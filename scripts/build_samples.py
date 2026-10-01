"""Stage 1.5 (NB01): build samples_v2 from prep_v1 on the local machine.

    python scripts/build_samples.py --config configs/uc04_v2.json [--workers 10] [--force]

Output (plan 3.7):
  <samples>/  labels/{train,calibration,validation}.parquet, features/{...}.parquet,
              events.parquet, case_index.parquet, wave_mask.npz, norm_tabular.json,
              label_report.csv, abstain_report.csv, subsets/dryrun.json, samples.json
  <sealed>/   labels_test.parquet, features_test.parquet, events_test.parquet,
              case_index_test.parquet, wave_mask_test.npz  (written, never summarised)
  <reports>/  cohort_flow.csv, cohort_excluded.csv

Everything is built in `<dir>.tmp` and renamed at the end. If samples.json already
records the same config digest and qc.csv sha256, the build is skipped.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from uc04.checks import EXPECTED
from uc04.columns import WINDOWS, numeric_cols, wave_cols
from uc04.config import load_config
from uc04.io import SPLITS, read_json, read_qc, sha256_file, write_json
from uc04.loaders import check_no_subject_overlap
from uc04.provenance import make_provenance, write_provenance
from uc04.samples import (SAMPLES_SECTIONS, TEST, BuildParams, build_case, case_specs, column_stats,
                          dryrun_subset, label_report, model_columns)

DRYRUN_SIZES = {"train": 30, "calibration": 15, "validation": 15}
IN_FLIGHT = 48          # bounded number of pending cases (memory)
FLUSH_ROWS = 150_000    # rows per parquet row group


class StreamWriter:
    """Append DataFrames to one parquet file in row groups of about FLUSH_ROWS rows."""

    def __init__(self, path: Path):
        self.path, self.buf, self.n, self.writer, self.rows = path, [], 0, None, 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, df: pd.DataFrame) -> None:
        self.buf.append(df)
        self.n += len(df)
        if self.n >= FLUSH_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        table = pa.Table.from_pandas(pd.concat(self.buf, ignore_index=True), preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table.cast(self.writer.schema))
        self.rows += table.num_rows
        self.buf, self.n = [], 0

    def close(self) -> int:
        self.flush()
        if self.writer is not None:
            self.writer.close()
        return self.rows


def run_cases(prep, specs, params, workers):
    """Yield build_case results in the order of `specs`, with bounded memory."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = []
        it = iter(specs)
        for spec in it:
            pending.append(pool.submit(build_case, prep, spec, params))
            if len(pending) >= IN_FLIGHT:
                yield pending.pop(0).result()
        for fut in pending:
            yield fut.result()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/uc04_v2.json")
    ap.add_argument("--prep"), ap.add_argument("--samples"), ap.add_argument("--sealed"), ap.add_argument("--reports")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, help="first N cases per split (debug; use other --samples/--sealed)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config).with_paths(prep=args.prep, samples=args.samples, sealed=args.sealed,
                                              reports=args.reports)
    prep, samples, sealed, reports = (cfg.path(n) for n in ("prep", "samples", "sealed", "reports"))
    qc_sha = sha256_file(prep / "qc.csv")
    if qc_sha != cfg.input.qc_sha256:
        print(f"qc.csv sha256 {qc_sha} != config input.qc_sha256; run check_data.py first", file=sys.stderr)
        return 1
    digest = cfg.digest()
    done = samples / "samples.json"
    if done.exists() and not args.force and not args.limit:
        rec = read_json(done)
        if rec.get("samples_digest") == cfg.digest(*SAMPLES_SECTIONS) and rec.get("qc_sha256") == qc_sha:
            print(f"up to date, skipping: {done} (use --force to rebuild)")
            return 0

    t0 = time.time()
    params = BuildParams.from_config(cfg)
    qc = read_qc(prep)
    check_no_subject_overlap(qc)
    qc["_order"] = qc["split"].map({s: i for i, s in enumerate(SPLITS)})
    qc = qc.sort_values(["_order", "caseid"])
    if args.limit:
        qc = qc.groupby("split", sort=False).head(args.limit)
    specs = case_specs(qc)

    tmp_s, tmp_x = samples.with_name(samples.name + ".tmp"), sealed.with_name(sealed.name + ".tmp")
    for d in (tmp_s, tmp_x):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    def target(split: str, kind: str) -> Path:
        return tmp_x / f"{kind}_test.parquet" if split == TEST else tmp_s / kind / f"{split}.parquet"

    writers: dict[tuple[str, str], StreamWriter] = {}
    events, index, masks = defaultdict(list), defaultdict(list), defaultdict(dict)
    report_rows, abstain = [], defaultdict(lambda: defaultdict(int))
    ppv_nan = {W: [0, 0] for W in WINDOWS}
    train_rows = 0
    print(f"building {len(specs)} cases with {args.workers} workers -> {samples} (+ sealed {sealed})", flush=True)
    for i, res in enumerate(run_cases(prep, specs, params, args.workers), 1):
        spec = res["spec"]
        grp = TEST if spec.split == TEST else "public"
        for kind in ("labels", "features"):
            key = (spec.split, kind)
            if key not in writers:
                writers[key] = StreamWriter(target(spec.split, kind))
            writers[key].add(res[kind])
        events[grp].append(res["events"])
        index[grp].append({"caseid": spec.caseid, "subjectid": spec.subjectid, "split": spec.split,
                           "start": res["start"], "surgery_seconds": res["surgery_seconds"],
                           "has_wave": spec.has_wave, "in_main_cohort": spec.in_main_cohort(params.min_map_coverage),
                           "map_coverage": spec.map_coverage, "baseline_source": spec.baseline_source})
        masks[grp][str(spec.caseid)] = res["wave_mask"]
        if spec.split != TEST:
            report_rows += res["report"]
            for reason, n in res["abstain"].items():
                abstain[spec.split][reason] += n
            if spec.split == "train":
                train_rows += res["rows"]
                for W, (o, n) in res["ppv_nan"].items():
                    ppv_nan[W][0] += o
                    ppv_nan[W][1] += n
        if i % 250 == 0 or i == len(specs):
            print(f"  {i}/{len(specs)} cases, {time.time() - t0:.0f} s", flush=True)
    rows = {f"{s}/{k}": w.close() for (s, k), w in writers.items()}

    # ---- per-case tables
    ev_cols = ["caseid", "subjectid", "split", "onset", "end", "duration", "min_map", "suspect_artefact",
               *[f"eligible_{h}" for h in cfg.labels.horizons_seconds]]
    for grp, base, suffix in (("public", tmp_s, ""), (TEST, tmp_x, "_test")):
        ev = pd.concat(events[grp], ignore_index=True) if events[grp] else pd.DataFrame(columns=ev_cols)
        ev[ev_cols].to_parquet(base / f"events{suffix}.parquet", index=False)
        pd.DataFrame(index[grp]).to_parquet(base / f"case_index{suffix}.parquet", index=False)
        np.savez_compressed(base / f"wave_mask{suffix}.npz", **masks[grp])

    # ---- alignment check (labels vs features), non-test and test alike, without reading labels values
    for s in SPLITS:
        lp, fp = target(s, "labels"), target(s, "features")
        if not lp.exists():
            continue
        a = pq.read_table(lp, columns=["caseid", "time"])
        b = pq.read_table(fp, columns=["caseid", "time"])
        if not (a.equals(b)):
            print(f"labels and features of {s} are not row-aligned", file=sys.stderr)
            return 1

    # ---- reports (non-test only)
    rep = label_report(report_rows)
    rep.to_csv(tmp_s / "label_report.csv", index=False)
    ab = pd.DataFrame([{"split": s, "abstain_reason": r, "rows": n} for s, d in abstain.items() for r, n in d.items()])
    ab.sort_values(["split", "rows"], ascending=[True, False]).to_csv(tmp_s / "abstain_report.csv", index=False)

    idx = pd.DataFrame(index["public"])
    evp = pd.read_parquet(tmp_s / "events.parquet")
    (tmp_s / "subsets").mkdir()
    write_json(tmp_s / "subsets" / "dryrun.json", dryrun_subset(evp, idx, DRYRUN_SIZES, cfg.seed))

    # ---- normalisation on train, eligible rows of the main cohort
    cols = model_columns(params)
    lab_train = pd.read_parquet(tmp_s / "labels" / "train.parquet", columns=["eligible", "in_main_cohort"])
    mask = (lab_train["eligible"] & lab_train["in_main_cohort"]).to_numpy()
    feat_file = tmp_s / "features" / "train.parquet"
    stats = column_stats(lambda c: pq.read_table(feat_file, columns=c).to_pandas(), cols, mask)
    write_json(tmp_s / "norm_tabular.json", {"scope": "train, eligible, main cohort", "rows": int(mask.sum()),
                                             "columns": stats})

    # ---- cohort flow (qc only)
    qc_all = read_qc(prep)
    flow, excluded = [], []
    for s in SPLITS:
        d = qc_all[qc_all.split == s]
        c = d[d.has_wave]
        dd = c[c.map_coverage >= cfg.input.min_map_coverage]
        flow.append({"split": s, "A_cohort": EXPECTED[s][0], "B_processed": len(d), "C_has_wave": len(c),
                     "D_main": len(dd), "subjects_B": d.subjectid.nunique(), "subjects_D": dd.subjectid.nunique()})
        excluded += [{"caseid": int(x), "split": s, "step": "C", "reason": "no_wave"} for x in d[~d.has_wave].caseid]
        excluded += [{"caseid": int(x), "split": s, "step": "D", "reason": "no_usable_arterial_line"}
                     for x in c[c.map_coverage < cfg.input.min_map_coverage].caseid]
    excluded += [{"caseid": int(x), "split": "", "step": "B", "reason": "prep_error"} for x in cfg.input.known_failed_cases]
    flow = pd.DataFrame(flow)
    reports.mkdir(parents=True, exist_ok=True)
    flow.to_csv(reports / "cohort_flow.csv", index=False)
    pd.DataFrame(excluded).to_csv(reports / "cohort_excluded.csv", index=False)
    flow[flow.split != TEST].to_csv(tmp_s / "cohort_flow.csv", index=False)  # uploaded copy: no test row

    # ---- summary
    ev_non_test = evp
    n_ev = len(ev_non_test)
    d_main = qc_all[(qc_all.split != TEST) & qc_all.has_wave & (qc_all.map_coverage >= cfg.input.min_map_coverage)]
    exposure_zero = {s: int((pd.read_parquet(tmp_s / "labels" / f"{s}.parquet", columns=["exposure_seconds"])
                             .exposure_seconds <= 0).sum()) for s in ("train", "calibration", "validation")}
    summary = {
        "config_digest": digest, "samples_digest": cfg.digest(*SAMPLES_SECTIONS), "qc_sha256": qc_sha,
        "exclusion_reset": cfg.labels.exclusion_reset,
        "label_policy": cfg.labels.label_policy, "negative_rule": cfg.labels.negative_rule,
        "min_map_coverage": cfg.input.min_map_coverage,
        "rows": rows, "cases": {s: int((idx.split == s).sum()) for s in ("train", "calibration", "validation")},
        "main_cohort_cases": {s: int(((idx.split == s) & idx.in_main_cohort).sum())
                              for s in ("train", "calibration", "validation")},
        "dropped_cases": pd.DataFrame(excluded).to_dict("records"),
        "suspect_artefact_events": {"n": int(ev_non_test.suspect_artefact.sum()), "of": n_ev,
                                    "rate": round(float(ev_non_test.suspect_artefact.mean()), 5) if n_ev else None},
        "ppv_nan_rate_train": {f"w{W}": {"old_bt_ppv": round(o / train_rows, 4), "new_bt_ppv30": round(n / train_rows, 4)}
                               for W, (o, n) in ppv_nan.items()} if train_rows else {},
        "columns_per_window": {f"w{W}": {"numeric": len(numeric_cols(W)), "waveform": len(wave_cols(W))} for W in WINDOWS},
        "model_columns": len(cols),
        "qc_summary": {  # plan 3.8, main cohort, non-test cases only
            "baseline_source_main_non_test": {int(k): int(v) for k, v in d_main.baseline_source.value_counts().sort_index().items()},
            "median_label_known_frac": round(float(d_main.label_known_frac.median()), 4),
            "median_beat_valid_frac": round(float(d_main.beat_valid_frac.median()), 4)},
        "rows_exposure_zero": exposure_zero,
        "sealed_dir": str(sealed), "seconds": round(time.time() - t0, 1),
        "limit": args.limit,
    }
    write_json(tmp_s / "samples.json", summary)
    write_provenance(tmp_s, make_provenance(cfg, "build_samples", inputs={"qc.csv": qc_sha}))

    for tmp, final in ((tmp_s, samples), (tmp_x, sealed)):
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)

    # ---- printout (plan 3.8)
    pd.set_option("display.width", 200)
    print("\nCohort flow:\n" + flow.to_string(index=False))
    print(f"threshold map_coverage >= {cfg.input.min_map_coverage}")
    d_non_test = qc_all[(qc_all.split != TEST) & qc_all.has_wave & (qc_all.map_coverage >= cfg.input.min_map_coverage)]
    print("baseline_source (all cases):", qc_all.baseline_source.value_counts().sort_index().to_dict())
    print(f"main cohort (non-test) median label_known_frac {d_non_test.label_known_frac.median():.3f}, "
          f"beat_valid_frac {d_non_test.beat_valid_frac.median():.3f}")
    for rule in ("event", "any_low"):
        sub = rep[rep.exclusion_reset == rule]
        for col, title in (("unknown_rate", "share of eligible rows labelled -1"),
                           ("positive_rate", "positive rate among labelled rows"),
                           ("events_eligible", "events with an eligible positive row before them")):
            piv = sub.pivot_table(index=["split", "horizon"], columns="label_policy", values=col)
            print(f"\n{title} (main cohort, exclusion_reset={rule}):\n" + piv.round(4).to_string())
    pol = f"lenient_{'possible' if cfg.labels.negative_rule == 'possible_event' else 'fraction'}"
    cur = rep[(rep.label_policy == pol) & (rep.exclusion_reset == cfg.labels.exclusion_reset)]
    print(f"\n-1 labels by reason ({pol}, exclusion_reset={cfg.labels.exclusion_reset}; share of eligible rows):\n"
          + cur[["split", "horizon", "unknown_rate", "unknown_end_of_case_rate", "unknown_other_rate"]]
          .to_string(index=False))
    cmp_ = rep[rep.label_policy == pol].pivot_table(index=["split", "horizon"], columns="exclusion_reset",
                                                    values=["eligible", "events_eligible"])
    print(f"\nexclusion_reset comparison ({pol}): eligible rows and alertable events\n" + cmp_.to_string())
    print(f"\nmain cohort (non-test): baseline_source {summary['qc_summary']['baseline_source_main_non_test']}, "
          f"median label_known_frac {summary['qc_summary']['median_label_known_frac']}, "
          f"median beat_valid_frac {summary['qc_summary']['median_beat_valid_frac']}; "
          f"rows with exposure 0 (prediction time at the end of the case): {summary['rows_exposure_zero']}")
    print(f"\nlabel tables use label_policy={cfg.labels.label_policy}, negative_rule={cfg.labels.negative_rule}, "
          f"exclusion_reset={cfg.labels.exclusion_reset}")
    print(f"\nsuspect_artefact events (non-test): {summary['suspect_artefact_events']}")
    print(f"PPV NaN rate on train: {summary['ppv_nan_rate_train']}")
    print(f"columns per window: 66 + 27; model columns {len(cols)}")
    print(f"rows written: {rows}")
    print(f"done in {summary['seconds']} s -> {samples}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
