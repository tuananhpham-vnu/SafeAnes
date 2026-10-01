"""03_dl_finalize (CPU): ensemble the seeds of each window, calibrate, choose thresholds.

    python scripts/dl_finalize.py --inputs <dirs of 03_dl_W60/W30/W120/W90 outputs> [--work .]

Uses only the per-seed logits written by train_dl.py (no GPU). For each W with at
least one finished seed:
- mean logit over seeds; one temperature and one bias shared by the 5 horizons,
  fitted on calibration (plan 8.1), so probabilities stay ordered;
- per horizon: threshold on validation (FA/h budget), metrics, bootstrap CIs;
- confidence: SD of the calibrated per-seed probabilities, tertiles on validation.
Writes artifacts/dl/W<W>/ (logits, calibration.json, threshold.json,
case_metrics_h<h>.parquet, val_predictions.parquet), reports/dl_validation.csv,
reports/calibration_check_dl.csv, reports/dl_confidence/W<W>.csv. Only this
notebook writes these files.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uc04.calibration import TemperatureBias
from uc04.dl_data import HORIZONS
from uc04.hub import hub_from_config, on_kaggle
from uc04.io import read_json, write_json, write_parquet
from uc04.metrics import row_metrics
from uc04.provenance import make_provenance, write_provenance
from uc04.runtime import common_args, load_data, samples_inputs, setup
from uc04.tabular import RESULT_COLUMNS, TabularData, _write_csv, evaluate_probs

LOGIT_COLS = [f"logit_{h}" for h in HORIZONS]


def seed_dirs(inputs: list[Path], W: int) -> dict[int, Path]:
    out = {}
    for d in inputs:
        for done in sorted((d / "artifacts" / "dl" / f"W{W}").glob("seed*/done.json")):
            out.setdefault(int(done.parent.name[4:]), done.parent)
    return out


def load_logits(dirs: dict[int, Path], split: str) -> tuple[pd.DataFrame, np.ndarray]:
    frames = [pd.read_parquet(d / f"logits_{split}.parquet") for d in dirs.values()]
    keys = frames[0][["caseid", "time"]]
    for f in frames[1:]:
        if not f[["caseid", "time"]].equals(keys):
            raise ValueError(f"{split} logits of different seeds are not row-aligned")
    return keys, np.stack([f[LOGIT_COLS].to_numpy(np.float64) for f in frames])  # [seeds, n, 5]


def align(labels: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    """Position in `keys` of every labels row (-1 if absent)."""
    k = pd.Series(np.arange(len(keys)), index=pd.MultiIndex.from_frame(keys))
    return k.reindex(pd.MultiIndex.from_frame(labels[["caseid", "time"]])).fillna(-1).astype(int).to_numpy()


def main(argv=None) -> int:
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--inputs", nargs="+", help="work dirs written by train_dl.py (default: --work)")
    ap.add_argument("--windows", nargs="+", type=int)
    ap.add_argument("--bootstrap", type=int)
    args = ap.parse_args(argv)
    cfg, samples, work, push, subset = setup(args)
    kaggle = on_kaggle()
    # On Kaggle, per-seed logits come only from Hugging Face (DEVIATIONS 31); --inputs is for local runs.
    inputs = [] if kaggle else [Path(p) for p in (args.inputs or [work])]
    windows = args.windows or list(cfg.dl.windows_seconds)
    boot = cfg.evaluation.bootstrap_repeats_validation if args.bootstrap is None else args.bootstrap
    hub = hub_from_config(cfg, work, push=push)
    t0 = time.time()
    labels, _, events = load_data(samples, [], subset, splits=("calibration", "validation"))
    prov = make_provenance(cfg, "dl_finalize", inputs=samples_inputs(samples))
    results, checks, written = [], [], []

    for W in windows:
        dirs = seed_dirs(inputs, W)
        if kaggle and not hub.enabled:
            raise SystemExit("on Kaggle 03_dl_finalize reads logits from Hugging Face, but HF is not set up")
        if not dirs and hub.enabled:
            for s in cfg.dl.seeds:
                for f in ("done.json", "logits_calibration.parquet", "logits_validation.parquet"):
                    try:
                        hub.pull(f"dl/W{W}/seed{s}/{f}", local_dir=work / "_pull")
                    except Exception:
                        break
            dirs = {int(p.parent.name[4:]): p.parent for p in (work / "_pull" / "dl" / f"W{W}").glob("seed*/done.json")}
        if not dirs:
            print(f"W={W}: no finished seed, skipping")
            continue
        keys_c, Lc = load_logits(dirs, "calibration")
        keys_v, Lv = load_logits(dirs, "validation")
        lab_c, lab_v = labels["calibration"], labels["validation"]
        pos_c = align(lab_c, keys_c)
        known_c = pos_c >= 0
        y_c = lab_c.loc[known_c, [f"y_{h}" for h in HORIZONS]].to_numpy()
        ens_c = Lc.mean(0)[pos_c[known_c]]
        tb = TemperatureBias().fit(ens_c, y_c)
        pos_v = align(lab_v, keys_v)
        has = pos_v >= 0
        ens_v = Lv.mean(0)
        p_v = np.full((len(lab_v), len(HORIZONS)), np.nan)
        p_v[has] = tb.predict(ens_v[pos_v[has]])
        p_seed = tb.predict(Lv)                                  # [seeds, n, 5]
        sd = np.full((len(lab_v), len(HORIZONS)), np.nan)
        sd[has] = p_seed.std(0)[pos_v[has]] if len(dirs) > 1 else np.nan
        out = work / "artifacts" / "dl" / f"W{W}"
        out.mkdir(parents=True, exist_ok=True)
        for split, keys, L in (("calibration", keys_c, Lc), ("validation", keys_v, Lv)):
            df = keys.copy()
            df[LOGIT_COLS] = L.mean(0)
            write_parquet(out / f"logits_{split}.parquet", df)
        write_json(out / "calibration.json", {**tb.to_dict(), "seeds": sorted(dirs), "fitted_on": "calibration"})
        data = TabularData(W, {"validation": lab_v}, {}, events)
        thresholds, conf_rows = {}, []
        preds = lab_v[["caseid", "subjectid", "time", "eligible", "exposure_seconds"]].copy()
        for k, h in enumerate(HORIZONS):
            prob = np.where(lab_v["eligible"].to_numpy(bool), p_v[:, k], np.nan)
            preds[f"probability_{h}"] = prob
            summary, thr, res, ev = evaluate_probs(cfg, data, h, prob, boot, cfg.seed)
            thresholds[str(h)] = thr
            write_parquet(out / f"case_metrics_h{h}.parquet", res.case_table)
            results.append({c: summary.get(c) for c in RESULT_COLUMNS} | {
                "model": "dl_conv_tf", "window": W, "horizon": h, "status": "ok", "n_seeds": len(dirs),
                "config_digest": prov["config_digest"], "git_commit": prov["git"]["commit"]})
            y = lab_v[f"y_{h}"].to_numpy()
            ok = lab_v["eligible"].to_numpy(bool) & (y != -1) & np.isfinite(prob)
            checks.append({"model": "dl_conv_tf", "window": W, "horizon": h, "mean_p_validation": float(prob[ok].mean()),
                           "positive_rate_validation": float((y[ok] == 1).mean())})
            if len(dirs) > 1:
                el = lab_v["eligible"].to_numpy(bool) & np.isfinite(sd[:, k])
                cuts = np.quantile(sd[el, k], [1 / 3, 2 / 3])
                lvl = np.where(sd[:, k] <= cuts[0], "high", np.where(sd[:, k] <= cuts[1], "medium", "low"))
                for lv in ("high", "medium", "low"):
                    m = el & (lvl == lv)
                    conf_rows.append({"window": W, "horizon": h, "confidence": lv, "rows": int(m.sum()),
                                      "sd_cut_low": float(cuts[0]), "sd_cut_high": float(cuts[1]),
                                      **row_metrics(y[m], prob[m])})
            print(f"  W{W} h{h}: sens {summary['event_sensitivity']:.3f} "
                  f"[{summary.get('event_sensitivity_lo', np.nan):.3f}, {summary.get('event_sensitivity_hi', np.nan):.3f}] "
                  f"FA/h {summary['false_alarms_per_hour']:.2f} AUROC {summary['auroc']:.3f} "
                  f"({len(dirs)} seeds, T={tb.temperature:.2f})", flush=True)
        write_json(out / "threshold.json", thresholds)
        write_parquet(out / "val_predictions.parquet", preds)
        write_provenance(out, {**prov, "window": W, "seeds": sorted(dirs)})
        written.append(f"artifacts/dl/W{W}")
        if conf_rows:
            _write_csv(work / "reports" / "dl_confidence" / f"W{W}.csv", pd.DataFrame(conf_rows))
            written.append(f"reports/dl_confidence/W{W}.csv")

    if not results:
        print("no window had finished seeds", file=sys.stderr)
        return 1
    _write_csv(work / "reports" / "dl_validation.csv", pd.DataFrame(results)[["n_seeds", *RESULT_COLUMNS]])
    _write_csv(work / "reports" / "calibration_check_dl.csv", pd.DataFrame(checks))
    written += ["reports/dl_validation.csv", "reports/calibration_check_dl.csv"]
    hub.register(*written)
    hub.push(written, message="03 dl finalize")
    print(f"done in {time.time() - t0:.0f} s; outputs in {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
