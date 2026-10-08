"""Export what the SafeAnes demo backend serves into deploy/backend/data/ (git-ignored).

    python deploy/backend/export_bundle.py --config configs/uc04_v3.json --work artifacts/v3 --cases 14

Run from the repo root with PYTHONPATH="src;.local_deps". Writes:
  data/models/<model>/h<h>.txt     LightGBM booster (text), no uc04 needed at runtime
  data/models/<model>/meta.json    columns, Platt calibrator, alarm threshold, validation metrics per horizon
  data/cases/<caseid>.parquet      validation cases for replay: model features + labels, one row per 30 s
  data/cases.json                  case list (duration, incision time, events)
  data/report.json                 v2 vs v3 and by-phase tables (from compare_versions / phase_thresholds)
The demo cases are VitalDB validation cases (open dataset, data use agreement): keep data/ out of git and
out of public deployments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from uc04.columns import CONTEXT_COLS, numeric_cols
from uc04.config import load_config
from uc04.loaders import load_split
from uc04.phases import read_surgery_start

HERE = Path(__file__).resolve().parent
HORIZONS = (300, 600, 900, 1200, 1800)
MODELS = {"lgbm_context": "LightGBM + ngữ cảnh ca (đề xuất)", "lgbm_numeric": "LightGBM chỉ số monitor (baseline)"}
METRICS = ("auroc", "auroc_lo", "auroc_hi", "auprc", "event_sensitivity", "event_sensitivity_lo",
           "event_sensitivity_hi", "false_alarms_per_hour", "alarm_ppv", "lead_median_s", "events_eligible",
           "events_detected", "prevalence", "ece")


def export_models(work: Path, W: int, out: Path) -> None:
    for model, title in MODELS.items():
        meta = {"name": model, "title": title, "window_s": W, "horizons": {}}
        for h in HORIZONS:
            d = work / "artifacts" / "tabular" / model / f"W{W}" / f"h{h}"
            b = joblib.load(d / "model.joblib")
            booster = b["model"].model.booster_
            (out / model).mkdir(parents=True, exist_ok=True)
            booster.save_model(str(out / model / f"h{h}.txt"))  # write_text would turn \n into \r\n on Windows
            row = pd.read_csv(work / "reports" / "tabular_validation" / f"{model}_W{W}_h{h}.csv").iloc[0]
            meta["columns"] = list(b["columns"])
            meta["horizons"][str(h)] = {
                "calibrator": {k: float(v) for k, v in b["calibrator"].items() if k != "method"},
                "threshold": json.loads((d / "threshold.json").read_text())["threshold"],
                "metrics": {k: (None if pd.isna(row[k]) else float(row[k])) for k in METRICS}}
        (out / model / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"model {model}: {len(meta['columns'])} columns, {len(HORIZONS)} horizons")


def pick_cases(labels: pd.DataFrame, events: pd.DataFrame, ss: dict, n: int, seed: int) -> list[int]:
    """A spread of validation cases: events before incision, during surgery, both, none."""
    ev = events.assign(pre=events.onset <= events.caseid.map(ss))
    per = ev.groupby("caseid").agg(n=("onset", "size"), pre=("pre", "sum"))
    cases = pd.Series(labels.caseid.unique())
    dur = labels.groupby("caseid").time.max() / 60
    ok = cases[(dur.reindex(cases).to_numpy() >= 90) & (dur.reindex(cases).to_numpy() <= 360)]
    per = per.reindex(ok, fill_value=0)
    rng = np.random.default_rng(seed)
    groups = [per[(per.pre > 0) & (per.n > per.pre)], per[(per.pre > 0) & (per.n == per.pre)],
              per[(per.pre == 0) & (per.n > 0)], per[per.n == 0]]
    k = [n * 3 // 10, n * 2 // 10, n * 3 // 10]
    k.append(n - sum(k))
    out = []
    for g, kk in zip(groups, k):
        ids = g.index.to_numpy()
        out += sorted(rng.choice(ids, min(kk, len(ids)), replace=False).tolist())
    return [int(c) for c in out]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--work", required=True, help="v3 work dir with artifacts/tabular and reports/")
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--cases", type=int, default=14)
    ap.add_argument("--report", nargs="*", default=["artifacts/v3/compare_interim.csv",
                                                    "artifacts/v3/reports/phase_thresholds_W120.csv"])
    ap.add_argument("--out", default=str(HERE / "data"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out, work, W = Path(args.out), Path(args.work), args.window
    export_models(work, W, out / "models")

    samples = cfg.path("samples")
    cols = list(dict.fromkeys(numeric_cols(W) + list(CONTEXT_COLS)))
    labels, feats = load_split(samples, "validation", cols)
    events = pd.read_parquet(samples / "events.parquet")
    events = events[events.split == "validation"]
    ss = read_surgery_start(samples)
    chosen = pick_cases(labels, events, ss, args.cases, cfg.seed)
    keep = ["caseid", "time", "eligible", "exposure_seconds", *[f"y_{h}" for h in HORIZONS]]
    (out / "cases").mkdir(parents=True, exist_ok=True)
    listing = []
    for c in chosen:
        m = (labels.caseid == c).to_numpy()
        df = pd.concat([labels.loc[m, keep].reset_index(drop=True), feats.loc[m, cols].reset_index(drop=True)], axis=1)
        df.to_parquet(out / "cases" / f"{c}.parquet", index=False)
        e = events[events.caseid == c].sort_values("onset")
        listing.append({
            "caseid": c, "start_s": float(df.time.min()), "end_s": float(df.time.max()),
            "surgery_start_s": ss[c],
            "events": [{"onset_s": float(r.onset), "end_s": float(r.end), "min_map": float(r.min_map),
                        "pre_incision": bool(r.onset <= ss[c])} for r in e.itertuples()]})
    (out / "cases.json").write_text(json.dumps(listing, indent=1), encoding="utf-8")
    print(f"{len(listing)} cases: {[x['caseid'] for x in listing]}")

    report = {}
    for p in map(Path, args.report):
        if p.exists():
            d = pd.read_csv(p)
            report[p.stem] = json.loads(d.to_json(orient="records"))
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(f"report tables: {list(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
