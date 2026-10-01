"""EXPLORATORY: lgbm_numeric with longer windows (W = 300, 600 s) on validation.

    python scripts/explore_long_windows.py [--workers 10] [--bootstrap 200]

Why exploratory: W* = 120 was the longest window tried (the edge of the range), and
EDA shows MAP starting to fall about 8 minutes before an event. This analysis was
decided AFTER seeing validation results, so it is reported separately and does not
replace W* or any pre-specified result.

Features for W = 120, 300, 600 are all recomputed from the 2 s grid of cases/ with
the same function (uc04.long_windows; it reproduces prep_v1's W = 30-120 columns in
99.9 % of rows), so the three windows are compared like for like. W-independent
columns (*_current, map_drop_pct) come from samples_v2. Each W x horizon: fit on
train, Platt on calibration, threshold on validation (FA <= 1/h), bootstrap CIs,
paired bootstrap against the recomputed W = 120.
Writes data/exploratory/ (features) and reports/exploratory/long_windows*.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from uc04.bootstrap import Bootstrap, ci, paired_ci
from uc04.columns import SIG, numeric_cols
from uc04.config import load_config
from uc04.io import write_parquet
from uc04.loaders import load_split
from uc04.long_windows import case_long_features
from uc04.runtime import REPO_CONFIG
from uc04.tabular import TabularData, _write_csv, evaluate_probs, fit_and_calibrate, predict_split

WINDOWS = (120, 300, 600)
CURRENT = [f"{s}_current" for s in SIG] + ["map_drop_pct"]
KEYS = ("event_sensitivity", "event_sensitivity_all", "false_alarms_per_hour", "auroc", "auprc")


def _one(args):
    prep, cid, times, mc = args
    return cid, case_long_features(prep, cid, times, mc, WINDOWS)


def build(cfg, split: str, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(cfg.root) / "data" / "exploratory" / f"features_long_{split}.parquet"
    labels, base = load_split(cfg.path("samples"), split, CURRENT)
    if out.exists():
        feats = pd.read_parquet(out)
        if len(feats) == len(labels):
            return labels, feats
    tasks = [(cfg.path("prep"), int(c), g["time"].to_numpy(), g["map_current"].to_numpy())
             for c, g in base.groupby("caseid", sort=False)]
    parts = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for cid, df in pool.map(_one, tasks, chunksize=8):
            parts[cid] = df
    long = pd.concat([parts[c] for c in base["caseid"].drop_duplicates()], ignore_index=True)
    feats = pd.concat([base.reset_index(drop=True), long], axis=1)
    write_parquet(out, feats)
    return labels, feats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_CONFIG))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--bootstrap", type=int)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    boot = cfg.evaluation.bootstrap_repeats_validation if args.bootstrap is None else args.bootstrap
    t0 = time.time()
    labels, feats = {}, {}
    for s in ("train", "calibration", "validation"):
        labels[s], feats[s] = build(cfg, s, args.workers)
        print(f"{s}: {len(feats[s]):,} rows x {feats[s].shape[1]} columns ({time.time() - t0:.0f} s)", flush=True)
    events = pd.read_parquet(cfg.path("samples") / "events.parquet")
    events = events[events.caseid.isin(pd.concat([labels[s].caseid for s in labels]).unique())]

    rows, pairs = [], []
    for h in cfg.labels.horizons_seconds:
        draws, bs = {}, None
        for W in WINDOWS:
            cols = numeric_cols(W)
            data = TabularData(W, labels, {s: feats[s][["caseid", "time", *cols]] for s in feats}, events)
            est, platt, status = fit_and_calibrate(cfg, "lgbm_numeric", data, h)
            if status != "ok":
                continue
            prob = predict_split(est, platt, labels["validation"], data.features["validation"])
            summary, thr, res, ev = evaluate_probs(cfg, data, h, prob, 0, cfg.seed)
            sel = ev.eligible
            if bs is None:
                bs = Bootstrap(labels["validation"]["subjectid"].to_numpy()[ev.order][sel],
                               res.case_table["subjectid"].to_numpy(), boot, cfg.seed)
            draws[W] = bs.metrics(res.case_table, ev.y[sel], ev.sort(prob)[sel])
            rows.append({"analysis": "exploratory", "model": "lgbm_numeric", "window": W, "horizon": h,
                         **{k: summary.get(k) for k in ("threshold", "budget_not_met", "event_sensitivity",
                                                        "events_detected", "events_eligible", "event_sensitivity_all",
                                                        "false_alarms_per_hour", "alarm_ppv", "auroc", "auprc",
                                                        "prevalence")}, **ci(draws[W], KEYS)})
            print(f"  W={W:>3} h={h:>4}: sens {summary['event_sensitivity']:.3f} FA/h {summary['false_alarms_per_hour']:.2f} "
                  f"AUROC {summary['auroc']:.3f}", flush=True)
        for W in (300, 600):
            if W in draws and 120 in draws:
                pairs.append({"analysis": "exploratory", "horizon": h, "pair": f"W{W} - W120",
                              **paired_ci(draws[W], draws[120], KEYS)})
    out = Path(cfg.root) / "reports" / "exploratory"
    _write_csv(out / "long_windows.csv", pd.DataFrame(rows))
    _write_csv(out / "long_windows_pairs.csv", pd.DataFrame(pairs))
    p = pd.DataFrame(pairs)
    pd.set_option("display.width", 220)
    for k in ("event_sensitivity", "auroc", "auprc"):
        p[k] = p.apply(lambda x: f"{x[k + '_diff_mean']:+.3f} [{x[k + '_lo']:+.3f}, {x[k + '_hi']:+.3f}]", axis=1)
    print("\nEXPLORATORY paired differences vs recomputed W=120 (validation):")
    print(p[["horizon", "pair", "event_sensitivity", "auroc", "auprc"]].to_string(index=False))
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
