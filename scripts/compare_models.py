"""Main comparison table on validation: does the arterial waveform add information?

    python scripts/compare_models.py [--window 120] [--policies drop_below cooldown] [--bootstrap 200]

Uses the validation predictions saved by train_tabular.py (no retraining). For each
alarm re-arm policy and each horizon, every model gets its own threshold (FA/h
budget on validation, plan 8.4), then metrics with patient-bootstrap CIs. The same
bootstrap draws are used for all models, so pairwise differences get paired CIs:
  lgbm_wave - lgbm_numeric, lgbm_wave - map_logistic, lgbm_numeric - map_threshold.
Writes reports/model_comparison.csv and reports/model_comparison_pairs.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from uc04.bootstrap import Bootstrap, ci, paired_ci
from uc04.io import read_json
from uc04.loaders import load_split
from uc04.metrics import Evaluator, lead_quantiles, row_metrics, summarize_counts
from uc04.runtime import REPO_CONFIG
from uc04.config import load_config
from uc04.tabular import _write_csv, combo_dir
from uc04.thresholds import candidates, choose_threshold

MODELS = ("map_threshold", "map_logistic", "lgbm_numeric", "lgbm_wave")
PAIRS = (("lgbm_wave", "lgbm_numeric"), ("lgbm_wave", "map_logistic"), ("lgbm_numeric", "map_threshold"))
KEYS = ("event_sensitivity", "event_sensitivity_all", "false_alarms_per_hour", "alarm_time_fraction", "auroc", "auprc")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_CONFIG))
    ap.add_argument("--work", help="root with artifacts/ and reports/ (default: repo root)")
    ap.add_argument("--window", type=int, help="default: W* from reports/w_star.json")
    ap.add_argument("--policies", nargs="+", default=["drop_below", "cooldown"])
    ap.add_argument("--bootstrap", type=int)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    work = Path(args.work) if args.work else Path(cfg.root)
    W = args.window or read_json(work / "reports" / "w_star.json")["w_star"]
    E = cfg.evaluation
    boot = E.bootstrap_repeats_validation if args.bootstrap is None else args.bootstrap
    samples = cfg.path("samples")
    labels, _ = load_split(samples, "validation", [])
    events = pd.read_parquet(samples / "events.parquet")
    events = events[events.split == "validation"]

    rows, pairs = [], []
    for h in cfg.labels.horizons_seconds:
        probs = {}
        for m in MODELS:
            vp = pd.read_parquet(work / "artifacts" / combo_dir(m, W, h) / "val_predictions.parquet")
            if not (np.array_equal(vp.caseid.to_numpy(), labels.caseid.to_numpy())
                    and np.array_equal(vp.time.to_numpy(), labels.time.to_numpy())
                    and np.array_equal(vp[f"y_{h}"].to_numpy(), labels[f"y_{h}"].to_numpy())):
                raise SystemExit(f"{m} h{h}: saved predictions do not match the current validation labels")
            probs[m] = vp["probability"].to_numpy(float)
        for policy in args.policies:
            ev = Evaluator(labels, events, h, persistence=E.alarm_persistence, cooldown=E.alarm_cooldown_seconds,
                           early_lead=E.early_lead_seconds, rearm=policy)
            sel = ev.eligible
            bs, draws = None, {}
            for m in MODELS:
                p = ev.sort(probs[m])
                thr = choose_threshold(candidates(p[sel & np.isfinite(p)], E.threshold_candidates),
                                       lambda t: ev.quick(p, t), budget=E.fa_per_hour_budget)
                res = ev.evaluate(p, thr["threshold"])
                if bs is None:
                    subj_rows = labels["subjectid"].to_numpy()[ev.order][sel]
                    bs = Bootstrap(subj_rows, res.case_table["subjectid"].to_numpy(), boot, cfg.seed)
                draws[m] = bs.metrics(res.case_table, ev.y[sel], p[sel])
                rows.append({"alarm_rearm": policy, "window": W, "horizon": h, "model": m,
                             "threshold": thr["threshold"], "budget_not_met": thr["budget_not_met"],
                             **row_metrics(ev.y[sel], p[sel], bins=E.ece_bins),
                             **summarize_counts(res.case_table.sum(numeric_only=True)), **lead_quantiles(res),
                             **ci(draws[m], KEYS)})
            for a, b in PAIRS:
                d = paired_ci(draws[a], draws[b], KEYS)
                pairs.append({"alarm_rearm": policy, "window": W, "horizon": h, "pair": f"{a} - {b}", **d})
        print(f"h={h} done", flush=True)

    res = pd.DataFrame(rows)
    pr = pd.DataFrame(pairs)
    _write_csv(work / "reports" / "model_comparison.csv", res)
    _write_csv(work / "reports" / "model_comparison_pairs.csv", pr)
    pd.set_option("display.width", 250)
    for policy in args.policies:
        r = res[res.alarm_rearm == policy]
        t = r.assign(sens=r.apply(lambda x: f"{x.event_sensitivity:.3f} [{x.event_sensitivity_lo:.3f}, "
                                            f"{x.event_sensitivity_hi:.3f}]", axis=1))
        print(f"\n=== validation, W={W}, alarm_rearm={policy}")
        print(t[["horizon", "model", "sens", "event_sensitivity_all", "false_alarms_per_hour", "alarm_ppv",
                 "alarm_time_fraction", "auroc", "auprc", "prevalence"]].round(3).to_string(index=False))
        p = pr[pr.alarm_rearm == policy]
        cols = []
        for k in ("event_sensitivity", "auroc", "auprc"):
            p = p.assign(**{k: p.apply(lambda x: f"{x[k + '_diff_mean']:+.3f} [{x[k + '_lo']:+.3f}, "
                                                 f"{x[k + '_hi']:+.3f}]", axis=1)})
            cols.append(k)
        print(f"\npaired differences (bootstrap by subject, {boot} draws), alarm_rearm={policy}")
        print(p[["horizon", "pair", *cols]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
