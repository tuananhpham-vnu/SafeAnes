"""Re-evaluate the 65 tabular combinations from saved validation predictions (no retraining).

    python scripts/reevaluate_tabular.py [--bootstrap 200]

For both alarm re-arm policies (drop_below, cooldown): threshold on validation
(FA <= 1/h), event and alarm metrics including alarm_time_fraction, and patient-
bootstrap CIs of the count-based metrics (same seed and draws as train_tabular.py).
Row metrics (AUROC, AUPRC, Brier, ECE and their CIs) do not depend on the policy or
the threshold and are kept from the existing per-combination file.

Writes reports/tabular_validation_by_policy.csv (65 x 2 rows) and rewrites
reports/tabular_validation/<name>.csv with the configured policy
(evaluation.alarm_rearm). For drop_below it first checks that the recomputed
event metrics equal the stored ones.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from uc04.bootstrap import Bootstrap, ci
from uc04.config import load_config
from uc04.loaders import load_split
from uc04.metrics import Evaluator, lead_quantiles, summarize_counts
from uc04.runtime import REPO_CONFIG
from uc04.tabular import MODELS, RESULT_COLUMNS, _write_csv, combo_dir, combo_name
from uc04.thresholds import candidates, choose_threshold

COUNT_KEYS = ("event_sensitivity", "false_alarms_per_hour", "alarm_time_fraction")
ROW_KEEP = ("n_rows", "n_cases", "prevalence", "auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo", "auprc_hi",
            "brier", "ece", "config_digest", "git_commit", "status")
CHECK = ("threshold", "event_sensitivity", "false_alarms_per_hour", "alarm_ppv", "events_detected")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_CONFIG))
    ap.add_argument("--bootstrap", type=int)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    E = cfg.evaluation
    boot = E.bootstrap_repeats_validation if args.bootstrap is None else args.bootstrap
    root, samples = Path(cfg.root), cfg.path("samples")
    labels, _ = load_split(samples, "validation", [])
    events = pd.read_parquet(samples / "events.parquet")
    events = events[events.split == "validation"]
    rows, mismatch = [], []
    for h in cfg.labels.horizons_seconds:
        evs = {p: Evaluator(labels, events, h, persistence=E.alarm_persistence, cooldown=E.alarm_cooldown_seconds,
                            early_lead=E.early_lead_seconds, rearm=p) for p in ("drop_below", "cooldown")}
        for m in MODELS:
            for W in (cfg.features.windows_seconds if m != "map_threshold" else [None]):
                name = combo_name(m, W, h)
                old = pd.read_csv(root / "reports" / "tabular_validation" / f"{name}.csv").iloc[0]
                vp = pd.read_parquet(root / "artifacts" / combo_dir(m, W, h) / "val_predictions.parquet")
                if not np.array_equal(vp[f"y_{h}"].to_numpy(), labels[f"y_{h}"].to_numpy()):
                    raise SystemExit(f"{name}: saved predictions do not match the validation labels")
                for policy, ev in evs.items():
                    p = ev.sort(vp["probability"].to_numpy(float))
                    thr = choose_threshold(candidates(p[ev.eligible & np.isfinite(p)], E.threshold_candidates),
                                           lambda t: ev.quick(p, t), budget=E.fa_per_hour_budget,
                                           curve_points=E.curve_points)
                    res = ev.evaluate(p, thr["threshold"])
                    bs = Bootstrap(labels["subjectid"].to_numpy()[ev.order][ev.eligible],
                                   res.case_table["subjectid"].to_numpy(), boot, cfg.seed)
                    row = {"model": m, "window": W, "horizon": h, "alarm_rearm": policy,
                           **{k: old[k] for k in ROW_KEEP if k in old.index},
                           **summarize_counts(res.case_table.sum(numeric_only=True)), **lead_quantiles(res),
                           "threshold": thr["threshold"], "budget_not_met": thr["budget_not_met"],
                           **ci(bs.metrics(res.case_table, ev.y, p, with_rows=False), COUNT_KEYS)}
                    if policy == "drop_below":
                        bad = [k for k in CHECK if not np.isclose(float(row[k]), float(old[k]), equal_nan=True)]
                        if bad:
                            mismatch.append((name, bad))
                    rows.append(row)
                    if policy == E.alarm_rearm:
                        _write_csv(root / "reports" / "tabular_validation" / f"{name}.csv",
                                   pd.DataFrame([{c: row.get(c) for c in RESULT_COLUMNS}]))
        print(f"h={h} done", flush=True)
    if mismatch:
        print(f"drop_below did not reproduce stored results: {mismatch[:5]}", file=sys.stderr)
        return 1
    out = pd.DataFrame(rows)
    _write_csv(root / "reports" / "tabular_validation_by_policy.csv", out)
    print("drop_below reproduces the stored threshold, sensitivity, FA/h, PPV and detections for all combinations")
    piv = out[(out.horizon.isin([300, 600])) & (out.window.isin([120]) | out.window.isna())].pivot_table(
        index=["horizon", "model"], columns="alarm_rearm",
        values=["event_sensitivity", "false_alarms_per_hour", "alarm_time_fraction"])
    pd.set_option("display.width", 220)
    print(piv.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
