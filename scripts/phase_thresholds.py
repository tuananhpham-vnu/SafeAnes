"""One alarm threshold vs one threshold per phase (pre-incision / surgery), samples v3.

    python scripts/phase_thresholds.py --config configs/uc04_v3.json --work artifacts/v3 \
        --extra-work artifacts/v3/kaggle_dl/uc04-v3-tabm-numeric/work ... --windows 120 --out reports/v3/phase_thresholds.csv

For each tabular model (W, h) with a model.joblib: calibrated probabilities on calibration (re-scored here)
and validation (val_predictions.parquet). Both policies are chosen on CALIBRATION under the same budget
(FA/h <= evaluation.fa_per_hour_budget over the whole case) and reported on VALIDATION, so neither gets the
optimism of choosing on the split it is reported on; `single_val` (the plan's threshold, chosen on
validation) is added for reference. Metrics by phase as uc04.phases.phase_metrics.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from uc04.calibration import Platt
from uc04.config import load_config
from uc04.io import read_json
from uc04.loaders import load_split
from uc04.phases import (align_probability, choose_phase_thresholds, evaluator_for, phase_metrics, phase_scaled,
                         read_surgery_start, row_phase)
from uc04.tabular import combo_dir
from uc04.thresholds import candidates, choose_threshold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tabm import OneHorizon  # noqa: E402,F401  (TabM bundles pickle __main__.OneHorizon)

MODELS = ("map_logistic", "lgbm_numeric", "lgbm_wave", "lgbm_context", "catboost_numeric", "tabm_numeric",
          "tabm_context")


def find_bundle(works: list[Path], model: str, W: int, h: int) -> Path | None:
    for w in works:
        p = w / "artifacts" / combo_dir(model, W, h)
        if (p / "model.joblib").exists() and (p / "val_predictions.parquet").exists():
            return p
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--work", required=True, nargs="+", help="work dirs with artifacts/tabular/")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--windows", nargs="+", type=int, default=[120])
    ap.add_argument("--horizons", nargs="+", type=int, default=[300, 600, 900, 1200, 1800])
    ap.add_argument("--grid", type=int, default=25, help="quantile candidates per phase")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    E, samples = cfg.evaluation, cfg.path("samples")
    works = [Path(w) for w in args.work]
    ss = read_surgery_start(samples)
    events = pd.read_parquet(samples / "events.parquet")
    lab_c, _ = load_split(samples, "calibration", [])
    lab_v, _ = load_split(samples, "validation", [])
    rows, t0 = [], time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for W in args.windows:
        todo = [(m, h, p) for m in args.models for h in args.horizons if (p := find_bundle(works, m, W, h))]
        if not todo:
            continue
        cols = sorted({c for _, _, p in todo for c in joblib.load(p / "model.joblib")["columns"]})
        _, feats_c = load_split(samples, "calibration", cols)
        for h in args.horizons:
            ev_c = evaluator_for(lab_c, events, h, E, split="calibration")
            ev_v = evaluator_for(lab_v, events, h, E)
            pre_c, pre_v = row_phase(ev_c, ss), row_phase(ev_v, ss)
            for m, hh, path in todo:
                if hh != h:
                    continue
                b = joblib.load(path / "model.joblib")
                el_c = lab_c["eligible"].to_numpy(bool)
                try:  # a model pickled with other library versions (e.g. TabM trained on Kaggle) may not load here
                    score_c = b["model"].score(feats_c)
                except Exception as exc:
                    print(f"W{W} h{h} {m}: skipped, cannot score calibration here ({type(exc).__name__}: {exc})")
                    continue
                p_c = ev_c.sort(np.where(el_c, Platt(**{k: v for k, v in b["calibrator"].items() if k != "method"})
                                         .predict(score_c), np.nan))
                p_v = ev_v.sort(align_probability(lab_v, pd.read_parquet(path / "val_predictions.parquet"),
                                                  "probability"))
                single = choose_threshold(candidates(p_c[ev_c.eligible & np.isfinite(p_c)], E.threshold_candidates),
                                          lambda t: ev_c.quick(p_c, t), budget=E.fa_per_hour_budget)
                phase = choose_phase_thresholds(ev_c, p_c, pre_c, budget=E.fa_per_hour_budget, n=args.grid)
                thr_val = read_json(path / "threshold.json")["threshold"]
                for policy, prob, thr, info in (
                        ("single_val", p_v, thr_val, {"threshold_pre": thr_val, "threshold_surgery": thr_val}),
                        ("single_cal", p_v, single["threshold"],
                         {"threshold_pre": single["threshold"], "threshold_surgery": single["threshold"]}),
                        ("phase_cal", phase_scaled(p_v, pre_v, phase["threshold_pre"], phase["threshold_surgery"]),
                         1.0, {k: phase[k] for k in ("threshold_pre", "threshold_surgery")})):
                    for r in phase_metrics(ev_v, prob, thr, ss, bins=E.ece_bins):
                        if policy == "phase_cal":  # AUROC etc. of the scaled score are meaningless
                            r.update(auroc=np.nan, auprc=np.nan, brier=np.nan, ece=np.nan)
                        rows.append({"model": m, "window": W, "horizon": h, "policy": policy, **info, **r})
                v = [r for r in rows if r["model"] == m and r["window"] == W and r["horizon"] == h
                     and r["phase"] == "all"]
                print(f"W{W} h{h} {m:<17} " + "  ".join(
                    f"{r['policy']}: sens {r['event_sensitivity']:.3f} FA/h {r['false_alarms_per_hour']:.2f}"
                    for r in v) + f"  ({time.time() - t0:.0f} s)", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)  # after every model: a crash keeps what is done
    print(f"{len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
