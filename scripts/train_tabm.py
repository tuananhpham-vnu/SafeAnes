"""TabM (E06/E08 model family) on the UC04 cohort: one 5-horizon model per W, then per horizon the same
Platt calibration, threshold choice, evaluation and outputs as train_tabular.py (uc04.tabular.run_combo).

    python scripts/train_tabm.py --config configs/uc04_v3.json --windows 60 120 [--context] [--seed 20260917]
    python scripts/train_tabm.py --config configs/uc04_v3.json --windows 60 --subset dryrun --epochs 1

Writes artifacts/tabular/tabm_numeric/W<W>/h<h>/ (tabm_context with --context) and
reports/tabular_validation/<name>.csv like the other tabular models. GPU if available.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from uc04.calibration import Platt
from uc04.columns import window_cols
from uc04.hub import hub_from_config
from uc04.provenance import make_provenance
from uc04.runtime import common_args, load_data, samples_inputs, setup
from uc04.tabm import HORIZONS, TabMMulti
from uc04.tabular import TabularData, combo_name, fit_rows, run_combo


class OneHorizon:
    """Scores of one horizon of a TabMMulti, with the interface run_combo expects."""

    def __init__(self, multi: TabMMulti, k: int):
        self.multi, self.k, self.columns = multi, k, multi.columns

    def score(self, X):
        return self.multi.logits(X)[:, self.k]


def main(argv=None) -> int:
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--windows", nargs="+", type=int)
    ap.add_argument("--context", action="store_true", help="numeric + case-context columns (samples v3)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--bootstrap", type=int)
    args = ap.parse_args(argv)
    cfg, samples, work, push, subset = setup(args)
    model = "tabm_context" if args.context else "tabm_numeric"
    windows = args.windows or list(cfg.features.windows_seconds)
    seed = cfg.seed if args.seed is None else args.seed
    boot = cfg.evaluation.bootstrap_repeats_validation if args.bootstrap is None else args.bootstrap
    hub = hub_from_config(cfg, work, push=push)
    prov = make_provenance(cfg, "train_tabm", inputs=samples_inputs(samples),
                           extra={"subset": args.subset, "bootstrap": boot, "seed": seed, "epochs": args.epochs,
                                  "batch_size": args.batch_size, "model": model})
    groups = ("numeric", "context") if args.context else ("numeric",)
    t_all = time.time()
    for W in windows:
        if all((work / "artifacts" / "tabular" / model / f"W{W}" / f"h{h}" / "done.json").exists() for h in HORIZONS):
            print(f"W={W}: done already")
            continue
        cols = window_cols(W, groups)
        t0 = time.time()
        labels, feats, events = load_data(samples, cols, subset)
        data = TabularData(W, labels, feats, events)
        tr = labels["train"]
        Y = tr[[f"y_{h}" for h in HORIZONS]].to_numpy(np.int8)
        rows = tr["eligible"].to_numpy(bool) & (Y != -1).any(1)
        print(f"W={W}: {model} on {rows.sum():,} train rows x {len(cols)} columns", flush=True)
        multi = TabMMulti(cols, seed, epochs=args.epochs, batch_size=args.batch_size).fit(
            feats["train"][rows], Y[rows], tr["subjectid"].to_numpy()[rows])
        multi.model.cpu()
        multi.device = "cpu"
        print(f"W={W}: trained in {time.time() - t0:.0f} s, best epoch {multi.best_epoch}", flush=True)
        ca, fc = labels["calibration"], feats["calibration"]
        L_cal = multi.logits(fc)
        for k, h in enumerate(HORIZONS):
            mc = fit_rows(ca, h)
            y_cal = ca[f"y_{h}"].to_numpy()[mc]
            fitted = (OneHorizon(multi, k), Platt().fit(L_cal[mc, k], y_cal), "ok")
            row, written = run_combo(cfg, model, data, h, work, bootstrap=boot, provenance=prov, fitted=fitted)
            local = [f"artifacts/{p}" if p.startswith("tabular/") else p for p in written]
            hub.register(*local)
            hub.push(local, message=f"{combo_name(model, W, h)}")
            print(f"  {combo_name(model, W, h):<24} sens {row['event_sensitivity']:.3f} "
                  f"FA/h {row['false_alarms_per_hour']:.2f}  AUROC {row['auroc']:.3f}", flush=True)
    print(f"done in {time.time() - t_all:.0f} s; outputs in {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
