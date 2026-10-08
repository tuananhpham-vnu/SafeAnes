"""NB02 (02a/02b/02c): train baselines and LightGBM, one result per (model, W, h).

    python scripts/train_tabular.py --models map_threshold map_logistic      # 02a
    python scripts/train_tabular.py --models lgbm_numeric                    # 02b
    python scripts/train_tabular.py --models lgbm_wave                       # 02c
    python scripts/train_tabular.py --models lgbm_wave --windows 60 --horizons 300 --subset dryrun

Each combination writes only its own files (plan 6.3) and a done.json; a rerun
skips combinations already done locally or on Hugging Face. map_threshold does
not depend on W and runs once per horizon.
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from uc04.columns import WINDOWS
from uc04.hub import hub_from_config
from uc04.provenance import make_provenance
from uc04.runtime import common_args, load_data, samples_inputs, setup
from uc04.tabular import BASE_MODELS, MODELS, TabularData, combo_dir, combo_name, model_columns, run_combo


def main(argv=None) -> int:
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--models", nargs="+", default=list(BASE_MODELS), choices=MODELS)
    ap.add_argument("--windows", nargs="+", type=int)
    ap.add_argument("--horizons", nargs="+", type=int)
    ap.add_argument("--bootstrap", type=int, help="bootstrap repeats (default: evaluation.bootstrap_repeats_validation)")
    args = ap.parse_args(argv)
    cfg, samples, work, push, subset = setup(args)
    windows = args.windows or list(cfg.features.windows_seconds)
    horizons = args.horizons or list(cfg.labels.horizons_seconds)
    boot = cfg.evaluation.bootstrap_repeats_validation if args.bootstrap is None else args.bootstrap
    hub = hub_from_config(cfg, work, push=push)  # local work/artifacts/tabular/... -> repo tabular/...
    remote_done = hub.done_set("tabular")
    prov = make_provenance(cfg, "train_tabular", inputs=samples_inputs(samples),
                           extra={"subset": args.subset, "bootstrap": boot})
    print(f"work dir {work}; models {args.models}; windows {windows}; horizons {horizons}; bootstrap {boot}")

    rows, t_all = [], time.time()
    for wi, W in enumerate(windows):
        todo = [(m, h) for h in horizons for m in args.models
                if not (m == "map_threshold" and wi > 0)
                and not (work / "artifacts" / combo_dir(m, W, h) / "done.json").exists()
                and combo_dir(m, W, h) not in remote_done]
        if not todo:
            print(f"W={W}: nothing to do")
            continue
        cols = sorted({c for m, _ in todo for c in model_columns(cfg, m, W)})
        t0 = time.time()
        labels, feats, events = load_data(samples, cols, subset)
        data = TabularData(W, labels, feats, events)
        print(f"W={W}: loaded {sum(len(v) for v in labels.values()):,} rows x {len(cols)} columns "
              f"in {time.time() - t0:.0f} s; {len(todo)} combinations")
        for m, h in todo:
            t1 = time.time()
            row, written = run_combo(cfg, m, data, h, work, bootstrap=boot, provenance=prov)
            rows.append(row)
            local = [f"artifacts/{p}" if p.startswith("tabular/") else p for p in written]
            hub.register(*local)
            hub.push(local, message=f"{combo_name(m, W, h)} ({row['status']})")
            if row["status"] == "ok":
                print(f"  {combo_name(m, W, h):<24} sens {row['event_sensitivity']:.3f} "
                      f"[{row.get('event_sensitivity_lo', float('nan')):.3f}, {row.get('event_sensitivity_hi', float('nan')):.3f}]"
                      f"  FA/h {row['false_alarms_per_hour']:.2f}  AUROC {row['auroc']:.3f}  "
                      f"thr {row['threshold']:.3f}{' (budget not met)' if row['budget_not_met'] else ''}"
                      f"  {time.time() - t1:.0f} s", flush=True)
    if rows:
        pd.set_option("display.width", 200)
        df = pd.DataFrame(rows)
        show = [c for c in ("model", "window", "horizon", "status", "event_sensitivity", "false_alarms_per_hour",
                            "alarm_ppv", "auroc", "prevalence", "budget_not_met") if c in df]
        print("\n" + df[show].to_string(index=False))
    print(f"done in {time.time() - t_all:.0f} s; outputs in {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
