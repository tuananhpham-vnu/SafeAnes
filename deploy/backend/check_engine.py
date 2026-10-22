"""Check the live ward engine against the offline test probabilities, and time it. Runs the models: Kaggle only.

    python check_engine.py --data <bundle of export_ward.py> [--out check_engine.json]

1. Bundle: every bed has its case and vitals files, the inputs of every model are the bundle's inputs.
2. Live = offline: for every bed and every row, the probability of each model and horizon computed by the engine
   (app.ward.Ward in live mode, the same code the server runs) equals the probability written by the training run
   (reference/<model>.parquet, uc04.tabular.predict_test). Tolerance 1e-6.
3. Speed: one tick that reaches a new 30 s row on all beds (50 rows x 4 models x 5 horizons), median of 20 ticks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.models import HORIZONS  # noqa: E402
from app.ward import Ward  # noqa: E402

TOL = 1e-6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="check_engine.json")
    args = ap.parse_args()
    data = Path(args.data)
    t0 = time.time()
    ward = Ward(data, "live")
    rep: dict = {"beds": len(ward.beds), "models": list(ward.models), "load_s": round(time.time() - t0, 2)}
    for m in ward.models.values():
        assert m.columns == ward.inputs, f"{m.name}: inputs differ from the bundle"

    # speed first, on fresh caches: every tick advances each bed by one row
    n_rows = min(len(b.time) for b in ward.beds.values())
    ticks = []
    for i in range(min(20, n_rows)):
        s = time.perf_counter()
        ward.ensure({b: i for b in ward.beds})
        ticks.append(time.perf_counter() - s)
    rep["tick_s_median"] = float(np.median(ticks[1:])) if len(ticks) > 1 else float(ticks[0])
    rep["tick_s_max"] = float(np.max(ticks))

    # then the whole cases
    s = time.perf_counter()
    ward.ensure({b: len(bed.time) - 1 for b, bed in ward.beds.items()})
    rep["all_rows_s"] = round(time.perf_counter() - s, 2)
    rep["rows"] = int(sum(len(b.time) for b in ward.beds.values()))

    worst = {}
    for name in ward.models:
        ref = pd.read_parquet(data / "reference" / f"{name}.parquet").set_index(["caseid", "time"])
        d = []
        for b, bed in ward.beds.items():
            live = ward.tracks[(b, name)].prob
            r = ref.reindex(pd.MultiIndex.from_arrays([np.full(len(bed.time), bed.caseid), bed.time]))
            off = r[[f"p_{h}" for h in HORIZONS]].to_numpy(float)
            both = np.isfinite(live) & np.isfinite(off)
            assert (np.isfinite(live) == np.isfinite(off)).all(), f"{name} bed {b}: eligible rows differ"
            d.append(np.abs(live[both] - off[both]))
        dd = np.concatenate(d)
        worst[name] = {"max_abs_diff": float(dd.max()), "n": int(dd.size)}
    rep["live_vs_offline"] = worst
    rep["ok"] = all(v["max_abs_diff"] <= TOL for v in worst.values())
    Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
