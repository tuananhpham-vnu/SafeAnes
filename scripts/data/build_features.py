"""Window features (plan 2.1.4 / 2.2.2) for every preprocessed case in data/prep_v1.

One row per decision time t = start + 30 k inside the surgery, features for W = 30/60/90/120 s
(see safeanes.features). No labels or splits here: the sample step joins them later on
(caseid, time). Resumable: cases with an existing parquet are skipped.

  python scripts/data/build_features.py [--prep data/prep_v1] [--workers 4] [--caseids 1 2]

Outputs:
  <prep>/features/<caseid>.parquet   time + features (float32), 1 file per case
  <prep>/features/columns.json       feature names grouped by source, windows, cadence
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np

from safeanes.data import write_json
from safeanes.features import CADENCE, WINDOWS, case_features

ROOT = Path(__file__).resolve().parents[2]


def build_one(path, out):
    caseid = int(path.stem)
    try:
        frame = case_features(dict(np.load(path)))
        frame.insert(0, "caseid", caseid)
        temp = out / f"{caseid}.tmp.parquet"
        frame.to_parquet(temp, index=False)
        temp.replace(out / f"{caseid}.parquet")
        return caseid, list(frame.columns), len(frame), None
    except Exception as error:
        return caseid, None, 0, repr(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, default=ROOT / "data/prep_v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--caseids", type=int, nargs="*")
    args = parser.parse_args()
    out = args.prep / "features"
    out.mkdir(parents=True, exist_ok=True)
    cases = sorted((p for p in (args.prep / "cases").glob("*.npz") if p.stem.isdigit()), key=lambda p: int(p.stem))
    if args.caseids:
        cases = [p for p in cases if int(p.stem) in set(args.caseids)]
    todo = [p for p in cases if not (out / f"{p.stem}.parquet").exists()]
    print(f"cases={len(cases)} todo={len(todo)} out={out}", flush=True)
    columns, rows, errors, start = None, 0, {}, time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_one, p, out) for p in todo]
        for i, future in enumerate(as_completed(futures), 1):
            caseid, cols, n, error = future.result()
            if error:
                errors[str(caseid)] = error
                print(f"ERROR {caseid} {error}", flush=True)
            else:
                columns, rows = columns or cols, rows + n
            if i % 200 == 0 or i == len(todo):
                rate = i / max(time.monotonic() - start, 1e-9)
                print(f"{i}/{len(todo)} lỗi={len(errors)} còn ~{(len(todo) - i) / rate / 60:.0f} phút", flush=True)
    if columns is None:
        done = sorted(out.glob("*.parquet"))
        if done:
            import pandas as pd
            columns = list(pd.read_parquet(done[0]).columns)
    if columns:
        feats = [c for c in columns if c not in ("caseid", "time")]
        write_json(out / "columns.json", {
            "windows_seconds": list(WINDOWS), "cadence_seconds": CADENCE, "n_features": len(feats),
            "numeric": [c for c in feats if not c.startswith("bt_")],
            "waveform": [c for c in feats if c.startswith("bt_")],
            "note": "No labels/splits; join on (caseid, time). Waveform ablation = drop 'waveform' columns."})
    write_json(out / "errors.json", errors)
    print(f"xong: {len(todo) - len(errors)} ca mới, {rows} dòng; lỗi {len(errors)}", flush=True)


if __name__ == "__main__":
    main()
