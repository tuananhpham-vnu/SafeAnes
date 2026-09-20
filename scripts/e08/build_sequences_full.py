"""Build the raw-sequence cache for the full VitalDB cohort (input for the DL family).

data/vitaldb_full/cases holds the 30s decision rows with 143 window statistics; a sequence model
needs the numeric grid behind them. This writes one float32 array per case, [steps, 14] =
7 resampled tracks + 7 observation ages, on exactly the grid sample_case produces, so
FullSequenceStore.window(caseid, time) lines up with the decision row at that time.

  python scripts/e08/build_sequences_full.py [--workers 8] [--limit N]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import pandas as pd

from safeanes.config import Protocol, TRACKS
from safeanes.data import write_json
from safeanes.sequences import build_case_sequence

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/sequences_full"
# Tracked copy; the data/ original is not in git. Same rows, same columns.
MANIFEST = ROOT / "reports/E07/cohort_manifest.csv"
RAW_DIRS = (ROOT / "data/vitaldb_development300/raw", ROOT / "data/vitaldb_full/raw")


def build_one(record):
    caseid = int(record["caseid"])
    target = OUT / f"{caseid}.npy"
    try:
        if target.exists():
            array = np.load(target, mmap_mode="r", allow_pickle=False)
            return caseid, {"start": float(record["_start"]), "steps": int(array.shape[0]),
                            "subjectid": int(record["subjectid"])}, None
        array, start = build_case_sequence(record, RAW_DIRS, Protocol())
        temp = target.with_suffix(".tmp.npy")
        np.save(temp, array, allow_pickle=False)
        temp.replace(target)
        return caseid, {"start": start, "steps": int(array.shape[0]),
                        "subjectid": int(record["subjectid"])}, None
    except Exception as error:
        return caseid, None, repr(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    rows = manifest[manifest.eligible].to_dict("records")
    for r in rows:
        r["_start"] = float(np.ceil(r["opstart"]))
    if args.limit:
        rows = rows[:args.limit]
    print(f"cases={len(rows)} out={OUT}", flush=True)

    cases, errors, done, start = {}, {}, 0, time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_one, r) for r in rows]
        for future in as_completed(futures):
            caseid, record, error = future.result()
            done += 1
            if error:
                errors[caseid] = error
                print(f"ERROR {caseid} {error}", flush=True)
            else:
                cases[str(caseid)] = record
            if done % 100 == 0:
                rate = done / max(time.monotonic() - start, 1e-9)
                print(f"{done}/{len(rows)} lỗi={len(errors)} "
                      f"còn ~{(len(rows)-done)/max(rate,1e-9)/60:.0f} phút", flush=True)

    write_json(OUT / "index.json", {"version": "uc04-sequences-full-v1",
        "protocol": asdict(Protocol()), "protocol_hash": Protocol().digest(),
        "tracks": list(TRACKS), "cases": cases, "errors": {str(k): v for k, v in errors.items()}})
    total = sum(p.stat().st_size for p in OUT.glob("*.npy"))
    print(f"xong {len(cases)}/{len(rows)}; lỗi {len(errors)}; {total/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
