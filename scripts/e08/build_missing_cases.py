"""Rebuild the preprocessed cases that data/vitaldb_full/cases is missing.

E07 processed all 3626 eligible cases, but only part of that output is tracked in git and the
csv_cases copy is gone, so E08 v3 silently ran on a subset. This redoes the original
preprocessing (no approximation: same build_case, same Protocol) for whatever is absent, pulling
raw tracks from the VitalDB API into data/vitaldb_full/raw when they are not already cached.

  python scripts/e08/build_missing_cases.py [--workers 8] [--limit N]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_full_vitaldb import DATA, build_one  # noqa: E402

# reports/E07 is the tracked copy: same 6388 rows and same columns as the data/ original,
# which .gitignore drops, so a fresh clone still has the tid_*/opstart/opend a rebuild needs.
MANIFEST = ROOT / "reports/E07/cohort_manifest.csv"


def pending(limit=None):
    eligible = pd.read_csv(MANIFEST).query("eligible")
    have = {int(p.stem) for p in (DATA / "cases").glob("*.joblib")}
    rows = eligible[~eligible.caseid.isin(have)].to_dict("records")
    return len(eligible), len(have), rows[:limit] if limit else rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    (DATA / "cases").mkdir(parents=True, exist_ok=True)
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    total, have, rows = pending(args.limit)
    print(f"eligible={total} có sẵn={have} cần dựng={len(rows)}", flush=True)
    if not rows:
        return
    done, errors, start = 0, {}, time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_one, r): int(r["caseid"]) for r in rows}
        for future in as_completed(futures):
            caseid, _, error = future.result()
            done += 1
            if error:
                errors[caseid] = error
                print(f"ERROR {caseid} {error}", flush=True)
            if done % 25 == 0:
                rate = done / max(time.monotonic() - start, 1e-9)
                print(f"{done}/{len(rows)} lỗi={len(errors)} "
                      f"còn ~{(len(rows)-done)/max(rate,1e-9)/60:.0f} phút", flush=True)
    print(f"xong {done-len(errors)}/{len(rows)}; lỗi {len(errors)}", flush=True)
    if errors:
        pd.DataFrame({"caseid": list(errors), "error": list(errors.values())}).to_csv(
            ROOT / "reports/E08/build_errors.csv", index=False)


if __name__ == "__main__":
    main()
