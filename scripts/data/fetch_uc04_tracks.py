"""Download every UC04-related numeric track (config.UC04_TRACKS) + labs for eligible cases.

Resumable: fetch_csv reuses cached files after a SHA-256 check, so rerunning only fetches
what is missing. Raw files go to the same cache as the 7 core tracks (data/vitaldb_full/raw).

  python scripts/data/fetch_uc04_tracks.py [--groups hemodynamic vasoactive ...] [--workers 8]

Outputs:
  data/vitaldb_full/meta/uc04_tracks.csv   caseid, group, tname, tid (what should exist)
  data/vitaldb_full/meta/labs.csv.gz       VitalDB labs (dt = seconds from case start)
  data/vitaldb_full/fetch_log.csv          tid, status, error
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

import pandas as pd

from safeanes.config import UC04_TRACKS
from safeanes.data import fetch_csv

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "data/vitaldb_full"
RAW, META = FULL / "raw", FULL / "meta"


def track_index(groups):
    manifest = pd.read_csv(META / "cohort_manifest.csv")
    eligible = set(manifest.loc[manifest.eligible, "caseid"])
    trks = pd.read_csv(META / "trks.csv.gz")
    group_of = {name: group for group in groups for name in UC04_TRACKS[group]}
    index = trks[trks.caseid.isin(eligible) & trks.tname.isin(group_of)].copy()
    index["group"] = index.tname.map(group_of)
    return index[["caseid", "group", "tname", "tid"]].sort_values(["caseid", "tname"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", default=list(UC04_TRACKS), choices=list(UC04_TRACKS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("1..16 workers")

    fetch_csv("labs", META)
    index = track_index(args.groups)
    index.to_csv(META / "uc04_tracks.csv", index=False)
    tids = sorted(set(index.tid))
    todo = [t for t in tids if not (RAW / f"{t}.csv.gz").exists()]
    print(f"tracks={len(tids)} cached={len(tids) - len(todo)} todo={len(todo)}", flush=True)

    def download(tid):
        try:
            fetch_csv(tid, RAW)
            return tid, "ok", ""
        except Exception as error:  # logged; rerun resumes
            return tid, "error", repr(error)

    log, start = [], time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, t) for t in todo]
        for i, future in enumerate(as_completed(futures), 1):
            log.append(future.result())
            if i % 2000 == 0 or i == len(todo):
                rate = i / max(time.monotonic() - start, 1e-9)
                errors = sum(s == "error" for _, s, _ in log)
                print(f"{i}/{len(todo)} lỗi={errors} còn ~{(len(todo) - i) / rate / 60:.0f} phút", flush=True)
    log = pd.DataFrame(log, columns=["tid", "status", "error"])
    log.to_csv(FULL / "fetch_log.csv", index=False)
    errors = int(log.status.eq("error").sum()) if len(log) else 0
    print(f"xong; lỗi {errors} (chạy lại để tải tiếp)", flush=True)


if __name__ == "__main__":
    main()
