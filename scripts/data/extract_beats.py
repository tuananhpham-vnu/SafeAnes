"""Stream SNUADC/ART per eligible case -> beat features on the 2 s grid; raw waveform discarded.

Grid lattice = ceil(opstart) + 2k (same as sample_case / sequences_full), extended back to the
start of the recording so post-induction hypotension before incision is covered.

  python scripts/data/extract_beats.py [--workers 4] [--limit N] [--snippets 6]

Outputs data/beats_full/<caseid>.npz (times, features, columns),
data/beats_full/index.json, and QC snippets (raw 20 s + detected beats) under snippets/.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import io
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from safeanes.config import WAVEFORM_TRACKS
from safeanes.data import API, write_json
from safeanes.waveform import GRID_COLUMNS, beat_table, grid_features, wave_times

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data/vitaldb_full/meta"
OUT = ROOT / "data/beats_full"


def download(tid):
    for attempt in range(4):
        try:
            request = Request(f"{API}/{tid}", headers={"User-Agent": "SafeAnes-UC04-research/0.1"})
            with urlopen(request, timeout=180) as response:
                payload = response.read()
            raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
            return pd.read_csv(io.BytesIO(raw)), hashlib.sha256(payload).hexdigest()
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def process(record, snippet, indexed=False):
    caseid, tid = int(record["caseid"]), record["tid"]
    target = OUT / f"{caseid}.npz"
    if target.exists() and indexed:
        return caseid, {"cached": True}, None
    try:
        frame, sha = download(tid)
        times, pressure = wave_times(frame), frame.iloc[:, 1].to_numpy(float)
        del frame
        beats = beat_table(times, pressure)
        anchor = float(np.ceil(record["opstart"]))
        first = anchor - 2 * np.floor((anchor - times[0]) / 2)
        grid = np.arange(first, times[-1] + 1e-9, 2.0)
        features = grid_features(beats, grid)
        if snippet:
            mid = int(len(times) * .5)
            sl = slice(mid, mid + 20 * 500)
            seg = beats[(beats.onset >= times[sl.start]) & (beats.onset <= times[sl.stop - 1])]
            np.savez_compressed(OUT / "snippets" / f"{caseid}.npz", t=times[sl], p=pressure[sl],
                                onset=seg.onset.to_numpy(), sbp=seg.sbp.to_numpy(),
                                dbp=seg.dbp.to_numpy(), valid=seg.valid.to_numpy())
        temp = OUT / f"{caseid}.tmp.npz"
        np.savez_compressed(temp, times=grid, features=features, columns=np.array(GRID_COLUMNS))
        temp.replace(target)
        info = {"tid": tid, "sha256": sha, "n_beats": int(len(beats)),
                "valid_frac": float(beats.valid.mean()) if len(beats) else 0.0,
                "steps": int(len(grid)), "start": float(grid[0]) if len(grid) else None,
                "retrieved_at": datetime.now(timezone.utc).isoformat()}
        return caseid, info, None
    except Exception as error:
        return caseid, None, repr(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--snippets", type=int, default=6)
    args = parser.parse_args()
    (OUT / "snippets").mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(META / "cohort_manifest.csv")
    trks = pd.read_csv(META / "trks.csv.gz")
    art = trks[trks.tname.eq(WAVEFORM_TRACKS["art"])].drop_duplicates("caseid")
    rows = manifest[manifest.eligible].merge(art[["caseid", "tid"]], on="caseid")
    rows = rows.sort_values("caseid").to_dict("records")[:args.limit]
    index_path = OUT / "index.json"
    import json
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"cases": {}, "errors": {}}
    print(f"cases={len(rows)} out={OUT}", flush=True)
    done, start = 0, time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, r, i < args.snippets, str(int(r["caseid"])) in index["cases"])
                   for i, r in enumerate(rows)]
        for future in as_completed(futures):
            caseid, info, error = future.result()
            done += 1
            if error:
                index["errors"][str(caseid)] = error
                print(f"ERROR {caseid} {error}", flush=True)
            elif not info.get("cached"):
                index["cases"][str(caseid)] = info
                index["errors"].pop(str(caseid), None)
            if done % 100 == 0:
                rate = done / max(time.monotonic() - start, 1e-9)
                print(f"{done}/{len(rows)} lỗi={len(index['errors'])} còn ~{(len(rows) - done) / rate / 60:.0f} phút",
                      flush=True)
                write_json(index_path, {**index, "version": "uc04-beats-v1", "columns": GRID_COLUMNS})
    write_json(index_path, {**index, "version": "uc04-beats-v1", "columns": GRID_COLUMNS})
    print(f"xong {len(index['cases'])}; lỗi {len(index['errors'])}", flush=True)


if __name__ == "__main__":
    main()
