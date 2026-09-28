"""Preprocess every eligible VitalDB case (research plan V.2) into data/prep_v1.

Per case: numeric monitor on a 2 s grid (value/age/flag), baseline MAP/SBP, label-grade MAP
at 1 s with artefacts marked unknown, ART beat table with per-beat reject reasons, 1 s
waveform artefact mask, and a 100 Hz float16 waveform for deep learning. SNUADC/ART is
streamed from the VitalDB API and only the 100 Hz copy is kept. Resumable: finished cases
are skipped. After all cases, z-score statistics are computed on split == "train" only.

  python scripts/data/preprocess_vitaldb.py [--workers 3] [--limit N] [--caseids 1 2 3] [--stats-only]

Needs only the repo (reports/E07/cohort_manifest.csv is tracked) and Internet: VitalDB `trks`
and numeric tracks are downloaded to data/vitaldb_full on first use (SHA-256 checked after).
Kaggle: set --out to /kaggle/working/prep_v1; ~1 GB RAM per worker for long cases.

Outputs (data/prep_v1):
  cases/<caseid>.npz      arrays from safeanes.preprocess.process_case
  wave100/<caseid>.npy    float16, sample k at start + k / 100 s (start stored in the npz)
  qc.csv                  one row per case (coverage, artefact fractions, beats, lag, baseline)
  errors.json             cases that failed, with the exception
  normalization.json      train-split mean/std per numeric channel and the 100 Hz waveform
  prep.json               config, config hash, cohort, outputs
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from safeanes.data import fetch_csv, read_numeric, write_json
from safeanes.preprocess import AUX_TRACKS, CHANNELS, VERSION, WAVE_TRACK, PrepConfig, process_case
from safeanes.waveform import wave_times

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "data/vitaldb_full"
MANIFEST = ROOT / "reports/E07/cohort_manifest.csv"
OUT = ROOT / "data/prep_v1"


def track_ids():
    """tid of every needed track per case. Built from VitalDB `trks` (downloaded and cached on
    first use, e.g. on Kaggle where data/vitaldb_full/meta is not present)."""
    cached = FULL / "meta/trks.csv.gz"
    trks = pd.read_csv(cached) if cached.exists() else fetch_csv("trks", FULL / "meta")
    wanted = {ch.track for ch in CHANNELS.values()} | set(AUX_TRACKS.values())
    index = trks[trks.tname.isin(wanted)].drop_duplicates(["caseid", "tname"])
    art = trks[trks.tname.eq(WAVE_TRACK)].drop_duplicates("caseid").set_index("caseid").tid
    return {c: g.set_index("tname").tid.to_dict() for c, g in index.groupby("caseid")}, art.to_dict()


def load_wave(tid):
    # Same streaming download as extract_beats.py (resumable per case, raw not kept).
    import sys
    sys.path.insert(0, str(ROOT / "scripts/data"))
    from extract_beats import download
    frame, sha = download(tid)
    times = wave_times(frame)
    pressure = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(float)
    return (times, pressure), sha


def run_case(record, tids, wave_tid, out=OUT):
    caseid = int(record["caseid"])
    target = out / "cases" / f"{caseid}.npz"
    try:
        tracks = {**{k: ch.track for k, ch in CHANNELS.items()}, **AUX_TRACKS}
        raw = {}
        for key, tname in tracks.items():
            tid = tids.get(tname)
            if isinstance(tid, str):   # cached raw file if present, otherwise downloaded + cached
                raw[key] = read_numeric(fetch_csv(tid, FULL / "raw"))
        wave, sha = (None, None)
        if isinstance(wave_tid, str):
            wave, sha = load_wave(wave_tid)
        started = time.monotonic()
        arrays, wave100, qc = process_case(record, raw, wave)
        del wave
        if wave100 is not None:
            temp = out / "wave100" / f"{caseid}.tmp.npy"
            np.save(temp, wave100, allow_pickle=False)
            temp.replace(out / "wave100" / f"{caseid}.npy")
            qc["wave100_samples"] = int(len(wave100))
        temp = out / "cases" / f"{caseid}.tmp.npz"
        np.savez_compressed(temp, **arrays)
        temp.replace(target)
        qc.update({"split": record["split"], "wave_sha256": sha, "seconds": round(time.monotonic() - started, 1)})
        return caseid, qc, None
    except Exception as error:
        return caseid, None, repr(error)
    finally:
        gc.collect()


class Moments:
    def __init__(self):
        self.n, self.s, self.ss = 0, 0.0, 0.0

    def add(self, x):
        x = np.asarray(x, np.float64)
        x = x[np.isfinite(x)]
        self.n += len(x)
        self.s += float(x.sum())
        self.ss += float((x * x).sum())

    def result(self):
        if not self.n:
            return {"mean": None, "std": None, "n": 0}
        mean = self.s / self.n
        return {"mean": mean, "std": float(np.sqrt(max(self.ss / self.n - mean ** 2, 0))), "n": self.n}


def train_statistics(qc):
    """Mean/std on train-split cases only; the same numbers are then applied to every split."""
    numeric, wave = {}, Moments()
    for caseid in qc.loc[qc.split.eq("train"), "caseid"]:
        z = np.load(OUT / "cases" / f"{caseid}.npz")
        for j, name in enumerate(z["channels"]):
            numeric.setdefault(str(name), Moments()).add(z["values"][:, j])
        path = OUT / "wave100" / f"{caseid}.npy"
        if path.exists():
            w = np.load(path, mmap_mode="r").astype(np.float32)
            mask = np.repeat(z["wave_mask"] > 0, 100)[:len(w)]
            wave.add(w[:len(mask)][~mask])
    return {"scope": "split == train", "cases": int(qc.split.eq("train").sum()),
            "numeric": {k: m.result() for k, m in numeric.items()},
            "wave100": wave.result()}


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--caseids", type=int, nargs="*")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out
    for sub in ("cases", "wave100"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST)
    cohort = manifest[manifest.eligible].sort_values("caseid")
    if args.caseids:
        cohort = cohort[cohort.caseid.isin(args.caseids)]
    cohort = cohort.head(args.limit) if args.limit else cohort
    qc_path, err_path = OUT / "qc.csv", OUT / "errors.json"
    qc = pd.read_csv(qc_path) if qc_path.exists() else pd.DataFrame(columns=["caseid"])
    errors = json.loads(err_path.read_text(encoding="utf-8")) if err_path.exists() else {}
    done = set(qc.caseid.astype(int)) & {int(p.stem) for p in (OUT / "cases").glob("*.npz") if p.stem.isdigit()}

    if not args.stats_only:
        tids, art = track_ids()
        todo = [r for r in cohort.to_dict("records") if int(r["caseid"]) not in done]
        print(f"cases={len(cohort)} done={len(done)} todo={len(todo)} out={OUT}", flush=True)
        rows, start = [], time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_case, r, tids.get(int(r["caseid"]), {}), art.get(int(r["caseid"])), OUT)
                       for r in todo]
            for i, future in enumerate(as_completed(futures), 1):
                caseid, row, error = future.result()
                if error:
                    errors[str(caseid)] = error
                    print(f"ERROR {caseid} {error}", flush=True)
                else:
                    errors.pop(str(caseid), None)
                    rows.append(row)
                if i % 50 == 0 or i == len(todo):
                    qc = pd.concat([qc[~qc.caseid.isin([r["caseid"] for r in rows])], pd.DataFrame(rows)],
                                   ignore_index=True).sort_values("caseid")
                    qc.to_csv(qc_path, index=False)
                    write_json(err_path, errors)
                    rate = i / max(time.monotonic() - start, 1e-9)
                    print(f"{i}/{len(todo)} lỗi={len(errors)} còn ~{(len(todo) - i) / rate / 60:.0f} phút", flush=True)

    if not qc_path.exists():
        print("Không có ca nào được xử lý (caseid không nằm trong cohort eligible?)", flush=True)
        return
    qc = pd.read_csv(qc_path)
    if args.stats_only or (not args.caseids and not args.limit):
        write_json(OUT / "normalization.json", train_statistics(qc))
    cfg = PrepConfig()
    write_json(OUT / "prep.json", {
        "version": VERSION, "config": asdict(cfg), "config_hash": cfg.digest(),
        "manifest": str(MANIFEST.relative_to(ROOT)), "eligible_cases": int(manifest.eligible.sum()),
        "processed_cases": int(len(qc)), "errors": len(errors),
        "channels": {k: asdict(v) for k, v in CHANNELS.items()},
    })
    print(f"xong {len(qc)} ca; lỗi {len(errors)}", flush=True)


if __name__ == "__main__":
    main()
