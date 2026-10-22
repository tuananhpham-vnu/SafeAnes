"""ECG_II + PLETH beat features (uc04.ecg_ppg) on the 30-s prediction grid of samples v3, one shard of the cases.

  python scripts/data/extract_ecg_ppg.py --tids ecg_ppg_tids.csv --samples <samples_v3> --shard 0 --n-shards 10 \
      --out out [--workers 4] [--limit N] [--qc 3]

Needs internet (VitalDB open API). Raw waveforms are streamed and discarded. Outputs:
  out/ecgppg_shard<i>.parquet   caseid, time, ep_* (every main-cohort grid time of train, calibration, validation)
  out/ecgppg_qc_shard<i>.csv    per case: n R peaks, % valid RR, n PPG pulses, % R with PAT, median PAT, error
  out/qc/<caseid>.png           10 s of ECG / PPG with the detected R peaks and PPG feet (first --qc cases)
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import gzip
import io
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from uc04 import ecg_ppg as ep

API = "https://api.vitaldb.net"


def download(tid: str) -> pd.DataFrame:
    for attempt in range(5):
        try:
            req = Request(f"{API}/{tid}", headers={"User-Agent": "SafeAnes-UC04-research/0.1"})
            with urlopen(req, timeout=300) as r:
                payload = r.read()
            raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
            return pd.read_csv(io.BytesIO(raw))
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))


def wave_times(frame: pd.DataFrame) -> np.ndarray:
    """VitalDB waveform CSV: Time is only filled at a few anchor rows; interpolate on the row index
    (= safeanes.waveform.wave_times)."""
    t = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(t))
    if len(known) < 2:
        raise ValueError("waveform needs at least two time anchors")
    return np.interp(np.arange(len(t)), known, t[known])


def load(tid):
    if not isinstance(tid, str) or not tid:
        return None, None
    f = download(tid)
    return wave_times(f), pd.to_numeric(f.iloc[:, 1], errors="coerce").to_numpy(float)


def qc_plot(path, t_e, ecg, t_p, ppg, eb, pb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref = t_e if t_e is not None else t_p
    t0 = ref[len(ref) // 2]
    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    for a, tt, x, marks, name in ((ax[0], t_e, ecg, eb["time"], "ECG II"), (ax[1], t_p, ppg, pb.get("foot_time"), "PLETH")):
        if tt is None:
            continue
        m = (tt >= t0) & (tt < t0 + 10)
        a.plot(tt[m], x[m], lw=0.6)
        mk = np.asarray(marks if marks is not None else [])
        mk = mk[(mk >= t0) & (mk < t0 + 10)]
        a.plot(mk, np.interp(mk, tt[m], x[m]) if m.any() else mk, "r.", ms=6)
        a.set_ylabel(name)
    ax[1].set_xlabel("giây (đỏ: đỉnh R / chân sóng PPG phát hiện được)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def process(rec: dict, grid: np.ndarray, qc_dir: Path | None):
    cid = int(rec["caseid"])
    try:
        t_e, ecg = load(rec.get("ecg_tid"))
        t_p, ppg = load(rec.get("ppg_tid"))
        eb, pb = ep.beat_tables(t_e, ecg, t_p, ppg)
        feats = ep.window_features(eb, pb, grid)
        feats.insert(0, "time", grid)
        feats.insert(0, "caseid", cid)
        if qc_dir is not None:
            qc_plot(qc_dir / f"{cid}.png", t_e, ecg, t_p, ppg, eb, pb)
        info = {"caseid": cid, "n_r": len(eb), "valid_rr": float(eb["valid"].mean()) if len(eb) else np.nan,
                "n_ppg": len(pb), "pat_frac": float(eb["pat"].notna().mean()) if len(eb) else np.nan,
                "pat_median_ms": float(1e3 * eb["pat"].median()) if len(eb) else np.nan, "error": ""}
        return feats, info
    except Exception as e:  # one bad case must not stop the shard
        return None, {"caseid": cid, "error": repr(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tids", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qc", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "qc").mkdir(exist_ok=True)

    grids = []
    for split in ("train", "calibration", "validation"):
        lab = pd.read_parquet(args.samples / "labels" / f"{split}.parquet", columns=["caseid", "time", "in_main_cohort"])
        grids.append(lab.loc[lab["in_main_cohort"], ["caseid", "time"]])
    grids = pd.concat(grids)
    grid_of = {c: g["time"].to_numpy(float) for c, g in grids.groupby("caseid")}
    tids = pd.read_csv(args.tids)
    tids = tids[tids["caseid"].isin(grid_of)].sort_values("caseid")
    tids = tids[np.arange(len(tids)) % args.n_shards == args.shard]
    if args.limit:
        tids = tids.head(args.limit)
    recs = tids.to_dict("records")
    print(f"shard {args.shard}/{args.n_shards}: {len(recs)} cases", flush=True)

    parts, infos, t0 = [], [], time.time()
    out_p = args.out / f"ecgppg_shard{args.shard}.parquet"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process, r, grid_of[int(r["caseid"])], args.out / "qc" if i < args.qc else None)
                for i, r in enumerate(recs)]
        for n, fut in enumerate(as_completed(futs), 1):
            feats, info = fut.result()
            infos.append(info)
            if feats is not None:
                parts.append(feats)
            if info["error"]:
                print(f"ERROR {info['caseid']}: {info['error']}", flush=True)
            if n % 25 == 0 or n == len(recs):
                rate = n / (time.time() - t0)
                print(f"{n}/{len(recs)} done, {sum(bool(i['error']) for i in infos)} errors, "
                      f"~{(len(recs) - n) / rate / 60:.0f} min left", flush=True)
                if parts:
                    pd.concat(parts, ignore_index=True).to_parquet(out_p, index=False)
                pd.DataFrame(infos).to_csv(args.out / f"ecgppg_qc_shard{args.shard}.csv", index=False)
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(out_p, index=False)
    pd.DataFrame(infos).to_csv(args.out / f"ecgppg_qc_shard{args.shard}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
