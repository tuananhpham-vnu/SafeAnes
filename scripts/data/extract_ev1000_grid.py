"""Device haemodynamics (EV1000 / Vigileo) on the 30-s prediction grid of samples v3, for checking the
component forecasts against measurements that are never model inputs (week 14 plan, A5).

  python scripts/data/extract_ev1000_grid.py [--splits calibration validation]

Reads only the cached raw tracks in data/vitaldb_full/raw (no download; a missing file is skipped). Each grid time t
takes the last device value at or before t, at most 60 s old (safeanes.signals.causal_sample: no interpolation).
Output: data/ev1000_grid/ev1000_grid.parquet with caseid, time, device, dev_sv, dev_co, dev_svr, dev_svv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from safeanes.data import read_numeric, decode_csv
from safeanes.signals import causal_sample

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/vitaldb_full/raw"
TRACKS = {"EV1000": {"dev_sv": "EV1000/SV", "dev_co": "EV1000/CO", "dev_svr": "EV1000/SVR", "dev_svv": "EV1000/SVV"},
          "Vigileo": {"dev_sv": "Vigileo/SV", "dev_co": "Vigileo/CO", "dev_svv": "Vigileo/SVV"}}


def read_track(tid: str):
    p = RAW / f"{tid}.csv.gz"
    if not p.exists():
        return None
    t, v = read_numeric(decode_csv(p.read_bytes()))
    return t, np.where(np.isfinite(v) & (v > 0), v, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["calibration", "validation"])
    ap.add_argument("--samples", type=Path, default=ROOT / "data/samples_v3")
    ap.add_argument("--out", type=Path, default=ROOT / "data/ev1000_grid")
    args = ap.parse_args()
    trk = pd.read_csv(ROOT / "data/vitaldb_full/meta/uc04_tracks.csv")
    rows = []
    for split in args.splits:
        lab = pd.read_parquet(args.samples / "labels" / f"{split}.parquet", columns=["caseid", "time", "in_main_cohort"])
        lab = lab[lab["in_main_cohort"]]
        for cid, g in lab.groupby("caseid"):
            tc = trk[trk["caseid"] == cid].set_index("tname")["tid"]
            for device, names in TRACKS.items():
                if names["dev_sv"] not in tc:
                    continue
                grid = g["time"].to_numpy(float)
                rec = {"caseid": cid, "time": grid, "device": device}
                for col, name in names.items():
                    tr = read_track(tc[name]) if name in tc else None
                    rec[col] = causal_sample(tr[0], tr[1], grid, 60)[0] if tr is not None else np.nan
                d = pd.DataFrame(rec)
                if d["dev_sv"].notna().sum() >= 20:
                    rows.append(d)
                break  # one device per case (EV1000 first)
        print(split, "cases with a device:", sum(r["caseid"].iloc[0] in set(lab["caseid"]) for r in rows), flush=True)
    out = pd.concat(rows, ignore_index=True)
    for c in ("dev_sv", "dev_co", "dev_svr", "dev_svv"):
        if c not in out:
            out[c] = np.nan
    args.out.mkdir(parents=True, exist_ok=True)
    out.astype({c: "float32" for c in ("dev_sv", "dev_co", "dev_svr", "dev_svv")}).to_parquet(
        args.out / "ev1000_grid.parquet", index=False)
    print(out.groupby("device")["caseid"].nunique().to_dict(), len(out))


if __name__ == "__main__":
    main()
