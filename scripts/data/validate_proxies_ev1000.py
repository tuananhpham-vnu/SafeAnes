"""Check pulse-contour proxies (plan 2.2.2) against EV1000 in cases where EV1000 was recorded.

Proxies are relative (uncalibrated), so agreement is judged WITHIN each case: Spearman rho
between the proxy and EV1000 on a 60 s grid (level) and between their 5-minute changes (trend).
PPV is compared with EV1000 SVV (both are dynamic preload indices). Descriptive only.

  python scripts/data/validate_proxies_ev1000.py [--prep data/prep_v1] [--limit N]

Outputs reports/PREP/proxy_vs_ev1000.csv (per case x pair) and proxy_vs_ev1000_summary.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from safeanes.data import fetch_csv, read_numeric
from safeanes.features import beat_frame
from safeanes.signals import causal_sample

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "data/vitaldb_full"
PAIRS = {"sv": "EV1000/SV", "co": "EV1000/CO", "svr": "EV1000/SVR", "ppv": "EV1000/SVV"}
STEP, MIN_POINTS = 60.0, 20


def proxy_series(beats, grid):
    """60 s mean of accepted-beat proxies ending at each grid time (PPV: per-window swing)."""
    good = beats[beats.reject.eq(0)]
    avail = good.avail.to_numpy()
    hi = np.searchsorted(avail, grid, side="right")
    lo = np.searchsorted(avail, grid - STEP, side="right")
    out = {k: np.full(len(grid), np.nan) for k in PAIRS}
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b - a < 10:
            continue
        seg = good.iloc[a:b]
        for k in ("sv", "co", "svr"):
            out[k][i] = seg[k].mean()
        p = seg.pp.to_numpy()
        out["ppv"][i] = (p.max() - p.min()) / ((p.max() + p.min()) / 2) * 100
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, default=ROOT / "data/prep_v1")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    trks = pd.read_csv(FULL / "meta/trks.csv.gz") if (FULL / "meta/trks.csv.gz").exists() \
        else fetch_csv("trks", FULL / "meta")
    ev = trks[trks.tname.isin(PAIRS.values())].drop_duplicates(["caseid", "tname"])
    done = {int(p.stem) for p in (args.prep / "cases").glob("*.npz") if p.stem.isdigit()}
    caseids = sorted(set(ev.caseid) & done)[:args.limit]
    print(f"ca có EV1000 và đã tiền xử lý: {len(caseids)}", flush=True)
    rows = []
    for caseid in caseids:
        case = dict(np.load(args.prep / "cases" / f"{caseid}.npz"))
        beats = beat_frame(case)
        if beats.empty:
            continue
        start = float(case["start"])
        grid = np.arange(start + STEP, start + len(case["label_map"]), STEP)
        proxies = proxy_series(beats, grid)
        tids = ev[ev.caseid.eq(caseid)].set_index("tname").tid
        for key, tname in PAIRS.items():
            if tname not in tids:
                continue
            t, v = read_numeric(fetch_csv(tids[tname], FULL / "raw"))
            v = np.where(np.isfinite(v) & (v > 0), v, np.nan)
            ref, _ = causal_sample(t, v, grid, 60)
            x = proxies[key]
            ok = np.isfinite(x) & np.isfinite(ref)
            if ok.sum() < MIN_POINTS:
                continue
            level = pd.Series(x[ok]).corr(pd.Series(ref[ok]), method="spearman")
            dx, dr = x[5:] - x[:-5], ref[5:] - ref[:-5]
            ok5 = np.isfinite(dx) & np.isfinite(dr)
            trend = pd.Series(dx[ok5]).corr(pd.Series(dr[ok5]), method="spearman") if ok5.sum() >= MIN_POINTS else np.nan
            rows.append({"caseid": caseid, "proxy": key, "ev1000": tname, "points": int(ok.sum()),
                         "rho_level": level, "rho_trend_5min": trend})
    table = pd.DataFrame(rows)
    out = ROOT / "reports/PREP"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "proxy_vs_ev1000.csv", index=False)
    summary = table.groupby(["proxy", "ev1000"]).agg(
        ca=("caseid", "nunique"),
        rho_level_median=("rho_level", "median"),
        rho_level_q25=("rho_level", lambda s: s.quantile(.25)),
        rho_level_q75=("rho_level", lambda s: s.quantile(.75)),
        rho_trend_median=("rho_trend_5min", "median"),
        ca_rho_level_gt_0_5=("rho_level", lambda s: float((s > 0.5).mean())),
    ).round(3).reset_index()
    summary.to_csv(out / "proxy_vs_ev1000_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
