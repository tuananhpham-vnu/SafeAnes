"""Numeric features for windows longer than prep_v1 provides (exploratory, W = 300, 600 s).

Recomputed from the causal 2 s grid `values` of cases/<caseid>.npz with the plan's
appendix-A definitions, over the grid cells with time in (t - W, t]:
mean / std / min / max / slope (per minute, least squares) / missing (fraction of
cells missing, over the cells of the window that lie inside the case) for map, sbp,
dbp, hr, spo2, etco2, rr, pp = sbp - dbp and shock_index = hr / sbp;
`map_time_65_75` = seconds with 65 <= MAP < 75 (prep_v1's half-open band);
`map_extrap` = map_current + slope * 5 min, clipped to [30, 150]. The W-independent
columns (`*_current`, `map_drop_pct`) are taken from prep_v1 unchanged.
The function is checked against prep_v1's own W = 60 and 120 columns before use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import SIG, STATS

BASE = ("map", "sbp", "dbp", "hr", "spo2", "etco2", "rr")


def grid_signals(values: np.ndarray, channels) -> pd.DataFrame:
    ch = [str(c) for c in channels]
    df = pd.DataFrame({s: values[:, ch.index(s)].astype(np.float64) for s in BASE})
    df["pp"] = df["sbp"] - df["dbp"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["shock_index"] = df["hr"] / df["sbp"]
    return df


def window_features(values: np.ndarray, channels, start: float, times: np.ndarray, W: int, step: float = 2.0,
                    ddof: int = 0) -> pd.DataFrame:
    g = grid_signals(values, channels)
    n_cells = int(round(W / step))
    tmin = pd.Series(np.arange(len(g)) * step / 60.0)             # grid time in minutes
    idx = np.rint((np.asarray(times, float) - start) / step).astype(int)  # cell exactly at t
    idx_c = np.clip(idx, 0, len(g) - 1)
    out = {}
    for s in SIG:
        x = g[s]
        r = x.rolling(n_cells, min_periods=1)
        cnt = x.notna().astype(float).rolling(n_cells, min_periods=1).sum()
        in_case = np.minimum(np.arange(1, len(g) + 1), n_cells).astype(float)  # window clipped at case start
        tt = tmin.where(x.notna())
        cov = tt.rolling(n_cells, min_periods=2).cov(x)
        var = tt.rolling(n_cells, min_periods=2).var()
        stats = {"mean": r.mean(), "std": r.std(ddof=ddof), "min": r.min(), "max": r.max(),
                 "slope": cov / var, "missing": 1.0 - cnt.to_numpy() / in_case}
        for st in STATS:
            v = np.asarray(stats[st])[idx_c]
            out[f"{s}_{st}_w{W}"] = np.where(idx >= 0, v, np.nan)
    m = g["map"]
    in_band = ((m >= 65) & (m < 75)).astype(float)
    out[f"map_time_65_75_w{W}"] = (in_band.rolling(n_cells, min_periods=1).sum() * step).to_numpy()[idx_c]
    return pd.DataFrame(out)


def add_extrap(df: pd.DataFrame, W: int, map_current: np.ndarray) -> pd.DataFrame:
    df[f"map_extrap_w{W}"] = np.clip(np.asarray(map_current, float) + df[f"map_slope_w{W}"].to_numpy() * 5.0, 30, 150)
    return df


def case_long_features(prep, caseid: int, times: np.ndarray, map_current: np.ndarray, windows) -> pd.DataFrame:
    """W-dependent numeric columns of one case for every W in `windows` (worker function)."""
    from .io import load_case
    z = load_case(prep, caseid, ["start", "values", "channels"])
    parts = [add_extrap(window_features(z["values"], z["channels"], float(z["start"]), times, W), W, map_current)
             for W in windows]
    return pd.concat(parts, axis=1).astype(np.float32)
