"""Recomputed PPV, `bt_ppv30_w{W}` (plan section 3.3, fast method).

ppv30(e) uses beats with avail in (e - 30, e]: NaN if < 5 beats, any rejected
beat, or period CV > 10 %; else (max PP - min PP) / mean(max, min) * 100.
Prediction times are on a 30 s grid, so ppv30 is computed once per grid point
(including up to W/30 - 1 points before the first prediction time) and
bt_ppv30_w{W}(t) = nanmedian of the W/30 most recent values, NaN when not
ventilated (6 <= RR <= 40 and EtCO2 >= 10 at t).
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd


def beat_table(beats: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
    """Beat table as a DataFrame with named columns, sorted by `avail`."""
    df = pd.DataFrame(np.asarray(beats, float), columns=[str(c) for c in columns])
    return df.sort_values("avail", kind="stable").reset_index(drop=True)


def ppv30_at(beats: pd.DataFrame, ends: np.ndarray, sub: float = 30.0, min_beats: int = 5,
             max_period_cv: float = 0.10) -> np.ndarray:
    """ppv30 for every sub-window end in `ends`."""
    ends = np.asarray(ends, float)
    out = np.full(len(ends), np.nan)
    if beats.empty:
        return out
    avail = beats["avail"].to_numpy(float)
    pp = beats["pp"].to_numpy(float)
    period = beats["period"].to_numpy(float)
    rejected = np.concatenate(([0], np.cumsum(beats["reject"].to_numpy() != 0)))
    s1 = np.concatenate(([0.0], np.cumsum(period)))
    s2 = np.concatenate(([0.0], np.cumsum(period ** 2)))
    lo = np.searchsorted(avail, ends - sub, side="right")
    hi = np.searchsorted(avail, ends, side="right")
    n = hi - lo
    ok = (n >= min_beats) & (rejected[hi] - rejected[lo] == 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (s1[hi] - s1[lo]) / n
        var = np.maximum((s2[hi] - s2[lo]) / n - mean ** 2, 0.0)
        ok &= np.sqrt(var) / mean <= max_period_cv
    for i in np.flatnonzero(ok):
        seg = pp[lo[i]:hi[i]]
        mx, mn = seg.max(), seg.min()
        out[i] = (mx - mn) / ((mx + mn) / 2.0) * 100.0
    return out


def ppv30_features(times: np.ndarray, start: float, beats: pd.DataFrame, rr_current: np.ndarray,
                   etco2_current: np.ndarray, windows: Sequence[int] = (30, 60, 90, 120), *,
                   sub: int = 30, min_beats: int = 5, max_period_cv: float = 0.10,
                   rr_range: Sequence[float] = (6, 40), min_etco2: float = 10) -> pd.DataFrame:
    """bt_ppv30_w{W} for prediction times `times` (must be start + 30k, k >= 1)."""
    t = np.asarray(times, float)
    k = np.rint((t - start) / sub).astype(int)
    if len(t) and (not np.allclose(t, start + sub * k) or k.min() < 1):
        raise ValueError("prediction times must be start + 30k with k >= 1")
    max_lag = max(windows) // sub - 1
    kmax = int(k.max()) if len(k) else 0
    grid_k = np.arange(1 - max_lag, kmax + 1)
    ppv = ppv30_at(beats, start + sub * grid_k, sub, min_beats, max_period_cv)
    offset = max_lag  # position of k = 1 in grid_k
    rr = np.asarray(rr_current, float)
    et = np.asarray(etco2_current, float)
    ventilated = (rr >= rr_range[0]) & (rr <= rr_range[1]) & (et >= min_etco2)
    out = pd.DataFrame({"time": t})
    for W in windows:
        m = W // sub
        idx = (k - 1 + offset)[:, None] - np.arange(m)[None, :]  # m most recent grid points
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN windows
            val = np.nanmedian(ppv[idx], axis=1) if len(t) else np.zeros(0)
        out[f"bt_ppv30_w{W}"] = np.where(ventilated, val, np.nan).astype(np.float32)
    return out
