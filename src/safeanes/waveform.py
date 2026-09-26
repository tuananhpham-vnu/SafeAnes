"""Beat-level features from the SNUADC/ART arterial waveform (500 Hz).

The raw waveform is never stored: extract_beats.py streams one case, calls `beat_table`
and `grid_features`, and keeps only the 2 s grid. Pulse-contour quantities here are
PROXIES (uncalibrated, relative within a case). They are not validated measurements
of stroke volume or vascular resistance; see reports/EDA for agreement with EV1000/Vigileo.

Causality: a beat is only "available" at `avail` = its end (next onset) + FILTER_MARGIN
seconds. The zero-phase filter and the onset search both look ahead within that margin,
so grid features at time t use only samples at <= t.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

FS = 500
FILTER_MARGIN = 1.0
BEAT_COLUMNS = ["onset", "avail", "sbp", "dbp", "map", "pp", "period", "dpdt_max",
                "sys_area", "sys_time", "valid"]
GRID_COLUMNS = ["bt_sbp", "bt_dbp", "bt_map", "bt_pp", "bt_hr", "bt_dpdt", "bt_sys_area",
                "bt_sv_lz", "bt_co_lz", "bt_svr_lz", "bt_ppv", "bt_spv", "bt_valid_frac", "bt_age"]


def wave_times(frame):
    """VitalDB waveform CSV: Time is only filled at a few anchor rows; interpolate on index."""
    time = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(time))
    if len(known) < 2:
        raise ValueError("Waveform needs at least two time anchors")
    return np.interp(np.arange(len(time)), known, time[known])


def beat_table(times, pressure, fs=FS):
    """Detect beats (systolic peaks, foot = minimum before each peak) and per-beat features."""
    pressure = np.asarray(pressure, float)
    finite = np.isfinite(pressure)
    if finite.sum() < fs * 10:
        return pd.DataFrame(columns=BEAT_COLUMNS)
    filled = pd.Series(pressure).interpolate(limit=fs // 10, limit_area="inside").to_numpy()
    work = np.where(np.isfinite(filled), filled, 0.0)
    smooth = sosfiltfilt(butter(4, 15, fs=fs, output="sos"), work)
    smooth[~np.isfinite(filled)] = np.nan
    peaks, _ = find_peaks(np.nan_to_num(smooth, nan=-1e3), distance=int(.33 * fs), prominence=8)
    if len(peaks) < 3:
        return pd.DataFrame(columns=BEAT_COLUMNS)
    # Foot of each beat: minimum between previous peak and this peak.
    onsets = np.array([p0 + np.nanargmin(smooth[p0:p1]) if np.isfinite(smooth[p0:p1]).any() else p0
                       for p0, p1 in zip(peaks[:-1], peaks[1:])])
    derivative = np.gradient(smooth) * fs
    rows = []
    for start, end in zip(onsets[:-1], onsets[1:]):
        segment = smooth[start:end]
        if len(segment) < 2 or not np.isfinite(segment).all():
            continue
        peak = int(np.argmax(segment))
        sbp, dbp = float(segment[peak]), float(segment[0])
        period = (end - start) / fs
        # Ejection time via Bazett-style estimate (robust fallback to notch detection).
        sys_n = min(len(segment) - 1, max(peak + 1, int(.37 * np.sqrt(period) * fs)))
        sys_area = float(np.sum(segment[:sys_n] - dbp) / fs)
        pp = sbp - dbp
        mean = float(segment.mean())
        valid = (20 <= dbp < mean < sbp <= 300) and (10 <= pp <= 150) and (.3 <= period <= 2.0) \
            and np.ptp(pressure[start:end][np.isfinite(pressure[start:end])]) < 200
        rows.append((times[start], times[min(end, len(times) - 1)] + FILTER_MARGIN, sbp, dbp, mean,
                     pp, period, float(np.nanmax(derivative[start:start + peak + 1])),
                     sys_area, sys_n / fs, bool(valid)))
    return pd.DataFrame(rows, columns=BEAT_COLUMNS)


def grid_features(beats, grid, window=10.0, variation_window=30.0, max_age=10.0):
    """Causal aggregation of VALID beats onto `grid` (times). Median over `window` seconds;
    PPV/SPV over `variation_window`. Uses only beats with avail <= t."""
    grid = np.asarray(grid, float)
    out = np.full((len(grid), len(GRID_COLUMNS)), np.nan, dtype="float32")
    if beats.empty:
        out[:, GRID_COLUMNS.index("bt_age")] = np.inf
        return out
    beats = beats.sort_values("avail")
    avail = beats.avail.to_numpy(float)
    valid = beats.valid.to_numpy(bool)
    values = beats[["sbp", "dbp", "map", "pp", "period", "dpdt_max", "sys_area"]].to_numpy(float)
    vt, vv = avail[valid], values[valid]
    hi_all = np.searchsorted(avail, grid, side="right")
    lo_all = np.searchsorted(avail, grid - window, side="right")
    hi = np.searchsorted(vt, grid, side="right")
    lo = np.searchsorted(vt, grid - window, side="right")
    lo_var = np.searchsorted(vt, grid - variation_window, side="right")
    for i in range(len(grid)):
        total = hi_all[i] - lo_all[i]
        age = grid[i] - vt[hi[i] - 1] if hi[i] > 0 else np.inf
        out[i, 13] = age
        if total:
            out[i, 12] = (hi[i] - lo[i]) / total
        if hi[i] - lo[i] < 3 or age > max_age:
            continue
        sbp, dbp, mbp, pp, period, dpdt, area = np.median(vv[lo[i]:hi[i]], axis=0)
        hr = 60 / period
        sv = pp / (sbp + dbp)          # Liljestrand-Zander, arbitrary units
        co = sv * hr
        out[i, :10] = (sbp, dbp, mbp, pp, hr, dpdt, area, sv, co, mbp / co)
        span = vv[lo_var[i]:hi[i]]
        if len(span) >= 5:
            out[i, 10] = (span[:, 3].max() - span[:, 3].min()) / span[:, 3].mean() * 100
            out[i, 11] = span[:, 0].max() - span[:, 0].min()
    return out


def case_grid(start, end, step=2.0):
    """Grid on the same lattice as sample_case (start + k*step), covering [start, end]."""
    return np.arange(start, end + 1e-9, step)
