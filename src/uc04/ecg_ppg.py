"""ECG + PPG beat features on the 30-s prediction grid (week 15), following HypoBridCast (2026): its gain over
ART alone came from adding ECG and PPG. Kept small for real time: hand-made beat features for the tabular model,
no waveform network.

Per beat (SNUADC/ECG_II and SNUADC/PLETH, 500 Hz):
  R peaks      band-pass 5-15 Hz, squared derivative, 150 ms integration, adaptive threshold, refined to |ECG| max
  PPG pulses   band-pass 0.5-8 Hz; systolic peak, foot = minimum before the peak; amplitude, rise time
  PAT          pulse arrival time = R peak to PPG foot (pair_pat; includes the monitor PLETH delay)
A beat is available `MARGIN` s after its last sample (zero-phase filters look a little ahead), so a feature at t
only uses beats available by t: causal.

Window features at t over (t - W, t] (`ep_` prefix, W = 120 s, baseline window 600 s):
  ep_hr_ecg_mean, ep_sdnn, ep_rmssd                    ECG heart rate and its variability (ms)
  ep_pat_mean, ep_pat_std, ep_pat_slope, ep_pat_rel600  PAT (ms, ms/min) and PAT now / PAT of the last 10 min
  ep_ppg_amp_rel600, ep_ppg_amp_cv, ep_ppg_resp_var    PPG amplitude now / last 10 min, its CV, and the mean
                                                       (max - min) / mean over 10-s blocks (ΔPOP-like, preload)
  ep_ppg_rise_mean                                      PPG foot-to-peak time (ms)
  ep_ecg_valid_frac, ep_ppg_valid_frac, ep_pat_frac     quality: fraction of the window covered by valid beats
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

FS = 500
MARGIN = 1.0
W, BASE = 120.0, 600.0
COLUMNS = ["ep_hr_ecg_mean", "ep_sdnn", "ep_rmssd", "ep_pat_mean", "ep_pat_std", "ep_pat_slope", "ep_pat_rel600",
           "ep_ppg_amp_rel600", "ep_ppg_amp_cv", "ep_ppg_resp_var", "ep_ppg_rise_mean",
           "ep_ecg_valid_frac", "ep_ppg_valid_frac", "ep_pat_frac"]
DESCRIBE = {
    "ep_hr_ecg_mean": ("Nhịp tim từ ECG, trung bình 120 s", "nhịp/phút"),
    "ep_sdnn": ("SDNN: độ lệch chuẩn RR (biến thiên nhịp tim, trương lực giao cảm/phó giao cảm)", "ms"),
    "ep_rmssd": ("RMSSD: biến thiên RR liên tiếp (phó giao cảm)", "ms"),
    "ep_pat_mean": ("PAT: thời gian từ đỉnh R tới chân sóng PPG (giảm khi HA/trương lực mạch tăng)", "ms"),
    "ep_pat_std": ("Độ lệch chuẩn PAT trong 120 s", "ms"),
    "ep_pat_slope": ("Độ dốc PAT (mỗi phút)", "ms/phút"),
    "ep_pat_rel600": ("PAT 120 s / PAT 10 phút (>1: mạch giãn, HA giảm)", "tỷ số"),
    "ep_ppg_amp_rel600": ("Biên độ PPG 120 s / 10 phút (tưới máu ngoại vi, giãn mạch)", "tỷ số"),
    "ep_ppg_amp_cv": ("Hệ số biến thiên biên độ PPG trong 120 s", "–"),
    "ep_ppg_resp_var": ("Biến thiên biên độ PPG theo hô hấp (ΔPOP, đáp ứng dịch)", "–"),
    "ep_ppg_rise_mean": ("Thời gian chân → đỉnh sóng PPG", "ms"),
    "ep_ecg_valid_frac": ("Tỷ lệ thời gian có nhịp ECG hợp lệ", "–"),
    "ep_ppg_valid_frac": ("Tỷ lệ thời gian có nhịp PPG hợp lệ", "–"),
    "ep_pat_frac": ("Tỷ lệ nhịp ECG ghép được với PPG", "–"),
}


def _fill(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Short gaps (<= 0.1 s) interpolated; returns (signal with zeros in long gaps, finite mask)."""
    s = pd.Series(np.asarray(x, float)).interpolate(limit=fs // 10, limit_area="inside").to_numpy()
    ok = np.isfinite(s)
    return np.where(ok, s, 0.0), ok


def r_peaks(ecg: np.ndarray, fs: int = FS) -> np.ndarray:
    """Sample indices of R peaks (Pan-Tompkins-like)."""
    x, ok = _fill(ecg, fs)
    if ok.sum() < 10 * fs:
        return np.array([], int)
    bp = sosfiltfilt(butter(2, [5, 15], btype="band", fs=fs, output="sos"), x)
    e = np.gradient(bp) ** 2
    k = int(0.15 * fs)
    integ = np.convolve(e, np.ones(k) / k, mode="same")
    integ[~ok] = 0
    # adaptive threshold: 30 % of the 98th percentile of each 10-s block
    blk = 10 * fs
    thr = np.repeat([np.percentile(integ[i:i + blk], 98) for i in range(0, len(integ), blk)], blk)[:len(integ)]
    pk, _ = find_peaks(integ, height=0.3 * thr, distance=int(0.25 * fs))
    a = np.abs(bp)
    w = int(0.075 * fs)
    out = np.array([max(0, p - w) + int(np.argmax(a[max(0, p - w):p + w])) for p in pk], int)
    return np.unique(out[ok[out]])


def ppg_pulses(ppg: np.ndarray, fs: int = FS) -> pd.DataFrame:
    """Per pulse: foot index, peak index, amplitude, rise time (s)."""
    x, ok = _fill(ppg, fs)
    if ok.sum() < 10 * fs:
        return pd.DataFrame(columns=["foot", "peak", "amp", "rise"])
    f = sosfiltfilt(butter(2, [0.5, 8], btype="band", fs=fs, output="sos"), x)
    q75, q25 = np.percentile(f[ok], [75, 25])
    pk, _ = find_peaks(f, distance=int(0.3 * fs), prominence=0.3 * (q75 - q25))
    df = np.gradient(f) * fs
    rows = []
    for p0, p1 in zip(pk[:-1], pk[1:]):
        if not ok[p0:p1 + 1].all():
            continue
        lo = p0 + int(np.argmin(f[p0:p1]))
        if lo >= p1 - 1:
            continue
        # intersecting tangent: tangent at the steepest upstroke meets the level of the minimum
        m = lo + int(np.argmax(df[lo:p1]))
        foot = m - (f[m] - f[lo]) / df[m] * fs if df[m] > 0 else float(lo)
        foot = float(np.clip(foot, lo, p1))
        rows.append((foot, p1, f[p1] - f[lo], (p1 - foot) / fs))
    d = pd.DataFrame(rows, columns=["foot", "peak", "amp", "rise"])
    return d[(d["amp"] > 0) & (d["rise"].between(0.05, 0.5))].reset_index(drop=True)


def beat_tables(t_ecg, ecg, t_ppg, ppg, fs: int = FS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ECG beats: time, rr, valid, pat, avail), (PPG pulses: time, amp, rise, avail); times in seconds."""
    r = r_peaks(ecg, fs) if ecg is not None else np.array([], int)
    rt = np.asarray(t_ecg)[r] if len(r) else np.array([])
    rr = np.r_[np.nan, np.diff(rt)]
    med = pd.Series(rr).rolling(9, center=True, min_periods=3).median().to_numpy()
    valid = (rr >= 0.3) & (rr <= 2.0) & (np.abs(rr - med) <= 0.3 * med)
    eb = pd.DataFrame({"time": rt, "rr": rr, "valid": valid})
    pp = ppg_pulses(ppg, fs) if ppg is not None else pd.DataFrame(columns=["foot", "peak", "amp", "rise"])
    tp = np.asarray(t_ppg) if t_ppg is not None else np.array([])
    pb = pd.DataFrame({"time": tp[pp["peak"].astype(int)] if len(pp) else [],
                       "foot_time": np.interp(pp["foot"].to_numpy(float), np.arange(len(tp)), tp) if len(pp) else [],
                       "amp": pp["amp"].to_numpy(float), "rise": pp["rise"].to_numpy(float)})
    pb["avail"] = pb["time"] + MARGIN
    eb["pat"] = pair_pat(rt, pb["foot_time"].to_numpy(float)) if len(pb) and len(eb) else np.nan
    eb["avail"] = eb["time"] + eb["pat"].fillna(0).to_numpy(float) + MARGIN
    return eb, pb


PAT_RANGE = (0.05, 1.5)  # s; the monitor PLETH lags ECG by about 0.5-0.7 s on top of the physiological PAT
PAT_TOL = 0.12
PAT_BLOCK = 300  # R peaks per block for the dominant delay


def pair_pat(rt: np.ndarray, ft: np.ndarray) -> np.ndarray:
    """PAT per R peak: the PPG foot nearest R + d0, within PAT_TOL, where d0 is the dominant R-to-foot delay of
    the block (mode of every R-to-foot delay in PAT_RANGE, 10-ms bins). Unlike "first foot after R", this stays
    right when PAT is longer than the RR interval. Includes the monitor's constant PLETH delay, so changes within
    a case (slope, ratio to the last 10 min) are the physiological signal."""
    pat = np.full(len(rt), np.nan)
    if not len(rt) or not len(ft):
        return pat
    bins = np.arange(PAT_RANGE[0], PAT_RANGE[1] + 0.01, 0.01)
    for b0 in range(0, len(rt), PAT_BLOCK):
        r = rt[b0:b0 + PAT_BLOCK]
        lo, hi = np.searchsorted(ft, r + PAT_RANGE[0]), np.searchsorted(ft, r + PAT_RANGE[1])
        d = np.concatenate([ft[a:b] - x for a, b, x in zip(lo, hi, r)]) if (hi > lo).any() else np.array([])
        if len(d) < 20:
            continue
        h = np.convolve(np.histogram(d, bins)[0], np.ones(3), mode="same")
        d0 = bins[np.argmax(h)] + 0.005
        j = np.clip(np.searchsorted(ft, r + d0), 1, len(ft) - 1)
        near = np.where(np.abs(ft[j] - (r + d0)) < np.abs(ft[j - 1] - (r + d0)), ft[j], ft[j - 1])
        dd = near - r
        pat[b0:b0 + PAT_BLOCK] = np.where(np.abs(dd - d0) <= PAT_TOL, dd, np.nan)
    return pat


def _slope(t, v):
    m = np.isfinite(v)
    if m.sum() < 5:
        return np.nan
    t, v = t[m], v[m]
    tc = t - t.mean()
    den = (tc ** 2).sum()
    return float((tc * (v - v.mean())).sum() / den * 60) if den > 0 else np.nan


def window_features(eb: pd.DataFrame, pb: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """COLUMNS at each grid time from beats available in (t - W, t]."""
    ea, pa = eb["avail"].to_numpy(float), pb["avail"].to_numpy(float)
    eo, po = np.argsort(ea, kind="stable"), np.argsort(pa, kind="stable")
    ea, pa = ea[eo], pa[po]
    et, rr, ev, pat = (eb[c].to_numpy(float)[eo] for c in ("time", "rr", "valid", "pat"))
    ev = ev.astype(bool)
    pt, amp, rise = (pb[c].to_numpy(float)[po] for c in ("time", "amp", "rise"))
    out = np.full((len(grid), len(COLUMNS)), np.nan, np.float32)
    for i, t in enumerate(grid):
        e0, e1 = np.searchsorted(ea, t - W, "right"), np.searchsorted(ea, t, "right")
        b0 = np.searchsorted(ea, t - BASE, "right")
        p0, p1 = np.searchsorted(pa, t - W, "right"), np.searchsorted(pa, t, "right")
        q0 = np.searchsorted(pa, t - BASE, "right")
        v = ev[e0:e1]
        r = rr[e0:e1][v]
        rec = {}
        if len(r) >= 10:
            rec["ep_hr_ecg_mean"] = 60 / r.mean()
            rec["ep_sdnn"] = 1e3 * r.std(ddof=1)
            d = np.diff(rr[e0:e1])
            dv = v[1:] & v[:-1]
            rec["ep_rmssd"] = 1e3 * np.sqrt(np.mean(d[dv] ** 2)) if dv.sum() >= 5 else np.nan
        rec["ep_ecg_valid_frac"] = float(np.nansum(rr[e0:e1][v]) / W) if e1 > e0 else 0.0
        pw = pat[e0:e1][v]
        pw = pw[np.isfinite(pw)]
        rec["ep_pat_frac"] = len(pw) / max(v.sum(), 1) if e1 > e0 else 0.0
        if len(pw) >= 10:
            rec["ep_pat_mean"] = 1e3 * pw.mean()
            rec["ep_pat_std"] = 1e3 * pw.std(ddof=1)
            rec["ep_pat_slope"] = 1e3 * _slope(et[e0:e1][v][np.isfinite(pat[e0:e1][v])], pw)
            pb_ = pat[b0:e1][ev[b0:e1]]
            pb_ = pb_[np.isfinite(pb_)]
            rec["ep_pat_rel600"] = pw.mean() / pb_.mean() if len(pb_) >= 30 else np.nan
        a = amp[p0:p1]
        rec["ep_ppg_valid_frac"] = min(1.0, float(np.sum(np.diff(pt[p0:p1]).clip(0, 2.0)) / W)) if p1 - p0 > 1 else 0.0
        if len(a) >= 10:
            ab = amp[q0:p1]
            rec["ep_ppg_amp_rel600"] = a.mean() / ab.mean() if len(ab) >= 30 else np.nan
            rec["ep_ppg_amp_cv"] = a.std(ddof=1) / a.mean()
            blocks = np.floor((pt[p0:p1] - (t - W)) / 10)
            rv = [(g.max() - g.min()) / g.mean() for _, g in pd.Series(a).groupby(blocks) if len(g) >= 5]
            rec["ep_ppg_resp_var"] = float(np.mean(rv)) if rv else np.nan
            rec["ep_ppg_rise_mean"] = 1e3 * rise[p0:p1].mean()
        for j, c in enumerate(COLUMNS):
            if c in rec:
                out[i, j] = rec[c]
    return pd.DataFrame(out, columns=COLUMNS)
