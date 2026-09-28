"""Window features (research plan 2.1.4 and 2.2.2) from one preprocessed case (data/prep_v1).

Every feature at decision time t uses only data in the window (t - W, t]:
* numeric: grid cells with time <= t (values are already causal: last valid sample, <= 30 s old);
* beats: accepted beats (reject == 0) whose `avail` (end of beat + filter margin) is <= t.

Column names: <signal>_<stat>_w<W>, e.g. map_mean_w60, bt_dpdt_max_slope_w120. Values that do
not depend on W (current value, MAP drop vs baseline) have no suffix. Slopes are per minute.
Pulse-contour SV/CO/SVR are relative proxies (Liljestrand-Zander), not calibrated measurements.
"""

import numpy as np
import pandas as pd

WINDOWS = (30, 60, 90, 120)
CADENCE = 30.0
NUMERIC_STATS = ("mean", "std", "min", "max", "slope", "missing")
BEAT_SIGNALS = ("dpdt_max", "sys_area", "ejection_time", "notch_rel", "decay_tau", "sv", "co", "svr")
BEAT_STATS = ("mean", "std", "slope")
PPV_MIN_BEATS = 5
PPV_MAX_PERIOD_CV = 0.10
VENTILATED_RR = (6.0, 40.0)
VENTILATED_ETCO2 = 10.0
EXTRAPOLATE_MINUTES = 5.0
EXTRAPOLATE_RANGE = (30.0, 150.0)
RISK_BAND = (65.0, 75.0)


def decision_times(case, cadence=CADENCE):
    """t = start + k * cadence (k >= 1) inside the surgery; the label step picks eligible ones."""
    start = float(case["start"])
    end = start + len(case["label_map"])
    return start + cadence * np.arange(1, int((end - start) // cadence) + 1)


def _window_moments(t, x, lo, hi):
    """Stats of x over index ranges [lo, hi) ignoring NaN, via prefix sums (vectorised)."""
    ok = np.isfinite(x)
    xv, tv = np.where(ok, x, 0.0), np.where(ok, t, 0.0)
    c = lambda a: np.r_[0.0, np.cumsum(a)]
    n, s, ss = c(ok.astype(float)), c(xv), c(xv * xv)
    st, stt, stx = c(tv), c(tv * tv), c(tv * xv)
    take = lambda p: p[hi] - p[lo]
    cnt, sx, sxx, sxt, stt_, stx_ = take(n), take(s), take(ss), take(st), take(stt), take(stx)
    with np.errstate(all="ignore"):
        mean = sx / cnt
        std = np.sqrt(np.maximum(sxx / cnt - mean ** 2, 0))
        cov = stx_ / cnt - (sxt / cnt) * mean
        var_t = stt_ / cnt - (sxt / cnt) ** 2
        slope = np.where((cnt >= 2) & (var_t > 0), cov / var_t * 60.0, np.nan)
    mean[cnt == 0] = np.nan
    std[cnt == 0] = np.nan
    return cnt, mean, std, slope


def _window_extrema(x, lo, hi):
    mins, maxs = np.full(len(lo), np.nan), np.full(len(lo), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        seg = x[a:b]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            mins[i], maxs[i] = seg.min(), seg.max()
    return mins, maxs


def numeric_features(case, times, windows=WINDOWS):
    names = [str(c) for c in case["channels"]]
    values = np.asarray(case["values"], float)
    start, step = float(case["start"]), float(case["step"])
    grid = start + step * np.arange(len(values))
    hi = np.searchsorted(grid, times, side="right")
    out = {}
    col = {n: values[:, j] for j, n in enumerate(names)}
    with np.errstate(all="ignore"):
        col["pp"] = col["sbp"] - col["dbp"]
        col["shock_index"] = col["hr"] / col["sbp"]
    cur = np.maximum(hi - 1, 0)
    for n, x in col.items():
        out[f"{n}_current"] = np.where(hi > 0, x[cur], np.nan)
    baseline = float(case["baseline"][0])
    with np.errstate(all="ignore"):
        out["map_drop_pct"] = (baseline - out["map_current"]) / baseline * 100 if np.isfinite(baseline) \
            else np.full(len(times), np.nan)
    for w in windows:
        lo = np.searchsorted(grid, np.asarray(times) - w, side="right")
        expected = np.maximum(hi - lo, 1)
        for n, x in col.items():
            cnt, mean, std, slope = _window_moments(grid, x, lo, hi)
            mins, maxs = _window_extrema(x, lo, hi)
            for stat, v in zip(NUMERIC_STATS, (mean, std, mins, maxs, slope, 1 - cnt / expected)):
                out[f"{n}_{stat}_w{w}"] = v
        m = col["map"]
        band = np.isfinite(m) & (m >= RISK_BAND[0]) & (m < RISK_BAND[1])
        prefix = np.r_[0, np.cumsum(band)]
        out[f"map_time_65_75_w{w}"] = (prefix[hi] - prefix[lo]) * step
        out[f"map_extrap_w{w}"] = np.clip(out["map_current"] + out[f"map_slope_w{w}"] * EXTRAPOLATE_MINUTES,
                                          *EXTRAPOLATE_RANGE)
    return out


def beat_frame(case):
    beats = pd.DataFrame(case["beats"], columns=[str(c) for c in case["beat_table_columns"]])
    if beats.empty:
        return beats
    with np.errstate(all="ignore"):
        beats["sv"] = beats.pp / (beats.sbp + beats.dbp)     # Liljestrand-Zander, relative
        beats["co"] = beats.sv * beats.hr
        beats["svr"] = beats["map"] / beats.co
    return beats.sort_values("avail", kind="stable").reset_index(drop=True)


def beat_features(case, times, numeric, windows=WINDOWS):
    """Per-beat values aggregated over accepted beats in the window, PPV and valid-beat share.
    PPV needs >= 5 beats, no rejected beat in the window, period CV <= 10 % and controlled
    ventilation (RR 6-40 /min and EtCO2 >= 10 mmHg at t)."""
    beats = beat_frame(case)
    times = np.asarray(times, float)
    out = {}
    names = [f"bt_{s}_{st}_w{w}" for w in windows for s in BEAT_SIGNALS for st in BEAT_STATS]
    names += [f"bt_{k}_w{w}" for w in windows for k in ("ppv", "valid_frac", "n_beats")]
    if beats.empty:
        return {n: np.full(len(times), np.nan) for n in names}
    avail = beats.avail.to_numpy(float)
    good = beats.reject.to_numpy() == 0
    vals = beats[list(BEAT_SIGNALS)].to_numpy(float)
    pp, period = beats.pp.to_numpy(float), beats.period.to_numpy(float)
    hi = np.searchsorted(avail, times, side="right")
    rr, etco2 = numeric["rr_current"], numeric["etco2_current"]
    ventilated = (rr >= VENTILATED_RR[0]) & (rr <= VENTILATED_RR[1]) & (etco2 >= VENTILATED_ETCO2)
    for w in windows:
        lo = np.searchsorted(avail, times - w, side="right")
        res = {n: np.full(len(times), np.nan) for n in names if n.endswith(f"_w{w}")}
        for i, (a, b) in enumerate(zip(lo, hi)):
            total = b - a
            res[f"bt_n_beats_w{w}"][i] = total
            if not total:
                continue
            g = good[a:b]
            res[f"bt_valid_frac_w{w}"][i] = g.mean()
            if g.sum() < 3:
                continue
            v, ta = vals[a:b][g], avail[a:b][g]
            mean = np.nanmean(v, axis=0)
            std = np.nanstd(v, axis=0)
            dt = ta - ta.mean()
            with np.errstate(all="ignore"):
                slope = np.nansum(dt[:, None] * (v - mean), axis=0) / np.sum(dt * dt) * 60.0
            for k, s in enumerate(BEAT_SIGNALS):
                res[f"bt_{s}_mean_w{w}"][i], res[f"bt_{s}_std_w{w}"][i] = mean[k], std[k]
                res[f"bt_{s}_slope_w{w}"][i] = slope[k]
            if g.all() and g.sum() >= PPV_MIN_BEATS and ventilated[i]:
                per = period[a:b]
                if per.std() / per.mean() <= PPV_MAX_PERIOD_CV:
                    p = pp[a:b]
                    res[f"bt_ppv_w{w}"][i] = (p.max() - p.min()) / ((p.max() + p.min()) / 2) * 100
        out.update(res)
    return out


def case_features(case, times=None, windows=WINDOWS):
    """DataFrame: one row per decision time, `time` + all window features (float32)."""
    times = decision_times(case) if times is None else np.asarray(times, float)
    numeric = numeric_features(case, times, windows)
    beats = beat_features(case, times, numeric, windows)
    frame = pd.DataFrame({**numeric, **beats})
    frame = frame.astype(np.float32)
    frame.insert(0, "time", times)
    return frame
