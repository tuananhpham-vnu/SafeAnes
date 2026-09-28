"""UC04 preprocessing v1 (research plan V.2): synchronisation, artefact removal, resampling.

One case in, one set of aligned arrays out. Nothing here builds samples or labels; that is the
next step (windows, horizons, splits). What this module guarantees:

* Timeline. VitalDB puts every track on one clock (seconds from case start), so tracks are
  joined on it and cropped to the surgery interval [ceil(opstart), floor(opend)).
* Numeric monitor -> 2 s grid, last valid value at or before t, at most `max_age` old,
  never interpolated. Every grid cell carries a flag byte saying WHY a value is missing.
* Two artefact views. `causal` flags use only samples <= t and feed model inputs/masks.
  `label` flags may look ahead (a spike is only known to be a spike once it returns) and
  are used ONLY to mark label-grade MAP as unknown. True low MAP is never removed by range
  alone above 20 mmHg; a sustained drop is kept (only excursions that come back are spikes).
* ART waveform (500 Hz) -> per-beat table (onset, peak, dicrotic notch, dP/dt, areas, decay)
  with per-beat reject reasons, a 1 s artefact mask, and a 100 Hz copy for deep learning
  (causal anti-alias low-pass, then decimation by 5). Window features (stats over W, PPV,
  PP / shock index / MAP drop vs baseline) are built later from these arrays, not here.
  The 2 s beat grid below is used internally only (waveform-vs-monitor MAP check).

Pulse-contour SV/CO/SVR are PROXIES (relative within a case), see waveform.py and reports/EDA.
"""

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import statistics

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfilt, sosfiltfilt

from .signals import causal_sample, label_grid

VERSION = "uc04-prep-v1"
STEP = 2.0            # numeric / beat grid (s)
WAVE_FS = 500         # SNUADC/ART native rate
DL_FS = 100           # deep-learning waveform rate
FILTER_MARGIN = 1.0   # beat available at next onset + margin (zero-phase smoothing look-ahead)

# Numeric flag bits (per raw sample, carried to the grid from the last sample <= t).
RANGE, PP, JUMP, FLAT, WAVE, SQI, STALE, NOTRACK = 1, 2, 4, 8, 16, 32, 64, 128
FLAG_NAMES = {RANGE: "range", PP: "pulse_pressure", JUMP: "jump", FLAT: "flat", WAVE: "waveform",
              SQI: "sqi", STALE: "stale", NOTRACK: "no_track"}
# Waveform 1 s block bits.
W_MISSING, W_FLUSH, W_FLAT, W_DAMPED = 1, 2, 4, 8
# Beat reject bits.
B_SHAPE, B_ARTIFACT, B_IRREGULAR, B_DAMPED = 1, 2, 4, 8


@dataclass(frozen=True)
class Channel:
    track: str
    lo: float
    hi: float
    max_age: float
    checks: tuple = ()


ART_CHECKS = ("pp", "jump", "flat", "wave")
# Plausibility ranges follow reports/EDA/INSIGHTS.md section 2 (engineering limits, not diagnoses).
CHANNELS = {
    "map": Channel("Solar8000/ART_MBP", 20, 200, 30, ART_CHECKS),
    "sbp": Channel("Solar8000/ART_SBP", 30, 260, 30, ART_CHECKS),
    "dbp": Channel("Solar8000/ART_DBP", 10, 160, 30, ART_CHECKS),
    "hr": Channel("Solar8000/HR", 25, 200, 30, ("jump", "flat")),
    "spo2": Channel("Solar8000/PLETH_SPO2", 50, 100, 30),
    "etco2": Channel("Solar8000/ETCO2", 0, 80, 30),
    "rr": Channel("Solar8000/RR_CO2", 0, 60, 30),
    "nibp_mbp": Channel("Solar8000/NIBP_MBP", 20, 200, 600),
    "cvp": Channel("Solar8000/CVP", -5, 40, 60),
    "bis": Channel("BIS/BIS", 1, 100, 60, ("sqi",)),
    "mac": Channel("Primus/MAC", 0, 3, 60),
    "ppf_ce": Channel("Orchestra/PPF20_CE", 0, 12, 60),
    "rftn_ce": Channel("Orchestra/RFTN20_CE", 0, 20, 60),
    "peep": Channel("Primus/PEEP_MBAR", 0, 30, 120),
}
AUX_TRACKS = {"bis_sqi": "BIS/SQI", "nibp_sbp": "Solar8000/NIBP_SBP"}
WAVE_TRACK = "SNUADC/ART"


@dataclass(frozen=True)
class PrepConfig:
    version: str = VERSION
    step_seconds: float = STEP
    min_pulse_pressure: float = 10.0      # SBP - DBP below this: damped / flush / zeroing
    jump_threshold: float = 30.0          # |value - median(last 30 s)|, mmHg or bpm
    jump_reference_seconds: float = 30.0
    jump_max_hold_seconds: float = 60.0   # longer excursions are accepted as a real level change
    jump_return_tolerance: float = 15.0   # excursion counts as a spike if it comes back this close
    flat_seconds: float = 60.0            # identical consecutive values for this long = frozen
    wave_mismatch_mmhg: float = 15.0      # |beat MAP - monitor MAP|
    wave_trust_seconds: float = 300.0     # waveform only checks the monitor when it was mostly valid
    wave_trust_min_valid: float = 0.5
    flush_mmhg: float = 250.0
    flat_ptp_mmhg: float = 5.0
    artefact_dilate_seconds: int = 3
    damped_ratio: float = 0.5
    irregular_ratio: float = 0.25
    beat_window_seconds: float = 10.0
    ppv_window_seconds: float = 30.0
    ppv_max_period_cv: float = 0.10
    label_max_gap_seconds: int = 10

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ numeric artefacts
def _mask_times(flag_times, flag_mask, times, tolerance=2.0):
    """Value of a boolean raw-sample mask at `times` (last flag sample <= t, within tolerance)."""
    if len(flag_times) == 0:
        return np.zeros(len(times), bool)
    idx = np.searchsorted(flag_times, times, side="right") - 1
    ok = (idx >= 0) & (times - flag_times[np.maximum(idx, 0)] <= tolerance)
    return ok & flag_mask[np.maximum(idx, 0)]


def jump_flags(times, values, valid, cfg=PrepConfig()):
    """Causal and retrospective jump flags.

    causal: sample differs from the median of the last `jump_reference_seconds` of accepted
    samples by more than `jump_threshold`; held for at most `jump_max_hold_seconds`, after which
    the new level is accepted. retro: only excursions that came back (spikes) are marked.
    """
    n = len(values)
    causal, retro = np.zeros(n, bool), np.zeros(n, bool)
    history = deque()
    start = ref = None
    members = []
    for i in np.flatnonzero(valid):
        t, v = times[i], values[i]
        while history and history[0][0] <= t - cfg.jump_reference_seconds:
            history.popleft()
        if start is not None:
            if abs(v - ref) <= cfg.jump_return_tolerance:
                retro[members] = True
                start, members = None, []
            elif t - start <= cfg.jump_max_hold_seconds:
                causal[i] = True
                members.append(i)
                continue
            else:
                start, members = None, []
                history.clear()
        if start is None and len(history) >= 3:
            median = statistics.median([h[1] for h in history])
            if abs(v - median) > cfg.jump_threshold:
                start, ref, members = t, median, [i]
                causal[i] = True
                continue
        history.append((t, v))
    return causal, retro


def flat_flags(times, values, valid, cfg=PrepConfig()):
    """Runs of identical consecutive valid values. causal: from `flat_seconds` into the run;
    retro: the whole run once it lasted `flat_seconds`."""
    n = len(values)
    causal, retro = np.zeros(n, bool), np.zeros(n, bool)
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return causal, retro
    v, t = values[idx], times[idx]
    new_run = np.r_[True, v[1:] != v[:-1]]
    run_id = np.cumsum(new_run) - 1
    run_start = t[new_run][run_id]
    run_end = pd.Series(t).groupby(run_id).transform("max").to_numpy()
    causal[idx] = t - run_start >= cfg.flat_seconds
    retro[idx] = run_end - run_start >= cfg.flat_seconds
    return causal, retro


def clean_numeric(raw, cfg=PrepConfig(), wave_check=None):
    """raw: name -> (times, values). Returns name -> (times, values, causal_bits, label_bits).

    wave_check(times, causal: bool) -> bool mask of waveform disagreement/artefact at `times`.
    """
    out = {}
    for name, ch in CHANNELS.items():
        if name not in raw:
            continue
        t, v = (np.asarray(a, float) for a in raw[name])
        finite = np.isfinite(v)
        bits = np.where(finite & ((v < ch.lo) | (v > ch.hi)), RANGE, 0).astype(np.uint8)
        bits[~finite] |= RANGE
        if "sqi" in ch.checks and "bis_sqi" in raw:
            st, sv = raw["bis_sqi"]
            sqi, _ = causal_sample(st, np.where(np.isfinite(sv), sv, np.nan), t, 60)
            bits[np.isfinite(sqi) & (sqi < 50)] |= SQI
        out[name] = [t, v, bits, bits.copy()]
    art = [k for k in ("map", "sbp", "dbp") if k in out]
    if len(art) == 3:
        for name in art:
            t = out[name][0]
            s = {k: causal_sample(out[k][0], np.where(out[k][2] == 0, out[k][1], np.nan), t, 2.0)[0]
                 for k in art}
            bad = ((s["sbp"] - s["dbp"] < cfg.min_pulse_pressure)
                   | (s["map"] < s["dbp"] - 2) | (s["map"] > s["sbp"] + 2))
            bad &= np.isfinite(s["sbp"]) & np.isfinite(s["dbp"]) & np.isfinite(s["map"])
            out[name][2][bad] |= PP
            out[name][3][bad] |= PP
    for name, (t, v, causal, label) in out.items():
        checks = CHANNELS[name].checks
        ok = causal == 0
        if "flat" in checks:
            c, r = flat_flags(t, v, ok, cfg)
            causal[c] |= FLAT
            label[r] |= FLAT
        if "jump" in checks:
            c, r = jump_flags(t, v, ok & (causal == 0), cfg)
            causal[c] |= JUMP
            label[r] |= JUMP
        if "wave" in checks and wave_check is not None:
            causal[wave_check(t, True)] |= WAVE
            label[wave_check(t, False)] |= WAVE
    return {k: tuple(v) for k, v in out.items()}


def grid_numeric(cleaned, grid):
    """Numeric channels on `grid`: value, age (s, capped 600) and flag byte, all causal."""
    names = list(CHANNELS)
    values = np.full((len(grid), len(names)), np.nan, np.float32)
    ages = np.full((len(grid), len(names)), 600.0, np.float32)
    flags = np.full((len(grid), len(names)), NOTRACK | STALE, np.uint8)
    for j, name in enumerate(names):
        if name not in cleaned:
            continue
        t, v, bits, _ = cleaned[name]
        good = bits == 0
        if good.any():
            val, age = causal_sample(t[good], v[good], grid, CHANNELS[name].max_age)
        else:
            val, age = np.full(len(grid), np.nan), np.full(len(grid), np.inf)
        values[:, j] = val
        ages[:, j] = np.minimum(age, 600)
        last = np.searchsorted(t, grid, side="right") - 1
        recent = (last >= 0) & (grid - t[np.maximum(last, 0)] <= CHANNELS[name].max_age)
        flags[:, j] = np.where(recent, bits[np.maximum(last, 0)], 0)
        flags[~np.isfinite(val), j] |= STALE
    return names, values, ages, flags


BASELINE_RANGE = (50.0, 150.0)   # a reference MAP outside this is not credible as a baseline


def baseline_pressure(raw, meta, cleaned=None):
    """MAP/SBP reference for `map_drop_pct`. VitalDB rarely records before induction (anestart is
    usually before the recording starts), so fall back in order and report the source:
    1 NIBP before anestart (median of last 3), 2 first NIBP before incision (median of first 3),
    3 artefact-free ART MAP, median of the first 300 s before incision, 0 missing.
    A tier is skipped when its MAP falls outside BASELINE_RANGE."""
    anestart, opstart = float(meta["anestart"]), float(meta["opstart"])

    def valid(key, lo, hi):
        if key not in raw:
            return np.array([]), np.array([])
        t, v = (np.asarray(a, float) for a in raw[key])
        ok = np.isfinite(v) & (v > lo) & (v < hi)
        return t[ok], v[ok]

    def credible(value):
        return BASELINE_RANGE[0] <= value <= BASELINE_RANGE[1]

    t, v = valid("nibp_mbp", 20, 200)
    ts, vs = valid("nibp_sbp", 40, 260)

    def sbp_near(times):
        if not len(ts):
            return np.nan
        s, _ = causal_sample(ts, vs, times, 5)
        return float(np.nanmedian(s)) if np.isfinite(s).any() else np.nan

    for source, sel in ((1, np.flatnonzero(t < anestart)[-3:]), (2, np.flatnonzero(t < opstart)[:3])):
        if len(sel) and credible(float(np.median(v[sel]))):
            return float(np.median(v[sel])), sbp_near(t[sel]), source
    if cleaned and "map" in cleaned:
        t, v, _, bits = cleaned["map"]
        ok = (bits == 0) & (t < opstart)
        if ok.sum() >= 10:
            first = t[ok][0]
            sel = ok & (t < first + 300)
            value = float(np.median(v[sel]))
            if credible(value):
                sbp = np.nan
                if "sbp" in cleaned:
                    st, sv, _, sb = cleaned["sbp"]
                    s = sv[(sb == 0) & (st >= first) & (st < first + 300)]
                    sbp = float(np.median(s)) if len(s) else np.nan
                return value, sbp, 3
    return np.nan, np.nan, 0


# ------------------------------------------------------------------ waveform
def wave_segment(times, pressure, start, end, fs=WAVE_FS):
    """Regular array covering [start, end) at `fs`; NaN outside the recording."""
    times, pressure = np.asarray(times, float), np.asarray(pressure, float)
    if len(times) < 2:
        return np.full(int(round((end - start) * fs)), np.nan)
    period = np.median(np.diff(times[:10000]))
    if abs(period * fs - 1) > 1e-3:
        raise ValueError(f"Waveform sample period {period} does not match {fs} Hz")
    n = int(round((end - start) * fs))
    first = int(round((start - times[0]) * fs))
    out = np.full(n, np.nan)
    lo, hi = max(first, 0), min(first + n, len(pressure))
    if hi > lo:
        out[lo - first:hi - first] = pressure[lo:hi]
    return out


def _dilate(mask, before, after):
    out = mask.copy()
    for k in range(1, before + 1):
        out[:-k] |= mask[k:]
    for k in range(1, after + 1):
        out[k:] |= mask[:-k]
    return out


def wave_blocks(x, cfg=PrepConfig(), fs=WAVE_FS):
    """1 s artefact blocks: missing, flush (high square wave), flat (no pulsatility: zeroing,
    sampling, closed stopcock). Returns (causal bits, retro bits); causal dilation is
    forward-only, so block k depends on samples up to the end of block k."""
    n = len(x) // fs
    blocks = x[:n * fs].reshape(n, fs)
    finite = np.isfinite(blocks)
    nan_frac = 1 - finite.mean(axis=1)
    with np.errstate(all="ignore"):
        bmax = np.where(finite.any(1), np.nanmax(np.where(finite, blocks, -np.inf), axis=1), np.nan)
        bmin = np.where(finite.any(1), np.nanmin(np.where(finite, blocks, np.inf), axis=1), np.nan)
    hi2 = np.fmax(bmax, np.r_[np.nan, bmax[:-1]])       # trailing 2 s peak-to-peak
    lo2 = np.fmin(bmin, np.r_[np.nan, bmin[:-1]])
    missing = nan_frac > 0.5
    flush = (bmax > cfg.flush_mmhg) | (bmin < -10)
    flat = ~missing & (hi2 - lo2 < cfg.flat_ptp_mmhg)
    d = cfg.artefact_dilate_seconds
    causal = (missing * W_MISSING | _dilate(flush, 0, d) * W_FLUSH | _dilate(flat, 0, d) * W_FLAT)
    retro = (missing * W_MISSING | _dilate(flush, d, d) * W_FLUSH | _dilate(flat, d, d) * W_FLAT)
    return causal.astype(np.uint8), retro.astype(np.uint8)


BEAT_COLUMNS = ["onset", "avail", "sbp", "dbp", "map", "pp", "period", "hr", "dpdt_max", "sys_area",
                "ejection_time", "notch_rel", "decay_tau", "notch_found", "reject"]


def beat_table(x, start, block_bits, cfg=PrepConfig(), fs=WAVE_FS):
    """Beats of a regular waveform starting at `start`. Onset = foot before each systolic peak;
    notch = first local minimum after the peak within the first 60% of the beat, else the
    maximum curvature point (inflection). Reject reasons are all causal (<= avail)."""
    finite = np.isfinite(x)
    if finite.sum() < fs * 10:
        return pd.DataFrame(columns=BEAT_COLUMNS)
    filled = pd.Series(x).interpolate(limit=fs // 10, limit_area="inside").to_numpy()
    ok = np.isfinite(filled)
    smooth = sosfiltfilt(butter(4, 15, fs=fs, output="sos"), np.where(ok, filled, 0.0))
    smooth[~ok] = np.nan
    peaks, _ = find_peaks(np.nan_to_num(smooth, nan=-1e3), distance=int(.33 * fs), prominence=8)
    if len(peaks) < 3:
        return pd.DataFrame(columns=BEAT_COLUMNS)
    onsets = np.array([p0 + int(np.nanargmin(smooth[p0:p1])) if np.isfinite(smooth[p0:p1]).any() else p0
                       for p0, p1 in zip(peaks[:-1], peaks[1:])])
    d1 = np.gradient(smooth) * fs
    d2 = np.gradient(d1) * fs
    rows = []
    for a, b in zip(onsets[:-1], onsets[1:]):
        seg = smooth[a:b]
        if len(seg) < 0.3 * fs or not np.isfinite(seg).all():
            continue
        peak = int(np.argmax(seg))
        sbp, dbp, period = float(seg[peak]), float(seg[0]), (b - a) / fs
        pp, mean = sbp - dbp, float(seg.mean())
        lo, hi = peak + int(.04 * fs), int(.6 * len(seg))
        notch, found = None, 0.0
        if hi - lo > 3:
            slope = d1[a + lo:a + hi]
            up = np.flatnonzero((slope[:-1] < 0) & (slope[1:] >= 0))
            if len(up):
                notch, found = lo + int(up[0]) + 1, 1.0
            else:
                notch = lo + int(np.argmax(d2[a + lo:a + hi]))
        if notch is None:
            notch = min(len(seg) - 1, max(peak + 1, int(.37 * np.sqrt(period) * fs)))
        p_notch, p_end = float(seg[notch]), float(seg[-1])
        tau = (len(seg) - notch) / fs / np.log(p_notch / p_end) if p_notch > p_end > 0 else np.nan
        raw_seg = x[a:b][np.isfinite(x[a:b])]
        shape_ok = ((20 <= dbp < mean < sbp <= 300) and (10 <= pp <= 150) and (.3 <= period <= 2.0)
                    and len(raw_seg) and np.ptp(raw_seg) < 200)
        k0, k1 = int(a // fs), min(int(b // fs), len(block_bits) - 1)
        artefact = bool(block_bits[k0:k1 + 1].any()) if k0 < len(block_bits) else True
        rows.append((start + a / fs, start + b / fs + FILTER_MARGIN, sbp, dbp, mean, pp, period,
                     60 / period, float(np.max(d1[a:a + peak + 1])),
                     float(np.sum(seg[:notch] - dbp) / fs), notch / fs, (p_notch - dbp) / pp if pp else np.nan,
                     tau, found, (0 if shape_ok else B_SHAPE) | (B_ARTIFACT if artefact else 0)))
    beats = pd.DataFrame(rows, columns=BEAT_COLUMNS)
    _reject_irregular_damped(beats, cfg)
    return beats


def _time_index(seconds):
    return pd.to_timedelta(np.asarray(seconds, float), unit="s")


def _reject_irregular_damped(beats, cfg):
    """Causal references from beats that passed the shape/artefact checks and came strictly
    earlier: period vs the median of the previous 10; PP and dP/dt vs the median of the
    previous 300 s (needs >= 20 beats)."""
    reject = beats.reject.to_numpy(np.int64)
    base = reject == 0
    if base.sum() < 6:
        return
    ok = beats.loc[base, ["avail", "period", "pp", "dpdt_max"]].reset_index(drop=True)
    ref_period = ok.period.rolling(10, min_periods=5).median().shift(1).to_numpy()
    timed = ok.set_index(_time_index(ok.avail))[["pp", "dpdt_max"]]
    ref = timed.rolling("300s", min_periods=20, closed="left").median().to_numpy()
    with np.errstate(invalid="ignore"):
        irregular = np.abs(ok.period.to_numpy() - ref_period) > cfg.irregular_ratio * ref_period
        damped = ((ok.pp.to_numpy() < cfg.damped_ratio * ref[:, 0])
                  & (ok.dpdt_max.to_numpy() < cfg.damped_ratio * ref[:, 1]))
    idx = np.flatnonzero(base)
    reject[idx[irregular]] |= B_IRREGULAR
    reject[idx[damped]] |= B_DAMPED
    beats["reject"] = reject


BEAT_GRID = ["wb_sbp", "wb_dbp", "wb_map", "wb_pp", "wb_hr", "wb_dpdt", "wb_sys_area", "wb_ejection_time",
             "wb_notch_rel", "wb_decay_tau", "wb_sv", "wb_co", "wb_svr", "wb_ppv", "wb_spv",
             "wb_valid_frac", "wb_age"]
_BEAT_VALUES = ["sbp", "dbp", "map", "pp", "hr", "dpdt_max", "sys_area", "ejection_time", "notch_rel", "decay_tau"]


def beat_grid(beats, grid, cfg=PrepConfig(), max_age=10.0):
    """Causal beat features on `grid`. Each cell takes the last accepted beat with avail <= t
    (at most `max_age` old) and the median over accepted beats in the `beat_window_seconds`
    ending at that beat (>= 3 beats). PPV/SPV use the `ppv_window_seconds` ending there, only
    when >= 5 beats, none rejected in the window and period CV <= `ppv_max_period_cv`."""
    grid = np.asarray(grid, float)
    out = np.full((len(grid), len(BEAT_GRID)), np.nan, np.float32)
    if beats.empty:
        out[:, -1] = np.inf
        return out
    avail = beats.avail.to_numpy(float)
    good = beats.reject.to_numpy() == 0
    all_hi = np.searchsorted(avail, grid, side="right")
    all_lo = np.searchsorted(avail, grid - cfg.ppv_window_seconds, side="right")
    cum = np.r_[0, np.cumsum(good)]
    total, accepted = all_hi - all_lo, cum[all_hi] - cum[all_lo]
    with np.errstate(all="ignore"):
        out[:, -2] = np.where(total > 0, accepted / np.maximum(total, 1), np.nan)
    if not good.any():
        out[:, -1] = np.inf
        return out
    g = beats.loc[good, ["avail", "period"] + _BEAT_VALUES].astype(float).reset_index(drop=True)
    g = g[~g.avail.duplicated(keep="last")].reset_index(drop=True)
    timed = g.set_index(_time_index(g.avail))
    win, pwin = f"{int(cfg.beat_window_seconds)}s", f"{int(cfg.ppv_window_seconds)}s"
    med = timed[_BEAT_VALUES].rolling(win, min_periods=1).median().to_numpy()
    count = timed.period.rolling(win).count().to_numpy()
    pp_roll, sbp_roll, per_roll = timed.pp.rolling(pwin), timed.sbp.rolling(pwin), timed.period.rolling(pwin)
    pp_max, pp_min = pp_roll.max().to_numpy(), pp_roll.min().to_numpy()
    spv = (sbp_roll.max() - sbp_roll.min()).to_numpy()
    p_cv = (per_roll.std(ddof=0) / per_roll.mean()).to_numpy()
    p_count = per_roll.count().to_numpy()
    gt = g.avail.to_numpy()
    last = np.searchsorted(gt, grid, side="right") - 1
    li = np.maximum(last, 0)
    age = np.where(last >= 0, grid - gt[li], np.inf)
    out[:, -1] = age
    use = (last >= 0) & (age <= max_age) & (count[li] >= 3)
    m = med[li]
    sbp, dbp, mbp, pp, hr = (m[:, k] for k in range(5))
    with np.errstate(all="ignore"):
        sv = pp / (sbp + dbp)                    # Liljestrand-Zander, arbitrary units
        derived = np.stack([sv, sv * hr, mbp / (sv * hr)], axis=1)
        ppv = (pp_max[li] - pp_min[li]) / ((pp_max[li] + pp_min[li]) / 2) * 100
        ppv_ok = use & (p_count[li] >= 5) & (accepted == total) & (p_cv[li] <= cfg.ppv_max_period_cv)
    out[use, :10] = m[use]
    out[use, 10:13] = derived[use]
    out[ppv_ok, 13] = ppv[ppv_ok]
    out[ppv_ok, 14] = spv[li][ppv_ok]
    return out


def centered_beat_map(beats, times, half=5.0):
    """Non-causal median MAP of accepted beats within +-half s of the beat nearest each time
    (>= 3 beats, nearest beat within 2 s). QC and label checks only, never model inputs."""
    times = np.asarray(times, float)
    good = beats[beats.reject.eq(0)]
    if len(good) < 3:
        return np.full(len(times), np.nan)
    mid = (good.onset + good.period / 2).to_numpy(float)
    maps = good["map"].to_numpy(float)
    order = np.argsort(mid, kind="stable")
    mid, maps = mid[order], maps[order]
    keep = np.r_[mid[1:] != mid[:-1], True]
    mid, maps = mid[keep], maps[keep]
    med = pd.Series(maps, index=_time_index(mid)).rolling(f"{int(2 * half)}s", center=True,
                                                           min_periods=3).median().to_numpy()
    j = np.clip(np.searchsorted(mid, times), 1, len(mid) - 1)
    j = np.where(np.abs(mid[j - 1] - times) < np.abs(mid[j] - times), j - 1, j)
    return np.where(np.abs(mid[j] - times) <= 2.0, med[j], np.nan)


def to_dl_rate(x, fs=WAVE_FS, target=DL_FS):
    """Causal anti-alias (Butterworth order 4 at 0.4 * target) and decimation. Sample k is at
    start + k / target and depends only on samples at or before it. Gaps (> 0.1 s) stay NaN."""
    factor = fs // target
    if fs % target:
        raise ValueError("fs must be a multiple of the target rate")
    filled = pd.Series(x).ffill(limit=fs // 10)
    nan = filled.isna().to_numpy()
    y = sosfilt(butter(4, 0.4 * target, fs=fs, output="sos"), filled.fillna(0).to_numpy())
    # Invalidate where any sample in the last 0.1 s was a gap (filter memory of the zero fill).
    recent_gap = pd.Series(nan.astype(np.int8)).rolling(fs // 10, min_periods=1).max().to_numpy() > 0
    y[recent_gap] = np.nan
    return y[::factor].astype(np.float16)


def estimate_lag(beats, map_t, map_v, start, end, lags=np.arange(-20, 21, 2)):
    """Lag (s) maximising corr(monitor MAP(t), centered beat MAP(t + lag)). Negative = the
    monitor reports later than the waveform. Returns (lag, corr, median bias wave - monitor)."""
    if beats.empty or len(map_t) < 50:
        return np.nan, np.nan, np.nan
    times = np.arange(start + 30, end - 30, 5.0)
    monitor, _ = causal_sample(map_t, map_v, times, 4.0)
    best = (np.nan, -np.inf, np.nan)
    for lag in lags:
        wave_map = centered_beat_map(beats, times + lag)
        ok = np.isfinite(monitor) & np.isfinite(wave_map)
        if ok.sum() < 30 or np.std(monitor[ok]) == 0 or np.std(wave_map[ok]) == 0:
            continue
        r = np.corrcoef(monitor[ok], wave_map[ok])[0, 1]
        if r > best[1]:
            best = (float(lag), float(r), float(np.median(wave_map[ok] - monitor[ok])))
    return best if np.isfinite(best[1]) else (np.nan, np.nan, np.nan)


DEFAULT_LAG = -8.0   # monitor MAP trails the waveform by ~8 s (trial cases); fallback per case
MIN_LAG_CORR = 0.9


# ------------------------------------------------------------------ one case
def process_case(meta, raw, wave=None, cfg=PrepConfig()):
    """meta: cohort row (caseid, subjectid, opstart, opend, anestart). raw: name -> (times, values)
    for CHANNELS/AUX_TRACKS present. wave: (times, pressure) of SNUADC/ART or None.

    Returns (arrays: dict for np.savez, wave100: float16 array or None, qc: dict)."""
    start, end = float(np.ceil(meta["opstart"])), float(np.floor(meta["opend"]))
    if end - start < 60:
        raise ValueError("Surgery interval shorter than 60 s")
    seconds = int(end - start)
    grid = start + STEP * np.arange(int(np.ceil(seconds / STEP)))
    grid1 = start + np.arange(seconds, dtype=float)

    beats, wave100 = pd.DataFrame(columns=BEAT_COLUMNS), None
    causal_blocks = retro_blocks = np.zeros(seconds, np.uint8)
    if wave is not None:
        x = wave_segment(*wave, start, end)
        causal_blocks, retro_blocks = wave_blocks(x, cfg)
        beats = beat_table(x, start, causal_blocks, cfg)
        wave100 = to_dl_rate(x)
        del x
    bgrid = beat_grid(beats, grid, cfg)

    map_t, map_v = (np.asarray(a, float) for a in raw.get("map", (np.array([]), np.array([]))))
    ch = CHANNELS["map"]
    usable = np.isfinite(map_v) & (map_v >= ch.lo) & (map_v <= ch.hi) & (map_t >= start) & (map_t < end)
    lag, lag_corr, bias = estimate_lag(beats, map_t[usable], map_v[usable], start, end)
    reliable = np.isfinite(lag_corr) and lag_corr >= MIN_LAG_CORR
    use_lag, use_bias = (lag, bias) if reliable else (DEFAULT_LAG, 0.0)

    wave_check = None
    if not beats.empty:
        # Trust: share of accepted beats over the last 300 s; a channel that is dead for long
        # (noise around 0 while the monitor still reports MAP) must not veto the monitor.
        accepted = beats.reject.eq(0).to_numpy()
        avail = beats.avail.to_numpy(float)
        cum = np.r_[0, np.cumsum(accepted)]
        bt_causal_map = bgrid[:, BEAT_GRID.index("wb_map")]

        def trusted(times):
            hi = np.searchsorted(avail, times, side="right")
            lo = np.searchsorted(avail, times - cfg.wave_trust_seconds, side="right")
            n = hi - lo
            return (n >= 20) & ((cum[hi] - cum[lo]) >= cfg.wave_trust_min_valid * np.maximum(n, 1))

        def wave_check(times, causal):
            times = np.asarray(times, float)
            inside = (times >= start) & (times < end)
            k = np.clip(((times - start) // 1).astype(int), 0, seconds - 1)
            blocks = causal_blocks if causal else retro_blocks
            artefact = inside & ((blocks[k] & (W_FLUSH | W_FLAT)) > 0)
            monitor, _ = causal_sample(map_t, map_v, times, 2.0)
            if causal:
                # The causal beat MAP trails like the monitor does; remove a running bias
                # (median difference over the previous 300 s) instead of the case-level one.
                g = np.searchsorted(grid, times, side="right") - 1
                diff = np.where(g >= 0, bt_causal_map[np.maximum(g, 0)], np.nan) - monitor
                ref = pd.Series(diff, index=_time_index(times)).rolling(
                    "300s", min_periods=30, closed="left").median().to_numpy()
                diff = diff - np.nan_to_num(ref, nan=0.0)
            else:
                diff = centered_beat_map(beats, times + use_lag) - monitor - use_bias
            with np.errstate(invalid="ignore"):
                mismatch = np.abs(diff) > cfg.wave_mismatch_mmhg
            return trusted(times) & inside & (artefact | mismatch)

    cleaned = clean_numeric(raw, cfg, wave_check)
    names, values, ages, flags = grid_numeric(cleaned, grid)
    base_map, base_sbp, base_source = baseline_pressure(raw, meta, cleaned)

    # Label-grade MAP at 1 s: retro artefacts become unknown (NaN), gaps > 10 s unknown.
    label_map = np.full(seconds, np.nan, np.float32)
    label_bits = np.full(seconds, NOTRACK, np.uint8)
    if "map" in cleaned:
        t, v, _, lbits = cleaned["map"]
        label_map = label_grid(t, np.where(lbits == 0, v, np.nan), grid1,
                               cfg.label_max_gap_seconds).astype(np.float32)
        last = np.searchsorted(t, grid1 + 1, side="right") - 1
        label_bits = np.where(last >= 0, lbits[np.maximum(last, 0)], NOTRACK).astype(np.uint8)

    qc = qc_row(meta, cleaned, names, values, beats, bgrid, label_map, causal_blocks,
                wave is not None, start, end)
    qc.update({"wave_lag_seconds": lag, "wave_lag_corr": lag_corr, "wave_map_bias": bias,
               "lag_used": use_lag, "baseline_map": base_map, "baseline_sbp": base_sbp,
               "baseline_source": base_source})
    arrays = {
        "start": np.float64(start), "step": np.float64(STEP),
        "channels": np.array(names), "values": values, "ages": ages, "flags": flags,
        "beat_table_columns": np.array(BEAT_COLUMNS), "beats": beats.to_numpy(np.float64),
        "label_map": label_map, "label_flags": label_bits, "wave_mask": causal_blocks,
        "baseline": np.array([base_map, base_sbp, base_source], np.float64),
    }
    return arrays, wave100, qc


def qc_row(meta, cleaned, names, values, beats, bgrid, label_map, blocks, has_wave, start, end):
    """Per-case quality summary; raw-sample fractions count only samples inside the surgery."""
    row = {"caseid": int(meta["caseid"]), "subjectid": int(meta["subjectid"]),
           "surgery_seconds": int(len(label_map)), "has_wave": bool(has_wave)}
    for j, name in enumerate(names):
        row[f"{name}_coverage"] = float(np.isfinite(values[:, j]).mean())
    for name in ("map", "sbp", "hr"):
        if name not in cleaned:
            continue
        t, _, causal, label = cleaned[name]
        inside = (t >= start) & (t < end)
        if not inside.any():
            continue
        causal, label = causal[inside], label[inside]
        for bit, label_name in FLAG_NAMES.items():
            if bit not in (STALE, NOTRACK, SQI):
                row[f"{name}_{label_name}_frac"] = float(((causal & bit) > 0).mean())
        row[f"{name}_input_artefact_frac"] = float((causal > 0).mean())
        row[f"{name}_label_artefact_frac"] = float((label > 0).mean())
    known = np.isfinite(label_map)
    row["label_known_frac"] = float(known.mean())
    row["label_below65_frac"] = float((label_map[known] < 65).mean()) if known.any() else np.nan
    if has_wave:
        row["wave_missing_frac"] = float(((blocks & W_MISSING) > 0).mean())
        row["wave_flush_frac"] = float(((blocks & W_FLUSH) > 0).mean())
        row["wave_flat_frac"] = float(((blocks & W_FLAT) > 0).mean())
        row["n_beats"] = int(len(beats))
        if len(beats):
            rej = beats.reject.to_numpy().astype(int)
            row["beat_valid_frac"] = float((rej == 0).mean())
            for bit, name in ((B_SHAPE, "shape"), (B_ARTIFACT, "artefact"), (B_IRREGULAR, "irregular"),
                              (B_DAMPED, "damped")):
                row[f"beat_{name}_frac"] = float(((rej & bit) > 0).mean())
            row["beat_notch_found_frac"] = float(beats.notch_found.mean())
        wb_map = bgrid[:, BEAT_GRID.index("wb_map")]
        mon = values[:, names.index("map")]
        ok = np.isfinite(wb_map) & np.isfinite(mon)
        row["wave_map_mae_causal"] = float(np.mean(np.abs(wb_map[ok] - mon[ok]))) if ok.any() else np.nan
    return row
