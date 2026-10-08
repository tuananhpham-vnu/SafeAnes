"""Model features from raw monitor vitals (the "Dự báo" page): the training pipeline, run on uploaded data.

Steps, all causal (only data at or before t):
1. `safeanes.preprocess.clean_numeric`: plausibility ranges, pulse-pressure, jump and flat checks (no waveform
   check: uploads have no ART waveform).
2. `safeanes.preprocess.grid_numeric`: 2 s grid, last valid value at most 30 s old, never interpolated.
3. Predictions every 30 s from 30 s after the first sample.
4. Window columns (W = 120 s) with `uc04.long_windows` (checked against prep_v1); `*_current` = grid value at t;
   `map_drop_pct` against a baseline MAP (given, else `baseline_pressure` of the pipeline).
5. Case-context columns `f1a_*`, definitions reconstructed and checked against samples v3 (DEVIATIONS v3-8):
   MAP mean/min/slope over 5/10/15/30 min; HR, pulse pressure, EtCO2 mean/slope over 5/15 min; MAP minus the
   running median of the case; MAP minus its 15 min maximum; minutes with 70 <= MAP < 80 in 15 min; minutes since
   induction / incision (missing before); number of hypotension events that have ended and minutes since the
   last one ended. Events: the pipeline's own label-grade MAP and uc04.events.detect_events.
"""
from __future__ import annotations

import io
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(REPO_SRC) not in sys.path:  # local deployment runs inside the repo: reuse the pipeline's own code
    sys.path.insert(0, str(REPO_SRC))

from safeanes.preprocess import RANGE, WAVE, baseline_pressure, clean_numeric, grid_numeric  # noqa: E402
from safeanes.signals import label_grid  # noqa: E402
from uc04.columns import SIG  # noqa: E402
from uc04.events import detect_events as pipeline_events  # noqa: E402
from uc04.long_windows import add_extrap, grid_signals, window_features  # noqa: E402

STEP, CADENCE, W = 2.0, 30.0, 120
SETTLE_S = 0.0           # stand-in for the waveform flush check: off (20 s made 3 of 4 checked cases worse)
LABEL_MAX_GAP_S = 10     # label-grade MAP: gaps longer than this are unknown (PrepConfig.label_max_gap_seconds)
MAP_THRESHOLD, EVENT_S, MERGE_GAP_S, POST_EVENT_S = 65.0, 60.0, 120.0, 120.0

# accepted column names (case-insensitive): our names, VitalDB track names, common monitor exports
ALIASES = {
    "time": ["time", "t", "time_s", "seconds", "timestamp", "datetime", "clock", "thoi_gian", "thời gian"],
    "map": ["map", "art_mbp", "solar8000/art_mbp", "mbp", "abp_mean", "ibp_mean", "art_mean", "map_art"],
    "sbp": ["sbp", "art_sbp", "solar8000/art_sbp", "abp_sys", "ibp_sys", "art_sys"],
    "dbp": ["dbp", "art_dbp", "solar8000/art_dbp", "abp_dia", "ibp_dia", "art_dia"],
    "hr": ["hr", "solar8000/hr", "heart_rate", "pulse"],
    "spo2": ["spo2", "pleth_spo2", "solar8000/pleth_spo2", "sao2"],
    "etco2": ["etco2", "solar8000/etco2", "et_co2", "co2_et"],
    "rr": ["rr", "rr_co2", "solar8000/rr_co2", "resp_rate", "awrr"],
    "nibp_mbp": ["nibp_mbp", "solar8000/nibp_mbp", "nibp_mean"],
    "nibp_sbp": ["nibp_sbp", "solar8000/nibp_sbp", "nibp_sys"],
}
REQUIRED = ("map",)


class InputError(ValueError):
    """A problem with the uploaded data, shown to the user as is."""


def parse_time(x, base_date=None) -> float:
    """Seconds: a number (already seconds), 'HH:MM[:SS]' (seconds of the day) or a date-time (epoch s)."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or str(x).strip() == "":
        return float("nan")
    s = str(x).strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?", s)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3) or 0)
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        raise InputError(f"không đọc được thời điểm {s!r} (dùng số giây, HH:MM:SS hoặc ngày giờ)")
    return ts.timestamp()


def read_table(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8-sig", errors="replace")
    sep = "\t" if text.count("\t") > text.count(",") and text.count("\t") > text.count(";") else (
        ";" if text.count(";") > text.count(",") else ",")
    df = pd.read_csv(io.StringIO(text), sep=sep, comment="#")
    if df.empty:
        raise InputError("file không có dòng dữ liệu nào")
    return df


def map_columns(df: pd.DataFrame) -> dict[str, str]:
    lower = {c.strip().lower(): c for c in df.columns}
    found = {}
    for key, names in ALIASES.items():
        for n in names:
            if n in lower:
                found[key] = lower[n]
                break
    if "time" not in found:
        raise InputError(f"thiếu cột thời gian (một trong: {', '.join(ALIASES['time'][:5])})")
    missing = [k for k in REQUIRED if k not in found]
    if missing:
        raise InputError(f"thiếu cột bắt buộc {missing} (MAP động mạch xâm lấn, ví dụ 'map' hoặc 'ART_MBP')")
    return found


def raw_tracks(df: pd.DataFrame, cols: dict[str, str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    t = np.array([parse_time(v) for v in df[cols["time"]]], float)
    raw = {}
    for key, col in cols.items():
        if key == "time":
            continue
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(v)
        if ok.sum() == 0:
            continue
        order = np.argsort(t[ok], kind="stable")
        tt, vv = t[ok][order], v[ok][order]
        keep = np.r_[True, np.diff(tt) > 0]  # one value per time stamp
        raw[key] = (tt[keep], vv[keep])
    if "map" not in raw:
        raise InputError("cột MAP không có giá trị số nào")
    return raw


def settle_after_range(cleaned: dict) -> dict:
    """Uploads have no ART waveform, so the pipeline's waveform check (flush, zeroing) cannot run. Stand-in, causal:
    an arterial sample (MAP, SBP, DBP) within SETTLE_S after any arterial sample out of range is an artefact (the
    tail of a flush or of zeroing, e.g. 240 -> 217 -> 199 mmHg), flagged WAVE in both the causal and label views."""
    art = [k for k in ("map", "sbp", "dbp") if k in cleaned]
    if not art:
        return cleaned
    viol = np.sort(np.concatenate([cleaned[k][0][(cleaned[k][2] & RANGE) != 0] for k in art]))
    if not len(viol):
        return cleaned
    out = dict(cleaned)
    for k in art:
        t, v, causal, label = cleaned[k]
        j = np.searchsorted(viol, t, side="right") - 1
        recent = (j >= 0) & (t - viol[np.maximum(j, 0)] <= SETTLE_S)
        causal, label = causal.copy(), label.copy()
        causal[recent & (causal == 0)] |= WAVE
        label[recent & (label == 0)] |= WAVE
        out[k] = (t, v, causal, label)
    return out


def label_events(cleaned: dict, start: float, end: float) -> list[dict]:
    """Events as the pipeline: label-grade 1 s MAP (preprocess.process_case) -> uc04.events.detect_events."""
    if "map" not in cleaned:
        return []
    t, v, _, lbits = cleaned["map"]
    grid1 = np.arange(start, end, 1.0)
    lm = label_grid(t, np.where(lbits == 0, v, np.nan), grid1, LABEL_MAX_GAP_S)
    ev = pipeline_events(lm, start, threshold=MAP_THRESHOLD, min_seconds=int(EVENT_S), merge_gap=int(MERGE_GAP_S))
    return [{"onset_s": float(r.onset), "end_s": float(r.end), "min_map": float(r.min_map)} for r in ev.itertuples()]


def _rolling_slope(tmin: pd.Series, x: pd.Series, n: int) -> np.ndarray:
    tt = tmin.where(x.notna())
    return (tt.rolling(n, min_periods=2).cov(x) / tt.rolling(n, min_periods=2).var()).to_numpy()


def _half(x: pd.Series, n: int) -> np.ndarray:
    """samples v3 leaves a context window statistic missing while fewer than half of its cells are valid."""
    return x.notna().astype(float).rolling(n, min_periods=1).sum().to_numpy() >= n / 2


def build(df: pd.DataFrame, anestart: float | None, opstart: float | None, baseline_map: float | None,
          grid_start: float | None = None, times: np.ndarray | None = None) -> dict:
    """`grid_start` / `times` fix the 2 s grid phase and the prediction times (used to check against samples v3;
    by default the grid starts at the first whole second of data and predictions follow every 30 s)."""
    cols = map_columns(df)
    raw = raw_tracks(df, cols)
    t_first = min(v[0][0] for v in raw.values())
    t_last = max(v[0][-1] for v in raw.values())
    if t_last - t_first < 5 * 60:
        raise InputError("cần ít nhất 5 phút dữ liệu")
    if t_last - t_first > 24 * 3600:
        raise InputError("dữ liệu dài hơn 24 giờ: kiểm tra cột thời gian")
    start = math.ceil(t_first) if grid_start is None else float(grid_start)
    grid = np.arange(start, t_last + 1e-9, STEP)
    cleaned = settle_after_range(clean_numeric(raw)) if SETTLE_S > 0 else clean_numeric(raw)
    names, values, _, _ = grid_numeric(cleaned, grid)
    times = np.arange(start + CADENCE, t_last + 1e-9, CADENCE) if times is None else np.asarray(times, float)
    times = times[(times >= start) & (times <= grid[-1])]
    idx = np.rint((times - start) / STEP).astype(int)

    g = grid_signals(values, names)
    feat = window_features(values, names, float(start), times, W)
    cur = {s: g[s].to_numpy()[idx] for s in SIG}
    for s in SIG:
        feat[f"{s}_current"] = cur[s]
    add_extrap(feat, W, cur["map"])

    # baseline MAP for map_drop_pct
    base_src = "user"
    if baseline_map is None or not np.isfinite(baseline_map):
        base_src = "auto"
        meta = {"anestart": anestart if anestart is not None else -np.inf,
                "opstart": opstart if opstart is not None else t_last + 1}
        baseline_map, _, src = baseline_pressure(raw, meta, cleaned)
        base_src = {1: "NIBP trước khởi mê", 2: "NIBP trước rạch da", 3: "MAP 5 phút đầu", 0: "không có"}[src]
    feat["map_drop_pct"] = (baseline_map - cur["map"]) / baseline_map * 100 if np.isfinite(baseline_map) else np.nan

    # case context
    tmin = pd.Series(np.arange(len(g)) * STEP / 60.0)
    m = g["map"]
    for mins in (5, 10, 15, 30):
        n = int(mins * 60 / STEP)
        ok = _half(m, n)[idx]
        feat[f"f1a_map_mean_m{mins}"] = np.where(ok, m.rolling(n, min_periods=1).mean().to_numpy()[idx], np.nan)
        feat[f"f1a_map_min_m{mins}"] = np.where(ok, m.rolling(n, min_periods=1).min().to_numpy()[idx], np.nan)
        feat[f"f1a_map_slope_m{mins}"] = np.where(ok, _rolling_slope(tmin, m, n)[idx], np.nan)
    for s in ("hr", "pp", "etco2"):
        for mins in (5, 15):
            n = int(mins * 60 / STEP)
            ok = _half(g[s], n)[idx]
            feat[f"f1a_{s}_mean_m{mins}"] = np.where(ok, g[s].rolling(n, min_periods=1).mean().to_numpy()[idx], np.nan)
            feat[f"f1a_{s}_slope_m{mins}"] = np.where(ok, _rolling_slope(tmin, g[s], n)[idx], np.nan)
    n15 = int(15 * 60 / STEP)
    feat["f1a_map_rel_case_median"] = cur["map"] - m.expanding(min_periods=1).median().to_numpy()[idx]
    feat["f1a_map_rel_max_m15"] = cur["map"] - m.rolling(n15, min_periods=1).max().to_numpy()[idx]
    feat["f1a_min_70_80_m15"] = (((m >= 70) & (m < 80)).astype(float).rolling(n15, min_periods=1).sum()
                                 * STEP / 60).to_numpy()[idx]  # no half-window rule here: it agreed less
    since = lambda ev: np.where(times >= ev, (times - ev) / 60, np.nan) if ev is not None else np.full(len(times), np.nan)
    feat["f1a_min_since_opstart"] = since(opstart)
    feat["f1a_min_since_anestart"] = since(anestart)
    events = label_events(cleaned, start, float(grid[-1]) + STEP)
    ends = np.array([e["end_s"] for e in events], float)
    n_prev = np.searchsorted(np.sort(ends), times, side="left")  # events that ended before t
    last_end = np.array([ends[ends < t].max() if (ends < t).any() else np.nan for t in times])
    feat["f1a_n_prev_events"] = n_prev.astype(float)
    feat["f1a_min_since_last_event"] = (times - last_end) / 60

    # when not to predict (as the samples: in or right after an event, low or missing MAP)
    in_event = np.zeros(len(times), bool)
    for e in events:
        in_event |= (times >= e["onset_s"]) & (times < e["end_s"] + POST_EVENT_S)
    eligible = (np.isfinite(cur["map"]) & (cur["map"] >= MAP_THRESHOLD) & ~in_event
                & (feat[f"map_missing_w{W}"].to_numpy() <= 0.5))

    coverage = {k: float(np.isfinite(cur[k]).mean()) for k in ("map", "sbp", "dbp", "hr", "spo2", "etco2", "rr")}
    warnings = []
    for k, label in (("sbp", "HA tâm thu"), ("dbp", "HA tâm trương"), ("hr", "nhịp tim"), ("etco2", "EtCO₂")):
        if coverage[k] < 0.5:
            warnings.append(f"Thiếu {label} ở {round((1 - coverage[k]) * 100)}% thời điểm: các đặc trưng liên quan bị bỏ trống.")
    if opstart is None:
        warnings.append("Chưa nhập thời điểm rạch da: mô hình coi như chưa rạch da.")
    if anestart is None:
        warnings.append("Chưa nhập thời điểm khởi mê: cột 'phút từ khởi mê' bị bỏ trống.")
    if t_first > (anestart if anestart is not None else t_first) + 30 * 60:
        warnings.append("Dữ liệu bắt đầu hơn 30 phút sau khởi mê: trung vị MAP của ca và số đợt tụt trước có thể chưa đủ.")
    removed = {k: int((cleaned[k][2] != 0).sum()) for k in cleaned}
    return {
        "times": times, "features": feat, "eligible": eligible, "events": events, "start_s": float(start),
        "vitals": {k: cur[k] for k in ("map", "sbp", "dbp", "hr", "spo2", "etco2")},
        "quality": {"columns": cols, "coverage": coverage, "samples": {k: int(len(v[0])) for k, v in raw.items()},
                    "artefacts_removed": removed, "baseline_map": None if not np.isfinite(baseline_map) else float(baseline_map),
                    "baseline_source": base_src, "duration_min": round((t_last - t_first) / 60, 1),
                    "warnings": warnings},
    }
