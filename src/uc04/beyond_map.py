"""Week 15: what predicts hypotension once MAP is already known.

Every gain here is measured against a MAP-only model (MAP_CORE: everything derived from the arterial pressure
level: MAP/SBP/DBP statistics of W, MAP context trends, MAP time-in-band, and the case's earlier MAP events).
Its out-of-fold logit is the *offset*: a second model that boosts from that offset can only score by using
information the MAP features do not already carry.

  L1  per feature:  partial Spearman with the MAP residual (case bootstrap CI), conditional MI given the offset
                    decile, incremental gain of offset + feature, AUROC inside the MAP grey zone (65-80 mmHg)
  L2  per pair:     synergy = gain(a, b) - gain(a) - gain(b) (weak alone, strong together), fold stability
  L3  composites:   physiology-driven features built from existing columns (MAP = HR x SV x SVR, compensation,
                    anaesthetic depth, preload, phase), each with the sign physiology expects
  L4  selection:    greedy forward selection of bundles (feature, stable pair) on the offset, then a prefix curve
                    of full models in grouped CV; validation only scores the chosen set

All folds are grouped by subjectid. Only train is used to choose anything.
"""
from __future__ import annotations

import os
import re
from typing import Sequence

import numpy as np
import pandas as pd

from .feature_eda import auroc

EPS = 1e-6
GREY = (65.0, 80.0)

# --------------------------------------------------------------------------------------------- MAP core

_MAP_RE = re.compile(r"^(map|sbp|dbp)_(mean|std|min|max|slope|missing|current)(_w\d+)?$")
MAP_EXTRA = ("map_drop_pct", "f1a_min_70_80_m15", "f1a_n_prev_events", "f1a_min_since_last_event",
             "f1a_map_rel_case_median", "f1a_map_rel_max_m15")


def is_map_core(col: str) -> bool:
    """Columns that only re-express the arterial pressure level (and its own event history)."""
    return bool(_MAP_RE.match(col)) or col in MAP_EXTRA or col.startswith(("f1a_map_", "map_time_65_75_w",
                                                                          "map_extrap_w", "cx_map_"))


def map_core(columns: Sequence[str]) -> list[str]:
    return [c for c in columns if is_map_core(c) and not c.startswith("cx_")]


# --------------------------------------------------------------------------------------------- L3 composites

def _div(a, b):
    with np.errstate(all="ignore"):
        out = np.asarray(a, float) / np.asarray(b, float)
    out[~np.isfinite(out)] = np.nan
    return out


def _sum_known(*parts):
    """Sum treating NaN as 0, NaN only when every part is NaN (drugs not given are 0, not unknown)."""
    a = np.vstack([np.asarray(p, float) for p in parts])
    out = np.nansum(a, axis=0)
    out[np.isnan(a).all(axis=0)] = np.nan
    return out


def _lz_sv(pp, map_):
    """Liljestrand-Zander SV proxy PP / (SBP + DBP), with SBP + DBP = 2 MAP + PP / 3 (MAP = DBP + PP / 3)."""
    return _div(pp, 2 * np.asarray(map_, float) + np.asarray(pp, float) / 3)


def _time_to_cross(map_now, slope, cap=60.0):
    """Minutes until MAP reaches 65 at the current slope (mmHg/min); 0 when already below, cap when not falling."""
    m, s = np.asarray(map_now, float), np.asarray(slope, float)
    with np.errstate(all="ignore"):
        t = np.where(s < -0.05, (m - 65) / -s, cap)
    t = np.where(m <= 65, 0.0, np.minimum(t, cap))
    t[np.isnan(m) | np.isnan(s)] = np.nan
    return t


# name: (hypothesis, expected direction of risk: +1 higher -> more risk, -1 lower -> more risk, 0 no prior)
COMPOSITES = {
    "cx_map_ttc_m5": ("Thời gian (phút) để MAP chạm 65 theo độ dốc 5 phút", -1),
    "cx_map_ttc_m15": ("Thời gian (phút) để MAP chạm 65 theo độ dốc 15 phút", -1),
    "cx_map_proj_5": ("MAP dự phóng 5 phút (MAP + độ dốc 5 phút × 5)", -1),
    "cx_map_proj_10": ("MAP dự phóng 10 phút (độ dốc 10 phút)", -1),
    "cx_map_proj_15": ("MAP dự phóng 15 phút (độ dốc 15 phút)", -1),
    "cx_map_cv": ("Hệ số biến thiên MAP trong W (bất ổn)", +1),
    "cx_map_grey_burden": ("Phút MAP ở vùng xám 65-80 (15 phút) + 65-75 (W)", +1),
    "cx_map_event_rate": ("Số đợt tụt mỗi giờ gây mê", +1),
    "cx_dsi": ("Chỉ số sốc tâm trương HR/DBP", +1),
    "cx_msi": ("Chỉ số sốc hiệu chỉnh HR/MAP", +1),
    "cx_dlog_hr": ("Δlog HR (TB 5 phút so với 15 phút)", 0),
    "cx_dlog_sv": ("Δlog SV proxy (Liljestrand-Zander) 5 vs 15 phút", -1),
    "cx_dlog_svr": ("Δlog SVR proxy = Δlog MAP − Δlog HR − Δlog SV", -1),
    "cx_baro_gain": ("ΔHR/ΔMAP khi MAP đang giảm (bù phản xạ áp cảm thụ; âm = có bù)", +1),
    "cx_no_hr_comp": ("MAP giảm mà HR không tăng (mất bù) [0/1]", +1),
    "cx_no_hr_comp_age": ("Mất bù HR × tuổi/60", +1),
    "cx_etco2_drop": ("EtCO2 TB 5 phút − 15 phút (giảm = cung lượng tim giảm)", -1),
    "cx_etco2_map_coslope": ("Độ dốc EtCO2 × độ dốc MAP khi cả hai giảm", +1),
    "cx_etco2_per_mv": ("EtCO2 / thông khí phút (CO2 đến phổi theo thông khí)", -1),
    "cx_depth": ("Chỉ số độ sâu mê gộp: Ce ppf/3 + Ce rftn/4 + MAC", +1),
    "cx_depth_d5": ("Thay đổi độ sâu mê gộp 5 phút", +1),
    "cx_depth_d5_grey": ("Tăng độ sâu mê khi MAP đang ở vùng xám", +1),
    "cx_depth_age": ("Độ sâu mê gộp × tuổi/60 (người già nhạy thuốc)", +1),
    "cx_bis_low_grey": ("BIS < 40 khi MAP ở vùng xám [0/1]", +1),
    "cx_pre_incision": ("Trước rạch da (chưa có kích thích phẫu thuật) [0/1]", +1),
    "cx_pre_incision_min": ("Số phút từ khởi mê khi chưa rạch da", 0),
    "cx_ppv_grey": ("PPV khi MAP ở vùng xám (phụ thuộc tiền tải)", +1),
    "cx_ppv_x_dp": ("PPV × áp lực đẩy (thông khí làm giảm hồi lưu tĩnh mạch)", +1),
    "cx_drop_htn": ("% MAP giảm so với nền × tăng huyết áp (dịch ngưỡng tự điều hòa)", +1),
    "cx_drop_age": ("% MAP giảm so với nền × tuổi/60", +1),
}
NEEDED = ("map_current", "map_mean_w{W}", "map_std_w{W}", "map_slope_w{W}", "dbp_current", "hr_current",
          "etco2_current", "map_time_65_75_w{W}", "map_drop_pct", "bt_ppv30_w{W}",
          "f1a_map_slope_m5", "f1a_map_slope_m10", "f1a_map_slope_m15", "f1a_map_mean_m5", "f1a_map_mean_m15",
          "f1a_hr_mean_m5", "f1a_hr_mean_m15", "f1a_hr_slope_m5", "f1a_pp_mean_m5", "f1a_pp_mean_m15",
          "f1a_etco2_mean_m5", "f1a_etco2_mean_m15", "f1a_etco2_slope_m5", "f1a_min_70_80_m15",
          "f1a_n_prev_events", "f1a_min_since_anestart", "f1a_min_since_opstart",
          "f1b_ppf", "f1b_rftn", "f1b_mac", "f1b_ppf_d5", "f1b_rftn_d5", "f1b_mac_d5", "f1b_bis",
          "f1b_age", "f1b_preop_htn", "f1c_vent_mv", "f1c_vent_driving_pressure")


def needed_columns(W: int) -> list[str]:
    return [c.format(W=W) for c in NEEDED]


def add_composites(f: pd.DataFrame, W: int) -> pd.DataFrame:
    """The COMPOSITES columns (float32), computed row by row from columns at the same time t (causal)."""
    g = lambda c: f[c.format(W=W)].to_numpy(np.float64)  # noqa: E731
    mp = g("map_current")
    grey = (mp >= GREY[0]) & (mp <= GREY[1])
    age = g("f1b_age") / 60
    falling = g("f1a_map_slope_m5") < -0.3
    out = {
        "cx_map_ttc_m5": _time_to_cross(mp, g("f1a_map_slope_m5")),
        "cx_map_ttc_m15": _time_to_cross(mp, g("f1a_map_slope_m15")),
        "cx_map_proj_5": mp + 5 * g("f1a_map_slope_m5"),
        "cx_map_proj_10": mp + 10 * g("f1a_map_slope_m10"),
        "cx_map_proj_15": mp + 15 * g("f1a_map_slope_m15"),
        "cx_map_cv": _div(g("map_std_w{W}"), g("map_mean_w{W}")),
        "cx_map_grey_burden": g("f1a_min_70_80_m15") + g("map_time_65_75_w{W}") / 60,
        "cx_map_event_rate": _div(g("f1a_n_prev_events"), np.maximum(g("f1a_min_since_anestart"), 10) / 60),
        "cx_dsi": _div(g("hr_current"), g("dbp_current")),
        "cx_msi": _div(g("hr_current"), mp),
    }
    with np.errstate(all="ignore"):
        dmap = np.log(g("f1a_map_mean_m5")) - np.log(g("f1a_map_mean_m15"))
        dhr = np.log(g("f1a_hr_mean_m5")) - np.log(g("f1a_hr_mean_m15"))
        dsv = (np.log(_lz_sv(g("f1a_pp_mean_m5"), g("f1a_map_mean_m5")))
               - np.log(_lz_sv(g("f1a_pp_mean_m15"), g("f1a_map_mean_m15"))))
    out["cx_dlog_hr"], out["cx_dlog_sv"], out["cx_dlog_svr"] = dhr, dsv, dmap - dhr - dsv
    out["cx_baro_gain"] = np.where(falling, _div(g("f1a_hr_slope_m5"), g("f1a_map_slope_m5")), np.nan)
    nocomp = np.where(np.isnan(g("f1a_map_slope_m5")) | np.isnan(g("f1a_hr_slope_m5")), np.nan,
                      (falling & (g("f1a_hr_slope_m5") <= 0)).astype(float))
    out["cx_no_hr_comp"] = nocomp
    out["cx_no_hr_comp_age"] = nocomp * age
    out["cx_etco2_drop"] = g("f1a_etco2_mean_m5") - g("f1a_etco2_mean_m15")
    es, ms = g("f1a_etco2_slope_m5"), g("f1a_map_slope_m5")
    out["cx_etco2_map_coslope"] = np.where((es < 0) & (ms < 0), es * ms, 0.0)
    out["cx_etco2_map_coslope"][np.isnan(es) | np.isnan(ms)] = np.nan
    out["cx_etco2_per_mv"] = _div(g("etco2_current"), np.where(g("f1c_vent_mv") > 0.5, g("f1c_vent_mv"), np.nan))
    depth = _sum_known(g("f1b_ppf") / 3, g("f1b_rftn") / 4, g("f1b_mac"))
    depth_d5 = _sum_known(g("f1b_ppf_d5") / 3, g("f1b_rftn_d5") / 4, g("f1b_mac_d5"))
    out["cx_depth"], out["cx_depth_d5"] = depth, depth_d5
    out["cx_depth_d5_grey"] = np.where(grey, depth_d5, np.where(np.isnan(mp), np.nan, 0.0))
    out["cx_depth_age"] = depth * age
    bis = g("f1b_bis")
    out["cx_bis_low_grey"] = np.where(np.isnan(bis) | np.isnan(mp), np.nan, ((bis < 40) & grey).astype(float))
    pre = np.isnan(g("f1a_min_since_opstart"))
    out["cx_pre_incision"] = pre.astype(float)
    out["cx_pre_incision_min"] = np.where(pre, g("f1a_min_since_anestart"), np.nan)
    ppv = g("bt_ppv30_w{W}")
    out["cx_ppv_grey"] = np.where(grey, ppv, np.nan)
    out["cx_ppv_x_dp"] = ppv * g("f1c_vent_driving_pressure")
    out["cx_drop_htn"] = g("map_drop_pct") * g("f1b_preop_htn")
    out["cx_drop_age"] = g("map_drop_pct") * age
    cx = pd.DataFrame({k: np.asarray(v, np.float64) for k, v in out.items()}, index=f.index)
    cx = cx.replace([np.inf, -np.inf], np.nan).astype("float32")
    return pd.concat([f, cx[list(COMPOSITES)]], axis=1)


# --------------------------------------------------------------------------------------------- extra inputs

def load_extra(root) -> pd.DataFrame | None:
    """ECG/PPG features (`ep_*`) from every ecgppg_shard*.parquet under root (Kaggle kernel outputs)."""
    if root is None:
        return None
    from pathlib import Path
    paths = sorted(Path(root).rglob("ecgppg_shard*.parquet"))
    if not paths:
        return None
    d = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return d.drop_duplicates(["caseid", "time"])


def merge_extra(X: pd.DataFrame, extra: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """X with the extra columns joined on (caseid, time), row order kept; NaN where a case has no extra row."""
    if extra is None:
        return X, []
    cols = [c for c in extra.columns if c not in ("caseid", "time")]
    keys = X[["caseid", "time"]].merge(extra, on=["caseid", "time"], how="left")
    assert len(keys) == len(X)
    out = pd.concat([X.reset_index(drop=True), keys[cols].astype("float32").reset_index(drop=True)], axis=1)
    out.index = X.index
    return out, cols


# --------------------------------------------------------------------------------------------- folds / offset

def subject_folds(subject: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold id per row; every row of one subject in one fold."""
    u = np.unique(subject)
    perm = np.random.default_rng(seed).permutation(len(u))
    fold_of = dict(zip(u[perm], np.arange(len(u)) % k))
    return np.array([fold_of[s] for s in subject], np.int8)


SMALL = {"objective": "binary", "num_leaves": 4, "learning_rate": 0.05, "min_child_samples": 300,
         "reg_lambda": 10, "verbosity": -1, "num_threads": os.cpu_count() or 4}
SMALL_TREES = 150


def _train(params, X, y, n_trees, init=None, seed=0):
    import lightgbm as lgb
    ds = lgb.Dataset(np.asarray(X, np.float32), y, init_score=init, free_raw_data=True)
    return lgb.train({**params, "seed": seed}, ds, num_boost_round=n_trees)


def map_offset_oof(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray, params: dict, n_trees: int,
                   seed: int) -> np.ndarray:
    """Out-of-fold raw score of a model on X (rows with unknown y keep NaN)."""
    out = np.full(len(y), np.nan)
    ok = np.isin(y, (0, 1))
    for k in np.unique(folds):
        tr, te = ok & (folds != k), ok & (folds == k)
        b = _train(params, X[tr], y[tr], n_trees, seed=seed)
        out[te] = b.predict(np.asarray(X[te], np.float32), raw_score=True)
    return out


def _logloss(y, raw):
    p = 1 / (1 + np.exp(-raw))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def incremental_gain(X: pd.DataFrame, y: np.ndarray, offset: np.ndarray, folds: np.ndarray,
                     holdouts: Sequence[int], seed: int = 0, params: dict = SMALL,
                     n_trees: int = SMALL_TREES) -> dict:
    """Per holdout fold: log-loss and AUROC of offset alone vs offset + a small model on X (trained on the other
    folds with the offset as init score). Returns the per-fold gains and their means."""
    from sklearn.metrics import average_precision_score as ap
    ok = np.isin(y, (0, 1)) & np.isfinite(offset)
    dll, dauc, dap = [], [], []
    for k in holdouts:
        tr, te = ok & (folds != k), ok & (folds == k)
        b = _train(params, X[tr], y[tr], n_trees, init=offset[tr], seed=seed)
        raw = offset[te] + b.predict(np.asarray(X[te], np.float32), raw_score=True)
        dll.append(_logloss(y[te], offset[te]) - _logloss(y[te], raw))
        dauc.append(auroc(y[te], raw) - auroc(y[te], offset[te]))
        dap.append(ap(y[te], raw) - ap(y[te], offset[te]))
    return {"dlogloss": float(np.mean(dll)), "dauroc": float(np.mean(dauc)), "dauprc": float(np.mean(dap)),
            "dlogloss_folds": dll, "dauroc_folds": dauc, "dauprc_folds": dap}


# --------------------------------------------------------------------------------------------- L1 statistics

def _ranks(x):
    from scipy.stats import rankdata
    return rankdata(x)


def partial_spearman(x: np.ndarray, resid: np.ndarray, case: np.ndarray, n_boot: int = 100,
                     seed: int = 0) -> tuple[float, float, float]:
    """Spearman of x with the MAP residual y - p_MAP, and a case-bootstrap 95 % CI."""
    r = spearman_boot(x, resid, case, n_boot, seed)
    return r["rho"], r["lo"], r["hi"]


def _p_from_z(est: float, se: float, null: float = 0.0) -> float:
    from scipy.stats import norm
    if not (np.isfinite(est) and np.isfinite(se)) or se <= 0:
        return np.nan
    return float(2 * norm.sf(abs(est - null) / se))


def spearman_boot(x: np.ndarray, r: np.ndarray, case: np.ndarray, n_boot: int = 100, seed: int = 0) -> dict:
    """Spearman of x with r, case-bootstrap 95 % CI and SE, and a two-sided p from the bootstrap SE (rows of one
    case are correlated, so a row-level p would be far too small). Ranks once, then resamples cases through
    per-case sums (fast)."""
    nan = {"rho": np.nan, "lo": np.nan, "hi": np.nan, "se": np.nan, "p": np.nan}
    m = np.isfinite(x) & np.isfinite(r)
    if m.sum() < 100 or np.nanstd(x[m]) == 0:
        return nan
    a, b = _ranks(x[m]), _ranks(r[m])
    codes, inv = np.unique(case[m], return_inverse=True)
    s = np.vstack([np.bincount(inv, w, len(codes)) for w in (np.ones_like(a), a, b, a * a, b * b, a * b)])

    def corr(w):
        n, sa, sb, saa, sbb, sab = s @ w
        cov = sab / n - sa / n * sb / n
        return cov / np.sqrt(max(saa / n - (sa / n) ** 2, EPS) * max(sbb / n - (sb / n) ** 2, EPS))

    rho = float(corr(np.ones(len(codes))))
    rng = np.random.default_rng(seed)
    bs = np.array([corr(rng.multinomial(len(codes), np.full(len(codes), 1 / len(codes)))) for _ in range(n_boot)])
    se = float(bs.std(ddof=1))
    return {"rho": rho, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "se": se,
            "p": _p_from_z(rho, se)}


def auroc_boot(y: np.ndarray, s: np.ndarray, case: np.ndarray, n_boot: int = 100, seed: int = 0,
               s_ref: np.ndarray | None = None) -> dict:
    """AUROC of s (or, with s_ref, the paired difference AUROC(s) - AUROC(s_ref)), case-bootstrap 95 % CI, SE and
    a two-sided p (H0: AUROC = 0.5, or difference = 0) from the bootstrap SE."""
    m = np.isin(y, (0, 1)) & np.isfinite(s) & (np.isfinite(s_ref) if s_ref is not None else True)
    y, s, case = (y[m] == 1).astype(np.int8), s[m], case[m]
    if len(y) < 50 or y.min() == y.max():
        return {"est": np.nan, "lo": np.nan, "hi": np.nan, "se": np.nan, "p": np.nan}
    ref = s_ref[m] if s_ref is not None else None
    stat = (lambda k: auroc(y[k], s[k]) - auroc(y[k], ref[k])) if ref is not None else (lambda k: auroc(y[k], s[k]))
    est = stat(np.arange(len(y)))
    u, inv = np.unique(case, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(u)))]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(u), len(u))
        k = np.concatenate([order[starts[i]:starts[i + 1]] for i in pick])
        if 0 < y[k].sum() < len(k):
            bs.append(stat(k))
    bs = np.array(bs)
    se = float(bs.std(ddof=1)) if len(bs) > 2 else np.nan
    return {"est": float(est), "lo": float(np.percentile(bs, 2.5)) if len(bs) else np.nan,
            "hi": float(np.percentile(bs, 97.5)) if len(bs) else np.nan, "se": se,
            "p": _p_from_z(est, se, 0.0 if ref is not None else 0.5)}


def logit_offset_test(x: np.ndarray, y: np.ndarray, offset: np.ndarray, cluster: np.ndarray) -> dict:
    """Logistic regression y ~ offset(MAP logit) + b * x/IQR + g * missing(x), cluster-robust SE by subject.
    exp(b) = odds ratio per IQR of x among rows with the same MAP-only risk; x clipped to its 1-99th percentiles,
    missing values set to the median and flagged."""
    nan = {"or_iqr": np.nan, "or_lo": np.nan, "or_hi": np.nan, "p": np.nan}
    try:
        import statsmodels.api as sm
    except ImportError:  # local test machine; Kaggle images have statsmodels
        return nan
    m = np.isin(y, (0, 1)) & np.isfinite(offset)
    x, y, o, g = np.asarray(x, float)[m], (y[m] == 1).astype(float), offset[m], cluster[m]
    fin = np.isfinite(x)
    if fin.sum() < 200 or len(np.unique(x[fin])) < 2:
        return nan
    lo, q1, med, q3, hi = np.percentile(x[fin], [1, 25, 50, 75, 99])
    iqr = q3 - q1 if q3 > q1 else (hi - lo if hi > lo else 1.0)
    z = (np.clip(np.where(fin, x, med), lo, hi) - med) / iqr
    cols = [np.ones_like(z), z] + ([(~fin).astype(float)] if (~fin).any() and fin.any() else [])
    try:
        res = sm.GLM(y, np.column_stack(cols), family=sm.families.Binomial(), offset=o).fit(
            cov_type="cluster", cov_kwds={"groups": g})
    except Exception:  # separation / singular design
        return nan
    b, se = float(res.params[1]), float(res.bse[1])
    return {"or_iqr": float(np.exp(b)), "or_lo": float(np.exp(b - 1.96 * se)), "or_hi": float(np.exp(b + 1.96 * se)),
            "p": float(res.pvalues[1])}


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaN p stay NaN)."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return q
    order = ok[np.argsort(p[ok])]
    n = len(order)
    v = p[order] * n / np.arange(1, n + 1)
    q[order] = np.minimum(np.minimum.accumulate(v[::-1])[::-1], 1.0)
    return q


def oof_incremental(X: pd.DataFrame, y: np.ndarray, offset: np.ndarray, folds: np.ndarray, seed: int = 0,
                    params: dict = SMALL, n_trees: int = SMALL_TREES) -> np.ndarray:
    """Out-of-fold raw score of offset + small model on X (every fold a holdout once), for paired tests."""
    ok = np.isin(y, (0, 1)) & np.isfinite(offset)
    out = np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr, te = ok & (folds != k), ok & (folds == k)
        b = _train(params, X[tr], y[tr], n_trees, init=offset[tr], seed=seed)
        out[te] = offset[te] + b.predict(np.asarray(X[te], np.float32), raw_score=True)
    return out


def _mi(xb: np.ndarray, y: np.ndarray) -> float:
    t = pd.crosstab(xb, y).to_numpy(float)
    if t.shape[1] < 2:
        return 0.0
    p = t / t.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(all="ignore"):
        v = p * np.log(p / (px * py))
    return float(np.nansum(v))


def conditional_mi(x: np.ndarray, y: np.ndarray, offset: np.ndarray, bins: int = 10) -> float:
    """I(x; y | offset decile) in nats, x in quantile bins (NaN is its own bin)."""
    m = np.isin(y, (0, 1)) & np.isfinite(offset)
    x, y, o = x[m], y[m], offset[m]
    xb = pd.qcut(pd.Series(x), bins, labels=False, duplicates="drop").fillna(-1).to_numpy()
    ob = pd.qcut(pd.Series(o), 10, labels=False, duplicates="drop").to_numpy()
    out = 0.0
    for b in np.unique(ob):
        k = ob == b
        out += k.mean() * _mi(xb[k], y[k])
    return out


def grey_auroc(x: np.ndarray, y: np.ndarray, map_now: np.ndarray) -> tuple[float, str]:
    """AUROC (folded to >= 0.5) of x among rows with MAP in the grey zone 65-80 mmHg, and its direction."""
    m = np.isin(y, (0, 1)) & np.isfinite(x) & (map_now >= GREY[0]) & (map_now <= GREY[1])
    if m.sum() < 100 or len(np.unique(x[m])) < 2:
        return np.nan, ""
    a = auroc((y[m] == 1).astype(np.int8), x[m])
    return max(a, 1 - a), ("+" if a >= 0.5 else "-")


# --------------------------------------------------------------------------------------------- L2 synergy

def pair_synergy(X: pd.DataFrame, y: np.ndarray, offset: np.ndarray, folds: np.ndarray, pool: Sequence[str],
                 single: dict[str, list[float]], holdouts: Sequence[int], seed: int = 0) -> pd.DataFrame:
    """syn(a, b) per holdout = dlogloss(a, b) - dlogloss(a) - dlogloss(b), on top of the MAP offset.
    `single[f]` holds the per-holdout dlogloss of f alone (same holdouts)."""
    rows = []
    pool = list(pool)
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            g = incremental_gain(X[[a, b]], y, offset, folds, holdouts, seed)
            syn = np.array(g["dlogloss_folds"]) - np.array(single[a]) - np.array(single[b])
            rows.append({"a": a, "b": b, "dlogloss_pair": g["dlogloss"], "dauroc_pair": g["dauroc"],
                         "synergy": float(syn.mean()), "synergy_pos_folds": int((syn > 0).sum()),
                         "n_folds": len(syn)})
    return pd.DataFrame(rows)


def shap_interactions(booster, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP interaction value| of every column pair (off-diagonal doubled: both halves); needs `shap`."""
    import shap
    iv = shap.TreeExplainer(booster).shap_interaction_values(X)
    if isinstance(iv, list):
        iv = iv[-1]
    m = np.abs(iv).mean(axis=0)
    m = m + m.T - np.diag(np.diag(m))
    return pd.DataFrame(m, index=X.columns, columns=X.columns)


# --------------------------------------------------------------------------------------------- L4 selection

def forward_bundles(X: pd.DataFrame, ys: dict[int, np.ndarray], offsets: dict[int, np.ndarray],
                    folds: np.ndarray, bundles: dict[str, list[str]], holdouts: Sequence[int],
                    max_steps: int = 15, min_gain: float = 0.001, seed: int = 0, log=print) -> pd.DataFrame:
    """Greedy forward selection on the MAP offset. Score of a set = mean over horizons and holdouts of
    (dAUROC + dAUPRC) / 2 against the offset; each step adds the bundle with the best score.
    Stops when the best step gains < min_gain or a step's gain is not positive in every holdout."""
    chosen: list[str] = []
    cols: list[str] = []
    best_prev = 0.0
    steps = []
    remaining = dict(bundles)
    for step in range(max_steps):
        cand = []
        for name, members in remaining.items():
            fs = cols + [c for c in members if c not in cols]
            per = []
            for h, y in ys.items():
                g = incremental_gain(X[fs], y, offsets[h], folds, holdouts, seed)
                per.append((np.array(g["dauroc_folds"]) + np.array(g["dauprc_folds"])) / 2)
            per = np.mean(per, axis=0)
            cand.append((float(per.mean()), name, per, fs))
        cand.sort(key=lambda t: -t[0])
        score, name, per, fs = cand[0]
        gain = score - best_prev
        stable = bool(np.all(per - (steps[-1]["per_fold"] if steps else 0) > 0))
        log(f"  step {step + 1}: + {name} -> score {score:.4f} (gain {gain:+.4f}, stable {stable})")
        if gain < min_gain or not stable:  # kept in the table (kept = False) so the report can say why it stopped
            steps.append({"step": step + 1, "bundle": name, "members": " ".join(bundles[name]), "score": score,
                          "gain": gain, "per_fold": per, "columns": " ".join(fs), "kept": False,
                          "reason": "gain < min_gain" if gain < min_gain else "not positive in every fold"})
            break
        chosen.append(name)
        cols = fs
        best_prev = score
        steps.append({"step": step + 1, "bundle": name, "members": " ".join(bundles[name]), "score": score,
                      "gain": gain, "per_fold": per, "columns": " ".join(cols), "kept": True, "reason": ""})
        remaining.pop(name)
    out = pd.DataFrame(steps)
    if len(out):
        out["per_fold"] = out["per_fold"].map(lambda v: " ".join(f"{x:.4f}" for x in v))
    return out
