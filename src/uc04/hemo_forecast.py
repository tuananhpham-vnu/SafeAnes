"""Forecast where each haemodynamic component is heading, and what drives it (week 14, plan A1-A3).

Targets, for h in 5 / 10 / 15 min (future rows of the same case, used only as labels):
    dlog X(t, h) = log X(t + h) - log X(t),  X in {HR, SV_LZ, SVR_LZ, MAP}, from 2-min means
so that dlog MAP = dlog HR + dlog SV + dlog SVR exactly (mechanism.lz). One LightGBM regressor per component and
horizon; the forecast MAP change is the sum of the three component forecasts, and the component with the most
negative forecast is the one "expected to drive" the fall.

Drivers: TreeSHAP of each component model, summed over readable input groups (DRIVER_GROUPS), so a forecast reads
"SVR -14 % in 15 min: 55 % from anaesthetic depth, 30 % from the MAP trend".
Counterfactual: the same forecast with the recent changes of anaesthetic drugs / ventilation set to "no change".
It shows what the model attributes to those changes; it is not a causal effect (confounding by indication, and
VitalDB has no timed vasopressor boluses).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import mechanism as mc

COMPONENTS = ("hr", "sv", "svr", "map")
HORIZONS = (300, 600, 900)
NOW = ["map_mean_w120", "sbp_mean_w120", "dbp_mean_w120", "hr_mean_w120"]

DRIVER_GROUPS = {  # matched by prefix, in this order
    "Độ mê / thuốc mê": ("f1b_mac", "f1b_ppf", "f1b_rftn", "f1b_bis", "f1c_drug_"),
    "Thở máy": ("f1b_peep", "f1c_vent_"),
    "Khí máu trong mổ": ("f1c_lab_",),
    "Giai đoạn / tiền sử tụt": ("f1a_min_since", "f1a_n_prev"),
    "Hồ sơ bệnh nhân": ("st_", "f1c_pre_", "f1b_age", "f1b_asa", "f1b_preop_htn", "f1b_dept_", "f1b_approach_",
                        "f1b_position_"),
    "Sinh lý riêng trong ca": ("pers_",),
    "Nhịp tim": ("hr_", "shock_index_", "f1a_hr_"),
    "Hô hấp / EtCO2": ("etco2_", "rr_", "spo2_", "f1a_etco2_", "f1b_etco2_"),
    "Huyết áp: mức và xu hướng": ("map_", "sbp_", "dbp_", "pp_", "f1a_map_", "f1a_pp_", "f1a_min_70_80"),
    "Sóng ART": ("bt_",),
}
# recent changes that a clinician makes; "no change" counterfactual sets them to 0
INTERVENTION_DELTAS = ("f1b_ppf_d1", "f1b_ppf_d5", "f1b_ppf_d10", "f1b_rftn_d1", "f1b_rftn_d5", "f1b_rftn_d10",
                       "f1b_mac_d1", "f1b_mac_d5", "f1b_mac_d10", "f1c_drug_exp_sevo_d5", "f1c_drug_exp_des_d5",
                       "f1c_drug_ppf_rate_d1", "f1c_drug_ppf_rate_d5", "f1c_drug_rftn_rate_d1", "f1c_drug_rftn_rate_d5",
                       "f1c_vent_pip_d1", "f1c_vent_pip_d5", "f1c_vent_pplat_d1", "f1c_vent_pplat_d5",
                       "f1c_vent_tv_d1", "f1c_vent_tv_d5", "f1c_vent_mv_d1", "f1c_vent_mv_d5")

LGB_REG = {"objective": "huber", "alpha": 0.05, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 200,
           "reg_lambda": 10, "verbosity": -1}
N_TREES = 400


def driver_group(col: str) -> str:
    return next((g for g, pre in DRIVER_GROUPS.items() if col.startswith(pre)), "Khác")


def targets(feats: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """dlog of each component from t to t + h (NaN when the row at t + h does not exist or is incomplete)."""
    now = mc.lz(feats["sbp_mean_w120"], feats["dbp_mean_w120"], feats["map_mean_w120"], feats["hr_mean_w120"])
    out = {}
    for h in horizons:
        fut = mc.add_lagged(feats[["caseid", "time", *NOW]], NOW, lag=-h)  # value at time + h
        later = mc.lz(fut["sbp_mean_w120_lag"], fut["dbp_mean_w120_lag"], fut["map_mean_w120_lag"],
                      fut["hr_mean_w120_lag"])
        with np.errstate(all="ignore"):
            out[f"d_hr_{h}"] = (later[0] - now[0]).to_numpy()
            out[f"d_sv_{h}"] = (later[1] - now[1]).to_numpy()
            out[f"d_svr_{h}"] = (later[2] - now[2]).to_numpy()
            out[f"d_map_{h}"] = np.log(fut["map_mean_w120_lag"].to_numpy()) - np.log(feats["map_mean_w120"].to_numpy())
    return pd.DataFrame(out, index=feats.index)


def fit(X: pd.DataFrame, y: np.ndarray, seed: int, threads: int):
    import lightgbm as lgb
    ok = np.isfinite(y)
    # clip extreme targets (artefacts, flushes) before fitting; Huber handles the rest
    yy = np.clip(y[ok], -1.0, 1.0)
    return lgb.train({**LGB_REG, "seed": seed, "num_threads": threads}, lgb.Dataset(X[ok], yy),
                     num_boost_round=N_TREES)


def group_shap(booster, X: pd.DataFrame) -> pd.DataFrame:
    """Per row: signed TreeSHAP summed by driver group (+ 'base' = expected value)."""
    contrib = booster.predict(X, pred_contrib=True)
    groups = [driver_group(c) for c in X.columns]
    out = pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index).T.groupby(groups).sum().T
    out["base"] = contrib[:, -1]
    return out


def counterfactual(booster, X: pd.DataFrame) -> np.ndarray:
    """Forecast with every recent intervention change set to 0 (no change)."""
    Xc = X.copy()
    for c in INTERVENTION_DELTAS:
        if c in Xc:
            Xc[c] = np.where(np.isfinite(Xc[c]), 0.0, Xc[c])
    return booster.predict(Xc)


def regression_metrics(y: np.ndarray, p: np.ndarray, dead: float = 0.02) -> dict:
    from scipy.stats import spearmanr
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    ss = ((y - y.mean()) ** 2).sum()
    moved = np.abs(y) >= dead
    return {"n": int(ok.sum()), "r2": float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan,
            "spearman": float(spearmanr(y, p).statistic) if len(y) > 2 else np.nan,
            "direction_acc": float((np.sign(p[moved]) == np.sign(y[moved])).mean()) if moved.any() else np.nan,
            "moved_share": float(moved.mean())}


def personal_physiology(feats: pd.DataFrame) -> pd.DataFrame:
    """Causal in-case personal physiology (plan C3), from the 30-s rows of each case up to t:
      pers_baro      slope of HR on MAP (2-min means) over the previous 30 min: negative = HR rises when MAP
                     falls (working baroreflex); near 0 = blunted response (e.g. diabetic autonomic neuropathy,
                     beta-blockade, deep anaesthesia)
      pers_case_min  lowest 2-min MAP mean of the case so far
      pers_map_sd    SD of the 2-min MAP mean over the previous 30 min
    `feats` must hold every main-cohort row of the cases (not only eligible rows)."""
    f = feats[["caseid", "time", "map_mean_w120", "hr_mean_w120"]].sort_values(["caseid", "time"])
    out = []
    for _, g in f.groupby("caseid", sort=False):
        m, h = g["map_mean_w120"].reset_index(drop=True), g["hr_mean_w120"].reset_index(drop=True)
        r = m.rolling(60, min_periods=20)  # 30 min of 30-s rows
        with np.errstate(all="ignore"):
            baro = (m.rolling(60, min_periods=20).cov(h) / r.var()).to_numpy()
        out.append(pd.DataFrame({"pers_baro": np.clip(baro, -5, 5), "pers_case_min": m.cummin().to_numpy(),
                                 "pers_map_sd": r.std().to_numpy()}, index=g.index))
    return pd.concat(out).reindex(feats.index).astype("float32")
