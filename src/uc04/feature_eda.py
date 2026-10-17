"""Feature-level EDA on samples v3 (week 14): what every input is, how it relates to each label,
which inputs carry the same information, which matter, and how few are enough.

Candidate inputs (all causal, known at the prediction time t):
  numeric  66 monitor columns of window W               (columns.numeric_cols)
  wave     27 arterial-waveform beat columns of W        (columns.wave_cols; missing before incision)
  context  31 case-context columns f1a_*                 (trends 5-30 min, time since induction/incision, ...)
  anaes    anaesthesia and ventilation f1b_*/f1c_*       (MAC, propofol/remifentanil Ce and pump, BIS, PEEP, vent, ABG)
  clinical pre-operative patient/surgery facts           (f1b age/ASA/HTN/department/approach/position, f1c_pre_*,
                                                          st_* from the cohort manifest: diabetes, glucose, platelets, ECG)
`anaes` and `clinical` are outside the monitor-only scope of plan 3.2: they are studied here, not put in the locked
models. End-of-case totals of the manifest (intraop_*: blood loss, fluids, vasopressor totals) are never used: they
are only known after the case.

Only train and validation are read; the test split is never touched. Statistics that choose anything (redundancy
clusters, univariate ranking, the selection order) come from train; validation only scores fitted models.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .columns import CONTEXT_COLS, numeric_cols, wave_cols

HORIZONS = (300, 600, 900)
LABEL_COLS = ("caseid", "subjectid", "time", "exposure_seconds", "eligible", "in_main_cohort",
              *(f"y_{h}" for h in HORIZONS))
GROUP_ORDER = ("numeric", "wave", "context", "anaes", "clinical")

# --------------------------------------------------------------------------------------------- dictionary

SIGNALS = {
    "map": ("MAP", "Huyết áp động mạch trung bình", "mmHg"),
    "sbp": ("SBP", "Huyết áp tâm thu", "mmHg"),
    "dbp": ("DBP", "Huyết áp tâm trương", "mmHg"),
    "hr": ("HR", "Nhịp tim", "nhịp/phút"),
    "spo2": ("SpO2", "Độ bão hòa oxy mạch nảy", "%"),
    "etco2": ("EtCO2", "CO2 cuối thì thở ra (gián tiếp phản ánh cung lượng tim và thông khí)", "mmHg"),
    "rr": ("RR", "Nhịp thở", "lần/phút"),
    "pp": ("PP", "Áp lực mạch = SBP − DBP (gián tiếp phản ánh thể tích nhát bóp)", "mmHg"),
    "shock_index": ("Shock index", "Chỉ số sốc = HR / SBP", "–"),
}
STATS = {"mean": "trung bình", "std": "độ lệch chuẩn", "min": "giá trị thấp nhất", "max": "giá trị cao nhất",
         "slope": "độ dốc (mỗi phút)", "missing": "tỷ lệ ô lưới 2 s bị thiếu", "current": "giá trị hiện tại",
         "d1": "thay đổi trong 1 phút", "d5": "thay đổi trong 5 phút", "d10": "thay đổi trong 10 phút"}
BEATS = {
    "dpdt_max": ("dP/dt max: độ dốc lớn nhất của pha tống máu (proxy sức co bóp)", "mmHg/s"),
    "sys_area": ("Diện tích dưới sóng tâm thu (liên quan thể tích nhát bóp)", "mmHg·s"),
    "ejection_time": ("Thời gian tống máu", "s"),
    "notch_rel": ("Độ cao khuyết dicrotic so với biên độ nhịp (phản ánh trương lực mạch)", "–"),
    "decay_tau": ("Hằng số thời gian giảm áp tâm trương ≈ kháng lực × độ đàn hồi mạch", "s"),
    "sv": ("Proxy thể tích nhát bóp (Liljestrand–Zander, tính từ huyết áp)", "tương đối"),
    "co": ("Proxy cung lượng tim = SV proxy × HR (tính từ huyết áp)", "tương đối"),
    "svr": ("Proxy kháng lực mạch hệ thống = MAP / CO proxy (tính từ huyết áp)", "tương đối"),
}
SIDE = {  # f1b_ / f1c_ stems
    "mac": ("Nồng độ phế nang tối thiểu của khí mê (MAC)", "MAC"),
    "ppf": ("Nồng độ đích Ce propofol (TCI, ước lượng dược động học)", "µg/mL"),
    "rftn": ("Nồng độ đích Ce remifentanil (TCI)", "ng/mL"),
    "bis": ("Chỉ số BIS (độ sâu mê)", "0–100"),
    "bis_mean_m5": ("BIS trung bình 5 phút", "0–100"),
    "peep": ("Áp lực dương cuối thì thở ra (PEEP)", "cmH2O"),
    "etco2_d1": ("Thay đổi EtCO2 trong 1 phút", "mmHg"), "etco2_d5": ("Thay đổi EtCO2 trong 5 phút", "mmHg"),
    "age": ("Tuổi", "năm"), "asa": ("Phân loại ASA", "1–6"), "preop_htn": ("Tăng huyết áp trước mổ", "0/1"),
    "vent_pip": ("Áp lực đỉnh đường thở", "cmH2O"), "vent_pplat": ("Áp lực bình nguyên", "cmH2O"),
    "vent_tv": ("Thể tích khí lưu thông", "mL"), "vent_mv": ("Thông khí phút", "L/phút"),
    "vent_compliance": ("Độ giãn nở phổi", "mL/cmH2O"), "vent_driving_pressure": ("Áp lực đẩy", "cmH2O"),
    "drug_ppf_rate": ("Tốc độ bơm propofol", "mL/h"), "drug_ppf_vol": ("Thể tích propofol đã bơm", "mL"),
    "drug_ppf_cp": ("Nồng độ huyết tương Cp propofol", "µg/mL"),
    "drug_rftn_rate": ("Tốc độ bơm remifentanil", "mL/h"), "drug_rftn_vol": ("Thể tích remifentanil đã bơm", "mL"),
    "drug_rftn_cp": ("Nồng độ huyết tương Cp remifentanil", "ng/mL"),
    "drug_exp_sevo": ("Nồng độ sevoflurane thở ra", "%"), "drug_exp_des": ("Nồng độ desflurane thở ra", "%"),
    "drug_fen2o": ("N2O thở ra", "%"), "drug_bis_sr": ("Tỷ lệ ức chế bùng phát EEG (BIS SR)", "%"),
    "drug_bt": ("Nhiệt độ", "°C"),
    "lab_hct": ("Hct khí máu gần nhất trong mổ", "%"), "lab_lac": ("Lactate gần nhất", "mmol/L"),
    "lab_ph": ("pH gần nhất", "–"), "lab_ica": ("Ca ion gần nhất", "mmol/L"), "lab_hco3": ("HCO3 gần nhất", "mmol/L"),
    "lab_min_since_sample": ("Số phút từ mẫu khí máu gần nhất", "phút"), "lab_n_abg": ("Số mẫu khí máu đã lấy", "mẫu"),
    "pre_hb": ("Hb trước mổ", "g/dL"), "pre_alb": ("Albumin trước mổ", "g/dL"), "pre_cr": ("Creatinine trước mổ", "mg/dL"),
    "pre_na": ("Na trước mổ", "mmol/L"), "pre_k": ("K trước mổ", "mmol/L"), "pre_bmi": ("BMI", "kg/m²"),
    "pre_male": ("Giới nam", "0/1"), "pre_emop": ("Mổ cấp cứu", "0/1"),
}
STATIC_MANIFEST = {  # st_<name>: (manifest column or rule, meaning, unit)
    "st_preop_dm": ("Đái tháo đường trước mổ", "0/1"),
    "st_preop_gluc": ("Glucose máu trước mổ", "mg/dL"),
    "st_preop_plt": ("Tiểu cầu trước mổ", "×10³/µL"),
    "st_preop_ecg_abn": ("ECG trước mổ bất thường (khác nhịp xoang bình thường)", "0/1"),
}
CLINICAL_F1B = ("f1b_age", "f1b_asa", "f1b_preop_htn", "f1b_dept_", "f1b_approach_", "f1b_position_")


def feature_group(col: str) -> str:
    if col.startswith("bt_"):
        return "wave"
    if col.startswith("f1a_"):
        return "context"
    if col.startswith(("st_", "f1c_pre_", *CLINICAL_F1B)):
        return "clinical"
    if col.startswith(("f1b_", "f1c_")):
        return "anaes"
    return "numeric"


def describe(col: str) -> tuple[str, str, str]:
    """(signal, meaning in Vietnamese, unit) of one candidate column."""
    if col in STATIC_MANIFEST:
        return ("Lâm sàng", *STATIC_MANIFEST[col])
    if col == "map_drop_pct":
        return "MAP", "% MAP giảm so với MAP nền của ca", "%"
    m = re.fullmatch(r"map_time_65_75_w(\d+)", col)
    if m:
        return "MAP", f"Số giây MAP nằm trong vùng 65–75 mmHg trong {m[1]} s", "s"
    m = re.fullmatch(r"map_extrap_w(\d+)", col)
    if m:
        return "MAP", f"MAP ngoại suy theo độ dốc {m[1]} s", "mmHg"
    m = re.fullmatch(r"([a-z_0-9]+?)_(mean|std|min|max|slope|missing|current)(?:_w(\d+))?", col)
    if m and m[1] in SIGNALS:
        short, name, unit = SIGNALS[m[1]]
        win = f" trong {m[3]} s" if m[3] else ""
        return short, f"{name}: {STATS[m[2]]}{win}", "tỷ lệ" if m[2] == "missing" else unit
    m = re.fullmatch(r"bt_([a-z_]+?)_(mean|std|slope)_w(\d+)", col)
    if m and m[1] in BEATS:
        name, unit = BEATS[m[1]]
        return "Sóng ART", f"{name}: {STATS[m[2]]} theo nhịp trong {m[3]} s", unit
    m = re.fullmatch(r"bt_(ppv30|ppv|valid_frac|n_beats)_w(\d+)", col)
    if m:
        name = {"ppv30": "PPV: biến thiên áp lực mạch theo hô hấp (đáp ứng dịch)", "ppv": "PPV",
                "valid_frac": "Tỷ lệ nhịp hợp lệ (chất lượng sóng)", "n_beats": "Số nhịp phát hiện"}[m[1]]
        return "Sóng ART", f"{name} trong {m[2]} s", "%" if m[1].startswith("ppv") else "–"
    if col.startswith("f1a_"):
        stem = col[4:]
        if stem == "map_rel_case_median":
            return "MAP", "MAP hiện tại so với trung vị MAP từ đầu ca tới t", "tỷ số"
        if stem == "map_rel_max_m15":
            return "MAP", "MAP hiện tại so với MAP cao nhất 15 phút qua", "tỷ số"
        if stem == "min_70_80_m15":
            return "MAP", "Số phút MAP trong vùng 70–80 mmHg trong 15 phút qua", "phút"
        if stem == "min_since_opstart":
            return "Thời gian", "Số phút từ lúc rạch da (thiếu trước rạch da)", "phút"
        if stem == "min_since_anestart":
            return "Thời gian", "Số phút từ lúc bắt đầu gây mê", "phút"
        if stem == "n_prev_events":
            return "Tiền sử trong ca", "Số đợt tụt đã xảy ra trong ca", "đợt"
        if stem == "min_since_last_event":
            return "Tiền sử trong ca", "Số phút từ đợt tụt gần nhất", "phút"
        m = re.fullmatch(r"([a-z0-9]+)_(mean|min|slope)_m(\d+)", stem)
        if m and m[1] in SIGNALS:
            short, name, unit = SIGNALS[m[1]]
            return short, f"{name}: {STATS[m[2]]} trong {m[3]} phút", unit
    if col.startswith(("f1b_", "f1c_")):
        stem = col[4:]
        for p in ("dept_", "approach_", "position_"):
            if stem.startswith(p):
                kind = {"dept_": "Khoa", "approach_": "Đường mổ", "position_": "Tư thế"}[p]
                return "Lâm sàng", f"{kind}: {stem[len(p):]}", "0/1"
        m = re.fullmatch(r"(.+?)(?:_(d1|d5|d10)|_mean_m5|_m5|_m15)?", stem)
        base, suffix = m[1], col[len("f1x_") + len(m[1]):]
        name, unit = SIDE.get(stem, SIDE.get(base, (base, "")))
        extra = {"_d1": ", thay đổi 1 phút", "_d5": ", thay đổi 5 phút", "_d10": ", thay đổi 10 phút",
                 "_mean_m5": ", trung bình 5 phút", "_m5": ", trong 5 phút", "_m15": ", trong 15 phút"}
        sig = "Lâm sàng" if feature_group(col) == "clinical" else "Gây mê/thở máy/khí máu"
        return sig, name + ("" if stem in SIDE else extra.get(suffix, "")), unit
    return "?", col, ""


# --------------------------------------------------------------------------------------------- loading

def candidate_columns(samples: Path, W: int) -> dict[str, list[str]]:
    import json
    side = json.loads((Path(samples) / "features_v5" / "columns.json").read_text())
    f1bc = [c for g, cols in side.items() if g != "f1a" for c in cols]
    out = {"numeric": numeric_cols(W), "wave": wave_cols(W), "context": list(CONTEXT_COLS),
           "anaes": [c for c in f1bc if feature_group(c) == "anaes"],
           "clinical": [c for c in f1bc if feature_group(c) == "clinical"] + list(STATIC_MANIFEST)}
    return out


def static_from_manifest(manifest: Path) -> pd.DataFrame:
    m = pd.read_csv(manifest, usecols=["caseid", "preop_dm", "preop_gluc", "preop_plt", "preop_ecg"])
    ecg = m["preop_ecg"].astype("string")
    return pd.DataFrame({
        "caseid": m["caseid"],
        "st_preop_dm": m["preop_dm"].astype("float32"),
        "st_preop_gluc": m["preop_gluc"].astype("float32"),
        "st_preop_plt": m["preop_plt"].astype("float32"),
        "st_preop_ecg_abn": np.where(ecg.isna(), np.nan, (ecg != "Normal Sinus Rhythm").astype(float)).astype("float32"),
    })


def load_rows(samples: Path, split: str, columns: Sequence[str], manifest: Path, cases: set[int] | None = None,
              all_main: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(labels, float32 features) of main-cohort eligible rows with at least one known label
    (`all_main=True`: every main-cohort row, as the alarm replay needs).

    Reads row group by row group and column batch by column batch so peak memory stays near the output size
    (the 8 GB laptop cannot hold a whole train table).
    """
    samples = Path(samples)
    lab = pd.read_parquet(samples / "labels" / f"{split}.parquet", columns=list(LABEL_COLS))
    keep = lab["in_main_cohort"].to_numpy(bool)
    if not all_main:
        keep &= lab["eligible"].to_numpy(bool)
        keep &= np.any([np.isin(lab[f"y_{h}"].to_numpy(), (0, 1)) for h in HORIZONS], axis=0)
    if cases is not None:
        keep &= lab["caseid"].isin(cases).to_numpy()
    lab = lab[keep].reset_index(drop=True)

    main = [c for c in columns if feature_group(c) in ("numeric", "wave")]
    side = [c for c in columns if c.startswith(("f1a_", "f1b_", "f1c_"))]
    stat = [c for c in columns if c.startswith("st_")]

    pf = pq.ParquetFile(samples / "features" / f"{split}.parquet")
    parts, start = [], 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        k = keep[start:start + n]
        start += n
        if k.any():
            t = pf.read_row_group(rg, columns=["caseid", "time", *main]).to_pandas()
            parts.append(t[k].astype({c: "float32" for c in main}))
    feats = pd.concat(parts, ignore_index=True)
    if not (np.array_equal(feats["caseid"].to_numpy(), lab["caseid"].to_numpy())
            and np.array_equal(feats["time"].to_numpy(), lab["time"].to_numpy())):
        raise ValueError(f"labels and features of {split} are not row-aligned")

    if side:
        sp = samples / "features_v5" / f"{split}.parquet"
        keys = pq.read_table(sp, columns=["caseid", "time"]).to_pandas()
        keys["_pos"] = np.arange(len(keys))
        pos = lab[["caseid", "time"]].merge(keys, on=["caseid", "time"], how="left")["_pos"].to_numpy()
        if np.isnan(pos).any():
            raise ValueError(f"features_v5/{split} misses {int(np.isnan(pos).sum())} rows")
        pos = pos.astype(np.int64)
        extra = {}
        for i in range(0, len(side), 16):
            batch = side[i:i + 16]
            t = pq.read_table(sp, columns=batch)
            for c in batch:
                extra[c] = t.column(c).to_numpy().astype("float32")[pos]
            del t
        feats = pd.concat([feats, pd.DataFrame(extra)], axis=1)
    if stat:
        st = static_from_manifest(manifest)
        feats = feats.merge(st[["caseid", *stat]], on="caseid", how="left")
    return lab, feats[["caseid", "time", *columns]]


def sample_cases(labels_path: Path, n: int, seed: int) -> set[int]:
    """`n` train cases drawn by subject (one case per drawn subject is not required; all cases of a subject go)."""
    lab = pd.read_parquet(labels_path, columns=["caseid", "subjectid", "in_main_cohort"])
    lab = lab[lab["in_main_cohort"]].drop_duplicates("caseid")
    rng = np.random.default_rng(seed)
    subj = rng.permutation(lab["subjectid"].unique())
    out: set[int] = set()
    for s in subj:
        out |= set(lab.loc[lab["subjectid"] == s, "caseid"].tolist())
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------------------------- statistics

def auroc(y: np.ndarray, x: np.ndarray) -> float:
    """AUROC of score x for binary y (ties averaged); NaN when one class is empty."""
    from scipy.stats import rankdata
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stratified_auroc(y: np.ndarray, x: np.ndarray, strata: np.ndarray) -> float:
    """AUROC within strata, pooled with weight n_pos * n_neg: how well x ranks rows that share a stratum."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        ys = y[m]
        n1 = ys.sum()
        w = n1 * (len(ys) - n1)
        if w == 0:
            continue
        num += w * auroc(ys, x[m])
        den += w
    return num / den if den else np.nan


MAP_BINS = np.r_[-np.inf, np.arange(60, 106, 5), np.inf]


def univariate(lab: pd.DataFrame, feats: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Per feature and horizon: coverage, AUROC folded to >= 0.5 with its direction, and the same AUROC within
    5 mmHg bins of current MAP (information beyond the current MAP)."""
    rows = []
    map_bin = np.digitize(feats["map_current"].to_numpy(), MAP_BINS)
    for c in columns:
        x = feats[c].to_numpy(np.float64)
        rec = {"feature": c, "group": feature_group(c), "missing": float(np.isnan(x).mean())}
        for h in HORIZONS:
            y = lab[f"y_{h}"].to_numpy()
            m = np.isin(y, (0, 1)) & np.isfinite(x)
            yy, xx = (y[m] == 1).astype(np.int8), x[m]
            if len(np.unique(xx)) < 2:
                a = s = np.nan
            else:
                a = auroc(yy, xx)
                mm = m & np.isfinite(feats["map_current"].to_numpy())
                s = stratified_auroc((y[mm] == 1).astype(np.int8), x[mm], map_bin[mm]) if c != "map_current" else 0.5
            rec[f"auroc_{h}"] = max(a, 1 - a) if np.isfinite(a) else np.nan
            rec[f"dir_{h}"] = ("cao → nguy cơ" if a >= 0.5 else "thấp → nguy cơ") if np.isfinite(a) else ""
            rec[f"auroc_given_map_{h}"] = max(s, 1 - s) if np.isfinite(s) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def risk_by_decile(lab: pd.DataFrame, feats: pd.DataFrame, columns: Sequence[str], h: int) -> pd.DataFrame:
    y = lab[f"y_{h}"].to_numpy()
    out = []
    for c in columns:
        x = feats[c].to_numpy(np.float64)
        m = np.isin(y, (0, 1)) & np.isfinite(x)
        q = pd.qcut(pd.Series(x[m]), 10, labels=False, duplicates="drop")
        d = pd.DataFrame({"bin": q.to_numpy(), "x": x[m], "y": y[m]}).groupby("bin").agg(
            x_median=("x", "median"), risk=("y", "mean"), n=("y", "size")).reset_index()
        d.insert(0, "feature", c)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def spearman(feats: pd.DataFrame, columns: Sequence[str], n_rows: int, seed: int) -> pd.DataFrame:
    s = feats[list(columns)].sample(min(n_rows, len(feats)), random_state=seed)
    return s.rank().corr(min_periods=500)  # Pearson of ranks = Spearman, pairwise complete


def redundancy_clusters(corr: pd.DataFrame, threshold: float = 0.8) -> pd.Series:
    """Average-linkage clusters on 1 - |rho|; features in one cluster have |rho| >= threshold on average."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    d = 1 - corr.abs().fillna(0).to_numpy()
    np.fill_diagonal(d, 0)
    d = (d + d.T) / 2
    z = linkage(squareform(d, checks=False), method="average")
    return pd.Series(fcluster(z, t=1 - threshold, criterion="distance"), index=corr.index, name="cluster")


# --------------------------------------------------------------------------------------------- models

LGB_PARAMS = {"objective": "binary", "num_leaves": 7, "learning_rate": 0.03, "min_child_samples": 150,
              "reg_lambda": 10, "verbosity": -1, "num_threads": os.cpu_count() or 4}  # configs/uc04_v3.json lightgbm.params
N_TREES = 500
MONOTONE = ("map_current", "map_mean_w{W}", "map_min_w{W}")


def fit_lgbm(X: pd.DataFrame, y: np.ndarray, W: int, seed: int):
    import lightgbm as lgb
    dec = {c.replace("{W}", str(W)) for c in MONOTONE}
    params = {**LGB_PARAMS, "seed": seed, "monotone_constraints": [-1 if c in dec else 0 for c in X.columns]}
    return lgb.train(params, lgb.Dataset(X, y, free_raw_data=True), num_boost_round=N_TREES)


def known(lab: pd.DataFrame, h: int) -> np.ndarray:
    return np.isin(lab[f"y_{h}"].to_numpy(), (0, 1))


def eval_auroc(booster, X: pd.DataFrame, y: np.ndarray) -> float:
    return auroc(y, booster.predict(X, raw_score=True))


def grouped_permutation(booster, X: pd.DataFrame, y: np.ndarray, groups: dict[str, list[str]], seed: int,
                        repeats: int = 2) -> pd.DataFrame:
    """AUROC drop when all columns of a group are permuted together (rows shuffled jointly, so the
    within-group relation is kept and only the link to the label is broken)."""
    base = eval_auroc(booster, X, y)
    rng = np.random.default_rng(seed)
    rows = []
    for g, cols in groups.items():
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[idx]
            drops.append(base - eval_auroc(booster, Xp, y))
        rows.append({"cluster": g, "n_features": len(cols), "auroc_drop": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("auroc_drop", ascending=False), base


def shap_by_feature(booster, X: pd.DataFrame) -> pd.Series:
    contrib = booster.predict(X, pred_contrib=True)[:, :-1]
    return pd.Series(np.abs(contrib).mean(axis=0), index=X.columns)
