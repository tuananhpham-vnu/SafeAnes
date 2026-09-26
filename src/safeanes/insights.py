"""Insight analyses on the full VitalDB cohort for UC04 (reports/EDA/INSIGHTS.md).

1. describe  — data dictionary of every feature at the 30 s decision rows (coverage, missing, range)
2. noise     — raw-track noise: gaps, invalid/implausible values, flatlines, spikes, bad beats, BIS SQI
3. relations — Spearman structure between features, association with the FUTURE 5-min MAP change,
               risk curves, MAP level x trend risk map
4. influence — univariate AUROC, LightGBM + TreeSHAP (pred_contrib), feature-group ablation

Scope: noise uses every eligible case (it does not touch outcomes). Everything with labels uses only
development cases (outside the locked global test). Model-based importance trains on FIT and reports
on VALIDATION; it is descriptive and never used to select a model for the test set.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import eda
from .config import BOUNDS, TRACKS
from .data import fetch_csv, read_numeric
from .signals import causal_sample

# Dynamic UC04 features added to the E08 window table: current = median of last 60 s,
# d300 = change versus the 60 s median 5 minutes earlier (log-ratio for positive scale signals).
DYNAMIC = {"bt_sv_lz": True, "bt_svr_lz": True, "bt_co_lz": True, "bt_dpdt": True, "bt_pp": True,
           "bt_ppv": False, "bt_valid_frac": False, "bis": False, "ppf_ce": False, "rftn_ce": False,
           "mac": False, "cvp": False, "peep": False}
STATIC_EXTRA = {"sex": lambda m: m.sex.eq("F").astype(float), "emop": lambda m: m.emop.astype(float),
                "preop_htn": lambda m: m.preop_htn.astype(float), "preop_hb": lambda m: m.preop_hb.astype(float)}
NOT_FEATURES = {"caseid", "subjectid", "time", "eligible", "history_coverage", "exposure_seconds",
                "y_300", "y_600", "role", "dmap_future_300"}
GROUPS = [  # (group, predicate on feature name) — first match wins
    ("MAP", lambda f: f.startswith("map_")),
    ("SBP/DBP", lambda f: f.startswith(("sbp_", "dbp_"))),
    ("HR", lambda f: f.startswith("hr_")),
    ("SpO2", lambda f: f.startswith("spo2_")),
    ("EtCO2/RR", lambda f: f.startswith(("etco2_", "rr_"))),
    ("Beat (ART waveform)", lambda f: f.startswith("bt_")),
    ("Độ mê / thuốc mê", lambda f: f.startswith(("bis_", "ppf_ce_", "rftn_ce_", "mac_"))),
    ("Thở máy / CVP", lambda f: f.startswith(("peep_", "cvp_"))),
    ("Static", lambda f: f.startswith("static_") or f in STATIC_EXTRA),
]
# Raw tracks audited for noise: name -> (track, max_age_s, implausible predicate description, predicate)
NOISE_TRACKS = {
    "ART MAP": ("Solar8000/ART_MBP", 30, "MAP < 20 hoặc > 200", lambda v: (v < 20) | (v > 200)),
    "ART SBP": ("Solar8000/ART_SBP", 30, "SBP < 30 hoặc > 260", lambda v: (v < 30) | (v > 260)),
    "ART DBP": ("Solar8000/ART_DBP", 30, "DBP < 10 hoặc > 160", lambda v: (v < 10) | (v > 160)),
    "HR": ("Solar8000/HR", 30, "HR < 25 hoặc > 200", lambda v: (v < 25) | (v > 200)),
    "SpO2": ("Solar8000/PLETH_SPO2", 30, "SpO2 < 50", lambda v: v < 50),
    "EtCO2": ("Solar8000/ETCO2", 30, "EtCO2 > 80", lambda v: v > 80),
    "RR": ("Solar8000/RR_CO2", 30, "RR > 60", lambda v: v > 60),
    "NIBP MAP": ("Solar8000/NIBP_MBP", 600, "MAP < 20 hoặc > 200", lambda v: (v < 20) | (v > 200)),
    "CVP": ("Solar8000/CVP", 60, "CVP < −5 hoặc > 40", lambda v: (v < -5) | (v > 40)),
    "BIS": ("BIS/BIS", 60, "BIS = 0 hoặc > 100", lambda v: (v <= 0) | (v > 100)),
    "BIS SQI": ("BIS/SQI", 60, "SQI < 50 (BIS kém tin cậy)", lambda v: v < 50),
    "Propofol Ce": ("Orchestra/PPF20_CE", 60, "Ce > 12 µg/mL", lambda v: v > 12),
    "Remifentanil Ce": ("Orchestra/RFTN20_CE", 60, "Ce > 20 ng/mL", lambda v: v > 20),
    "MAC": ("Primus/MAC", 60, "MAC > 3", lambda v: v > 3),
    "Nhiệt độ": ("Solar8000/BT", 60, "BT < 30 hoặc > 42", lambda v: (v < 30) | (v > 42)),
}
FLAT_SPIKE = {"ART MAP": 30, "ART SBP": 40, "HR": 30}   # spike = |jump| between successive samples


def group_of(feature):
    return next((g for g, match in GROUPS if match(feature)), "Khác")


# ---------------------------------------------------------------- 1. decision-row feature table
def case_rows(args):
    root, record, stride = args
    import joblib
    p = eda.Paths(root)
    frame = joblib.load(p.cases / f"{int(record['caseid'])}.joblib")["frame"]
    frame = frame[frame.eligible].iloc[::stride].copy()
    if frame.empty:
        return None
    grid = eda.case_grid(root, record).set_index("time")
    roll = grid[list(DYNAMIC) + ["map"]].rolling(30, min_periods=10).median()
    cur = roll.reindex(frame.time.to_numpy())
    prev = roll.reindex(frame.time.to_numpy() - 300)
    future = roll["map"].reindex(frame.time.to_numpy() + 300)
    for name, log in DYNAMIC.items():
        c, b = cur[name].to_numpy(), prev[name].to_numpy()
        frame[f"{name}_cur"] = c
        with np.errstate(all="ignore"):
            frame[f"{name}_d300"] = np.where((c > 0) & (b > 0), np.log(c / b), np.nan) if log else c - b
    frame["dmap_future_300"] = future.to_numpy() - cur["map"].to_numpy()
    for name, fn in STATIC_EXTRA.items():
        frame[name] = float(fn(pd.DataFrame([record])).iloc[0])
    frame["role"] = record["role"]
    numeric = frame.select_dtypes("number").columns.difference(["caseid", "subjectid", "time"])
    frame[numeric] = frame[numeric].astype("float32")
    return frame


def build_table(root, manifest, stride=2, workers=8, limit=None):
    # Imported here, not at module level: e08_methods pulls in torch, which each spawned worker
    # would otherwise load (exhausts RAM/paging file on an 8 GB Windows machine).
    from .e08_methods import FULL_ROLES
    records = eda.attach_extra_tids(root, manifest)
    records = records[records.outcome_scope & records.evaluation_group.isin(FULL_ROLES)].copy()
    records["role"] = records.evaluation_group.map(FULL_ROLES)
    records = records.to_dict("records")[:limit]
    parts = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(case_rows, [(root, r, stride) for r in records], chunksize=8):
            if part is not None:
                parts.append(part)
    return pd.concat(parts, ignore_index=True)


def feature_columns(table):
    return [c for c in table.columns if c not in NOT_FEATURES]


def dictionary(table):
    """One row per feature: group, % missing, median [IQR], range."""
    rows = []
    for f in feature_columns(table):
        v = table[f].to_numpy(float)
        ok = v[np.isfinite(v)]
        q = np.percentile(ok, [1, 25, 50, 75, 99]) if len(ok) else [np.nan] * 5
        rows.append({"feature": f, "nhóm": group_of(f), "% thiếu": round(100 - len(ok) / len(v) * 100, 1),
                     "p1": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p99": q[4]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. noise
def noise_case(args):
    root, record, tids = args
    p = eda.Paths(root)
    start, end = float(np.ceil(record["opstart"])), float(np.floor(record["opend"]))
    grid = np.arange(start, end, 2.0)
    out = []
    for name, (track, max_age, _, implausible) in NOISE_TRACKS.items():
        tid = tids.get(track)
        if not isinstance(tid, str) or not (p.raw / f"{tid}.csv.gz").exists():
            out.append({"caseid": record["caseid"], "signal": name, "present": False})
            continue
        t, v = read_numeric(fetch_csv(tid, p.raw))
        inside = (t >= start) & (t < end)
        t, v = t[inside], v[inside]
        finite = np.isfinite(v)
        row = {"caseid": record["caseid"], "signal": name, "present": True, "n": int(len(v)),
               "nonfinite": int((~finite).sum()), "implausible": int(implausible(v[finite]).sum())}
        key = next((k for k, trk in TRACKS.items() if trk == track), None)
        if key:   # outside the engineering bounds used by the pipeline (or 0 for BP/HR)
            lo, hi = BOUNDS[key]
            bad = (v[finite] < lo) | (v[finite] > hi)
            if key in ("map", "sbp", "dbp", "hr"):
                bad |= v[finite] == 0
            row["out_of_bounds"] = int(bad.sum())
        if len(t):
            valid = finite.copy()
            valid[finite] &= ~implausible(v[finite])
            _, age = causal_sample(t[valid], v[valid], grid, max_age) if valid.any() else (None, np.full(len(grid), np.inf))
            row["stale_frac"] = float(np.mean(age > max_age))
        else:
            row["stale_frac"] = 1.0
        if name in FLAT_SPIKE and finite.sum() > 10:
            vv, tt = v[finite], t[finite]
            same = np.diff(vv) == 0
            # time covered by runs of identical consecutive values lasting >= 60 s
            run_time, run_start = 0.0, None
            for i, s in enumerate(same):
                if s and run_start is None:
                    run_start = tt[i]
                if (not s or i == len(same) - 1) and run_start is not None:
                    stop = tt[i + 1] if s else tt[i]
                    run_time += (stop - run_start) if stop - run_start >= 60 else 0
                    run_start = None
            row["flat_frac"] = run_time / max(end - start, 1)
            row["spikes_per_h"] = float((np.abs(np.diff(vv)) > FLAT_SPIKE[name]).sum() / max((end - start) / 3600, 1e-9))
        out.append(row)
    return out


def noise_profile(root, manifest, workers=8, limit=None):
    records = manifest[manifest.eligible]
    index = pd.read_csv(eda.Paths(root).meta / "uc04_tracks.csv").drop_duplicates(["caseid", "tname"])
    tids = {c: g.set_index("tname").tid.to_dict() for c, g in index.groupby("caseid")}
    args = [(root, r, tids.get(r["caseid"], {})) for r in records.to_dict("records")[:limit]]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(noise_case, args, chunksize=16):
            rows += part
    per_case = pd.DataFrame(rows)
    summary = []
    n_cases = len(args)
    for name, g in per_case.groupby("signal", sort=False):
        have = g[g.present]
        n = have.n.sum()
        row = {"tín hiệu": name, "track": NOISE_TRACKS[name][0], "% ca có track": round(len(have) / n_cases * 100, 1),
               "% thời gian mổ thiếu/cũ (median ca)": round(have.stale_frac.median() * 100, 2) if len(have) else np.nan,
               "% ca thiếu >20% thời gian": round((have.stale_frac > .2).mean() * 100, 1) if len(have) else np.nan,
               "% mẫu không hữu hạn": round(have.nonfinite.sum() / max(n, 1) * 100, 3),
               "tiêu chí bất thường": NOISE_TRACKS[name][2],
               "% mẫu bất thường": round(have.implausible.sum() / max(n, 1) * 100, 3)}
        if "out_of_bounds" in have and have.out_of_bounds.notna().any():
            row["% mẫu ngoài BOUNDS/=0"] = round(have.out_of_bounds.sum() / max(n, 1) * 100, 3)
        if "flat_frac" in have and have.flat_frac.notna().any():
            row["% thời gian đứng yên ≥60 s"] = round(have.flat_frac.mean() * 100, 2)
            row["spike/giờ (mean)"] = round(have.spikes_per_h.mean(), 2)
        summary.append(row)
    return per_case, pd.DataFrame(summary)


# ---------------------------------------------------------------- 3. relations
KEY_FEATURES = [
    "map_current", "map_300_mean", "map_300_slope", "map_300_std", "map_600_min", "sbp_current", "dbp_current",
    "hr_current", "hr_300_slope", "spo2_current", "etco2_current", "etco2_300_slope", "rr_current",
    "bt_sv_lz_cur", "bt_sv_lz_d300", "bt_svr_lz_cur", "bt_svr_lz_d300", "bt_co_lz_d300", "bt_ppv_cur",
    "bt_ppv_d300", "bt_dpdt_cur", "bt_dpdt_d300", "bt_pp_cur", "bis_cur", "bis_d300", "ppf_ce_cur",
    "ppf_ce_d300", "rftn_ce_cur", "rftn_ce_d300", "mac_cur", "peep_cur", "cvp_cur",
    "static_age", "static_asa", "static_bmi", "preop_hb", "preop_htn",
]


def spearman(table, features=KEY_FEATURES, sample=150_000, seed=0):
    d = table[features].sample(min(sample, len(table)), random_state=seed)
    return d.rank().corr(min_periods=2000)


def future_association(table, target="dmap_future_300", sample=300_000, seed=0):
    d = table.sample(min(sample, len(table)), random_state=seed)
    rows = []
    for f in feature_columns(d):
        both = d[[f, target]].dropna()
        if len(both) >= 2000 and both[f].nunique() > 5:
            rows.append({"feature": f, "nhóm": group_of(f), "n": len(both),
                         "rho": both[f].rank().corr(both[target].rank())})
    return pd.DataFrame(rows).sort_values("rho", key=np.abs, ascending=False)


def risk_curve(table, feature, label="y_300", bins=10):
    d = table[table[label].ge(0)][[feature, label]].dropna()
    if len(d) < 2000:
        return pd.DataFrame()
    d["bin"] = pd.qcut(d[feature], bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(x=(feature, "median"), risk=(label, "mean"), n=(label, "size"))
    return g.reset_index(drop=True)


def level_trend_map(table, label="y_300"):
    d = table[table[label].ge(0)][["map_current", "map_300_slope", label]].dropna()
    d = d[(d.map_current.between(55, 110)) & (d.map_300_slope.between(-0.1, 0.1))]
    d["lv"] = pd.cut(d.map_current, np.arange(55, 111, 5))
    d["tr"] = pd.cut(d.map_300_slope * 60, np.linspace(-6, 6, 13))   # mmHg / min
    g = d.groupby(["tr", "lv"], observed=False)[label].agg(["mean", "size"])
    risk = g["mean"].where(g["size"] >= 200).unstack("lv")
    return risk


# ---------------------------------------------------------------- 4. influence
def univariate_auroc(table, labels=("y_300", "y_600")):
    from sklearn.metrics import roc_auc_score
    rows = []
    for f in feature_columns(table):
        row = {"feature": f, "nhóm": group_of(f)}
        for label in labels:
            d = table[table[label].ge(0)][[f, label]].dropna()
            if len(d) < 2000 or d[label].nunique() < 2 or d[f].nunique() < 2:
                continue
            auc = roc_auc_score(d[label], d[f])
            row[f"auroc_{label}"] = max(auc, 1 - auc)
            row[f"hướng_{label}"] = "cao → nguy cơ" if auc >= .5 else "thấp → nguy cơ"
            row[f"n_{label}"] = len(d)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("auroc_y_300", ascending=False)


def _fit(train, features, label, seed=20260917):
    import lightgbm as lgb
    from .e08_methods import LIGHTGBM_PARAMS
    d = train[train[label].ge(0)]
    model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS, random_state=seed, verbose=-1, n_jobs=8)
    model.fit(d[features], d[label].astype(int))
    return model


def _score(model, valid, features, label):
    from sklearn.metrics import average_precision_score, roc_auc_score
    d = valid[valid[label].ge(0)]
    p = model.predict_proba(d[features])[:, 1]
    return d, p, roc_auc_score(d[label], p), average_precision_score(d[label], p)


def shap_importance(table, label="y_300", sample=40_000, seed=0):
    features = feature_columns(table)
    fit, valid = table[table.role.eq("fit")], table[table.role.eq("validation")]
    model = _fit(fit, features, label)
    d, p, auc, ap = _score(model, valid, features, label)
    s = d.sample(min(sample, len(d)), random_state=seed)
    contrib = model.booster_.predict(s[features], pred_contrib=True)[:, :-1]
    imp = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(contrib).mean(axis=0)})
    imp["nhóm"] = imp.feature.map(group_of)
    imp["share_%"] = imp.mean_abs_shap / imp.mean_abs_shap.sum() * 100
    # direction: correlation between feature value and its SHAP contribution
    direction = []
    for j, f in enumerate(features):
        x = s[f].to_numpy(float)
        ok = np.isfinite(x)
        r = np.corrcoef(x[ok], contrib[ok, j])[0, 1] if ok.sum() > 100 and np.nanstd(x[ok]) > 0 else np.nan
        direction.append(r)
    imp["tương quan giá trị–SHAP"] = direction
    return imp.sort_values("mean_abs_shap", ascending=False), {"auroc": auc, "ap": ap, "n_valid": len(d)}


def group_ablation(table, label="y_300", bootstrap=200, seed=0):
    """AUROC/AP on VALIDATION for: MAP only, MAP + each group, all, all minus each group."""
    features = feature_columns(table)
    groups = {g: [f for f in features if group_of(f) == g] for g, _ in GROUPS}
    fit, valid = table[table.role.eq("fit")], table[table.role.eq("validation")]
    configs = {"Chỉ MAP": groups["MAP"], "Tất cả": features}
    for g, fs in groups.items():
        if g != "MAP" and fs:
            configs[f"MAP + {g}"] = groups["MAP"] + fs
            configs[f"Tất cả − {g}"] = [f for f in features if f not in fs]
    rows, preds = [], {}
    for name, fs in configs.items():
        model = _fit(fit, fs, label)
        d, p, auc, ap = _score(model, valid, fs, label)
        preds[name] = p
        rows.append({"cấu hình": name, "số feature": len(fs), "AUROC": auc, "AP": ap})
    out = pd.DataFrame(rows)
    # case-cluster bootstrap for "Tất cả" vs "Chỉ MAP"
    d = valid[valid[label].ge(0)]
    y, cases = d[label].to_numpy(), d.caseid.to_numpy()
    from sklearn.metrics import roc_auc_score, average_precision_score
    uniq = np.unique(cases)
    index = {c: np.flatnonzero(cases == c) for c in uniq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(bootstrap):
        pick = np.concatenate([index[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        if y[pick].min() == y[pick].max():
            continue
        diffs.append((roc_auc_score(y[pick], preds["Tất cả"][pick]) - roc_auc_score(y[pick], preds["Chỉ MAP"][pick]),
                      average_precision_score(y[pick], preds["Tất cả"][pick])
                      - average_precision_score(y[pick], preds["Chỉ MAP"][pick])))
    diffs = np.array(diffs)
    ci = {"dAUROC": np.percentile(diffs[:, 0], [2.5, 50, 97.5]).tolist(),
          "dAP": np.percentile(diffs[:, 1], [2.5, 50, 97.5]).tolist(), "n_cases": int(len(uniq))}
    return out, ci
