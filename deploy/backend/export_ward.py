"""Export the SafeAnes ward bundle: 50 test cases and the four features_v8 models -> deploy/backend/data_ward/.

    PYTHONPATH="src;.local_deps" python deploy/backend/export_ward.py --work <work dir of the *_v8 runs> \
        --test-dir data/samples_v3_test --vitaldb data/vitaldb_full --cases 50

Run from the repo root. Only reads and reshapes files; no model is run. Writes (git-ignored, patient data):
  ward.json                    beds: case info (age, sex, ASA, department, surgery type), incision, events
  cases/<caseid>.parquet       one row per 30 s: time, eligible, y_*, the model inputs (float32)
  wave/<caseid>.parquet        2 s vitals from the raw Solar8000 tracks: map, sbp, dbp, hr, spo2
  models/<model>/meta.json     inputs with Vietnamese labels and groups; per horizon: Platt, alarm threshold,
                               validation metrics and the descriptive test metrics
  models/lgbm_v8/h<h>.txt, models/catboost_v8/h<h>.cbm, models/logistic_v8/h<h>.json, models/tabm_v8/model.joblib
  reference/<model>.parquet    offline test probabilities of the exported cases (to check the live engine)
The test split of safeanes-v3 was already used to confirm locked models: results on it are descriptive only.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from uc04.ablation import FAMILY_VI, family
from uc04.feature_eda import describe
from safeanes.preprocess import CHANNELS
from uc04.loaders import load_v8

HERE = Path(__file__).resolve().parent
HORIZONS = (300, 600, 900, 1200, 1800)
W = 120
MODELS = {"lgbm_v8": "LightGBM", "catboost_v8": "CatBoost", "logistic_v8": "Hồi quy logistic", "tabm_v8": "TabM"}
TRACKS = {"map": "Solar8000/ART_MBP", "sbp": "Solar8000/ART_SBP", "dbp": "Solar8000/ART_DBP", "hr": "Solar8000/HR",
          "spo2": "Solar8000/PLETH_SPO2"}
WAVE_STEP, WAVE_MAX_AGE, WAVE_LEAD = 2.0, 10.0, 1800.0  # 2 s grid, last value at most 10 s old, 30 min before start
METRICS = ("auroc", "auroc_lo", "auroc_hi", "auprc", "event_sensitivity", "false_alarms_per_hour", "alarm_ppv",
           "lead_median_s", "events_eligible", "events_detected", "prevalence", "ece")
TEST_METRICS = ("auroc", "auprc", "event_sensitivity", "false_alarms_per_hour", "alarm_ppv", "lead_median_s",
                "events_eligible", "events_detected", "n_cases")

# labels of the features_v8 columns that uc04.feature_eda.describe does not know (f2_* of lgbm_full_v4)
F2_SIGNAL = {
    "pip": ("Áp lực đỉnh đường thở (PIP)", "cmH₂O"), "pplat": ("Áp lực bình nguyên (Pplat)", "cmH₂O"),
    "tv": ("Thể tích khí lưu thông", "mL"), "mv": ("Thông khí phút", "L/phút"),
    "compliance": ("Độ giãn nở phổi", "mL/cmH₂O"), "exp_sevo": ("Sevoflurane thở ra", "%"),
    "exp_des": ("Desflurane thở ra", "%"), "ppf_vol": ("Thể tích propofol đã bơm", "mL"),
    "ppf_cp": ("Nồng độ propofol huyết tương (TCI)", "µg/mL"), "rftn_rate": ("Tốc độ bơm remifentanil", "mL/h"),
    "bis_sr": ("BIS: tỷ lệ ức chế điện não", "%"), "bis_sqi": ("BIS: chất lượng tín hiệu", "%"),
    "bt": ("Nhiệt độ cơ thể", "°C"), "pleth_hr": ("Nhịp tim từ pleth", "lần/phút"),
    "nibp_dbp": ("Huyết áp tâm trương không xâm lấn", "mmHg"), "ev_svi": ("Chỉ số thể tích nhát bóp SVI (EV1000)", "mL/m²"),
    "fem_mbp_age": ("MAP động mạch đùi: tuổi của giá trị gần nhất", "s"),
    "lab_gluc": ("Glucose (khí máu)", "mg/dL"), "lab_hct": ("Hct (khí máu)", "%"), "lab_po2": ("pO₂ (khí máu)", "mmHg"),
    "lab_pco2": ("pCO₂ (khí máu)", "mmHg"), "lab_ica": ("Ca ion (khí máu)", "mmol/L"), "lab_k": ("K⁺ (khí máu)", "mmol/L"),
    "lab_min_since_sample": ("Số phút từ mẫu khí máu gần nhất", "phút"),
    "st_preop_aptt": ("aPTT trước mổ", "s"), "st_preop_alt": ("ALT trước mổ", "U/L"), "st_preop_ast": ("AST trước mổ", "U/L"),
    "st_preop_k": ("K⁺ trước mổ", "mmol/L"), "st_preop_cr": ("Creatinin trước mổ", "mg/dL"),
    "st_preop_alb": ("Albumin trước mổ", "g/dL"), "st_preop_hb": ("Hb trước mổ", "g/dL"),
    "st_weight": ("Cân nặng", "kg"), "st_height": ("Chiều cao", "cm"),
}
F2_STAT = {"mean": "trung bình", "sd": "độ lệch chuẩn"}


def label(col: str) -> tuple[str, str]:
    """(Vietnamese label, unit) of one input."""
    if col.startswith("f2_"):
        rest = col[3:]
        for sig in sorted(F2_SIGNAL, key=len, reverse=True):
            if rest == sig:
                return F2_SIGNAL[sig]
            if rest.startswith(sig + "_"):
                stat, _, win = rest[len(sig) + 1:].partition("_m")
                name, unit = F2_SIGNAL[sig]
                return f"{name}, {F2_STAT.get(stat, stat)} {win} phút" if win else f"{name}, {stat}", unit
        return col, ""
    _, meaning, unit = describe(col)
    return meaning, unit


# ------------------------------------------------------------------------------------------------ cases

def pick_cases(labels: pd.DataFrame, events: pd.DataFrame, ss: dict, n: int, seed: int, usable: set) -> list[int]:
    """A spread of cases: events before incision and during surgery, only before, only during, none (30/20/30/20 %),
    90-360 min long, with the raw vitals available."""
    ev = events.assign(pre=events.onset <= events.caseid.map(ss))
    per = ev.groupby("caseid").agg(n=("onset", "size"), pre=("pre", "sum"))
    dur = labels.groupby("caseid").time.agg(lambda t: (t.max() - t.min()) / 60)
    ok = [c for c in dur.index if 90 <= dur[c] <= 360 and c in usable and c in ss]
    per = per.reindex(ok, fill_value=0)
    rng = np.random.default_rng(seed)
    groups = [per[(per.pre > 0) & (per.n > per.pre)], per[(per.pre > 0) & (per.n == per.pre)],
              per[(per.pre == 0) & (per.n > 0)], per[per.n == 0]]
    k = [n * 3 // 10, n * 2 // 10, n * 3 // 10]
    k.append(n - sum(k))
    out = []
    for g, kk in zip(groups, k):
        ids = g.index.to_numpy()
        out += sorted(rng.choice(ids, min(kk, len(ids)), replace=False).tolist())
    return [int(c) for c in out]


def wave(vitaldb: Path, tracks: pd.DataFrame, caseid: int, start: float, end: float) -> pd.DataFrame:
    """2 s grid of the raw Solar8000 vitals: the last value at most WAVE_MAX_AGE s old, never interpolated. Values
    outside the plausibility range of the pipeline (safeanes.preprocess.CHANNELS, e.g. MAP 20-200 mmHg) are monitor
    artefacts (zeroing, flush) and dropped, as in the training data."""
    grid = np.arange(np.floor((start - WAVE_LEAD) / WAVE_STEP) * WAVE_STEP, end + WAVE_STEP, WAVE_STEP)
    out = {"time": grid}
    for key, tname in TRACKS.items():
        tid = tracks[(tracks.caseid == caseid) & (tracks.tname == tname)].tid
        v = np.full(len(grid), np.nan)
        if len(tid):
            d = pd.read_csv(vitaldb / "raw" / f"{tid.iloc[0]}.csv.gz")
            d = d.rename(columns={d.columns[0]: "t", d.columns[1]: "v"}).dropna().sort_values("t")
            ch = CHANNELS[key]
            d = d[(d.v >= ch.lo) & (d.v <= ch.hi)]
            t, x = d.t.to_numpy(float), d.v.to_numpy(float)
            i = np.searchsorted(t, grid, side="right") - 1
            okk = (i >= 0)
            okk[okk] &= grid[okk] - t[i[okk]] <= WAVE_MAX_AGE
            v[okk] = x[i[okk]]
        out[key] = v
    df = pd.DataFrame(out)
    return df.astype({c: "float32" for c in TRACKS})


# ------------------------------------------------------------------------------------------------ models

def model_dir(work: Path, model: str, h: int) -> Path:
    return work / "artifacts" / "tabular" / model / f"W{W}" / f"h{h}"


def export_model(work: Path, model: str, out: Path, cases: list[int]) -> dict:
    d = out / "models" / model
    d.mkdir(parents=True, exist_ok=True)
    meta = {"name": model, "title": MODELS[model], "window_s": W, "horizons": {}}
    refs = []
    for k, h in enumerate(HORIZONS):
        src = model_dir(work, model, h)
        b = joblib.load(src / "model.joblib")
        est = b["model"]
        if model == "lgbm_v8":
            est.model.booster_.save_model(str(d / f"h{h}.txt"))   # write_text would turn \n into \r\n on Windows
        elif model == "catboost_v8":
            est.model.save_model(str(d / f"h{h}.cbm"))
        elif model == "logistic_v8":                                # score = logit(LR proba) = coef . z + intercept
            (d / f"h{h}.json").write_text(json.dumps({
                "medians": [est.medians[c] for c in est.columns], "mean": est.scaler.mean_.tolist(),
                "scale": est.scaler.scale_.tolist(), "coef": est.lr.coef_[0].tolist(),
                "intercept": float(est.lr.intercept_[0])}), encoding="utf-8")
        elif k == 0:                                                # one TabM for all horizons (OneHorizon.k)
            shutil.copy2(src / "model.joblib", d / "model.joblib")
        meta["columns"] = list(b["columns"])
        val = pd.read_csv(work / "reports" / "tabular_validation" / f"{model}_W{W}_h{h}.csv").iloc[0]
        tp = work / "test_descriptive" / "reports" / f"{model}_W{W}_h{h}.csv"
        test = pd.read_csv(tp).iloc[0] if tp.exists() else None
        meta["horizons"][str(h)] = {
            "calibrator": {k2: float(v) for k2, v in b["calibrator"].items() if k2 != "method"},
            "threshold": json.loads((src / "threshold.json").read_text())["threshold"], "tabm_index": k,
            "validation": {m: (None if pd.isna(val[m]) else float(val[m])) for m in METRICS},
            "test": None if test is None else {m: (None if pd.isna(test[m]) else float(test[m])) for m in TEST_METRICS}}
        tpp = work / "test_descriptive" / "tabular" / model / f"W{W}" / f"h{h}" / "test_predictions.parquet"
        if tpp.exists():
            r = pd.read_parquet(tpp, columns=["caseid", "time", "probability"])
            refs.append(r[r.caseid.isin(cases)].rename(columns={"probability": f"p_{h}"}).set_index(["caseid", "time"]))
    meta["features"] = [{"column": c, "label": label(c)[0], "unit": label(c)[1], "group": family(c),
                         "group_label": FAMILY_VI[family(c)]} for c in meta["columns"]]
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if refs:
        (out / "reference").mkdir(exist_ok=True)
        pd.concat(refs, axis=1).reset_index().to_parquet(out / "reference" / f"{model}.parquet", index=False)
    print(f"model {model}: {len(meta['columns'])} inputs, {len(HORIZONS)} horizons, reference {'yes' if refs else 'no'}")
    return meta


# ------------------------------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", help="work dir with artifacts/tabular/<model>/W120/h<h>/ of the *_v8 runs "
                                   "(the TabM run may be a second dir: --work-tabm)")
    ap.add_argument("--work-tabm")
    ap.add_argument("--columns", default="configs/features_v8_71.json")
    ap.add_argument("--test-dir", default="data/samples_v3_test")
    ap.add_argument("--vitaldb", default="data/vitaldb_full")
    ap.add_argument("--cases", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out", default=str(HERE / "data_ward"))
    ap.add_argument("--skip-models", action="store_true", help="cases and vitals only")
    args = ap.parse_args()
    out, test, vdb = Path(args.out), Path(args.test_dir), Path(args.vitaldb)
    cols = json.loads(Path(args.columns).read_text(encoding="utf-8"))["columns"]

    lab, X = load_v8(test, "test", cols)
    events = pd.read_parquet(test / "events.parquet")
    ss = pd.read_csv(test / "surgery_start.csv").set_index("caseid")["surgery_start"].to_dict()
    tracks = pd.read_csv(vdb / "meta" / "trks.csv.gz")
    usable = set(tracks.loc[tracks.tname == TRACKS["map"], "caseid"])
    chosen = pick_cases(lab, events, ss, args.cases, args.seed, usable)
    man = pd.read_csv(vdb / "meta" / "cohort_manifest.csv").set_index("caseid")

    for sub in ("cases", "wave"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    keep = ["caseid", "time", "eligible", *[f"y_{h}" for h in HORIZONS]]
    beds = []
    for bed, c in enumerate(chosen, start=1):
        m = (lab.caseid == c).to_numpy()
        df = pd.concat([lab.loc[m, keep].reset_index(drop=True),
                        X.loc[m, cols].astype("float32").reset_index(drop=True)], axis=1)
        df.to_parquet(out / "cases" / f"{c}.parquet", index=False)
        start, end = float(df.time.min()), float(df.time.max())
        wave(vdb, tracks, c, start, end).to_parquet(out / "wave" / f"{c}.parquet", index=False)
        e = events[events.caseid == c].sort_values("onset")
        r = man.loc[c]
        beds.append({
            "bed": bed, "caseid": c, "age": None if pd.isna(r.age) else float(r.age), "sex": str(r.sex),
            "asa": None if pd.isna(r.asa) else int(r.asa), "department": str(r.department),
            "optype": str(r.optype), "emergency": bool(r.emop == 1) if not pd.isna(r.emop) else False,
            "start_s": start, "end_s": end, "surgery_start_s": float(ss[c]),
            "events": [{"onset_s": float(x.onset), "end_s": float(x.end), "min_map": float(x.min_map),
                        "pre_incision": bool(x.onset <= ss[c])} for x in e.itertuples()]})
        print(f"bed {bed:2d}: case {c}, {len(df)} rows, {(end - start) / 60:.0f} min, {len(e)} events")
    meta = {}
    if not args.skip_models:
        for mname in MODELS:
            work = Path(args.work_tabm if mname == "tabm_v8" and args.work_tabm else args.work)
            meta[mname] = export_model(work, mname, out, chosen)
    (out / "ward.json").write_text(json.dumps({
        "source": "safeanes-v3 test split (descriptive: already used to confirm locked models)",
        "horizons": list(HORIZONS), "window_s": W, "inputs": cols, "models": list(meta), "beds": beds},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(beds)} beds -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
