"""Does the patient's clinical profile help, and for whom? -> reports/weekly/week14/clinical/.

  $env:PYTHONPATH = "$PWD/src;$PWD/.local_deps"
  python scripts/clinical_encoding.py [--train-cases 700]

Two ways to use the pre-operative profile (age, ASA, hypertension, diabetes, emergency, BMI, sex, department,
approach, position, pre-operative labs), compared with the week-13 lgbm_context inputs (66 monitor + 31 context):
  A. as model inputs      LightGBM base vs base + clinical (same settings), h = 5 and 15 min
  B. as alarm policy      one threshold for everyone vs one threshold per risk profile (high risk = age >= 70 or
                          ASA >= 3 or hypertension or diabetes), each within <= 1 false alarm per hour
Thresholds and Platt scaling are chosen on the calibration split; validation only scores, overall and per subgroup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uc04 import feature_eda as fe
from uc04.calibration import Platt
from uc04.columns import CONTEXT_COLS, numeric_cols
from uc04.metrics import Evaluator, summarize_counts
from uc04.thresholds import candidates, choose_threshold

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data/samples_v3"
MANIFEST = ROOT / "data/vitaldb_full/meta/cohort_manifest.csv"
OUT = ROOT / "reports/weekly/week14/clinical"
SEED, W = 20260917, 120


def subgroups(caseids: pd.Series) -> dict[str, np.ndarray]:
    m = pd.read_csv(MANIFEST, usecols=["caseid", "age", "asa", "preop_htn", "preop_dm", "emop"]).set_index("caseid")
    m = m.reindex(caseids.to_numpy())
    high = (m["age"] >= 70) | (m["asa"] >= 3) | (m["preop_htn"] == 1) | (m["preop_dm"] == 1)
    return {"Tất cả": np.ones(len(m), bool), "≥ 70 tuổi": (m["age"] >= 70).to_numpy(),
            "< 70 tuổi": (m["age"] < 70).to_numpy(), "ASA ≥ 3": (m["asa"] >= 3).to_numpy(),
            "Tăng huyết áp": (m["preop_htn"] == 1).to_numpy(), "Đái tháo đường": (m["preop_dm"] == 1).to_numpy(),
            "Mổ cấp cứu": (m["emop"] == 1).to_numpy(), "Nguy cơ cao (bất kỳ)": high.to_numpy(),
            "Nguy cơ thấp": (~high).to_numpy()}


def evaluate(lab, events, risk, thr, h, rows) -> dict:
    sub = lab[rows]
    ev = Evaluator(sub, events, h)
    t = np.asarray(thr)[rows] if np.ndim(thr) else thr
    p = ev.sort(risk[rows])
    if np.ndim(t):
        # per-row threshold: compare p - t with 0
        res = ev.evaluate(p - ev.sort(t), 0.0)
    else:
        res = ev.evaluate(p, t)
    s = summarize_counts(res.case_table.sum(numeric_only=True))
    k = fe.known(sub, h) & sub["eligible"].to_numpy(bool)
    y = sub[f"y_{h}"].to_numpy()[k]
    return {"n_cases": int(sub["caseid"].nunique()), "events": s["events_eligible"],
            "auroc": fe.auroc(y, risk[rows][k]) if 0 < y.sum() < len(y) else np.nan,
            "event_sensitivity": s["event_sensitivity"], "fa_per_hour": s["false_alarms_per_hour"],
            "ppv": s["alarm_ppv"], "early_sensitivity": s["early_sensitivity"]}


def pick(lab, events, risk, h, rows) -> float:
    ev = Evaluator(lab[rows], events, h)
    p = ev.sort(risk[rows])
    elig = lab["eligible"].to_numpy(bool)[rows]
    return choose_threshold(candidates(risk[rows][elig]), lambda t: ev.quick(p, t), budget=1.0)["threshold"]


def main():
    global SAMPLES, MANIFEST, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cases", type=int, default=700)
    ap.add_argument("--samples", type=Path, default=SAMPLES)
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="VitalDB clinical table (cases API columns)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    SAMPLES, MANIFEST, OUT = args.samples, args.manifest, args.out
    OUT.mkdir(parents=True, exist_ok=True)

    groups = fe.candidate_columns(SAMPLES, W)
    base = numeric_cols(W) + list(CONTEXT_COLS)
    clin = groups["clinical"]
    sets = {"Monitor + ngữ cảnh": base, "+ lâm sàng trước mổ": base + clin}
    cases = fe.sample_cases(SAMPLES / "labels/train.parquet", args.train_cases, SEED)
    lab, X = fe.load_rows(SAMPLES, "train", base + clin, MANIFEST, cases)
    clab, CX = fe.load_rows(SAMPLES, "calibration", base + clin, MANIFEST, all_main=True)
    vlab, VX = fe.load_rows(SAMPLES, "validation", base + clin, MANIFEST, all_main=True)
    events = pd.read_parquet(SAMPLES / "events.parquet")
    vg, cg = subgroups(vlab["caseid"]), subgroups(clab["caseid"])
    rows, policy, imp = [], [], []
    for h in (300, 900):
        m = fe.known(lab, h)
        for name, cols in sets.items():
            b = fe.fit_lgbm(X.loc[m, cols], lab[f"y_{h}"].to_numpy()[m], W, SEED)
            cm = fe.known(clab, h) & clab["eligible"].to_numpy(bool)
            platt = Platt().fit(b.predict(CX.loc[cm, cols], raw_score=True), clab[f"y_{h}"].to_numpy()[cm])
            rc = np.where(clab["eligible"], platt.predict(b.predict(CX[cols], raw_score=True)), np.nan)
            rv = np.where(vlab["eligible"], platt.predict(b.predict(VX[cols], raw_score=True)), np.nan)
            thr = pick(clab, events, rc, h, cg["Tất cả"])
            for g, r in vg.items():
                rows.append({"horizon": h, "inputs": name, "subgroup": g, **evaluate(vlab, events, rv, thr, h, r)})
            if name.startswith("+"):
                sh = fe.shap_by_feature(b, VX.loc[fe.known(vlab, h) & vlab["eligible"].to_numpy(bool), cols]
                                        .sample(40000, random_state=SEED))
                imp.append(sh[clin].rename(f"shap_{h}") / sh.sum())
            # B. one threshold per risk profile (chosen on calibration), base inputs only
            if name == "Monitor + ngữ cảnh":
                t_hi = pick(clab, events, rc, h, cg["Nguy cơ cao (bất kỳ)"])
                t_lo = pick(clab, events, rc, h, cg["Nguy cơ thấp"])
                thr_rows = np.where(vg["Nguy cơ cao (bất kỳ)"], t_hi, t_lo)
                for g, r in vg.items():
                    policy.append({"horizon": h, "policy": "1 ngưỡng chung", "subgroup": g,
                                   "threshold": thr, **evaluate(vlab, events, rv, thr, h, r)})
                    policy.append({"horizon": h, "policy": "ngưỡng theo hồ sơ nguy cơ", "subgroup": g,
                                   "threshold": f"{t_hi:.3f} / {t_lo:.3f}",
                                   **evaluate(vlab, events, rv, thr_rows, h, r)})
            print(h, name, "done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "inputs_by_subgroup.csv", index=False, float_format="%.4f")
    pd.DataFrame(policy).to_csv(OUT / "threshold_policy.csv", index=False, float_format="%.4f")
    pd.concat(imp, axis=1).sort_values("shap_900", ascending=False).to_csv(OUT / "clinical_shap_share.csv",
                                                                          float_format="%.5f")
    (OUT / "summary.json").write_text(json.dumps({"train_cases": len(cases), "clinical_inputs": clin},
                                                 indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
