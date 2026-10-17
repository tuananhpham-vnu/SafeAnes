"""Personalisation mechanisms beyond "add the profile as columns" (week 14 plan, C1-C3) -> <out>/.

  python scripts/personalize.py --samples data/samples_v3 --manifest <clinical csv> --out <dir> [--train-cases N]

C1  personalised label: MAP < max(65, 0.8 x baseline MAP) for >= 60 s (uc04.rel_labels), checked by rebuilding the
    MAP < 65 labels with the same block procedure; a model trained on it; alarm policy "high-risk patients are
    warned against their own threshold".
C2  profile as conditioning: (i) offset model, logit = a * score_dynamics + g(profile) fitted on calibration;
    (ii) Mondrian: one threshold per risk profile (high = age >= 70 or ASA >= 3 or hypertension or diabetes).
C3  in-case personal physiology (uc04.hemo_forecast.personal_physiology) + distance to the personal threshold.
Every policy: thresholds on calibration (<= 1 false alarm per hour), scored on validation, overall and per subgroup,
patient-level bootstrap CIs and paired differences against the week-13 baseline (lgbm_context, one threshold).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uc04 import feature_eda as fe
from uc04 import hemo_forecast as hf
from uc04 import policy_eval as pe
from uc04 import rel_labels as rl
from uc04.calibration import Platt
from uc04.columns import CONTEXT_COLS, numeric_cols

ROOT = Path(__file__).resolve().parents[1]
SEED, W, HS = 20260917, 120, (300, 900)
W30 = ["map_min_w30", "map_max_w30", "map_missing_w30"]
PROFILE = ["f1b_age", "f1b_asa", "f1b_preop_htn", "st_preop_dm", "f1c_pre_emop", "f1c_pre_male", "f1c_pre_bmi"]


def subgroups(caseids: pd.Series, manifest: Path) -> dict[str, np.ndarray]:
    m = pd.read_csv(manifest, usecols=["caseid", "age", "asa", "preop_htn", "preop_dm", "emop"]).set_index("caseid")
    m = m.reindex(caseids.to_numpy())
    high = ((m["age"] >= 70) | (m["asa"] >= 3) | (m["preop_htn"] == 1) | (m["preop_dm"] == 1)).to_numpy()
    return {"Tất cả": np.ones(len(m), bool), "Nguy cơ cao": high, "Nguy cơ thấp": ~high,
            "≥ 70 tuổi": (m["age"] >= 70).to_numpy(), "ASA ≥ 3": (m["asa"] >= 3).to_numpy(),
            "Tăng huyết áp": (m["preop_htn"] == 1).to_numpy(), "Đái tháo đường": (m["preop_dm"] == 1).to_numpy(),
            "Mổ cấp cứu": (m["emop"] == 1).to_numpy()}


def load(samples, split, cols, manifest, cases=None):
    lab, f = fe.load_rows(samples, split, cols, manifest, cases, all_main=True)
    thr = rl.personal_threshold(f)
    plab, pev = rl.build(f, thr, HS)
    for h in HS:
        pev[f"eligible_{h}"] = rl.event_eligibility(pev, f, plab, h)
    pers = hf.personal_physiology(f)
    f = pd.concat([f, pers], axis=1)
    f["pers_thr"] = thr.astype("float32")
    f["pers_margin"] = (f["map_current"] - thr).astype("float32")
    pl = pe.labels_frame(lab, {h: plab[f"y_{h}"].to_numpy() for h in HS}, plab["eligible"].to_numpy())
    return lab, f, pl, pev


def fit_cal(X, y, Xc, yc, cols):
    m = np.isin(y, (0, 1))
    b = fe.fit_lgbm(X.loc[m, cols], y[m], W, SEED)
    mc_ = np.isin(yc, (0, 1)) & np.isfinite(Xc["map_current"].to_numpy())
    pl = Platt().fit(b.predict(Xc.loc[mc_, cols], raw_score=True), yc[mc_])
    return b, pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=ROOT / "data/samples_v3")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/vitaldb_full/meta/cohort_manifest.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/weekly/week14/personalize")
    ap.add_argument("--train-cases", type=int, default=100000)
    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    groups = fe.candidate_columns(args.samples, W)
    base = numeric_cols(W) + list(CONTEXT_COLS)
    cols = list(dict.fromkeys(base + W30 + groups["clinical"]))
    pers = ["pers_baro", "pers_case_min", "pers_map_sd", "pers_margin", "pers_thr"]
    cases = fe.sample_cases(args.samples / "labels/train.parquet", args.train_cases, SEED)

    tl, tf, tpl, _ = load(args.samples, "train", cols, args.manifest, cases)
    cl, cf, cpl, cev = load(args.samples, "calibration", cols, args.manifest)
    vl, vf, vpl, vev = load(args.samples, "validation", cols, args.manifest)
    events = pd.read_parquet(args.samples / "events.parquet")
    print("loaded", len(tl), len(cl), len(vl), flush=True)

    # C1 checks and descriptives
    chk = pd.concat([rl.check_against_abs(cf, cl, HS).assign(split="calibration"),
                     rl.check_against_abs(vf, vl, HS).assign(split="validation")])
    chk.to_csv(out / "label_check.csv", index=False, float_format="%.4f")
    vg, cg = subgroups(vl["caseid"], args.manifest), subgroups(cl["caseid"], args.manifest)
    case_thr = vf.groupby("caseid")["pers_thr"].median()
    hrs = vl.groupby("caseid")["exposure_seconds"].sum() / 3600
    n_abs = events[events["caseid"].isin(vl["caseid"])].groupby("caseid").size().reindex(hrs.index).fillna(0)
    n_pers = vev.groupby("caseid").size().reindex(hrs.index).fillna(0)
    stats = []
    for g, mrow in vg.items():
        ids = vl.loc[mrow, "caseid"].unique()
        stats.append({"subgroup": g, "cases": len(ids), "thr_above_65": float((case_thr.reindex(ids) > 65).mean()),
                      "thr_median": float(case_thr.reindex(ids).median()),
                      "events_per_h_abs": float(n_abs.reindex(ids).sum() / hrs.reindex(ids).sum()),
                      "events_per_h_pers": float(n_pers.reindex(ids).sum() / hrs.reindex(ids).sum())})
    pd.DataFrame(stats).to_csv(out / "label_stats.csv", index=False, float_format="%.4f")
    print(chk.to_string(), "\n", pd.DataFrame(stats).to_string(), flush=True)

    rows, pairs = [], []

    def record(policy, truth, h, lab, ev, p, thr, base_draws=None):
        nonlocal rows, pairs
        draws_by = {}
        for g, mrow in vg.items():
            r, d = pe.evaluate(lab, ev, p, thr, h, mrow, args.bootstrap, SEED, return_draws=True)
            rows.append({"policy": policy, "truth": truth, "horizon": h, "subgroup": g, **r})
            draws_by[g] = d
            if base_draws is not None and g in base_draws:
                pairs.append({"policy": policy, "truth": truth, "horizon": h, "subgroup": g,
                              **pe.paired(d, base_draws[g])})
        return draws_by

    for h in HS:
        y_t, y_c = tl[f"y_{h}"].to_numpy(), cl[f"y_{h}"].to_numpy()
        tm, cm = tl["eligible"].to_numpy(bool), cl["eligible"].to_numpy(bool)
        # A: week-13 inputs, absolute label (baseline)
        bA, pA = fit_cal(tf[tm], y_t[tm], cf[cm], y_c[cm], base)
        # B: + personal physiology (C3)
        bB, pB = fit_cal(tf[tm], y_t[tm], cf[cm], y_c[cm], base + pers)
        # C: personalised label model (C1)
        ty, cy = tpl[f"y_{h}"].to_numpy(), cpl[f"y_{h}"].to_numpy()
        tpm, cpm = tpl["eligible"].to_numpy(bool), cpl["eligible"].to_numpy(bool)
        bC, pC = fit_cal(tf[tpm], ty[tpm], cf[cpm], cy[cpm], base + pers)

        def prob(b, pl, f, elig):
            return np.where(elig, pl.predict(b.predict(f[b.feature_name()], raw_score=True)), np.nan)

        vA, cA = prob(bA, pA, vf, vl["eligible"]), prob(bA, pA, cf, cl["eligible"])
        vB, cB = prob(bB, pB, vf, vl["eligible"]), prob(bB, pB, cf, cl["eligible"])
        vC, cC = prob(bC, pC, vf, vpl["eligible"]), prob(bC, pC, cf, cpl["eligible"])
        # E: offset model (C2 i): logistic on [score A, profile] fitted on calibration
        from sklearn.linear_model import LogisticRegression
        from scipy.special import logit

        def zA(f):
            return np.c_[logit(np.clip(np.where(np.isfinite(f[0]), f[0], 0.5), 1e-6, 1 - 1e-6)),
                         f[1][PROFILE].fillna(f[1][PROFILE].median()).to_numpy()]
        ok = cm & np.isin(y_c, (0, 1)) & np.isfinite(cA)
        mu, sd = zA((cA, cf))[ok].mean(0), zA((cA, cf))[ok].std(0) + 1e-9
        lr = LogisticRegression(C=1.0, max_iter=2000).fit((zA((cA, cf))[ok] - mu) / sd, y_c[ok])
        vE = np.where(vl["eligible"], lr.predict_proba((zA((vA, vf)) - mu) / sd)[:, 1], np.nan)
        cE = np.where(cl["eligible"], lr.predict_proba((zA((cA, cf)) - mu) / sd)[:, 1], np.nan)

        # thresholds on calibration
        tA, tB, tE = (pe.pick(cl, events, c, h) for c in (cA, cB, cE))
        hi_c, lo_c = cg["Nguy cơ cao"], cg["Nguy cơ thấp"]
        hi_v = vg["Nguy cơ cao"]
        mondA = np.where(hi_v, pe.pick(cl, events, cA, h, hi_c), pe.pick(cl, events, cA, h, lo_c))
        mondB = np.where(hi_v, pe.pick(cl, events, cB, h, hi_c), pe.pick(cl, events, cB, h, lo_c))

        # truth 1: absolute label
        base_abs = record("A tuần 13 (1 ngưỡng)", "MAP<65", h, vl, events, vA, tA)
        record("B + sinh lý riêng (C3)", "MAP<65", h, vl, events, vB, tB, base_abs)
        record("E offset hồ sơ (C2i)", "MAP<65", h, vl, events, vE, tE, base_abs)
        record("A + ngưỡng theo hồ sơ (C2ii)", "MAP<65", h, vl, events, vA, mondA, base_abs)
        record("B + ngưỡng theo hồ sơ (C2ii+C3)", "MAP<65", h, vl, events, vB, mondB, base_abs)

        # truth 2: personal (high-risk -> own threshold, others -> MAP < 65)
        mix_lab = pe.labels_frame(vl, {h: np.where(hi_v, vpl[f"y_{h}"], vl[f"y_{h}"])},
                                  np.where(hi_v, vpl["eligible"], vl["eligible"]))
        hi_cases = set(vl.loc[hi_v, "caseid"])
        mix_ev = pd.concat([vev[vev["caseid"].isin(hi_cases)],
                            events[events["caseid"].isin(set(vl["caseid"]) - hi_cases)]], ignore_index=True)
        cmix_lab = pe.labels_frame(cl, {h: np.where(hi_c, cpl[f"y_{h}"], cl[f"y_{h}"])},
                                   np.where(hi_c, cpl["eligible"], cl["eligible"]))
        chi = set(cl.loc[hi_c, "caseid"])
        cmix_ev = pd.concat([cev[cev["caseid"].isin(chi)], events[events["caseid"].isin(set(cl["caseid"]) - chi)]],
                            ignore_index=True)
        vMix = np.where(hi_v, vC, vB)
        cMix = np.where(hi_c, cC, cB)
        mixthr = np.where(hi_v, pe.pick(cmix_lab, cmix_ev, cMix, h, hi_c), pe.pick(cmix_lab, cmix_ev, cMix, h, lo_c))
        base_mix = record("A tuần 13 (1 ngưỡng)", "cá nhân", h, mix_lab, mix_ev, vA,
                          pe.pick(cmix_lab, cmix_ev, cA, h))
        record("Cá nhân: nhãn riêng + ngưỡng theo hồ sơ (C1+C2ii+C3)", "cá nhân", h, mix_lab, mix_ev, vMix, mixthr,
               base_mix)
        print("h", h, "done", flush=True)

    pd.DataFrame(rows).to_csv(out / "policies.csv", index=False, float_format="%.4f")
    pd.DataFrame(pairs).to_csv(out / "paired_vs_baseline.csv", index=False, float_format="%.4f")
    (out / "summary.json").write_text(json.dumps({"train_cases": len(cases), "bootstrap": args.bootstrap,
                                                  "personal_features": pers, "profile": PROFILE},
                                                 indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
