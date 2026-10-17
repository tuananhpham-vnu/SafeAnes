"""Where is each haemodynamic component heading, and why (week 14 plan, A1-A3) -> <out>/.

  python scripts/hemo_forecast.py --samples data/samples_v3 --manifest <clinical csv> --out <dir> [--train-cases N]

A1  LightGBM (Huber) per component (HR, SV_LZ, SVR_LZ, MAP) and horizon (5 / 10 / 15 min) forecasts dlog X(t, h).
A2  Drivers: TreeSHAP of each component model summed by input group (uc04.hemo_forecast.DRIVER_GROUPS).
A3  Counterfactual: forecast with the recent anaesthetic / ventilation changes set to "no change".
Outputs
  accuracy.csv          validation R2, Spearman, direction accuracy per component x horizon; MAP sum vs direct model
  fall_link.csv         does the forecast MAP fall rank future hypotension? (AUROC of -dlog MAP forecast for y_h)
  chain_pre_event.csv   rows 0-15 min before an event vs controls: expected dominant component x its top driver
  drivers_by_group.csv  mean signed group SHAP per component before events vs controls
  counterfactual.csv    forecast SVR / MAP change with vs without the recent anaesthetic deepening
  chain_h900.parquet    per validation row: forecasts, top drivers, counterfactual (for the demo and the LLM card)
  models/<comp>_<h>.txt LightGBM boosters (for deploy/)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from uc04 import feature_eda as fe
from uc04 import hemo_forecast as hf
from uc04 import mechanism as mc
from uc04.columns import CONTEXT_COLS, numeric_cols

ROOT = Path(__file__).resolve().parents[1]
SEED, W = 20260917, 120
THREADS = os.cpu_count() or 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=ROOT / "data/samples_v3")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/vitaldb_full/meta/cohort_manifest.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/weekly/week14/hemo")
    ap.add_argument("--train-cases", type=int, default=100000)
    ap.add_argument("--shap-rows", type=int, default=20000)
    args = ap.parse_args()
    out = args.out
    (out / "models").mkdir(parents=True, exist_ok=True)

    groups = fe.candidate_columns(args.samples, W)
    inputs = list(dict.fromkeys(numeric_cols(W) + list(CONTEXT_COLS) + groups["anaes"] + groups["clinical"]))
    cases = fe.sample_cases(args.samples / "labels/train.parquet", args.train_cases, SEED)

    def load(split, cs=None):
        lab, f = fe.load_rows(args.samples, split, inputs, args.manifest, cs, all_main=True)
        f = pd.concat([f, hf.personal_physiology(f), hf.targets(f)], axis=1)
        return lab, f

    tl, tf = load("train", cases)
    vl, vf = load("validation")
    X_cols = inputs + ["pers_baro", "pers_case_min", "pers_map_sd"]
    te, ve = tl["eligible"].to_numpy(bool), vl["eligible"].to_numpy(bool)
    print("loaded", len(tl), len(vl), flush=True)

    models, pred = {}, {}
    acc = []
    for h in hf.HORIZONS:
        for c in hf.COMPONENTS:
            y = tf[f"d_{c}_{h}"].to_numpy()
            b = hf.fit(tf.loc[te, X_cols], y[te], SEED, THREADS)
            b.save_model(str(out / "models" / f"{c}_{h}.txt"))
            models[c, h] = b
            pred[c, h] = b.predict(vf[X_cols])
            acc.append({"component": c, "horizon": h, **hf.regression_metrics(vf[f"d_{c}_{h}"].to_numpy()[ve],
                                                                              pred[c, h][ve])})
            print("fit", c, h, acc[-1], flush=True)
        s = pred["hr", h] + pred["sv", h] + pred["svr", h]
        ok = ve & np.isfinite(vf[f"d_map_{h}"].to_numpy())
        acc.append({"component": "sum(hr+sv+svr) vs map model", "horizon": h, "n": int(ok.sum()),
                    "r2": np.nan, "spearman": float(pd.Series(s[ok]).corr(pd.Series(pred["map", h][ok]),
                                                                          method="spearman"))})
    pd.DataFrame(acc).to_csv(out / "accuracy.csv", index=False, float_format="%.4f")

    # does the forecast MAP fall rank future hypotension? (same rows / labels as the risk models)
    link = []
    for h in hf.HORIZONS:
        y = vl[f"y_{h}"].to_numpy()
        m = ve & np.isin(y, (0, 1))
        link.append({"horizon": h, "auroc_minus_dlogmap_forecast": fe.auroc((y[m] == 1).astype(int), -pred["map", h][m]),
                     "auroc_map_current": fe.auroc((y[m] == 1).astype(int), -vf["map_current"].to_numpy()[m])})
    pd.DataFrame(link).to_csv(out / "fall_link.csv", index=False, float_format="%.4f")

    # A2 drivers before events (y_900 == 1) vs controls (y_900 == 0), h = 15 min
    h = 900
    y = vl[f"y_{h}"].to_numpy()
    rng = np.random.default_rng(SEED)
    pre = np.flatnonzero(ve & (y == 1))
    ctl = np.flatnonzero(ve & (y == 0))
    pre = rng.choice(pre, min(args.shap_rows, len(pre)), replace=False)
    ctl = rng.choice(ctl, min(args.shap_rows, len(ctl)), replace=False)
    drv = []
    gs = {}
    for c in ("hr", "sv", "svr", "map"):
        for name, idx in (("trước tụt (y15=1)", pre), ("đối chứng (y15=0)", ctl)):
            g = hf.group_shap(models[c, h], vf.loc[idx, X_cols])
            gs[c, name] = g
            row = g.drop(columns="base").mean()
            drv.append({"component": c, "rows": name, "forecast_mean": float(pred[c, h][idx].mean()),
                        **row.to_dict()})
    pd.DataFrame(drv).to_csv(out / "drivers_by_group.csv", index=False, float_format="%.5f")

    # chain: expected dominant component (most negative forecast) and its most negative driver group
    chain = []
    for name, idx in (("trước tụt (y15=1)", pre), ("đối chứng (y15=0)", ctl)):
        comp = np.array(["hr", "sv", "svr"])[np.argmin(np.c_[pred["hr", h][idx], pred["sv", h][idx],
                                                             pred["svr", h][idx]], axis=1)]
        top = []
        for k, c in enumerate(comp):
            g = gs[c, name].drop(columns="base").iloc[k]
            top.append(g.idxmin())
        chain.append(pd.DataFrame({"rows": name, "component": comp, "driver": top}))
    chain = pd.concat(chain)
    tab = chain.groupby(["rows", "component", "driver"]).size().rename("n").reset_index()
    tab["share"] = tab["n"] / tab.groupby("rows")["n"].transform("sum")
    tab.sort_values(["rows", "share"], ascending=[True, False]).to_csv(out / "chain_pre_event.csv", index=False,
                                                                      float_format="%.4f")

    # A3 counterfactual on rows with recent anaesthetic deepening
    flags = mc.clinical_flags(vf)
    deep = ve & flags["anaesthetic_deepening"].to_numpy()
    cf_rows = []
    for c in ("svr", "map"):
        for name, m in (("trước tụt, mê sâu thêm", deep & (y == 1)), ("đối chứng, mê sâu thêm", deep & (y == 0))):
            idx = np.flatnonzero(m)
            if not len(idx):
                continue
            idx = rng.choice(idx, min(args.shap_rows, len(idx)), replace=False)
            f_obs = pred[c, h][idx]
            f_cf = hf.counterfactual(models[c, h], vf.loc[idx, X_cols])
            cf_rows.append({"component": c, "rows": name, "n": len(idx),
                            "forecast_pct": float(np.median(np.expm1(f_obs)) * 100),
                            "forecast_no_change_pct": float(np.median(np.expm1(f_cf)) * 100),
                            "median_diff_pct_points": float(np.median(np.expm1(f_obs) - np.expm1(f_cf)) * 100)})
    pd.DataFrame(cf_rows).to_csv(out / "counterfactual.csv", index=False, float_format="%.3f")

    # per-row chain for the demo / LLM card (validation eligible rows, h = 15 min)
    idx = np.flatnonzero(ve)
    rec = {"caseid": vf["caseid"].to_numpy()[idx], "time": vf["time"].to_numpy()[idx]}
    for c in hf.COMPONENTS:
        rec[f"f_{c}"] = pred[c, h][idx].astype("float32")
    rec["f_map_no_change"] = hf.counterfactual(models["map", h], vf.loc[idx, X_cols]).astype("float32")
    dom = np.array(["hr", "sv", "svr"])[np.argmin(np.c_[rec["f_hr"], rec["f_sv"], rec["f_svr"]], axis=1)]
    rec["dominant"] = dom
    for c in ("hr", "sv", "svr"):
        g = hf.group_shap(models[c, h], vf.loc[idx, X_cols]).drop(columns="base")
        rec[f"{c}_driver1"] = g.idxmin(axis=1).to_numpy()
        rec[f"{c}_driver1_val"] = g.min(axis=1).to_numpy().astype("float32")
    pd.DataFrame(rec).to_parquet(out / "chain_h900.parquet", index=False)
    (out / "summary.json").write_text(json.dumps({"train_cases": len(cases), "inputs": len(X_cols),
                                                  "driver_groups": list(hf.DRIVER_GROUPS)}, indent=2,
                                                 ensure_ascii=False))
    print("done", flush=True)


if __name__ == "__main__":
    main()
