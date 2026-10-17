"""Check the component forecasts against device measurements that are never model inputs (week 14 plan, A5).

  python scripts/hemo_validate.py --samples data/samples_v3 --manifest <clinical csv> --models <dir with
         <comp>_<h>.txt> --device data/ev1000_grid/ev1000_grid.parquet --out <dir>

For calibration + validation cases with EV1000 or Vigileo (held out from the forecast models), at each eligible row:
  forecast   dlog SV / SVR / CO (= HR + SV) from t to t + h (hemo_forecast models)
  device     log X_dev(t + h) - log X_dev(t)
  proxy      the same change of the LZ proxy (what the forecast was trained to predict)
Reported per device, component and horizon, on rows where the device moved >= 5 % (smaller changes are within
device noise): direction agreement and Spearman rho of forecast vs device, the same for the observed proxy change
(the ceiling set by the proxy), and for "persistence" (the last 10-min change continues) as a naive baseline.
Case-level bootstrap CIs (cases drawn with replacement).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from uc04 import feature_eda as fe
from uc04 import hemo_forecast as hf
from uc04 import mechanism as mc
from uc04.columns import CONTEXT_COLS, numeric_cols

ROOT = Path(__file__).resolve().parents[1]
SEED, W = 20260917, 120
DEV = {"sv": "dev_sv", "svr": "dev_svr", "co": "dev_co"}
MOVE = np.log(1.05)


def agree(f, d):
    ok = np.isfinite(f) & np.isfinite(d) & (np.abs(d) >= MOVE)
    if ok.sum() < 30:
        return np.nan, np.nan, int(ok.sum())
    return (float((np.sign(f[ok]) == np.sign(d[ok])).mean()),
            float(pd.Series(f[ok]).corr(pd.Series(d[ok]), method="spearman")), int(ok.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=ROOT / "data/samples_v3")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/vitaldb_full/meta/cohort_manifest.csv")
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--device", type=Path, default=ROOT / "data/ev1000_grid/ev1000_grid.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/weekly/week14/hemo_validate")
    ap.add_argument("--train-cases", type=int, default=0, help="unused (kernel interface)")
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import lightgbm as lgb
    if not (args.models / "svr_900.txt").exists():  # Kaggle: the forecast kernel's output mounted as an input
        hits = sorted(Path("/kaggle/input").rglob("svr_900.txt"))
        if not hits:
            raise FileNotFoundError(f"no forecast models in {args.models} or /kaggle/input")
        args.models = hits[0].parent

    groups = fe.candidate_columns(args.samples, W)
    inputs = list(dict.fromkeys(numeric_cols(W) + list(CONTEXT_COLS) + groups["anaes"] + groups["clinical"]))
    dev = pd.read_parquet(args.device)
    rows = []
    for split in ("calibration", "validation"):
        lab, f = fe.load_rows(args.samples, split, inputs, args.manifest, set(dev["caseid"]), all_main=True)
        if not len(f):
            continue
        f = pd.concat([f, hf.personal_physiology(f), hf.targets(f)], axis=1)
        f = f.merge(dev, on=["caseid", "time"], how="left")
        f["eligible"] = lab["eligible"].to_numpy(bool)
        f["split"] = split
        rows.append(f)
    f = pd.concat(rows, ignore_index=True)
    X_cols = inputs + ["pers_baro", "pers_case_min", "pers_map_sd"]
    el = f["eligible"].to_numpy(bool)
    out, per_case = [], []
    for h in hf.HORIZONS:
        fc = {c: lgb.Booster(model_file=str(args.models / f"{c}_{h}.txt")).predict(f[X_cols])
              for c in ("hr", "sv", "svr")}
        fc["co"] = fc["hr"] + fc["sv"]
        obs = {c: f[f"d_{c}_{h}"].to_numpy() for c in ("hr", "sv", "svr")}
        obs["co"] = obs["hr"] + obs["sv"]
        # persistence: the change over the last 10 min, scaled to h
        past = mc.decompose(mc.add_lagged(f, ["map_mean_w120", "sbp_mean_w120", "dbp_mean_w120", "hr_mean_w120"]))
        pers = {"sv": past["dlog_sv"].to_numpy(), "svr": past["dlog_svr"].to_numpy(),
                "hr": past["dlog_hr"].to_numpy()}
        pers["co"] = pers["hr"] + pers["sv"]
        fut = mc.add_lagged(f[["caseid", "time", *DEV.values()]], list(DEV.values()), lag=-h)
        for c, dcol in DEV.items():
            with np.errstate(all="ignore"):
                d = np.log(fut[f"{dcol}_lag"].to_numpy()) - np.log(f[dcol].to_numpy())
            for device in ("EV1000", "Vigileo"):
                m = el & (f["device"].to_numpy() == device)
                if not np.isfinite(d[m]).any():
                    continue
                rec = {"horizon": h, "component": c, "device": device,
                       "cases": int(f.loc[m & np.isfinite(d), "caseid"].nunique())}
                for name, x in (("forecast", fc[c]), ("proxy_observed", obs[c]), ("persistence", pers[c] * h / 600)):
                    a, r, n = agree(x[m], d[m])
                    rec[f"{name}_dir"], rec[f"{name}_rho"], rec["n_moved"] = a, r, n
                # case bootstrap of the forecast direction agreement
                ids = f.loc[m, "caseid"].to_numpy()
                ok = m & np.isfinite(fc[c]) & np.isfinite(d) & (np.abs(d) >= MOVE)
                hit = pd.DataFrame({"caseid": f.loc[ok, "caseid"].to_numpy(),
                                    "hit": np.sign(fc[c][ok]) == np.sign(d[ok])}).groupby("caseid")["hit"].agg(
                    ["sum", "size"])
                if len(hit) >= 5:
                    rng = np.random.default_rng(SEED)
                    bs = [hit.iloc[rng.integers(0, len(hit), len(hit))].sum() for _ in range(args.bootstrap)]
                    v = np.array([b["sum"] / b["size"] for b in bs])
                    rec["forecast_dir_lo"], rec["forecast_dir_hi"] = np.quantile(v, [0.025, 0.975])
                out.append(rec)
                del ids
    res = pd.DataFrame(out)
    res.to_csv(args.out / "device_agreement.csv", index=False, float_format="%.4f")
    print(res.to_string(), flush=True)


if __name__ == "__main__":
    main()
