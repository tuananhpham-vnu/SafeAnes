"""Causal-chain explanation for the demo (week 14 plan B): driver -> haemodynamic component -> MAP -> risk.

At a time point it reports
  past       how MAP changed over the last 10 min, split into HR, SV and SVR (Liljestrand-Zander proxies)
  forecast   where HR, SV, SVR and MAP are heading over the next 15 min (uc04.hemo_forecast models)
  drivers    for each component, the input groups that push its forecast up or down (TreeSHAP, % points)
  no_change  the MAP forecast if the recent anaesthetic / ventilation changes had not happened (model-based, not a
             causal effect)
  findings   clinical findings at t (uc04.mechanism.clinical_flags) and the patient's profile
  personal   in-case personal physiology (HR response to MAP, lowest MAP so far)
Models come from the hemo_forecast kernel (data/chain/<comp>_<h>.txt + meta.json, written by export_bundle.py).
Inputs the demo does not have (drug pumps, ventilator, labs for an uploaded CSV) are missing values; LightGBM handles
them, and the response says which groups were missing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(REPO_SRC) not in sys.path:  # as extract.py: the demo runs inside the repo and reuses the pipeline code
    sys.path.insert(0, str(REPO_SRC))

from uc04 import hemo_forecast as hf  # noqa: E402
from uc04 import mechanism as mc  # noqa: E402

H = 900
LAGGED = ["map_mean_w120", "sbp_mean_w120", "dbp_mean_w120", "hr_mean_w120"]
COMP_LABEL = {"hr": "Nhịp tim (HR)", "sv": "Thể tích nhát bóp (SV)", "svr": "Kháng lực mạch (SVR)", "map": "MAP"}


def pct(v) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(np.expm1(v) * 100), 1)


class Chain:
    def __init__(self, root: Path):
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        self.columns: list[str] = meta["columns"]
        self.models = {c: lgb.Booster(model_file=str(root / f"{c}_{H}.txt")) for c in hf.COMPONENTS}
        self.groups = [hf.driver_group(c) for c in self.columns]

    def prepare(self, frame: pd.DataFrame, times=None) -> pd.DataFrame:
        """Whole-case frame (sorted by time) with personal physiology and the 10-min lag columns added."""
        f = frame.reset_index(drop=True).copy()
        if "caseid" not in f:
            f["caseid"] = 0
        if "time" not in f:
            f["time"] = np.asarray(times, float)
        f = pd.concat([f, hf.personal_physiology(f)], axis=1)
        return mc.add_lagged(f, LAGGED)

    def explain(self, f: pd.DataFrame, i: int) -> dict:
        row = f.iloc[[i]]
        X = row.reindex(columns=self.columns).astype(float)
        fc = {c: float(m.predict(X)[0]) for c, m in self.models.items()}
        drivers = {}
        for c in ("hr", "sv", "svr"):
            contrib = self.models[c].predict(X, pred_contrib=True)[0][:-1]
            g = pd.Series(contrib).groupby(self.groups).sum()
            g = g[g.abs() > 1e-4].sort_values(key=np.abs, ascending=False).head(4)
            drivers[c] = [{"group": k, "pp": round(float(v * 100), 2)} for k, v in g.items()]
        Xc = X.copy()
        changed = [c for c in hf.INTERVENTION_DELTAS if c in Xc and np.isfinite(Xc[c].iat[0]) and Xc[c].iat[0] != 0]
        no_change = None
        if changed:
            no_change = {c: pct(float(hf.counterfactual(self.models[c], X)[0])) for c in ("svr", "map")}
        past = mc.decompose(row).iloc[0]
        flags = mc.clinical_flags(row).iloc[0]
        missing = sorted({hf.driver_group(c) for c in self.columns if c not in f or f[c].isna().all()})
        dom = min(("hr", "sv", "svr"), key=lambda c: fc[c])
        pers = row[["pers_baro", "pers_case_min", "pers_map_sd"]].iloc[0]
        return {
            "horizon_s": H,
            "past": {c: pct(past[f"dlog_{c}"]) for c in ("map", "hr", "sv", "svr")},
            "past_pattern": mc.MECHANISMS[mc.primary_mechanism(past.to_frame().T).iat[0]],
            "forecast": {c: pct(v) for c, v in fc.items()},
            "dominant": dom if fc[dom] < 0 else None,
            "labels": COMP_LABEL,
            "drivers": drivers,
            "no_change": no_change,
            "changed_inputs": changed,
            "findings": [t for k, t in mc.FLAG_TEXT.items() if bool(flags.get(k, False))],
            "profile": [t for k, t in mc.PROFILE_TEXT.items() if bool(flags.get(k, False))],
            "personal": {"hr_per_map": None if not np.isfinite(pers["pers_baro"]) else round(float(pers["pers_baro"]), 2),
                         "lowest_map": None if not np.isfinite(pers["pers_case_min"]) else round(float(pers["pers_case_min"])),
                         "map_sd_30min": None if not np.isfinite(pers["pers_map_sd"]) else round(float(pers["pers_map_sd"]), 1)},
            "missing_groups": missing,
        }


def load(data: Path) -> Chain | None:
    root = data / "chain"
    return Chain(root) if (root / "meta.json").exists() else None
