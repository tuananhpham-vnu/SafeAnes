"""Models, predictions, explanations and alarms of the SafeAnes demo backend.

Mirrors the UC04 evaluation: probability = Platt(raw LightGBM score) per horizon; an alarm fires after
`PERSISTENCE` consecutive eligible predictions at or above the horizon's threshold, at most once per
`COOLDOWN` s, and re-arms only after the probability drops below threshold (uc04.alarms.replay_reference,
plan 8.2). Explanations are LightGBM TreeSHAP contributions mapped to the calibrated logit (Platt is
linear in the raw score), summed by physiological group.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import GROUPS, UNITS, group_of, label_of

HORIZONS = (300, 600, 900, 1200, 1800)
PERSISTENCE, COOLDOWN, CADENCE = 2, 300.0, 30.0
VITALS = ("map", "sbp", "dbp", "hr", "spo2", "etco2")


@dataclass
class HorizonModel:
    booster: lgb.Booster
    mean: float
    std: float
    a: float
    b: float
    threshold: float
    metrics: dict

    def raw(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X, raw_score=True)

    def prob(self, raw: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * (raw - self.mean) / self.std + self.b)))


class Model:
    def __init__(self, root: Path):
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        self.name, self.title, self.columns = meta["name"], meta["title"], meta["columns"]
        self.horizons: dict[int, HorizonModel] = {}
        for h in HORIZONS:
            m = meta["horizons"][str(h)]
            c = m["calibrator"]
            self.horizons[h] = HorizonModel(lgb.Booster(model_file=str(root / f"h{h}.txt")), c["mean"], c["std"],
                                            c["a"], c["b"], m["threshold"], m["metrics"])

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.reindex(columns=self.columns).to_numpy(np.float64)

    def predict(self, frame: pd.DataFrame) -> dict[int, np.ndarray]:
        X = self.matrix(frame)
        return {h: m.prob(m.raw(X)) for h, m in self.horizons.items()}

    def explain(self, row: pd.Series, h: int, top: int = 8) -> dict:
        """Contributions to the calibrated logit at one time point."""
        m = self.horizons[h]
        x = self.matrix(row.to_frame().T)
        contrib = m.booster.predict(x, pred_contrib=True)[0]          # [n_features + 1], raw logit units
        scale = m.a / m.std
        phi, base = contrib[:-1] * scale, (contrib[-1] - m.mean) * scale + m.b
        groups: dict[str, float] = {}
        for c, v in zip(self.columns, phi):
            groups[group_of(c)] = groups.get(group_of(c), 0.0) + float(v)
        order = np.argsort(-np.abs(phi))[:top]
        feats = []
        for i in order:
            c, val = self.columns[i], x[0, i]
            sig = c.split("_")[0] if not c.startswith("f1a_") else c.split("_")[1]
            feats.append({"column": c, "label": label_of(c), "group": group_of(c), "contribution": float(phi[i]),
                          "value": None if not np.isfinite(val) else float(val),
                          "unit": UNITS.get(sig, "") if not c.endswith(("missing_w120", "_pct")) else ""})
        logit = base + phi.sum()
        return {"horizon_s": h, "probability": float(1 / (1 + np.exp(-logit))), "base_logit": float(base),
                "groups": [{"id": g, "label": GROUPS[g][0], "kind": GROUPS[g][1], "contribution": v}
                           for g, v in sorted(groups.items(), key=lambda kv: -abs(kv[1]))],
                "features": feats}


def replay_alarms(time: np.ndarray, eligible: np.ndarray, prob: np.ndarray, thr: float) -> list[float]:
    """uc04.alarms.replay_reference for one case, rearm='drop_below'."""
    alarms, streak, active, next_allowed, prev_t = [], 0, False, -np.inf, -np.inf
    for t, el, p in zip(time, eligible, prob):
        if t - prev_t > CADENCE:
            streak, active = 0, False
        prev_t = t
        if not el or not np.isfinite(p) or p < thr:
            streak, active = 0, False
            continue
        streak += 1
        if not active and streak >= PERSISTENCE and t >= next_allowed:
            alarms.append(float(t))
            active, next_allowed = True, t + COOLDOWN
    return alarms


def classify_alarm(t: float, h: int, onsets: np.ndarray, y_at_t: float) -> dict:
    """true if an event starts in (t, t + h], censored if the label at t is unknown, else false."""
    nxt = onsets[onsets > t]
    if len(nxt) and nxt[0] <= t + h:
        return {"time_s": t, "kind": "true", "lead_s": float(nxt[0] - t)}
    return {"time_s": t, "kind": "censored" if y_at_t == -1 else "false", "lead_s": None}


def risk_and_alarms(mdl: Model, frame: pd.DataFrame, t: np.ndarray, el: np.ndarray, onsets: np.ndarray,
                    y_unknown) -> tuple[dict, dict]:
    """Probabilities (None where not predicted) and replayed alarms per horizon. `y_unknown(h, i)` says whether
    the outcome of an alarm at row i is unknown (censored)."""
    risk, alarms = {}, {}
    for h, p in mdl.predict(frame).items():
        p = np.where(el, p, np.nan)
        risk[str(h)] = [None if not np.isfinite(v) else round(float(v), 4) for v in p]
        fired = replay_alarms(t, el, p, mdl.horizons[h].threshold)
        idx = np.searchsorted(t, fired)
        alarms[str(h)] = [classify_alarm(ft, h, onsets, -1 if y_unknown(h, i) else 0) for ft, i in zip(fired, idx)]
    return risk, alarms


def rounded(a) -> list:
    return [None if not np.isfinite(x) else round(float(x), 1) for x in np.asarray(a, float)]


class Store:
    def __init__(self, data: Path):
        self.data = data
        self.models = {p.name: Model(p) for p in sorted((data / "models").iterdir()) if (p / "meta.json").exists()}
        self.cases = {c["caseid"]: c for c in json.loads((data / "cases.json").read_text(encoding="utf-8"))}
        rp = data / "report.json"
        self.report = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}

    @lru_cache(maxsize=64)
    def frame(self, caseid: int) -> pd.DataFrame:
        return pd.read_parquet(self.data / "cases" / f"{caseid}.parquet").sort_values("time").reset_index(drop=True)

    @lru_cache(maxsize=64)
    def timeline(self, caseid: int, model: str) -> dict:
        df, meta, mdl = self.frame(caseid), self.cases[caseid], self.models[model]
        t = df.time.to_numpy(float)
        el = df.eligible.to_numpy(bool)
        onsets = np.array([e["onset_s"] for e in meta["events"]], float)
        risk, alarms = risk_and_alarms(mdl, df, t, el, onsets, lambda h, i: df[f"y_{h}"].iat[i] == -1)
        vit = {v: rounded(df[f"{v}_current"]) for v in VITALS}
        return {"caseid": caseid, "model": model, "time_s": t.tolist(), "eligible": el.tolist(),
                "surgery_start_s": meta["surgery_start_s"], "events": meta["events"], "vitals": vit, "risk": risk,
                "alarms": alarms, "thresholds": {str(h): m.threshold for h, m in mdl.horizons.items()}}

    def explain(self, caseid: int, model: str, t: float, h: int) -> dict:
        df = self.frame(caseid)
        i = int(np.clip(np.searchsorted(df.time.to_numpy(), t, side="right") - 1, 0, len(df) - 1))
        row = df.iloc[i]
        pre = row.time <= self.cases[caseid]["surgery_start_s"]
        return {**explain_row(self.models[model], row, h, float(row.time), bool(row.eligible), pre), "caseid": caseid}


CONTEXT_SHOWN = ("f1a_min_since_anestart", "f1a_min_since_opstart", "f1a_n_prev_events", "f1a_min_since_last_event",
                 "f1a_map_rel_case_median", "map_drop_pct")


def explain_row(mdl: Model, row: pd.Series, h: int, t: float, eligible: bool, pre: bool) -> dict:
    """Explanation at one time point, with the case context and the missing fraction of each vital (2 min)."""
    val = lambda c: None if c not in row or pd.isna(row[c]) else float(row[c])
    return {**mdl.explain(row.reindex(mdl.columns), h), "time_s": t, "eligible": eligible,
            "phase": "pre_incision" if pre else "surgery", "context": {c: val(c) for c in CONTEXT_SHOWN},
            "data_quality": {v: val(f"{v}_missing_w120") for v in VITALS}}
