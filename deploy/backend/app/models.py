"""The four features_v8 models of the ward demo, each behind one interface.

score(X) -> raw scores [n, 5 horizons] (the score Platt was fitted on: logit of the model probability), and
explain(x, h) -> contributions of each input to that raw score (or None). Probability = Platt(score), exactly as in
uc04.tabular (calibrator mean / std / a / b of model.joblib). Explanations are mapped to the calibrated logit, which is
linear in the raw score.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (300, 600, 900, 1200, 1800)


class Platt:
    def __init__(self, c: dict):
        self.mean, self.std, self.a, self.b = c["mean"], c["std"], c["a"], c["b"]

    def logit(self, s: np.ndarray) -> np.ndarray:
        return self.a * (np.asarray(s, float) - self.mean) / self.std + self.b

    def prob(self, s: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(s)))


class Adapter:
    """Common part: meta.json (inputs, labels, per-horizon Platt / threshold / metrics)."""
    explains = True

    def __init__(self, root: Path):
        self.meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        self.name, self.title, self.columns = self.meta["name"], self.meta["title"], self.meta["columns"]
        self.platt = {h: Platt(self.meta["horizons"][str(h)]["calibrator"]) for h in HORIZONS}
        self.threshold = {h: float(self.meta["horizons"][str(h)]["threshold"]) for h in HORIZONS}

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.reindex(columns=self.columns).to_numpy(np.float64)

    def score(self, X: np.ndarray) -> np.ndarray:  # [n, 5]
        raise NotImplementedError

    def prob(self, X: np.ndarray) -> np.ndarray:
        s = self.score(X)
        return np.stack([self.platt[h].prob(s[:, k]) for k, h in enumerate(HORIZONS)], axis=1)

    def contrib(self, x: np.ndarray, h: int) -> tuple[np.ndarray, float] | None:
        """(per-input contributions, base) to the raw score of one row, or None."""
        return None


class LightGBM(Adapter):
    def __init__(self, root: Path):
        import lightgbm as lgb
        super().__init__(root)
        self.boosters = {h: lgb.Booster(model_file=str(root / f"h{h}.txt")) for h in HORIZONS}

    def score(self, X):
        return np.stack([self.boosters[h].predict(X, raw_score=True) for h in HORIZONS], axis=1)

    def contrib(self, x, h):
        c = self.boosters[h].predict(x[None, :], pred_contrib=True)[0]  # TreeSHAP, raw score units
        return c[:-1], float(c[-1])


class CatBoost(Adapter):
    def __init__(self, root: Path):
        from catboost import CatBoostClassifier
        super().__init__(root)
        self.models = {}
        for h in HORIZONS:
            m = CatBoostClassifier()
            m.load_model(str(root / f"h{h}.cbm"))
            self.models[h] = m

    def score(self, X):
        return np.stack([self.models[h].predict(X, prediction_type="RawFormulaVal") for h in HORIZONS], axis=1)

    def contrib(self, x, h):
        from catboost import Pool
        c = self.models[h].get_feature_importance(Pool(x[None, :]), type="ShapValues")[0]
        return c[:-1], float(c[-1])


class Logistic(Adapter):
    """Median imputation, standardisation, logistic regression: score = coef . z + intercept."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.p = {}
        for h in HORIZONS:
            d = json.loads((root / f"h{h}.json").read_text(encoding="utf-8"))
            self.p[h] = {k: np.asarray(v, float) for k, v in d.items()}

    def _z(self, X, h):
        p = self.p[h]
        X = np.where(np.isnan(X), p["medians"], X)
        return (X - p["mean"]) / p["scale"]

    def score(self, X):
        return np.stack([self._z(X, h) @ self.p[h]["coef"] + self.p[h]["intercept"] for h in HORIZONS], axis=1)

    def contrib(self, x, h):
        return self._z(x[None, :], h)[0] * self.p[h]["coef"], float(self.p[h]["intercept"])


class TabM(Adapter):
    """One multi-horizon TabM (uc04.tabm.TabMMulti inside the saved OneHorizon). No per-input attribution."""
    explains = False

    def __init__(self, root: Path):
        import joblib
        super().__init__(root)
        self.multi = joblib.load(root / "model.joblib")["model"].multi
        self.multi.device = "cpu"

    def score(self, X):
        return self.multi.logits(pd.DataFrame(X, columns=self.columns))


KINDS = {"lgbm_v8": LightGBM, "catboost_v8": CatBoost, "logistic_v8": Logistic, "tabm_v8": TabM}


def load_models(root: Path, names) -> dict[str, Adapter]:
    return {n: KINDS[n](root / n) for n in names}


def explain(m: Adapter, x: np.ndarray, h: int, top: int = 8) -> dict | None:
    """Contributions to the calibrated logit at one row, summed by input group, plus the top inputs."""
    c = m.contrib(x, h)
    if c is None:
        return None
    phi, base = c
    pl = m.platt[h]
    scale = pl.a / pl.std
    phi, base = phi * scale, (base - pl.mean) * scale + pl.b
    feats = m.meta["features"]
    groups: dict[str, list] = {}
    for f, v in zip(feats, phi):
        g = groups.setdefault(f["group"], [f["group_label"], 0.0])
        g[1] += float(v)
    order = np.argsort(-np.abs(phi))[:top]
    logit = base + float(phi.sum())
    return {"horizon_s": h, "probability": float(1 / (1 + np.exp(-logit))), "base_logit": float(base),
            "groups": [{"id": g, "label": v[0], "contribution": v[1]}
                       for g, v in sorted(groups.items(), key=lambda kv: -abs(kv[1][1]))],
            "features": [{"column": feats[i]["column"], "label": feats[i]["label"], "unit": feats[i]["unit"],
                          "group": feats[i]["group"], "contribution": float(phi[i]),
                          "value": None if not np.isfinite(x[i]) else float(x[i])} for i in order]}
