"""Calibration (plan section 8.1).

- Tabular: Platt on the raw score s: standardise s on the calibration rows, then
  LogisticRegression(C=1e6); p = sigmoid(a * z + b).
- DL: one temperature T > 0 and one bias b shared by the 5 horizons,
  p = sigmoid(logit / T + b), fitted by L-BFGS-B on all known labels. A shared
  monotone map keeps p increasing across horizons.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


@dataclass
class Platt:
    mean: float = 0.0
    std: float = 1.0
    a: float = 1.0
    b: float = 0.0

    def fit(self, s: np.ndarray, y: np.ndarray) -> "Platt":
        s, y = np.asarray(s, float), np.asarray(y, int)
        ok = np.isfinite(s) & ((y == 0) | (y == 1))
        s, y = s[ok], y[ok]
        if len(np.unique(y)) < 2:
            raise ValueError("calibration needs both classes")
        self.mean, self.std = float(s.mean()), float(s.std() or 1.0)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(((s - self.mean) / self.std)[:, None], y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def predict(self, s: np.ndarray) -> np.ndarray:
        z = (np.asarray(s, float) - self.mean) / self.std
        return expit(self.a * z + self.b)  # NaN score -> NaN probability

    def to_dict(self) -> dict:
        return {"method": "platt_on_score", "mean": self.mean, "std": self.std, "a": self.a, "b": self.b}


@dataclass
class TemperatureBias:
    log_t: float = 0.0
    b: float = 0.0

    @property
    def temperature(self) -> float:
        return float(np.exp(self.log_t))

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureBias":
        """logits, y: [n, K]; y == -1 is ignored."""
        z, y = np.asarray(logits, float), np.asarray(y)
        known = (y == 0) | (y == 1)
        z, t = z[known], y[known].astype(float)

        def loss(theta):
            lt, b = theta
            u = z / np.exp(lt) + b
            p = expit(u)
            # mean BCE and its gradient
            val = np.mean(np.logaddexp(0, u) - t * u)
            g_u = (p - t) / len(u)
            return val, np.array([np.sum(g_u * (-z / np.exp(lt))), np.sum(g_u)])

        res = minimize(loss, x0=np.array([0.0, 0.0]), jac=True, method="L-BFGS-B",
                       bounds=[(-4, 4), (-20, 20)])
        self.log_t, self.b = float(res.x[0]), float(res.x[1])
        return self

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return expit(np.asarray(logits, float) / self.temperature + self.b)

    def to_dict(self) -> dict:
        return {"method": "shared_temperature_and_bias", "temperature": self.temperature, "log_t": self.log_t,
                "b": self.b}
