"""Confidence levels from seed disagreement, and a check that they mean something.

Levels: SD across seeds of the logit (not the probability: on the probability
scale the SD grows with the risk itself, so "low confidence" would just mean
"high risk"). Tertiles of that SD on validation eligible rows -> high / medium / low.

Check (per review, 29/09): within each risk decile (deciles of the mean
probability), the low-confidence rows must be worse than the high-confidence rows:
a larger calibration error |mean p - observed rate| and a higher Brier score. The
levels pass if low is worse on calibration error in at least 8 of 10 deciles AND
the pooled (row-weighted) calibration error of low is at least 1.2 x that of high
AND the pooled Brier is worse for low. (With random levels this passes in about 1 %
of simulations; a looser 6-of-10 rule passed about 20 % of the time.) If they
fail, LightGBM confidence levels are dropped and only DL seed disagreement is kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import logit

LEVELS = ("high", "medium", "low")


def seed_levels(prob_seeds: np.ndarray, fit_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """prob_seeds [seeds, n] -> (mean_p, sd_logit, level, cuts); cuts from `fit_rows` (bool mask)."""
    P = np.asarray(prob_seeds, float)
    sd = logit(P).std(0)
    mean_p = P.mean(0)
    ok = fit_rows & np.isfinite(sd)
    cuts = np.quantile(sd[ok], [1 / 3, 2 / 3])
    level = np.where(~np.isfinite(sd), "", np.where(sd <= cuts[0], "high", np.where(sd <= cuts[1], "medium", "low")))
    return mean_p, sd, level, cuts


def check_by_risk_decile(y: np.ndarray, p: np.ndarray, level: np.ndarray, min_rows: int = 30) -> tuple[pd.DataFrame, dict]:
    """Compare low vs high confidence within deciles of p (rows with y in {0,1})."""
    y, p, level = np.asarray(y), np.asarray(p, float), np.asarray(level)
    ok = ((y == 0) | (y == 1)) & np.isfinite(p) & (level != "")
    y, p, level = y[ok].astype(float), p[ok], level[ok]
    dec = np.minimum((pd.Series(p).rank(pct=True, method="first").to_numpy() * 10).astype(int), 9)
    rows = []
    for d in range(10):
        for lv in LEVELS:
            m = (dec == d) & (level == lv)
            if m.sum() < min_rows:
                rows.append({"decile": d, "confidence": lv, "rows": int(m.sum())})
                continue
            rows.append({"decile": d, "confidence": lv, "rows": int(m.sum()), "mean_p": float(p[m].mean()),
                         "observed": float(y[m].mean()), "calibration_error": float(abs(p[m].mean() - y[m].mean())),
                         "brier": float(np.mean((p[m] - y[m]) ** 2))})
    tab = pd.DataFrame(rows).reindex(columns=["decile", "confidence", "rows", "mean_p", "observed",
                                             "calibration_error", "brier"])  # columns exist even if all groups are small
    piv = tab.pivot(index="decile", columns="confidence", values="calibration_error")
    both = piv[["low", "high"]].dropna() if {"low", "high"} <= set(piv.columns) else pd.DataFrame()
    worse = int((both["low"] > both["high"]).sum()) if len(both) else 0

    def pooled(lv, col):
        t = tab[(tab.confidence == lv) & tab[col].notna()] if col in tab else tab.iloc[:0]
        return float(np.average(t[col], weights=t["rows"])) if len(t) else np.nan

    verdict = {"deciles_compared": int(len(both)), "deciles_low_worse_calibration": worse,
               "pooled_calibration_error_low": pooled("low", "calibration_error"),
               "pooled_calibration_error_high": pooled("high", "calibration_error"),
               "pooled_brier_low": pooled("low", "brier"), "pooled_brier_high": pooled("high", "brier")}
    verdict["passed"] = bool(len(both) >= 8 and worse >= 0.8 * len(both)
                             and verdict["pooled_calibration_error_low"] >= 1.2 * verdict["pooled_calibration_error_high"]
                             and verdict["pooled_brier_low"] > verdict["pooled_brier_high"])
    return tab, verdict
