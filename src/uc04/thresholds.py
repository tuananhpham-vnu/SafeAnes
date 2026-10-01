"""Threshold choice under a false-alarm budget (plan section 8.4).

Candidates: 200 quantiles of the eligible validation probabilities plus the grid
0.01..0.99. Every candidate is evaluated (full scan) and the one with the highest
event sensitivity among those with FA/h <= budget is kept, ties broken by the
higher alarm PPV, then the lower threshold. If none is within budget, the lowest
FA/h is kept and `budget_not_met` is set.

Deviation from the plan (DEVIATIONS): the plan bisects on FA/h assuming it falls
monotonically with the threshold. It does not: at very low thresholds the
probability stays above threshold, the alarm stays active and fires once per
case, so FA/h is low again. Bisection can then miss the best threshold. With the
numba replay a full scan of ~300 candidates costs a few seconds, so it is used.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def candidates(prob_eligible: np.ndarray, n_quantiles: int = 200) -> np.ndarray:
    p = np.asarray(prob_eligible, float)
    p = p[np.isfinite(p)]
    q = np.quantile(p, np.linspace(0, 1, n_quantiles)) if len(p) else np.zeros(0)
    grid = np.round(np.arange(1, 100) / 100, 2)
    return np.unique(np.concatenate([q, grid]))


def _key(r: tuple[float, float, float], i: int):
    return (np.nan_to_num(r[0], nan=-1.0), np.nan_to_num(r[2], nan=-1.0), -i)


def choose_threshold(cands: np.ndarray, quick: Callable[[float], tuple[float, float, float]],
                     budget: float = 1.0, curve_points: int = 40) -> dict:
    """`quick(thr)` returns (event_sensitivity, false_alarms_per_hour, alarm_ppv)."""
    cands = np.sort(np.asarray(cands, float))
    res = [quick(float(t)) for t in cands]
    fa = np.array([np.inf if np.isnan(r[1]) else r[1] for r in res])
    ok = np.flatnonzero(fa <= budget)
    if len(ok):
        best = max(ok, key=lambda i: _key(res[i], i))
        not_met = False
    else:
        best = int(np.argmin(fa))
        not_met = True
    idx = np.unique(np.linspace(0, len(cands) - 1, curve_points).round().astype(int))
    curve = [{"threshold": float(cands[i]), "event_sensitivity": res[i][0], "false_alarms_per_hour": res[i][1],
              "alarm_ppv": res[i][2]} for i in idx]
    s, f, p = res[best]
    return {"threshold": float(cands[best]), "budget_not_met": bool(not_met), "event_sensitivity": s,
            "false_alarms_per_hour": f, "alarm_ppv": p, "n_candidates": int(len(cands)),
            "fa_monotone": bool(np.all(np.diff(fa[np.isfinite(fa)]) <= 1e-12)), "curve": curve}
