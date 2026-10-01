"""Alarm replay (plan section 8.2).

Rows must be sorted by (caseid, time). An alarm fires on the second consecutive
eligible row above threshold, then no new alarm until the probability has gone
below threshold (or a row is not eligible / missing) and 300 s have passed since
the last alarm. A gap > 30 s in the grid, or a new case, resets the streak.
`replay_reference` is the plan pseudo-code; `replay` is the fast version.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


REARM = ("drop_below", "cooldown")  # plan 8.2 / repeat every `cooldown` s while still above threshold


@njit(cache=True)
def _replay(case_idx, time, ok, prob, thr, persistence, cooldown, cadence, repeat):
    n = len(time)
    fired = np.zeros(n, np.bool_)
    on = np.zeros(n, np.bool_)  # alarm state is on at this row
    streak = 0
    active = False
    next_allowed = -np.inf
    prev_t = -np.inf
    prev_case = -1
    for i in range(n):
        if case_idx[i] != prev_case:
            streak = 0
            active = False
            next_allowed = -np.inf
            prev_t = -np.inf
            prev_case = case_idx[i]
        if time[i] - prev_t > cadence:
            streak = 0
            active = False
        prev_t = time[i]
        p = prob[i]
        if (not ok[i]) or np.isnan(p) or p < thr:
            streak = 0
            active = False
            continue
        streak += 1
        if (repeat or not active) and streak >= persistence and time[i] >= next_allowed:
            fired[i] = True
            active = True
            next_allowed = time[i] + cooldown
        on[i] = active
    return fired, on


def replay(case_idx: np.ndarray, time: np.ndarray, eligible: np.ndarray, prob: np.ndarray, thr: float,
           persistence: int = 2, cooldown: float = 300.0, cadence: float = 30.0,
           rearm: str = "drop_below") -> np.ndarray:
    """Boolean mask of rows where an alarm fires. `case_idx`: integer case code per row.

    rearm="drop_below" (plan 8.2): after an alarm, the probability must go below threshold before
    another alarm. rearm="cooldown": while it stays above threshold, alarm again every `cooldown` s.
    """
    return replay_state(case_idx, time, eligible, prob, thr, persistence, cooldown, cadence, rearm)[0]


def replay_state(case_idx, time, eligible, prob, thr, persistence=2, cooldown=300.0, cadence=30.0,
                 rearm="drop_below") -> tuple[np.ndarray, np.ndarray]:
    """(fired, on): rows where an alarm fires, and rows where the alarm state is on (from an alarm
    until the probability drops below threshold, a row is not eligible, or the grid has a gap)."""
    if rearm not in REARM:
        raise ValueError(rearm)
    return _replay(np.asarray(case_idx, np.int64), np.asarray(time, np.float64), np.asarray(eligible, np.bool_),
                   np.asarray(prob, np.float64), float(thr), int(persistence), float(cooldown), float(cadence),
                   rearm == "cooldown")


def replay_reference(case_rows, thr, persistence=2, cooldown=300.0, cadence=30.0, rearm="drop_below"):
    """Plan 8.2 pseudo-code for one case: rows = iterable of (time, eligible, probability).
    With rearm="cooldown", `not active` is dropped from the alarm condition."""
    alarms, streak, active = [], 0, False
    next_allowed, prev_t = -np.inf, -np.inf
    for t, el, p in sorted(case_rows):
        if t - prev_t > cadence:
            streak, active = 0, False
        prev_t = t
        if not el or np.isnan(p) or p < thr:
            streak, active = 0, False
            continue
        streak += 1
        if (rearm == "cooldown" or not active) and streak >= persistence and t >= next_allowed:
            alarms.append(t)
            active = True
            next_allowed = t + cooldown
    return alarms
