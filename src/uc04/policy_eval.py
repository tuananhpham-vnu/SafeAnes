"""Alarm-policy evaluation helpers shared by the week-14 analyses: threshold choice under the false-alarm budget,
evaluation on any subset of cases, per-row thresholds, and patient-level bootstrap CIs (plan 8.5).

`lab` holds every main-cohort row of a split (caseid, subjectid, time, exposure_seconds, eligible, y_<h>);
`events` has caseid, onset, eligible_<h>. A per-row threshold array is applied as score - thr against 0, which
replays the same alarm logic (2 consecutive rows above, 5-min cooldown).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap import Bootstrap, ci, paired_ci
from .metrics import Evaluator, lead_quantiles, row_metrics, summarize_counts
from .thresholds import candidates, choose_threshold

KEYS = ("auroc", "auprc", "event_sensitivity", "false_alarms_per_hour", "alarm_ppv", "early_sensitivity")


def _sub(lab: pd.DataFrame, mask) -> pd.DataFrame:
    return lab if mask is None else lab[np.asarray(mask, bool)]


def pick(lab, events, p, h, mask=None, budget: float = 1.0) -> float:
    sub = _sub(lab, mask)
    pp = np.asarray(p, float) if mask is None else np.asarray(p, float)[np.asarray(mask, bool)]
    ev = Evaluator(sub, events, h)
    ps = ev.sort(pp)
    return choose_threshold(candidates(pp[sub["eligible"].to_numpy(bool)]), lambda t: ev.quick(ps, t),
                            budget=budget)["threshold"]


def evaluate(lab, events, p, thr, h, mask=None, bootstrap: int = 0, seed: int = 20260917,
             return_draws: bool = False):
    """Metrics of one policy on the cases selected by `mask` (row mask over `lab`)."""
    m = None if mask is None else np.asarray(mask, bool)
    sub = _sub(lab, m)
    pp = np.asarray(p, float) if m is None else np.asarray(p, float)[m]
    ev = Evaluator(sub, events, h)
    ps = ev.sort(pp)
    if np.ndim(thr):
        tt = np.asarray(thr, float) if m is None else np.asarray(thr, float)[m]
        res = ev.evaluate(ps - ev.sort(tt), 0.0)
    else:
        res = ev.evaluate(ps, float(thr))
    sel = ev.eligible
    out = {"n_cases": int(len(ev.cases)), **row_metrics(ev.y[sel], ps[sel]),
           **summarize_counts(res.case_table.sum(numeric_only=True)), **lead_quantiles(res)}
    draws = None
    if bootstrap:
        subj_rows = sub["subjectid"].to_numpy()[ev.order][sel]
        bs = Bootstrap(subj_rows, res.case_table["subjectid"].to_numpy(), bootstrap, seed)
        draws = bs.metrics(res.case_table, ev.y[sel], ps[sel])
        out.update(ci(draws, KEYS))
    return (out, draws) if return_draws else out


def paired(draws_a, draws_b) -> dict:
    return paired_ci(draws_a, draws_b, KEYS)


def labels_frame(lab: pd.DataFrame, y: dict[int, np.ndarray], eligible: np.ndarray) -> pd.DataFrame:
    """A copy of `lab` with other labels / eligibility (e.g. the personalised label)."""
    out = lab[["caseid", "subjectid", "time", "exposure_seconds"]].copy()
    out["eligible"] = np.asarray(eligible, bool)
    for h, v in y.items():
        out[f"y_{h}"] = v
    return out
