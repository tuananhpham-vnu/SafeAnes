"""Eligibility and labels at each prediction time (plan section 3.5).

`eligible` is causal except `in_event` / `post_event`, which use events found in
label_map (accepted deviation, DEVIATIONS item 5). Labels: 1, 0 or -1 (unknown).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .events import longest_true_run

REASONS = ("map_unknown", "in_hypotension", "in_event", "post_event", "map_sparse", "recent_low")


@dataclass(frozen=True)
class LabelRules:
    threshold: float = 65.0
    horizons: tuple[int, ...] = (300, 600, 900, 1200, 1800)
    post_event_seconds: int = 120
    max_map_missing_w120: float = 0.5
    recent_low_seconds: int = 120
    neg_max_unknown_fraction: float = 0.05
    neg_max_unknown_run: int = 30
    confirm_seconds: int = 60  # an event starting at t + h needs 60 s after it
    merge_gap: int = 120
    negative_rule: str = "possible_event"

    @classmethod
    def from_config(cls, labels) -> "LabelRules":
        return cls(threshold=float(labels.map_threshold), horizons=tuple(labels.horizons_seconds),
                   merge_gap=int(labels.merge_gap_seconds), negative_rule=str(labels.negative_rule),
                   post_event_seconds=int(labels.post_event_exclusion_seconds),
                   max_map_missing_w120=float(labels.max_map_missing_w120),
                   neg_max_unknown_fraction=float(labels.negative_max_unknown_fraction),
                   neg_max_unknown_run=int(labels.negative_max_unknown_run_seconds),
                   confirm_seconds=int(labels.event_seconds))


def eligibility(times: np.ndarray, map_current: np.ndarray, map_missing_w120: np.ndarray,
                events: pd.DataFrame, rule: str, rules: LabelRules = LabelRules(),
                grid_start: float | None = None, grid_map: np.ndarray | None = None,
                grid_step: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (eligible, abstain_reason). The first reason met is recorded.

    `rule` is "event" or "any_low". "any_low" also needs the causal 2 s MAP grid
    (`values[:, map]` from the npz) to find any MAP < threshold in (t - 120, t].
    """
    if rule not in ("event", "any_low"):
        raise ValueError(rule)
    t = np.asarray(times, float)
    n = len(t)
    reason = np.full(n, "", dtype=object)

    def mark(mask: np.ndarray, name: str) -> None:
        reason[(reason == "") & mask] = name

    mc = np.asarray(map_current, float)
    mark(~np.isfinite(mc), "map_unknown")
    mark(mc < rules.threshold, "in_hypotension")
    on = events["onset"].to_numpy(float)
    en = events["end"].to_numpy(float)
    if len(on):
        mark(((on[None, :] <= t[:, None]) & (t[:, None] < en[None, :])).any(1), "in_event")
        mark(((en[None, :] <= t[:, None]) & (t[:, None] < en[None, :] + rules.post_event_seconds)).any(1),
             "post_event")
    miss = np.asarray(map_missing_w120, float)
    mark(miss > rules.max_map_missing_w120, "map_sparse")
    if rule == "any_low":
        if grid_map is None or grid_start is None:
            raise ValueError("rule 'any_low' needs grid_start and grid_map")
        g = np.asarray(grid_map, float)
        gt = grid_start + grid_step * np.arange(len(g))
        low = np.isfinite(g) & (g < rules.threshold)
        low_cum = np.concatenate(([0], np.cumsum(low)))
        lo = np.searchsorted(gt, t - rules.recent_low_seconds, side="right")  # gt > t - 120
        hi = np.searchsorted(gt, t, side="right")                            # gt <= t
        mark(low_cum[hi] - low_cum[lo] > 0, "recent_low")
    return reason == "", reason


NEGATIVE_RULES = ("possible_event", "unknown_fraction")
# label variants: lenient with the configured negative rule, lenient with each rule explicitly, strict
POLICIES = ("lenient", "lenient_possible", "lenient_fraction", "strict")


def _could_hide_event(future: np.ndarray, h: int, threshold: float, min_seconds: int) -> bool:
    """True if some run of seconds that are unknown or below threshold, at least `min_seconds`
    long, starts in (t, t + h]. `future[i]` is the second t + 1 + i."""
    possible = ~np.isfinite(future) | (future < threshold)
    d = np.diff(np.concatenate(([0], possible.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return bool(((ends - starts >= min_seconds) & (starts <= h - 1)).any())


UNKNOWN_REASONS = ("", "end_of_case", "possible_event", "unknown_fraction", "incomplete_future")


def horizon_labels(times: np.ndarray, label_map: np.ndarray, start: float, events: pd.DataFrame,
                   h: int, policy: str = "lenient", rules: LabelRules = LabelRules()) -> np.ndarray:
    return horizon_labels_with_reason(times, label_map, start, events, h, policy, rules)[0]


def horizon_labels_with_reason(times: np.ndarray, label_map: np.ndarray, start: float, events: pd.DataFrame,
                               h: int, policy: str = "lenient", rules: LabelRules = LabelRules()
                               ) -> tuple[np.ndarray, np.ndarray]:
    """y_h for every prediction time: 1, 0 or -1 (plan 3.5; negative rule revised, DEVIATIONS 23).

    Positive (lenient): an event starts in (t, t + h]. Otherwise -1 if t + h + 60 is past the
    end of the case, else by the negative rule:
      possible_event   : -1 if the unknown or below-threshold seconds in (t, t + h + 60] could hold
                         an event starting in (t, t + h] (a run >= 60 s starting there), else 0;
      unknown_fraction : (old) -1 if > 5 % unknown or an unknown run >= 30 s, else 0.
    strict: -1 unless every second of (t, t + h + 60] is known.

    Also returns why a label is -1 (`UNKNOWN_REASONS`, "" when known): end_of_case (checked
    first), possible_event, unknown_fraction, or incomplete_future (strict).
    """
    if policy not in POLICIES:
        raise ValueError(policy)
    neg = {"lenient": rules.negative_rule, "lenient_possible": "possible_event",
           "lenient_fraction": "unknown_fraction"}.get(policy)
    if neg is not None and neg not in NEGATIVE_RULES:
        raise ValueError(neg)
    lm = np.asarray(label_map, float)
    end = start + len(lm)
    on = events["onset"].to_numpy(float)
    y = np.empty(len(times), np.int8)
    why = np.full(len(times), "", dtype=object)
    for i, t in enumerate(np.asarray(times, float)):
        positive = bool(((on > t) & (on <= t + h)).any())
        j0 = int(t - start) + 1
        j1 = int(t - start) + h + rules.confirm_seconds + 1
        future = lm[j0:j1]
        beyond_end = t + h + rules.confirm_seconds > end
        if policy == "strict":
            if beyond_end:
                y[i], why[i] = -1, "end_of_case"
            elif not np.isfinite(future).all():
                y[i], why[i] = -1, "incomplete_future"
            else:
                y[i] = int(positive)
        elif positive:
            y[i] = 1
        elif beyond_end:
            y[i], why[i] = -1, "end_of_case"
        elif neg == "possible_event":
            hidden = _could_hide_event(future, h, rules.threshold, rules.confirm_seconds)
            y[i], why[i] = (-1, "possible_event") if hidden else (0, "")
        else:
            unknown = ~np.isfinite(future)
            bad = unknown.mean() > rules.neg_max_unknown_fraction or longest_true_run(unknown) >= rules.neg_max_unknown_run
            y[i], why[i] = (-1, "unknown_fraction") if bad else (0, "")
    return y, why


def event_eligible_h(events: pd.DataFrame, times: np.ndarray, eligible: np.ndarray, y_h: np.ndarray,
                     h: int) -> np.ndarray:
    """eligible_h(ev): some eligible row with y_h == 1 and onset - h <= t < onset."""
    t = np.asarray(times, float)
    ok = np.asarray(eligible, bool) & (np.asarray(y_h) == 1)
    on = events["onset"].to_numpy(float)
    if not len(on):
        return np.zeros(0, bool)
    win = (t[None, :] >= on[:, None] - h) & (t[None, :] < on[:, None])
    return (win & ok[None, :]).any(1)


def exposure_seconds(times: np.ndarray, end: float, cadence: float = 30.0) -> np.ndarray:
    return np.minimum(cadence, end - np.asarray(times, float))


def label_case(times: np.ndarray, map_current: np.ndarray, map_missing_w120: np.ndarray,
               label_map: np.ndarray, start: float, events: pd.DataFrame, *,
               rules: LabelRules = LabelRules(), grid_map: np.ndarray | None = None,
               policies: Sequence[str] = POLICIES,
               exclusion_rules: Sequence[str] = ("event", "any_low")) -> pd.DataFrame:
    """All label columns for one case.

    Columns: time, exposure_seconds, eligible_<rule>, abstain_reason_<rule>,
    y_<h>_<policy> for every rule, h and policy requested.
    """
    t = np.asarray(times, float)
    out = pd.DataFrame({"time": t, "exposure_seconds": exposure_seconds(t, start + len(label_map))})
    for rule in exclusion_rules:
        el, rs = eligibility(t, map_current, map_missing_w120, events, rule, rules,
                             grid_start=start, grid_map=grid_map)
        out[f"eligible_{rule}"] = el
        out[f"abstain_reason_{rule}"] = rs
    for policy in policies:
        for h in rules.horizons:
            y, why = horizon_labels_with_reason(t, label_map, start, events, h, policy, rules)
            out[f"y_{h}_{policy}"] = y
            out[f"unknown_reason_{h}_{policy}"] = why
    return out
