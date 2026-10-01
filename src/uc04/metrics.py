"""Row, event and alarm metrics (plan section 8.3).

`Evaluator` is built once per (split, horizon) from the labels of every row of
the cases and the events table; `evaluate(prob, thr)` replays alarms and returns
per-case counts (`case_metrics`) and per-event leads. `summarize` turns (possibly
bootstrap-weighted) per-case counts into the reported metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .alarms import replay_state

CASE_COLUMNS = ("events_all", "events_eligible", "events_detected", "events_detected_all", "events_early",
                "true_alarms", "false_alarms", "censored_alarms", "evaluable_seconds", "eligible_seconds",
                "scheduled_seconds", "alarm_seconds")
_KEY = 1e7  # case code * _KEY + onset: sortable composite key (times < 1e7 s)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10, w: np.ndarray | None = None) -> float:
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    idx = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    tot = w.sum()
    out = 0.0
    for b in range(bins):
        m = idx == b
        wb = w[m].sum()
        if wb > 0:
            out += wb / tot * abs(np.average(p[m], weights=w[m]) - np.average(y[m], weights=w[m]))
    return float(out)


def row_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None, bins: int = 10) -> dict:
    """AUROC, AUPRC, Brier, ECE and prevalence on rows with y in {0,1} and finite p."""
    y, p = np.asarray(y), np.asarray(p, float)
    ok = ((y == 0) | (y == 1)) & np.isfinite(p)
    y, p = y[ok].astype(int), p[ok]
    w = None if w is None else np.asarray(w, float)[ok]
    out = {"n_rows": int(ok.sum()), "prevalence": float(np.average(y, weights=w)) if len(y) else np.nan}
    if len(np.unique(y)) < 2:
        return {**out, "auroc": np.nan, "auprc": np.nan, "brier": np.nan, "ece": np.nan}
    out["auroc"] = float(roc_auc_score(y, p, sample_weight=w))
    out["auprc"] = float(average_precision_score(y, p, sample_weight=w))
    out["brier"] = float(np.average((p - y) ** 2, weights=w))
    out["ece"] = ece(y, p, bins, w)
    return out


@dataclass
class Evaluation:
    case_table: pd.DataFrame      # one row per case, CASE_COLUMNS
    event_lead: np.ndarray        # per event: max lead of true alarms (NaN if not detected)
    event_case: np.ndarray        # per event: case position
    event_eligible: np.ndarray    # per event: eligible_h
    alarms: pd.DataFrame          # time, caseid, kind (true/false/censored), lead


class Evaluator:
    def __init__(self, labels: pd.DataFrame, events: pd.DataFrame, horizon: int, *, persistence: int = 2,
                 cooldown: float = 300.0, cadence: float = 30.0, early_lead: float = 300.0,
                 rearm: str = "drop_below"):
        self.order = np.lexsort((labels["time"].to_numpy(float), labels["caseid"].to_numpy()))  # row positions
        lab = labels.iloc[self.order].reset_index(drop=True)
        self.h = int(horizon)
        self.persistence, self.cooldown, self.cadence, self.early_lead = persistence, cooldown, cadence, early_lead
        self.rearm = rearm
        self.cases = np.unique(lab["caseid"].to_numpy())
        self.case_code = np.searchsorted(self.cases, lab["caseid"].to_numpy())
        self.subject_of_case = lab.groupby("caseid")["subjectid"].first().reindex(self.cases).to_numpy()
        self.time = lab["time"].to_numpy(float)
        self.eligible = lab["eligible"].to_numpy(bool)
        self.y = lab[f"y_{self.h}"].to_numpy(np.int8)
        self.exposure = lab["exposure_seconds"].to_numpy(float)
        ev = events[events["caseid"].isin(self.cases)].sort_values(["caseid", "onset"], kind="stable")
        self.ev_case = np.searchsorted(self.cases, ev["caseid"].to_numpy())
        self.ev_onset = ev["onset"].to_numpy(float)
        self.ev_eligible = ev[f"eligible_{self.h}"].to_numpy(bool)
        self.ev_key = self.ev_case * _KEY + self.ev_onset
        n = len(self.cases)
        self.scheduled = np.bincount(self.case_code, self.exposure, n)
        self.eligible_s = np.bincount(self.case_code, self.exposure * self.eligible, n)
        self.events_all = np.bincount(self.ev_case, minlength=n)
        self.events_eligible = np.bincount(self.ev_case, self.ev_eligible, n)

    def sort(self, values: np.ndarray) -> np.ndarray:
        """Reorder a per-row array given in the labels' original order."""
        return np.asarray(values)[self.order]

    def evaluate(self, prob_sorted: np.ndarray, thr: float) -> Evaluation:
        """`prob_sorted` must be in (caseid, time) order (use `sort`)."""
        n = len(self.cases)
        p = np.asarray(prob_sorted, float)
        fired_mask, on = replay_state(self.case_code, self.time, self.eligible, p, thr, self.persistence,
                                      self.cooldown, self.cadence, self.rearm)
        fired = np.flatnonzero(fired_mask)
        a_case, a_time, a_y = self.case_code[fired], self.time[fired], self.y[fired]
        censored = a_y == -1
        j = np.searchsorted(self.ev_key, a_case * _KEY + a_time, side="right")
        jj = np.minimum(j, max(len(self.ev_key) - 1, 0))
        hit = (~censored) & (j < len(self.ev_key)) & (len(self.ev_key) > 0)
        if len(self.ev_key):
            hit &= (self.ev_case[jj] == a_case) & (self.ev_onset[jj] <= a_time + self.h)
        true_ = hit
        false_ = (~censored) & (~hit)
        lead_alarm = np.where(true_, self.ev_onset[jj] - a_time if len(self.ev_key) else np.nan, np.nan)

        lead = np.full(len(self.ev_key), np.nan)
        if true_.any():
            ev_idx = jj[true_]
            np.fmax.at(lead, ev_idx, lead_alarm[true_])
        detected = np.isfinite(lead)
        early = detected & (lead >= self.early_lead)
        evaluable_row = self.eligible & (self.y != -1) & np.isfinite(p)
        table = pd.DataFrame({
            "caseid": self.cases, "subjectid": self.subject_of_case,
            "events_all": self.events_all, "events_eligible": self.events_eligible,
            "events_detected": np.bincount(self.ev_case, detected & self.ev_eligible, n),
            "events_detected_all": np.bincount(self.ev_case, detected, n),
            "events_early": np.bincount(self.ev_case, early & self.ev_eligible, n),
            "true_alarms": np.bincount(a_case, true_, n), "false_alarms": np.bincount(a_case, false_, n),
            "censored_alarms": np.bincount(a_case, censored, n),
            "evaluable_seconds": np.bincount(self.case_code, self.exposure * evaluable_row, n),
            "eligible_seconds": self.eligible_s, "scheduled_seconds": self.scheduled,
            "alarm_seconds": np.bincount(self.case_code, self.exposure * (on & self.eligible), n),
        })
        kind = np.where(censored, "censored", np.where(true_, "true", "false"))
        alarms = pd.DataFrame({"caseid": self.cases[a_case], "time": a_time, "kind": kind, "lead": lead_alarm})
        return Evaluation(table, lead, self.ev_case, self.ev_eligible, alarms)

    def quick(self, prob_sorted: np.ndarray, thr: float) -> tuple[float, float, float]:
        """(event_sensitivity, false_alarms_per_hour, alarm_ppv) at one threshold."""
        s = summarize_counts(self.evaluate(prob_sorted, thr).case_table.sum(numeric_only=True))
        return s["event_sensitivity"], s["false_alarms_per_hour"], s["alarm_ppv"]


def _div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else np.nan


def summarize_counts(c) -> dict:
    """Metrics from summed CASE_COLUMNS (a Series or dict)."""
    return {
        "event_sensitivity": _div(c["events_detected"], c["events_eligible"]),
        "events_detected": int(c["events_detected"]), "events_eligible": int(c["events_eligible"]),
        "event_sensitivity_all": _div(c["events_detected_all"], c["events_all"]),
        "early_sensitivity": _div(c["events_early"], c["events_eligible"]),
        "false_alarms_per_hour": _div(c["false_alarms"], c["evaluable_seconds"] / 3600.0),
        "alarm_ppv": _div(c["true_alarms"], c["true_alarms"] + c["false_alarms"]),
        "prediction_coverage": _div(c["eligible_seconds"], c["scheduled_seconds"]),
        # share of eligible exposure time during which the alarm state is on (plan 2.6, 8.3)
        "alarm_time_fraction": _div(c.get("alarm_seconds", np.nan), c["eligible_seconds"]),
        "true_alarms": int(c["true_alarms"]), "false_alarms": int(c["false_alarms"]),
        "censored_alarms": int(c["censored_alarms"]),
    }


def lead_quantiles(ev: Evaluation) -> dict:
    lead = ev.event_lead[np.isfinite(ev.event_lead) & ev.event_eligible]
    if not len(lead):
        return {"lead_median_s": np.nan, "lead_q25_s": np.nan, "lead_q75_s": np.nan}
    q25, q50, q75 = np.percentile(lead, [25, 50, 75])
    return {"lead_median_s": float(q50), "lead_q25_s": float(q25), "lead_q75_s": float(q75)}
