"""Continuous replay and metrics. Selection-bias rationale: S03; calibration: S05.

Alarm persistence, cooldown, matching and FA/hour grid are PROJECT choices.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .config import Protocol


def window_metrics(y, probability):
    y, probability = np.asarray(y), np.asarray(probability, float)
    if len(y) == 0:
        return {k: None for k in ("auroc", "average_precision", "brier", "ece", "prevalence")}
    if not np.isin(y, [0, 1]).all() or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Metrics require known binary labels and finite probabilities")
    two_classes = len(np.unique(y)) == 2
    bins = np.minimum((probability * 10).astype(int), 9)  # fixed 10 equal-width bins
    ece = sum(np.mean(bins == i) * abs(np.mean(y[bins == i]) - np.mean(probability[bins == i]))
              for i in range(10) if np.any(bins == i))
    return {"auroc": float(roc_auc_score(y, probability)) if two_classes else None,
            "average_precision": float(average_precision_score(y, probability)) if two_classes else None,
            "brier": float(brier_score_loss(y, probability)), "ece": float(ece),
            "prevalence": float(np.mean(y))}


def replay_alarms(frame, threshold, protocol=Protocol()):
    """Return emitted timestamps; unknown future labels never control this state."""
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    frame = frame.sort_values("time")
    if frame.time.duplicated().any():
        raise ValueError("Duplicate decision timestamps")
    emitted, streak, active = [], 0, False
    next_allowed, previous = -np.inf, -np.inf
    for row in frame.itertuples():
        if row.time - previous > protocol.cadence_seconds:
            streak, active = 0, False
        previous = row.time
        if not row.eligible or not np.isfinite(row.probability):
            streak, active = 0, False
            continue
        if row.probability < threshold:
            streak, active = 0, False
            continue
        streak += 1
        if not active and streak >= protocol.alarm_persistence and row.time >= next_allowed:
            emitted.append(float(row.time))
            active = True
            next_allowed = row.time + protocol.alarm_cooldown_seconds
    return emitted


def evaluate_predictions(frame, events, horizon, threshold, protocol=Protocol()):
    if frame.duplicated(["caseid", "time"]).any():
        raise ValueError("Duplicate case/time predictions")
    label = f"y_{horizon}"
    known = frame.eligible & frame[label].ge(0) & frame.probability.notna()
    metrics = window_metrics(frame.loc[known, label], frame.loc[known, "probability"])
    case_stats, alarm_rows, leads = [], [], []
    for caseid, case in frame.groupby("caseid", sort=False):
        if case.subjectid.nunique() != 1:
            raise ValueError("A case cannot belong to multiple patients")
        case_events = events[events.caseid.eq(caseid)].sort_values("onset")
        eligible_events = case_events[case_events[f"eligible_{horizon}"].astype(bool)]
        matched, early = set(), set()
        true_count = false_count = censored_count = 0
        indexed = case.set_index("time")
        for t in replay_alarms(case, threshold, protocol):
            if indexed.loc[t, label] < 0:
                censored_count += 1
                alarm_rows.append({"caseid": int(caseid), "time": t, "outcome": "censored", "lead_seconds": None})
                continue
            upcoming = case_events[(case_events.onset > t) & (case_events.onset <= t + horizon)]
            if len(upcoming):
                event = upcoming.iloc[0]
                lead = float(event.onset - t)
                matched.add(float(event.onset))
                if lead >= 300:
                    early.add(float(event.onset))
                true_count += 1
                alarm_rows.append({"caseid": int(caseid), "time": t, "outcome": "true", "lead_seconds": lead})
            else:
                false_count += 1
                alarm_rows.append({"caseid": int(caseid), "time": t, "outcome": "false", "lead_seconds": None})
        eligible_onsets = set(eligible_events.onset)
        for onset in matched & eligible_onsets:
            event_leads = [a["lead_seconds"] for a in alarm_rows if a["caseid"] == caseid
                           and a["outcome"] == "true" and a["time"] + a["lead_seconds"] == onset]
            leads.append(max(event_leads))
        eligible = case.eligible
        evaluable = eligible & case[label].ge(0) & case.probability.notna()
        case_stats.append({"caseid": int(caseid), "subjectid": int(case.subjectid.iloc[0]),
            "events_all": len(case_events), "events_eligible": len(eligible_onsets),
            "events_detected": len(matched & eligible_onsets),
            "events_early_5min": len(early & eligible_onsets),
            "true_alarms": true_count, "false_alarms": false_count, "censored_alarms": censored_count,
            "evaluable_seconds": float(case.loc[evaluable, "exposure_seconds"].sum()),
            "eligible_seconds": float(case.loc[eligible, "exposure_seconds"].sum()),
            "scheduled_seconds": float(case.exposure_seconds.sum()),
            "known_windows": int(evaluable.sum())})
    summary = pd.DataFrame(case_stats)
    metrics.update(aggregate_cases(summary))
    metrics["lead_seconds_median"] = float(np.median(leads)) if leads else None
    metrics["lead_seconds_q25"] = float(np.quantile(leads, .25)) if leads else None
    metrics["lead_seconds_q75"] = float(np.quantile(leads, .75)) if leads else None
    metrics["threshold"] = float(threshold)
    metrics["horizon_seconds"] = horizon
    metrics["exposure_method"] = "30-second decision-grid approximation; see PROTOCOL.md"
    return metrics, summary, alarm_rows


def aggregate_cases(summary):
    count_names = ("events_all", "events_eligible", "events_detected", "events_early_5min",
                   "true_alarms", "false_alarms", "censored_alarms", "known_windows")
    sums = summary.sum(numeric_only=True) if len(summary) else {}
    out = {key: int(sums.get(key, 0)) for key in count_names}
    hours = float(sums.get("evaluable_seconds", 0)) / 3600
    def ratio(a, b):
        return float(a / b) if b else None
    out.update({"event_sensitivity": ratio(out["events_detected"], out["events_eligible"]),
        "event_sensitivity_all": ratio(out["events_detected"], out["events_all"]),
        "early_5min_sensitivity": ratio(out["events_early_5min"], out["events_eligible"]),
        "alarm_ppv": ratio(out["true_alarms"], out["true_alarms"] + out["false_alarms"]),
        "false_alarms_per_hour": ratio(out["false_alarms"], hours),
        "prediction_coverage": ratio(sums.get("eligible_seconds", 0), sums.get("scheduled_seconds", 0)),
        "evaluable_hours": hours})
    return out


TARGETS = {300: {"auroc": .95, "event_sensitivity": .90, "alarm_ppv": .70},
           600: {"auroc": .90, "event_sensitivity": .85, "alarm_ppv": .60}}


def quality_gates(metrics, horizon):
    lower = {**TARGETS[horizon], "prediction_coverage": .90}
    if horizon == 600:
        lower["early_5min_sensitivity"] = .80
    gates = {key: metrics.get(key) is not None and metrics[key] >= target for key, target in lower.items()}
    for key, target in {"false_alarms_per_hour": .5, "ece": .05}.items():
        gates[key] = metrics.get(key) is not None and metrics[key] <= target
    return {"criteria": gates, "all_point_targets_met": all(gates.values()),
            "note": "Proposed research targets; pilot results never establish clinical validation"}


def select_threshold(validation, events, horizon, protocol=Protocol()):
    candidates = []
    for threshold in np.r_[np.linspace(.01, .99, 40), 1.000001]:
        metric, _, _ = evaluate_predictions(validation, events, horizon, threshold, protocol)
        gates = quality_gates(metric, horizon)
        candidates.append((metric, gates))
    feasible = [item for item in candidates if item[1]["all_point_targets_met"]]
    under_budget = [item for item in candidates if item[0]["false_alarms_per_hour"] is not None
                    and item[0]["false_alarms_per_hour"] <= .5]
    pool = feasible or under_budget or candidates
    chosen = max(pool, key=lambda item: (item[0]["event_sensitivity"] or 0,
                                        item[0]["alarm_ppv"] or 0, item[0]["threshold"]))
    return chosen[0]["threshold"], {"selection": "all_targets" if feasible else "exploratory_fallback_targets_unmet",
                                   "metrics": chosen[0], "gates": chosen[1]}


def bootstrap_ci(frame, events, horizon, threshold, protocol=Protocol(), repeats=200):
    """Patient-cluster bootstrap; alarm replay performed once, not on duplicated cases."""
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    _, cases, _ = evaluate_predictions(frame, events, horizon, threshold, protocol)
    subjects = frame.subjectid.unique()
    rng, samples = np.random.default_rng(protocol.seed), []
    for _ in range(repeats):
        selected = rng.choice(subjects, len(subjects), replace=True)
        rows = pd.concat([frame[frame.subjectid.eq(s)] for s in selected], ignore_index=True)
        stats = pd.concat([cases[cases.subjectid.eq(s)] for s in selected], ignore_index=True)
        known = rows.eligible & rows[f"y_{horizon}"].ge(0) & rows.probability.notna()
        m = window_metrics(rows.loc[known, f"y_{horizon}"], rows.loc[known, "probability"])
        m.update(aggregate_cases(stats))
        samples.append(m)
    keys = ["auroc", "average_precision", "brier", "ece", "event_sensitivity", "alarm_ppv",
            "false_alarms_per_hour", "early_5min_sensitivity", "prediction_coverage"]
    result = {}
    for key in keys:
        values = [m[key] for m in samples if m[key] is not None]
        result[key] = {"low": float(np.quantile(values, .025)) if values else None,
                       "high": float(np.quantile(values, .975)) if values else None,
                       "valid_replicates": len(values)}
    return {"unit": "subjectid", "repeats": repeats, "intervals": result}

