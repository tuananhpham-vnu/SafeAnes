"""Equivalent alarm-policy search with arrays cached once per validation cohort."""
import numpy as np
from .evaluation import evaluate_predictions, quality_gates
from .threshold_audit import threshold_grid


def prepare_validation(frame, events, horizon, protocol):
    baseline, _, _ = evaluate_predictions(frame, events, horizon, 1.000001, protocol)
    prepared = []
    for caseid, group in frame.groupby("caseid", sort=False):
        group = group.sort_values("time")
        ev = events[events.caseid.eq(caseid)].sort_values("onset")
        prepared.append((list(zip(group.time.to_numpy(), group.eligible.to_numpy(),
            group.probability.to_numpy(), group[f"y_{horizon}"].to_numpy())),
            ev.onset.to_numpy(), set(ev.loc[ev[f"eligible_{horizon}"].astype(bool), "onset"])))
    return baseline, prepared


def cached_metrics(baseline, prepared, horizon, threshold, protocol):
    true = false = censored = detected = early = 0
    leads = []
    for rows, onsets, eligible_onsets in prepared:
        streak, active, previous, next_allowed = 0, False, -np.inf, -np.inf
        best_leads = {}
        for time, eligible, probability, label in rows:
            if time - previous > protocol.cadence_seconds:
                streak, active = 0, False
            previous = time
            if not eligible or not np.isfinite(probability) or probability < threshold:
                streak, active = 0, False
                continue
            streak += 1
            if active or streak < protocol.alarm_persistence or time < next_allowed:
                continue
            active, next_allowed = True, time + protocol.alarm_cooldown_seconds
            if label < 0:
                censored += 1
                continue
            index = np.searchsorted(onsets, time, side="right")
            if index < len(onsets) and onsets[index] <= time + horizon:
                true += 1
                onset = onsets[index]
                if onset in eligible_onsets:
                    best_leads[onset] = max(best_leads.get(onset, 0), float(onset-time))
            else:
                false += 1
        detected += len(best_leads)
        early += sum(lead >= 300 for lead in best_leads.values())
        leads.extend(best_leads.values())
    m = dict(baseline)
    ratio = lambda a, b: float(a/b) if b else None
    m.update(events_detected=detected, events_early_5min=early, true_alarms=true, false_alarms=false,
        censored_alarms=censored, event_sensitivity=ratio(detected, m["events_eligible"]),
        event_sensitivity_all=ratio(detected, m["events_all"]), early_5min_sensitivity=ratio(early, m["events_eligible"]),
        alarm_ppv=ratio(true, true+false), false_alarms_per_hour=ratio(false, m["evaluable_hours"]),
        threshold=float(threshold), lead_seconds_median=float(np.median(leads)) if leads else None,
        lead_seconds_q25=float(np.quantile(leads, .25)) if leads else None,
        lead_seconds_q75=float(np.quantile(leads, .75)) if leads else None)
    return m


def select_fast(validation, events, horizon, protocol, adaptive=False):
    baseline, prepared = prepare_validation(validation, events, horizon, protocol)
    curve = []
    for threshold in threshold_grid(validation, adaptive):
        m = cached_metrics(baseline, prepared, horizon, threshold, protocol)
        curve.append({"metrics": m, "gates": quality_gates(m, horizon)})
    feasible = [x for x in curve if x["gates"]["all_point_targets_met"]]
    budget = [x for x in curve if x["metrics"]["false_alarms_per_hour"] is not None and x["metrics"]["false_alarms_per_hour"] <= .5]
    chosen = max(feasible or budget or curve, key=lambda x: (
        x["metrics"]["event_sensitivity"] or 0, x["metrics"]["alarm_ppv"] or 0, x["metrics"]["threshold"]))
    return chosen["metrics"]["threshold"], {**chosen, "selection": "all_targets" if feasible else "exploratory_fallback_targets_unmet"}, curve
