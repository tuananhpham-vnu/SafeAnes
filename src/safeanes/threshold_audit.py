"""Versioned threshold ablation; historical evaluator remains unchanged."""
import numpy as np
from .evaluation import evaluate_predictions, quality_gates


def threshold_grid(validation, adaptive):
    fixed = np.r_[np.linspace(.01, .99, 40), 1.000001]
    p = validation.loc[validation.eligible, "probability"].to_numpy(float)
    p = p[np.isfinite(p)]
    if not adaptive or not len(p):
        return fixed
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities")
    return np.unique(np.r_[fixed, np.quantile(p, np.linspace(0, 1, 201))])


def select_audited_threshold(validation, events, horizon, protocol, adaptive=True):
    candidates = []
    for threshold in threshold_grid(validation, adaptive):
        metric, _, _ = evaluate_predictions(validation, events, horizon, threshold, protocol)
        candidates.append({"metrics": metric, "gates": quality_gates(metric, horizon)})
    feasible = [x for x in candidates if x["gates"]["all_point_targets_met"]]
    budget = [x for x in candidates if x["metrics"]["false_alarms_per_hour"] is not None
              and x["metrics"]["false_alarms_per_hour"] <= .5]
    chosen = max(feasible or budget or candidates, key=lambda x: (
        x["metrics"]["event_sensitivity"] or 0, x["metrics"]["alarm_ppv"] or 0,
        x["metrics"]["threshold"]))
    selection = {**chosen, "selection": "all_targets" if feasible else "exploratory_fallback_targets_unmet",
                 "policy": "validation_quantiles_201" if adaptive else "historical_40",
                 "candidate_count": len(candidates)}
    return chosen["metrics"]["threshold"], selection, candidates
