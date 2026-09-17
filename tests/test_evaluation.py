import numpy as np
import pandas as pd
import pytest

from safeanes.config import Protocol
from safeanes.evaluation import (bootstrap_ci, evaluate_predictions, quality_gates,
                                replay_alarms, select_threshold, window_metrics)


def frame(probabilities, times=None):
    n = len(probabilities)
    return pd.DataFrame({"caseid": 1, "subjectid": 9, "time": np.arange(n) * 30 if times is None else times,
        "eligible": True, "probability": probabilities, "y_300": 0, "y_600": 0, "exposure_seconds": 30})


def events(onsets=()):
    return pd.DataFrame({"caseid": [1] * len(onsets), "subjectid": [9] * len(onsets),
        "onset": list(onsets), "end": [x + 60 for x in onsets],
        "eligible_300": [True] * len(onsets), "eligible_600": [True] * len(onsets)})


def test_persistence_groups_sustained_alarm_and_cooldown():
    f = frame([.9] * 20)
    assert replay_alarms(f, .5) == [30]
    f.loc[2:3, "probability"] = .1
    assert replay_alarms(f, .5) == [30, 330]


def test_missing_and_timestamp_gap_reset_persistence():
    assert replay_alarms(frame([.9, .9], times=[0, 60]), .5) == []
    f = frame([.9, .9, .9])
    f.loc[1, "eligible"] = False
    assert replay_alarms(f, .5) == []


def test_event_matching_uses_actual_alarm_time_and_does_not_count_event_twice():
    f = frame([.9] * 30)
    f["y_600"] = (f.time < 600).astype(int)
    metric, _, alarms = evaluate_predictions(f, events([600]), 600, .5)
    assert metric["events_detected"] == 1
    assert metric["true_alarms"] == 1
    assert metric["lead_seconds_median"] == 570
    assert metric["early_5min_sensitivity"] == 1
    assert alarms[0]["time"] == 30


def test_false_alarms_rate_and_single_class_metrics():
    f = frame([.9] * 120)
    metric, _, _ = evaluate_predictions(f, events(), 300, .5)
    assert metric["false_alarms_per_hour"] == 1
    assert metric["alarm_ppv"] == 0
    assert metric["auroc"] is None
    assert metric["event_sensitivity"] is None


def test_unknown_future_does_not_prevent_emission_but_censors_scoring():
    f = frame([.9] * 5)
    f["y_300"] = -1
    assert replay_alarms(f, .5) == [30]
    metric, _, _ = evaluate_predictions(f, events(), 300, .5)
    assert metric["censored_alarms"] == 1
    assert metric["false_alarms"] == 0
    assert metric["false_alarms_per_hour"] is None


def test_no_alerts_cannot_pass_ppv_or_quality_gates():
    metric, _, _ = evaluate_predictions(frame([.1] * 120), events(), 300, .5)
    assert metric["alarm_ppv"] is None
    assert not quality_gates(metric, 300)["all_point_targets_met"]


def test_probability_validation_and_perfect_calibration():
    result = window_metrics([0, 1], [0, 1])
    assert result["ece"] == 0
    assert result["brier"] == 0
    assert result["auroc"] == 1
    with pytest.raises(ValueError):
        window_metrics([0, 1], [.2, 2])


def test_threshold_fallback_is_explicit_and_ci_cluster_unit():
    f = frame([.1] * 120)
    threshold, choice = select_threshold(f, events(), 300)
    assert choice["selection"] == "exploratory_fallback_targets_unmet"
    ci = bootstrap_ci(f, events(), 300, threshold, repeats=3)
    assert ci["unit"] == "subjectid"
    assert ci["intervals"]["auroc"]["valid_replicates"] == 0

