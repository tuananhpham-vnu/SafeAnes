import numpy as np
import pytest

from safeanes.config import Protocol
from safeanes.signals import Event, causal_sample, detect_events, future_label, label_grid


def test_causal_sample_never_uses_future_and_expires():
    sampled, ages = causal_sample([10, 20, 100], [80, 60, 200], [0, 10, 19, 20, 51], 30)
    np.testing.assert_allclose(sampled, [np.nan, 80, 80, 60, np.nan], equal_nan=True)
    assert ages[-1] == 31


def test_invalid_observation_does_not_resurrect_previous_value():
    values, _ = causal_sample([0, 2, 4], [80, np.nan, 70], [1, 2, 3, 4], 30)
    np.testing.assert_allclose(values, [80, np.nan, np.nan, 70], equal_nan=True)


def test_long_gap_not_bridged_by_label_reconstruction():
    truth = label_grid([0, 2, 90, 92], [60, 60, 60, 60], np.arange(93), 10)
    assert np.isnan(truth[2:90]).all()
    assert np.isnan(truth[-1])
    assert detect_events(np.arange(93), truth) == []


def test_fractional_timestamps_do_not_destroy_continuous_coverage():
    times = np.arange(0.3, 200, 2)
    truth = label_grid(times, np.full(len(times), 60), np.arange(200), 10)
    assert np.isfinite(truth[1:197]).all()
    events = detect_events(np.arange(200), truth)
    assert len(events) == 1
    assert events[0].onset == 1


def test_subsecond_reading_crossing_threshold_breaks_low_duration():
    truth = label_grid([0, .5, 1, 2], [60, 80, 60, 60], [0, 1], 10)
    np.testing.assert_equal(truth, [80, 60])


@pytest.mark.parametrize("duration,expected", [(59, 0), (60, 1), (61, 1)])
def test_minimum_event_duration(duration, expected):
    truth = np.r_[np.full(10, 80), np.full(duration, 60), np.full(10, 80)]
    events = detect_events(np.arange(len(truth)), truth)
    assert len(events) == expected


def test_exact_threshold_is_not_low():
    assert detect_events(np.arange(100), np.full(100, 65)) == []


def test_recurrent_events_merge_only_within_recovery_without_missing():
    truth = np.r_[np.full(60, 60.), np.full(60, 80.), np.full(60, 60.)]
    assert detect_events(np.arange(180), truth) == [Event(0, 180)]
    truth[90] = np.nan
    assert len(detect_events(np.arange(180), truth)) == 2


def test_horizon_boundary_and_nested_labels():
    grid, truth, events = np.arange(1500), np.full(1500, 80.), [Event(600, 660)]
    assert future_label(300, 300, grid, truth, events) == 1
    assert future_label(299, 300, grid, truth, events) == 0
    assert future_label(299, 600, grid, truth, events) == 1
    assert future_label(600, 300, grid, truth, events) == 0


def test_missing_future_and_right_censoring_are_not_negative():
    grid, truth = np.arange(1000), np.full(1000, 80.)
    assert future_label(700, 300, grid, truth, []) == -1
    truth[500] = np.nan
    assert future_label(300, 300, grid, truth, [Event(400, 460)]) == -1


def test_bad_times_fail():
    with pytest.raises(ValueError):
        causal_sample([0, 0], [80, 90], [1], 10)
    with pytest.raises(ValueError):
        Protocol(numeric_step_seconds=7)
