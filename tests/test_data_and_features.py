import gzip
import numpy as np
import pandas as pd
import pytest

from safeanes.config import Protocol, TRACKS
from safeanes.data import cohort_manifest, decode_csv, read_numeric, subject_bucket
from safeanes.dataset import build_case, feature_columns
from safeanes.experiment import pilot_roles


def test_gzip_api_and_duplicate_timestamp():
    data = decode_csv(gzip.compress(b"Time,Value\n0,80\n2,60\n2,70\n"))
    times, values = read_numeric(data)
    np.testing.assert_equal(times, [0, 2])
    np.testing.assert_equal(values, [80, 70])


def test_reject_waveform_style_missing_timestamps():
    with pytest.raises(ValueError):
        read_numeric(pd.DataFrame({"Time": [0, .002, np.nan], "Value": [80, 81, 82]}))


def test_subject_split_stable_when_cohort_grows():
    first = {i: subject_bucket(i) for i in range(100)}
    grown = {i: subject_bucket(i) for i in range(150)}
    assert all(first[i] == grown[i] for i in first)
    assert subject_bucket(12.) == subject_bucket(12)
    with pytest.raises(ValueError):
        subject_bucket(np.nan)


def test_cohort_filters_and_missing_optional_tracks():
    cases = pd.DataFrame({"caseid": [1, 2, 3], "subjectid": [10, 10, 20],
        "age": [50, 17, 60], "ane_type": ["General"] * 3,
        "opstart": [0] * 3, "opend": [4000] * 3})
    tracks = pd.DataFrame({"caseid": [1, 2], "tname": [TRACKS["map"]] * 2, "tid": ["a", "b"]})
    out = cohort_manifest(cases, tracks)
    assert list(out.eligible) == [True, False, False]
    assert out.iloc[0].split == out.iloc[1].split
    assert out.iloc[0].tid_hr is np.nan or pd.isna(out.iloc[0].tid_hr)


def test_grouped_pilot_partition_never_splits_repeat_surgeries():
    manifest = pd.DataFrame({"caseid": range(40), "subjectid": np.repeat(range(20), 2), "split": "train"})
    assigned = pilot_roles(manifest, 1)
    assert assigned.groupby("subjectid").role.nunique().max() == 1
    assert set(assigned.role) == {"fit", "calibration", "validation", "pilot_test"}
    manifest.loc[0, "split"] = "test"
    with pytest.raises(ValueError):
        pilot_roles(manifest, 1)


def example_case():
    meta = {"caseid": 1, "subjectid": 1, "opstart": 0, "opend": 2400,
            "age": 50, "bmi": 24, "asa": 2}
    times = np.arange(0, 2401, 2)
    values = np.full(len(times), 80.)
    return meta, times, values


def test_future_perturbation_changes_labels_not_features_or_eligibility():
    meta, times, values = example_case()
    first, _, _ = build_case(meta, {"map": (times, values)})
    values[(times >= 800) & (times < 900)] = 60
    second, _, _ = build_case(meta, {"map": (times, values)})
    cols = feature_columns(first) + ["eligible", "history_coverage"]
    pd.testing.assert_frame_equal(first.loc[first.time <= 780, cols], second.loc[second.time <= 780, cols])
    assert first.loc[first.time.eq(600), "y_300"].item() == 0
    assert second.loc[second.time.eq(600), "y_300"].item() == 1
    assert not second.loc[second.time.eq(810), "eligible"].item()


def test_future_labels_and_identifiers_not_features():
    columns = pd.DataFrame(columns=["map_current", "static_age", "y_300", "subjectid", "caseid", "time", "exposure_seconds"])
    assert feature_columns(columns) == ["map_current", "static_age"]

