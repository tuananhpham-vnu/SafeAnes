import hashlib
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from safeanes.config import Protocol
from safeanes.data import write_json
from safeanes.experiment import run_pilot


def test_end_to_end_pilot_saves_provenance_and_never_reads_global_test(tmp_path):
    """Synthetic integration fixture, never reported as medical performance."""
    data, output = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    protocol = Protocol()
    rows, events = [], []
    for subject in range(20):
        for t in range(600, 2400, 30):
            y = int(1200 <= t < 1500)
            rows.append({"caseid": subject, "subjectid": subject, "time": t,
                "eligible": True, "exposure_seconds": 30,
                "y_300": y, "y_600": int(900 <= t < 1500),
                "map_current": 80 - 10*y + subject/10,
                "map_300_slope": -.01*y, "map_300_std": 2 + y})
        events.append({"caseid": subject, "subjectid": subject, "onset": 1500, "end": 1560,
                       "eligible_300": True, "eligible_600": True})
    pd.DataFrame(rows).to_csv(data / "windows.csv.gz", index=False, compression="gzip")
    pd.DataFrame(events).to_csv(data / "events.csv", index=False)
    pd.DataFrame({"caseid": range(20), "subjectid": range(20), "split": "train"}).to_csv(data / "manifest.csv", index=False)
    write_json(data / "dataset.json", {"scope": "pilot_train_pool_only", "protocol": asdict(protocol),
        "protocol_hash": protocol.digest(), "features": ["map_current", "map_300_slope", "map_300_std"],
        "windows_sha256": hashlib.sha256((data / "windows.csv.gz").read_bytes()).hexdigest()})
    report = run_pilot(data, output, models=["map", "logistic"], repeats=2)
    assert not report["errors"]
    assert len(report["models"]) == 4
    assert (output / "map_300.joblib").exists()
    persisted = json.loads((output / "report.json").read_text())
    assert persisted["scope"] == "exploratory_pilot_not_final_test"
    roles = pd.read_csv(output / "pilot_roles.csv")
    assert roles.groupby("subjectid").role.nunique().max() == 1
    assert set(roles.role) == {"fit", "calibration", "validation", "pilot_test"}
    with pytest.raises(FileExistsError):
        run_pilot(data, output, models=["map"], repeats=2)


def test_optional_lightgbm_estimator():
    pytest.importorskip("lightgbm")
    from safeanes.experiment import make_model
    X = pd.DataFrame({"map_current": np.r_[np.arange(50, 90), np.arange(90, 130)]})
    y = np.r_[np.ones(40), np.zeros(40)]
    model = make_model("lightgbm", 1).fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]
    assert np.isfinite(probabilities).all()
    assert probabilities[:40].mean() > probabilities[40:].mean()


def test_adaptive_threshold_recovers_low_risk_alarm_without_changing_budget():
    from safeanes.threshold_audit import select_audited_threshold
    from safeanes.evaluation import select_threshold
    # A narrow calibrated risk range lies entirely below the historical grid.
    times = np.arange(0, 3600, 30)
    frame = pd.DataFrame({"caseid": 1, "subjectid": 1, "time": times,
        "eligible": True, "exposure_seconds": 30, "y_300": ((times >= 900) & (times < 1200)).astype(int),
        "probability": np.where((times >= 900) & (times < 1200), .005, .001)})
    events = pd.DataFrame({"caseid": [1], "onset": [1200], "eligible_300": [True]})
    old, old_selection = select_threshold(frame, events, 300)
    fixed, fixed_selection, _ = select_audited_threshold(frame, events, 300, Protocol(), False)
    new, selection, _ = select_audited_threshold(frame, events, 300, Protocol(), True)
    assert old == fixed and old_selection["metrics"] == fixed_selection["metrics"]
    assert selection["metrics"]["events_detected"] == 1
    assert selection["metrics"]["false_alarms_per_hour"] == 0
    assert new < .01


def test_expansion_keeps_historical_patients_and_is_stable_when_appending():
    from safeanes.development import expanded_roles
    old = pd.DataFrame({"caseid": [1, 2], "subjectid": [11, 22], "role": ["pilot_test", "validation"]})
    manifest = pd.DataFrame({"caseid": [1, 2, 3, 4], "subjectid": [11, 22, 11, 33], "split": "train"})
    roles = expanded_roles(manifest, old)
    assert roles.loc[roles.subjectid.eq(11), "role"].eq("pilot_test").all()
    larger = pd.concat([manifest, pd.DataFrame({"caseid": [5], "subjectid": [44], "split": "train"})])
    pd.testing.assert_frame_equal(roles, expanded_roles(larger, old).iloc[:4])
    with pytest.raises(ValueError):
        expanded_roles(manifest.assign(split="test"), old)


def test_cached_replay_matches_reference_with_gaps_censoring_and_cooldown():
    from safeanes.fast_selection import prepare_validation, cached_metrics
    from safeanes.evaluation import evaluate_predictions
    rng = np.random.default_rng(42)
    times = np.r_[np.arange(0, 1200, 30), np.arange(1500, 3600, 30)]
    frame = pd.DataFrame({"caseid": np.repeat([1, 2], len(times)), "subjectid": np.repeat([10, 20], len(times)),
        "time": np.tile(times, 2), "eligible": rng.random(2*len(times)) > .15,
        "exposure_seconds": 30, "y_300": rng.choice([-1, 0, 1], 2*len(times)),
        "probability": rng.choice([.01, .1, .5, .9, np.nan], 2*len(times))})
    events = pd.DataFrame({"caseid": [1, 1, 2], "onset": [900, 2700, 2100], "eligible_300": [True, False, True]})
    base, prepared = prepare_validation(frame, events, 300, Protocol())
    for threshold in (.01, .1, .5, .9, 1.000001):
        reference, _, _ = evaluate_predictions(frame, events, 300, threshold)
        assert cached_metrics(base, prepared, 300, threshold, Protocol()) == reference


@pytest.mark.parametrize("architecture", ["inception", "timesnet"])
def test_version03_models_are_independent_of_other_patients_at_inference(architecture):
    torch = pytest.importorskip("torch")
    from safeanes.models import NumericModel, masked_bce
    torch.set_num_threads(2)
    torch.manual_seed(123)
    model = NumericModel(architecture, 21, 6, width=4).eval()
    x, static = torch.randn(3, 21, 300), torch.randn(3, 6)
    together = model(x, static)
    alone = model(x[:1], static[:1])
    torch.testing.assert_close(alone, together[:1], rtol=1e-5, atol=1e-6)
    assert torch.all(together[:, 1] >= together[:, 0])
    # The all-zero signal has no nonzero spectral peak and must remain finite.
    assert torch.isfinite(model(torch.zeros_like(x), torch.zeros_like(static))).all()
    model.train()
    loss = masked_bce(model(x, static), torch.tensor([[0., 1.], [-1., 0.], [1., 1.]]))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_version03_blend_preserves_order_and_rejects_invalid_weights():
    pytest.importorskip("torch")
    from safeanes.ensemble import blend_logits
    scores = np.array([[[-3., -1.], [1., 2.]], [[-1., 0.], [-1., 3.]]])
    np.testing.assert_equal(blend_logits(scores, [.5, .5]), [[-2., -.5], [0., 2.5]])
    with pytest.raises(ValueError):
        blend_logits(scores, [-.1, 1.1])
    with pytest.raises(ValueError):
        blend_logits(scores, [.2, .2])
    scores[0, 0] = [1., -1.]
    with pytest.raises(ValueError, match="horizon"):
        blend_logits(scores, [.5, .5])


def test_version03_training_and_ensemble_on_synthetic_fixture(tmp_path):
    torch = pytest.importorskip("torch")
    from safeanes.config import TRACKS
    from safeanes.ensemble import run_ensemble
    from safeanes.sequences import file_hash
    from safeanes.training import TrainConfig, train_sequence
    from dataclasses import replace
    data, sequences = tmp_path / "data", tmp_path / "sequences"
    data.mkdir()
    sequences.mkdir()
    protocol = Protocol()
    rows, event_rows, records = [], [], {}
    for caseid in range(20):
        for t in range(600, 1800, 30):
            rows.append({"caseid": caseid, "subjectid": caseid, "time": t, "eligible": True,
                "exposure_seconds": 30, "y_300": int(900 <= t < 1200), "y_600": int(600 <= t < 1200),
                "static_age": 40 + caseid, "static_bmi": 24., "static_asa": 2.})
        event_rows.append({"caseid": caseid, "subjectid": caseid, "onset": 1200, "end": 1260,
                           "eligible_300": True, "eligible_600": True})
        array = np.zeros((900, len(TRACKS) * 2), np.float32)
        array[:, :len(TRACKS)] = 80 + caseid
        array[:, 0] += np.sin(np.arange(900) / 20)
        path = sequences / f"{caseid}.npy"
        np.save(path, array)
        records[str(caseid)] = {"start": 0., "steps": 900, "subjectid": caseid, "sha256": file_hash(path)}
    pd.DataFrame(rows).to_csv(data / "windows.csv.gz", index=False)
    pd.DataFrame(event_rows).to_csv(data / "events.csv", index=False)
    pd.DataFrame({"caseid": range(20), "subjectid": range(20), "split": "train"}).to_csv(data / "manifest.csv", index=False)
    meta = {"scope": "pilot_train_pool_only", "protocol": asdict(protocol), "protocol_hash": protocol.digest(),
            "windows_sha256": file_hash(data / "windows.csv.gz")}
    write_json(data / "dataset.json", meta)
    write_json(sequences / "sequences.json", {**meta, "tracks": list(TRACKS), "cases": records,
        "manifest_sha256": file_hash(data / "manifest.csv")})
    config = TrainConfig(architecture="inception", width=4, epochs=1, max_windows=32, batch_size=32, threads=2, device="cpu")
    runs = [tmp_path / "inception", tmp_path / "timesnet"]
    for architecture, run in zip(("inception", "timesnet"), runs):
        train_sequence(data, sequences, run, replace(config, architecture=architecture), repeats=2)
    output = tmp_path / "ensemble"
    result = run_ensemble(data, sequences, runs, output, repeats=2, threads=2)
    assert not result["errors"] and result["monotonicity_violations"] == 0
    assert set(result["models"]) == {"ensemble_300", "ensemble_600"}
    assert result["weights"] == [.5, .5]
    bundle = torch.load(output / "ensemble.pt", weights_only=True)
    assert len(bundle["components"]) == 2
    frame = pd.read_csv(output / "ensemble_300_predictions.csv.gz")
    assert frame.role.eq("pilot_test").all()
    assert (frame.p_600 >= frame.p_300).all()
    with pytest.raises(FileExistsError):
        run_ensemble(data, sequences, runs, output, repeats=2)
    checkpoint = torch.load(runs[0] / "model.pt", weights_only=True)
    checkpoint["identity"]["dataset_hash"] = "wrong dataset"
    torch.save(checkpoint, runs[0] / "model.pt")
    with pytest.raises(ValueError, match="mismatch"):
        run_ensemble(data, sequences, runs, tmp_path / "invalid", repeats=2)
