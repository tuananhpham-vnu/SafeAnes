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

