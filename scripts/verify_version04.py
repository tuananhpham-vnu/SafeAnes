"""Check saved models/provenance and disclose independent-horizon violations."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from safeanes.data import write_json
from safeanes.experiment import raw_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "artifacts/v0_4"
    registration = json.loads((out / "registration.json").read_text())["identity"]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    current_differences = []
    for path, expected in registration["sources"].items():
        assert digest(out / "source" / path) == expected, path
        if not (ROOT / path).exists() or digest(ROOT / path) != expected:
            current_differences.append(path)
    for path, expected in registration["inputs"].items():
        assert digest(ROOT / path) == expected, path
    assert digest(ROOT / "data/pilot_v1/windows.csv.gz") == registration["dataset_hash"]
    records = []
    for run in sorted(out.glob("catboost_*")):
        predictions = []
        parameters = {}
        for horizon in (300, 600):
            saved = joblib.load(run / f"{horizon}_model.joblib")
            parameters[str(horizon)] = saved["model"].named_steps["catboostclassifier"].get_all_params()
            frame = pd.read_csv(run / f"{horizon}_test.csv.gz")
            valid = frame[frame.eligible]
            p = saved["calibrator"].predict_proba(raw_score(run.name, saved["model"], valid, saved["features"]).reshape(-1, 1))[:, 1]
            np.testing.assert_allclose(p, valid.probability, rtol=1e-10, atol=1e-12)
            assert frame.role.eq("pilot_test").all()
            predictions.append(valid[["caseid", "time", "probability"]])
        paired = predictions[0].merge(predictions[1], on=["caseid", "time"], suffixes=("_300", "_600"), validate="one_to_one")
        records.append({"run": run.name, "prediction_reproduction": "passed",
            "eligible_windows": len(paired), "p600_below_p300": int((paired.probability_600 < paired.probability_300).sum()),
            "effective_model_parameters": parameters})
    assert len(records) == 6
    write_json(ROOT / "reports/v0_4/verification.json", {"source_and_input_hashes": "frozen snapshot and original inputs passed",
        "current_source_differences": current_differences, "release": "0.2.0", "experiment": "E04", "runs": records,
        "note": "Separate horizon classifiers do not enforce p600 >= p300. Do not claim ordered risks."})
    print(json.dumps([{k: v for k, v in row.items() if k != "effective_model_parameters"} for row in records], indent=2))


if __name__ == "__main__":
    main()
