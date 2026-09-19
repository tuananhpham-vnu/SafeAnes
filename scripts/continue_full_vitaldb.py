"""E07 correction: training-compatible CSV input, parallel fixed-model inference."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
import io
import json
import shutil
import time

import joblib
import pandas as pd
import evaluate_full_vitaldb as base
from safeanes.data import write_json

ROOT, DATA, REPORT = base.ROOT, base.DATA, base.REPORT
OUT = ROOT / "artifacts/E07b"
MODELS = THRESHOLDS = None


def initialize_models():
    global MODELS, THRESHOLDS
    base.OUT = OUT
    MODELS, THRESHOLDS = base.models_and_thresholds()


def infer_one(path, record):
    cid = int(record["caseid"])
    target = OUT / "cases" / f"{cid}.joblib"
    try:
        normalized = DATA / "csv_cases" / f"{cid}.joblib"
        if not normalized.exists():
            data = joblib.load(path)
            data["frame"] = pd.read_csv(io.StringIO(data["frame"].to_csv(index=False)))
            temp = normalized.with_suffix(".tmp")
            joblib.dump(data, temp, compress=3)
            temp.replace(normalized)
        result = base.predict_case(str(normalized), record, MODELS, THRESHOLDS)
        result["preprocessed_path"] = str(normalized.relative_to(ROOT))
        temp = target.with_suffix(".tmp")
        joblib.dump(result, temp, compress=3)
        temp.replace(target)
        return cid, None
    except Exception as error:
        return cid, repr(error)


def main():
    manifest, selected = base.register()
    for folder in (OUT / "cases", DATA / "csv_cases"):
        folder.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / "docs/experiments/E07_CORRECTION.md", ROOT / "scripts/report_full_vitaldb.py"]
    identity = {"parent_registration_sha256": base.sha(ROOT / "artifacts/E07/registration.json"),
        "sources": {str(p.relative_to(ROOT)): base.sha(p) for p in sources},
        "input_conversion": "build_case -> to_csv(index=False) -> pd.read_csv(default)",
        "build_workers": 8, "inference_workers": 3, "model_threshold_protocol_changes": False}
    registration = OUT / "registration.json"
    if registration.exists():
        assert json.loads(registration.read_text())["identity"] == identity, "Corrected execution changed"
    else:
        write_json(registration, {"created_utc": datetime.now(timezone.utc).isoformat(), "identity": identity})
        for source in sources:
            target = OUT / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    records = {int(r["caseid"]): r for r in selected.to_dict("records")}
    pending = [r for cid, r in records.items() if not (OUT / "cases" / f"{cid}.joblib").exists()]
    completed, errors, start = len(selected)-len(pending), {}, time.monotonic()
    def progress():
        write_json(REPORT / "progress.json", {"execution": "E07b_csv_parity", "total_metadata_cases": len(manifest),
            "eligible_cases": len(selected), "completed_cases": completed, "errors": errors,
            "elapsed_seconds_this_run": time.monotonic()-start, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "final_test_authorized": True, "retuning": False})
    progress()
    print("E07 corrected input registered; completed", completed, "pending", len(pending), flush=True)
    with ProcessPoolExecutor(max_workers=8) as builders, ProcessPoolExecutor(max_workers=3, initializer=initialize_models) as predictors:
        building = {builders.submit(base.build_one, r): int(r["caseid"]) for r in pending}
        predicting = {}
        while building or predicting:
            done, _ = wait(set(building) | set(predicting), timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                if future in building:
                    building.pop(future)
                    cid, path, error = future.result()
                    if error:
                        errors[str(cid)] = error
                        print("ERROR build", cid, error, flush=True)
                    else:
                        predicting[predictors.submit(infer_one, path, records[cid])] = cid
                else:
                    predicting.pop(future)
                    cid, error = future.result()
                    if error:
                        errors[str(cid)] = error
                        print("ERROR predict", cid, error, flush=True)
                    else:
                        completed += 1
                        if completed % 50 == 0:
                            print("E07 corrected", completed, "/", len(selected), "errors", len(errors), flush=True)
                            progress()
    progress()
    if errors or completed != len(selected):
        raise RuntimeError("Incomplete cohort; rerun corrected execution to resume")
    print("E07 corrected full cohort complete", flush=True)


if __name__ == "__main__":
    main()
