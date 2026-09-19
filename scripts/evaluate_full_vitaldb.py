"""E07: full eligible VitalDB, fixed models, subject-isolated final-test reporting."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd

from safeanes.config import Protocol, TRACKS
from safeanes.data import fetch_csv, read_numeric, write_json
from safeanes.dataset import build_case

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/vitaldb_full"
OUT = ROOT / "artifacts/E07"
REPORT = ROOT / "reports/E07"
NAMES = ("tabm_ensemble", "catboost_numeric", "catboost_all", "map", "lightgbm_monotone")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_one(record):
    """Workers import no model libraries; original preprocessing, no approximation."""
    import joblib
    caseid = int(record["caseid"])
    target = DATA / "cases" / f"{caseid}.joblib"
    if target.exists():
        return caseid, str(target), None
    try:
        raw = {}
        for name in TRACKS:
            tid = record[f"tid_{name}"]
            if pd.isna(tid):
                continue
            cache = ROOT / "data/vitaldb_development300/raw"
            if not (cache / f"{tid}.csv.gz").exists():
                cache = DATA / "raw"
            raw[name] = read_numeric(fetch_csv(str(tid), cache))
        frame, events, audit = build_case(record, raw, Protocol())
        if frame.empty:
            raise ValueError("No scheduled windows for eligible case")
        payload = {"frame": frame, "events": events, "audit": audit}
        temp = target.with_suffix(".tmp")
        joblib.dump(payload, temp, compress=3)
        temp.replace(target)
        return caseid, str(target), None
    except Exception as error:
        return caseid, None, repr(error)


def register():
    for folder in (DATA / "raw", DATA / "cases", OUT / "cases", REPORT):
        folder.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data/vitaldb_development300/cohort_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    roles_path = ROOT / "artifacts/E06/roles.csv"
    roles = pd.read_csv(roles_path).drop_duplicates("subjectid").set_index("subjectid")
    manifest["evaluation_group"] = np.where(manifest.subjectid.isin(roles.index), "development_seen", "unseen_" + manifest.split)
    manifest["development_role"] = manifest.subjectid.map(roles.role)
    selected = manifest[manifest.eligible].copy()
    assert selected.groupby("subjectid").evaluation_group.nunique().eq(1).all()
    model_paths = [ROOT / "artifacts/E06" / name / "bundle.joblib" for name in
                   ("tabm_20260917", "tabm_20260918", "tabm_20260919", "lightgbm_monotone")]
    for name in ("catboost_numeric", "catboost_all", "map_all"):
        for h in (300, 600):
            model_paths.append(ROOT / "artifacts/E05" / f"{name}_20260917_{h}" / "model.joblib")
    selection_paths = [ROOT / "artifacts/E06/validation_selection.json", ROOT / "reports/E05/comparison.json"]
    inputs = [manifest_path, roles_path, *model_paths, *selection_paths,
              ROOT / "data/vitaldb_development300/raw/cases.csv.gz", ROOT / "data/vitaldb_development300/raw/trks.csv.gz"]
    sources = [Path(__file__), ROOT / "docs/experiments/E07_PLAN.md", *sorted((ROOT / "src/safeanes").glob("*.py"))]
    identity = {"experiment": "E07", "release": "0.2.0", "protocol_hash": Protocol().digest(),
        "inputs": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in sources}, "models": list(NAMES),
        "primary": "tabm_ensemble minus catboost_numeric on unseen_test; fixed thresholds",
        "authorization": "User: Danh gia toan bo VitalDB di; includes global test, no retuning"}
    registration = OUT / "registration.json"
    if registration.exists():
        assert json.loads(registration.read_text())["identity"] == identity, "Frozen E07 changed"
    else:
        write_json(registration, {"created_utc": datetime.now(timezone.utc).isoformat(), "identity": identity})
        for source in sources:
            target = OUT / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    manifest.to_csv(REPORT / "cohort_manifest.csv", index=False)
    selected.to_csv(DATA / "eligible_manifest.csv", index=False)
    return manifest, selected


def models_and_thresholds():
    import joblib
    import torch
    torch.set_num_threads(2)
    models = {name: joblib.load(ROOT / "artifacts/E06" / name / "bundle.joblib") for name in
              ("tabm_20260917", "tabm_20260918", "tabm_20260919", "lightgbm_monotone")}
    old = json.loads((ROOT / "reports/E05/comparison.json").read_text())
    new = json.loads((ROOT / "artifacts/E06/validation_selection.json").read_text())["selections"]
    thresholds = {}
    for name in NAMES:
        for h in (300, 600):
            if name in ("tabm_ensemble", "lightgbm_monotone"):
                thresholds[(name, h)] = new[f"{name}_{h}_fixed"]["metrics"]["threshold"]
            else:
                prefix = "map_all" if name == "map" else name
                key = f"{prefix}_20260917_{h}"
                models[(name, h)] = joblib.load(ROOT / "artifacts/E05" / key / "model.joblib")
                thresholds[(name, h)] = old[key]["fixed"]["selection_on_validation"]["metrics"]["threshold"]
    write_json(OUT / "thresholds.json", {f"{n}_{h}": v for (n, h), v in thresholds.items()})
    return models, thresholds


def predict_case(path, record, models, thresholds):
    import joblib
    from safeanes.tabular_sota import calibrated_risk
    from safeanes.experiment import raw_score
    from safeanes.evaluation import evaluate_predictions
    data = joblib.load(path)
    frame = data["frame"]
    eligible = frame.eligible.to_numpy()
    x = frame[frame.eligible]
    probabilities = {}
    for name in NAMES:
        p = np.full((len(frame), 2), np.nan)
        if len(x):
            if name == "tabm_ensemble":
                p[eligible] = np.mean([calibrated_risk(models[f"tabm_{s}"], x) for s in (20260917, 20260918, 20260919)], axis=0)
            elif name == "lightgbm_monotone":
                p[eligible] = calibrated_risk(models[name], x)
            else:
                for i, h in enumerate((300, 600)):
                    b = models[(name, h)]
                    raw = raw_score(b["name"], b["model"], x, b["features"])
                    p[eligible, i] = b["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]
            assert np.isfinite(p[eligible]).all()
        probabilities[name] = p
    events = pd.DataFrame(data["events"], columns=["caseid", "subjectid", "onset", "end", "eligible_300", "eligible_600"])
    slim = frame[["caseid", "subjectid", "time", "eligible", "exposure_seconds", "y_300", "y_600"]].copy()
    stats = []
    for name in NAMES:
        for i, h in enumerate((300, 600)):
            slim["probability"] = probabilities[name][:, i]
            _, cases, alarms = evaluate_predictions(slim, events, h, thresholds[(name, h)], Protocol())
            row = cases.iloc[0].to_dict()
            stats.append({"model": name, "horizon": h, **row})
    slim.drop(columns="probability", inplace=True)
    return {"frame": slim, "probabilities": probabilities, "case_stats": stats, "events": data["events"],
            "audit": data["audit"], "group": record["evaluation_group"], "preprocessed_sha256": sha(Path(path))}


def main():
    import joblib
    manifest, selected = register()
    models, thresholds = models_and_thresholds()
    records = {int(r["caseid"]): r for r in selected.to_dict("records")}
    pending = [r for cid, r in records.items() if not (OUT / "cases" / f"{cid}.joblib").exists()]
    completed, errors, start = len(selected) - len(pending), {}, time.monotonic()
    print("E07 registered", len(manifest), "total", len(selected), "eligible", completed, "completed", flush=True)
    def progress():
        write_json(REPORT / "progress.json", {"total_metadata_cases": len(manifest), "eligible_cases": len(selected),
            "completed_cases": completed, "errors": errors, "elapsed_seconds_this_run": time.monotonic()-start,
            "updated_utc": datetime.now(timezone.utc).isoformat(), "final_test_authorized": True})
    progress()
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(build_one, r): int(r["caseid"]) for r in pending}
        for future in as_completed(futures):
            caseid, path, error = future.result()
            if error:
                errors[str(caseid)] = error
                progress()
                print("ERROR", caseid, error, flush=True)
                continue
            target = OUT / "cases" / f"{caseid}.joblib"
            try:
                result = predict_case(path, records[caseid], models, thresholds)
                temp = target.with_suffix(".tmp")
                joblib.dump(result, temp, compress=3)
                temp.replace(target)
                completed += 1
            except Exception as error:
                errors[str(caseid)] = repr(error)
            if completed % 25 == 0 or errors:
                progress()
                print("E07", completed, "/", len(selected), "errors", len(errors), flush=True)
    progress()
    if errors or completed != len(selected):
        raise RuntimeError("Incomplete cohort; rerun to resume; see progress.json")
    print("All eligible cases processed. Run scripts/report_full_vitaldb.py", flush=True)


if __name__ == "__main__":
    main()
