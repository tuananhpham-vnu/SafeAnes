"""E05: registered 300-case development, immutable old roles, fresh subset report."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import time
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from safeanes.config import Protocol
from safeanes.data import write_json
from safeanes.development import expanded_roles
from safeanes.experiment import make_model, raw_score
from safeanes.evaluation import evaluate_predictions, bootstrap_ci, quality_gates
from safeanes.fast_selection import select_fast

ROOT = Path(__file__).resolve().parents[1]
MATRIX = [(name, "all", 20260917) for name in ("map", "logistic", "lightgbm")] + [
    ("catboost", "all", seed) for seed in (20260917, 20260918, 20260919)] + [
    ("catboost", scope, 20260917) for scope in ("map_only", "numeric")]


def main():
    data, out, reports = ROOT / "data/development300", ROOT / "artifacts/E05", ROOT / "reports/E05"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / "dataset.json").read_text())
    protocol = Protocol(**{**meta["protocol"], "horizons_seconds": tuple(meta["protocol"]["horizons_seconds"])})
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(data / "windows.csv.gz") == meta["windows_sha256"]
    assert protocol.digest() == meta["protocol_hash"]
    roles = pd.read_csv(ROOT / "data/vitaldb_development300/development_roles.csv")
    manifest = pd.read_csv(data / "manifest.csv")
    old = pd.read_csv(ROOT / "artifacts/tcn_v1/pilot_roles.csv")
    pd.testing.assert_frame_equal(roles, expanded_roles(manifest, old).reset_index(drop=True))
    sources = [Path(__file__), ROOT / "docs/experiments/E05_PLAN.md", *sorted((ROOT / "src/safeanes").glob("*.py"))]
    identity = {"release": "0.2.0", "experiment": "E05", "dataset_hash": meta["windows_sha256"],
        "roles_hash": digest(ROOT / "data/vitaldb_development300/development_roles.csv"), "matrix": [list(x) for x in MATRIX],
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "packages": {p: importlib.metadata.version(p) for p in ("catboost", "lightgbm", "numpy", "pandas", "scikit-learn")}}
    if (out / "registration.json").exists():
        assert json.loads((out / "registration.json").read_text())["identity"] == identity, "Frozen experiment changed"
    else:
        write_json(out / "registration.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "identity": identity})
        for p in sources:
            target = out / "source" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    roles.to_csv(out / "roles.csv", index=False)
    frame = pd.read_csv(data / "windows.csv.gz").merge(roles, on=["caseid", "subjectid"], validate="many_to_one")
    events = pd.read_csv(data / "events.csv")
    support = []
    for (role, historical), group in frame.groupby(["role", "historical_subject"]):
        for h in (300, 600):
            known = group[group.eligible & group[f"y_{h}"].ge(0)]
            ev = events[events.caseid.isin(group.caseid.unique())]
            support.append({"role": role, "historical": bool(historical), "horizon": h,
                "subjects": int(group.subjectid.nunique()), "windows": len(group), "known_eligible": len(known),
                "positive": int(known[f"y_{h}"].sum()), "eligible_events": int(ev[f"eligible_{h}"].sum())})
    write_json(reports / "data_support.json", support)
    # Metadata audit does not infer clinical exclusions from outcomes.
    audit = manifest[["caseid", "subjectid", "department", "optype", "opname", "ane_type"]].copy()
    audit["cardiac_name_review"] = audit.opname.fillna("").str.contains(r"cardiac|coronary|cardiopulmonary|heart|valve|CABG", case=False, regex=True)
    audit.to_csv(reports / "cohort_audit.csv", index=False)
    result = {}
    for name, feature_set, seed in MATRIX:
        for h in (300, 600):
            key = f"{name}_{feature_set}_{seed}_{h}"
            directory = out / key
            directory.mkdir(exist_ok=True)
            if (directory / "results.json").exists():
                result[key] = json.loads((directory / "results.json").read_text())
                continue
            started = time.perf_counter()
            label = f"y_{h}"
            fit = frame[frame.role.eq("fit") & frame.eligible & frame[label].ge(0)]
            cal = frame[frame.role.eq("calibration") & frame.eligible & frame[label].ge(0)]
            assert fit[label].nunique() == cal[label].nunique() == 2
            features = list(meta["features"])
            if feature_set == "map_only":
                features = [x for x in features if x.startswith("map_")]
            elif feature_set == "numeric":
                features = [x for x in features if not x.startswith("static_")]
            if name in ("map", "logistic"):
                features = ["map_current", "map_300_slope", "map_300_std"]
            if name == "catboost":
                model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                    CatBoostClassifier(iterations=300, depth=5, learning_rate=.05, l2_leaf_reg=5,
                        auto_class_weights="None", random_seed=seed, thread_count=2, verbose=False, allow_writing_files=False))
            else:
                model = None if name == "map" else make_model(name, seed)
            if model is not None:
                model.fit(fit[features], fit[label].astype(int))
            calibrator = make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=2000))
            calibrator.fit(raw_score(name, model, cal, features).reshape(-1, 1), cal[label].astype(int))
            val = frame[frame.role.eq("validation")].copy()
            test = frame[frame.role.eq("pilot_test")].copy()
            for subset in (val, test):
                subset["probability"] = np.nan
                ok = subset.eligible
                subset.loc[ok, "probability"] = calibrator.predict_proba(raw_score(name, model, subset[ok], features).reshape(-1, 1))[:, 1]
            joblib.dump({"name": name, "model": model, "calibrator": calibrator, "features": features,
                "dataset_hash": meta["windows_sha256"], "protocol_hash": protocol.digest(), "horizon": h,
                "fit_calibration_seconds": time.perf_counter()-started}, directory / "model.joblib")
            val.to_csv(directory / "validation.csv.gz", index=False)
            test.to_csv(directory / "test.csv.gz", index=False)
            policies, interval_cache = {}, {}
            for policy, adaptive in (("fixed", False), ("adaptive", True)):
                threshold, selection, curve = select_fast(val, events, h, protocol, adaptive)
                # Verify optimized selection against the reference on the chosen point.
                reference, _, _ = evaluate_predictions(val, events, h, threshold, protocol)
                assert reference == selection["metrics"], "Cached replay diverged"
                write_json(directory / f"{policy}_curve.json", curve)
                scopes = {}
                for scope, subset in (("new_patients", test[~test.historical_subject]),
                                      ("historical_patients", test[test.historical_subject]), ("combined", test)):
                    m, cases, alarms = evaluate_predictions(subset, events, h, threshold, protocol)
                    cache_key = (scope, cases.to_csv(index=False))
                    if cache_key not in interval_cache:
                        interval_cache[cache_key] = bootstrap_ci(subset, events, h, threshold, protocol, 200)
                    scopes[scope] = {"metrics": m, "gates": quality_gates(m, h), "ci95": interval_cache[cache_key]}
                    cases.to_csv(directory / f"{policy}_{scope}_cases.csv", index=False)
                    write_json(directory / f"{policy}_{scope}_alarms.json", alarms)
                policies[policy] = {"selection_on_validation": selection, "scopes": scopes}
                new = scopes["new_patients"]["metrics"]
                print(key, policy, "new", new["events_detected"], "/", new["events_eligible"], "PPV", new["alarm_ppv"], flush=True)
            policies["elapsed_seconds"] = time.perf_counter()-started
            write_json(directory / "results.json", policies)
            result[key] = policies
            write_json(reports / "comparison.json", result)
    write_json(reports / "comparison.json", result)
    print("E05 completed", len(result), "horizon models", flush=True)


if __name__ == "__main__":
    main()
