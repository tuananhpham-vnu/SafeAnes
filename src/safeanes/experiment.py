"""Exploratory TRAIN-pool experiment; the global final test set stays unopened."""

from dataclasses import asdict
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Protocol
from .data import write_json
from .evaluation import bootstrap_ci, evaluate_predictions, quality_gates, select_threshold


def pilot_roles(manifest, seed):
    if not manifest.split.eq("train").all():
        raise ValueError("This experiment accepts global TRAIN patients only")
    subjects = np.sort(manifest.subjectid.unique()).copy()
    if len(subjects) < 20:
        raise ValueError("Use >=20 patients for the pilot split; fewer is only a data audit")
    np.random.default_rng(seed).shuffle(subjects)
    n = len(subjects)
    pieces = np.split(subjects, [int(n * .60), int(n * .75), int(n * .85)])
    assignment = {s: role for role, group in zip(("fit", "calibration", "validation", "pilot_test"), pieces) for s in group}
    out = manifest[["caseid", "subjectid"]].copy()
    out["role"] = out.subjectid.map(assignment)
    return out


def make_model(name, seed):
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), LogisticRegression(max_iter=2000, C=1, random_state=seed))
    if name == "hist_gradient":
        return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
            l2_regularization=1, learning_rate=.05, early_stopping=False, random_state=seed)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=.05,
                              reg_lambda=1, random_state=seed, n_jobs=2, verbosity=-1)
    raise ValueError(f"Unknown model {name}")


def raw_score(name, model, frame, features):
    if name == "map":
        return -frame.map_current.to_numpy(float)
    if hasattr(model, "decision_function"):
        return model.decision_function(frame[features])
    probability = np.clip(model.predict_proba(frame[features])[:, 1], 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def run_pilot(dataset, out, models=("map", "logistic", "hist_gradient"), repeats=200):
    dataset, out = Path(dataset), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new output directory for each experiment")
    meta = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    if meta["scope"] != "pilot_train_pool_only":
        raise ValueError("Only exploratory training-pool datasets accepted")
    config = meta["protocol"]
    config["horizons_seconds"] = tuple(config["horizons_seconds"])
    protocol = Protocol(**config)
    if meta["protocol_hash"] != protocol.digest():
        raise ValueError("Protocol hash mismatch")
    if hashlib.sha256((dataset / "windows.csv.gz").read_bytes()).hexdigest() != meta["windows_sha256"]:
        raise ValueError("Dataset checksum mismatch")
    frames = pd.read_csv(dataset / "windows.csv.gz")
    events = pd.read_csv(dataset / "events.csv")
    roles = pilot_roles(pd.read_csv(dataset / "manifest.csv"), protocol.seed)
    frames = frames.merge(roles, on=["caseid", "subjectid"], validate="many_to_one")
    out.mkdir(parents=True, exist_ok=True)
    roles.to_csv(out / "pilot_roles.csv", index=False)
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "lightgbm"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    write_json(out / "environment.json", {"python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__, "protocol": asdict(protocol),
        "dataset_hash": meta["windows_sha256"], "scope": "exploratory_pilot_not_final_test",
        "package_versions": versions,
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))}})
    report = {"scope": "exploratory_pilot_not_final_test", "models": {}, "errors": {}}
    for horizon in protocol.horizons_seconds:
        label = f"y_{horizon}"
        fit = frames[frames.role.eq("fit") & frames.eligible & frames[label].ge(0)]
        cal = frames[frames.role.eq("calibration") & frames.eligible & frames[label].ge(0)]
        if fit[label].nunique() < 2 or cal[label].nunique() < 2:
            report["errors"][str(horizon)] = "Fit/calibration need both classes; expand pilot, do not reshuffle by outcome"
            continue
        for name in models:
            started = time.perf_counter()
            key = f"{name}_{horizon}"
            features = (["map_current", "map_300_slope", "map_300_std"]
                        if name in ("map", "logistic") else meta["features"])
            try:
                model = None if name == "map" else make_model(name, protocol.seed)
            except ImportError:
                report["errors"][key] = "Optional dependency missing: pip install lightgbm"
                continue
            if model is not None:
                model.fit(fit[features], fit[label].astype(int))
            # Sigmoid calibration on patients disjoint from model fitting (S05).
            calibrator = make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=2000))
            calibrator.fit(raw_score(name, model, cal, features).reshape(-1, 1), cal[label].astype(int))
            validation = frames[frames.role.eq("validation")].copy()
            test = frames[frames.role.eq("pilot_test")].copy()
            for subset in (validation, test):
                subset["probability"] = np.nan
                ok = subset.eligible
                if ok.any():
                    subset.loc[ok, "probability"] = calibrator.predict_proba(
                        raw_score(name, model, subset[ok], features).reshape(-1, 1))[:, 1]
            threshold, selection = select_threshold(validation, events, horizon, protocol)
            metrics, cases, alarms = evaluate_predictions(test, events, horizon, threshold, protocol)
            intervals = bootstrap_ci(test, events, horizon, threshold, protocol, repeats)
            result = {"selection_on_validation": selection, "pilot_test": metrics,
                      "ci95": intervals, "gates": quality_gates(metrics, horizon),
                      "elapsed_seconds": time.perf_counter() - started}
            report["models"][key] = result
            joblib.dump({"model": model, "name": name, "calibrator": calibrator,
                         "features": features, "threshold": threshold, "horizon": horizon,
                         "protocol_hash": protocol.digest()}, out / f"{key}.joblib")
            test.to_csv(out / f"{key}_predictions.csv.gz", index=False, compression="gzip")
            cases.to_csv(out / f"{key}_case_metrics.csv", index=False)
            write_json(out / f"{key}_alarms.json", alarms)
            write_json(out / "report.json", report)
    write_json(out / "report.json", report)
    return report
