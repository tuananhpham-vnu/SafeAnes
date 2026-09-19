"""E06: registered CPU TabM benchmark; all holdout results are exploratory."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from safeanes.config import Protocol
from safeanes.data import write_json
from safeanes.evaluation import bootstrap_ci, evaluate_predictions, quality_gates
from safeanes.fast_selection import select_fast
from safeanes.tabular_sota import TabMRisk, calibrated_risk, log_odds

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260917, 20260918, 20260919)
NAMES = [f"tabm_{s}" for s in SEEDS] + ["lightgbm_monotone", "tabm_ensemble"]
META = ["caseid", "subjectid", "time", "eligible", "exposure_seconds", "y_300", "y_600",
        "historical_subject", "role"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data, out, report = ROOT / "data/development300", ROOT / "artifacts/E06", ROOT / "reports/E06"
    meta = json.loads((data / "dataset.json").read_text())
    protocol = Protocol(**{**meta["protocol"], "horizons_seconds": tuple(meta["protocol"]["horizons_seconds"])})
    assert meta["scope"] == "pilot_train_pool_only"
    assert digest(data / "windows.csv.gz") == meta["windows_sha256"]
    assert protocol.digest() == meta["protocol_hash"]
    roles_path = ROOT / "data/vitaldb_development300/development_roles.csv"
    sources = [Path(__file__), ROOT / "docs/experiments/E06_PLAN.md", ROOT / "docs/SOTA_E06.md",
               ROOT / "pyproject.toml", *sorted((ROOT / "src/safeanes").glob("*.py"))]
    identity = {"experiment": "E06", "release": "0.2.0", "models": NAMES,
        "data_hashes": {str(p.relative_to(ROOT)): digest(p) for p in
                        (data / "dataset.json", data / "windows.csv.gz", data / "events.csv", data / "manifest.csv", roles_path)},
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "packages": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "torch", "tabm", "rtdl_num_embeddings", "lightgbm")}}
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    if (out / "registration.json").exists():
        assert json.loads((out / "registration.json").read_text())["identity"] == identity, "Frozen E06 changed; use a new experiment"
    else:
        write_json(out / "registration.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "identity": identity})
        for source in sources:
            dest = out / "source" / source.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    roles = pd.read_csv(roles_path)
    manifest = pd.read_csv(data / "manifest.csv")
    assert manifest.split.eq("train").all()
    assert roles.groupby("subjectid").role.nunique().eq(1).all()
    pd.testing.assert_frame_equal(roles, pd.read_csv(ROOT / "artifacts/E05/roles.csv"))
    roles.to_csv(out / "roles.csv", index=False)
    frame = pd.read_csv(data / "windows.csv.gz").merge(roles, on=["caseid", "subjectid"], validate="many_to_one")
    assert frame.role.notna().all()
    events = pd.read_csv(data / "events.csv")
    features = [f for f in meta["features"] if not f.startswith("static_")]
    fit = frame[frame.role.eq("fit") & frame.eligible & frame[["y_300", "y_600"]].ge(0).any(axis=1)]
    cal = frame[frame.role.eq("calibration") & frame.eligible]
    val = frame[frame.role.eq("validation")]
    test = frame[frame.role.eq("pilot_test")]
    validation, timings = {}, {}
    for name in NAMES[:-1]:
        directory = out / name
        directory.mkdir(exist_ok=True)
        started = time.perf_counter()
        if (directory / "bundle.joblib").exists():
            bundle = joblib.load(directory / "bundle.joblib")
        else:
            bundle = {"features": features, "kind": "tabm" if name.startswith("tabm_") else "lightgbm"}
            if bundle["kind"] == "tabm":
                model = TabMRisk(seed=int(name.rsplit("_", 1)[1])).fit(
                    fit[features], fit[["y_300", "y_600"]], fit.subjectid.to_numpy())
                bundle["model"] = model
                raw = model.predict_risk(cal[features])
                write_json(directory / "training.json", {"history": model.history, "best_epoch": model.best_epoch,
                                                          "inner_subjects": model.inner_subjects})
            else:
                constraints = [-1 if f in ("map_current", "map_60_mean", "map_300_mean") else 0 for f in features]
                bundle["models"] = []
                for h in (300, 600):
                    known = fit[f"y_{h}"].ge(0)
                    model = LGBMClassifier(n_estimators=500, num_leaves=7, learning_rate=.03,
                        min_child_samples=150, reg_lambda=10, monotone_constraints=constraints,
                        random_state=SEEDS[0], n_jobs=2, verbosity=-1)
                    model.fit(fit.loc[known, features], fit.loc[known, f"y_{h}"].astype(int))
                    bundle["models"].append(model)
                raw = np.column_stack([m.predict_proba(cal[features])[:, 1] for m in bundle["models"]])
            bundle["calibrators"] = []
            for i, h in enumerate((300, 600)):
                known = cal[f"y_{h}"].ge(0).to_numpy()
                calibrator = make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=2000))
                calibrator.fit(log_odds(raw[known, i]).reshape(-1, 1), cal.loc[known, f"y_{h}"].astype(int))
                bundle["calibrators"].append(calibrator)
            bundle["fit_calibration_seconds"] = time.perf_counter() - started
            joblib.dump(bundle, directory / "bundle.joblib")
        timings[name] = bundle["fit_calibration_seconds"]
        p = np.full((len(val), 2), np.nan)
        p[val.eligible] = calibrated_risk(bundle, val[val.eligible])
        validation[name] = p
    validation["tabm_ensemble"] = np.mean([validation[f"tabm_{s}"] for s in SEEDS], axis=0)
    selections = {}
    for name, p in validation.items():
        directory = out / name
        directory.mkdir(exist_ok=True)
        for i, h in enumerate((300, 600)):
            v = val[META].copy()
            v["probability"] = p[:, i]
            v.to_csv(directory / f"validation_{h}.csv.gz", index=False)
            for policy, adaptive in (("fixed", False), ("adaptive", True)):
                threshold, selection, curve = select_fast(v, events, h, protocol, adaptive)
                reference, _, _ = evaluate_predictions(v, events, h, threshold, protocol)
                assert reference == selection["metrics"]
                selections[f"{name}_{h}_{policy}"] = selection
                write_json(directory / f"curve_{h}_{policy}.json", curve)
    chosen = {}
    for h in (300, 600):
        def rank(name):
            s = selections[f"{name}_{h}_fixed"]
            m = s["metrics"]
            return (s["gates"]["all_point_targets_met"],
                    m["false_alarms_per_hour"] is not None and m["false_alarms_per_hour"] <= .5,
                    m["event_sensitivity"] or 0, m["alarm_ppv"] or 0, m["auroc"] or 0, name)
        chosen[str(h)] = max(("tabm_ensemble", "lightgbm_monotone"), key=rank)
    # This file is written before computing any E06 holdout prediction or metric.
    write_json(out / "validation_selection.json", {"selected": chosen, "selections": selections})
    test_predictions, verification = {}, []
    for name in NAMES[:-1]:
        bundle = joblib.load(out / name / "bundle.joblib")
        p = np.full((len(test), 2), np.nan)
        p[test.eligible] = calibrated_risk(bundle, test[test.eligible])
        test_predictions[name] = p
        check = joblib.load(out / name / "bundle.joblib")
        probe = test[test.eligible].iloc[:137]
        expected = p[test.eligible][:len(probe)]
        np.testing.assert_allclose(calibrated_risk(check, probe), expected, rtol=1e-5, atol=1e-7)
        assert np.isfinite(p[test.eligible]).all()
        assert np.all(p[test.eligible, 1] >= p[test.eligible, 0])
        verification.append({"model": name, "reload_and_batch_invariance": "passed",
                             "finite_eligible_fraction": 1., "horizon_violations": 0})
    test_predictions["tabm_ensemble"] = np.mean([test_predictions[f"tabm_{s}"] for s in SEEDS], axis=0)
    rows, results = [], {}
    for name, p in test_predictions.items():
        for i, h in enumerate((300, 600)):
            t = test[META].copy()
            t["probability"] = p[:, i]
            t.to_csv(out / name / f"test_{h}.csv.gz", index=False)
            for policy in ("fixed", "adaptive"):
                selection = selections[f"{name}_{h}_{policy}"]
                threshold = selection["metrics"]["threshold"]
                for scope, subset in (("new_patients", t[~t.historical_subject]),
                                      ("historical_patients", t[t.historical_subject]), ("combined", t)):
                    m, cases, alarms = evaluate_predictions(subset, events, h, threshold, protocol)
                    key = f"{name}_{h}_{policy}_{scope}"
                    gates = quality_gates(m, h)
                    result = {"metrics": m, "gates": gates}
                    if policy == "fixed" and scope == "new_patients":
                        result["ci95"] = bootstrap_ci(subset, events, h, threshold, protocol, 200)
                    results[key] = result
                    cases.to_csv(out / name / f"cases_{h}_{policy}_{scope}.csv", index=False)
                    write_json(out / name / f"alarms_{h}_{policy}_{scope}.json", alarms)
                    rows.append({"model": name, "horizon": h, "policy": policy, "scope": scope,
                                 "selected_on_validation": name == chosen[str(h)],
                                 "all_gates": gates["all_point_targets_met"], **m})
                print(name, h, policy, "evaluated", flush=True)
        write_json(report / "results.json", results)
    pd.DataFrame(rows).to_csv(report / "comparison.csv", index=False)
    write_json(report / "verification.json", {"checks": verification, "fit_seconds": timings,
        "source_dataset_roles_hashes": "passed", "validation_replay": "matched reference",
        "selected_on_validation_before_holdout": chosen, "scope": "exploratory_reused_E05_holdout"})
    print("E06 complete", chosen, flush=True)


if __name__ == "__main__":
    main()
