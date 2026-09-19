"""Frozen pilot policy ablation and three-seed CatBoost experiment."""
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
from safeanes.config import Protocol
from safeanes.data import write_json
from safeanes.experiment import pilot_roles, raw_score
from safeanes.evaluation import evaluate_predictions, quality_gates, bootstrap_ci
from safeanes.threshold_audit import select_audited_threshold

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT / "data/pilot_v1"
    out = ROOT / "artifacts/v0_4"
    reports = ROOT / "reports/v0_4"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / "dataset.json").read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(data / "windows.csv.gz") == meta["windows_sha256"]
    protocol = Protocol(**{**meta["protocol"], "horizons_seconds": tuple(meta["protocol"]["horizons_seconds"])})
    assert protocol.digest() == meta["protocol_hash"]
    sources = [Path(__file__), ROOT / "docs/versions/V0_4_PLAN.md", *sorted((ROOT / "src/safeanes").glob("*.py"))]
    identity = {"dataset_hash": meta["windows_sha256"], "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "seeds": [20260917, 20260918, 20260919], "iterations": 300, "depth": 5,
                "learning_rate": .05, "l2_leaf_reg": 5,
                "catboost_version": "1.2.10",
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn")},
                "inputs": {str(p.relative_to(ROOT)): digest(p) for folder in ("artifacts/tcn_v1", "artifacts/v0_3/ensemble")
                           for p in sorted((ROOT / folder).glob("*_predictions.csv.gz"))}}
    for folder in ("artifacts/tcn_v1", "artifacts/v0_3/ensemble"):
        assert json.loads((ROOT / folder / "environment.json").read_text())["dataset_hash"] == meta["windows_sha256"]
    registration = out / "registration.json"
    if registration.exists():
        assert json.loads(registration.read_text())["identity"] == identity, "Frozen source changed; use new version"
    else:
        write_json(registration, {"identity": identity, "created_utc": datetime.now(timezone.utc).isoformat()})
        for p in sources:
            target = out / "source" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    roles = pilot_roles(pd.read_csv(data / "manifest.csv"), protocol.seed)
    old_roles = pd.read_csv(ROOT / "artifacts/tcn_v1/pilot_roles.csv")
    pd.testing.assert_frame_equal(roles.reset_index(drop=True), old_roles.reset_index(drop=True))
    roles.to_csv(out / "pilot_roles.csv", index=False)
    frames = pd.read_csv(data / "windows.csv.gz").merge(roles, on=["caseid", "subjectid"], validate="many_to_one")
    events = pd.read_csv(data / "events.csv")
    results = {}

    def evaluate(name, horizon, val, test):
        run = out / name
        run.mkdir(exist_ok=True)
        key = f"{name}_{horizon}"
        result_path = run / f"{horizon}_results.json"
        if result_path.exists():
            results.update(json.loads(result_path.read_text()))
            return
        val.to_csv(run / f"{horizon}_validation.csv.gz", index=False)
        test.to_csv(run / f"{horizon}_test.csv.gz", index=False)
        pair = {}
        for policy, adaptive in (("fixed", False), ("adaptive", True)):
            threshold, selection, curve = select_audited_threshold(val, events, horizon, protocol, adaptive)
            write_json(run / f"{horizon}_{policy}_validation_curve.json", curve)
            metrics, cases, alarms = evaluate_predictions(test, events, horizon, threshold, protocol)
            cases.to_csv(run / f"{horizon}_{policy}_cases.csv", index=False)
            write_json(run / f"{horizon}_{policy}_alarms.json", alarms)
            pair[f"{key}_{policy}"] = {"selection_on_validation": selection, "pilot_test": metrics,
                "gates": quality_gates(metrics, horizon),
                "ci95": bootstrap_ci(test, events, horizon, threshold, protocol, 200)}
            print(key, policy, metrics["events_detected"], metrics["alarm_ppv"], metrics["false_alarms_per_hour"], flush=True)
        write_json(result_path, pair)
        results.update(pair)

    for name, folder in (("tcn", "artifacts/tcn_v1"), ("ensemble", "artifacts/v0_3/ensemble")):
        for h in protocol.horizons_seconds:
            run = ROOT / folder
            evaluate(name, h, pd.read_csv(run / f"{name}_{h}_validation_predictions.csv.gz"),
                     pd.read_csv(run / f"{name}_{h}_predictions.csv.gz"))
    from catboost import CatBoostClassifier
    assert importlib.metadata.version("catboost") == identity["catboost_version"]
    for weight in ("None", "SqrtBalanced"):
        for seed in identity["seeds"]:
            name = f"catboost_{weight}_{seed}"
            run = out / name
            run.mkdir(exist_ok=True)
            for h in protocol.horizons_seconds:
                if (run / f"{h}_results.json").exists():
                    results.update(json.loads((run / f"{h}_results.json").read_text()))
                    continue
                started = time.perf_counter()
                label = f"y_{h}"
                fit = frames[frames.role.eq("fit") & frames.eligible & frames[label].ge(0)]
                cal = frames[frames.role.eq("calibration") & frames.eligible & frames[label].ge(0)]
                assert fit[label].nunique() == cal[label].nunique() == 2
                features = meta["features"]
                model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                    CatBoostClassifier(iterations=300, depth=5, learning_rate=.05, l2_leaf_reg=5,
                        auto_class_weights=weight, random_seed=seed, thread_count=2,
                        verbose=False, allow_writing_files=False))
                model.fit(fit[features], fit[label].astype(int))
                calibrator = make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=2000))
                calibrator.fit(raw_score(name, model, cal, features).reshape(-1, 1), cal[label].astype(int))
                subsets = []
                for role in ("validation", "pilot_test"):
                    sub = frames[frames.role.eq(role)].copy()
                    sub["probability"] = np.nan
                    ok = sub.eligible
                    sub.loc[ok, "probability"] = calibrator.predict_proba(raw_score(name, model, sub[ok], features).reshape(-1, 1))[:, 1]
                    subsets.append(sub)
                joblib.dump({"model": model, "calibrator": calibrator, "features": features,
                    "protocol_hash": protocol.digest(), "dataset_hash": meta["windows_sha256"],
                    "elapsed_seconds": time.perf_counter()-started}, run / f"{h}_model.joblib")
                evaluate(name, h, *subsets)
    write_json(reports / "comparison.json", results)
    lines = ["# v0.4 — CatBoost và ablation ngưỡng", "",
        "Pilot 60 ca, split giữ nguyên; 9 bệnh nhân pilot_test, 3/4 event đủ điều kiện ở 5/10 phút. CPU. Không mở global test.", "",
        "Thiết kế và nguồn: [plan](../../docs/versions/V0_4_PLAN.md). CI 200 bootstrap bệnh nhân và validation metrics: [JSON](comparison.json).", "",
        "| Run / policy | AUROC | AP | Event phát hiện | PPV | FA/giờ | ECE | Threshold | Đủ gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    fmt = lambda x: "N/A" if x is None else f"{x:.4f}"
    for key, r in results.items():
        m = r["pilot_test"]
        values = [key, fmt(m["auroc"]), fmt(m["average_precision"]), f"{m['events_detected']}/{m['events_eligible']}",
                  *[fmt(m[k]) for k in ("alarm_ppv", "false_alarms_per_hour", "ece", "threshold")],
                  str(r["gates"]["all_point_targets_met"])]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "Không lựa chọn seed tốt nhất bằng bảng test. Mọi kết quả exploratory vì pilot_test đã được xem trước. Thay đổi ngưỡng không thay đổi AUROC/AP; ngưỡng >1 biểu thị không cảnh báo. PPV N/A không phải kết quả tốt. Chưa mở rộng cohort trong phép ablation này."]
    (reports / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
