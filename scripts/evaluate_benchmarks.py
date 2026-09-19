"""Audit real predictions and paired patient-bootstrap differences; no tests/ run."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import torch

from safeanes.config import Protocol
from safeanes.data import write_json
from safeanes.evaluation import evaluate_predictions, quality_gates
from safeanes.experiment import raw_score
from safeanes.tabular_sota import calibrated_risk

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/benchmarks"
SCOPES = {"pilot_holdout": "historical_patients", "vital_expansion": "new_patients",
          "vital_combined": "combined"}
COUNT_COLS = ["events_detected", "events_eligible", "true_alarms", "false_alarms", "evaluable_seconds"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_metrics(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        other = expected[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            np.testing.assert_allclose(value, other, rtol=1e-10, atol=1e-12, err_msg=key)
        else:
            assert value == other, (key, value, other)


def rates(counts):
    def divide(a, b):
        return np.divide(a, b, out=np.full(np.shape(a), np.nan, dtype=float), where=b > 0)
    return np.column_stack((divide(counts[..., 0], counts[..., 1]),
        divide(counts[..., 2], counts[..., 2] + counts[..., 3]),
        divide(counts[..., 3] * 3600, counts[..., 4])))


def paired_bootstrap(a, b, repeats=5000):
    a = a.groupby("subjectid")[COUNT_COLS].sum().sort_index()
    b = b.groupby("subjectid")[COUNT_COLS].sum().sort_index()
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a.events_eligible, b.events_eligible)
    np.testing.assert_allclose(a.evaluable_seconds, b.evaluable_seconds)
    rng = np.random.default_rng(20260918)
    weights = rng.multinomial(len(a), np.full(len(a), 1 / len(a)), size=repeats)
    aa, bb = rates(weights @ a.to_numpy()), rates(weights @ b.to_numpy())
    point = (rates(a.sum().to_numpy()[None, :]) - rates(b.sum().to_numpy()[None, :]))[0]
    result = {}
    for i, key in enumerate(("event_sensitivity", "alarm_ppv", "false_alarms_per_hour")):
        d = aa[:, i] - bb[:, i]
        good = d[np.isfinite(d)]
        lower, upper = np.quantile(good, [.025, .975]) if len(good) else (np.nan, np.nan)
        result[key] = {"difference": float(point[i]) if np.isfinite(point[i]) else None,
            "low": float(lower) if np.isfinite(lower) else None,
            "high": float(upper) if np.isfinite(upper) else None,
            "valid_replicates": len(good),
            "favorable_ci_excludes_zero": bool(lower > 0 if i < 2 else upper < 0)}
    return {"unit": "subjectid", "subjects": len(a), "repeats": repeats,
            "direction": "TabM ensemble minus comparator; lower FA/hour is better", "metrics": result}


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / "data/development300/dataset.json").read_text())
    pilot_meta = json.loads((ROOT / "data/pilot_v1/dataset.json").read_text())
    assert meta["protocol_hash"] == pilot_meta["protocol_hash"]
    protocol = Protocol(**{**meta["protocol"], "horizons_seconds": tuple(meta["protocol"]["horizons_seconds"])})
    assert protocol.digest() == meta["protocol_hash"]
    for folder, metadata in (("development300", meta), ("pilot_v1", pilot_meta)):
        assert sha(ROOT / "data" / folder / "windows.csv.gz") == metadata["windows_sha256"]
        assert pd.read_csv(ROOT / "data" / folder / "manifest.csv").split.eq("train").all()
    # Validate frozen experiment source snapshots and input files, without changing them.
    for experiment in ("E05", "E06"):
        identity = json.loads((ROOT / f"artifacts/{experiment}/registration.json").read_text())["identity"]
        for path, expected in identity["sources"].items():
            assert sha(ROOT / f"artifacts/{experiment}/source" / path) == expected
        for path, expected in identity.get("data_hashes", {}).items():
            assert sha(ROOT / path) == expected
        if "dataset_hash" in identity:
            assert identity["dataset_hash"] == meta["windows_sha256"]
    pilot = pd.read_csv(ROOT / "data/pilot_v1/windows.csv.gz").set_index(["caseid", "time"]).sort_index()
    development = pd.read_csv(ROOT / "data/development300/windows.csv.gz").set_index(["caseid", "time"]).sort_index()
    pd.testing.assert_frame_equal(pilot, development.loc[pilot.index, pilot.columns], rtol=1e-12, atol=1e-12)
    roles = pd.read_csv(ROOT / "artifacts/E06/roles.csv")
    old_roles = pd.read_csv(ROOT / "artifacts/tcn_v1/pilot_roles.csv")
    assert roles.groupby("subjectid").role.nunique().eq(1).all()
    overlap = old_roles.merge(roles, on=["caseid", "subjectid"], validate="one_to_one", suffixes=("_old", "_new"))
    assert len(overlap) == len(old_roles) and overlap.role_old.eq(overlap.role_new).all()
    assert overlap.historical_subject.all()
    holdout = roles[roles.role.eq("pilot_test")]
    used = set(roles.loc[~roles.role.eq("pilot_test"), "subjectid"])
    assert used.isdisjoint(holdout.subjectid)
    subjects = {
        "pilot_holdout": set(old_roles.loc[old_roles.role.eq("pilot_test"), "subjectid"]),
        "vital_expansion": set(holdout.loc[~holdout.historical_subject, "subjectid"]),
        "vital_combined": set(holdout.subjectid)}
    assert subjects["pilot_holdout"].isdisjoint(subjects["vital_expansion"])
    assert subjects["pilot_holdout"] | subjects["vital_expansion"] == subjects["vital_combined"]
    base = development.reset_index().merge(roles, on=["caseid", "subjectid"], validate="many_to_one")
    base = base[base.role.eq("pilot_test")].sort_values(["caseid", "time"])
    events = pd.read_csv(ROOT / "data/development300/events.csv")
    pilot_events = pd.read_csv(ROOT / "data/pilot_v1/events.csv")
    pd.testing.assert_frame_equal(pilot_events.sort_values(["caseid", "onset"]).reset_index(drop=True),
        events[events.caseid.isin(pilot.index.get_level_values("caseid"))].sort_values(["caseid", "onset"]).reset_index(drop=True))
    old = json.loads((ROOT / "reports/E05/comparison.json").read_text())
    new = json.loads((ROOT / "reports/E06/results.json").read_text())
    selection = json.loads((ROOT / "artifacts/E06/validation_selection.json").read_text())["selections"]
    predictions, checks, entries = {}, [], []
    for key, result in old.items():
        directory = ROOT / "artifacts/E05" / key
        h = int(key.rsplit("_", 1)[1])
        model_name = "E05/" + key.rsplit("_", 1)[0]
        pred = pd.read_csv(directory / "test.csv.gz", float_precision="round_trip").sort_values(["caseid", "time"]).reset_index(drop=True)
        bundle = joblib.load(directory / "model.joblib")
        eligible = base.eligible.to_numpy()
        reproduced = bundle["calibrator"].predict_proba(raw_score(bundle["name"], bundle["model"], base[base.eligible], bundle["features"]).reshape(-1, 1))[:, 1]
        np.testing.assert_allclose(reproduced, pred.loc[eligible, "probability"], rtol=1e-10, atol=1e-12)
        for policy in ("fixed", "adaptive"):
            entries.append((model_name, h, policy, pred, result[policy]["selection_on_validation"]["metrics"]["threshold"],
                            {scope: result[policy]["scopes"][old_scope]["metrics"] for scope, old_scope in SCOPES.items()}))
        checks.append(key)
    for name in ("tabm_20260917", "tabm_20260918", "tabm_20260919", "lightgbm_monotone", "tabm_ensemble"):
        if name == "tabm_ensemble":
            reproduced = np.mean([predictions[f"tabm_{s}"] for s in (20260917, 20260918, 20260919)], axis=0)
        else:
            bundle = joblib.load(ROOT / "artifacts/E06" / name / "bundle.joblib")
            reproduced = calibrated_risk(bundle, base[base.eligible])
        assert np.isfinite(reproduced).all() and np.all(reproduced[:, 1] >= reproduced[:, 0])
        predictions[name] = reproduced
        for i, h in enumerate((300, 600)):
            pred = pd.read_csv(ROOT / "artifacts/E06" / name / f"test_{h}.csv.gz", float_precision="round_trip").sort_values(["caseid", "time"]).reset_index(drop=True)
            np.testing.assert_allclose(reproduced[:, i], pred.loc[base.eligible.to_numpy(), "probability"], rtol=1e-5, atol=1e-7)
            for policy in ("fixed", "adaptive"):
                entries.append(("E06/" + name, h, policy, pred, selection[f"{name}_{h}_{policy}"]["metrics"]["threshold"],
                                {scope: new[f"{name}_{h}_{policy}_{old_scope}"]["metrics"] for scope, old_scope in SCOPES.items()}))
        checks.append(name)
    print("Reloaded all saved models; predictions reproduced", flush=True)
    rows, case_cache, audits = [], {}, []
    for scope, members in subjects.items():
        data = (pilot.reset_index() if scope == "pilot_holdout" else development.reset_index())
        data = data[data.subjectid.isin(members)].sort_values(["caseid", "time"])
        ev = pilot_events if scope == "pilot_holdout" else events
        for h in (300, 600):
            known = data.eligible & data[f"y_{h}"].ge(0)
            e = ev[ev.subjectid.isin(members)]
            audits.append({"benchmark": scope, "horizon": h, "subjects": len(members), "cases": int(data.caseid.nunique()),
                "windows": len(data), "known_eligible_windows": int(known.sum()), "positive_windows": int(data.loc[known, f"y_{h}"].sum()),
                "events_all": len(e), "events_eligible": int(e[f"eligible_{h}"].sum()),
                "evaluable_hours": float(data.loc[known, "exposure_seconds"].sum()/3600)})
        for model, h, policy, pred, threshold, expected in entries:
            saved = pred[pred.subjectid.isin(members)]
            cols = ["caseid", "subjectid", "time", "eligible", "exposure_seconds", "y_300", "y_600"]
            pd.testing.assert_frame_equal(data[cols].reset_index(drop=True), saved[cols].reset_index(drop=True), rtol=1e-12, atol=1e-12)
            evaluation = data[cols].copy()
            evaluation["probability"] = saved.probability.to_numpy()
            assert np.isfinite(evaluation.loc[evaluation.eligible, "probability"]).all()
            metrics, cases, _ = evaluate_predictions(evaluation, ev, h, threshold, protocol)
            try:
                verify_metrics(metrics, expected[scope])
            except AssertionError as error:
                raise AssertionError(f"Replay mismatch: {scope}, {model}, {h}, {policy}: {error}") from error
            rows.append({"benchmark": scope, "model": model, "horizon": h, "policy": policy,
                "subjects": len(members), "all_gates": quality_gates(metrics, h)["all_point_targets_met"], **metrics})
            if policy == "fixed":
                case_cache[(scope, model, h)] = cases
        print("Replayed", scope, len(members), "patients", flush=True)
    table = pd.DataFrame(rows)
    assert len(table) == 156
    table.to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(audits).to_csv(OUT / "data_support.csv", index=False)
    pairs = []
    for scope in ("pilot_holdout", "vital_expansion"):
        for h in (300, 600):
            for comparator in ("E05/catboost_all_20260917", "E05/catboost_numeric_20260917", "E05/map_all_20260917"):
                result = paired_bootstrap(case_cache[(scope, "E06/tabm_ensemble", h)], case_cache[(scope, comparator, h)])
                pairs.append({"benchmark": scope, "horizon": h, "model": "E06/tabm_ensemble", "comparator": comparator, **result})
    write_json(OUT / "paired_differences.json", pairs)
    write_json(OUT / "verification.json", {"script_sha256": sha(Path(__file__)), "protocol_hash": protocol.digest(),
        "dataset_hashes": {"pilot": pilot_meta["windows_sha256"], "development": meta["windows_sha256"]},
        "pilot_features_labels_events_equal_development_subset": True, "historical_roles_unchanged": True,
        "holdout_disjoint_from_fit_calibration_validation": True, "global_train_only": True,
        "saved_models_reproduced": checks, "metrics_replayed_and_matched": len(rows),
        "paired_patient_bootstrap_repeats": 5000, "tests_directory_executed": False,
        "model_or_threshold_retuned": False, "final_test_opened": False})
    render(table, audits, pairs)


def render(table, audits, pairs):
    fmt = lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.3f}"
    lines = ["# Đánh giá riêng Pilot và VitalDB development", "",
        "Pilot là tập con VitalDB, không phải nguồn kiểm định ngoài độc lập. Cohort gốc: pilot 60 ca/60 bệnh nhân; "
        "development 300 ca/297 bệnh nhân. Chỉ đánh giá holdout tách bệnh nhân; không chấm cả cohort lẫn dữ liệu fit. "
        "Đây không phải benchmark trên toàn bộ VitalDB.", "",
        "Model E05/E06 và ngưỡng validation đã khóa được giữ nguyên. Đánh giá lại prediction trên dữ liệu gốc từng tập; "
        "không retrain hoặc tune theo kết quả. Không chạy tests/ theo yêu cầu. Final test chưa mở.", "",
        "[Bảng đầy đủ primary/secondary](comparison.csv) · [Audit cỡ mẫu](data_support.csv) · "
        "[Chênh lệch ghép cặp và CI](paired_differences.json) · [Kiểm chứng thực nghiệm](verification.json)", "",
        "## Thành phần benchmark", "", "| Tập | Phút | Bệnh nhân | Ca | Window có nhãn/đủ điều kiện | Event đủ điều kiện | Giờ |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for a in audits:
        lines.append(f"| {a['benchmark']} | {a['horizon']//60} | {a['subjects']} | {a['cases']} | {a['known_eligible_windows']} | {a['events_eligible']} | {a['evaluable_hours']:.2f} |")
    lines += ["", "pilot_holdout = 9 bệnh nhân holdout pilot cũ; vital_expansion = 37 bệnh nhân ngoài pilot; "
              "vital_combined = hợp của hai nhóm, 46 bệnh nhân. Nhóm gộp không phải lần xác nhận độc lập thứ ba."]
    for scope in SCOPES:
        lines += ["", f"## {scope} — primary, cùng model/ngưỡng đã khóa", "",
            "| Model | Phút | AUROC | AP | Detect/event | Recall | PPV | FA/giờ | Đủ gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in table[table.benchmark.eq(scope) & table.policy.eq("fixed")].to_dict("records"):
            lines.append("| " + " | ".join([r["model"], str(r["horizon"]//60), fmt(r["auroc"]), fmt(r["average_precision"]),
                f"{r['events_detected']}/{r['events_eligible']}", fmt(r["event_sensitivity"]), fmt(r["alarm_ppv"]),
                fmt(r["false_alarms_per_hour"]), str(r["all_gates"])]) + " |")
    lines += ["", "## Kiểm chứng chênh lệch, 5.000 bootstrap ghép cặp theo bệnh nhân", "",
        "Chênh lệch = TabM ensemble − đối chứng. Recall/PPV dương tốt hơn; FA/giờ âm tốt hơn. "
        "Mỗi lần lấy mẫu dùng cùng bệnh nhân và số lần lặp bệnh nhân cho hai model; giữ nguyên ngưỡng và replay. "
        "CI percentile 95% không bao gồm bất định do huấn luyện/chọn model và không hiệu chỉnh so sánh nhiều lần. "
        "Các holdout đã được xem, nên kể cả CI không chứa 0 cũng chỉ là bằng chứng thăm dò.", "",
        "| Tập | Phút | Đối chứng | Δ recall [95% CI] | Δ PPV [95% CI] | Δ FA/giờ [95% CI] |",
        "|---|---:|---|---|---|---|"]
    for p in pairs:
        ci = lambda key: "{} [{}, {}]".format(*[fmt(p["metrics"][key][k]) for k in ("difference", "low", "high")])
        lines.append("| " + " | ".join([p["benchmark"], str(p["horizon"]//60), p["comparator"],
            ci("event_sensitivity"), ci("alarm_ppv"), ci("false_alarms_per_hour")]) + " |")
    lines += ["", "## Cách diễn giải", "",
        "- So sánh hiện tại đánh giá các pipeline train trên development300. Đây không phải tái huấn luyện TabM chỉ bằng "
        "36 bệnh nhân fit của pilot. Kết quả pilot-only E01/E04 dùng train/calibration/validation nhỏ hơn, "
        "nên không đặt ngang hàng để quy chênh lệch cho kiến trúc.",
        "- TabM có inner stopping riêng trong FIT; CatBoost dùng toàn bộ FIT. Khác biệt phản ánh cả pipeline.",
        "- Pilot chỉ có 3/4 biến cố đủ điều kiện: một biến cố làm recall đổi 33,3/25 điểm phần trăm. "
        "Số window nhiều không bù được cỡ mẫu bệnh nhân/biến cố thấp.",
        "- Nếu CI của chênh lệch chứa 0, chưa đủ bằng chứng để kết luận tốt hơn trên chỉ số đó. "
        "Không kết luận thắng chỉ từ AUROC hay chỉ chọn benchmark có điểm đẹp.",
        "- Global final test và kiểm định ngoài vẫn chưa dùng. Muốn xác nhận cải thiện phải khóa pipeline "
        "và đánh giá trên bệnh nhân chưa từng được dùng để phát triển/chọn ý tưởng.", "",
        "Tái lập: `PYTHONPATH=src;.local_deps python scripts/evaluate_benchmarks.py` trên PowerShell cần đặt "
        "`$env:PYTHONPATH = \"$PWD/src;$PWD/.local_deps\"` rồi chạy `python scripts/evaluate_benchmarks.py`."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table[table.model.eq("E06/tabm_ensemble") & table.policy.eq("fixed")][
        ["benchmark", "horizon", "events_detected", "events_eligible", "alarm_ppv", "false_alarms_per_hour", "auroc"]].to_string(index=False))


if __name__ == "__main__":
    main()
