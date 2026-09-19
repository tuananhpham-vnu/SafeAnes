"""E05 reports and saved-model checks; never select a model on test metrics."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from safeanes.config import Protocol
from safeanes.data import write_json
from safeanes.experiment import raw_score
from safeanes.evaluation import evaluate_predictions, bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]


def main():
    out, directory = ROOT / "artifacts/E05", ROOT / "reports/E05"
    results = json.loads((directory / "comparison.json").read_text())
    assert len(results) == 16, "Wait for all registered models"
    registration = json.loads((out / "registration.json").read_text())["identity"]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for source, expected in registration["sources"].items():
        assert digest(out / "source" / source) == expected
    assert digest(ROOT / "data/development300/windows.csv.gz") == registration["dataset_hash"]
    rows, checks = [], []
    for key, policies in results.items():
        for policy in ("fixed", "adaptive"):
            v = policies[policy]["selection_on_validation"]["metrics"]
            for scope, value in policies[policy]["scopes"].items():
                rows.append({"model": key, "policy": policy, "scope": scope, **value["metrics"],
                    "all_gates": value["gates"]["all_point_targets_met"],
                    "validation_recall": v["event_sensitivity"], "validation_ppv": v["alarm_ppv"],
                    "validation_fah": v["false_alarms_per_hour"]})
        saved = joblib.load(out / key / "model.joblib")
        test = pd.read_csv(out / key / "test.csv.gz")
        valid = test[test.eligible]
        reproduced = saved["calibrator"].predict_proba(raw_score(saved["name"], saved["model"], valid, saved["features"]).reshape(-1, 1))[:, 1]
        np.testing.assert_allclose(reproduced, valid.probability, rtol=1e-10, atol=1e-12)
        checks.append({"model": key, "prediction_reproduction": "passed", "features": len(saved["features"]),
                       "finite_probability_fraction_among_eligible": float(np.isfinite(valid.probability).mean()),
                       "fit_calibration_seconds": saved["fit_calibration_seconds"]})
    table = pd.DataFrame(rows)
    table.to_csv(directory / "comparison.csv", index=False)
    support = pd.DataFrame(json.loads((directory / "data_support.json").read_text()))
    lines = ["# v0.2 — E05: development 300 ca", "",
        "Đã chạy 16 model theo horizon: MAP/logistic/LightGBM; CatBoost ba seed; ablation MAP-only/numeric/+static. Mỗi model có hai policy ngưỡng. Không tăng release và không mở global final test.", "",
        "300 ca/297 bệnh nhân, 99.132 cửa sổ, 497 episode IOH. Nhóm đánh giá mới: 37 bệnh nhân, 35/37 event đủ điều kiện ở 5/10 phút. Nhóm cũ: 9 bệnh nhân, 3/4 event. Bảo toàn roles cũ; mở rộng cohort thay đổi dữ liệu train/calibration/validation nên không quy mọi khác biệt cho kiến trúc.", "",
        "[Plan khóa trước chạy](../../docs/experiments/E05_PLAN.md) · [JSON gồm CI và mọi scope](comparison.json) · [CSV](comparison.csv) · [Kiểm tra model](verification.json) · [Subgroup/lead time](SUBGROUPS.md) · [Hình và replay](FIGURES.md)", ""]
    fmt = lambda x: "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
    for scope, policy, title in (("new_patients", "fixed", "Primary: bệnh nhân mới, ngưỡng cố định"),
                                 ("new_patients", "adaptive", "Secondary: bệnh nhân mới, ngưỡng adaptive"),
                                 ("historical_patients", "fixed", "Đối chiếu nhóm bệnh nhân cũ")):
        lines += [f"## {title}", "", "| Model | AUROC | AP | Detect/event | PPV | FA/giờ | ECE | Đủ gate |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for r in table[(table.scope == scope) & (table.policy == policy)].to_dict("records"):
            lines.append("| " + " | ".join([r["model"], fmt(r["auroc"]), fmt(r["average_precision"]),
                f"{r['events_detected']}/{r['events_eligible']}", fmt(r["alarm_ppv"]), fmt(r["false_alarms_per_hour"]),
                fmt(r["ece"]), str(r["all_gates"])]) + " |")
    seed_rows = []
    for h in (300, 600):
        group = table[table.model.isin([f"catboost_all_{s}_{h}" for s in (20260917, 20260918, 20260919)]) & table.scope.eq("new_patients") & table.policy.eq("fixed")]
        seed_rows.append({"horizon": h, "recall_mean": float(group.event_sensitivity.mean()),
            "recall_min": float(group.event_sensitivity.min()), "recall_max": float(group.event_sensitivity.max()),
            "auroc_mean": float(group.auroc.mean()), "ppv_mean_defined": float(group.alarm_ppv.mean()) if group.alarm_ppv.notna().any() else None,
            "fah_mean": float(group.false_alarms_per_hour.mean())})
    write_json(directory / "seed_summary.json", seed_rows)
    lines += ["", "## CatBoost ba seed — primary trên bệnh nhân mới", ""]
    for r in seed_rows:
        lines.append(f"- {r['horizon']//60} phút: recall mean {r['recall_mean']:.3f}, min–max {r['recall_min']:.3f}–{r['recall_max']:.3f}; AUROC mean {r['auroc_mean']:.3f}; PPV mean (seed có cảnh báo) {fmt(r['ppv_mean_defined'])}; FA/giờ mean {r['fah_mean']:.3f}.")
    gates = int(table[table.scope.eq("new_patients") & table.policy.eq("fixed")].all_gates.sum())
    lines += ["", "## Kết luận và giới hạn", "",
        f"- Primary trên nhóm mới: {gates}/16 model theo horizon đạt toàn bộ gate. Báo cáo không chọn seed tốt nhất theo test.",
        "- Global final test chưa sử dụng. E05 là development holdout mới đã khóa trước chạy; từ sau báo cáo này, việc thử ý tưởng phát sinh trên cùng nhóm phải ghi exploratory hoặc dùng holdout mới.",
        "- CI 200 bootstrap theo bệnh nhân, không theo window. Các seed dùng cùng bệnh nhân nên không phải cohort độc lập. Báo cả budget FA/giờ trên test ngay khi vượt mức validation.",
        "- Numeric+static so với numeric và MAP-only dùng cùng seed 20260917; ablation không đại diện độ ổn định ba seed cho mọi feature set.",
        "- Source VitalDB mô tả cohort non-cardiac; audit E05 có General surgery 190, Thoracic surgery 87, Gynecology 16, Urology 7 và không có cờ tên tim theo regex sơ bộ. Đây là bằng chứng nguồn/metadata, không phải gán nhãn thủ thuật lại bởi chuyên gia. [Bài gốc](https://www.nature.com/articles/s41597-022-01411-5).",
        "- FA/giờ vẫn theo exposure xấp xỉ 30 giây của protocol v1; hai classifier tabular chưa ràng buộc p600>=p300. Không gọi bản này là nghiệm thu chất lượng dự báo hoặc kết quả triển khai lâm sàng.", "",
        "- `prediction_coverage` trong evaluator v1 là eligible_seconds/scheduled_seconds (tỉ lệ thời gian đủ điều kiện), không phải tỉ lệ model trả xác suất trong các window đủ điều kiện. Chỉ số availability riêng được ghi trong verification.json; không thay gate sau khi xem kết quả.", "",
        "## Tái lập", "", "```powershell", '$env:PYTHONPATH = "$PWD/src;$PWD/.local_deps"',
        "python scripts/prepare_development.py", "# Chỉ build khi output dataset chưa tồn tại:",
        "python -m safeanes.cli build-pilot --root data/vitaldb_development300 --out data/development300",
        "python scripts/run_development.py", "python scripts/report_development.py",
        "python scripts/analyze_development.py", "python scripts/plot_development.py", "```", "",
        "Run có registration/source snapshot và resume từng horizon. Không ghi đè input/output đã khóa nếu đổi code hoặc protocol."]
    (directory / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    coherence = []
    for prefix in sorted({key.rsplit("_", 1)[0] for key in results}):
        a = pd.read_csv(out / f"{prefix}_300/test.csv.gz")
        b = pd.read_csv(out / f"{prefix}_600/test.csv.gz")
        pair = a[a.eligible][["caseid", "time", "probability", "historical_subject"]].merge(
            b[b.eligible][["caseid", "time", "probability"]], on=["caseid", "time"], suffixes=("_300", "_600"), validate="one_to_one")
        for scope, subset in (("new", pair[~pair.historical_subject]), ("historical", pair[pair.historical_subject])):
            coherence.append({"model": prefix, "scope": scope, "eligible_windows": len(subset),
                              "p600_below_p300": int((subset.probability_600 < subset.probability_300).sum())})
    write_json(directory / "verification.json", {"release": "0.2.0", "experiment": "E05",
        "frozen_source_and_dataset_hashes": "passed", "registered_horizon_models": 16,
        "report_source_hashes": {p.name: digest(p) for p in (Path(__file__), ROOT / "scripts/analyze_development.py", ROOT / "scripts/plot_development.py")},
        "cached_validation_replay": "each selected threshold matched reference evaluator during run", "runs": checks,
        "horizon_coherence": coherence})
    release_dir = ROOT / "reports/v0_2"
    release_dir.mkdir(exist_ok=True)
    summary = ["# v0.2 — báo cáo tổng hợp trước push", "",
        "Toàn bộ sửa lỗi và đợt thí nghiệm hiện tại đã gộp vào v0.2. Không tạo release mới cho E03/E04/E05 và chưa push. Package metadata/README/model card dùng 0.2.0; tên artifact cũ giữ để bảo toàn source snapshot/hashes.", "",
        "## Đã hoàn thành thêm", "",
        "- E04: CatBoost hai cấu hình weighting × ba seed × hai horizon; ablation ngưỡng trên cùng pilot 60 ca.",
        "- E05: tải và dựng 300 ca/297 bệnh nhân, 99.132 windows, 497 episode; giữ role của 60 bệnh nhân cũ; nhóm đánh giá mới 37 bệnh nhân với 35/37 event đủ điều kiện.",
        "- Chạy đủ 16 model theo horizon, mỗi model hai policy; so MAP/logistic/LightGBM/CatBoost, ba seed và MAP-only/numeric/+static. Báo riêng bệnh nhân mới/cũ/gộp.",
        "- CI bootstrap 200 theo bệnh nhân; thêm 28 bản ghi subgroup (kể cả nhóm trống) và CI lead time có điều kiện đã phát hiện; hình so sánh và replay từ dữ liệu thật.",
        "- Sửa model card về tính nhất quán horizon, thêm split ổn định khi mở rộng và kiểm tra fast selection khớp evaluator. 45 kiểm thử phần mềm đã qua; 16 model lưu tái tạo đúng predictions.", "",
        "## Kết quả E05 chính trên bệnh nhân mới", "",
        f"{gates}/16 model theo horizon đạt toàn bộ quality gate. Cohort lớn cho thấy kết quả pilot nhỏ chưa đủ để kết luận chất lượng; không chọn lại model/seed/threshold bằng test.", ""]
    for row in seed_rows:
        summary.append(f"- CatBoost all-features, ba seed, {row['horizon']//60} phút: recall trung bình {row['recall_mean']:.3f}; AUROC {row['auroc_mean']:.3f}; PPV {fmt(row['ppv_mean_defined'])}; FA/giờ {row['fah_mean']:.3f}.")
    summary += ["", "## Hồ sơ chi tiết", "",
        "- [E05: kết quả đầy đủ và CI](../E05/REPORT.md)", "- [Phân nhóm/lead time](../E05/SUBGROUPS.md)",
        "- [Hình và case replay](../E05/FIGURES.md)", "- [E04: pilot/threshold/CatBoost](../v0_4/REPORT.md#nhan-xet)",
        "- [E03: CNN/ensemble](../v0_3/REPORT.md#nhan-xet)", "- [Tiến độ plan và phần chưa hoàn thành](../../docs/IMPLEMENTATION_STEPS.md#tien-do)", "",
        "Exposure từng giây chưa triển khai. T4/Kaggle, kiểm định ngoài/tiến cứu và nhãn cơ chế cần môi trường/dữ liệu/chuyên gia tương ứng. Final test vẫn chưa mở; waveform còn theo điều kiện validation. Không đánh dấu toàn bộ nghiên cứu hoặc MVP hiệu năng hoàn thành."]
    (release_dir / "REPORT.md").write_text("\n".join(summary)+"\n", encoding="utf-8")
    print(json.dumps(seed_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
