"""Frozen v0.3 numeric benchmark; reusable commands, measured reports, no test tuning."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from safeanes.data import write_json
from safeanes.ensemble import run_ensemble
from safeanes.reporting import build_report
from safeanes.sequences import file_hash, load_dataset
from safeanes.training import TrainConfig, train_sequence


ROOT = Path(__file__).resolve().parents[1]


def summary(dataset, runs, report_root):
    meta, _, _, _ = load_dataset(dataset)
    lines = ["# UC04 v0.3 — kết quả mô hình mới và ensemble", "",
        "Thử nghiệm exploratory trên cùng pilot 60 ca; không phải kết quả paper hoặc final test mới.", "",
        "- Phương pháp và nguồn: [SOTA review](../../docs/SOURCES.md#review-e03).",
        "- Thiết kế khóa trước chạy: [kế hoạch v0.3](../../docs/versions/V0_3_PLAN.md).",
        f"- Dataset SHA-256: `{meta['windows_sha256']}`.",
        "- Numeric 2 giây/bước, history 600 giây; cùng patient split, calibration và policy với v0.2.",
        "- Bốn thành viên ensemble có trọng số logits 0,25, cố định trước kết quả; một calibrator chung fit riêng trên calibration patients.",
        "- Hai mô hình mới chạy CPU; chưa có benchmark GPU/T4. Mỗi mô hình mới một seed.", "",
        "## So sánh trên pilot_test", "",
        "| Mô hình | Phút | AUROC | AP | Event recall | PPV | FA/giờ | ECE | Phát hiện/đủ điều kiện | Đạt mọi mục tiêu? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    fmt = lambda x: "N/A" if x is None else f"{x:.3f}"
    results, validation_rows, resources = {}, [], []
    for label, run in runs.items():
        run = Path(run)
        report = json.loads((run / "report.json").read_text(encoding="utf-8"))
        env = json.loads((run / "environment.json").read_text(encoding="utf-8"))
        if env["dataset_hash"] != meta["windows_sha256"] or report["errors"]:
            raise ValueError("Comparison contains incompatible or incomplete runs")
        for key, result in report["models"].items():
            m = result["pilot_test"]
            row = [key, str(m["horizon_seconds"] // 60)] + [fmt(m[k]) for k in
                ("auroc", "average_precision", "event_sensitivity", "alarm_ppv", "false_alarms_per_hour", "ece")]
            row += [f"{m['events_detected']}/{m['events_eligible']}", "Có (pilot)" if result["gates"]["all_point_targets_met"] else "Chưa đạt"]
            lines.append("| " + " | ".join(row) + " |")
            results[key] = result
            v = result["selection_on_validation"]["metrics"]
            validation_rows.append("| " + " | ".join([key, fmt(v["threshold"]),
                f"{v['events_detected']}/{v['events_eligible']}", fmt(v["alarm_ppv"]), fmt(v["false_alarms_per_hour"]),
                result["selection_on_validation"]["selection"]]) + " |")
        if "resources" in report:
            r = report["resources"]
            resources.append(f"| {label} | {r['parameters']:,} | {r['epochs_completed']} / {r['best_epoch']} | {r['training_seconds']:.1f} |")
    lines += ["", "## Validation dùng để chọn ngưỡng", "",
        "| Model | Threshold | Event phát hiện/đủ điều kiện | PPV | FA/giờ | Trạng thái chọn |",
        "|---|---:|---:|---:|---:|---|", *validation_rows,
        "", "Threshold >1 nghĩa policy đã chọn không phát cảnh báo trong fallback; PPV N/A không phải PPV tốt.",
        "", "## Chi phí thực thi", "",
        "| Model | Tham số | Epoch chạy / chọn | Tổng giây các epoch |", "|---|---:|---:|---:|", *resources,
        "", "TCN/Transformer là số đo cũ v0.2, có lúc chạy đồng thời. Hai model v0.3 chạy tuần tự. Không dùng bảng này để kết luận tốc độ giữa kiến trúc trên cùng điều kiện. Ensemble tái sử dụng weights, thêm chi phí suy luận bốn model; không có epoch train mới.",
        "", "## Diễn giải", "",
        "- Đây là kiểm tra chuyển giao kiến trúc và ensemble. Không gán danh hiệu SOTA IOH từ AUROC hoặc thứ hạng một pilot.",
        "- Mọi run dùng 9 bệnh nhân pilot_test; chỉ có 3/4 event đủ điều kiện ở 5/10 phút. CI bootstrap 200 lần theo subjectid nằm trong [comparison.json](comparison.json) và báo cáo chi tiết.",
        "- Không chọn thành viên, trọng số, seed hoặc threshold bằng pilot_test; pilot_test đã được xem ở version trước nên kết quả vẫn exploratory.",
        "- FA/giờ còn xấp xỉ lưới 30 giây, và chưa hoàn tất audit cohort không tim. Không nới mục tiêu khi kết quả không đạt.",
        "", "## Báo cáo chi tiết", "",
        "- [Inception-style](inception/REPORT.md)", "- [TimesNet adaptation](timesnet/REPORT.md)",
        "- [Ensemble bốn model](ensemble/REPORT.md)",
        "- [TCN v0.2](../tcn_v1/REPORT.md)", "- [Transformer v0.2](../transformer_v1/REPORT.md)",
        "- [Baseline v0.1](../PILOT_BASELINE.md)",
        "", "## Version tiếp theo", "",
        "v0.4: ưu tiên audit/mở rộng cohort và tập calibration/validation, rồi ablation và ba seed. Nếu thay grid threshold hoặc exposure, chạy lại mọi baseline trên cùng protocol mới và báo riêng tác động của dữ liệu/evaluator. Chưa chuyển sang waveform chỉ vì mô hình mới không đạt."]
    report_root.mkdir(parents=True, exist_ok=True)
    write_json(report_root / "comparison.json", {"scope": "exploratory_pilot_not_final_test", "dataset_hash": meta["windows_sha256"],
        "models": results, "run_paths": {k: str(v) for k, v in runs.items()}})
    (report_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    dataset, sequences = ROOT / "data/pilot_v1", ROOT / "data/sequences_v1"
    output, reports = ROOT / "artifacts/v0_3", ROOT / "reports/v0_3"
    configs = {name: TrainConfig(**json.loads((ROOT / f"configs/v0_3/{name}.json").read_text()))
               for name in ("inception", "timesnet")}
    sources = list((ROOT / "src/safeanes").glob("*.py")) + [Path(__file__).resolve()]
    spec = {"version": "0.3.0", "plan_sha256": file_hash(ROOT / "docs/versions/V0_3_PLAN.md"),
        "dataset_sha256": file_hash(dataset / "windows.csv.gz"), "configs": {n: asdict(c) for n, c in configs.items()},
        "source_hashes": {p.name: file_hash(p) for p in sources},
        "frozen_components": ["tcn_v1", "transformer_v1", "v0_3/inception", "v0_3/timesnet"],
        "frozen_weights": [.25] * 4, "bootstrap_repeats": 200,
        "reused_checkpoints": {name: file_hash(ROOT / f"artifacts/{name}/model.pt") for name in ("tcn_v1", "transformer_v1")}}
    registration = output / "registration.json"
    if registration.exists():
        old = json.loads(registration.read_text())
        if old["spec"] != spec:
            raise ValueError("Registered version changed; make a new version rather than overwrite")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(registration, {"registered_at": datetime.now(timezone.utc).isoformat(), "spec": spec})
        snapshot = output / "source"
        snapshot.mkdir()
        for source in sources:
            shutil.copy2(source, snapshot / source.name)
    for name, config in configs.items():
        run = output / name
        if not (run / "report.json").exists():
            print(f"Training {name}", flush=True)
            train_sequence(dataset, sequences, run, config, repeats=200, resume=(run / "last.pt").exists())
    component_runs = [ROOT / f"artifacts/{name}" for name in spec["frozen_components"]]
    if not (output / "ensemble/report.json").exists():
        print("Evaluating frozen four-model ensemble", flush=True)
        run_ensemble(dataset, sequences, component_runs, output / "ensemble", spec["frozen_weights"], repeats=200)
    for name in ("inception", "timesnet", "ensemble"):
        if not (reports / name / "REPORT.md").exists():
            build_report(dataset, output / name, reports / name)
    runs = {"baseline": ROOT / "artifacts/pilot_v1", "lightgbm": ROOT / "artifacts/pilot_lightgbm_v1",
        "tcn": ROOT / "artifacts/tcn_v1", "transformer": ROOT / "artifacts/transformer_v1",
        **{n: output / n for n in ("inception", "timesnet", "ensemble")}}
    summary(dataset, runs, reports)
    print(f"Completed: {reports / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
