"""Regenerate aggregate Markdown from measured pilot artifacts; no invented metrics."""

import argparse
import json
from pathlib import Path

import pandas as pd


def fmt(value):
    return "N/A" if value is None else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/pilot_v1")
    parser.add_argument("--runs", nargs="+", default=["artifacts/pilot_v1", "artifacts/pilot_lightgbm_v1"])
    parser.add_argument("--out", default="reports/PILOT_BASELINE.md")
    args = parser.parse_args()
    dataset = Path(args.dataset)
    meta = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    quality = pd.read_csv(dataset / "quality.csv")
    models = {}
    for run in args.runs:
        report = json.loads((Path(run) / "report.json").read_text(encoding="utf-8"))
        if report["scope"] != "exploratory_pilot_not_final_test" or report["errors"]:
            raise ValueError(f"Invalid/incomplete pilot: {run}")
        env = json.loads((Path(run) / "environment.json").read_text(encoding="utf-8"))
        if env["dataset_hash"] != meta["windows_sha256"]:
            raise ValueError("Cannot compare runs using different datasets")
        for key, result in report["models"].items():
            if key in models:
                raise ValueError(f"Duplicate model results: {key}")
            models[key] = result
    roles = pd.read_csv(Path(args.runs[0]) / "pilot_roles.csv")
    lines = ["# Kết quả pilot UC04 trên dữ liệu thật", "",
        "Báo cáo sinh từ artifacts bằng `scripts/summarize_pilot.py`. Đây là **pilot bên trong global train**, không phải final test hoặc kiểm định lâm sàng.", "",
        f"- Cohort pilot: {meta['cases']} ca / {meta['subjects']} bệnh nhân.",
        f"- Dataset: {meta['windows']:,} decision windows, {meta['events']} đợt IOH theo protocol hiện tại.",
        f"- Độ phủ nhãn 1 giây trung bình theo ca: {quality.label_coverage.mean():.2%}.",
        f"- Protocol hash: `{meta['protocol_hash']}`.",
        f"- Dataset hash: `{meta['windows_sha256']}`.", "", "## Chia tập pilot", "",
        "| Vai trò | Ca | Bệnh nhân |", "|---|---:|---:|"]
    for role, group in roles.groupby("role"):
        lines.append(f"| {role} | {len(group)} | {group.subjectid.nunique()} |")
    lines += ["", "## Hiệu năng tại ngưỡng chọn trên validation", "",
        "Các threshold thuộc chế độ nghiên cứu; giữ nguyên khi chạy pilot_test. Mọi số ở đây là kết quả đo, không phải số lấy từ paper.", "",
        "| Model / horizon (giây) | AUROC | AP | Event sensitivity | Alarm PPV | False alarms/giờ | Event phát hiện/đủ điều kiện | Đạt mọi mục tiêu điểm? |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for key, result in models.items():
        m = result["pilot_test"]
        entries = [key] + [fmt(m[k]) for k in ("auroc", "average_precision", "event_sensitivity", "alarm_ppv", "false_alarms_per_hour")]
        entries += [f"{m['events_detected']}/{m['events_eligible']}", "Có (pilot)" if result["gates"]["all_point_targets_met"] else "Chưa đạt"]
        lines.append("| " + " | ".join(entries) + " |")
    lines += ["", "## Giới hạn phải đọc cùng kết quả", "",
        "- Cỡ mẫu pilot_test nhỏ; xem số event đủ điều kiện, không chỉ số window. Khoảng tin cậy 95% và số bootstrap replicate hợp lệ nằm trong `report.json` của từng run.",
        "- N/A PPV nghĩa không có alarm đánh giá được, không có nghĩa PPV bằng 100%.",
        "- AUROC cao chỉ phản ánh thứ hạng nguy cơ; còn cần độ nhạy và PPV ở ngưỡng cảnh báo đã khóa. Chưa thể tuyên bố đạt yêu cầu chỉ nhờ AUROC.",
        "- Eligibility, history, censoring và recovery làm một số event không có cơ hội dự báo. Metrics lưu cả `events_all` và `event_sensitivity_all` để công bố phần này.",
        "- FA/hour hiện xấp xỉ trên lưới 30 giây; xem [protocol](../docs/PROTOCOL.md). Chưa dùng metric này để nghiệm thu lâm sàng.",
        "- Báo cáo này giữ kết quả baseline CPU. Kết quả DL bổ sung: [TCN](tcn_v1/REPORT.md), [Transformer](transformer_v1/REPORT.md); chưa có benchmark T4.", "",
        "## Bước tiếp theo dựa trên pilot", "",
        "1. Audit các event không đủ cơ hội dự báo và alarm bị censored; rà soát nhãn bằng timeline MAP gốc.",
        "2. Mở rộng pilot trong global train để calibration/validation có nhiều bệnh nhân và biến cố hơn; giữ final test chưa mở.",
        "3. Cải thiện evaluator exposure từng giây và kiểm tra độ nhạy với recovery/persistence/cooldown trong development; không hạ mục tiêu chất lượng.",
        "4. TCN và sequence loader đã được bổ sung với cùng split/nhãn/evaluator; cần mở rộng dữ liệu, đo VRAM và thời gian trên T4 trước kết luận giá trị tăng thêm.", "",
        "## Nguồn của phương pháp", "",
        "Dữ liệu từ [VitalDB](https://doi.org/10.1038/s41597-022-01411-5); API theo [tài liệu chính thức](https://vitaldb.net/docs/?documentId=API%2FWeb_API_OpenDataset.md). Đánh giá liên tục được thúc đẩy bởi nghiên cứu [selection bias của Yang et al.](https://pubmed.ncbi.nlm.nih.gov/40404499/). Calibration dựa trên [scikit-learn](https://scikit-learn.org/stable/modules/calibration.html). Các quy tắc cụ thể là thiết kế dự án, được phân biệt trong [sổ nguồn](../docs/SOURCES.md).", "",
        "Artifacts dùng để sinh báo cáo:", ""]
    lines += [f"- `{run}`" for run in args.runs]
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
