"""Readable E06 report with unchanged E05 comparators; no test-based selection."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "reports/E06"
    table = pd.read_csv(directory / "comparison.csv")
    verification = json.loads((directory / "verification.json").read_text())
    selection = json.loads((ROOT / "artifacts/E06/validation_selection.json").read_text())
    primary = table[table.scope.eq("new_patients") & table.policy.eq("fixed")]
    assert len(table) == 60 and len(primary) == 10, "Wait for all registered models/scopes"
    assert len(verification["checks"]) == 4
    old = pd.read_csv(ROOT / "reports/E05/comparison.csv")
    old = old[old.scope.eq("new_patients") & old.policy.eq("fixed")]
    fmt = lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.3f}"
    lines = ["# E06 — TabM và phương án thay thế trong v0.2", "",
        "Đã chạy TabM+PLE chính thức ba seed, ensemble xác suất đều và LightGBM numeric có regularization/ràng buộc MAP. "
        "Dùng lại 300 ca/297 bệnh nhân của E05; bảng chính là nhóm 37 bệnh nhân/35–37 event từng được gọi là nhóm mới ở E05. "
        "**Đây là đánh giá thăm dò trên holdout đã xem, không phải bằng chứng độc lập hay SOTA IOH.**", "",
        "[Review phương pháp](../../docs/SOTA_E06.md) · [Plan](../../docs/experiments/E06_PLAN.md) · "
        "[CSV mọi scope/policy](comparison.csv) · [JSON và CI](results.json) · [Kiểm chứng](verification.json)", "",
        "## Chọn bằng validation trước khi đánh giá holdout", "",
        "| Horizon | Đề xuất khóa trước holdout | Validation recall | PPV | FA/giờ |", "|---|---|---:|---:|---:|"]
    for h, name in selection["selected"].items():
        m = selection["selections"][f"{name}_{h}_fixed"]["metrics"]
        lines.append(f"| {int(h)//60} phút | {name} | {fmt(m['event_sensitivity'])} | {fmt(m['alarm_ppv'])} | {fmt(m['false_alarms_per_hour'])} |")
    lines += ["", "Đề xuất chỉ chọn giữa ensemble TabM và LightGBM mới; không chọn seed theo holdout. "
              "Đối chiếu MAP và CatBoost dưới đây vẫn cần thiết để quyết định có đáng nghiên cứu tiếp hay không.", "",
              "## Cùng holdout và ngưỡng primary", "",
              "| Model | Phút | AUROC | AP | Phát hiện/event | Recall | PPV | FA/giờ | Đủ gate |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    rows = []
    for prefix in ("map_all_20260917", "catboost_all_20260917", "catboost_numeric_20260917"):
        for h in (300, 600):
            r = old[old.model.eq(f"{prefix}_{h}")].iloc[0].to_dict()
            r.update(model=f"E05/{prefix}", horizon=h)
            rows.append(r)
    rows.extend(primary.to_dict("records"))
    for r in rows:
        lines.append("| " + " | ".join([r["model"], str(int(r["horizon"])//60), fmt(r["auroc"]),
            fmt(r["average_precision"]), f"{int(r['events_detected'])}/{int(r['events_eligible'])}",
            fmt(r["event_sensitivity"]), fmt(r["alarm_ppv"]), fmt(r["false_alarms_per_hour"]), str(r["all_gates"])]) + " |")
    lines += ["", "## Trung bình ba seed, không chọn seed tốt nhất", "",
              "| Model | Phút | Recall | PPV | FA/giờ | AUROC |", "|---|---:|---:|---:|---:|---:|"]
    seed_summary = []
    for h in (300, 600):
        groups = (("CatBoost E05", old[old.model.isin([f"catboost_all_{s}_{h}" for s in (20260917, 20260918, 20260919)])]),
                  ("TabM E06", primary[primary.horizon.eq(h) & primary.model.isin([f"tabm_{s}" for s in (20260917, 20260918, 20260919)])]))
        for name, group in groups:
            assert len(group) == 3
            metrics = {k: float(group[k].mean()) for k in ("event_sensitivity", "alarm_ppv", "false_alarms_per_hour", "auroc")}
            seed_summary.append({"model": name, "horizon": h, **metrics})
            lines.append("| " + " | ".join([name, str(h//60), *[fmt(v) for v in metrics.values()]]) + " |")
    (directory / "seed_summary.json").write_text(json.dumps(seed_summary, indent=2) + "\n", encoding="utf-8")
    lines += ["", "## Mức thay đổi của đề xuất validation so với CatBoost all-features seed 20260917", ""]
    for h, name in selection["selected"].items():
        r = primary[primary.model.eq(name) & primary.horizon.eq(int(h))].iloc[0]
        b = old[old.model.eq(f"catboost_all_20260917_{h}")].iloc[0]
        lines.append(f"- {int(h)//60} phút: recall {b.event_sensitivity:.3f} → {r.event_sensitivity:.3f}; "
                     f"PPV {b.alarm_ppv:.3f} → {r.alarm_ppv:.3f}; FA/giờ {b.false_alarms_per_hour:.3f} → {r.false_alarms_per_hour:.3f}; "
                     f"AUROC {b.auroc:.3f} → {r.auroc:.3f}.")
    lines += ["", f"Có {int(primary.all_gates.sum())}/{len(primary)} model/horizon primary đạt toàn bộ gate. "
        "Thay đổi điểm số không tự chứng minh ưu thế có ý nghĩa thống kê; CI trong JSON là CI từng model, không phải CI chênh lệch ghép cặp.", "",
        "## Kết luận thực nghiệm", "",
        "- TabM ba seed cải thiện trung bình ở 10 phút so với CatBoost all-features ba seed: "
        "recall 0.505 → 0.604, PPV 0.281 → 0.337, FA/giờ 0.808 → 0.746. "
        "Ensemble TabM cố định đạt recall 0.649, PPV 0.361, FA/giờ 0.715; đây là ứng viên cho kiểm định tiếp.",
        "- Ở 5 phút, TabM ba seed tăng recall 0.410 → 0.514 nhưng FA/giờ tăng 0.737 → 0.903; "
        "không gọi là cải thiện đồng đều hoặc đã giải quyết báo động giả.",
        "- Đề xuất LightGBM khóa bằng validation không giữ được lợi thế trên holdout: "
        "ở 5 phút PPV giảm và FA/giờ tăng so với CatBoost; ở 10 phút recall giảm so với trung bình CatBoost. "
        "Không nâng nó thành mặc định và không thay đề xuất đã khóa bằng một model thắng trên holdout.",
        "- Giữ các model như ứng viên nghiên cứu. Cần validation lớn hơn hoặc đánh giá theo nhóm bệnh nhân nhiều fold "
        "trước khi khóa lựa chọn cho holdout chưa từng xem; không tiếp tục tune trên cùng bảng kết quả này.", "",
        "## Kiểm chứng và giới hạn", "",
        "- Source/config/dataset/events/roles hashes được đăng ký; roles khớp E05, chỉ global train. "
        "Mọi ngưỡng đã đối chiếu fast replay với evaluator tham chiếu.",
        "- Nạp lại cả bốn bundle và kiểm tra batch nhỏ tái tạo dự báo. 100% cửa sổ đủ điều kiện có xác suất hữu hạn; "
        "không có p10 < p5 sau calibration/projection. Ensemble đều giữ thứ tự này.",
        "- TabM dùng inner stopping tách bệnh nhân trong FIT và không refit; LightGBM dùng toàn bộ FIT. "
        "Do đó đây là so sánh pipeline đã đăng ký, không phải ablation chỉ riêng kiến trúc.",
        "- Mỗi TabM có 16 thành viên thay vì cấu hình 32 trong paper; PLE/head/training budget là cấu hình dự án. "
        "Không tái lập toàn bộ paper và không suy điểm IOH từ benchmark bảng.",
        "- Calibration tách bệnh nhân; projection xác suất hai horizon là thay đổi được đăng ký trước E06. "
        "Không thay nhãn, policy cảnh báo, budget hoặc denominator E05 để làm đẹp số.",
        "- CI primary 200 bootstrap theo bệnh nhân; adaptive là secondary. Nhóm cũ/gộp đều có trong CSV. "
        "Coverage gate v1 vẫn là eligible/scheduled, khác availability của model; exposure còn xấp xỉ 30 giây.",
        "- Final test chưa mở; chưa benchmark CUDA/T4, waveform, kiểm định ngoài hoặc tiến cứu. "
        "TabICLv2/TabPFN-3.5 đã rà soát nguồn, chưa benchmark SafeAnes.", "",
        "Phần cứng có RTX 3050 Laptop 4 GB (nvidia-smi); môi trường hiện dùng PyTorch 2.14.0+cpu. "
        "E06 thực thi CPU, không suy rằng máy không có GPU và không gọi đây là benchmark T4.", "",
        "## Chạy lại", "", "```powershell", '$env:PYTHONPATH = "$PWD/src;$PWD/.local_deps"',
        'python -m pip install -e ".[tabular,dev]"', "python scripts/run_sota.py", "python scripts/report_sota.py", "```", "",
        "Runner tiếp tục từ bundle hoàn tất nếu hashes không đổi; training bị ngắt giữa model chạy lại model đó. "
        "Khi thay code/config, dùng đợt thí nghiệm mới; không ghi đè E05/E06 đã khóa. Không tăng release hoặc push."]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(primary[["model", "horizon", "event_sensitivity", "alarm_ppv", "false_alarms_per_hour", "auroc"]].to_string(index=False))


if __name__ == "__main__":
    main()
