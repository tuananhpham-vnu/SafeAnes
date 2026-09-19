# DEPRECATED — E08 v1. Giữ lại để truy vết, KHÔNG dùng số liệu sinh ra từ file này.
# Audit 19/09/2026 tìm 6 lỗi trong v1; dùng scripts/e08/run_comparison.py thay thế.
# Chi tiết từng lỗi và bằng chứng: scripts/e08/README.md
"""E08 Giai đoạn 1, phương pháp #1: trần recall bằng cách hạ ngưỡng (không train lại model).

Dùng lại threshold curve đã tính sẵn trên validation của E05/E06 (không tính lại, không đổi
model/bundle). Với mỗi model/horizon, tìm ứng viên có event_sensitivity cao nhất trên toàn bộ
lưới ngưỡng đã quét (bỏ qua ràng buộc ngân sách FA/giờ và gate), để biết "trần recall" mà model
hiện tại có thể đạt được chỉ bằng cách đổi ngưỡng, đối lập với điểm vận hành đã chọn (ràng buộc
budget) trong validation_selection.json/comparison.csv. Đây là dữ liệu validation đã dùng để chọn
model ở E05/E06 (không phải holdout mới); chỉ là chẩn đoán, không phải kết quả xác nhận độc lập.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]

E05_MODELS = ["map_all_20260917", "logistic_all_20260917", "lightgbm_all_20260917",
              "catboost_all_20260917", "catboost_numeric_20260917", "catboost_map_only_20260917"]
E06_MODELS = ["tabm_ensemble", "lightgbm_monotone", "tabm_20260917", "tabm_20260918", "tabm_20260919"]
HORIZONS = (300, 600)


def load_e05_curve(name, horizon):
    path = ROOT / "artifacts/E05" / f"{name}_{horizon}" / "fixed_curve.json"
    return json.loads(path.read_text())


def load_e06_curve(name, horizon):
    path = ROOT / "artifacts/E06" / name / f"curve_{horizon}_fixed.json"
    return json.loads(path.read_text())


def recall_ceiling(curve):
    valid = [c for c in curve if c["metrics"]["event_sensitivity"] is not None]
    return max(valid, key=lambda c: c["metrics"]["event_sensitivity"])


def current_operating_point(model_prefix, horizon, source):
    if source == "e05":
        table = pd.read_csv(ROOT / "reports/E05/comparison.csv")
        row = table[(table.model.eq(f"{model_prefix}_{horizon}")) & (table.policy.eq("fixed")) &
                     (table.scope.eq("new_patients"))]
    else:
        table = pd.read_csv(ROOT / "reports/E06/comparison.csv")
        row = table[(table.model.eq(model_prefix)) & (table.horizon.eq(horizon)) &
                     (table.policy.eq("fixed")) & (table.scope.eq("new_patients"))]
    return None if row.empty else row.iloc[0].to_dict()


def main():
    rows = []
    for name in E05_MODELS:
        for h in HORIZONS:
            curve = load_e05_curve(name, h)
            ceiling = recall_ceiling(curve)
            current = current_operating_point(name, h, "e05")
            rows.append(build_row("E05", name, h, ceiling, current))
    for name in E06_MODELS:
        for h in HORIZONS:
            curve = load_e06_curve(name, h)
            ceiling = recall_ceiling(curve)
            current = current_operating_point(name, h, "e06")
            rows.append(build_row("E06", name, h, ceiling, current))
    frame = pd.DataFrame(rows)
    out_dir = ROOT / "reports/E08/version/v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "stage1_recall_ceiling.csv", index=False)
    write_report(frame, out_dir)
    print(frame[["experiment", "model", "horizon", "recall_current", "recall_ceiling",
                  "ppv_ceiling", "fa_per_hour_ceiling", "threshold_ceiling"]].to_string(index=False))


def build_row(experiment, name, horizon, ceiling, current):
    m = ceiling["metrics"]
    return {"experiment": experiment, "model": name, "horizon": horizon,
            "recall_current": None if current is None else current.get("event_sensitivity"),
            "ppv_current": None if current is None else current.get("alarm_ppv"),
            "fa_per_hour_current": None if current is None else current.get("false_alarms_per_hour"),
            "threshold_current": None if current is None else current.get("threshold"),
            "recall_ceiling": m["event_sensitivity"], "ppv_ceiling": m["alarm_ppv"],
            "fa_per_hour_ceiling": m["false_alarms_per_hour"], "threshold_ceiling": m["threshold"],
            "auroc": m["auroc"], "prediction_coverage": m["prediction_coverage"],
            "events_eligible": m["events_eligible"], "events_detected_at_ceiling": m["events_detected"]}


def write_report(frame, out_dir):
    fmt = lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.3f}"
    lines = ["# E08 Giai đoạn 1 — trần recall theo ngưỡng, chưa đổi model (chẩn đoán)", "",
        "Dùng curve validation đã có từ E05/E06 (không tính lại, không đổi bundle/threshold đã khóa "
        "của hai đợt đó). 'recall_ceiling' là event_sensitivity lớn nhất trên toàn bộ lưới 41 ngưỡng "
        "(0,01 đến ≥1) trên tập validation đang dùng để chọn model ở E05/E06, bỏ qua ràng buộc FA/giờ "
        "≤0,5 và các gate khác. 'recall_current' là điểm vận hành đã chọn trước đó (có ràng buộc "
        "budget). Đây là dữ liệu validation đã xem cho lựa chọn model, không phải holdout/test mới và "
        "không phải bằng chứng xác nhận độc lập; chỉ để trả lời câu hỏi 'ngưỡng có đủ để đạt recall "
        "mục tiêu hay không'.", "",
        "| Nguồn | Model | Phút | Recall (điểm hiện tại) | Recall (trần, đổi ngưỡng) | PPV tại trần | FA/giờ tại trần | Ngưỡng tại trần |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in frame.itertuples():
        lines.append("| " + " | ".join([r.experiment, r.model, str(r.horizon // 60),
            fmt(r.recall_current), fmt(r.recall_ceiling), fmt(r.ppv_ceiling),
            fmt(r.fa_per_hour_ceiling), fmt(r.threshold_ceiling)]) + " |")
    best5 = frame[frame.horizon.eq(300)].recall_ceiling.max()
    best10 = frame[frame.horizon.eq(600)].recall_ceiling.max()
    lines += ["",
        f"Trần recall cao nhất quan sát được trong nhóm model hiện có: 5 phút {best5:.3f}, "
        f"10 phút {best10:.3f}. Mục tiêu mục 8 là ≥0,90 (5 phút) và ≥0,85 (10 phút).",
        "",
        "## Đọc số này thế nào",
        "",
        "- **5 phút: trần recall của MỌI model đều dưới mục tiêu 0,90** (cao nhất 0,826, catboost_map_only) "
        "— chỉ đổi ngưỡng không thể đạt gate recall ở horizon này với các model hiện có. Cần cải thiện "
        "discrimination của model (loss, dữ liệu, đặc trưng — mục Giai đoạn 1 của E08_PLAN.md) hoặc kết "
        "hợp nhiều model (OR alarms), không chỉ tune threshold.",
        "- **10 phút: trần recall đã vượt mục tiêu 0,85 ở hầu hết model** (catboost_map_only đạt 1,000; "
        "hầu hết ≥0,90) — ở horizon này, threshold hiện tại (map_all_20260917 chỉ 0,730) đang chọn quá "
        "thận trọng so với khả năng thật của model. Có thể tăng recall 10 phút đáng kể chỉ bằng cách hạ "
        "ngưỡng, không cần model mới.",
        "- FA/giờ tại điểm trần recall rất cao (2–4/giờ, gấp 4–8 lần ngân sách 0,5) — đây là chi phí đã "
        "biết trước, chưa đánh giá ở giai đoạn này theo đúng nguyên tắc Giai đoạn 1 của E08_PLAN.md "
        "(chấp nhận PPV/FA xấu hơn tạm thời, xử lý ở Giai đoạn 2–3).",
        "- Bước tiếp theo trong E08_PLAN.md Giai đoạn 1: với horizon 5 phút, thử class-weighted/focal "
        "loss, oversampling, hoặc OR-ensemble giữa các model ở đây (chưa chạy trong script này) để nâng "
        "trần recall lên gần 0,90 trước khi sang Giai đoạn 2.",
        "- Ngưỡng đạt trần không luôn là ngưỡng thấp nhất trong lưới (0,01): do alarm persistence/"
        "cooldown, ngưỡng quá thấp có thể giữ trạng thái 'active' liên tục (probability hiếm khi tụt "
        "dưới 0,01) nên không tạo lại cảnh báo mới sau lần đầu; một ngưỡng nhỏ hơn 0,01 một chút lại tạo "
        "nhiều cảnh báo hơn và do đó recall cao hơn. Đây là tính chất của chính sách persistence/"
        "cooldown, không phải lỗi.", "",
        "[Tạo bởi scripts/e08/version/v1_sequential_stages/stage1_recall.py](../../../../scripts/e08/version/v1_sequential_stages/stage1_recall.py) — đọc lại "
        "curve_*.json/fixed_curve.json đã có, không train lại, không đổi artifact E05/E06."]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
