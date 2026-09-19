# DEPRECATED — E08 v1. Giữ lại để truy vết, KHÔNG dùng số liệu sinh ra từ file này.
# Audit 19/09/2026 tìm 6 lỗi trong v1; dùng scripts/e08/run_comparison.py thay thế.
# Chi tiết từng lỗi và bằng chứng: scripts/e08/README.md
"""E08 Giai đoạn 4: gộp lại, kiểm tra đủ 5 gate cùng lúc trên đường cong ensemble VOTE.

Dùng lại ensemble VOTE (giống Giai đoạn 2-3, đã đổi khỏi trung bình cộng vì nhạy với việc đổi
model thành phần) và quét qua 6 mức đồng thuận (k/6), kiểm tra ĐỦ 5 gate ở mục 8 (AUROC,
event_sensitivity, alarm_ppv, false_alarms_per_hour, ece — coverage tách riêng) cùng lúc, không
chỉ tối ưu 1 metric. Nếu không có mức nào đạt đủ 5 gate (dự kiến, theo Giai đoạn 1-3 đã cho thấy
trade-off mạnh), báo cáo mức gần đạt nhất theo đúng thứ tự ưu tiên đã chọn: recall > PPV > FA/giờ.
Đây vẫn là validation, không phải holdout/test mới; không dùng kết quả này để mở test.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from safeanes.config import Protocol
from safeanes.evaluation import evaluate_predictions, quality_gates

ROOT = Path(__file__).resolve().parents[4]
MODELS = ["tabm_20260917", "tabm_20260918", "tabm_20260919", "lightgbm_gaussian_jitter",
          "map_all_20260917", "catboost_balanced_plus_jitter"]
E05_MODELS = {"map_all_20260917"}
E08_MODELS = {"lightgbm_gaussian_jitter", "catboost_balanced_plus_jitter"}
HORIZONS = (300, 600)
META = ["caseid", "subjectid", "time", "eligible", "exposure_seconds", "y_300", "y_600",
        "historical_subject", "role"]
GATE_KEYS = ("auroc", "event_sensitivity", "alarm_ppv", "prediction_coverage",
             "false_alarms_per_hour", "ece")


def load_validation(name, horizon):
    if name in E05_MODELS:
        return pd.read_csv(ROOT / "artifacts/E05" / f"{name}_{horizon}" / "validation.csv.gz")
    if name in E08_MODELS:
        return pd.read_csv(ROOT / "artifacts/E08/version/v1" / name / f"validation_{horizon}.csv.gz")
    return pd.read_csv(ROOT / "artifacts/E06" / name / f"validation_{horizon}.csv.gz")


def ceiling_thresholds():
    frame = pd.read_csv(ROOT / "reports/E08/version/v1/stage1_recall_ceiling.csv")
    frame = frame[frame.model.isin(MODELS)]
    return {(row.model, row.horizon): row.threshold_ceiling for row in frame.itertuples()}


def ensemble_vote_score(horizon, thresholds):
    frames = {m: load_validation(m, horizon) for m in MODELS}
    base = frames[MODELS[0]][META].reset_index(drop=True)
    for m in MODELS[1:]:
        other = frames[m]
        if not other[["caseid", "time"]].reset_index(drop=True).equals(base[["caseid", "time"]]):
            raise ValueError(f"Row order mismatch for {m} at horizon {horizon}")
    votes = pd.Series(0.0, index=base.index)
    for m in MODELS:
        p = frames[m]["probability"].reset_index(drop=True)
        flag = (p >= thresholds[(m, horizon)]).astype(float)
        flag[p.isna()] = 0.0
        votes = votes + flag
    base["probability"] = (votes / len(MODELS)).where(base.eligible.to_numpy(), other=np.nan)
    return base


def main():
    data = ROOT / "data/development300"
    meta = json.loads((data / "dataset.json").read_text())
    protocol = Protocol(**{**meta["protocol"], "horizons_seconds": tuple(meta["protocol"]["horizons_seconds"])})
    events = pd.read_csv(data / "events.csv")
    out_dir = ROOT / "reports/E08/version/v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = ceiling_thresholds()

    rows = []
    for h in HORIZONS:
        frame = ensemble_vote_score(h, thresholds)
        for k in range(1, len(MODELS) + 1):
            t = k / len(MODELS)
            m, _, _ = evaluate_predictions(frame, events, h, t, protocol)
            gates = quality_gates(m, h)
            rows.append({"horizon": h, "required_votes": k, "total_models": len(MODELS), "threshold": t,
                         "gates_met": sum(gates["criteria"].values()),
                         "all_gates_met": gates["all_point_targets_met"], **{k2: m[k2] for k2 in
                         ("auroc", "event_sensitivity", "alarm_ppv", "false_alarms_per_hour",
                          "prediction_coverage", "ece")}})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "stage4_combined_curve.csv", index=False)

    best_rows = []
    for h in HORIZONS:
        subset = frame[frame.horizon.eq(h)]
        passing = subset[subset.all_gates_met]
        if len(passing):
            best = passing.iloc[0]
            status = "all_5_gates_met"
        else:
            best = subset.sort_values(
                by=["gates_met", "event_sensitivity", "alarm_ppv", "false_alarms_per_hour"],
                ascending=[False, False, False, True]).iloc[0]
            status = "no_threshold_meets_all_5_gates"
        best_rows.append({"horizon": h, "status": status, **best.to_dict()})
    best = pd.DataFrame(best_rows)
    best.to_csv(out_dir / "stage4_best_point.csv", index=False)
    print(best[["horizon", "status", "threshold", "gates_met", "event_sensitivity", "alarm_ppv",
                "false_alarms_per_hour", "auroc"]].to_string(index=False))
    append_report(best, out_dir)


def append_report(best, out_dir):
    fmt = lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.3f}"
    report_path = out_dir / "REPORT.md"
    existing = report_path.read_text(encoding="utf-8")
    marker = "\n## Giai đoạn 4"
    if marker in existing:
        existing = existing[: existing.index(marker)]
    lines = ["", "## Giai đoạn 4 — gộp lại, kiểm tra đủ 5 gate cùng lúc", "",
        "Quét qua 6 mức đồng thuận trên ensemble VOTE (6 model đa dạng, giống Giai đoạn 2-3), kiểm tra "
        "ĐỦ 5 gate mục 8 (AUROC, event_sensitivity, alarm_ppv, false_alarms_per_hour, ece; coverage "
        "0,90 tính riêng) tại từng mức — không chỉ tối ưu 1 metric như Giai đoạn 1–3. Nếu không mức nào "
        "đạt đủ 5, chọn mức gần nhất theo đúng thứ tự ưu tiên đã dùng xuyên suốt E08: số gate đạt được "
        "→ recall → PPV → FA/giờ (thấp hơn tốt hơn). Vẫn là validation, chưa mở test.",
        "", "| Phút | Trạng thái | Đồng thuận | Số gate đạt/6 | Recall | PPV | FA/giờ | AUROC |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in best.itertuples():
        lines.append(f"| {r.horizon // 60} | {r.status} | {int(r.required_votes)}/{int(r.total_models)} | "
                      f"{int(r.gates_met)}/6 | {fmt(r.event_sensitivity)} | {fmt(r.alarm_ppv)} | "
                      f"{fmt(r.false_alarms_per_hour)} | {fmt(r.auroc)} |")
    lines += ["", "## Kết luận Giai đoạn 4", ""]
    for r in best.itertuples():
        if r.status == "all_5_gates_met":
            lines.append(f"- {r.horizon // 60} phút: **có ngưỡng đạt đủ 5 gate** ({fmt(r.threshold)}) — "
                         "cần xác nhận lại trên holdout/E07 trước khi khóa, đây vẫn là validation.")
        else:
            lines.append(f"- {r.horizon // 60} phút: **không có ngưỡng nào trên ensemble 6 model này "
                         f"đạt đủ 5 gate cùng lúc** (tốt nhất {int(r.gates_met)}/6, tại ngưỡng "
                         f"{r.threshold:.3f}: recall {r.event_sensitivity:.3f}, PPV {r.alarm_ppv:.3f}, "
                         f"FA/giờ {r.false_alarms_per_hour:.3f}). Xác nhận lại phát hiện của Giai đoạn "
                         "1-3: recall cao và FA/giờ thấp đối kháng quá mạnh với 6 model MAP-based hiện "
                         "có trên 300 ca; cần tín hiệu mới (waveform) hoặc cỡ mẫu lớn hơn (E07), không "
                         "phải chỉnh ngưỡng/ensemble thêm trên cùng model.")
    lines += ["", "Đây là điểm dừng hợp lý của vòng E08 dựa trên development300: đã trả lời câu hỏi "
        "'tăng từng metric riêng có được không' (được, xem Giai đoạn 1-3) và 'gộp lại có đạt cả 5 gate "
        "không' (số liệu ở trên). Bước tiếp theo nằm ngoài phạm vi chỉnh ngưỡng: chờ E07 (cỡ mẫu lớn "
        "hơn, CI hẹp hơn) hoặc mở nhánh waveform D3 đã nêu ở Giai đoạn 1.", "",
        "[Tạo bởi scripts/e08/version/v1_sequential_stages/stage4_combine.py](../../../../scripts/e08/version/v1_sequential_stages/stage4_combine.py) — không "
        "train lại, không đổi threshold/bundle đã khóa của E05/E06 hoặc kết quả Giai đoạn 1–3."]
    report_path.write_text(existing.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
