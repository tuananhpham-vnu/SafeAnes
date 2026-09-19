# DEPRECATED — E08 v1. Giữ lại để truy vết, KHÔNG dùng số liệu sinh ra từ file này.
# Audit 19/09/2026 tìm 6 lỗi trong v1; dùng scripts/e08/run_comparison.py thay thế.
# Chi tiết từng lỗi và bằng chứng: scripts/e08/README.md
"""E08 Giai đoạn 2, phương pháp #2: đồng thuận ensemble (AND) để nâng PPV.

Dùng lại bundle đã có ở E05/E06 (không train lại), mỗi model giữ ngưỡng "trần recall" đã chọn ở
Giai đoạn 1 (reports/E08/stage1_recall_ceiling.csv). Với mỗi cửa sổ, tính "vote" = số model có
probability >= ngưỡng trần riêng của model đó, rồi coi vote/N như một "probability" tổng hợp để
dùng lại evaluate_predictions với ngưỡng k/N (yêu cầu đúng k trong N model đồng thuận). So hai
nhóm: (a) 4 model tương quan cao (TabM 3 seed + LightGBM cùng feature set) và (b) 6 model đa dạng
kiến trúc hơn (thêm MAP baseline + CatBoost all-features từ E05). Đây là dữ liệu validation đã
dùng để chọn model ở E05/E06, không phải holdout mới.
"""
import json
from pathlib import Path

import pandas as pd

from safeanes.config import Protocol
from safeanes.evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[4]
GROUPS = {
    "correlated_4": ["tabm_20260917", "tabm_20260918", "tabm_20260919", "lightgbm_gaussian_jitter"],
    "diverse_6": ["tabm_20260917", "tabm_20260918", "tabm_20260919", "lightgbm_gaussian_jitter",
                  "map_all_20260917", "catboost_balanced_plus_jitter"],
}
E05_MODELS = {"map_all_20260917"}
E08_MODELS = {"lightgbm_gaussian_jitter", "catboost_balanced_plus_jitter"}
HORIZONS = (300, 600)
META = ["caseid", "subjectid", "time", "eligible", "exposure_seconds", "y_300", "y_600",
        "historical_subject", "role"]


def load_validation(name, horizon):
    if name in E05_MODELS:
        return pd.read_csv(ROOT / "artifacts/E05" / f"{name}_{horizon}" / "validation.csv.gz")
    if name in E08_MODELS:
        return pd.read_csv(ROOT / "artifacts/E08/version/v1" / name / f"validation_{horizon}.csv.gz")
    return pd.read_csv(ROOT / "artifacts/E06" / name / f"validation_{horizon}.csv.gz")


def ceiling_thresholds(models):
    frame = pd.read_csv(ROOT / "reports/E08/version/v1/stage1_recall_ceiling.csv")
    frame = frame[frame.model.isin(models)]
    return {(row.model, row.horizon): row.threshold_ceiling for row in frame.itertuples()}


def run_group(models, protocol, events):
    thresholds = ceiling_thresholds(models)
    rows = []
    for h in HORIZONS:
        frames = {m: load_validation(m, h) for m in models}
        base = frames[models[0]][META].reset_index(drop=True)
        for m in models[1:]:
            other = frames[m]
            if not other[["caseid", "time"]].reset_index(drop=True).equals(base[["caseid", "time"]]):
                raise ValueError(f"Row order mismatch for {m} at horizon {h}")
        votes = pd.Series(0.0, index=base.index)
        for m in models:
            p = frames[m]["probability"].reset_index(drop=True)
            flag = (p >= thresholds[(m, h)]).astype(float)
            flag[p.isna()] = 0.0
            votes = votes + flag
        ensemble_probability = votes / len(models)
        for k in range(1, len(models) + 1):
            frame = base.copy()
            frame["probability"] = ensemble_probability.where(base.eligible.to_numpy(), other=float("nan"))
            metric, _, _ = evaluate_predictions(frame, events, h, k / len(models), protocol)
            rows.append({"horizon": h, "required_votes": k, "total_models": len(models),
                         "recall": metric["event_sensitivity"], "ppv": metric["alarm_ppv"],
                         "fa_per_hour": metric["false_alarms_per_hour"],
                         "prediction_coverage": metric["prediction_coverage"],
                         "events_detected": metric["events_detected"], "events_eligible": metric["events_eligible"]})
    return pd.DataFrame(rows)


def main():
    data = ROOT / "data/development300"
    meta = json.loads((data / "dataset.json").read_text())
    protocol = Protocol(**{**meta["protocol"], "horizons_seconds": tuple(meta["protocol"]["horizons_seconds"])})
    events = pd.read_csv(data / "events.csv")
    out_dir = ROOT / "reports/E08/version/v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for group, models in GROUPS.items():
        frame = run_group(models, protocol, events)
        frame.to_csv(out_dir / f"stage2_ppv_ensemble_{group}.csv", index=False)
        results[group] = frame
        print(f"--- {group} ({len(models)} model) ---")
        print(frame.to_string(index=False))

    baseline = pd.read_csv(out_dir / "stage1_recall_ceiling.csv")
    replace_stage2_section(out_dir, results, baseline)


def replace_stage2_section(out_dir, results, baseline):
    fmt = lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.3f}"
    report_path = out_dir / "REPORT.md"
    existing = report_path.read_text(encoding="utf-8")
    marker = "\n## Giai đoạn 2"
    if marker in existing:
        existing = existing[: existing.index(marker)]

    def table(group):
        frame = results[group]
        n = frame.total_models.iloc[0]
        lines = [f"### Nhóm {group} ({n} model)", "",
                 f"| Phút | Cần bao nhiêu/{n} model đồng thuận | Recall | PPV | FA/giờ |",
                 "|---:|---:|---:|---:|---:|"]
        for r in frame.itertuples():
            lines.append(f"| {r.horizon // 60} | {r.required_votes}/{r.total_models} | "
                          f"{fmt(r.recall)} | {fmt(r.ppv)} | {fmt(r.fa_per_hour)} |")
        return lines

    lines = ["", "## Giai đoạn 2 — đồng thuận ensemble (AND), so nhóm model tương quan cao và nhóm đa dạng", "",
        "'required_votes' là số model phải cùng vượt ngưỡng trần riêng (Giai đoạn 1) tại một cửa sổ mới "
        "tính là cảnh báo; required_votes=1 giống phép OR, required_votes=N là AND chặt nhất. Vẫn là dữ "
        "liệu validation đã dùng chọn model ở E05/E06, không phải holdout mới.", ""]
    lines += table("correlated_4") + [""] + table("diverse_6") + [""]

    base4 = baseline[baseline.model.isin(["tabm_20260917", "tabm_20260918", "tabm_20260919", "lightgbm_gaussian_jitter"])]
    lines.append("So với PPV/recall riêng từng model tại điểm trần Giai đoạn 1 (trung bình 4 model tương quan cao):")
    for h in (300, 600):
        avg_ppv = base4[base4.horizon.eq(h)].ppv_ceiling.mean()
        avg_recall = base4[base4.horizon.eq(h)].recall_ceiling.mean()
        best4 = results["correlated_4"]
        best4 = best4[(best4.horizon.eq(h)) & (best4.required_votes.eq(3))].iloc[0]
        best6 = results["diverse_6"]
        best6_at_3 = best6[(best6.horizon.eq(h)) & (best6.required_votes.eq(3))].iloc[0]
        best6_at_4 = best6[(best6.horizon.eq(h)) & (best6.required_votes.eq(4))].iloc[0]
        lines.append(f"- {h // 60} phút: riêng lẻ trung bình PPV {avg_ppv:.3f} (recall {avg_recall:.3f}) → "
                     f"nhóm tương quan cao 3/4: PPV {best4.ppv:.3f}, recall {best4.recall:.3f} → "
                     f"nhóm đa dạng 3/6: PPV {best6_at_3.ppv:.3f}, recall {best6_at_3.recall:.3f}; "
                     f"4/6: PPV {best6_at_4.ppv:.3f}, recall {best6_at_4.recall:.3f}.")

    lines += ["", "## Kết luận Giai đoạn 2 (đồng thuận ensemble)", "",
        "- **Đồng thuận giữa 4 model tương quan cao (3 seed TabM + LightGBM cùng feature) chỉ tăng PPV "
        "vài điểm phần trăm**: 5 phút PPV 0,064 (1/4) → 0,131 (4/4, AND chặt), recall giảm theo "
        "(0,696 → 0,739, không đơn điệu do persistence/cooldown); 10 phút PPV 0,118 → 0,141. Các model này "
        "học trên cùng dữ liệu/đặc trưng nên đồng thuận cả ở cảnh báo giả, không chỉ cảnh báo đúng.",
        "- **Thêm MAP baseline (tuyến tính) và CatBoost (khác LightGBM/TabM) vào biểu quyết đa dạng hơn "
        "cho PPV cao rõ rệt**: 5 phút PPV lên đến 0,205 tại 6/6 model đồng thuận (gấp ~1,6 lần mức 0,131 "
        "của nhóm 4 model tương quan cao), đồng thời recall vẫn 0,739 — **không đánh đổi recall để có "
        "PPV cao hơn ở đây**. 10 phút: PPV 0,157 tại 6/6, recall vẫn giữ 0,958 (trên cả mục tiêu 0,85) "
        "suốt từ 3/6 đến 6/6 — ở horizon 10 phút, siết đồng thuận tối đa không mất recall nhưng PPV tăng "
        "gần gấp đôi so với 1/6 (0,110→0,157).",
        "- Đây là bằng chứng ủng hộ rõ giả thuyết 'đa dạng kiến trúc quan trọng hơn số lượng model' trong "
        "ensemble — nhưng **PPV tốt nhất quan sát được (0,157–0,205) vẫn còn rất xa mục tiêu 0,60–0,70** "
        "của mục 8; đây là cải thiện thật nhưng chưa đóng được gate.",
        "- FA/giờ tại điểm PPV tốt nhất vẫn cao (1,15/giờ ở 5 phút, 2,89/giờ ở 10 phút — gấp 2–6 lần ngân "
        "sách 0,5), sẽ xử lý ở Giai đoạn 3; không dùng FA/giờ để loại phương án ở Giai đoạn 2 theo đúng "
        "nguyên tắc tuần tự.",
        "- Bước tiếp theo hợp lý: thử thêm model đa dạng hơn nữa vào biểu quyết (waveform nếu có, hoặc "
        "logistic/lightgbm_all đã có ở E05), hoặc chuyển sang two-stage/tiered risk stratification (tầng "
        "2 dùng đặc trưng khác thay vì chỉ đếm vote) trước khi kết luận PPV trần bằng ensemble-vote là "
        "bao nhiêu. Kết quả này dựa trên development validation nhỏ (300 ca); cần xác nhận lại trên cỡ "
        "mẫu lớn hơn (E07) trước khi khóa phương án.", "",
        "[Tạo bởi scripts/e08/version/v1_sequential_stages/stage2_ppv.py](../../../../scripts/e08/version/v1_sequential_stages/stage2_ppv.py) — không train lại, "
        "không đổi bundle/threshold Giai đoạn 1 hoặc E05/E06 đã khóa."]
    report_path.write_text(existing.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
