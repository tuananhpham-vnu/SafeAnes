# DEPRECATED — E08 v1. Giữ lại để truy vết, KHÔNG dùng số liệu sinh ra từ file này.
# Audit 19/09/2026 tìm 6 lỗi trong v1; dùng scripts/e08/run_comparison.py thay thế.
# Chi tiết từng lỗi và bằng chứng: scripts/e08/README.md
"""E08 Giai đoạn 3, phương pháp #1: risk-controlling threshold selection để giảm FA/giờ.

Sửa so với bản đầu: ensemble ở đây dùng VOTE (số model vượt ngưỡng trần riêng của nó ở Giai đoạn 1,
giống Giai đoạn 2) thay vì trung bình cộng calibrated probability. Lý do đổi: bản trung bình cộng
nhạy với việc đổi model thành phần (khi thay lightgbm_monotone/catboost_all bằng bản đã
augment/weight, kết quả Giai đoạn 3 bị lệch không nhất quán dù Giai đoạn 1-2 đều cải thiện) —
vote ổn định hơn vì chỉ quan tâm model có vượt ngưỡng của chính nó hay không, không phụ thuộc hình
dạng phân phối xác suất của từng model. Với mỗi mức 'required_votes' (k trong N model), tính điểm
trên validation rồi bootstrap theo bệnh nhân (evaluation.bootstrap_ci) để lấy cận trên 95% của
FA/giờ — chỉ nhận k có CẢ điểm ước lượng lẫn cận trên đều <= ngân sách 0,5, đây là phần
'risk-controlling'. Vẫn là dữ liệu validation, không phải holdout mới.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from safeanes.config import Protocol
from safeanes.evaluation import bootstrap_ci, evaluate_predictions

ROOT = Path(__file__).resolve().parents[4]
MODELS = ["tabm_20260917", "tabm_20260918", "tabm_20260919", "lightgbm_gaussian_jitter",
          "map_all_20260917", "catboost_balanced_plus_jitter"]
E05_MODELS = {"map_all_20260917"}
E08_MODELS = {"lightgbm_gaussian_jitter", "catboost_balanced_plus_jitter"}
HORIZONS = (300, 600)
META = ["caseid", "subjectid", "time", "eligible", "exposure_seconds", "y_300", "y_600",
        "historical_subject", "role"]
FA_BUDGET = 0.5


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

    summary_rows, curve_rows = [], []
    for h in HORIZONS:
        frame = ensemble_vote_score(h, thresholds)
        point = []
        for k in range(1, len(MODELS) + 1):
            t = k / len(MODELS)
            m, _, _ = evaluate_predictions(frame, events, h, t, protocol)
            m["threshold"] = t  # keep as the vote fraction, not the raw grid value, for readability
            point.append(m)
            curve_rows.append({"horizon": h, "required_votes": k, "threshold": t, **{k2: m[k2] for k2 in
                ("event_sensitivity", "alarm_ppv", "false_alarms_per_hour", "auroc")}})
        under_budget = [m for m in point if m["false_alarms_per_hour"] is not None
                        and m["false_alarms_per_hour"] <= FA_BUDGET]
        if not under_budget:
            chosen_point = min(point, key=lambda m: m["false_alarms_per_hour"] or np.inf)
            status = "no_point_estimate_under_budget"
        else:
            chosen_point = max(under_budget, key=lambda m: (m["event_sensitivity"] or 0, m["alarm_ppv"] or 0))
            status = "point_estimate_under_budget"
        # Risk-controlling check: require the bootstrap upper 95% CI of FA/hour to also respect budget.
        candidates = sorted(point, key=lambda m: m["threshold"])
        checked = None
        for m in candidates:
            if m["false_alarms_per_hour"] is None or m["false_alarms_per_hour"] > FA_BUDGET * 1.5:
                continue
            ci = bootstrap_ci(frame, events, h, m["threshold"], protocol, repeats=200)
            fa_high = ci["intervals"]["false_alarms_per_hour"]["high"]
            if fa_high is not None and fa_high <= FA_BUDGET:
                checked = (m, ci)
                break
        if checked is None:
            final_metric, final_ci, rc_status = chosen_point, None, "risk_controlling_threshold_not_found"
        else:
            final_metric, final_ci, rc_status = checked[0], checked[1], "risk_controlling_threshold_found"
        required_votes = round(final_metric["threshold"] * len(MODELS))
        summary_rows.append({"horizon": h, "required_votes": required_votes, "total_models": len(MODELS),
            "recall": final_metric["event_sensitivity"], "ppv": final_metric["alarm_ppv"],
            "fa_per_hour": final_metric["false_alarms_per_hour"],
            "fa_per_hour_ci_high": None if final_ci is None else final_ci["intervals"]["false_alarms_per_hour"]["high"],
            "recall_ci_low": None if final_ci is None else final_ci["intervals"]["event_sensitivity"]["low"],
            "point_estimate_status": status, "risk_controlling_status": rc_status})
        print(f"horizon {h}: point_status={status} rc_status={rc_status} -> {summary_rows[-1]}")

    pd.DataFrame(curve_rows).to_csv(out_dir / "stage3_fa_curve.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "stage3_fa_summary.csv", index=False)
    append_report(summary, out_dir)


def append_report(summary, out_dir):
    fmt = lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.3f}"
    report_path = out_dir / "REPORT.md"
    existing = report_path.read_text(encoding="utf-8")
    marker = "\n## Giai đoạn 3"
    if marker in existing:
        existing = existing[: existing.index(marker)]
    lines = ["", "## Giai đoạn 3 — risk-controlling threshold trên ensemble VOTE (6 model đa dạng)", "",
        "Ensemble = vote (số model vượt ngưỡng trần riêng của Giai đoạn 1), giống Giai đoạn 2 — đã đổi "
        "khỏi trung bình cộng liên tục vì bản đó nhạy với việc đổi model thành phần. Với mỗi mức "
        "required_votes (k/6), kiểm tra cả điểm ước lượng và cận trên 95% bootstrap theo bệnh nhân (200 "
        "lần, unit=subjectid) của FA/giờ; chỉ nhận k có CẢ HAI đều <= ngân sách 0,5 — đây là phần "
        "'risk-controlling'. Dữ liệu vẫn là validation đã dùng chọn model ở E05/E06/E08, không phải "
        "holdout mới.", "",
        "| Phút | Đồng thuận | Recall | PPV | FA/giờ (điểm) | FA/giờ cận trên 95% | Trạng thái risk-controlling |",
        "|---:|---:|---:|---:|---:|---:|---|"]
    for r in summary.itertuples():
        lines.append(f"| {r.horizon // 60} | {r.required_votes}/{r.total_models} | {fmt(r.recall)} | {fmt(r.ppv)} | "
                      f"{fmt(r.fa_per_hour)} | {fmt(r.fa_per_hour_ci_high)} | {r.risk_controlling_status} |")
    lines += ["", "## Kết luận Giai đoạn 3", ""]
    for r in summary.itertuples():
        if r.risk_controlling_status == "risk_controlling_threshold_found":
            lines.append(f"- {r.horizon // 60} phút: đồng thuận {r.required_votes}/{r.total_models} model, "
                         f"FA/giờ cận trên 95% {r.fa_per_hour_ci_high:.3f} <= 0,5 (đạt ngân sách có kiểm "
                         f"chứng CI), tại đó recall {r.recall:.3f}, PPV {r.ppv:.3f}.")
        else:
            lines.append(f"- {r.horizon // 60} phút: **không tìm được mức đồng thuận nào có cận trên 95% CI "
                         f"của FA/giờ <= 0,5** trên ensemble 6 model này; điểm gần nhất theo ước lượng điểm "
                         f"cho recall {r.recall:.3f}, PPV {r.ppv:.3f}, FA/giờ {r.fa_per_hour:.3f}. Cần "
                         f"model có discrimination tốt hơn hoặc thêm dữ liệu (E07), không chỉ chỉnh "
                         f"ngưỡng/ensemble hiện có, để đạt ngân sách 0,5 ở horizon này.")
    lines += ["", "So sánh xuyên suốt 3 giai đoạn (điểm vận hành khác nhau, không phải cùng một model "
        "chọn một lần): Giai đoạn 1 tối ưu recall một mình (bỏ ngân sách FA) → Giai đoạn 2 tối ưu PPV "
        "bằng đồng thuận đa dạng (bỏ ngân sách FA) → Giai đoạn 3 áp lại ngân sách FA có kiểm chứng CI, "
        "nên recall/PPV ở đây thường thấp hơn hai giai đoạn trước — đây là đánh đổi đã biết trước theo "
        "đúng nguyên tắc tối ưu tuần tự của E08_PLAN.md, chưa phải bước gộp (Giai đoạn 4).", "",
        "[Tạo bởi scripts/e08/version/v1_sequential_stages/stage3_fa.py](../../../../scripts/e08/version/v1_sequential_stages/stage3_fa.py) — không train lại, "
        "không đổi threshold/bundle đã khóa của E05/E06 hoặc kết quả Giai đoạn 1–2."]
    report_path.write_text(existing.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
