# DEPRECATED — E08 v1. Giữ lại để truy vết, KHÔNG dùng số liệu sinh ra từ file này.
# Audit 19/09/2026 tìm 6 lỗi trong v1; dùng scripts/e08/run_comparison.py thay thế.
# Chi tiết từng lỗi và bằng chứng: scripts/e08/README.md
"""E08 Giai đoạn 1, phương pháp #2: class-weighted / focal-loss-style LightGBM để nâng trần recall.

Train lại LightGBM (không dùng bundle E06 cũ) với 3 biến thể trọng số lớp trên đúng fit split của
development300 (giống E06_PLAN.md: monotone constraints theo MAP), so trần recall (bỏ ràng buộc
FA/giờ, giống scripts/e08/version/v1_sequential_stages/stage1_recall.py) với lightgbm_monotone gốc (không trọng số) đã có ở
E06. Đây là train mới trên development, không phải holdout/test; chỉ để trả lời "trọng số lớp có
nâng trần recall 5 phút lên gần 0,90 không", câu hỏi còn mở ở Giai đoạn 1.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from safeanes.config import Protocol
from safeanes.evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[4]
HORIZONS = (300, 600)
VARIANTS = {
    "unweighted_baseline": {},
    "class_weight_balanced": {"class_weight": "balanced"},
    "scale_pos_weight_5x": {},  # scale_pos_weight set per-horizon below (5x class ratio)
    "scale_pos_weight_20x": {},  # 20x class ratio, aggressive
}


def recall_ceiling_curve(frame, events, horizon, protocol):
    best = None
    for t in np.r_[np.linspace(.01, .99, 40), 1.000001]:
        m, _, _ = evaluate_predictions(frame, events, horizon, t, protocol)
        if m["event_sensitivity"] is not None and (best is None or m["event_sensitivity"] > best["event_sensitivity"]):
            best = m
    return best


def main():
    data = ROOT / "data/development300"
    meta = json.loads((data / "dataset.json").read_text())
    protocol = Protocol(**{**meta["protocol"], "horizons_seconds": tuple(meta["protocol"]["horizons_seconds"])})
    roles = pd.read_csv(ROOT / "artifacts/E05/roles.csv")
    frame = pd.read_csv(data / "windows.csv.gz").merge(roles, on=["caseid", "subjectid"], validate="many_to_one")
    events = pd.read_csv(data / "events.csv")
    features = [f for f in meta["features"] if not f.startswith("static_")]
    fit = frame[frame.role.eq("fit") & frame.eligible & frame[["y_300", "y_600"]].ge(0).any(axis=1)]
    val = frame[frame.role.eq("validation")]
    constraints = [-1 if f in ("map_current", "map_60_mean", "map_300_mean") else 0 for f in features]

    rows = []
    for h in HORIZONS:
        known = fit[f"y_{h}"].ge(0)
        y = fit.loc[known, f"y_{h}"].astype(int)
        ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
        for name in VARIANTS:
            kwargs = dict(n_estimators=500, num_leaves=7, learning_rate=.03, min_child_samples=150,
                          reg_lambda=10, monotone_constraints=constraints, random_state=20260917,
                          n_jobs=2, verbosity=-1)
            if name == "class_weight_balanced":
                kwargs["class_weight"] = "balanced"
            elif name == "scale_pos_weight_5x":
                kwargs["scale_pos_weight"] = ratio * 5
            elif name == "scale_pos_weight_20x":
                kwargs["scale_pos_weight"] = ratio * 20
            model = LGBMClassifier(**kwargs)
            model.fit(fit.loc[known, features], y)
            v = val[["caseid", "subjectid", "time", "eligible", "exposure_seconds", "y_300", "y_600",
                     "historical_subject", "role"]].copy()
            p = np.full(len(val), np.nan)
            p[val.eligible.to_numpy()] = model.predict_proba(val.loc[val.eligible, features])[:, 1]
            v["probability"] = p
            ceiling = recall_ceiling_curve(v, events, h, protocol)
            rows.append({"horizon": h, "variant": name, "class_ratio_neg_over_pos": ratio,
                         "recall_ceiling": ceiling["event_sensitivity"], "ppv_at_ceiling": ceiling["alarm_ppv"],
                         "fa_per_hour_at_ceiling": ceiling["false_alarms_per_hour"],
                         "threshold_at_ceiling": ceiling["threshold"], "auroc": ceiling["auroc"]})
            print(rows[-1])
    out = pd.DataFrame(rows)
    out_dir = ROOT / "reports/E08/version/v1"
    out.to_csv(out_dir / "stage1_class_weight_variants.csv", index=False)
    append_report(out, out_dir)


def append_report(frame, out_dir):
    fmt = lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.3f}"
    report_path = out_dir / "REPORT.md"
    existing = report_path.read_text(encoding="utf-8")
    marker = "\n## Giai đoạn 1, thêm: class-weighted LightGBM"
    if marker in existing:
        existing = existing[: existing.index(marker)]
    lines = ["", "## Giai đoạn 1, thêm: class-weighted LightGBM (phương pháp #2)", "",
        "Train lại LightGBM (monotone constraints giống E06) với trọng số lớp khác nhau trên fit split "
        "development300; so trần recall (bỏ ràng buộc FA/giờ, giống chẩn đoán đầu Giai đoạn 1) giữa các "
        "biến thể. 'unweighted_baseline' tương đương lightgbm_monotone gốc của E06 nhưng train lại từ "
        "đầu (không dùng bundle cũ) nên số có thể lệch nhẹ do không cùng random_state/hash chính xác.",
        "", "| Phút | Biến thể | Tỷ lệ lớp (âm/dương) | Recall trần | PPV tại trần | FA/giờ tại trần |",
        "|---:|---|---:|---:|---:|---:|"]
    for r in frame.itertuples():
        lines.append(f"| {r.horizon // 60} | {r.variant} | {r.class_ratio_neg_over_pos:.1f} | "
                      f"{fmt(r.recall_ceiling)} | {fmt(r.ppv_at_ceiling)} | {fmt(r.fa_per_hour_at_ceiling)} |")
    lines += ["", "## Nhận xét", ""]
    for h in (300, 600):
        sub = frame[frame.horizon.eq(h)]
        base = sub[sub.variant.eq("unweighted_baseline")].iloc[0]
        best = sub.sort_values("recall_ceiling", ascending=False).iloc[0]
        if best.variant == "unweighted_baseline":
            lines.append(f"- {h // 60} phút: **trọng số lớp không nâng được trần recall** — biến thể tốt "
                         f"nhất vẫn là unweighted ({base.recall_ceiling:.3f}). Trọng số lớp không phải "
                         "nút thắt ở horizon này.")
        else:
            lines.append(f"- {h // 60} phút: **{best.variant} nâng trần recall {base.recall_ceiling:.3f} → "
                         f"{best.recall_ceiling:.3f}** (PPV {best.ppv_at_ceiling:.3f}, FA/giờ "
                         f"{best.fa_per_hour_at_ceiling:.3f} tại điểm đó, so với baseline PPV "
                         f"{base.ppv_at_ceiling:.3f}, FA/giờ {base.fa_per_hour_at_ceiling:.3f}).")
    lines += ["", "[Tạo bởi scripts/e08/version/v1_sequential_stages/stage1_class_weight.py](../../../../scripts/e08/version/v1_sequential_stages/stage1_class_weight.py) "
        "— train mới trên development300 fit split, không đổi bundle E05/E06 đã khóa."]
    report_path.write_text(existing.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
