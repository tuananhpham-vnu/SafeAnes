# DEPRECATED — E08 v1. Giữ lại để truy vết, KHÔNG dùng số liệu sinh ra từ file này.
# Audit 19/09/2026 tìm 6 lỗi trong v1; dùng scripts/e08/run_comparison.py thay thế.
# Chi tiết từng lỗi và bằng chứng: scripts/e08/README.md
"""E08 Giai đoạn 1: thử CatBoost (khác LightGBM) và kết hợp augmentation + trọng số lớp.

Trả lời hai câu hỏi: (1) ngoài LightGBM còn model nào khác đáng thử không — CatBoost đã có sẵn
trong .local_deps và dùng được auto_class_weights='Balanced' (khác monotone-constrained LightGBM);
(2) kết hợp gaussian_jitter (thắng ở thí nghiệm dữ liệu) với scale_pos_weight (thắng ở thí nghiệm
trọng số lớp) có đẩy trần recall xa hơn từng cái riêng lẻ không (0,826 là kỷ lục hiện tại ở 5 phút).
Train mới trên development300 fit split; validation không đổi.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer

from safeanes.config import Protocol
from safeanes.evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[4]
HORIZONS = (300, 600)
RNG = np.random.default_rng(20260919)


def gaussian_jitter(X, y, multiplier=3, scale=.1):
    pos_idx = np.where(y == 1)[0]
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    std = X_imp.std(axis=0)
    copies = []
    for _ in range(multiplier):
        noise = RNG.normal(0, scale * std, size=(len(pos_idx), X_imp.shape[1]))
        copies.append(X_imp[pos_idx] + noise)
    X_aug = np.concatenate(copies)
    y_aug = np.ones(len(X_aug))
    return np.concatenate([X_imp, X_aug]), np.concatenate([y, y_aug])


def recall_ceiling(frame, events, horizon, protocol):
    best = None
    for t in np.r_[np.linspace(.01, .99, 40), 1.000001]:
        m, _, _ = evaluate_predictions(frame, events, horizon, t, protocol)
        if m["event_sensitivity"] is not None and (best is None or m["event_sensitivity"] > best["event_sensitivity"]):
            best = m
    return best


def score(model, val, features, is_pipeline_input_imputed, imputer=None):
    X = val.loc[val.eligible, features]
    if imputer is not None:
        X = imputer.transform(X)
    p = np.full(len(val), np.nan)
    p[val.eligible.to_numpy()] = model.predict_proba(X)[:, 1]
    return p


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
    monotone = [-1 if f in ("map_current", "map_60_mean", "map_300_mean") else 0 for f in features]

    rows = []
    for h in HORIZONS:
        known = fit[f"y_{h}"].ge(0)
        X = fit.loc[known, features].to_numpy(float)
        y = fit.loc[known, f"y_{h}"].astype(int).to_numpy()
        ratio = float((y == 0).sum() / max((y == 1).sum(), 1))

        # Variant 1: LightGBM, jitter augmentation + scale_pos_weight combined.
        X_aug, y_aug = gaussian_jitter(X, y, multiplier=3, scale=.1)
        lgbm_combo = LGBMClassifier(n_estimators=500, num_leaves=7, learning_rate=.03, min_child_samples=150,
                                     reg_lambda=10, monotone_constraints=monotone, random_state=20260917,
                                     scale_pos_weight=ratio * 10, n_jobs=2, verbosity=-1)
        lgbm_combo.fit(X_aug, y_aug)
        p1 = score(lgbm_combo, val, features, False)
        rows.append(("lightgbm_jitter_plus_weight", h, p1))

        # Variant 2: CatBoost, auto_class_weights=Balanced, no augmentation.
        cat_balanced = CatBoostClassifier(iterations=300, depth=5, learning_rate=.05, l2_leaf_reg=5,
            auto_class_weights="Balanced", random_seed=20260917, thread_count=2, verbose=False,
            allow_writing_files=False)
        cat_balanced.fit(fit.loc[known, features], y)
        p2 = score(cat_balanced, val, features, False)
        rows.append(("catboost_balanced", h, p2))

        # Variant 3: CatBoost, auto_class_weights=Balanced + jitter augmentation.
        cat_combo = CatBoostClassifier(iterations=300, depth=5, learning_rate=.05, l2_leaf_reg=5,
            auto_class_weights="Balanced", random_seed=20260917, thread_count=2, verbose=False,
            allow_writing_files=False)
        cat_combo.fit(X_aug, y_aug)
        p3 = score(cat_combo, val, features, False)
        rows.append(("catboost_balanced_plus_jitter", h, p3))

    out_dir = ROOT / "reports/E08/version/v1"
    results = []
    for name, h, p in rows:
        v = val[["caseid", "subjectid", "time", "eligible", "exposure_seconds", "y_300", "y_600",
                 "historical_subject", "role"]].copy()
        v["probability"] = p
        ceiling = recall_ceiling(v, events, h, protocol)
        results.append({"horizon": h, "variant": name, "recall_ceiling": ceiling["event_sensitivity"],
                         "ppv_at_ceiling": ceiling["alarm_ppv"], "fa_per_hour_at_ceiling": ceiling["false_alarms_per_hour"],
                         "auroc": ceiling["auroc"], "threshold_at_ceiling": ceiling["threshold"]})
        print(results[-1])
    out = pd.DataFrame(results)
    out.to_csv(out_dir / "stage1_more_models.csv", index=False)
    append_report(out, out_dir)


def append_report(frame, out_dir):
    fmt = lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.3f}"
    report_path = out_dir / "REPORT.md"
    existing = report_path.read_text(encoding="utf-8")
    marker = "\n## Giai đoạn 1, thêm: CatBoost và kết hợp augmentation"
    if marker in existing:
        existing = existing[: existing.index(marker)]
    lines = ["", "## Giai đoạn 1, thêm: CatBoost và kết hợp augmentation", "",
        "So sánh với kỷ lục trần recall trước đó (LightGBM scale_pos_weight_20x: 5 phút 0,826, "
        "10 phút 0,958). 'lightgbm_jitter_plus_weight' kết hợp Gaussian jitter + scale_pos_weight cùng "
        "lúc; 'catboost_balanced' và 'catboost_balanced_plus_jitter' thử kiến trúc khác (gradient "
        "boosting oblivious trees, khác LightGBM leaf-wise).", "",
        "| Phút | Biến thể | Recall trần | PPV tại trần | FA/giờ tại trần |",
        "|---:|---|---:|---:|---:|"]
    for r in frame.itertuples():
        lines.append(f"| {r.horizon // 60} | {r.variant} | {fmt(r.recall_ceiling)} | "
                      f"{fmt(r.ppv_at_ceiling)} | {fmt(r.fa_per_hour_at_ceiling)} |")
    lines += ["", "## Nhận xét", "",
        "So với kỷ lục cũ (LightGBM scale_pos_weight_20x): 5 phút 0,826; 10 phút 0,958.", ""]
    for h in (300, 600):
        sub = frame[frame.horizon.eq(h)]
        best = sub.sort_values("recall_ceiling", ascending=False).iloc[0]
        record = 0.826 if h == 300 else 0.958
        verdict = "VƯỢT kỷ lục" if best.recall_ceiling > record + 1e-9 else "không vượt kỷ lục cũ"
        lines.append(f"- {h // 60} phút: tốt nhất trong nhóm này là {best.variant} "
                     f"(recall {best.recall_ceiling:.3f}, PPV {best.ppv_at_ceiling:.3f}) — {verdict}.")
    lines += ["", "[Tạo bởi scripts/e08/version/v1_sequential_stages/stage1_more_models.py](../../../../scripts/e08/version/v1_sequential_stages/stage1_more_models.py) "
        "— train mới trên development300 fit split, không đổi bundle E05/E06 đã khóa."]
    report_path.write_text(existing.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
