# DEPRECATED — E08 v1. Giữ lại để truy vết, KHÔNG dùng số liệu sinh ra từ file này.
# Audit 19/09/2026 tìm 6 lỗi trong v1; dùng scripts/e08/run_comparison.py thay thế.
# Chi tiết từng lỗi và bằng chứng: scripts/e08/README.md
"""E08: chốt catboost_balanced_plus_jitter (kỷ lục trần recall 5 phút 0,870) vào artifact dùng chung.

Train lại CatBoost auto_class_weights=Balanced + Gaussian-jitter augmentation (đã thắng ở
stage1_more_models.py), lưu bundle + prediction validation vào artifacts/E08/ để dùng
trong ensemble Giai đoạn 2-4, thay cho catboost_all_20260917 (E05, chưa augment/weight).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer

from safeanes.config import Protocol
from safeanes.evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[4]
HORIZONS = (300, 600)
RNG = np.random.default_rng(20260919)
NAME = "catboost_balanced_plus_jitter"


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

    out_dir = ROOT / "artifacts/E08/version/v1" / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    ceiling_rows = []
    for h in HORIZONS:
        known = fit[f"y_{h}"].ge(0)
        X = fit.loc[known, features].to_numpy(float)
        y = fit.loc[known, f"y_{h}"].astype(int).to_numpy()
        X_aug, y_aug = gaussian_jitter(X, y, multiplier=3, scale=.1)
        model = CatBoostClassifier(iterations=300, depth=5, learning_rate=.05, l2_leaf_reg=5,
            auto_class_weights="Balanced", random_seed=20260917, thread_count=2, verbose=False,
            allow_writing_files=False)
        model.fit(X_aug, y_aug)
        joblib.dump({"model": model, "features": features}, out_dir / f"model_{h}.joblib")

        v = val[["caseid", "subjectid", "time", "eligible", "exposure_seconds", "y_300", "y_600",
                 "historical_subject", "role"]].copy()
        p = np.full(len(val), np.nan)
        p[val.eligible.to_numpy()] = model.predict_proba(val.loc[val.eligible, features])[:, 1]
        v["probability"] = p
        v.to_csv(out_dir / f"validation_{h}.csv.gz", index=False)

        ceiling = recall_ceiling(v, events, h, protocol)
        ceiling_rows.append({"experiment": "E08", "model": NAME, "horizon": h,
            "recall_current": None, "ppv_current": None, "fa_per_hour_current": None, "threshold_current": None,
            "recall_ceiling": ceiling["event_sensitivity"], "ppv_ceiling": ceiling["alarm_ppv"],
            "fa_per_hour_ceiling": ceiling["false_alarms_per_hour"], "threshold_ceiling": ceiling["threshold"],
            "auroc": ceiling["auroc"], "prediction_coverage": ceiling["prediction_coverage"],
            "events_eligible": ceiling["events_eligible"], "events_detected_at_ceiling": ceiling["events_detected"]})
        print(ceiling_rows[-1])

    ceiling_path = ROOT / "reports/E08/version/v1/stage1_recall_ceiling.csv"
    existing = pd.read_csv(ceiling_path)
    existing = existing[~((existing.experiment.eq("E08")) & (existing.model.eq(NAME)))]
    updated = pd.concat([existing, pd.DataFrame(ceiling_rows)], ignore_index=True)
    updated.to_csv(ceiling_path, index=False)


if __name__ == "__main__":
    main()
