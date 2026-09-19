# DEPRECATED — E08 v1. Giữ lại để truy vết, KHÔNG dùng số liệu sinh ra từ file này.
# Audit 19/09/2026 tìm 6 lỗi trong v1; dùng scripts/e08/run_comparison.py thay thế.
# Chi tiết từng lỗi và bằng chứng: scripts/e08/README.md
"""E08 Giai đoạn 1, phương pháp #3: xử lý dữ liệu (oversampling/SMOTE/jitter) để nâng trần recall.

Chỉ sửa TẬP FIT (train) của development300; validation giữ nguyên, không đụng tới. So 4 biến thể:
- baseline: fit gốc, không augment, không trọng số (đối chứng khác baseline weighted ở bước trước).
- oversample_3x: nhân bản nguyên văn window dương (event) thêm 2 lần (tổng 3x).
- smote_5nn: sinh mẫu dương tổng hợp bằng nội suy với 5 láng giềng dương gần nhất (không gian đã
  chuẩn hóa, sau đó impute median để tính khoảng cách — synthetic dùng giá trị đã impute).
- gaussian_jitter: nhân bản window dương kèm nhiễu Gaussian nhỏ (std = 0,1 x độ lệch chuẩn từng
  đặc trưng trên fit) — một dạng augmentation đơn giản khác SMOTE.

Đây là train mới trên development300 fit split; validation dùng để đo trần recall (bỏ ràng buộc
FA/giờ) giống các script Giai đoạn 1 trước, không phải holdout/test.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from safeanes.config import Protocol
from safeanes.evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[4]
HORIZONS = (300, 600)
RNG = np.random.default_rng(20260919)


def recall_ceiling(frame, events, horizon, protocol):
    best = None
    for t in np.r_[np.linspace(.01, .99, 40), 1.000001]:
        m, _, _ = evaluate_predictions(frame, events, horizon, t, protocol)
        if m["event_sensitivity"] is not None and (best is None or m["event_sensitivity"] > best["event_sensitivity"]):
            best = m
    return best


def oversample(X, y, factor):
    pos = np.where(y == 1)[0]
    extra = np.tile(pos, factor - 1)
    return np.concatenate([X, X[extra]]), np.concatenate([y, y[extra]])


def smote(X, y, k=5, multiplier=3):
    pos_idx = np.where(y == 1)[0]
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler().fit(X_imp)
    X_scaled = scaler.transform(X_imp)
    pos_scaled = X_scaled[pos_idx]
    k_eff = min(k, len(pos_idx) - 1)
    if k_eff < 1:
        return X_imp, y
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(pos_scaled)
    _, neighbors = nn.kneighbors(pos_scaled)
    synthetic = []
    for _ in range(multiplier):
        chosen_neighbor = neighbors[np.arange(len(pos_idx)), RNG.integers(1, k_eff + 1, len(pos_idx))]
        alpha = RNG.uniform(0, 1, (len(pos_idx), 1))
        synth_scaled = pos_scaled + alpha * (pos_scaled[chosen_neighbor] - pos_scaled)
        synthetic.append(scaler.inverse_transform(synth_scaled))
    X_synth = np.concatenate(synthetic)
    y_synth = np.ones(len(X_synth))
    return np.concatenate([X_imp, X_synth]), np.concatenate([y, y_synth])


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


def train_and_score(X, y, features, val, val_features, events, horizon, protocol, monotone):
    model = LGBMClassifier(n_estimators=500, num_leaves=7, learning_rate=.03, min_child_samples=150,
                            reg_lambda=10, monotone_constraints=monotone, random_state=20260917,
                            n_jobs=2, verbosity=-1)
    model.fit(X, y)
    v = val[["caseid", "subjectid", "time", "eligible", "exposure_seconds", "y_300", "y_600",
             "historical_subject", "role"]].copy()
    p = np.full(len(val), np.nan)
    p[val.eligible.to_numpy()] = model.predict_proba(val.loc[val.eligible, val_features])[:, 1]
    v["probability"] = p
    return recall_ceiling(v, events, horizon, protocol)


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
        variants = {
            "baseline_no_augment": (X, y),
            "oversample_3x": oversample(X, y, 3),
            "smote_5nn_3x": smote(X, y, k=5, multiplier=3),
            "gaussian_jitter_3x": gaussian_jitter(X, y, multiplier=3, scale=.1),
        }
        for name, (Xv, yv) in variants.items():
            ceiling = train_and_score(Xv, yv, features, val, features, events, h, protocol, monotone)
            rows.append({"horizon": h, "variant": name, "n_train": len(Xv), "n_positive": int(yv.sum()),
                         "recall_ceiling": ceiling["event_sensitivity"], "ppv_at_ceiling": ceiling["alarm_ppv"],
                         "fa_per_hour_at_ceiling": ceiling["false_alarms_per_hour"], "auroc": ceiling["auroc"],
                         "threshold_at_ceiling": ceiling["threshold"]})
            print(rows[-1])
    out = pd.DataFrame(rows)
    out_dir = ROOT / "reports/E08/version/v1"
    out.to_csv(out_dir / "stage1_data_methods.csv", index=False)
    append_report(out, out_dir)


def append_report(frame, out_dir):
    fmt = lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.3f}"
    report_path = out_dir / "REPORT.md"
    existing = report_path.read_text(encoding="utf-8")
    marker = "\n## Giai đoạn 1, thêm: xử lý dữ liệu"
    if marker in existing:
        existing = existing[: existing.index(marker)]
    lines = ["", "## Giai đoạn 1, thêm: xử lý dữ liệu (oversampling/SMOTE/jitter, phương pháp #3)", "",
        "Chỉ sửa fit split (train) của development300, validation giữ nguyên. So trần recall (bỏ ràng "
        "buộc FA/giờ) giữa 4 biến thể dữ liệu, không đổi trọng số lớp (khác thí nghiệm trước).", "",
        "| Phút | Biến thể | Số mẫu train | Số dương | Recall trần | PPV tại trần | FA/giờ tại trần |",
        "|---:|---|---:|---:|---:|---:|---:|"]
    for r in frame.itertuples():
        lines.append(f"| {r.horizon // 60} | {r.variant} | {r.n_train} | {r.n_positive} | "
                      f"{fmt(r.recall_ceiling)} | {fmt(r.ppv_at_ceiling)} | {fmt(r.fa_per_hour_at_ceiling)} |")
    lines += ["", "## Nhận xét", ""]
    for h in (300, 600):
        sub = frame[frame.horizon.eq(h)]
        base = sub[sub.variant.eq("baseline_no_augment")].iloc[0]
        best = sub.sort_values("recall_ceiling", ascending=False).iloc[0]
        if best.variant == "baseline_no_augment":
            lines.append(f"- {h // 60} phút: **không biến thể dữ liệu nào vượt baseline** "
                         f"({base.recall_ceiling:.3f}) — xử lý dữ liệu (ở quy mô/cách làm này) không phải "
                         "nút thắt chính ở horizon này.")
        else:
            lines.append(f"- {h // 60} phút: **{best.variant} nâng trần recall {base.recall_ceiling:.3f} → "
                         f"{best.recall_ceiling:.3f}** (PPV {best.ppv_at_ceiling:.3f} so với baseline "
                         f"{base.ppv_at_ceiling:.3f}).")
    lines += ["", "So với class-weighted LightGBM (thí nghiệm trước, scale_pos_weight_20x đạt trần "
        "5 phút 0,826): xem biến thể tốt nhất ở đây có vượt được mức đó không, hay hai kỹ thuật đang "
        "chạm cùng một giới hạn của model/feature hiện có.", "",
        "[Tạo bởi scripts/e08/version/v1_sequential_stages/stage1_data_methods.py](../../../../scripts/e08/version/v1_sequential_stages/stage1_data_methods.py) "
        "— train mới trên development300 fit split (đã augment), validation không đổi, không đổi "
        "bundle E05/E06 đã khóa."]
    report_path.write_text(existing.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
