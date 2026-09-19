# DEPRECATED — E08 v1. Giữ lại để truy vết, KHÔNG dùng số liệu sinh ra từ file này.
# Audit 19/09/2026 tìm 6 lỗi trong v1; dùng scripts/e08/run_comparison.py thay thế.
# Chi tiết từng lỗi và bằng chứng: scripts/e08/README.md
"""E08 Giai đoạn 1: thử TabM (neural, khác hẳn boosting) + Gaussian jitter cho horizon 5 phút.

TabM gốc (E06, chưa augment) có trần recall 5 phút chỉ 0,74-0,78 — thấp hơn CatBoost đã augment
(0,870). Thử augment: nhân bản window dương y_300 kèm nhiễu Gaussian nhỏ (giống
stage1_data_methods.py) trước khi đưa vào TabMRisk.fit, giữ subjectid gốc của mỗi bản sao
để inner_stop_mask (tách bệnh nhân trong FIT) không bị phá vỡ. So trần recall với TabM gốc.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from safeanes.config import Protocol
from safeanes.evaluation import evaluate_predictions
from safeanes.tabular_sota import TabMRisk

ROOT = Path(__file__).resolve().parents[4]
RNG = np.random.default_rng(20260919)


def gaussian_jitter_with_subjects(X, y_300, y_600, subjects, multiplier=3, scale=.1):
    pos_idx = np.where(y_300 == 1)[0]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_imp = imputer.fit_transform(X)
    std = X_imp.std(axis=0)
    copies_x, copies_y300, copies_y600, copies_subj = [], [], [], []
    for _ in range(multiplier):
        noise = RNG.normal(0, scale * std, size=(len(pos_idx), X_imp.shape[1]))
        copies_x.append(X_imp[pos_idx] + noise)
        copies_y300.append(y_300[pos_idx])
        copies_y600.append(y_600[pos_idx])
        copies_subj.append(subjects[pos_idx])
    X_aug = np.concatenate([X_imp] + copies_x)
    y300_aug = np.concatenate([y_300] + copies_y300)
    y600_aug = np.concatenate([y_600] + copies_y600)
    subj_aug = np.concatenate([subjects] + copies_subj)
    return X_aug, y300_aug, y600_aug, subj_aug


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

    X = fit[features].to_numpy(float)
    y300 = fit["y_300"].to_numpy(float)
    y600 = fit["y_600"].to_numpy(float)
    subjects = fit["subjectid"].to_numpy()
    X_aug, y300_aug, y600_aug, subj_aug = gaussian_jitter_with_subjects(X, y300, y600, subjects, multiplier=3, scale=.1)
    y_aug = np.column_stack([y300_aug, y600_aug])

    print(f"fit rows: {len(X)} -> augmented: {len(X_aug)}; positives y300: {int((y300==1).sum())} -> {int((y300_aug==1).sum())}")
    model = TabMRisk(seed=20260917).fit(X_aug, y_aug, subj_aug)
    raw = model.predict_risk(val[features])

    rows = []
    for i, h in enumerate((300, 600)):
        v = val[["caseid", "subjectid", "time", "eligible", "exposure_seconds", "y_300", "y_600",
                 "historical_subject", "role"]].copy()
        p = np.full(len(val), np.nan)
        p[val.eligible.to_numpy()] = raw[val.eligible.to_numpy(), i]
        v["probability"] = p
        ceiling = recall_ceiling(v, events, h, protocol)
        rows.append({"horizon": h, "variant": "tabm_gaussian_jitter_y300",
                     "recall_ceiling": ceiling["event_sensitivity"], "ppv_at_ceiling": ceiling["alarm_ppv"],
                     "fa_per_hour_at_ceiling": ceiling["false_alarms_per_hour"], "auroc": ceiling["auroc"],
                     "threshold_at_ceiling": ceiling["threshold"]})
        print(rows[-1])
    out = pd.DataFrame(rows)
    out_dir = ROOT / "reports/E08/version/v1"
    out.to_csv(out_dir / "stage1_tabm_augment.csv", index=False)
    append_report(out, out_dir)


def append_report(frame, out_dir):
    fmt = lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.3f}"
    report_path = out_dir / "REPORT.md"
    existing = report_path.read_text(encoding="utf-8")
    marker = "\n## Giai đoạn 1, thêm: TabM + Gaussian jitter"
    if marker in existing:
        existing = existing[: existing.index(marker)]
    lines = ["", "## Giai đoạn 1, thêm: TabM + Gaussian jitter", "",
        "TabM gốc (E06, chưa augment) có trần recall 5 phút chỉ 0,739-0,783 (bảng đầu Giai đoạn 1), "
        "thấp hơn CatBoost đã augment (0,870, kỷ lục hiện tại). Thử augment window dương y_300 bằng "
        "Gaussian jitter (giống LightGBM/CatBoost) trước khi train TabM, giữ subjectid gốc cho mỗi bản "
        "sao để không phá inner-patient-split của TabM.", "",
        "| Phút | Biến thể | Recall trần | PPV tại trần | FA/giờ tại trần | AUROC |",
        "|---:|---|---:|---:|---:|---:|"]
    for r in frame.itertuples():
        lines.append(f"| {r.horizon // 60} | {r.variant} | {fmt(r.recall_ceiling)} | "
                      f"{fmt(r.ppv_at_ceiling)} | {fmt(r.fa_per_hour_at_ceiling)} | {fmt(r.auroc)} |")
    record5, record10 = .8695652173913043, .9583333333333334
    lines += ["", "## Nhận xét", ""]
    for h, record in ((300, record5), (600, record10)):
        row = frame[frame.horizon.eq(h)].iloc[0]
        verdict = "VƯỢT kỷ lục CatBoost" if row.recall_ceiling > record + 1e-9 else "không vượt kỷ lục CatBoost"
        lines.append(f"- {h // 60} phút: TabM + jitter đạt {row.recall_ceiling:.3f} — {verdict} "
                     f"({record:.3f}).")
    lines += ["", "[Tạo bởi scripts/e08/version/v1_sequential_stages/stage1_tabm_augment.py](../../../../scripts/e08/version/v1_sequential_stages/stage1_tabm_augment.py) "
        "— train mới trên development300 fit split (đã augment), validation không đổi."]
    report_path.write_text(existing.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
