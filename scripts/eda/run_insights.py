"""Insight report on VitalDB for UC04 -> reports/EDA/INSIGHTS.md (+ figures/insight_*.png, insight_*.csv).

  python scripts/eda/run_insights.py [--workers 8] [--limit N] [--reuse]

Four parts, matching the questions the report must answer:
  1. what the data is (feature dictionary, missing)      2. how noisy it is
  3. how features relate (to each other, to future MAP)  4. which features matter most (vs literature)
Heavy intermediates go to data/eda_cache (not in git). The narrative is reports/EDA/insights_notes.md,
included verbatim; every number in INSIGHTS.md is computed here.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eda import md  # noqa: E402  (same markdown table helper)
from safeanes import eda, eda_plots as plots, insights as ins  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT, FIG, CACHE = ROOT / "reports/EDA", ROOT / "reports/EDA/figures", ROOT / "data/eda_cache"
RISK_FEATURES = {
    "MAP hiện tại (mmHg)": "map_current", "Xu hướng MAP 5 phút (mmHg/s)": "map_300_slope",
    "Độ dao động MAP 5 phút (SD)": "map_300_std", "HR hiện tại": "hr_current",
    "Δlog SVR proxy 5 phút": "bt_svr_lz_d300", "Δlog SV proxy 5 phút": "bt_sv_lz_d300",
    "Δlog dP/dt 5 phút": "bt_dpdt_d300", "PPV hiện tại (%)": "bt_ppv_cur", "BIS hiện tại": "bis_cur",
    "Δ Ce remifentanil 5 phút": "rftn_ce_d300", "Ce propofol hiện tại": "ppf_ce_cur", "Tuổi": "static_age",
}
LITERATURE = [
    ("Mức MAP hiện tại + xu hướng", "Jacquet-Lagrèze et al., *Eur J Anaesthesiol* 2022;39:574–581 — ngoại suy tuyến tính MAP (LepMAP) dự báo IOH tốt hơn ΔMAP đơn thuần ([PubMed](https://pubmed.ncbi.nlm.nih.gov/35695749/)); Massari et al., *Anesthesiology* 2024 so sánh HPI với MAP và LepMAP ([PubMed](https://pubmed.ncbi.nlm.nih.gov/39377485/))"),
    ("MAP là yếu tố chi phối của mô hình waveform", "Mulder et al., *Anesthesiology* 2024 (HPI ≈ ngưỡng MAP); Frassanito et al., *Eur J Anaesthesiol* 2024 (HPI tương quan mạnh với MAP, [PubMed](https://pubmed.ncbi.nlm.nih.gov/38264965/))"),
    ("Đặc trưng waveform ART: tiền tải (PPV/SVV), co bóp (dP/dt), hậu tải (SVR, Eadyn), độ phức tạp sóng", "Hatib et al., *Anesthesiology* 2018 — HPI ([PubMed](https://pubmed.ncbi.nlm.nih.gov/29894315/)); mô tả nhóm đặc trưng theo tài liệu tổng quan HPI"),
    ("SV, SVR, HR, SVV (cơ chế IOH)", "Kouz et al., *BJA* 2023; Jian et al., *BJA* 2025; Zhu et al., *Perioper Med* 2025 (VitalDB EV1000) — xem proposal §3"),
    ("Kết hợp nhiều waveform (ABP + EEG + ECG)", "*PLOS One* 2022, “Predicting intraoperative hypotension using deep learning with waveforms of arterial blood pressure, electroencephalogram, and electrocardiogram” — ABP + EEG cải thiện hiệu năng và calibration so với ABP đơn thuần ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9362925/); mới đối chiếu qua tóm tắt tìm kiếm)"),
    ("Biến trước mổ / thuốc khởi mê (IOH sau khởi mê)", "Kendale et al., *Anesthesiology* 2018;129:675–688 — gradient boosting AUROC 0,76 cho MAP < 55 trong 10 phút sau khởi mê ([PubMed](https://pubmed.ncbi.nlm.nih.gov/30074930/))"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    for d in (OUT, FIG, CACHE):
        d.mkdir(parents=True, exist_ok=True)
    manifest = eda.load_manifest(ROOT)

    if args.reuse and (CACHE / "insight_table.parquet").exists():
        table = pd.read_parquet(CACHE / "insight_table.parquet")
        noise_case = pd.read_parquet(CACHE / "noise_case.parquet")
        noise = pd.read_csv(OUT / "insight_noise.csv")
    else:
        print("feature table...", flush=True)
        table = ins.build_table(ROOT, manifest, workers=args.workers, limit=args.limit)
        table.to_parquet(CACHE / "insight_table.parquet")
        print("noise...", flush=True)
        noise_case, noise = ins.noise_profile(ROOT, manifest, workers=args.workers, limit=args.limit)
        noise_case.to_parquet(CACHE / "noise_case.parquet")
        noise.to_csv(OUT / "insight_noise.csv", index=False)

    # 1. description
    dictionary = ins.dictionary(table)
    dictionary.to_csv(OUT / "insight_dictionary.csv", index=False)
    by_group = dictionary.groupby("nhóm").agg(**{"số feature": ("feature", "size"),
                                               "% thiếu median": ("% thiếu", "median"),
                                               "% thiếu max": ("% thiếu", "max")}).sort_values("số feature", ascending=False)
    key_dict = dictionary.set_index("feature").loc[[f for f in ins.KEY_FEATURES if f in dictionary.feature.values]]
    missing_plot = dictionary[dictionary.feature.isin(ins.KEY_FEATURES)].sort_values("% thiếu", ascending=False)
    plots.hbar_by_group(missing_plot, "% thiếu", "% decision row thiếu giá trị", FIG / "insight_missing.png",
                        "Mức thiếu của các feature chính tại decision row", top=len(missing_plot))
    labelled = table[table.y_300.ge(0)]
    scope = {"decision row (lấy mẫu 1/2, mỗi 60 s)": len(table), "ca": table.caseid.nunique(),
             "bệnh nhân": table.subjectid.nunique(), "feature": len(ins.feature_columns(table)),
             "row có nhãn y5": len(labelled), "tỉ lệ y5 dương (%)": round(labelled.y_300.mean() * 100, 2),
             "tỉ lệ y10 dương (%)": round(table[table.y_600.ge(0)].y_600.mean() * 100, 2)}

    # 2. noise
    plots.noise_bars(noise, FIG / "insight_noise.png")
    events = pd.read_csv(OUT / "events.csv") if (OUT / "events.csv").exists() else pd.DataFrame()
    beats = json.loads((ROOT / "data/beats_full/index.json").read_text(encoding="utf-8"))
    vf = pd.Series({k: v["valid_frac"] for k, v in beats["cases"].items()}, dtype=float)
    label_noise = {
        "biến cố IOH (phạm vi phát triển)": len(events),
        "biến cố có MAP thấp nhất ≤ 30 mmHg (nghi artefact)": f"{int((events.min_map <= 30).sum())} ({(events.min_map <= 30).mean() * 100:.1f}%)" if len(events) else "–",
        "biến cố ≤ 20 mmHg": f"{int((events.min_map <= 20).sum())} ({(events.min_map <= 20).mean() * 100:.1f}%)" if len(events) else "–",
        "ca có waveform ART": len(vf), "beat hợp lệ, median theo ca (%)": round(vf.median() * 100, 1),
        "ca < 50% beat hợp lệ (kênh ART hỏng/không nối)": int((vf < .5).sum()),
        "decision row thiếu beat features (%)": round(table.bt_sv_lz_cur.isna().mean() * 100, 1),
    }

    # 3. relations
    corr = ins.spearman(table)
    corr.to_csv(OUT / "insight_spearman.csv")
    plots.corr_heatmap(corr, FIG / "insight_corr.png")
    pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().rename("rho").reset_index()
    pairs["nhóm 1"], pairs["nhóm 2"] = pairs.level_0.map(ins.group_of), pairs.level_1.map(ins.group_of)
    cross = pairs[pairs["nhóm 1"] != pairs["nhóm 2"]].sort_values("rho", key=np.abs, ascending=False).head(15)
    future = ins.future_association(table)
    future.to_csv(OUT / "insight_future_map.csv", index=False)
    plots.hbar_by_group(future, "rho", "Spearman ρ với ΔMAP 5 phút tới (âm = giá trị cao → MAP sẽ giảm)",
                        FIG / "insight_future.png", "Feature liên quan nhất tới thay đổi MAP trong 5 phút tới",
                        fmt="{:+.2f}", signed=True)
    future_non_bp = future[~future["nhóm"].isin(["MAP", "SBP/DBP"])].head(12)
    curves = {name: ins.risk_curve(table, f) for name, f in RISK_FEATURES.items() if f in table}
    plots.risk_curves(curves, FIG / "insight_risk_curves.png", labelled.y_300.mean())
    lt = ins.level_trend_map(table)
    plots.level_trend(lt, FIG / "insight_level_trend.png")

    # 4. influence
    print("univariate / SHAP / ablation...", flush=True)
    uni = ins.univariate_auroc(table)
    uni.to_csv(OUT / "insight_univariate_auroc.csv", index=False)
    plots.hbar_by_group(uni.rename(columns={"auroc_y_300": "AUROC"}), "AUROC", "AUROC đơn biến (y5)",
                        FIG / "insight_univariate.png", "25 feature đơn lẻ phân biệt IOH 5 phút tốt nhất",
                        fmt="{:.3f}", xlim=(.5, 1))
    uni_best = uni.groupby("nhóm").head(1).sort_values("auroc_y_300", ascending=False)
    imp5, perf5 = ins.shap_importance(table, "y_300")
    imp10, perf10 = ins.shap_importance(table, "y_600")
    imp5.to_csv(OUT / "insight_shap_y5.csv", index=False)
    imp10.to_csv(OUT / "insight_shap_y10.csv", index=False)
    plots.shap_panels(imp5, FIG / "insight_shap.png")
    group_share = pd.DataFrame({"y5 (%)": imp5.groupby("nhóm")["share_%"].sum(),
                                "y10 (%)": imp10.groupby("nhóm")["share_%"].sum()}).sort_values("y5 (%)", ascending=False).round(1)
    ablation, ci = ins.group_ablation(table, "y_300")
    ablation.to_csv(OUT / "insight_ablation.csv", index=False)
    plots.ablation_bars(ablation, FIG / "insight_ablation.png")
    (OUT / "insight_ablation_ci.json").write_text(json.dumps(ci, indent=2), encoding="utf-8")

    notes = (OUT / "insights_notes.md").read_text(encoding="utf-8") if (OUT / "insights_notes.md").exists() \
        else "_(chưa có nhận xét)_"
    top_shap = imp5.head(20)[["feature", "nhóm", "share_%", "tương quan giá trị–SHAP"]]
    report = f"""# VitalDB cho UC04 — Insight về dữ liệu

Sinh tự động bởi `scripts/eda/run_insights.py` (module `safeanes.insights`). Bổ sung cho [REPORT.md](REPORT.md) (EDA tổng quát).
Phạm vi: nhiễu đo trên **mọi ca eligible** (không chạm outcome); mọi phân tích có nhãn chỉ dùng **ca phát triển ngoài global test**.
Mô hình dùng để đo tầm quan trọng (LightGBM, tham số E08) train trên FIT và báo cáo trên VALIDATION — chỉ mang tính mô tả,
không dùng để chọn mô hình.

## Insight chính

{notes}

---

## 1. Dữ liệu gồm những gì

Mỗi **decision row** là một thời điểm dự báo (30 s một lần; ở đây lấy 1/2 → 60 s) chỉ dùng dữ liệu quá khứ.

{md(pd.Series(scope).to_frame('giá trị'))}

Các nhóm feature (bảng đầy đủ: `insight_dictionary.csv`):

{md(by_group)}

Cách đọc tên feature:
- `map_300_slope`: độ dốc MAP trong 300 s gần nhất.
- `*_60/300/600_*`: thống kê trong 1/5/10 phút gần nhất (`missing` = tỉ lệ thiếu, `mean`, `std`, `min`, `max`, `slope`).
- `*_current`: giá trị mới nhất; `*_age`: số giây từ lần đo cuối.
- `bt_*`: đặc trưng theo nhịp từ waveform ART.
- `*_cur`: median 60 s gần nhất; `*_d300`: thay đổi so với 5 phút trước (log-ratio với tín hiệu dương).
- `static_*`, `sex`, `emop`, `preop_*`: thông tin trước mổ.

Feature chính (phân vị theo decision row):

{md(key_dict[['nhóm', '% thiếu', 'p1', 'p25', 'median', 'p75', 'p99']])}

![missing](figures/insight_missing.png)

## 2. Dữ liệu nhiễu đến mức nào

Đo trên track **thô** trong khoảng opstart → opend của {noise_case.caseid.nunique():,} ca eligible. "Thiếu/cũ" = thời điểm
không có giá trị hợp lệ mới trong 30 s (60 s với BIS/thuốc/CVP/BT; 600 s với NIBP vì đo gián đoạn). "Đứng yên" = chuỗi giá
trị giống hệt kéo dài ≥ 60 s. "Spike" = bước nhảy giữa hai mẫu liên tiếp > 30 mmHg (MAP, HR) hoặc > 40 mmHg (SBP).

![noise](figures/insight_noise.png)

{md(noise, index=False, floatfmt=3)}

Nhiễu ở mức waveform và nhãn:

{md(pd.Series(label_noise).to_frame('giá trị'))}

## 3. Quan hệ giữa các feature

### 3.1 Giữa các feature với nhau

![corr](figures/insight_corr.png)

Cặp tương quan mạnh nhất **giữa hai nhóm khác nhau** (các cặp trong cùng nhóm, như MAP–MAP, hiển nhiên dư thừa):

{md(cross.rename(columns={'level_0': 'feature 1', 'level_1': 'feature 2'})[['feature 1', 'feature 2', 'nhóm 1', 'nhóm 2', 'rho']], index=False, floatfmt=3)}

### 3.2 Với diễn biến MAP trong 5 phút tới

ρ Spearman giữa feature tại thời điểm t và ΔMAP(t → t+5 phút). Đây là quan hệ dự báo, không phải nhân quả.

![future](figures/insight_future.png)

Ngoài nhóm huyết áp, các feature liên quan nhất:

{md(future_non_bp[['feature', 'nhóm', 'rho', 'n']], index=False, floatfmt=3)}

### 3.3 Nguy cơ theo giá trị feature

![risk](figures/insight_risk_curves.png)

![level-trend](figures/insight_level_trend.png)

## 4. Feature ảnh hưởng nhất

### 4.1 Từng feature đơn lẻ (AUROC đơn biến)

![uni](figures/insight_univariate.png)

Feature tốt nhất của mỗi nhóm:

{md(uni_best[['feature', 'nhóm', 'auroc_y_300', 'hướng_y_300', 'auroc_y_600']], index=False, floatfmt=3)}

### 4.2 Trong mô hình đa biến (LightGBM + TreeSHAP)

Hiệu năng VALIDATION của mô hình đầy đủ ({len(ins.feature_columns(table))} feature): y5 AUROC {perf5['auroc']:.3f}, AP {perf5['ap']:.3f};
y10 AUROC {perf10['auroc']:.3f}, AP {perf10['ap']:.3f} ({perf5['n_valid']:,} decision row). "Tương quan giá trị–SHAP" âm nghĩa là
giá trị thấp làm tăng nguy cơ.

![shap](figures/insight_shap.png)

{md(top_shap, index=False, floatfmt=3)}

Tỉ phần mean|SHAP| theo nhóm:

{md(group_share)}

### 4.3 Ablation nhóm tín hiệu (y5, VALIDATION)

![ablation](figures/insight_ablation.png)

{md(ablation.sort_values('AUROC', ascending=False), index=False, floatfmt=4)}

Chênh lệch "Tất cả" − "Chỉ MAP" (bootstrap theo ca, {ci['n_cases']} ca, 95% CI): ΔAUROC {ci['dAUROC'][1]:+.4f}
[{ci['dAUROC'][0]:+.4f}; {ci['dAUROC'][2]:+.4f}], ΔAP {ci['dAP'][1]:+.4f} [{ci['dAP'][0]:+.4f}; {ci['dAP'][2]:+.4f}].

### 4.4 Đối chiếu với các nghiên cứu liên quan UC04

| Feature / nhóm trong y văn | Nguồn |
|---|---|
""" + "\n".join(f"| {a} | {b} |" for a, b in LITERATURE) + """

Ghi chú xác minh: metadata các nguồn đã đối chiếu qua PubMed/PMC/nhà xuất bản hoặc kết quả tìm kiếm ngày 24/09/2026; chưa đọc
toàn văn tất cả. Số liệu của paper là số tác giả công bố, không phải kết quả tái lập trên VitalDB.
"""
    (OUT / "INSIGHTS.md").write_text(eda.apply_captions(report, eda.load_captions(OUT / "figure_captions.md")), encoding="utf-8")
    print("xong:", OUT / "INSIGHTS.md", flush=True)


if __name__ == "__main__":
    main()
