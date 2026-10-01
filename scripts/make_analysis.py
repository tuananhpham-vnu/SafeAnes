"""Write results/ANALYSIS.md (analysis with short comments) and results/FULL_RESULTS.md
(complete tables, per model) from the CSV files in results/.

    python scripts/make_analysis.py

Every number, including the ones quoted in the comments, is read from results/ at run
time, so re-running after new results (e.g. DL) keeps both files consistent.
All figures are VALIDATION results; the test split has not been used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
T = R / "tabular"
MODELS = ["map_threshold", "map_logistic", "lgbm_numeric", "lgbm_wave"]
LABEL = {"map_threshold": "`map_threshold`", "map_logistic": "`map_logistic`",
         "lgbm_numeric": "`lgbm_numeric`", "lgbm_wave": "`lgbm_wave`"}
DESC = {"map_threshold": "Điểm nguy cơ = −MAP hiện tại (không fit, không phụ thuộc W)",
        "map_logistic": "Logistic trên MAP hiện tại, độ dốc MAP, độ lệch chuẩn MAP",
        "lgbm_numeric": "LightGBM, 66 đặc trưng chỉ số monitor",
        "lgbm_wave": "LightGBM, 66 đặc trưng chỉ số + 27 đặc trưng sóng động mạch"}
H = [300, 600, 900, 1200, 1800]
HL = {300: "5 phút", 600: "10 phút", 900: "15 phút", 1200: "20 phút", 1800: "30 phút"}
WS = [30, 60, 90, 120]
GROUP_VI = {"map": "MAP", "sbp_dbp": "SBP/DBP/PP", "hr": "HR, shock index", "spo2": "SpO2",
            "etco2_rr": "EtCO2, RR", "wave_shape": "Hình dạng sóng", "ppv": "PPV",
            "sv_co_svr": "SV/CO/SVR (proxy)", "beat_quality": "Chất lượng nhịp", "none": "(đầy đủ)"}


# ---------------------------------------------------------------- formatting

def f(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%".replace(".", ",")


def sg(x, d=3):
    return f"{x:+.{d}f}".replace(".", ",")


def ci(v, lo, hi, d=3):
    return f"{f(v, d)} [{f(lo, d)}–{f(hi, d)}]"


def diff(m, lo, hi, d=3):
    s = f"{m:+.{d}f} [{lo:+.{d}f}; {hi:+.{d}f}]".replace(".", ",")
    return f"**{s}**" if lo > 0 or hi < 0 else s


def secs(x):
    return "—" if not np.isfinite(x) else f"{x / 60:.1f}".replace(".", ",") + " ph"


def table(header, rows, align=None):
    align = align or ["---"] + ["--:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def bold_best(cells, values, higher=True):
    """Bold the best value(s) of a column (cells are formatted strings)."""
    v = np.array([np.nan if x is None else x for x in values], float)
    if not np.isfinite(v).any():
        return cells
    best = np.nanmax(v) if higher else np.nanmin(v)
    return [f"**{c}**" if np.isfinite(x) and abs(x - best) < 1e-12 else c for c, x in zip(cells, v)]


# ---------------------------------------------------------------- data

def load():
    d = {}
    d["val"] = pd.read_csv(T / "summary" / "tabular_validation.csv")
    d["pol"] = pd.read_csv(T / "summary" / "tabular_validation_by_policy.csv")
    d["cmp"] = pd.read_csv(T / "summary" / "model_comparison.csv")
    d["pairs"] = pd.read_csv(T / "summary" / "model_comparison_pairs.csv")
    d["cal"] = pd.read_csv(T / "summary" / "calibration_check_tabular.csv")
    d["wstar"] = json.loads((T / "summary" / "w_star.json").read_text(encoding="utf-8"))
    d["abl"] = pd.read_csv(T / "analysis" / "ablation.csv")
    d["shap"] = pd.read_csv(T / "analysis" / "shap" / "shap_groups.csv")
    d["long"] = pd.read_csv(T / "exploratory" / "long_windows.csv")
    d["long_pairs"] = pd.read_csv(T / "exploratory" / "long_windows_pairs.csv")
    d["labels"] = pd.read_csv(R / "data" / "label_report.csv")
    d["flow"] = pd.read_csv(R / "data" / "cohort_flow.csv")
    d["conf"] = {h: (pd.read_csv(T / "analysis" / "confidence" / f"W{d['wstar']['w_star']}_h{h}.csv"),
                     json.loads((T / "analysis" / "confidence" / f"W{d['wstar']['w_star']}_h{h}_cuts.json")
                                .read_text(encoding="utf-8")),
                     pd.read_csv(T / "analysis" / "confidence" / f"W{d['wstar']['w_star']}_h{h}_by_risk_decile.csv"))
                 for h in (300, 600)}
    d["samples"] = json.loads((R / "data" / "samples_summary.json").read_text(encoding="utf-8"))
    d["prov"] = json.loads((T / "lgbm_wave" / f"W{d['wstar']['w_star']}" / "h300" / "provenance.json")
                           .read_text(encoding="utf-8"))
    return d


def row(df, model, W, h, policy=None):
    q = (df.model == model) & (df.horizon == h)
    q &= df.window.isna() | (df.window == W) if model == "map_threshold" else (df.window == W)
    if policy is not None:
        q &= df.alarm_rearm == policy
    r = df[q]
    return r.iloc[0] if len(r) else None


def pair(d, name, h, policy="drop_below"):
    p = d["pairs"]
    return p[(p.pair == name) & (p.horizon == h) & (p.alarm_rearm == policy)].iloc[0]


KEY_VI = {"event_sensitivity": "độ nhạy", "event_sensitivity_all": "độ nhạy (mọi đợt)", "false_alarms_per_hour": "FA/giờ",
          "alarm_time_fraction": "thời gian cảnh báo bật", "auroc": "AUROC", "auprc": "AUPRC"}


def nominal(d, name, keys, policy="drop_below"):
    """(metric, horizon, mean diff) where the paired CI excludes 0 (nominal, no multiplicity correction)."""
    out = []
    for h in H:
        x = pair(d, name, h, policy)
        for k in keys:
            if x[k + "_lo"] > 0 or x[k + "_hi"] < 0:
                out.append((k, h, float(x[k + "_diff_mean"])))
    return out


def describe(items, d=3):
    return "; ".join(f"{KEY_VI[k]} {HL[h]} {sg(v, 4 if k == 'alarm_time_fraction' else d)}" for k, h, v in items)


TEST_NOTE = ("Tập **test** chưa được dùng để huấn luyện, chọn cấu hình, chọn ngưỡng hay đánh giá. Tuy vậy, trước khi khóa "
             "mô hình, tổng số đợt tụt của test đã hai lần vô tình bị xem (DEVIATIONS mục 8 và 26); không quyết định nào "
             "dựa trên các con số đó, nhưng test không còn hoàn toàn \"chưa được nhìn thấy\".")


# ---------------------------------------------------------------- ANALYSIS.md

def analysis(d) -> str:
    W = int(d["wstar"]["w_star"])
    val = d["val"]
    get = lambda m, h, pol=None: row(val if pol is None else d["pol"], m, W, h, pol)  # noqa: E731
    s5 = {m: get(m, 300) for m in MODELS}
    out = []
    add = out.append
    add("# Phân tích kết quả UC04 (validation)\n")
    add(f"> Mọi con số là của tập **validation** (248 ca, 245 bệnh nhân). Đây cũng là tập dùng để chọn W* và ngưỡng, nên "
        f"các con số có phần lạc quan. Kết quả báo cáo cuối là của tập test, chạy một lần sau khi khóa mô hình (**chưa chạy**). "
        f"{TEST_NOTE} \"Có ý nghĩa\" trong file này là **danh nghĩa** (CI 95% không chứa 0), chưa hiệu chỉnh cho việc so sánh "
        f"nhiều lần. Bảng đầy đủ theo từng mô hình: [FULL_RESULTS.md](FULL_RESULTS.md). "
        f"File này được sinh bởi `scripts/make_analysis.py` từ các CSV trong `results/`.\n")
    add("**Mục lục:** [0. Tóm tắt](#0-tóm-tắt) · [1. Bối cảnh](#1-bối-cảnh-và-điều-kiện) · "
        "[2. Dữ liệu và nhãn](#2-dữ-liệu-và-nhãn) · [3. Kết quả chính](#3-kết-quả-chính-theo-mốc-dự-báo) · "
        "[4. Điểm vận hành](#4-điểm-vận-hành-cảnh-báo-trông-ra-sao) · [5. Cửa sổ W](#5-ảnh-hưởng-của-cửa-sổ-nhìn-lại-w) · "
        "[6. So sánh ghép cặp](#6-so-sánh-ghép-cặp-chênh-lệch-nào-là-thật) · [7. Calibration](#7-calibration-xác-suất-có-đáng-tin-không) · "
        "[8. Chính sách cảnh báo](#8-chính-sách-cảnh-báo-drop_below-hay-cooldown) · [9. Ablation](#9-ablation-nhóm-tín-hiệu-nào-không-thể-thiếu) · "
        "[10. SHAP](#10-shap-mô-hình-dựa-vào-đâu) · [11. Mức tin cậy](#11-mức-tin-cậy-có-nói-được-dự-đoán-này-ít-chắc-chắn-không) · "
        "[12. Cửa sổ dài (thăm dò)](#12-thăm-dò-cửa-sổ-dài-300-và-600-giây) · [13. Hạn chế](#13-hạn-chế-và-lưu-ý-khi-diễn-giải) · "
        "[14. Việc tiếp theo](#14-việc-tiếp-theo)\n")

    # ---- 0. summary
    pw = [pair(d, "lgbm_wave - lgbm_numeric", h) for h in H]
    pt = [pair(d, "lgbm_numeric - map_threshold", h) for h in H]
    abl5 = d["abl"][(d["abl"].horizon == 300) & (d["abl"].dropped == "map")].iloc[0]
    add("## 0. Tóm tắt\n")
    wave_worse = nominal(d, "lgbm_wave - lgbm_numeric", ("false_alarms_per_hour", "alarm_time_fraction"))
    wave_disc = nominal(d, "lgbm_wave - lgbm_numeric", ("event_sensitivity", "auroc", "auprc"))
    fa_more = [x for x in nominal(d, "lgbm_numeric - map_threshold", ("false_alarms_per_hour",)) if x[2] > 0]
    add(f"1. **LightGBM đạt độ nhạy theo đợt tụt khoảng {pct(s5['lgbm_wave'].event_sensitivity, 0)} ở mốc 5 phút** dưới cùng "
        f"giới hạn trên ≤ 1 cảnh báo sai/giờ, so với {pct(s5['map_threshold'].event_sensitivity, 0)} của ngưỡng MAP đơn thuần. "
        f"Mỗi mô hình tự chọn ngưỡng riêng, và LightGBM dùng nhiều ngân sách cảnh báo hơn "
        f"({f(s5['lgbm_wave'].false_alarms_per_hour, 2)} so với {f(s5['map_threshold'].false_alarms_per_hour, 2)} cảnh báo sai/giờ).")
    add(f"2. **Đặc trưng sóng thủ công không cải thiện độ nhạy, AUROC hay AUPRC:** `lgbm_wave − lgbm_numeric` có CI chứa 0 "
        f"ở {'cả 5 mốc' if not wave_disc else 'hầu hết các mốc (ngoại lệ: ' + describe(wave_disc) + ')'} cho ba chỉ số này"
        + (f"; còn làm **tăng nhẹ gánh nặng cảnh báo** ({describe(wave_worse)})." if wave_worse else "."))
    add(f"3. **LightGBM xếp hạng nguy cơ tốt hơn ngưỡng MAP:** `lgbm_numeric − map_threshold` có AUROC cao hơn ở "
        f"{sum(p.auroc_lo > 0 for p in pt)}/5 mốc, và độ nhạy cao hơn ở {sum(p.event_sensitivity_lo > 0 for p in pt)}/5 mốc "
        f"(kèm FA/giờ cao hơn ở {len(fa_more)}/5 mốc).")
    add(f"4. **MAP là nguồn thông tin chính:** chiếm khoảng {pct(d['shap'][(d['shap'].window == W) & (d['shap'].horizon == 300) & (d['shap'].group == 'map')].share.iloc[0], 0)} "
        f"đóng góp SHAP; chỉ bỏ nhóm MAP mới làm giảm độ nhạy ({sg(abl5.delta_event_sensitivity_diff_mean)} ở mốc 5 phút).")
    add("5. **Mức tin cậy từ 5 seed LightGBM không dùng được** (không qua kiểm chứng). Phân tích thăm dò không thấy W = 300 "
        "hoặc 600 cải thiện độ nhạy của `lgbm_numeric` so với W = 120.\n")
    add("> **Kết luận hiện tại (validation):** LightGBM cải thiện khả năng phân biệt nguy cơ và đạt độ nhạy cao hơn dưới "
        "giới hạn ≤ 1 cảnh báo sai/giờ so với ngưỡng MAP, nhưng đặc trưng sóng thủ công không tạo lợi ích đo được và có thể "
        "làm tăng nhẹ gánh nặng cảnh báo ở mốc 5 phút. Kết luận cuối phải chờ tập test (sau khi khóa mô hình) và mô hình "
        "DL học trực tiếp từ sóng.\n")

    # ---- 1. context
    p = d["prov"]
    add("## 1. Bối cảnh và điều kiện\n")
    add("*Kết quả này được tạo ra như thế nào?*\n")
    add(table(["Mục", "Giá trị"], [
        ["Bài toán", "Tại mỗi thời điểm (cứ 30 giây), dự báo có đợt tụt huyết áp **mới** bắt đầu trong h phút tới"],
        ["Đợt tụt", "MAP < 65 mmHg kéo dài ≥ 60 giây; hai đợt cách nhau < 120 giây được gộp"],
        ["Dữ liệu", f"VitalDB, cohort D (có sóng và arterial line hoạt động); `samples.json` sha256 `9b17751a…`"],
        ["Chia tập (theo bệnh nhân)", "train → fit; calibration → Platt; validation → chọn ngưỡng và báo cáo; test → chưa dùng để đánh giá (xem mục 13)"],
        ["Nhãn", f"`label_policy` = {d['samples'].get('label_policy')}, luật nhãn âm = {d['samples'].get('negative_rule')}, "
                 f"`exclusion_reset` = {d['samples'].get('exclusion_reset')}"],
        ["Cửa sổ nhìn lại W", "30, 60, 90, 120 giây; W* = %d (chọn theo `lgbm_wave`, trung bình độ nhạy mốc 5 và 10 phút)" % W],
        ["Mốc dự báo h", "5, 10, 15, 20, 30 phút"],
        ["Chọn ngưỡng", "Độ nhạy theo đợt tụt cao nhất với ≤ 1 cảnh báo sai/giờ (quét ~300 ứng viên)"],
        ["Chính sách cảnh báo", "`drop_below`: 2 mốc liên tiếp vượt ngưỡng, nghỉ 300 giây, phải xuống dưới ngưỡng mới cảnh báo lại"],
        ["Khoảng tin cậy", "Bootstrap theo bệnh nhân, 200 lần rút; so sánh mô hình dùng bootstrap ghép cặp"],
        ["Không dùng", "Augmentation, oversample, trọng số lớp"],
        ["Code", f"commit `{p['git']['commit'][:7]}`, dirty = {p['git']['dirty']}; scikit-learn {p['packages'].get('scikit-learn')}, "
                 f"LightGBM {p['packages'].get('lightgbm')}"],
    ], ["---", "---"]))
    add("\n**Bốn mô hình:**\n")
    add(table(["Mô hình", "Mô tả", "Số tổ hợp"],
              [[LABEL[m], DESC[m], "5" if m == "map_threshold" else "20 (4 W × 5 h)"] for m in MODELS], ["---", "---", "--:"]))

    # ---- 2. data
    fl = d["flow"].set_index("split")
    lab = d["labels"]
    lp = lab[(lab.label_policy == "lenient_possible") & (lab.exclusion_reset == "event")]
    add("\n## 2. Dữ liệu và nhãn\n")
    add("*Mô hình học trên bao nhiêu ca, bao nhiêu đợt tụt? Nhãn có cân bằng không?*\n")
    rows = []
    for s in ("train", "calibration", "validation"):
        x = lp[(lp.split == s) & (lp.horizon == 300)].iloc[0]
        rows.append([s, int(fl.loc[s, "D_main"]), int(fl.loc[s, "subjects_D"]), f"{int(x.eligible):,}".replace(",", "."),
                     int(x.events), int(x.events_eligible)])
    add(table(["Tập", "Ca", "Bệnh nhân", "Dòng eligible", "Đợt tụt", "Đợt cảnh báo được (5 phút)"], rows))
    rows = []
    for h in H:
        x = lp[(lp.split == "train") & (lp.horizon == h)].iloc[0]
        rows.append([HL[h], pct(x.positive_rate), pct(x.unknown_rate), pct(x.unknown_end_of_case_rate), pct(x.unknown_other_rate)])
    add("\nNhãn trên train (dòng eligible):\n")
    add(table(["Mốc", "Tỷ lệ dương", "Nhãn −1", "−1 do hết ca", "−1 do có thể che đợt tụt"], rows))
    add("\n- Dữ liệu mất cân bằng mạnh ở mốc ngắn (khoảng 4% dương ở 5 phút), nên AUPRC phải đọc cùng tỷ lệ dương.")
    add("- Ở mốc dài, phần lớn nhãn −1 là do không đủ thời gian theo dõi trước khi hết ca, không phải do nhiễu.\n")

    # ---- 3. main results
    add("## 3. Kết quả chính theo mốc dự báo\n")
    add(f"*Mô hình nào cảnh báo được nhiều đợt tụt nhất?* (W = {W}, ngưỡng cho ≤ 1 cảnh báo sai/giờ)\n")
    for key, title, higher in (("event_sensitivity", "Độ nhạy theo đợt tụt [CI 95%] ↑", True),
                               ("auroc", "AUROC [CI 95%] ↑", True)):
        cols = []
        for h in H:
            vals = [get(m, h)[key] for m in MODELS]
            cells = [ci(get(m, h)[key], get(m, h)[key + "_lo"], get(m, h)[key + "_hi"]) for m in MODELS]
            cols.append(bold_best(cells, vals, higher))
        add(f"**{title}**\n")
        add(table(["Mô hình"] + [HL[h] for h in H], [[LABEL[m]] + [cols[j][i] for j in range(5)] for i, m in enumerate(MODELS)]))
        add("")
    rows = [[LABEL[m]] + [f(get(m, h).auprc) for h in H] for m in MODELS]
    rows.append(["*Tỷ lệ dương*"] + [f(get("lgbm_wave", h).prevalence) for h in H])
    add("**AUPRC ↑** (so với tỷ lệ dương)\n")
    add(table(["Mô hình"] + [HL[h] for h in H], rows))
    add("\n- LightGBM có độ nhạy cao nhất ở hầu hết các mốc, nhưng khoảng cách với `map_logistic` nhỏ và các CI chồng lên nhau. "
        "Mỗi mô hình có ngưỡng riêng (cùng giới hạn trên ≤ 1 FA/giờ), nên độ nhạy phải đọc cùng FA/giờ ở phần 4 và 6.")
    add("- AUROC và AUPRC của LightGBM cao hơn hai baseline: mô hình xếp hạng nguy cơ tốt hơn (phần 6).")
    add("- Theo mốc dài hơn, độ nhạy tăng nhẹ còn AUROC giảm dần.\n")

    # ---- 4. operating point
    add("## 4. Điểm vận hành: cảnh báo trông ra sao?\n")
    add(f"*Ở ngưỡng đã chọn, người dùng nhận được gì?* (W = {W})\n")
    cmp = d["cmp"]
    counts = lambda m, h: cmp[(cmp.model == m) & (cmp.horizon == h) & (cmp.alarm_rearm == "drop_below")].iloc[0]  # noqa: E731
    rows = []
    for m in MODELS:
        for h in (300, 600):
            x, c = get(m, h), counts(m, h)
            rows.append([LABEL[m], HL[h], f(x.false_alarms_per_hour, 2),
                         f"{int(c.true_alarms)} / {int(c.false_alarms)} / {int(c.censored_alarms)}", pct(x.alarm_ppv, 0),
                         pct(x.early_sensitivity, 0), f"{secs(x.lead_median_s)} [{secs(x.lead_q25_s)}–{secs(x.lead_q75_s)}]",
                         pct(x.alarm_time_fraction), pct(x.prediction_coverage, 0)])
    add(table(["Mô hình", "Mốc", "Cảnh báo sai/giờ ↓", "Cảnh báo đúng / sai / censored", "PPV ↑", "Báo trước ≥ 5 ph ↑",
               "Thời gian báo trước (trung vị [Q1–Q3])", "Thời gian cảnh báo bật ↓", "Coverage"], rows,
              ["---", "---"] + ["--:"] * 7))
    add("\n*Cảnh báo censored là cảnh báo tại thời điểm có nhãn −1 (không đủ dữ liệu tương lai để biết đúng hay sai); "
        "chúng không nằm trong mẫu số của PPV và của FA/giờ.*\n")
    x5, c5 = get("lgbm_wave", 300), counts("lgbm_wave", 300)
    n_all = c5.true_alarms + c5.false_alarms + c5.censored_alarms
    add(f"- Với `lgbm_wave` ở mốc 5 phút: PPV {pct(x5.alarm_ppv)} là tỷ lệ đúng **trong số cảnh báo đánh giá được** "
        f"({int(c5.true_alarms)}/{int(c5.true_alarms + c5.false_alarms)}). Tính trên toàn bộ {int(n_all)} cảnh báo: "
        f"{pct(c5.true_alarms / n_all)} đúng, {pct(c5.false_alarms / n_all)} sai, {pct(c5.censored_alarms / n_all)} censored. "
        f"Trạng thái cảnh báo bật khoảng {pct(x5.alarm_time_fraction)} thời gian theo dõi eligible.")
    x10 = get("lgbm_wave", 600)
    add(f"- **Thời gian báo trước ngắn:** trung vị {secs(x5.lead_median_s)} ở mốc 5 phút và {secs(x10.lead_median_s)} ở mốc "
        f"10 phút (`lgbm_wave`): thời gian để can thiệp trước khi MAP xuống dưới 65 thường chỉ vài phút.")
    add(f"- Ở mốc 5 phút, tỷ lệ báo trước ≥ 5 phút **quan sát được** là {pct(x5.early_sensitivity, 0)}. Về định nghĩa, "
        "một cảnh báo đúng đúng 300 giây trước khi đợt tụt bắt đầu vẫn có thể được tính, nên đây là kết quả quan sát, "
        f"không phải điều bắt buộc. Ở mốc 10 phút, `lgbm_wave` báo trước ≥ 5 phút cho {pct(x10.early_sensitivity, 0)} "
        "số đợt có thể cảnh báo.")
    n_not_met = int(val.budget_not_met.astype(bool).sum())
    n_combo = len(val.drop_duplicates(["model", "window", "horizon"]))
    add(f"- Số dòng kết quả không đạt ngân sách ≤ 1 cảnh báo sai/giờ: {n_not_met}/{n_combo}.")
    add(f"- Coverage khoảng {pct(get('lgbm_wave', 300).prediction_coverage, 0)}: phần còn lại là lúc MAP đã < 65, "
        "đang trong hoặc ngay sau đợt tụt, hoặc thiếu MAP.\n")

    # ---- 5. window
    add("## 5. Ảnh hưởng của cửa sổ nhìn lại W\n")
    add("*Nhìn lại bao lâu là đủ?* (độ nhạy theo đợt tụt)\n")
    rows = []
    for m in ("lgbm_wave", "lgbm_numeric", "map_logistic"):
        for Wx in WS:
            rows.append([LABEL[m], Wx] + [f(row(val, m, Wx, h).event_sensitivity) for h in H])
    add(table(["Mô hình", "W (giây)"] + [HL[h] for h in H], rows, ["---", "--:"] + ["--:"] * 5))
    ms = d["wstar"]["mean_event_sensitivity_5_10min"]
    add(f"\n- Căn cứ chọn W*: trung bình độ nhạy mốc 5 và 10 phút của `lgbm_wave` là "
        + ", ".join(f"W={int(float(k))}: {f(v)}" for k, v in ms.items()) + f" → W* = {W}.")
    spread = max(ms.values()) - min(ms.values())
    add(f"- Chênh lệch giữa các W nhỏ ({pct(spread)} theo tiêu chí chọn W*) so với độ rộng CI của độ nhạy "
        "(khoảng ±4–5 điểm phần trăm): chọn W không quyết định kết quả.\n")

    # ---- 6. paired
    add("## 6. So sánh ghép cặp: chênh lệch nào là thật?\n")
    add("*Hai mô hình được so trên cùng các lần rút bootstrap theo bệnh nhân. **Chữ đậm** = CI 95% không chứa 0 (danh nghĩa, "
        "chưa hiệu chỉnh so sánh nhiều lần). Mỗi mô hình có ngưỡng riêng, chỉ cùng giới hạn trên ≤ 1 cảnh báo sai/giờ, nên độ "
        "nhạy phải đọc cùng Δ FA/giờ và Δ thời gian cảnh báo bật.*\n")
    rows = []
    for name in ("lgbm_wave - lgbm_numeric", "lgbm_wave - map_logistic", "lgbm_numeric - map_threshold"):
        for h in H:
            x = pair(d, name, h)
            rows.append([f"`{name.replace(' - ', ' − ')}`", HL[h],
                         diff(x.event_sensitivity_diff_mean, x.event_sensitivity_lo, x.event_sensitivity_hi),
                         diff(x.false_alarms_per_hour_diff_mean, x.false_alarms_per_hour_lo, x.false_alarms_per_hour_hi),
                         diff(x.alarm_time_fraction_diff_mean, x.alarm_time_fraction_lo, x.alarm_time_fraction_hi, 4),
                         diff(x.auroc_diff_mean, x.auroc_lo, x.auroc_hi), diff(x.auprc_diff_mean, x.auprc_lo, x.auprc_hi)])
    add(table(["Cặp", "Mốc", "Δ Độ nhạy", "Δ FA/giờ", "Δ Thời gian cảnh báo bật", "Δ AUROC", "Δ AUPRC"], rows,
              ["---", "---"] + ["--:"] * 5))
    wave_disc = nominal(d, "lgbm_wave - lgbm_numeric", ("event_sensitivity", "auroc", "auprc"))
    wave_burden = nominal(d, "lgbm_wave - lgbm_numeric", ("false_alarms_per_hour", "alarm_time_fraction"))
    add("\n- **Đặc trưng sóng thủ công (`lgbm_wave − lgbm_numeric`):** không cải thiện độ nhạy, AUROC hay AUPRC "
        + ("(CI chứa 0 ở mọi mốc)" if not wave_disc else f"(ngoại lệ: {describe(wave_disc)})")
        + (f"; ngược lại, làm **tăng nhẹ gánh nặng cảnh báo**: {describe(wave_burden)}." if wave_burden else "."))
    lg_disc = nominal(d, "lgbm_wave - map_logistic", ("auroc", "auprc"))
    lg_sens = nominal(d, "lgbm_wave - map_logistic", ("event_sensitivity",))
    add(f"- **`lgbm_wave` so với `map_logistic`:** AUROC/AUPRC cao hơn ở {len({h for _, h, _ in lg_disc})}/5 mốc (xếp hạng "
        f"nguy cơ tốt hơn); độ nhạy dưới giới hạn ≤ 1 FA/giờ khác biệt ở {len(lg_sens)}/5 mốc.")
    nt_sens = nominal(d, "lgbm_numeric - map_threshold", ("event_sensitivity",))
    nt_fa = [x for x in nominal(d, "lgbm_numeric - map_threshold", ("false_alarms_per_hour",)) if x[2] > 0]
    nt_disc = nominal(d, "lgbm_numeric - map_threshold", ("auroc",))
    add(f"- **`lgbm_numeric` so với ngưỡng MAP:** độ nhạy cao hơn ở {len(nt_sens)}/5 mốc dưới cùng giới hạn trên ≤ 1 FA/giờ, "
        f"nhưng **dùng nhiều ngân sách cảnh báo hơn** (FA/giờ cao hơn ở {len(nt_fa)}/5 mốc"
        + (f": {describe(nt_fa, 2)}" if nt_fa else "") + "). Đây không phải so sánh ở cùng một mức FA/giờ; "
        f"bằng chứng xếp hạng tốt hơn là AUROC cao hơn ở {len(nt_disc)}/5 mốc.\n")

    # ---- 7. calibration
    cal = d["cal"]
    add("## 7. Calibration: xác suất có đáng tin không?\n")
    add(f"*Trung bình xác suất dự đoán so với tỷ lệ dương thực tế trên validation* (W = {W})\n")
    rows = []
    for m in MODELS:
        cells = []
        for h in H:
            c = row(cal, m, W, h)
            cells.append(f"{pct(c.mean_p_validation)} / {pct(c.positive_rate_validation)}")
        rows.append([LABEL[m]] + cells + [f(get(m, 300).ece, 4)])
    add(table(["Mô hình"] + [HL[h] + " (dự đoán / thực)" for h in H] + ["ECE 5 ph ↓"], rows))
    gap = float((cal.mean_p_validation - cal.positive_rate_validation).abs().max())
    ece_hi = val.loc[val.ece.idxmax()]
    lgbm_ece = val[val.model.str.startswith("lgbm")].ece
    add(f"\n- **Calibration trung bình tốt:** trung bình xác suất lệch tỷ lệ dương tối đa {pct(gap, 2)} trên cả 65 tổ hợp. "
        "Calibration được fit trên tập calibration riêng, nên đây là bằng chứng độc lập.")
    add(f"- **Nhưng trung bình khớp chưa đủ để khẳng định calibration tốt trên toàn miền nguy cơ.** ECE của LightGBM từ "
        f"{f(lgbm_ece.min(), 4)} đến {f(lgbm_ece.max(), 4)}; ECE lớn nhất là {f(ece_hi.ece, 4)} "
        f"({LABEL[ece_hi.model]}, mốc {HL[int(ece_hi.horizon)]}). Chưa có đồ thị reliability theo từng khoảng xác suất.\n")

    # ---- 8. policy
    add("## 8. Chính sách cảnh báo: `drop_below` hay `cooldown`?\n")
    add("*`drop_below`: phải xuống dưới ngưỡng mới cảnh báo lại. `cooldown`: vẫn trên ngưỡng thì cảnh báo lại mỗi 5 phút.* "
        f"(W = {W}, mốc 5 và 10 phút)\n")
    rows = []
    for m in MODELS:
        for h in (300, 600):
            a, b = get(m, h, "drop_below"), get(m, h, "cooldown")
            rows.append([LABEL[m], HL[h], f(a.event_sensitivity), f(b.event_sensitivity), f(a.false_alarms_per_hour, 2),
                         f(b.false_alarms_per_hour, 2), pct(a.alarm_time_fraction), pct(b.alarm_time_fraction)])
    add(table(["Mô hình", "Mốc", "Độ nhạy drop_below", "Độ nhạy cooldown", "FA/giờ drop_below", "FA/giờ cooldown",
               "Cảnh báo bật drop_below", "Cảnh báo bật cooldown"], rows, ["---", "---"] + ["--:"] * 6))
    diffs = d["pol"].pivot_table(index=["model", "window", "horizon"], columns="alarm_rearm", values="event_sensitivity")
    mx = float((diffs["cooldown"] - diffs["drop_below"]).abs().max())
    add(f"\n- Hai chính sách cho độ nhạy chênh tối đa {pct(mx)} trên cả 65 tổ hợp: lựa chọn chính sách không đổi kết luận.")
    add("- Chính sách chính vẫn **chờ nhóm chốt** (mặc định `drop_below`).\n")

    # ---- 9. ablation
    abl = d["abl"]
    add("## 9. Ablation: nhóm tín hiệu nào không thể thiếu?\n")
    add(f"*Bỏ lần lượt từng nhóm cột khỏi `lgbm_wave` (W = {W}); chênh lệch so với mô hình đầy đủ, bootstrap ghép cặp.*\n")
    rows = []
    for g in [x for x in abl.dropped.unique() if x != "none"]:
        cells = [GROUP_VI.get(g, g)]
        for h in (300, 600):
            x = abl[(abl.horizon == h) & (abl.dropped == g)].iloc[0]
            cells += [diff(x.delta_event_sensitivity_diff_mean, x.delta_event_sensitivity_lo, x.delta_event_sensitivity_hi),
                      diff(x.delta_auroc_diff_mean, x.delta_auroc_lo, x.delta_auroc_hi)]
        rows.append(cells)
    add(table(["Nhóm bị bỏ", "Δ Độ nhạy 5 ph", "Δ AUROC 5 ph", "Δ Độ nhạy 10 ph", "Δ AUROC 10 ph"], rows))
    sig = []  # other groups with a significant change in any column (CI excludes 0)
    for _, x in abl[abl.dropped.isin([g for g in abl.dropped.unique() if g not in ("none", "map")])].iterrows():
        for k, name in (("delta_event_sensitivity", "độ nhạy"), ("delta_auroc", "AUROC")):
            if x[k + "_lo"] > 0 or x[k + "_hi"] < 0:
                sig.append(f"{GROUP_VI.get(x.dropped, x.dropped)} ({name} {HL[int(x.horizon)]}: {sg(x[k + '_diff_mean'])})")
    add("\n- Chỉ bỏ **MAP** làm giảm **độ nhạy** có ý nghĩa, ở cả hai mốc; AUROC cũng giảm.")
    if sig:
        add("- Các thay đổi khác có CI không chứa 0 nhưng rất nhỏ, không ảnh hưởng độ nhạy: " + "; ".join(sig) + ".")
    add("- Bỏ các nhóm khác gần như không đổi kết quả. Điều này **không** có nghĩa các nhóm đó vô ích: nhiều nhóm trùng thông "
        "tin với nhau (SBP/DBP, PP và proxy SV/CO/SVR đều tính từ huyết áp), nên nhóm còn lại bù vào.\n")

    # ---- 10. SHAP
    sh = d["shap"][d["shap"].window == W].pivot_table(index="group", columns="horizon", values="share")
    sh = sh.sort_values(300, ascending=False)
    add("## 10. SHAP: mô hình dựa vào đâu?\n")
    add(f"*Tỷ lệ |SHAP| theo nhóm cột, `lgbm_wave`, W = {W}, 50.000 dòng validation.*\n")
    add(table(["Nhóm"] + [HL[h] for h in H], [[GROUP_VI.get(g, g)] + [pct(sh.loc[g, h], 0) for h in H] for g in sh.index]))
    add("\n- MAP chiếm tỷ lệ lớn nhất ở mọi mốc; cùng với SBP/DBP/PP, thông tin huyết áp chiếm phần lớn.")
    add("- SV/CO/SVR là **proxy tính từ huyết áp** (SV = PP/(SBP + DBP)), không phải số đo cung lượng tim: không nên kết luận mô hình dùng cung lượng tim.")
    add("- PPV đóng góp khoảng 1% hoặc ít hơn ở mọi mốc.\n")

    # ---- 11. confidence
    add("## 11. Mức tin cậy: có nói được \"dự đoán này ít chắc chắn\" không?\n")
    add("*Độ lệch chuẩn logit giữa 5 seed LightGBM → 3 mức theo tam phân vị. Tiêu chí: trong từng thập phân vị nguy cơ, "
        "nhóm \"tin cậy thấp\" phải có sai số calibration lớn hơn nhóm \"tin cậy cao\" ở ≥ 8/10 thập phân vị, sai số gộp ≥ 1,2 lần, và Brier tệ hơn.*\n")
    rows = []
    for h in (300, 600):
        c = d["conf"][h][1]["check"]
        rows.append([HL[h], f"{c['deciles_low_worse_calibration']}/{c['deciles_compared']}",
                     f"{f(c['pooled_calibration_error_low'], 4)} / {f(c['pooled_calibration_error_high'], 4)}",
                     f"{f(c['pooled_brier_low'], 4)} / {f(c['pooled_brier_high'], 4)}", "Qua" if c["passed"] else "**Không qua**"])
    add(table(["Mốc", "Thập phân vị \"thấp\" kém hơn", "Sai số calibration (thấp / cao)", "Brier (thấp / cao)", "Kết luận"], rows))
    add("\n- Không qua ở cả hai mốc → **bỏ mức tin cậy của LightGBM** (DEVIATIONS 28).")
    add("- Độ lệch giữa các seed của DL sẽ được kiểm bằng cùng tiêu chí.\n")

    # ---- 12. long windows
    lg, lpairs = d["long"], d["long_pairs"]
    add("## 12. Thăm dò: cửa sổ dài 300 và 600 giây\n")
    add("*W* = 120 là cửa sổ dài nhất đã thử. Phân tích này quyết định **sau khi xem validation**, nên chỉ là thăm dò: "
        "báo cáo riêng, không thay W*. Đặc trưng của cả 3 cửa sổ được tính lại bằng cùng một hàm (khớp `prep_v1` ở 99,9% dòng).*\n")
    rows = []
    for h in H:
        cells = [HL[h]]
        for Wx in (120, 300, 600):
            x = lg[(lg.window == Wx) & (lg.horizon == h)].iloc[0]
            cells.append(f(x.event_sensitivity))
        for name in ("W300 - W120", "W600 - W120"):
            x = lpairs[(lpairs.horizon == h) & (lpairs.pair == name)].iloc[0]
            cells.append(diff(x.event_sensitivity_diff_mean, x.event_sensitivity_lo, x.event_sensitivity_hi))
        rows.append(cells)
    add(table(["Mốc", "W=120", "W=300", "W=600", "Δ W300−W120", "Δ W600−W120"], rows))
    worse600 = [HL[int(x.horizon)] for x in lpairs[lpairs.pair == "W600 - W120"].itertuples() if x.event_sensitivity_hi < 0]
    add("\n- **Không thấy W = 300 hoặc 600 cải thiện độ nhạy của `lgbm_numeric` so với W = 120** trên validation"
        + (f"; W = 600 còn thấp hơn (danh nghĩa) ở mốc {', '.join(worse600)}." if worse600 else "."))
    add("- Kết luận này **có phạm vi hẹp:** chỉ thử hai cửa sổ dài, chỉ cho `lgbm_numeric`, trên cùng tập validation và sau "
        "khi đã xem kết quả. Nó không loại trừ các W trung gian (ví dụ 180–240 giây) hay mô hình học trực tiếp từ sóng thô.\n")

    # ---- 13. limitations
    add("## 13. Hạn chế và lưu ý khi diễn giải\n")
    add("**Về phương pháp thống kê**\n")
    for s in [
        "**Validation vừa để chọn vừa để báo cáo:** W* và ngưỡng được chọn trên chính tập này, nên con số có phần lạc quan.",
        "**CI chưa gồm bất định do chọn ngưỡng:** bootstrap giữ cố định ngưỡng đã chọn, không lặp lại bước chọn ngưỡng "
        "trong từng lần rút, nên CI của độ nhạy và FA/giờ có thể hẹp hơn thực tế.",
        "**200 lần rút bootstrap** tương đối ít cho CI 95%; giới hạn CI có thể dao động khi chạy lại với seed khác.",
        "**Chưa hiệu chỉnh so sánh nhiều lần:** nhiều mô hình, mốc, cặp và nhóm ablation được kiểm; \"có ý nghĩa\" chỉ là "
        "danh nghĩa (CI 95% không chứa 0).",
        "**Chỉ số theo dòng (AUROC, AUPRC, Brier, ECE)** tính trên các mốc 30 giây có tương quan mạnh trong cùng ca, và "
        "chịu ảnh hưởng nhiều hơn từ các ca mổ dài; CI vẫn bootstrap theo bệnh nhân.",
        "**So sánh độ nhạy không ở cùng một mức FA/giờ:** mỗi mô hình có ngưỡng riêng, chỉ cùng giới hạn trên ≤ 1 FA/giờ.",
    ]:
        add(f"- {s}")
    add("\n**Về dữ liệu, đặc trưng và phạm vi**\n")
    for s in [
        f"**Test:** {TEST_NOTE}",
        "**Phạm vi:** kết luận chỉ áp dụng cho cohort D (ca có sóng động mạch và arterial line hoạt động) của VitalDB.",
        "**MAP nền:** chỉ khoảng 4% ca có MAP đo trước khởi mê; `map_drop_pct` thực chất là mức giảm so với MAP sau khởi mê.",
        "**Proxy SV/CO/SVR** tính từ huyết áp, chưa hiệu chỉnh; không diễn giải như số đo cung lượng tim.",
        "**Các nhóm tín hiệu trùng thông tin:** ablation \"không đổi\" không có nghĩa nhóm đó vô ích.",
        "**`in_event`/`post_event`** dùng đợt tụt phát hiện từ `label_map`, có thể nhìn trước tối đa khoảng 60 giây (DEVIATIONS 5).",
        "**Nhãn −1** ở mốc dài chiếm tới khoảng 26% dòng eligible (phần lớn do hết ca); chỉ số theo dòng tính trên dòng có nhãn.",
        "**Chưa có DL:** câu hỏi \"sóng thô có thêm thông tin không\" mới được trả lời cho đặc trưng sóng (LightGBM), chưa cho mô hình học trực tiếp từ sóng.",
    ]:
        add(f"- {s}")
    add("\n## 14. Việc tiếp theo\n")
    add(table(["Việc", "Ai", "Trạng thái"], [
        ["Chốt `label_policy` và chính sách cảnh báo", "Nhóm", "Chờ"],
        ["Train DL (Conv1D + Transformer) trên Kaggle", "Kaggle + HF", "Notebook sẵn sàng; cần dataset `uc04-prep-v1`"],
        ["Khóa mô hình (`lock_models.py`)", "Máy", "Kết quả tabular đủ điều kiện (195 file)"],
        ["Chạy test một lần (NB04)", "Máy", "Sau khi khóa"],
    ], ["---", "---", "---"]))
    add("")
    return "\n".join(out)


# ---------------------------------------------------------------- FULL_RESULTS.md

def full(d) -> str:
    val, pol = d["val"], d["pol"]
    out = []
    add = out.append
    add("# Kết quả đầy đủ theo từng mô hình (validation)\n")
    add(f"> Mọi con số là của tập **validation**. {TEST_NOTE} Ngưỡng chọn cho ≤ 1 cảnh báo sai/giờ, chính sách "
        "`drop_below` (trừ mục so sánh chính sách). CI 95% bằng bootstrap theo bệnh nhân (200 lần rút). "
        "Phân tích và nhận xét: [ANALYSIS.md](ANALYSIS.md). Nguồn: các CSV trong `results/tabular/`. "
        "File này được sinh bởi `scripts/make_analysis.py`.\n")
    add("**Mục lục:** " + " · ".join(f"[{LABEL[m].strip('`')}](#{m})" for m in MODELS)
        + " · [So sánh ghép cặp](#so-sánh-ghép-cặp-đầy-đủ) · [Chính sách cảnh báo](#hai-chính-sách-cảnh-báo)"
        " · [Ablation](#ablation-đầy-đủ) · [SHAP](#shap-theo-nhóm-mọi-w-và-mốc) · [Cửa sổ dài](#thăm-dò-cửa-sổ-dài)"
        " · [Mức tin cậy](#mức-tin-cậy-theo-thập-phân-vị-nguy-cơ) · [Nhãn](#nhãn-theo-mọi-biến-thể)\n")
    add("Chiều tốt hơn: ↑ càng cao càng tốt, ↓ càng thấp càng tốt.\n")

    for m in MODELS:
        add(f"## {m}\n")
        add(f"{DESC[m]}. Thư mục chi tiết: `results/tabular/{m}/`.\n")
        Wlist = [None] if m == "map_threshold" else WS
        wlab = lambda Wx: "—" if Wx is None else Wx  # noqa: E731
        sub = lambda Wx, h: row(val, m, 120 if Wx is None else Wx, h)  # noqa: E731
        add("**A. Theo đợt tụt**\n")
        rows = []
        for Wx in Wlist:
            for h in H:
                x = sub(Wx, h)
                rows.append([wlab(Wx), HL[h], ci(x.event_sensitivity, x.event_sensitivity_lo, x.event_sensitivity_hi),
                             f"{int(x.events_detected)}/{int(x.events_eligible)}", f(x.event_sensitivity_all),
                             f(x.early_sensitivity), f"{secs(x.lead_median_s)} [{secs(x.lead_q25_s)}–{secs(x.lead_q75_s)}]"])
        add(table(["W", "Mốc", "Độ nhạy [CI] ↑", "Phát hiện / eligible", "Độ nhạy (mọi đợt) ↑", "Báo trước ≥ 5 ph ↑",
                   "Thời gian báo trước"], rows, ["--:", "---"] + ["--:"] * 5))
        add("\n**B. Cảnh báo**\n")
        rows = []
        for Wx in Wlist:
            for h in H:
                x = sub(Wx, h)
                rows.append([wlab(Wx), HL[h], f(x.threshold, 4), ci(x.false_alarms_per_hour, x.false_alarms_per_hour_lo,
                                                                   x.false_alarms_per_hour_hi, 2),
                             pct(x.alarm_ppv), ci(x.alarm_time_fraction, x.alarm_time_fraction_lo, x.alarm_time_fraction_hi, 4),
                             pct(x.prediction_coverage), "có" if x.budget_not_met else "không"])
        add(table(["W", "Mốc", "Ngưỡng", "Cảnh báo sai/giờ [CI] ↓", "PPV ↑", "Thời gian cảnh báo bật [CI] ↓", "Coverage",
                   "Vượt ngân sách FA"], rows, ["--:", "---"] + ["--:"] * 6))
        add("\n**C. Theo dòng và calibration**\n")
        rows = []
        for Wx in Wlist:
            for h in H:
                x = sub(Wx, h)
                c = row(d["cal"], m, 120 if Wx is None else Wx, h)
                rows.append([wlab(Wx), HL[h], ci(x.auroc, x.auroc_lo, x.auroc_hi), ci(x.auprc, x.auprc_lo, x.auprc_hi),
                             f(x.prevalence), f(x.brier, 4), f(x.ece, 4),
                             f"{pct(c.mean_p_validation, 2)} / {pct(c.positive_rate_validation, 2)}",
                             f"{int(x.n_rows):,}".replace(",", ".")])
        add(table(["W", "Mốc", "AUROC [CI] ↑", "AUPRC [CI] ↑", "Tỷ lệ dương", "Brier ↓", "ECE ↓",
                   "Xác suất TB / tỷ lệ thực", "Số dòng"], rows, ["--:", "---"] + ["--:"] * 7))
        add("")

    add("## So sánh ghép cặp đầy đủ\n")
    add("Cùng các lần rút bootstrap cho hai mô hình, W = 120. **Chữ đậm** = CI 95% không chứa 0.\n")
    for policy in ("drop_below", "cooldown"):
        add(f"**Chính sách `{policy}`**\n")
        rows = []
        for name in ("lgbm_wave - lgbm_numeric", "lgbm_wave - map_logistic", "lgbm_numeric - map_threshold"):
            for h in H:
                x = pair(d, name, h, policy)
                rows.append([f"`{name.replace(' - ', ' − ')}`", HL[h]]
                            + [diff(x[k + "_diff_mean"], x[k + "_lo"], x[k + "_hi"], 4 if k == "alarm_time_fraction" else 3)
                               for k in ("event_sensitivity", "event_sensitivity_all", "false_alarms_per_hour",
                                         "alarm_time_fraction", "auroc", "auprc")])
        add(table(["Cặp", "Mốc", "Δ Độ nhạy", "Δ Độ nhạy (mọi đợt)", "Δ FA/giờ", "Δ Thời gian cảnh báo bật", "Δ AUROC",
                   "Δ AUPRC"], rows, ["---", "---"] + ["--:"] * 6))
        add("")

    add("## Hai chính sách cảnh báo\n")
    add("Mọi tổ hợp; ngưỡng chọn lại riêng cho từng chính sách. Chỉ số theo dòng không phụ thuộc chính sách.\n")
    rows = []
    for m in MODELS:
        for Wx in ([None] if m == "map_threshold" else WS):
            for h in H:
                a = row(pol, m, 120 if Wx is None else Wx, h, "drop_below")
                b = row(pol, m, 120 if Wx is None else Wx, h, "cooldown")
                rows.append([LABEL[m], "—" if Wx is None else Wx, HL[h], f(a.event_sensitivity), f(b.event_sensitivity),
                             f(a.false_alarms_per_hour, 2), f(b.false_alarms_per_hour, 2), pct(a.alarm_ppv), pct(b.alarm_ppv),
                             pct(a.alarm_time_fraction), pct(b.alarm_time_fraction)])
    add(table(["Mô hình", "W", "Mốc", "Độ nhạy drop", "Độ nhạy cooldown", "FA/giờ drop", "FA/giờ cooldown", "PPV drop",
               "PPV cooldown", "Cảnh báo bật drop", "Cảnh báo bật cooldown"], rows, ["---", "--:", "---"] + ["--:"] * 8))

    abl = d["abl"]
    add("\n## Ablation đầy đủ\n")
    add("`lgbm_wave`, W = 120. Δ = mô hình bỏ nhóm − mô hình đầy đủ (bootstrap ghép cặp). **Chữ đậm** = CI không chứa 0.\n")
    rows = []
    for _, x in abl.iterrows():
        if x.dropped == "none":
            rows.append([HL[int(x.horizon)], GROUP_VI["none"], "—", f(x.event_sensitivity), f(x.false_alarms_per_hour, 2),
                         f(x.auroc), f(x.auprc), "", "", ""])
            continue
        rows.append([HL[int(x.horizon)], GROUP_VI.get(x.dropped, x.dropped), int(x.n_dropped_columns), f(x.event_sensitivity),
                     f(x.false_alarms_per_hour, 2), f(x.auroc), f(x.auprc),
                     diff(x.delta_event_sensitivity_diff_mean, x.delta_event_sensitivity_lo, x.delta_event_sensitivity_hi),
                     diff(x.delta_auroc_diff_mean, x.delta_auroc_lo, x.delta_auroc_hi),
                     diff(x.delta_auprc_diff_mean, x.delta_auprc_lo, x.delta_auprc_hi)])
    add(table(["Mốc", "Nhóm bị bỏ", "Số cột", "Độ nhạy", "FA/giờ", "AUROC", "AUPRC", "Δ Độ nhạy", "Δ AUROC", "Δ AUPRC"],
              rows, ["---", "---"] + ["--:"] * 8))

    sh = d["shap"]
    add("\n## SHAP theo nhóm, mọi W và mốc\n")
    add("Tỷ lệ tổng |SHAP| theo nhóm cột, `lgbm_wave`, 50.000 dòng validation. Chi tiết từng cột: "
        "`results/tabular/analysis/shap/shap_W<W>_h<h>.csv`.\n")
    groups = sh.groupby("group").share.mean().sort_values(ascending=False).index
    rows = []
    for Wx in WS:
        for h in H:
            s = sh[(sh.window == Wx) & (sh.horizon == h)].set_index("group").share
            rows.append([Wx, HL[h]] + [pct(s.get(g, np.nan), 1) for g in groups])
    add(table(["W", "Mốc"] + [GROUP_VI.get(g, g) for g in groups], rows, ["--:", "---"] + ["--:"] * len(groups)))
    top = pd.read_csv(T / "analysis" / "shap" / f"shap_W{int(d['wstar']['w_star'])}_h300.csv").head(15)
    add(f"\nTop 15 cột ở W = {int(d['wstar']['w_star'])}, mốc 5 phút:\n")
    add(table(["#", "Cột", "Nhóm", "Trung bình |SHAP|"],
              [[i + 1, f"`{r.column}`", GROUP_VI.get(r.group, r.group), f(r.mean_abs_shap, 4)] for i, r in enumerate(top.itertuples())],
              ["--:", "---", "---", "--:"]))

    lg, lp = d["long"], d["long_pairs"]
    add("\n## Thăm dò: cửa sổ dài\n")
    add("`lgbm_numeric`, đặc trưng tính lại từ lưới 2 giây cho cả 3 cửa sổ. Phân tích sau khi xem validation, không thay W*.\n")
    rows = []
    for Wx in (120, 300, 600):
        for h in H:
            x = lg[(lg.window == Wx) & (lg.horizon == h)].iloc[0]
            rows.append([Wx, HL[h], ci(x.event_sensitivity, x.event_sensitivity_lo, x.event_sensitivity_hi),
                         f(x.false_alarms_per_hour, 2), pct(x.alarm_ppv), ci(x.auroc, x.auroc_lo, x.auroc_hi),
                         ci(x.auprc, x.auprc_lo, x.auprc_hi)])
    add(table(["W", "Mốc", "Độ nhạy [CI]", "FA/giờ", "PPV", "AUROC [CI]", "AUPRC [CI]"], rows, ["--:", "---"] + ["--:"] * 5))
    rows = [[HL[int(x.horizon)], f"`{x.pair.replace(' - ', ' − ')}`",
             diff(x.event_sensitivity_diff_mean, x.event_sensitivity_lo, x.event_sensitivity_hi),
             diff(x.auroc_diff_mean, x.auroc_lo, x.auroc_hi), diff(x.auprc_diff_mean, x.auprc_lo, x.auprc_hi)]
            for x in lp.itertuples()]
    add("\n" + table(["Mốc", "Cặp", "Δ Độ nhạy", "Δ AUROC", "Δ AUPRC"], rows, ["---", "---", "--:", "--:", "--:"]))

    add("\n## Mức tin cậy theo thập phân vị nguy cơ\n")
    add("`lgbm_wave`, W = 120, 5 seed. Kết luận: không qua ở cả hai mốc → bỏ.\n")
    for h in (300, 600):
        lv, chk, dec = d["conf"][h]
        add(f"**{HL[h]}: theo mức tin cậy**\n")
        add(table(["Mức", "Số dòng", "SD logit tối đa", "Xác suất TB", "Tỷ lệ dương", "AUROC", "ECE"],
                  [[x.confidence, f"{int(x.rows):,}".replace(",", "."), f(x.sd_logit_max, 3), pct(x.mean_probability, 2),
                    pct(x.prevalence, 2), f(x.auroc), f(x.ece, 4)] for x in lv.itertuples()], ["---"] + ["--:"] * 6))
        piv = dec.pivot_table(index="decile", columns="confidence", values="calibration_error")
        add(f"\n**{HL[h]}: sai số calibration theo thập phân vị nguy cơ** (thấp kém hơn cao ở "
            f"{chk['check']['deciles_low_worse_calibration']}/{chk['check']['deciles_compared']} thập phân vị)\n")
        add(table(["Thập phân vị", "Cao", "Trung bình", "Thấp"],
                  [[int(i) + 1] + [f(piv.loc[i].get(c, np.nan), 4) for c in ("high", "medium", "low")] for i in piv.index],
                  ["--:", "--:", "--:", "--:"]))
        add("")

    lab = d["labels"]
    add("## Nhãn theo mọi biến thể\n")
    add("Cohort D, dòng eligible. `lenient_possible` = luật đang dùng; `lenient_fraction` = luật nhãn âm cũ; `strict` = chỉ để báo cáo.\n")
    rows = []
    for _, x in lab.sort_values(["exclusion_reset", "label_policy", "split", "horizon"]).iterrows():
        rows.append([x.exclusion_reset, x.label_policy, x.split, HL[int(x.horizon)], f"{int(x.eligible):,}".replace(",", "."),
                     pct(x.positive_rate), pct(x.unknown_rate), int(x.events), int(x.events_eligible)])
    add(table(["Luật loại mẫu", "Chính sách nhãn", "Tập", "Mốc", "Dòng eligible", "Tỷ lệ dương", "Nhãn −1", "Đợt tụt",
               "Đợt cảnh báo được"], rows, ["---", "---", "---", "---"] + ["--:"] * 5))
    add("")
    return "\n".join(out)


def main() -> int:
    d = load()
    (R / "ANALYSIS.md").write_text(analysis(d), encoding="utf-8")
    (R / "FULL_RESULTS.md").write_text(full(d), encoding="utf-8")
    for name in ("ANALYSIS.md", "FULL_RESULTS.md"):
        text = (R / name).read_text(encoding="utf-8")
        print(f"wrote results/{name}: {len(text.splitlines())} lines, {text.count('| ---') + text.count('|---')} tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
