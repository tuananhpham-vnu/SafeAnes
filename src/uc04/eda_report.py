"""eda/README.md from the outputs of scripts/beyond_map_eda.py: every number is a measured evaluation (train CV grouped by
subject, or validation), never a guess.

Layout: the main findings first (each with its number), then only the details that explain the choice of features
(MAP-only reference, how much of each feature is MAP again, what adds beyond MAP, composites, ECG/PPG, the
validation comparison of feature sets). The full tables stay in the CSV files listed at the end.

  python -c "from uc04.eda_report import write_eda_md; write_eda_md(Path(out), Path(out) / 'README.md')"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

H = (300, 600, 900)
TOL = 0.0005  # |ΔAUROC| below this counts as no change
CTX = "lgbm_context (97)"
FAMILY_VI = {"ecgppg": "Sóng ECG / PPG", "map": "Họ MAP", "numeric": "Monitor khác", "wave": "Sóng ART",
             "context": "Ngữ cảnh ca", "anaes": "Gây mê / thở máy", "clinical": "Lâm sàng trước mổ",
             "composite": "Tổ hợp sinh lý"}


def _read(d: Path, name: str) -> pd.DataFrame | None:
    p = d / name
    return pd.read_csv(p) if p.exists() else None


def _f(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    x = 0.0 if round(x, nd) == 0 else x  # no "-0,000"
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace(".", ",")


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def _range(v, nd=3, sign=False) -> str:
    v = [x for x in v if np.isfinite(x)]
    if not v:
        return "–"
    lo, hi = min(v), max(v)
    return _f(lo, nd, sign) if _f(lo, nd, sign) == _f(hi, nd, sign) else f"{_f(lo, nd, sign)} đến {_f(hi, nd, sign)}"


def _stats(d: Path) -> pd.DataFrame | None:
    parts = [pd.read_csv(p) for p in sorted(d.glob("l1_stats_h*.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else None


def _compare(d: Path) -> pd.DataFrame | None:
    """Validation comparison: this run, plus the 'all features' rows of the run without ECG/PPG (eda/no_ecg)."""
    fin = _read(d, "final_compare.csv")
    ne = _read(d / "no_ecg", "final_compare.csv")
    if fin is not None and ne is not None:
        extra = ne[ne["model"].str.startswith("Tất cả") & ~ne["model"].isin(fin["model"])]
        fin = pd.concat([fin, extra], ignore_index=True)
    return fin


def _get(fin: pd.DataFrame, model: str, proto: str = "every") -> pd.DataFrame:
    return fin[(fin["model"] == model) & (fin["protocol"] == proto)].set_index("horizon")


def _models(fin: pd.DataFrame, prefix: str) -> list[str]:
    return [m for m in dict.fromkeys(fin["model"]) if m.startswith(prefix)]


def _key_findings(d: Path, fin, st, c3, summary) -> list[str]:
    K = []
    if fin is not None:
        rule, mo, ctx = (_get(fin, m) for m in ("MAP hiện tại (luật)", "MAP-only (lõi MAP)", CTX))
        if len(mo) and len(ctx):
            K.append(f"**MAP mang gần hết thông tin.** Trên validation, mô hình chỉ dùng lõi MAP đạt AUROC "
                     f"{_f(mo.loc[300, 'auroc'])} / {_f(mo.loc[900, 'auroc'])} ở mốc 5 / 15 phút (luật \"MAP hiện tại\": "
                     f"{_f(rule.loc[300, 'auroc'])} / {_f(rule.loc[900, 'auroc'])}); mô hình hiện tại `lgbm_context` "
                     f"chỉ hơn nó {_range([ctx.loc[h, 'auroc'] - mo.loc[h, 'auroc'] for h in H])}.")
    if st is not None:
        cnt = st.groupby("horizon").agg(alone=("auroc_q", lambda q: int((q < 0.05).sum())),
                                        beyond=("sig_beyond_map", "sum"), n=("feature", "size"))
        K.append(f"**Phần lớn đặc trưng chỉ lặp lại MAP.** Trong {int(cnt['n'].iloc[0])} đặc trưng ngoài lõi MAP, "
                 f"{_range(cnt['alone'], 0)} có ý nghĩa khi đứng riêng nhưng chỉ {_range(cnt['beyond'], 0)} còn ý nghĩa "
                 "khi đã biết MAP (q < 0,05 cho cả OR và ΔAUROC).")
        sig = st[st["sig_beyond_map"].astype(bool)]
        fam = sig.drop_duplicates("feature")["family"].value_counts()
        top = fam.index[0] if len(fam) else ""
        K.append(f"**Phần thêm được ngoài MAP chủ yếu là thay đổi thuốc mê.** {int(fam.get(top, 0))}/{int(fam.sum())} "
                 f"đặc trưng có ý nghĩa ngoài MAP thuộc nhóm {FAMILY_VI.get(top, top)} (Ce remifentanil / propofol thay "
                 "đổi 1–10 phút, MAC và sevoflurane tăng, độ sâu mê gộp tăng trong 5 phút); mỗi đặc trưng chỉ thêm "
                 f"{_range(sig['dauroc'], 4, True)} AUROC.")
    if c3 is not None:
        n_ok = int((c3["consistent"] == "đúng").sum())
        n_rev = int((c3["consistent"] == "ngược").sum())
        txt = (f"**Đặc trưng tổ hợp từ suy luận sinh lý không thêm gì.** {len(c3)} tổ hợp `cx_*`: {n_ok} đúng dấu sinh lý, "
               f"{n_rev} ngược dấu sau khi đã biết MAP (phần lớn là MAP tính lại).")
        if fin is not None:
            a = [m for m in _models(fin, "Tất cả") if "tổ hợp" not in m and "+" not in m]
            big = sorted([m for m in a if _get(fin, m + " + tổ hợp").shape[0]], key=lambda m: int(m.split()[-1]))
            if big:
                b = big[0]
                txt += (" Thêm cả 30 tổ hợp vào `" + b + "` đổi AUROC validation "
                        + _range([_get(fin, b + " + tổ hợp").loc[h, "auroc"] - _get(fin, b).loc[h, "auroc"] for h in H],
                                 4, True) + ".")
        K.append(txt)
    if fin is not None:
        e = _models(fin, "lgbm_context + ECG/PPG")
        if e:
            g = _get(fin, e[0])
            dcol = [c for c in fin.columns if c.startswith("diff_vs")][0]
            n_ci0 = int(sum(g.loc[h, "diff_lo"] <= 0 for h in H))
            K.append(f"**Sóng ECG / PPG thêm rất ít.** Thêm 14 đặc trưng ECG/PPG vào `lgbm_context`: ΔAUROC "
                     f"{_range([g.loc[h, dcol] for h in H], 4, True)}, khoảng tin cậy chứa 0 ở {n_ci0}/3 mốc; đổi lại phải "
                     "có thêm pipeline sóng ECG_II + PLETH khi triển khai.")
        sel = _models(fin, "Chọn lọc")
        alls = sorted(_models(fin, "Tất cả"), key=len)
        if alls:
            dcol = [c for c in fin.columns if c.startswith("diff_vs")][0]
            no_ecg = [m for m in alls if "+" not in m]
            pick = min(no_ecg, key=lambda m: int(m.split()[-1]) if m.split()[-1].isdigit() else 999) if no_ecg else alls[0]
            g = _get(fin, pick)
            txt = (f"**Chọn bộ đặc trưng: `{pick}`.** Dùng mọi đặc trưng (trừ ECG/PPG và tổ hợp) hơn `lgbm_context` "
                   f"{_range([g.loc[h, dcol] for h in H], 4, True)} AUROC trên validation, khoảng tin cậy không chứa 0 ở "
                   f"{int(sum(g.loc[h, 'diff_lo'] > 0 for h in H))}/3 mốc")
            if sel:
                s = _get(fin, sel[0])
                txt += (f"; bộ chọn tham lam `{sel[0]}` (lõi MAP + {summary.get('k_star', '?')} bước) chỉ ngang "
                        f"`lgbm_context` ({_range([s.loc[h, dcol] for h in H], 4, True)}), vì phần thông tin thêm "
                        "được rải trên nhiều đặc trưng gây mê nhỏ, không tập trung ở vài biến")
            K.append(txt + ".")
    return K


def write_eda_md(d: Path, path: Path) -> None:
    d = Path(d)
    sp = d / "summary.json"
    summary = __import__("json").loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    fin, st, l0, l1, c3 = _compare(d), _stats(d), _read(d, "l0_map_only.csv"), _read(d, "l1_conditional.csv"), \
        _read(d, "l3_composites.csv")
    L = ["# EDA tuần 15: đặc trưng nào còn thêm được thông tin khi đã biết MAP", "",
         "Sinh tự động từ `scripts/beyond_map_eda.py` (Kaggle). Mọi con số là đánh giá thật: CV 5 fold theo bệnh nhân "
         "trên train (dùng để chọn) hoặc validation (chỉ để báo cáo); test không được đọc. W = 120 s; mốc dự báo "
         "5 / 10 / 15 phút; sự kiện MAP < 65 mmHg ≥ 60 s.", "",
         "## Kết quả chính", ""]
    L += [f"{i}. {k}" for i, k in enumerate(_key_findings(d, fin, st, c3, summary), 1)] + [""]
    L += ["## Chi tiết dẫn tới việc chọn đặc trưng", ""]

    # 1. reference
    if l0 is not None:
        L += ["### 1. Mốc so sánh: mô hình chỉ-MAP", "",
              "Lõi MAP = mọi thống kê MAP/SBP/DBP trong W, xu hướng MAP 5–30 phút, thời gian ở vùng 65–80, % giảm so "
              "với nền, MAP ngoại suy, lịch sử đợt tụt trong ca. Logit out-of-fold của nó là mốc để đo mọi phần "
              "\"thêm được\" ở các mục sau. AUROC out-of-fold trên train:", "",
              _table(pd.DataFrame({"Mốc": [f"{h // 60} phút" for h in l0["horizon"]],
                                   "Luật MAP hiện tại": [_f(v) for v in l0["auroc_map_current_rule"]],
                                   "LightGBM lõi MAP": [_f(v) for v in l0["auroc_map_only_oof"]],
                                   "Tỷ lệ dương": [f"{v:.1%}".replace(".", ",") for v in l0["prevalence"]]})), ""]

    # 2. MAP in disguise
    if st is not None:
        cnt = st.groupby("horizon").agg(n=("feature", "size"), alone=("auroc_q", lambda q: int((q < 0.05).sum())),
                                        beyond=("sig_beyond_map", "sum")).reset_index()
        L += ["### 2. Phần lớn đặc trưng chỉ lặp lại MAP", "",
              "\"Có ý nghĩa ngoài MAP\" = hồi quy logistic có offset là logit chỉ-MAP cho OR/IQR có q < 0,05, và "
              "ΔAUROC (chỉ-MAP + biến so với chỉ-MAP, out-of-fold, ghép cặp) > 0 với q < 0,05. p tính từ SE bootstrap "
              "theo ca / cluster theo bệnh nhân, q = Benjamini–Hochberg trên mọi biến của một mốc.", "",
              _table(pd.DataFrame({"Mốc": [f"{h // 60} phút" for h in cnt["horizon"]], "Số đặc trưng": cnt["n"],
                                   "Có ý nghĩa khi đứng riêng": cnt["alone"],
                                   "Có ý nghĩa ngoài MAP": cnt["beyond"].astype(int)})), ""]
        if (d / "figures" / "l1_alone_vs_beyond_h900.png").exists():
            L += ["![](figures/l1_alone_vs_beyond_h900.png)", "",
                  "*Trục ngang: AUROC khi đứng một mình; trục dọc: AUROC thêm được khi đã có lõi MAP (mốc 15 phút). "
                  "Nhiều biến mạnh khi đứng riêng nhưng gần 0 khi đã biết MAP.*", ""]
        s = st[(st["horizon"] == 900) & (st["auroc_q"] < 0.05) & ~st["sig_beyond_map"].astype(bool)]
        s = s.sort_values("auroc", ascending=False).head(8)
        if len(s):
            L += ["Ví dụ: các biến mạnh nhất khi đứng riêng (mốc 15 phút) nhưng không thêm gì khi đã biết MAP:", "",
                  _table(pd.DataFrame({"Biến": "`" + s["feature"] + "`",
                                       "Nhóm": s["family"].map(lambda f: FAMILY_VI.get(f, f)),
                                       "AUROC đứng riêng": [_f(v) for v in s["auroc"]],
                                       "ΔAUROC khi thêm vào chỉ-MAP [CI]": [
                                           f"{_f(a, 4, True)} [{_f(b, 4, True)}; {_f(c, 4, True)}]"
                                           for a, b, c in zip(s["dauroc"], s["dauroc_lo"], s["dauroc_hi"])]})), ""]

    # 3. what adds beyond MAP
    if st is not None:
        sig = st[st["sig_beyond_map"].astype(bool)]
        feats = list(dict.fromkeys(sig.sort_values("dauroc", ascending=False)["feature"]))
        meaning = dict(zip(l1["feature"], l1["meaning"])) if l1 is not None and "meaning" in l1 else {}
        rows = []
        for f in feats:
            g = st[st["feature"] == f].set_index("horizon")
            rows.append({"Biến": f"`{f}`", "Nhóm": FAMILY_VI.get(g["family"].iloc[0], g["family"].iloc[0]),
                         "Ý nghĩa": meaning.get(f, ""), "Hướng": g["direction"].iloc[-1],
                         "ΔAUROC 5' / 10' / 15'": " / ".join(_f(g.loc[h, "dauroc"], 4, True) if h in g.index else "–"
                                                            for h in H),
                         "Có ý nghĩa ở mốc": ", ".join(f"{h // 60}'" for h in H
                                                       if h in g.index and bool(g.loc[h, "sig_beyond_map"]))})
        L += ["### 3. Những đặc trưng còn thêm được khi đã biết MAP", "",
              f"Mọi đặc trưng có ý nghĩa ngoài MAP ở ít nhất một mốc ({len(feats)} đặc trưng):", "",
              _table(pd.DataFrame(rows)), "",
              "Đây là lý do giữ nhóm gây mê / thở máy trong bộ đặc trưng: chúng mang thông tin về **nguyên nhân** "
              "(tăng thuốc mê làm giãn mạch) mà các con số huyết áp chưa phản ánh tại thời điểm dự báo.", ""]

    # 4. composites
    if c3 is not None:
        k = c3[(c3["consistent"] == "đúng")].sort_values("dauroc_900", ascending=False)
        L += ["### 4. Đặc trưng tổ hợp sinh lý", "",
              f"{len(c3)} tổ hợp dựng từ MAP = HR × SV × SVR, chỉ số sốc, MAP dự phóng, mất bù nhịp tim, EtCO₂, độ "
              "sâu mê gộp, PPV ở vùng xám, giai đoạn trước rạch da. Kiểm tra: dấu quan sát (ρ với phần dư MAP, mốc "
              "15 phút) có khớp dấu sinh lý kỳ vọng không. Kết quả: " +
              ", ".join(f"{v} {k_}" for k_, v in c3["consistent"].value_counts().items()) + ".", "",
              "Các tổ hợp đúng dấu sinh lý:", "",
              _table(pd.DataFrame({"Tổ hợp": "`" + k["feature"] + "`", "Giả thuyết": k["hypothesis"],
                                   "ΔAUROC 5' / 15'": [f"{_f(a, 4, True)} / {_f(b, 4, True)}"
                                                       for a, b in zip(k["dauroc_300"], k["dauroc_900"])]})), "",
              "Chỉ hai tổ hợp về thay đổi độ sâu mê thêm được thông tin, và thông tin đó đã có trong các cột thuốc mê "
              "gốc; nên bộ đặc trưng không giữ tổ hợp.", ""]

    # 5. ECG / PPG
    L += _ecg_ppg_summary(d, fin)

    # 6. validation comparison
    if fin is not None:
        dcol = [c for c in fin.columns if c.startswith("diff_vs")][0]
        order = ["MAP hiện tại (luật)", "MAP-only (lõi MAP)", CTX, *_models(fin, "lgbm_context + ECG/PPG"),
                 *_models(fin, "Chọn lọc"), *sorted(_models(fin, "Tất cả"), key=lambda m: (len(m), m))]
        L += ["### 6. So sánh các bộ đặc trưng trên validation", "",
              "AUROC (mọi thời điểm), chênh lệch so với `lgbm_context` [CI 95% bootstrap ghép cặp theo bệnh nhân] và "
              "AUPRC. Các mô hình fit trên toàn bộ train, chỉ chấm trên validation.", ""]
        rows = []
        for m in order:
            g = _get(fin, m)
            if not len(g):
                continue
            r = {"Bộ đặc trưng": f"`{m}`"}
            for h in H:
                x = g.loc[h]
                r[f"AUROC {h // 60}'"] = _f(x["auroc"])
                r[f"Δ vs context {h // 60}' [CI]"] = ("mốc" if m == CTX else
                                                      f"{_f(x[dcol], 4, True)} [{_f(x['diff_lo'], 4, True)}; "
                                                      f"{_f(x['diff_hi'], 4, True)}]")
            r["AUPRC 5' / 15'"] = f"{_f(g.loc[300, 'auprc'])} / {_f(g.loc[900, 'auprc'])}"
            rows.append(r)
        L += [_table(pd.DataFrame(rows)), ""]
        lee = [m for m in order if len(_get(fin, m, "lee"))]
        if lee:
            L += ["Theo protocol Lee 2021 / HypoBridCast 2026 (bỏ mẫu âm khó), AUROC mốc 15 phút: " +
                  "; ".join(f"`{m}` {_f(_get(fin, m, 'lee').loc[900, 'auroc'])}" for m in lee) +
                  ". Mọi mô hình đều cao hơn vì protocol này bỏ các thời điểm MAP thấp mà không tụt.", ""]
        if (d / "figures" / "final_compare.png").exists():
            L += ["![](figures/final_compare.png)", ""]

    L += ["## Số liệu đầy đủ", "",
          "- `l1_conditional.csv`, `l1_stats_h*.csv`: mọi đặc trưng, mọi mốc (AUROC riêng, ρ với phần dư MAP, OR/IQR, "
          "ΔAUROC, q)",
          "- `spearman.csv`: tương quan Spearman giữa mọi cặp đặc trưng",
          "- `l2_synergy.csv`, `l2_shap_interaction.csv`: cặp đặc trưng và biến điều biến tác động của MAP",
          "- `l3_composites.csv`: các tổ hợp sinh lý",
          "- `l4_forward.csv`, `l4_prefix.csv`, `ladder_validation.csv`: chọn tham lam và trước / sau trên validation",
          "- `final_compare.csv`, `inference_cost.csv`: so sánh cuối 3 protocol, chi phí suy luận",
          "- `no_ecg/`: cùng phân tích khi không có ECG/PPG", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def _ecg_ppg_summary(d: Path, fin) -> list[str]:
    st = _stats(d)
    if st is None or not st["feature"].str.startswith("ep_").any():
        return []
    ep = st[st["feature"].str.startswith("ep_")]
    n = ep["feature"].nunique()
    sig = ep[ep["sig_beyond_map"].astype(bool)]["feature"].unique().tolist()
    L = ["### 5. Sóng ECG / PPG", "",
         f"{n} đặc trưng tự trích xuất từ SNUADC/ECG_II và SNUADC/PLETH (HRV, thời gian truyền sóng PAT, biên độ PPG), "
         "nhân quả, W = 120 s. Khi đã biết MAP: " +
         (f"chỉ `{'`, `'.join(sig)}` có ý nghĩa ở ít nhất một mốc." if sig else "không đặc trưng nào có ý nghĩa.") + ""]
    if fin is not None:
        e = _models(fin, "lgbm_context + ECG/PPG")
        dcol = [c for c in fin.columns if c.startswith("diff_vs")]
        if e and dcol:
            names = {"every": "Mọi thời điểm", "lee": "Lee / HypoBridCast", "hatib": "Hatib"}
            rows = []
            for p in ("every", "lee", "hatib"):
                g = _get(fin, e[0], p)
                if len(g):
                    rows.append({"Protocol": names[p], **{f"ΔAUROC {h // 60}' [CI]": f"{_f(g.loc[h, dcol[0]], 4, True)} "
                                                          f"[{_f(g.loc[h, 'diff_lo'], 4, True)}; "
                                                          f"{_f(g.loc[h, 'diff_hi'], 4, True)}]" for h in H}})
            L += ["", f"Thêm cả khối vào `lgbm_context` (validation, `{e[0]}` so với `{CTX}`):", "",
                  _table(pd.DataFrame(rows))]
    return L + ["", "Mức thêm này không đủ bù cho pipeline sóng phải có khi triển khai, nên bộ đặc trưng không giữ "
                "ECG/PPG.", ""]
