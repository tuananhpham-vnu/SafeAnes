"""Week 16: relations between the 349 cleaned inputs of lgbm_full_v4 (features_v8), which ones duplicate each other,
and whether dropping the duplicates loses anything -> <out>/{README.md, *.csv, figures/, summary.json}.

  python scripts/redundancy_v8.py --samples <dir with features_v7/ + features_v8/> --labels <lab_v8> \
      --events <lab_v8>/events.parquet

Heavy (LightGBM CV + bootstrap): run it on Kaggle, never on the laptop. features_v8 has no train split, so every
statistic and every choice uses calibration; validation is report only; test is never read.

  features.csv        per input: family, meaning, missing, distinct values, share of the most frequent value,
                      univariate AUROC (and within current-MAP bins), strongest partner |rho|, multiple R^2 from all
                      other inputs, CV |SHAP| share in the 349 model
  spearman.csv        349 x 349 pairwise-complete Spearman; n_both.csv rows where both are observed
  pairs_080.csv       every pair with |rho| >= 0.8
  family_block.csv    mean |rho| between / within families
  near_dup_<T>.csv    near-duplicate groups (every pair |rho| >= T) and the member kept, for each threshold T
  cv.csv              5-fold subject CV on calibration: 349 vs 349 without the duplicates of each threshold
  validation.csv      validation (report only), every-time-point / Hatib / Lee protocols, subject-bootstrap CI
  drop_list.csv / kept_columns.json   the recommended pruning (largest threshold step that passes the CV rule)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation_214 as a2  # noqa: E402  (validate(), table helpers)
from uc04 import ablation as ab  # noqa: E402
from uc04 import beyond_map as bm  # noqa: E402
from uc04 import feature_eda as fe  # noqa: E402
from uc04 import redundancy as rd  # noqa: E402
from uc04.columns import CONTEXT_COLS, numeric_cols  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda/redundancy_v8"
SEED = 20261021  # = scripts/ablation_214.py: same subject folds
H = fe.HORIZONS
_f, _table = a2._f, a2._table


def _n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def log(msg: str, t0: float = time.time()) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def tag(T: float) -> str:
    return f"{int(round(T * 100)):03d}"


def pruned_name(T: float, n: int) -> str:
    return f"Bỏ trùng |ρ| ≥ {_f(T, 2)} ({n})"


# ------------------------------------------------------------------------------------------------ figures

def figures(rho: pd.DataFrame, feats: pd.DataFrame, block: pd.DataFrame, cv: pd.DataFrame | None, fig: Path):
    plt = a2.plt_setup()
    colors = {"map": "#2a6fdb", "numeric": "#5d6d7e", "context": "#16a085", "anaes": "#e67e22", "vent": "#8e44ad",
              "hemo": "#c0392b", "lab": "#b7950b", "clinical": "#7f8c8d", "wave": "#1abc9c"}
    o = rd.order(rho)
    a = rho.loc[o, o].abs().to_numpy()
    f, ax = plt.subplots(figsize=(8.6, 8))
    im = ax.imshow(a, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    fam = [ab.family(c) for c in o]
    for i, g in enumerate(fam):
        ax.add_patch(plt.Rectangle((-8, i - 0.5), 6, 1, color=colors.get(g, "#999"), clip_on=False, lw=0))
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlim(-8.5, len(o) - 0.5)
    ax.set_title(f"|ρ| Spearman giữa {len(o)} đặc trưng (thứ tự theo cụm; dải màu = nhóm)", fontsize=9)
    f.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in colors if k in set(fam)]
    ax.legend(handles, [ab.FAMILY_VI[k] for k in colors if k in set(fam)], fontsize=6, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4)
    f.tight_layout()
    f.savefig(fig / "heatmap.png")
    plt.close(f)

    f, ax = plt.subplots(figsize=(6.2, 5))
    lab = [ab.FAMILY_VI[k] for k in block.index]
    ax.imshow(block.to_numpy(float), cmap="viridis", vmin=0, vmax=max(0.5, float(np.nanmax(block.to_numpy()))))
    for i in range(len(block)):
        for j in range(len(block)):
            v = block.iloc[i, j]
            ax.text(j, i, "" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v < 0.35 else "black")
    ax.set_xticks(range(len(lab)), lab, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(lab)), lab, fontsize=7)
    ax.set_title("|ρ| trung bình giữa / trong các nhóm", fontsize=9)
    f.tight_layout()
    f.savefig(fig / "family_block.png")
    plt.close(f)

    f, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].hist(feats["max_abs_rho"].dropna(), bins=np.linspace(0, 1, 41), color="#5d6d7e")
    for t in (0.8, 0.9, 0.95):
        axes[0].axvline(t, color="black", ls=":", lw=0.8)
    axes[0].set_xlabel("|ρ| lớn nhất với một đặc trưng khác")
    axes[0].set_ylabel("Số đặc trưng")
    axes[1].hist(feats["multiple_r2"].dropna(), bins=np.linspace(0, 1, 41), color="#e67e22")
    axes[1].set_xlabel("R² khi hồi quy từ mọi đặc trưng còn lại (trên hạng)")
    f.tight_layout()
    f.savefig(fig / "max_rho_r2.png")
    plt.close(f)

    if cv is not None and len(cv) > 1:
        d = cv.iloc[1:]
        f, ax = plt.subplots(figsize=(6, 0.4 * len(d) + 1.4))
        ax.barh(d["config"], d["d_auroc"] * 100, color="#2a6fdb", label="ΔAUROC")
        ax.barh(d["config"], d["d_auprc"] * 100, color="#e67e22", alpha=0.5, height=0.4, label="ΔAUPRC")
        ax.axvline(0, color="grey", lw=0.7)
        ax.set_xlabel("Δ × 100 so với bộ 349 (TB 3 mốc, CV trên calibration)")
        ax.legend(frameon=False, fontsize=7)
        f.tight_layout()
        f.savefig(fig / "cv.png")
        plt.close(f)


# ------------------------------------------------------------------------------------------------ report

def write_readme(out: Path, s: dict, feats: pd.DataFrame, groups: dict[float, pd.DataFrame],
                 pairs: pd.DataFrame, cv: pd.DataFrame | None, val: pd.DataFrame | None) -> None:
    n = s["n_inputs"]
    L = [f"# Quan hệ giữa {n} đặc trưng của `lgbm_full_v4` (`features_v8`)", "",
         "Sinh tự động từ `scripts/redundancy_v8.py` (Kaggle). W = 120 s, mốc 5 / 10 / 15 phút. Mọi thống kê và mọi "
         f"lựa chọn dùng **calibration** ({_n(s['rows'])} dòng, {s['cases']} ca; `features_v8` không có train); "
         "validation chỉ để báo cáo; test không đọc. Tương quan là Spearman theo cặp, chỉ trên các dòng cả hai cùng "
         f"có giá trị (bỏ cặp có < {s['min_periods']} dòng chung).", ""]

    L += ["## 1. Tổng quan", "",
          f"- Số cặp có |ρ| ≥ 0,95: **{s['n_pairs']['0.95']}**; ≥ 0,90: **{s['n_pairs']['0.9']}**; ≥ 0,80: "
          f"**{s['n_pairs']['0.8']}** (trên {_n(n * (n - 1) // 2)} cặp).",
          f"- Số đặc trưng có ít nhất một bạn |ρ| ≥ 0,95: **{s['n_with_partner']['0.95']}**; ≥ 0,90: "
          f"{s['n_with_partner']['0.9']}; ≥ 0,80: {s['n_with_partner']['0.8']}.",
          f"- R² bội ≥ 0,99 (gần như là tổ hợp tuyến tính của các biến khác, trên hạng): **{s['n_r2_099']}** biến; "
          f"trong đó {s['n_r2_099_no_pair']} biến không có cặp nào ≥ 0,95 (trùng theo tổ hợp, không theo cặp).",
          f"- Gần hằng (giá trị phổ biến nhất ≥ 99 % số dòng có giá trị): {s['n_quasi_constant']} biến; "
          f"thiếu ≥ 90 %: {s['n_missing_90']} biến.", "",
          "![](figures/heatmap.png)", "", "![](figures/family_block.png)", "", "![](figures/max_rho_r2.png)", ""]

    fp = s["family_pairs_095"]
    if fp:
        L += ["Số cặp |ρ| ≥ 0,95 theo cặp nhóm:", "",
              _table(pd.DataFrame([{"Nhóm A": ab.FAMILY_VI[a], "Nhóm B": ab.FAMILY_VI[b], "Số cặp": k}
                                   for (a, b), k in ((tuple(x.split("|")), v) for x, v in fp.items())])), ""]

    top = pairs.head(40)
    L += ["## 2. Các cặp tương quan mạnh nhất", "",
          f"40 cặp đầu (đủ danh sách |ρ| ≥ 0,8 trong `pairs_080.csv`, {len(pairs)} cặp).", "",
          _table(pd.DataFrame({"A": "`" + top["feature_a"] + "`", "B": "`" + top["feature_b"] + "`",
                               "ρ": top["rho"].map(lambda x: _f(x, 3, True)),
                               "Dòng chung": top["n_both"].map(_n)})), ""]

    qc = feats[feats["quasi_constant"] | (feats["missing"] >= 0.9)]
    if len(qc):
        L += ["## 3. Đặc trưng gần hằng hoặc thiếu nhiều", "",
              _table(pd.DataFrame({"Biến": "`" + qc["feature"] + "`", "Nhóm": qc["family"].map(ab.FAMILY_VI),
                                   "Thiếu": qc["missing"].map(lambda x: _f(x, 3)),
                                   "Số giá trị khác nhau": qc["n_unique"],
                                   "Tỷ lệ giá trị phổ biến nhất": qc["top_share"].map(lambda x: _f(x, 3)),
                                   "AUROC theo MAP (TB)": qc["auroc_given_map"].map(lambda x: _f(x, 3))})), ""]

    L += ["## 4. Nhóm gần trùng và biến được giữ", "",
          "Nhóm: mọi cặp trong nhóm có |ρ| ≥ T (liên kết đầy đủ). Mỗi nhóm giữ một biến theo thứ tự ưu tiên: thuộc "
          "lõi MAP > |SHAP| trong mô hình 349 (CV) > AUROC trong cùng mức MAP > ít thiếu hơn. Một biến khác trong nhóm "
          "cũng được giữ khi nó có giá trị trên ≥ 5 % số dòng mà biến giữ bị thiếu (ρ theo cặp bỏ qua chỗ thiếu).", "",
          _table(pd.DataFrame([{"T": _f(T, 2), "Số nhóm": int(g["group_id"].nunique()) if len(g) else 0,
                                "Biến trong nhóm": len(g), "Bỏ": int((~g["keep"]).sum()),
                                "Giữ thêm vì phủ dòng thiếu": int((g["role"] == "phủ thêm dòng").sum()),
                                "Còn lại": n - int((~g["keep"]).sum())} for T, g in groups.items()])), ""]
    T0 = s["show_groups_T"]
    g = groups[T0]
    if len(g):
        rows = []
        for gid, x in g.groupby("group_id", sort=True):
            k = x[x["role"] == "giữ"].iloc[0]
            rows.append({"Nhóm": gid, "Giữ": f"`{k['feature']}`" + "".join(
                f", `{c}`" for c in x.loc[x["role"] == "phủ thêm dòng", "feature"]),
                "Bỏ": ", ".join(f"`{c}`" for c in x.loc[~x["keep"], "feature"]) or "–",
                "|ρ| nhỏ nhất": _f(x["min_abs_rho_in_group"].iloc[0], 3)})
        L += [f"Nhóm ở T = {_f(T0, 2)} (đủ trong `near_dup_{tag(T0)}.csv`):", "", _table(pd.DataFrame(rows)), ""]

    if cv is not None:
        t = cv.assign(**{"Bộ": cv["config"], "Số biến": cv["n"], "AUROC": cv["auroc"].map(_f),
                         "AUPRC": cv["auprc"].map(_f), "ΔAUROC": cv["d_auroc"].map(lambda x: _f(x, sign=True)),
                         "ΔAUROC 5' / 10' / 15'": [" / ".join(_f(r[f"d_auroc_{h}"], sign=True) for h in H)
                                                    for _, r in cv.iterrows()],
                         "ΔAUPRC": cv["d_auprc"].map(lambda x: _f(x, sign=True)),
                         "Fold xấu hơn": cv["d_auroc_neg_folds"].astype(str) + "/" + str(s["holdouts"])})
        L += ["## 5. Bỏ biến trùng có mất thông tin không (CV trên calibration)", "",
              f"CV {s['holdouts']} fold theo bệnh nhân, cùng fold với ablation; LightGBM tham số `configs/uc04_v3.json`."
              f" Quy tắc chấp nhận: ΔAUROC ≥ −{_f(s['tol_auroc'])} và ΔAUPRC ≥ −{_f(s['tol_auprc'])} (TB 3 mốc).", "",
              _table(t[["Bộ", "Số biến", "AUROC", "AUPRC", "ΔAUROC", "ΔAUROC 5' / 10' / 15'", "ΔAUPRC",
                        "Fold xấu hơn"]]), "", "![](figures/cv.png)", ""]
    if val is not None:
        L += ["## 6. Validation (chỉ báo cáo)", "", "Khác biệt so với bộ 349 là bootstrap theo bệnh nhân, ghép cặp.", ""]
        for proto in ("every", "lee", "hatib"):
            vp = val[val["protocol"] == proto]
            rows = []
            for m in dict.fromkeys(vp["model"]):
                r = {"Mô hình": m, "Số biến": int(vp.loc[vp["model"] == m, "n_inputs"].iloc[0])}
                for h in H:
                    x = vp[(vp["model"] == m) & (vp["horizon"] == h)].iloc[0]
                    r[f"AUROC {h // 60}'"] = _f(x["auroc"], 3)
                    r[f"Δ {h // 60}' [CI]"] = (f"{_f(x['diff_vs_' + a2.FULL], 4, True)} "
                                              f"[{_f(x['diff_lo'], 4, True)}; {_f(x['diff_hi'], 4, True)}]")
                    r[f"AUPRC {h // 60}'"] = _f(x["auprc"], 3)
                rows.append(r)
            L += [f"Protocol `{proto}`:", "", _table(pd.DataFrame(rows)), ""]

    rec = s.get("recommended")
    L += ["## 7. Đề xuất", ""]
    if rec:
        L += [f"Ngưỡng lớn nhất bước qua quy tắc CV: **|ρ| ≥ {_f(rec['T'], 2)}**, bỏ **{rec['n_dropped']}** biến, còn "
              f"**{rec['n_kept']}** (CV ΔAUROC {_f(rec['d_auroc'], sign=True)}, ΔAUPRC {_f(rec['d_auprc'], sign=True)})."
              " Danh sách bỏ: `drop_list.csv`; bộ còn lại: `kept_columns.json`.", ""]
    else:
        L += ["Không ngưỡng nào qua quy tắc CV; giữ nguyên 349.", ""]
    (out / "README.md").write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True, help="dir holding features_v7/ and features_v8/")
    ap.add_argument("--labels", type=Path, required=True, help="dir with <split>.parquet main-cohort labels")
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--W", type=int, default=120)
    ap.add_argument("--train-cases", type=int, default=0, help="smoke: first N calibration cases (0 = all)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--holdouts", type=int, default=5)
    ap.add_argument("--shap-rows", type=int, default=20000)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--min-periods", type=int, default=500)
    ap.add_argument("--thresholds", default="0.98,0.95,0.9,0.85,0.8")
    ap.add_argument("--tol-auroc", type=float, default=0.0005)
    ap.add_argument("--tol-auprc", type=float, default=0.002)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()
    out, W = args.out, args.W
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    Ts = sorted({float(t) for t in args.thresholds.split(",")}, reverse=True)

    cols, fams = ab.v8_columns(args.samples)
    lab, X = ab.load_v8(args.samples, args.labels, "calibration", cols)
    if args.train_cases:
        keep = lab["caseid"].isin(sorted(lab["caseid"].unique())[:args.train_cases]).to_numpy()
        lab, X = lab[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
    X = X[cols]
    a2.FULL = f"Tất cả {len(cols)}"
    s = {"W": W, "n_inputs": len(cols), "families": {f: len(c) for f, c in fams.items()}, "rows": len(lab),
         "cases": int(lab["caseid"].nunique()), "min_periods": args.min_periods, "thresholds": Ts,
         "tol_auroc": args.tol_auroc, "tol_auprc": args.tol_auprc}
    log(f"calibration rows {len(lab):,} ({s['cases']} cases); inputs {len(cols)}")

    # ---------------------------------------------------------------- per-input description
    vals = X.to_numpy(np.float32)
    obs = ~np.isnan(vals)
    feats = pd.DataFrame({"feature": cols, "family": [ab.family(c) for c in cols],
                          "meaning": [fe.describe(c)[1] for c in cols], "missing": 1 - obs.mean(0)})
    nu, top = [], []
    for j, c in enumerate(cols):
        v = vals[obs[:, j], j]
        if len(v):
            u, k = np.unique(v, return_counts=True)
            nu.append(len(u)), top.append(k.max() / len(v))
        else:
            nu.append(0), top.append(np.nan)
    feats["n_unique"], feats["top_share"] = nu, top
    feats["quasi_constant"] = (feats["top_share"] >= 0.99) | (feats["n_unique"] <= 1)
    uni = fe.univariate(lab, X, cols)
    feats["auroc"] = uni[[f"auroc_{h}" for h in H]].mean(axis=1).to_numpy()
    feats["auroc_given_map"] = uni[[f"auroc_given_map_{h}" for h in H]].mean(axis=1).to_numpy()
    log("univariate done")

    # ---------------------------------------------------------------- correlation structure
    rho, n_both = rd.spearman(X, cols, args.min_periods)
    rho.to_csv(out / "spearman.csv", float_format="%.4f")
    n_both.to_csv(out / "n_both.csv")
    a = rho.abs().to_numpy().copy()
    np.fill_diagonal(a, np.nan)
    feats["max_abs_rho"] = np.nanmax(np.where(np.isfinite(a), a, -1), axis=1)
    feats["partner"] = [cols[i] for i in np.nanargmax(np.where(np.isfinite(a), a, -1), axis=1)]
    feats.loc[feats["max_abs_rho"] < 0, ["max_abs_rho", "partner"]] = [np.nan, ""]
    feats["multiple_r2"] = rd.multiple_r2(X, cols).to_numpy()
    log("spearman + multiple R2 done")
    pairs = rd.pairs(rho, n_both, 0.8)
    pairs.to_csv(out / "pairs_080.csv", index=False, float_format="%.4f")
    fam_order = [f for f in ab.FAMILIES if f in fams]
    block = rd.block_mean(rho, ab.family, fam_order)
    block.to_csv(out / "family_block.csv", float_format="%.4f")
    s["n_pairs"] = {str(t): int((pairs["rho"].abs() >= t).sum()) for t in (0.95, 0.9, 0.8)}
    s["n_with_partner"] = {str(t): int((feats["max_abs_rho"] >= t).sum()) for t in (0.95, 0.9, 0.8)}
    s["n_r2_099"] = int((feats["multiple_r2"] >= 0.99).sum())
    s["n_r2_099_no_pair"] = int(((feats["multiple_r2"] >= 0.99) & ~(feats["max_abs_rho"] >= 0.95)).sum())
    s["n_quasi_constant"] = int(feats["quasi_constant"].sum())
    s["n_missing_90"] = int((feats["missing"] >= 0.9).sum())
    p95 = pairs[pairs["rho"].abs() >= 0.95]
    fp = pd.Series(["|".join(sorted((ab.family(x), ab.family(y)), key=fam_order.index))
                    for x, y in zip(p95["feature_a"], p95["feature_b"])], dtype=str).value_counts()
    s["family_pairs_095"] = {k: int(v) for k, v in fp.items()}

    # ---------------------------------------------------------------- CV reference (+ |SHAP| for the keep rule)
    ys = {h: lab[f"y_{h}"].to_numpy() for h in H}
    folds = bm.subject_folds(lab["subjectid"].to_numpy(), args.folds, SEED)
    hold = list(range(min(args.holdouts, args.folds)))
    s["holdouts"] = len(hold)
    cv = lambda fs, shap=0: ab.cv_eval(X, ys, folds, fs, hold, W, SEED, shap)  # noqa: E731
    ref = cv(cols, args.shap_rows)
    feats["shap_share"] = ref["importance"].reindex(cols).to_numpy()
    log(f"CV 349: AUROC {ref['auroc']:.4f} AUPRC {ref['auprc']:.4f}")

    core = set(fams["map"])
    info = feats.set_index("feature")
    priority = lambda c: (c in core, float(info.at[c, "shap_share"]),  # noqa: E731
                          float(np.nan_to_num(info.at[c, "auroc_given_map"])), -float(info.at[c, "missing"]))
    groups, sets = {}, {}
    for T in Ts:
        g = rd.near_dup_groups(rho, X, T, priority)
        g.insert(3, "family", g["feature"].map(ab.family))
        g["shap_share"] = g["feature"].map(info["shap_share"])
        g.to_csv(out / f"near_dup_{tag(T)}.csv", index=False, float_format="%.4f")
        groups[T] = g
        drop = set(g.loc[~g["keep"], "feature"])
        sets[T] = [c for c in cols if c not in drop]
        feats[f"drop_{tag(T)}"] = feats["feature"].isin(drop)
        log(f"T={T}: {g['group_id'].nunique() if len(g) else 0} groups, drop {len(drop)} -> {len(sets[T])}")
    feats.to_csv(out / "features.csv", index=False, float_format="%.5f")

    # ---------------------------------------------------------------- CV of the pruned sets
    rows = [{"config": a2.FULL, "T": np.nan, "n": len(cols), "auroc": ref["auroc"], "auprc": ref["auprc"],
             **{f"auroc_{h}": ref[f"auroc_{h}"] for h in H}, **ab.paired_diff(ref, ref, H)}]
    done = {}
    for T in Ts:
        key = tuple(sets[T])
        if key not in done:
            done[key] = cv(sets[T]) if len(sets[T]) < len(cols) else ref
        r = done[key]
        rows.append({"config": pruned_name(T, len(sets[T])), "T": T, "n": len(sets[T]), "auroc": r["auroc"],
                     "auprc": r["auprc"], **{f"auroc_{h}": r[f"auroc_{h}"] for h in H}, **ab.paired_diff(r, ref, H)})
        log(f"CV T={T} ({len(sets[T])}): AUROC {r['auroc']:.4f} (d {rows[-1]['d_auroc']:+.4f}, "
            f"dAUPRC {rows[-1]['d_auprc']:+.4f})")
        pd.DataFrame(rows).to_csv(out / "cv.csv", index=False, float_format="%.5f")
    cvt = pd.DataFrame(rows)
    ok = cvt[(cvt["T"].notna()) & (cvt["d_auroc"] >= -args.tol_auroc) & (cvt["d_auprc"] >= -args.tol_auprc)]
    if len(ok):
        best = ok.sort_values(["n", "T"]).iloc[0]
        T = float(best["T"])
        s["recommended"] = {"T": T, "n_kept": int(best["n"]), "n_dropped": len(cols) - int(best["n"]),
                            "d_auroc": float(best["d_auroc"]), "d_auprc": float(best["d_auprc"])}
        g = groups[T]
        g[~g["keep"]].to_csv(out / "drop_list.csv", index=False, float_format="%.4f")
        (out / "kept_columns.json").write_text(json.dumps({"T": T, "columns": sets[T]}, indent=1), encoding="utf-8")
    s["show_groups_T"] = s.get("recommended", {}).get("T", 0.95 if 0.95 in groups else Ts[0])
    figures(rho, feats, block, cvt, fig)

    # ---------------------------------------------------------------- validation
    val = None
    if not args.no_validate:
        vlab, VX = ab.load_v8(args.samples, args.labels, "validation", cols, all_main=True)
        vrows = pd.concat([vlab, VX[["map_current"]]], axis=1)
        events = pd.read_parquet(args.events)
        events = events[events["split"] == "validation"]
        ctx = [c for c in numeric_cols(W) + list(CONTEXT_COLS) if c in set(cols)]
        models = {"MAP-only (lõi MAP)": fams["map"], f"lgbm_context ({len(ctx)})": ctx, a2.FULL: cols}
        for T in Ts:
            if len(sets[T]) < len(cols) and tuple(sets[T]) not in map(tuple, models.values()):
                models[pruned_name(T, len(sets[T]))] = sets[T]
        val = a2.validate(models, X, ys, VX, vrows, events, W, args.boot)
        val.to_csv(out / "validation.csv", index=False, float_format="%.4f")
        s["validation_rows"] = len(vrows)
    (out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(out, s, feats, groups, pairs, cvt, val)
    log("done")


if __name__ == "__main__":
    main()
