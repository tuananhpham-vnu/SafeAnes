"""Feature-level EDA on samples v3 -> reports/weekly/week14/eda/{*.csv, figures/*.png, summary.json}.

  $env:PYTHONPATH = "$PWD/src;$PWD/.local_deps"
  python scripts/feature_eda.py [--train-cases 900] [--W 120] [--skip-models]

What it answers (advisor feedback, week 13):
  1. dictionary.csv      every candidate input: group, signal, meaning, unit, missing, quantiles
  2. univariate.csv      AUROC of each input alone for y5/y10/y15, and the same AUROC among rows with the same current
                         MAP (does the input add anything beyond "MAP is already low"?)
     deciles_h900.csv    event risk by decile of the top inputs
  3. clusters.csv        redundancy clusters (|Spearman| >= 0.8); one representative per cluster
  4. importance.csv      LightGBM on every input: mean |TreeSHAP| per feature, AUROC drop when a cluster is permuted
  5. selection.csv       validation AUROC with the k most important clusters (representatives only), k = 1 .. all
     blocks.csv          validation AUROC per input block: monitor / + context / + anaesthesia / + clinical
Statistics that choose things use a train subsample; validation only scores. Test is never read.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uc04 import feature_eda as fe

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data/samples_v3"
MANIFEST = ROOT / "data/vitaldb_full/meta/cohort_manifest.csv"
OUT = ROOT / "reports/weekly/week14/eda"
FIG = OUT / "figures"
SEED = 20260917
GROUP_VI = {"numeric": "Chỉ số monitor", "wave": "Sóng ART", "context": "Ngữ cảnh ca",
            "anaes": "Gây mê / thở máy", "clinical": "Lâm sàng trước mổ"}
GROUP_COLOR = {"numeric": "#2a6fdb", "wave": "#8e44ad", "context": "#16a085", "anaes": "#e67e22", "clinical": "#c0392b"}


def log(msg: str, t0: float = time.time()) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def dictionary(lab, feats, cols):
    pre = feats["f1a_min_since_opstart"].isna().to_numpy()
    rows = []
    for c in cols:
        x = feats[c].to_numpy(np.float64)
        sig, meaning, unit = fe.describe(c)
        q = np.nanpercentile(x, [1, 50, 99]) if np.isfinite(x).any() else [np.nan] * 3
        rows.append({"feature": c, "group": fe.feature_group(c), "signal": sig, "meaning": meaning, "unit": unit,
                     "missing": np.isnan(x).mean(), "missing_pre_incision": np.isnan(x[pre]).mean(),
                     "missing_surgery": np.isnan(x[~pre]).mean(), "p1": q[0], "median": q[1], "p99": q[2]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ figures

def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def fig_univariate(uni: pd.DataFrame, h: int, path: Path, top: int = 25):
    plt = plt_setup()
    d = uni.sort_values(f"auroc_{h}", ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * top + 1))
    y = np.arange(len(d))
    ax.barh(y, d[f"auroc_{h}"] - 0.5, left=0.5, color=[GROUP_COLOR[g] for g in d["group"]], alpha=0.85,
            label="Một mình")
    ax.scatter(d[f"auroc_given_map_{h}"], y, color="black", s=14, zorder=3, label="Khi đã biết MAP hiện tại")
    ax.set_yticks(y, d["feature"])
    ax.set_xlim(0.5, max(0.85, d[f"auroc_{h}"].max() + 0.02))
    ax.set_xlabel(f"AUROC đơn biến, mốc {h // 60} phút (train)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUP_COLOR]
    leg1 = ax.legend(handles, [GROUP_VI[g] for g in GROUP_COLOR], loc="lower right", fontsize=7, frameon=False)
    ax.add_artist(leg1)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_deciles(dec: pd.DataFrame, path: Path, prevalence: float):
    plt = plt_setup()
    feats = dec["feature"].unique()
    n = len(feats)
    cols = 3
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(9, 2.4 * np.ceil(n / cols)))
    for ax, f in zip(np.ravel(axes), feats):
        d = dec[dec["feature"] == f]
        ax.plot(d["x_median"], d["risk"] * 100, marker="o", ms=3, color="#2a6fdb")
        ax.axhline(prevalence * 100, color="grey", lw=0.8, ls="--")
        ax.set_title(f, fontsize=8)
        ax.set_ylabel("% có tụt trong 15 ph", fontsize=7)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_corr(corr: pd.DataFrame, clusters: pd.Series, feats: list[str], path: Path):
    plt = plt_setup()
    order = sorted(feats, key=lambda c: (clusters[c], c))
    c = corr.loc[order, order].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6)
    ax.set_yticks(range(len(order)), order, fontsize=6)
    # outline clusters
    cl = [clusters[o] for o in order]
    start = 0
    for i in range(1, len(cl) + 1):
        if i == len(cl) or cl[i] != cl[start]:
            if i - start > 1:
                ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), i - start, i - start, fill=False, lw=1.2))
            start = i
    fig.colorbar(im, ax=ax, shrink=0.6, label="Spearman ρ")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_group_share(imp: pd.DataFrame, path: Path):
    plt = plt_setup()
    hs = [c for c in imp.columns if c.startswith("shap_")]  # validation SHAP
    share = imp.groupby("group")[hs].sum()
    share = share / share.sum()
    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    left = np.zeros(len(hs))
    for g in fe.GROUP_ORDER:
        if g in share.index:
            v = share.loc[g].to_numpy()
            ax.barh(range(len(hs)), v, left=left, color=GROUP_COLOR[g], label=GROUP_VI[g])
            for i, (l, w) in enumerate(zip(left, v)):
                if w > 0.05:
                    ax.text(l + w / 2, i, f"{w:.0%}", ha="center", va="center", color="white", fontsize=8)
            left += v
    ax.set_yticks(range(len(hs)), [f"{int(h.split('_')[1]) // 60} phút" for h in hs])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Tỷ phần mean |SHAP| (validation)")
    ax.legend(ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.02), loc="lower center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_selection(sel: pd.DataFrame, full: dict, path: Path):
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for h, color in zip(sorted(sel["horizon"].unique()), ("#2a6fdb", "#e67e22", "#16a085")):
        d = sel[sel["horizon"] == h]
        ax.plot(d["k"], d["auroc"], marker="o", ms=3, color=color, label=f"{h // 60} phút")
        ax.axhline(full[h], color=color, ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Số cụm đặc trưng giữ lại (mỗi cụm 1 đại diện)")
    ax.set_ylabel("AUROC validation")
    ax.legend(frameon=False, fontsize=8, title="Mốc (nét đứt: mọi đặc trưng)", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main

def main():
    global SAMPLES, MANIFEST, OUT, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cases", type=int, default=900)
    ap.add_argument("--W", type=int, default=120)
    ap.add_argument("--val-rows", type=int, default=80000, help="validation rows for SHAP and permutation")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--samples", type=Path, default=SAMPLES)
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="VitalDB clinical table (cases API columns)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    SAMPLES, MANIFEST, OUT = args.samples, args.manifest, args.out
    FIG = OUT / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    groups = fe.candidate_columns(SAMPLES, args.W)
    cols = [c for g in fe.GROUP_ORDER for c in groups[g]]
    cases = fe.sample_cases(SAMPLES / "labels/train.parquet", args.train_cases, SEED)
    log(f"{len(cols)} candidate inputs; {len(cases)} train cases")
    lab, X = fe.load_rows(SAMPLES, "train", cols, MANIFEST, cases)
    log(f"train rows {len(lab):,}; {X.memory_usage().sum() / 1e6:.0f} MB")
    summary = {"W": args.W, "train_cases": len(cases), "train_rows": len(lab), "n_inputs": len(cols),
               "inputs_per_group": {g: len(groups[g]) for g in fe.GROUP_ORDER},
               "prevalence_train": {h: float(lab[f"y_{h}"][fe.known(lab, h)].mean()) for h in fe.HORIZONS}}

    # 1. dictionary
    dic = dictionary(lab, X, cols)
    dic.to_csv(OUT / "dictionary.csv", index=False, float_format="%.4g")
    log("dictionary.csv")

    # 2. univariate
    uni = fe.univariate(lab, X, cols)
    uni.to_csv(OUT / "univariate.csv", index=False, float_format="%.4f")
    for h in fe.HORIZONS:
        fig_univariate(uni, h, FIG / f"univariate_h{h}.png")
    log("univariate.csv")

    # 3. redundancy
    corr = fe.spearman(X, cols, 40000, SEED)
    clusters = fe.redundancy_clusters(corr, 0.8)
    cl = pd.DataFrame({"feature": cols, "cluster": clusters.reindex(cols).to_numpy()}).merge(
        uni[["feature", "group", "auroc_900", "auroc_given_map_900"]], on="feature")
    rep = cl.sort_values("auroc_900", ascending=False).groupby("cluster")["feature"].first()
    cl["representative"] = cl["cluster"].map(rep)
    cl["cluster_size"] = cl.groupby("cluster")["feature"].transform("size")
    cl.sort_values(["cluster", "auroc_900"], ascending=[True, False]).to_csv(OUT / "clusters.csv", index=False,
                                                                              float_format="%.4f")
    corr.to_csv(OUT / "spearman.csv", float_format="%.3f")
    top = uni.sort_values("auroc_900", ascending=False)["feature"].head(45).tolist()
    fig_corr(corr, clusters, top, FIG / "spearman_top45.png")
    summary["n_clusters"] = int(cl["cluster"].nunique())
    log(f"{summary['n_clusters']} redundancy clusters")

    # top independent features: best of each cluster, by h=900 AUROC
    top_rep = (cl[cl["feature"] == cl["representative"]].sort_values("auroc_900", ascending=False)["feature"]
               .head(12).tolist())
    dec = fe.risk_by_decile(lab, X, top_rep, 900)
    dec.to_csv(OUT / "deciles_h900.csv", index=False, float_format="%.4f")
    fig_deciles(dec, FIG / "deciles_h900.png", summary["prevalence_train"][900])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.skip_models:
        return

    # 4. importance
    vlab, VX = fe.load_rows(SAMPLES, "validation", cols, MANIFEST)
    log(f"validation rows {len(vlab):,}")
    summary["validation_rows"] = len(vlab)
    rng = np.random.default_rng(SEED)
    imp = dic[["feature", "group"]].copy()
    perm_all, full = [], {}
    cluster_cols = {f"c{k}": g["feature"].tolist() for k, g in cl.groupby("cluster")}
    for h in (300, 900):
        m, vm = fe.known(lab, h), fe.known(vlab, h)
        booster = fe.fit_lgbm(X.loc[m, cols], lab[f"y_{h}"].to_numpy()[m], args.W, SEED)
        Xv, yv = VX.loc[vm, cols], vlab[f"y_{h}"].to_numpy()[vm]
        full[h] = fe.eval_auroc(booster, Xv, yv)
        sub = rng.choice(len(Xv), min(args.val_rows, len(Xv)), replace=False)
        imp[f"shap_{h}"] = fe.shap_by_feature(booster, Xv.iloc[sub]).reindex(imp["feature"]).to_numpy()
        tsub = rng.choice(int(m.sum()), min(args.val_rows, int(m.sum())), replace=False)
        imp[f"train_shap_{h}"] = fe.shap_by_feature(booster, X.loc[m, cols].iloc[tsub]).reindex(imp["feature"]).to_numpy()
        perm, _ = fe.grouped_permutation(booster, Xv.iloc[sub], yv[sub], cluster_cols, SEED)
        perm["horizon"] = h
        perm_all.append(perm)
        log(f"importance h={h}: validation AUROC {full[h]:.4f}")
    imp.to_csv(OUT / "importance.csv", index=False, float_format="%.5f")
    perm = pd.concat(perm_all)
    perm["representative"] = perm["cluster"].str[1:].astype(int).map(rep)
    perm["members"] = perm["cluster"].map(lambda k: " ".join(cluster_cols[k]))
    perm.to_csv(OUT / "permutation.csv", index=False, float_format="%.5f")
    fig_group_share(imp, FIG / "shap_group_share.png")
    summary["auroc_all_inputs"] = full

    # 5. selection: clusters ranked by summed mean |TreeSHAP| on train rows (both horizons), so validation
    #    only scores the curve and never chooses the order
    tr = imp.merge(cl[["feature", "cluster"]], on="feature")
    tr["s"] = tr["train_shap_300"] / tr["train_shap_300"].sum() + tr["train_shap_900"] / tr["train_shap_900"].sum()
    order = [rep[k] for k in tr.groupby("cluster")["s"].sum().sort_values(ascending=False).index]
    ks = [k for k in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 45, len(order)) if k <= len(order)]
    sel = []
    for h in (300, 900):
        m, vm = fe.known(lab, h), fe.known(vlab, h)
        for k in ks:
            fs = order[:k]
            b = fe.fit_lgbm(X.loc[m, fs], lab[f"y_{h}"].to_numpy()[m], args.W, SEED)
            a = fe.eval_auroc(b, VX.loc[vm, fs], vlab[f"y_{h}"].to_numpy()[vm])
            sel.append({"horizon": h, "k": k, "auroc": a, "features": " ".join(fs)})
            log(f"selection h={h} k={k}: {a:.4f}")
    sel = pd.DataFrame(sel)
    sel.to_csv(OUT / "selection.csv", index=False, float_format="%.4f")
    fig_selection(sel, full, FIG / "selection.png")

    # 6. blocks
    blocks = {
        "monitor (66)": groups["numeric"],
        "monitor + ngữ cảnh": groups["numeric"] + groups["context"],
        "monitor + ngữ cảnh + gây mê": groups["numeric"] + groups["context"] + groups["anaes"],
        "monitor + ngữ cảnh + lâm sàng": groups["numeric"] + groups["context"] + groups["clinical"],
        "monitor + ngữ cảnh + gây mê + lâm sàng": groups["numeric"] + groups["context"] + groups["anaes"]
                                                 + groups["clinical"],
        "tất cả (+ sóng)": cols,
    }
    out = []
    for h in (300, 900):
        m, vm = fe.known(lab, h), fe.known(vlab, h)
        for name, fs in blocks.items():
            a = full[h] if name.startswith("tất cả") else fe.eval_auroc(
                fe.fit_lgbm(X.loc[m, fs], lab[f"y_{h}"].to_numpy()[m], args.W, SEED),
                VX.loc[vm, fs], vlab[f"y_{h}"].to_numpy()[vm])
            out.append({"horizon": h, "block": name, "n_features": len(fs), "auroc": a})
            log(f"block h={h} {name}: {a:.4f}")
    pd.DataFrame(out).to_csv(OUT / "blocks.csv", index=False, float_format="%.4f")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log("done")


if __name__ == "__main__":
    main()
