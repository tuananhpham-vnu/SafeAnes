"""Week 15 EDA "beyond MAP" on samples v3, W = 120 -> eda/{README.md, *.csv, figures/, summary.json}.

  $env:PYTHONPATH = "$PWD/src;$PWD/.local_deps"
  python scripts/beyond_map_eda.py [--train-cases 0] [--row-frac 0.4] [--pool 32] [--max-steps 15]

Heavy (hours on CPU): run it on Kaggle, never on the laptop.

  l0_map_only.csv        MAP-only reference (MAP_CORE): out-of-fold AUROC on train per horizon; the offset
  l1_conditional.csv     every non-MAP-core input and composite: univariate AUROC, partial Spearman with the
                         MAP residual (case-bootstrap CI), I(x; y | MAP-offset decile), grey-zone AUROC,
                         incremental dAUROC/dAUPRC/dlogloss of offset + x (5 subject folds)
  l3_composites.csv      physiology composites: hypothesis, expected sign, observed sign, consistency
  l2_synergy.csv         pairs of the pool: synergy = gain(a, b) - gain(a) - gain(b) on the MAP offset
  l2_shap_interaction.csv  mean |SHAP interaction| (MAP core x others and others x others), h = 900
  l4_forward.csv         greedy bundle selection on the offset; l4_prefix.csv full models on its prefixes (CV)
  final_compare.csv      validation: MAP rule, MAP-only, lgbm_context, all 214, all + composites, selected,
                         lgbm_context + selected, under the every-time-point / Hatib / Lee-HypoBridCast protocols,
                         subject-bootstrap CI and paired difference to lgbm_context; gain over MAP-only
Everything that chooses (offset, ranking, pool, selection, stop point) uses train only. Test is never read.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uc04 import beyond_map as bm
from uc04 import ecg_ppg as ep
from uc04 import feature_eda as fe
from uc04 import peer_protocols as pp
from uc04.eda_report import write_eda_md
from uc04.metrics import row_metrics

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data/samples_v3"
MANIFEST = ROOT / "data/vitaldb_full/meta/cohort_manifest.csv"
OUT = ROOT / "eda"
SEED = 20261017
EPS = 1e-9
H = fe.HORIZONS
FAMILY_COLOR = {"ecgppg": "#e74c3c", "map": "#2a6fdb", "numeric": "#5d6d7e", "wave": "#8e44ad", "context": "#16a085",
                "anaes": "#e67e22", "clinical": "#c0392b", "composite": "#d4ac0d"}
FAMILY_VI = {"ecgppg": "Sóng ECG / PPG", "map": "Họ MAP", "numeric": "Monitor khác", "wave": "Sóng ART", "context": "Ngữ cảnh ca",
             "anaes": "Gây mê / thở máy", "clinical": "Lâm sàng trước mổ", "composite": "Tổ hợp sinh lý"}


def log(msg: str, t0: float = time.time()) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def family(c: str) -> str:
    if bm.is_map_core(c):
        return "map"
    if c.startswith("ep_"):
        return "ecgppg"
    return "composite" if c.startswith("cx_") else fe.feature_group(c)


def describe(c: str) -> str:
    if c in ep.DESCRIBE:
        return ep.DESCRIBE[c][0]
    return bm.COMPOSITES[c][0] if c in bm.COMPOSITES else fe.describe(c)[1]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inference_cost(booster, X: pd.DataFrame, rng, n_single: int = 300) -> dict:
    """Real-time cost: median latency of one-row predictions (as at the bedside, every 30 s), batch throughput,
    model size."""
    A = X.to_numpy(np.float32)
    rows = rng.choice(len(A), min(n_single, len(A)), replace=False)
    t = []
    for r in rows:
        t0 = time.perf_counter()
        booster.predict(A[r:r + 1], raw_score=True)
        t.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    booster.predict(A[:20000], raw_score=True)
    batch = time.perf_counter() - t0
    return {"latency_ms_row": 1e3 * float(np.median(t)), "latency_ms_row_p95": 1e3 * float(np.percentile(t, 95)),
            "rows_per_s_batch": min(20000, len(A)) / batch, "n_trees": booster.num_trees(),
            "model_kb": len(booster.model_to_string()) / 1024}


# ------------------------------------------------------------------------------------------------ figures

def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def fig_alone_vs_beyond(l1: pd.DataFrame, h: int, path: Path):
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for fam, d in l1.groupby("family"):
        ax.scatter(d[f"auroc_{h}"], d[f"dauroc_{h}"] * 100, s=14, alpha=0.8, color=FAMILY_COLOR[fam],
                   label=FAMILY_VI[fam])
    top = l1.sort_values(f"dauroc_{h}", ascending=False).head(12)
    for r in top.itertuples():
        ax.annotate(r.feature, (getattr(r, f"auroc_{h}"), getattr(r, f"dauroc_{h}") * 100), fontsize=6,
                    xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_xlabel(f"AUROC khi đứng một mình, mốc {h // 60} phút")
    ax.set_ylabel("ΔAUROC × 100 khi thêm vào mô hình chỉ-MAP")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_synergy(syn: pd.DataFrame, pool: list[str], h: int, path: Path):
    plt = plt_setup()
    m = pd.DataFrame(np.nan, index=pool, columns=pool)
    for r in syn.itertuples():
        v = getattr(r, f"synergy_{h}") * 1e4
        m.loc[r.a, r.b] = m.loc[r.b, r.a] = v
    lim = np.nanpercentile(np.abs(m.to_numpy()), 98) if np.isfinite(m.to_numpy()).any() else 1
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(m.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(pool)), pool, rotation=90, fontsize=6)
    ax.set_yticks(range(len(pool)), pool, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.6, label=f"Synergy (Δlog-loss × 1e4), mốc {h // 60} phút")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_modulation(lab, X, feats: list[str], h: int, path: Path):
    """Event risk against current MAP, one line per tertile of the modulating feature."""
    plt = plt_setup()
    y = lab[f"y_{h}"].to_numpy()
    k = np.isin(y, (0, 1))
    mp = X["map_current"].to_numpy()
    bins = np.arange(60, 106, 5)
    n = len(feats)
    fig, axes = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(10, 2.8 * np.ceil(n / 3)), squeeze=False)
    for ax, f in zip(axes.ravel(), feats):
        x = X[f].to_numpy(np.float64)
        m = k & np.isfinite(x) & np.isfinite(mp)
        q = np.nanpercentile(x[m], [33.3, 66.7])
        for lo, hi, name, color in ((-np.inf, q[0], "thấp", "#2a6fdb"), (q[0], q[1], "giữa", "#7f8c8d"),
                                    (q[1], np.inf, "cao", "#c0392b")):
            mm = m & (x > lo) & (x <= hi)
            b = np.digitize(mp[mm], bins)
            r = pd.Series(y[mm] == 1).groupby(b).agg(["mean", "size"])
            r = r[r["size"] >= 200]
            centers = [bins[i - 1] + 2.5 if 0 < i < len(bins) else np.nan for i in r.index]
            ax.plot(centers, r["mean"] * 100, marker="o", ms=3, color=color, label=f"{name}")
        ax.set_title(f, fontsize=8)
        ax.set_xlabel("MAP hiện tại (mmHg)", fontsize=7)
        ax.set_ylabel(f"% tụt trong {h // 60} ph", fontsize=7)
        ax.legend(fontsize=6, frameon=False, title="tam phân vị", title_fontsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_prefix(pre: pd.DataFrame, path: Path, k_star: int):
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for h, color in zip(H, ("#2a6fdb", "#e67e22", "#16a085")):
        d = pre[pre["horizon"] == h]
        ax.plot(d["k"], d["auroc_cv"], marker="o", ms=3, color=color, label=f"{h // 60} phút")
    ax.axvline(k_star, color="black", ls=":", lw=0.8)
    ax.set_xlabel("Số bundle thêm vào lõi MAP (0 = chỉ MAP)")
    ax.set_ylabel("AUROC CV trên train")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_final(fin: pd.DataFrame, path: Path):
    plt = plt_setup()
    protos = list(dict.fromkeys(fin["protocol"]))
    models = list(dict.fromkeys(fin["model"]))
    fig, axes = plt.subplots(1, len(protos), figsize=(4.2 * len(protos), 3.4), sharey=False, squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, p in zip(axes.ravel(), protos):
        d = fin[fin["protocol"] == p]
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            dm = d[d["model"] == m].sort_values("horizon")
            x = np.arange(len(dm)) + i * w
            ax.bar(x, dm["auroc"], w, color=colors[i], label=m)
            ax.errorbar(x, dm["auroc"], yerr=[(dm["auroc"] - dm["lo"]).clip(lower=0),
                                              (dm["hi"] - dm["auroc"]).clip(lower=0)], fmt="none",
                        ecolor="black", lw=0.6)
        ax.set_xticks(np.arange(len(H)) + 0.4 - w / 2, [f"{h // 60} ph" for h in H])
        ax.set_ylim(0.6, 1.0)
        ax.set_title(p)
    axes.ravel()[0].set_ylabel("AUROC validation")
    axes.ravel()[-1].legend(fontsize=6, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=120)
    ap.add_argument("--train-cases", type=int, default=0, help="0 = every train case")
    ap.add_argument("--row-frac", type=float, default=0.4, help="train rows used by L1, L2, L4 screening")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pool", type=int, default=32, help="features in the L2 pair search")
    ap.add_argument("--bundles", type=int, default=40, help="single-feature bundles in L4")
    ap.add_argument("--pair-bundles", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--shap-rows", type=int, default=2000)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--samples", type=Path, default=SAMPLES)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--resume", type=Path, help="earlier output dir: its finished stages are reused")
    ap.add_argument("--extra", type=Path, help="dir with ecgppg_shard*.parquet (ECG/PPG features) to add")
    args = ap.parse_args()
    out_dir, FIG = args.out, args.out / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    if args.resume and args.resume.resolve() != out_dir.resolve():
        for p in args.resume.glob("*"):
            if p.is_file():
                shutil.copy2(p, out_dir / p.name)
    W, rng = args.W, np.random.default_rng(SEED)
    summary = json.loads((out_dir / "summary.json").read_text()) if (out_dir / "summary.json").exists() else {}
    save = lambda: (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))  # noqa

    # ---------------------------------------------------------------- data
    groups = fe.candidate_columns(args.samples, W)
    cols = [c for g in fe.GROUP_ORDER for c in groups[g]]
    need = [c for c in bm.needed_columns(W) if c not in cols]
    assert not need, f"composites need columns outside the candidates: {need}"
    cases = fe.sample_cases(args.samples / "labels/train.parquet", args.train_cases, SEED) if args.train_cases else None
    lab, X = fe.load_rows(args.samples, "train", cols, args.manifest, cases)
    X = bm.add_composites(X, W)
    ep_extra = bm.load_extra(args.extra)
    X, epc = bm.merge_extra(X, ep_extra)
    cols = cols + epc
    cx = list(bm.COMPOSITES)
    core = bm.map_core(cols)
    cand = [c for c in cols + cx if c not in core]
    ys = {h: lab[f"y_{h}"].to_numpy() for h in H}
    folds = bm.subject_folds(lab["subjectid"].to_numpy(), args.folds, SEED)
    sub = rng.random(len(lab)) < args.row_frac
    Xs, ys_s, folds_s, case_s = X[sub].reset_index(drop=True), {h: y[sub] for h, y in ys.items()}, folds[sub], \
        lab["caseid"].to_numpy()[sub]
    log(f"train rows {len(lab):,} ({lab['caseid'].nunique()} cases); screening rows {sub.sum():,}; "
        f"MAP core {len(core)}; candidates beyond the core {len(cand)}")
    summary.update({"W": W, "train_rows": len(lab), "train_cases": int(lab["caseid"].nunique()),
                    "screen_rows": int(sub.sum()), "map_core": core, "n_candidates": len(cand)})

    # ---------------------------------------------------------------- L0: MAP-only offset
    off_path = out_dir / "l0_offsets.parquet"
    if off_path.exists():
        o = pd.read_parquet(off_path)
        assert len(o) == len(lab) and (o["time"].to_numpy() == lab["time"].to_numpy()).all()
        offsets = {h: o[f"offset_{h}"].to_numpy() for h in H}
        log("L0 reused")
    else:
        params = {k: v for k, v in fe.LGB_PARAMS.items()}
        offsets = {h: bm.map_offset_oof(X[core], ys[h], folds, params, fe.N_TREES, SEED) for h in H}
        pd.DataFrame({"caseid": lab["caseid"], "time": lab["time"],
                      **{f"offset_{h}": offsets[h] for h in H}}).to_parquet(off_path, index=False)
        rows = []
        for h in H:
            k = np.isin(ys[h], (0, 1))
            yy = (ys[h][k] == 1).astype(np.int8)
            rows.append({"horizon": h, "auroc_map_only_oof": fe.auroc(yy, offsets[h][k]),
                         "auroc_map_current_rule": fe.auroc(yy, -X["map_current"].to_numpy()[k]),
                         "prevalence": float(yy.mean())})
            log(f"L0 h={h}: MAP-only OOF AUROC {rows[-1]['auroc_map_only_oof']:.4f}, "
                f"rule {rows[-1]['auroc_map_current_rule']:.4f}")
        pd.DataFrame(rows).to_csv(out_dir / "l0_map_only.csv", index=False, float_format="%.4f")
    off_s = {h: offsets[h][sub] for h in H}

    # ---------------------------------------------------------------- L1: conditional association
    l1_path = out_dir / "l1_conditional.csv"
    single_path = out_dir / "l1_single_folds.json"
    if l1_path.exists() and single_path.exists():
        l1 = pd.read_csv(l1_path)
        single = json.loads(single_path.read_text())
        log("L1 reused")
    else:
        rows, single = [], {str(h): {} for h in H}
        mp = Xs["map_current"].to_numpy()
        for i, c in enumerate(cand):
            x = Xs[c].to_numpy(np.float64)
            rec = {"feature": c, "family": family(c), "meaning": describe(c), "missing": float(np.isnan(x).mean())}
            for h in H:
                y = ys_s[h]
                k = np.isin(y, (0, 1)) & np.isfinite(x)
                yy = (y[k] == 1).astype(np.int8)
                a = fe.auroc(yy, x[k]) if k.sum() > 100 and len(np.unique(x[k])) > 1 else np.nan
                rec[f"auroc_{h}"] = max(a, 1 - a) if np.isfinite(a) else np.nan
                kk = np.isin(y, (0, 1))
                resid = np.where(kk, (y == 1) - sigmoid(off_s[h]), np.nan)
                rho, lo, hi = bm.partial_spearman(x, resid, case_s, n_boot=100, seed=SEED)
                rec.update({f"pcorr_{h}": rho, f"pcorr_lo_{h}": lo, f"pcorr_hi_{h}": hi})
                rec[f"cmi_{h}"] = bm.conditional_mi(x, y, off_s[h])
                rec[f"grey_auroc_{h}"], rec[f"grey_dir_{h}"] = bm.grey_auroc(x, y, mp)
                g = bm.incremental_gain(Xs[[c]], y, off_s[h], folds_s, range(args.folds), SEED)
                rec.update({f"dauroc_{h}": g["dauroc"], f"dauprc_{h}": g["dauprc"], f"dlogloss_{h}": g["dlogloss"],
                            f"dauroc_pos_folds_{h}": int((np.array(g["dauroc_folds"]) > 0).sum())})
                single[str(h)][c] = g["dlogloss_folds"]
            rec["gain_mean"] = float(np.mean([(rec[f"dauroc_{h}"] + rec[f"dauprc_{h}"]) / 2 for h in H]))
            rows.append(rec)
            if i % 20 == 0:
                log(f"L1 {i + 1}/{len(cand)} {c}: dAUROC 5/15 min {rec['dauroc_300']:+.4f} {rec['dauroc_900']:+.4f}")
        l1 = pd.DataFrame(rows).sort_values("gain_mean", ascending=False)
        l1.to_csv(l1_path, index=False, float_format="%.5f")
        single_path.write_text(json.dumps(single))
    fig_alone_vs_beyond(l1, 900, FIG / "l1_alone_vs_beyond_h900.png")
    fig_alone_vs_beyond(l1, 300, FIG / "l1_alone_vs_beyond_h300.png")

    # L3 view of the composites: does the observed direction agree with physiology?
    c3 = l1[l1["family"].isin(["composite", "map"]) & l1["feature"].str.startswith("cx_")].copy()
    c3["hypothesis"] = c3["feature"].map(lambda c: bm.COMPOSITES[c][0])
    c3["expected_sign"] = c3["feature"].map(lambda c: bm.COMPOSITES[c][1])
    c3["observed_sign_900"] = np.sign(c3["pcorr_900"].to_numpy(np.float64))
    sig = ((c3["pcorr_lo_900"] > 0) | (c3["pcorr_hi_900"] < 0)).to_numpy()
    c3["consistent"] = np.where(c3["expected_sign"] == 0, "không có kỳ vọng",
                                np.where(~sig, "không có ý nghĩa",
                                         np.where(c3["observed_sign_900"] == c3["expected_sign"], "đúng", "ngược")))
    c3[["feature", "hypothesis", "expected_sign", "observed_sign_900", "pcorr_900", "pcorr_lo_900", "pcorr_hi_900",
        "consistent", "auroc_900", "dauroc_300", "dauroc_900", "gain_mean", "missing"]].to_csv(
        out_dir / "l3_composites.csv", index=False, float_format="%.5f")

    # ---------------------------------------------------------------- L2: pairs
    syn_path = out_dir / "l2_synergy.csv"
    beyond = l1[l1["family"] != "map"]
    corr_all = fe.spearman(Xs, core + l1["feature"].tolist(), 40000, SEED)
    corr_all.to_csv(out_dir / "spearman.csv", float_format="%.3f")
    bf = beyond["feature"].tolist()
    clusters = fe.redundancy_clusters(corr_all.loc[bf, bf], 0.8)
    pool = beyond.assign(cl=beyond["feature"].map(clusters)).drop_duplicates("cl")["feature"].head(args.pool).tolist()
    summary["l2_pool"] = pool
    if syn_path.exists():
        syn = pd.read_csv(syn_path)
        log("L2 reused")
    else:
        parts = []
        for h in (300, 900):
            s = bm.pair_synergy(Xs, ys_s[h], off_s[h], folds_s, pool, single[str(h)], range(args.folds), SEED)
            parts.append(s.rename(columns={c: f"{c}_{h}" for c in s.columns if c not in ("a", "b")}))
            log(f"L2 h={h}: {len(s)} pairs, best synergy {s['synergy'].max():.5f}")
        syn = parts[0].merge(parts[1], on=["a", "b"])
        syn["synergy_mean"] = (syn["synergy_300"] + syn["synergy_900"]) / 2
        syn["stable"] = ((syn["synergy_pos_folds_300"] + syn["synergy_pos_folds_900"])
                         >= 0.8 * (syn["n_folds_300"] + syn["n_folds_900"])) & (syn["synergy_mean"] > 0)
        syn["meaning_a"] = syn["a"].map(describe)
        syn["meaning_b"] = syn["b"].map(describe)
        syn = syn.sort_values("synergy_mean", ascending=False)
        syn.to_csv(syn_path, index=False, float_format="%.6f")
    for h in (300, 900):
        fig_synergy(syn, pool, h, FIG / f"l2_synergy_h{h}.png")

    shap_path = out_dir / "l2_shap_interaction.csv"
    if not shap_path.exists():
        try:
            fs = core + [c for c in pool if c not in core]
            k = np.isin(ys_s[900], (0, 1))
            b = fe.fit_lgbm(Xs.loc[k, fs], ys_s[900][k], W, SEED)
            pick = rng.choice(np.flatnonzero(k), min(args.shap_rows, int(k.sum())), replace=False)
            m = bm.shap_interactions(b, Xs.loc[pick, fs])
            long = (m.where(np.triu(np.ones(m.shape, bool), 1)).stack().rename("mean_abs_interaction")
                    .reset_index().rename(columns={"level_0": "a", "level_1": "b"}))
            long["kind"] = np.where(long["a"].isin(core) & long["b"].isin(core), "MAP × MAP",
                                    np.where(long["a"].isin(core) | long["b"].isin(core), "MAP × khác",
                                             "khác × khác"))
            long.sort_values("mean_abs_interaction", ascending=False).to_csv(shap_path, index=False,
                                                                            float_format="%.6f")
            log("L2 SHAP interactions")
        except Exception as e:  # shap missing or too slow: the synergy table still stands
            log(f"L2 SHAP interactions skipped: {e!r}")
    if shap_path.exists():
        si = pd.read_csv(shap_path)
        mx = si[si["kind"] == "MAP × khác"].assign(other=lambda d: np.where(d["a"].isin(core), d["b"], d["a"]))
        mod = mx.groupby("other")["mean_abs_interaction"].sum().sort_values(ascending=False).head(6).index.tolist()
        summary["map_modulators"] = mod
        fig_modulation(lab[sub].reset_index(drop=True), Xs, mod, 900, FIG / "l2_map_modulation_h900.png")

    # ---------------------------------------------------------------- L4: selection
    fwd_path, pre_path = out_dir / "l4_forward.csv", out_dir / "l4_prefix.csv"
    lf = l1["feature"].tolist()
    cl_all = fe.redundancy_clusters(corr_all.loc[lf, lf], 0.8)
    top = l1.assign(cl=l1["feature"].map(cl_all)).drop_duplicates("cl")
    top = top[top["gain_mean"] > 0]["feature"].head(args.bundles).tolist()
    bundles = {f: [f] for f in top}
    for r in syn[syn["stable"]].head(args.pair_bundles).itertuples():
        bundles[f"{r.a} + {r.b}"] = [r.a, r.b]
    if fwd_path.exists():
        fwd = pd.read_csv(fwd_path)
        log("L4 forward reused")
    else:
        log(f"L4 forward over {len(bundles)} bundles")
        fwd = bm.forward_bundles(Xs, {h: ys_s[h] for h in (300, 900)}, {h: off_s[h] for h in (300, 900)}, folds_s,
                                 bundles, holdouts=(0, 1, 2), max_steps=args.max_steps, seed=SEED, log=log)
        fwd.to_csv(fwd_path, index=False, float_format="%.5f")
    kept = fwd[fwd["kept"].astype(bool)] if len(fwd) else fwd
    prefixes = [[]] + [r.split() for r in kept["columns"].fillna("")]
    if pre_path.exists():
        pre = pd.read_csv(pre_path)
    else:
        rows = []
        for k, extra in enumerate(prefixes):
            fs = core + extra
            for h in H:
                y = ys_s[h]
                a = []
                for hold in (0, 1):
                    tr = np.isin(y, (0, 1)) & (folds_s != hold)
                    te = np.isin(y, (0, 1)) & (folds_s == hold)
                    b = fe.fit_lgbm(Xs.loc[tr, fs], y[tr], W, SEED)
                    a.append(fe.eval_auroc(b, Xs.loc[te, fs], y[te]))
                rows.append({"k": k, "horizon": h, "n_features": len(fs), "auroc_cv": float(np.mean(a)),
                             "added": " ".join(extra[len(prefixes[k - 1]):]) if k else ""})
            log(f"L4 prefix k={k}: " + " ".join(f"{r['auroc_cv']:.4f}" for r in rows[-len(H):]))
        pre = pd.DataFrame(rows)
        pre.to_csv(pre_path, index=False, float_format="%.5f")
    mean_k = pre.groupby("k")["auroc_cv"].mean()
    k_star = int(mean_k.index[np.argmax(mean_k.to_numpy() >= mean_k.max() - 0.001)])
    selected = core + prefixes[k_star]
    fig_prefix(pre, FIG / "l4_prefix.png", k_star)
    summary.update({"k_star": k_star, "selected_beyond_map": prefixes[k_star], "n_selected": len(selected)})
    save()
    log(f"selected: MAP core ({len(core)}) + {prefixes[k_star]}")

    # ---------------------------------------------------------------- final: validation under 3 protocols
    vcols = list(dict.fromkeys([c for c in cols if c not in epc] + ["map_min_w30", "map_missing_w30"]))
    vlab, VX = fe.load_rows(args.samples, "validation", vcols, args.manifest, all_main=True)
    VX = bm.add_composites(VX, W)
    VX, _ = bm.merge_extra(VX, ep_extra)
    vrows = pd.concat([vlab.reset_index(drop=True),
                       VX[["map_current", "map_min_w30", "map_missing_w30"]].reset_index(drop=True)], axis=1)
    events = pd.read_parquet(args.samples / "events.parquet")
    events = events[events["split"] == "validation"]
    ctx = groups["numeric"] + groups["context"]
    models = {
        "MAP-only (lõi MAP)": core,
        "lgbm_context (97)": ctx,
        f"Tất cả {len(cols)}": cols,
        f"Tất cả {len(cols)} + tổ hợp": cols + cx,
        **({f"lgbm_context + ECG/PPG ({len(epc)})": ctx + epc} if epc else {}),
        f"Chọn lọc ({len(selected)})": selected,
        "lgbm_context + chọn lọc": list(dict.fromkeys(ctx + selected)),
    }
    preds = {h: {"MAP hiện tại (luật)": -VX["map_current"].to_numpy(np.float64)} for h in H}
    cost = []
    for name, fs in models.items():
        for h in H:
            k = np.isin(ys[h], (0, 1))
            b = fe.fit_lgbm(X.loc[k, fs], ys[h][k], W, SEED)
            preds[h][name] = b.predict(VX[fs], raw_score=True)
        cost.append({"model": name, "n_inputs": len(fs), **inference_cost(b, VX[fs], rng)})
        log(f"final fit {name} ({len(fs)} inputs): {cost[-1]['latency_ms_row']:.3f} ms/row")
    pd.DataFrame(cost).to_csv(out_dir / "inference_cost.csv", index=False, float_format="%.4f")
    pd.DataFrame({"caseid": vrows["caseid"], "time": vrows["time"],
                  **{f"{n}|{h}": preds[h][n] for h in H for n in preds[h]}}).to_parquet(out_dir / "val_scores.parquet",
                                                                                        index=False)
    negs = {"hatib": pp.hatib_negatives(vrows, events), "lee": pp.lee_negatives(vrows, events)}
    out = []
    for proto in ("every", "hatib", "lee"):
        for h in H:
            idx, y = pp.protocol_rows(vrows, events, h, proto, negs)
            sc = {n: v[idx] for n, v in preds[h].items()}
            ok = np.all([np.isfinite(v) for v in sc.values()], axis=0)
            res = pp.paired_bootstrap(y[ok], {n: v[ok] for n, v in sc.items()}, vrows["subjectid"].to_numpy()[idx][ok],
                                      "lgbm_context (97)", args.boot, SEED)
            res["auprc"] = [row_metrics(y[ok], sigmoid(np.clip(sc[n][ok], -30, 30)) if n != "MAP hiện tại (luật)"
                                        else (sc[n][ok] - sc[n][ok].min()) / (np.ptp(sc[n][ok]) + EPS))["auprc"]
                            for n in res["model"]]
            res["gain_over_map_only"] = res["auroc"] - res.loc[res["model"] == "MAP-only (lõi MAP)", "auroc"].iloc[0]
            res.insert(0, "horizon", h)
            res.insert(0, "protocol", proto)
            out.append(res)
            log(f"{proto} h={h}: " + "; ".join(f"{r.model} {r.auroc:.3f}" for r in res.itertuples()))
    fin = pd.concat(out, ignore_index=True)
    fin.to_csv(out_dir / "final_compare.csv", index=False, float_format="%.4f")
    fig_final(fin, FIG / "final_compare.png")
    summary["validation_rows"] = len(vrows)
    save()

    # ---------------------------------------------------------------- ladder: real before / after on validation
    # Base = the current model (lgbm_context); add the forward bundles one at a time (and the bundle that
    # stopped the search). The choice was made on train; here validation only reports what each step did.
    lad_rows, cur = [], list(ctx)
    steps = [("lgbm_context (97)", [])] + [(r.bundle, r.members.split()) for r in fwd.itertuples()]
    if epc:  # the whole ECG/PPG block as one step, so its before / after is measured directly
        steps.append((f"khối ECG/PPG ({len(epc)} biến)", epc))
    prev = None
    for i, (name, members) in enumerate(steps):
        fs = list(dict.fromkeys(cur + members))
        rec = {"step": i, "added": name, "n_inputs": len(fs),
               "kept_on_train": True if i == 0 or i > len(fwd) else bool(fwd["kept"].iloc[i - 1])}
        for h in H:
            k = np.isin(ys[h], (0, 1))
            p = fe.fit_lgbm(X.loc[k, fs], ys[h][k], W, SEED).predict(VX[fs], raw_score=True)
            for proto in ("every", "lee"):
                idx, y = pp.protocol_rows(vrows, events, h, proto, negs)
                v = p[idx]
                ok = np.isfinite(v)
                rec[f"auroc_{proto}_{h}"] = fe.auroc(y[ok], v[ok])
                if proto == "every":
                    rec[f"auprc_every_{h}"] = row_metrics(y[ok], sigmoid(np.clip(v[ok], -30, 30)))["auprc"]
        if prev is not None:
            for key in [c for c in rec if c.startswith(("auroc_", "auprc_"))]:
                rec[f"d_{key}"] = rec[key] - prev[key]
            d = np.mean([rec[f"d_auroc_every_{h}"] for h in H])
            rec["verdict"] = "tốt hơn" if d > 0.0005 else ("xấu hơn" if d < -0.0005 else "không đổi")
        lad_rows.append(rec)
        log(f"ladder {i} + {name}: " + " ".join(f"{rec[f'auroc_every_{h}']:.4f}" for h in H))
        if rec.get("verdict") != "xấu hơn" and rec["kept_on_train"]:
            cur, prev = fs, rec  # a step that hurt on validation is noted, not carried forward
    pd.DataFrame(lad_rows).to_csv(out_dir / "ladder_validation.csv", index=False, float_format="%.5f")
    write_eda_md(out_dir, out_dir / "README.md")
    log("done")


if __name__ == "__main__":
    main()
