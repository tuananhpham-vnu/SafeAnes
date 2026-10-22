"""Week 15 ablation on samples v3, W = 120 -> <out>/{README.md, *.csv, figures/, summary.json}.

  $env:PYTHONPATH = "$PWD/src;$PWD/.local_deps"
  python scripts/ablation_214.py --stages groups,validate        # kernel 1
  python scripts/ablation_214.py --stages rfe,loo,validate       # kernel 2

Inputs: the 214 of week 15 (samples_v3 tree: labels/, features/, features_v5/; fit on train), or with --v8 the 349
cleaned columns of lgbm_full_v4 (<samples>/features_v8 + features_v7/clean_columns.json; features_v8 has no train
split, so CV and the reported fits use calibration; labels and events come from --labels / --events).

Heavy (hours on CPU): run it on Kaggle, never on the laptop.

  groups_cv.csv        leave one group out of the full set, MAP core + one group (grouped CV on train, paired with 214)
  rfe_path.csv         recursive elimination path: CV AUROC / AUPRC per size, paired change against 214,
                       inputs dropped after each step; rfe_steps.jsonl keeps every step's set and SHAP ranking
  rfe_order.csv        every input with the size of the set it was dropped from (kept = never dropped)
  loo_chosen.csv       on the chosen set: CV change when one input or one group is left out
  validation_*.csv     validation (report only): MAP rule, MAP-only, lgbm_context, 214 and the ablated sets under
                       the every-time-point / Hatib / Lee-HypoBridCast protocols, subject-bootstrap CI, paired
                       difference to the full model
Everything that chooses (ranking, stop point) uses train only. Test is never read.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uc04 import ablation as ab
from uc04 import beyond_map as bm
from uc04 import feature_eda as fe
from uc04 import peer_protocols as pp
from uc04.columns import CONTEXT_COLS, numeric_cols
from uc04.metrics import row_metrics

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data/samples_v3"
MANIFEST = ROOT / "data/vitaldb_full/meta/cohort_manifest.csv"
OUT = ROOT / "eda/ablation"
SEED = 20261021
H = fe.HORIZONS
FULL = "Tất cả 214"  # reset in main to the size of the input set
RULE = "MAP hiện tại (luật)"
PATH_SIZES = (150, 100, 75, 50, 40, 30, 25, 20, 15, 10, 5)
COLORS = {"map": "#2a6fdb", "numeric": "#5d6d7e", "context": "#16a085", "wave": "#8e44ad", "anaes": "#e67e22",
          "clinical": "#c0392b"}


def log(msg: str, t0: float = time.time()) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def describe(c: str) -> str:
    return fe.describe(c)[1]


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


# ------------------------------------------------------------------------------------------------ figures

def fig_path(path: pd.DataFrame, chosen_n: int, tol: float, out: Path):
    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, m in zip(axes, ("auroc", "auprc")):
        for h, color in zip(H, ("#2a6fdb", "#e67e22", "#16a085")):
            ax.plot(path["n"], path[f"{m}_{h}"], marker="o", ms=2.5, lw=1, color=color, label=f"{h // 60} phút")
        ax.axvline(chosen_n, color="black", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("Số biến còn lại (thang log)")
        ax.set_ylabel(f"{m.upper()} CV trên train")
        ax.legend(frameon=False, fontsize=7)
    ax = axes[0].twinx()
    ax.plot(path["n"], path["d_auroc"] * 100, color="black", lw=0.8, ls="--")
    ax.axhline(-tol * 100, color="grey", lw=0.6)
    ax.set_ylabel("ΔAUROC × 100 so với bộ đủ (TB 3 mốc, nét đứt)", fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_bars(df: pd.DataFrame, label_col: str, color_col: str | None, title: str, out: Path):
    plt = plt_setup()
    d = df.sort_values("d_auroc")
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(d) + 1.2))
    colors = [COLORS.get(c, "#7f8c8d") for c in d[color_col]] if color_col else "#5d6d7e"
    ax.barh(d[label_col], d["d_auroc"] * 100, color=colors)
    ax.axvline(0, color="grey", lw=0.7)
    ax.set_xlabel("ΔAUROC × 100 (TB 3 mốc, CV)")
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_validation(val: pd.DataFrame, out: Path):
    plt = plt_setup()
    d = val[val["model"].str.startswith(("RFE", FULL)) & (val["protocol"] == "every")].copy()
    d["n"] = d["n_inputs"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for h, color in zip(H, ("#2a6fdb", "#e67e22", "#16a085")):
        dh = d[d["horizon"] == h].sort_values("n")
        ax.plot(dh["n"], dh["auroc"], marker="o", ms=3, color=color, label=f"{h // 60} phút")
        ax.fill_between(dh["n"], dh["lo"], dh["hi"], color=color, alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Số biến (thang log)")
    ax.set_ylabel("AUROC validation (mọi thời điểm)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ validation

def validate(models: dict[str, list[str]], X, ys, VX, vrows, events, W: int, boot: int) -> pd.DataFrame:
    preds = {h: {RULE: -VX["map_current"].to_numpy(np.float64)} for h in H}
    for name, fs in models.items():
        for h in H:
            k = np.isin(ys[h], (0, 1))
            preds[h][name] = fe.fit_lgbm(X.loc[k, fs], ys[h][k], W, SEED).predict(VX[fs], raw_score=True)
        log(f"validation fit {name} ({len(fs)} inputs)")
    negs = {"hatib": pp.hatib_negatives(vrows, events), "lee": pp.lee_negatives(vrows, events)}
    n_in = {RULE: 1, **{n: len(f) for n, f in models.items()}}
    out = []
    for proto in ("every", "hatib", "lee"):
        for h in H:
            idx, y = pp.protocol_rows(vrows, events, h, proto, negs)
            sc = {n: v[idx] for n, v in preds[h].items()}
            ok = np.all([np.isfinite(v) for v in sc.values()], axis=0)
            res = pp.paired_bootstrap(y[ok], {n: v[ok] for n, v in sc.items()},
                                      vrows["subjectid"].to_numpy()[idx][ok], FULL, boot, SEED)
            res["auprc"] = [row_metrics(y[ok], sigmoid(sc[n][ok]) if n != RULE
                                        else (sc[n][ok] - sc[n][ok].min()) / (np.ptp(sc[n][ok]) + 1e-9))["auprc"]
                            for n in res["model"]]
            res["d_auprc_vs_full"] = res["auprc"] - res.loc[res["model"] == FULL, "auprc"].iloc[0]
            res.insert(1, "n_inputs", res["model"].map(n_in))
            res.insert(0, "horizon", h)
            res.insert(0, "protocol", proto)
            out.append(res)
            log(f"{proto} h={h}: " + "; ".join(f"{r.model} {r.auroc:.3f}" for r in res.itertuples()))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------------------------------------ report

def _f(x, nd=4, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",")


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def verdict(d: float, tol: float) -> str:
    return "mất thông tin" if d < -tol else ("tốt hơn khi bỏ" if d > tol else "không đổi")


def write_readme(d: Path, summary: dict) -> None:
    tol = summary.get("tol_auroc", 0.0005)
    fam = "; ".join(f"{ab.FAMILY_VI[f]} {n}" for f, n in summary["families"].items())
    fit = summary.get("fit_split", "train")
    L = [f"# Ablation {summary['n_inputs']} đặc trưng (tuần 15)", "",
         f"Sinh tự động từ `scripts/ablation_214.py` (Kaggle). Bộ gốc: {summary['source']}; {summary['n_inputs']} "
         f"đặc trưng = {fam}. W = 120 s, mốc 5 / 10 / 15 phút. Chọn trên **{fit}** (CV 5 fold theo bệnh nhân, "
         + f"{summary.get('screen_rows', 0):,}".replace(",", ".") + " dòng); mô hình báo cáo trên validation được fit "
         f"trên toàn bộ {fit}; validation chỉ để báo cáo; test không đọc.",
         "", f"Quy ước: ΔAUROC trung bình 3 mốc < −{_f(tol)} là **mất thông tin**, > {_f(tol)} là **tốt hơn khi bỏ**, "
         "còn lại **không đổi**. Mô hình: LightGBM tham số `configs/uc04_v3.json` (500 cây, 7 lá).", ""]
    g = d / "groups_cv.csv"
    if g.exists():
        g = pd.read_csv(g)
        L += ["## 1. Ablation theo nhóm (CV trên train)", "",
              f"Mốc: {FULL}, AUROC CV {_f(summary['groups_ref_auroc'])}, AUPRC {_f(summary['groups_ref_auprc'])} "
              "(trung bình 3 mốc).", ""]
        t = g.assign(**{"Cấu hình": g["config"], "Số biến": g["n"], "AUROC": g["auroc"].map(_f),
                        "ΔAUROC": g["d_auroc"].map(lambda x: _f(x, sign=True)),
                        "ΔAUROC 5' / 10' / 15'": [" / ".join(_f(r[f"d_auroc_{h}"], sign=True) for h in H)
                                                   for _, r in g.iterrows()],
                        "ΔAUPRC": g["d_auprc"].map(lambda x: _f(x, sign=True)),
                        "Fold xấu hơn": g["d_auroc_neg_folds"].astype(str) + "/" + str(summary["holdouts"]),
                        "Kết luận": [verdict(x, tol) if c.startswith("−") else "" for c, x in zip(g["config"],
                                                                                              g["d_auroc"])]})
        L += [_table(t[["Cấu hình", "Số biến", "AUROC", "ΔAUROC", "ΔAUROC 5' / 10' / 15'", "ΔAUPRC",
                        "Fold xấu hơn", "Kết luận"]]), "", "![](figures/groups.png)", ""]
    p = d / "rfe_path.csv"
    if p.exists():
        p = pd.read_csv(p)
        ch = summary["chosen"]
        L += ["## 2. Loại dần từng đặc trưng (RFE theo |SHAP|)", "",
              "Mỗi bước: fit CV trên bộ hiện tại cho cả 3 mốc, xếp hạng bằng |SHAP| trung bình trên fold giữ lại, "
              "bỏ 15 % biến yếu nhất khi còn > 60 biến, 5 % (ít nhất 1) khi ≤ 60; biến chưa từng được dùng luôn bị bỏ.",
              "", f"**Điểm dừng (chọn trên train):** bộ nhỏ nhất có ΔAUROC ≥ −{_f(tol)} và ΔAUPRC ≥ "
              f"−{_f(summary['tol_auprc'])} so với {summary['n_inputs']}: **{ch['n']} biến** (bước {ch['step']}, ΔAUROC "
              f"{_f(ch['d_auroc'], sign=True)}, ΔAUPRC {_f(ch['d_auprc'], sign=True)}).", "",
              "Kích thước nhỏ nhất theo từng ngưỡng chịu mất AUROC: " +
              "; ".join(f"−{_f(float(k))}: {v} biến" for k, v in summary["smallest_by_tol"].items()) + ".", "",
              "![](figures/rfe_path.png)", ""]
        show = p[p["n"].isin(sorted(set(summary["path_points"]) | {ch["n"], int(p["n"].max())}))]
        t = show.assign(**{"Số biến": show["n"], "AUROC": show["auroc"].map(_f), "AUPRC": show["auprc"].map(_f),
                           "ΔAUROC": show["d_auroc"].map(lambda x: _f(x, sign=True)),
                           "ΔAUROC 5' / 10' / 15'": [" / ".join(_f(r[f"d_auroc_{h}"], sign=True) for h in H)
                                                      for _, r in show.iterrows()],
                           "ΔAUPRC": show["d_auprc"].map(lambda x: _f(x, sign=True)),
                           "Fold xấu nhất": show["d_auroc_worst_fold"].map(lambda x: _f(x, sign=True))})
        L += [_table(t[["Số biến", "AUROC", "AUPRC", "ΔAUROC", "ΔAUROC 5' / 10' / 15'", "ΔAUPRC", "Fold xấu nhất"]]),
              ""]
        o = pd.read_csv(d / "rfe_order.csv")
        cnt = o.groupby("family").agg(total=("feature", "size"), kept=("kept", "sum"))
        L += ["Số biến còn lại trong bộ chọn theo nhóm: " + "; ".join(
            f"{ab.FAMILY_VI[f]} {int(cnt.loc[f, 'kept'])}/{int(cnt.loc[f, 'total'])}" for f in ab.FAMILIES
            if f in cnt.index) + ".", ""]
        k = o[o["kept"]].sort_values("importance", ascending=False)
        t = k.assign(**{"Biến": "`" + k["feature"] + "`", "Nhóm": k["family"].map(ab.FAMILY_VI),
                        "Ý nghĩa": k["meaning"], "|SHAP| chuẩn hóa": k["importance"].map(lambda x: _f(x, 3))})
        L += [f"### Bộ {ch['n']} biến được chọn", "", _table(t[["Biến", "Nhóm", "Ý nghĩa", "|SHAP| chuẩn hóa"]]), ""]
    lo = d / "loo_chosen.csv"
    if lo.exists():
        lo = pd.read_csv(lo)
        t = lo.sort_values("d_auroc").assign(**{
            "Bỏ ra": lambda x: x["unit"], "Số biến bỏ": lambda x: x["n_removed"],
            "ΔAUROC": lambda x: x["d_auroc"].map(lambda v: _f(v, sign=True)),
            "ΔAUROC 5' / 10' / 15'": lambda x: [" / ".join(_f(r[f"d_auroc_{h}"], sign=True) for h in H)
                                                 for _, r in x.iterrows()],
            "ΔAUPRC": lambda x: x["d_auprc"].map(lambda v: _f(v, sign=True)),
            "Fold xấu hơn": lambda x: x["d_auroc_neg_folds"].astype(str) + "/" + str(summary["holdouts"]),
            "Kết luận": lambda x: x["d_auroc"].map(lambda v: verdict(v, tol))})
        L += ["## 3. Bỏ từng biến / từng nhóm khỏi bộ đã chọn (CV)", "",
              "Mốc là bộ đã chọn; ΔAUROC âm = biến đó mang thông tin không thay được bằng các biến còn lại.", "",
              _table(t[["Bỏ ra", "Số biến bỏ", "ΔAUROC", "ΔAUROC 5' / 10' / 15'", "ΔAUPRC", "Fold xấu hơn",
                        "Kết luận"]]), "", "![](figures/loo_chosen.png)", ""]
    for name, title in (("validation_groups.csv", "Validation: ablation theo nhóm"),
                        ("validation_path.csv", "Validation: các điểm trên đường RFE")):
        v = d / name
        if not v.exists():
            continue
        v = pd.read_csv(v)
        L += [f"## 4. {title}", "", "Chỉ báo cáo, không dùng để chọn. Khác biệt so với bộ đủ là bootstrap theo bệnh "
              "nhân, ghép cặp (cùng mẫu lại).", ""]
        for proto in ("every", "lee", "hatib"):
            vp = v[v["protocol"] == proto]
            models = list(dict.fromkeys(vp["model"]))
            rows = []
            for m in models:
                r = {"Mô hình": m, "Số biến": int(vp.loc[vp["model"] == m, "n_inputs"].iloc[0])}
                for h in H:
                    x = vp[(vp["model"] == m) & (vp["horizon"] == h)].iloc[0]
                    r[f"AUROC {h // 60}'"] = _f(x["auroc"], 3)
                    r[f"Δ vs {summary['n_inputs']} {h // 60}' [CI]"] = (f"{_f(x['diff_vs_' + FULL], 4, True)} "
                                                      f"[{_f(x['diff_lo'], 4, True)}; {_f(x['diff_hi'], 4, True)}]")
                    r[f"AUPRC {h // 60}'"] = _f(x["auprc"], 3)
                rows.append(r)
            L += [f"Protocol `{proto}`:", "", _table(pd.DataFrame(rows)), ""]
        if name == "validation_path.csv":
            L += ["![](figures/validation_path.png)", ""]
    (d / "README.md").write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="groups,rfe,loo,validate")
    ap.add_argument("--W", type=int, default=120)
    ap.add_argument("--train-cases", type=int, default=0, help="0 = every train case")
    ap.add_argument("--row-frac", type=float, default=0.4, help="train rows used by the CV stages")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--holdouts", type=int, default=5, help="folds used as holdout in CV")
    ap.add_argument("--min-features", type=int, default=5)
    ap.add_argument("--shap-rows", type=int, default=20000)
    ap.add_argument("--tol-auroc", type=float, default=0.0005)
    ap.add_argument("--tol-auprc", type=float, default=0.002)
    ap.add_argument("--loo-max", type=int, default=0, help="limit single-input units in loo (0 = all)")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--samples", type=Path, default=SAMPLES)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--resume", type=Path, help="earlier output dir: its finished stages are reused")
    ap.add_argument("--v8", action="store_true", help="349 columns of features_v8 under --samples, fit on calibration")
    ap.add_argument("--labels", type=Path, help="--v8: dir with <split>.parquet main-cohort labels (+ map_*_w30)")
    ap.add_argument("--events", type=Path, help="--v8: events.parquet (default <samples>/events.parquet)")
    args = ap.parse_args()
    global FULL
    stages = set(args.stages.split(","))
    out, FIG = args.out, args.out / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    if args.resume and args.resume.resolve() != out.resolve():
        for p in args.resume.glob("*"):
            if p.is_file():
                shutil.copy2(p, out / p.name)
    W, rng = args.W, np.random.default_rng(SEED)
    sp = out / "summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    save = lambda: sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")  # noqa
    hold = list(range(min(args.holdouts, args.folds)))

    # ---------------------------------------------------------------- data
    fit = "calibration" if args.v8 else "train"
    if args.v8:
        cols, fams = ab.v8_columns(args.samples)
        lab, X = ab.load_v8(args.samples, args.labels, fit, cols)
        if args.train_cases:
            keep = lab["caseid"].isin(sorted(lab["caseid"].unique())[:args.train_cases]).to_numpy()
            lab, X = lab[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
        source = "349 cột đã làm sạch của `lgbm_full_v4` (`features_v8`)"
    else:
        cols, fams = ab.base_columns(args.samples, W)
        cases = fe.sample_cases(args.samples / "labels/train.parquet", args.train_cases, SEED) if args.train_cases else None
        lab, X = fe.load_rows(args.samples, "train", cols, args.manifest, cases)
        source = "214 đặc trưng tuần 15 (không ECG/PPG, không tổ hợp `cx_*`)"
    FULL = f"Tất cả {len(cols)}"
    core = fams["map"]
    ys = {h: lab[f"y_{h}"].to_numpy() for h in H}
    folds = bm.subject_folds(lab["subjectid"].to_numpy(), args.folds, SEED)
    sub = rng.random(len(lab)) < args.row_frac
    Xs, ys_s, folds_s = X[sub].reset_index(drop=True), {h: y[sub] for h, y in ys.items()}, folds[sub]
    log(f"train rows {len(lab):,} ({lab['caseid'].nunique()} cases); CV rows {sub.sum():,}; inputs {len(cols)} = "
        + ", ".join(f"{f} {len(c)}" for f, c in fams.items()))
    summary.update({"W": W, "n_inputs": len(cols), "families": {f: len(c) for f, c in fams.items()},
                    "source": source, "fit_split": fit,
                    "train_rows": len(lab), "train_cases": int(lab["caseid"].nunique()), "screen_rows": int(sub.sum()),
                    "holdouts": len(hold), "tol_auroc": args.tol_auroc, "tol_auprc": args.tol_auprc})
    save()
    cv = lambda fs, shap=0: ab.cv_eval(Xs, ys_s, folds_s, fs, hold, W, SEED, shap)  # noqa

    # ---------------------------------------------------------------- groups
    if "groups" in stages:
        gp = out / "groups_cv.csv"
        done = pd.read_csv(gp) if gp.exists() else None
        rows = [] if done is None else done.to_dict("records")
        seen = {r["config"] for r in rows}
        ref = cv(cols)
        summary.update({"groups_ref_auroc": ref["auroc"], "groups_ref_auprc": ref["auprc"]})
        configs = {f"− {ab.FAMILY_VI[f]}": (f, [c for c in cols if c not in set(fams[f])]) for f in fams}
        configs["Chỉ lõi MAP"] = ("map", core)
        configs.update({f"Lõi MAP + {ab.FAMILY_VI[f]}": (f, core + fams[f]) for f in fams if f != "map"})
        for name, (fam, fs) in configs.items():
            if name in seen:
                continue
            r = cv(fs)
            rows.append({"config": name, "family": fam, "n": len(fs), "auroc": r["auroc"], "auprc": r["auprc"],
                         **{f"auroc_{h}": r[f"auroc_{h}"] for h in H}, **ab.paired_diff(r, ref, H)})
            pd.DataFrame(rows).to_csv(gp, index=False, float_format="%.5f")
            log(f"groups {name} ({len(fs)}): AUROC {r['auroc']:.4f} (d {rows[-1]['d_auroc']:+.4f})")
        g = pd.DataFrame(rows)
        fig_bars(g[g["config"].str.startswith("−")], "config", "family",
                 "Bỏ một nhóm khỏi bộ đủ (âm = nhóm đó mang thông tin riêng)", FIG / "groups.png")
        save()

    # ---------------------------------------------------------------- rfe
    jp = out / "rfe_steps.jsonl"
    if "rfe" in stages or jp.exists():
        start = [json.loads(s) for s in jp.read_text(encoding="utf-8").splitlines()] if jp.exists() else []
        if "rfe" in stages and (not start or start[-1]["dropped"]):
            def on_step(rec):
                with jp.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            steps = ab.rfe_path(Xs, ys_s, folds_s, cols, hold, W, SEED, args.min_features, args.shap_rows,
                                start, on_step, log)
        else:
            steps = start
        path = ab.path_table(steps, H)
        path.to_csv(out / "rfe_path.csv", index=False, float_format="%.5f")
        k = ab.choose(path, args.tol_auroc, args.tol_auprc)
        chosen = steps[k]["columns"]
        ch = path[path["step"] == k].iloc[0]
        summary["chosen"] = {"step": k, "n": len(chosen), "d_auroc": float(ch["d_auroc"]),
                             "d_auprc": float(ch["d_auprc"]), "columns": chosen}
        summary["smallest_by_tol"] = {str(t): int(path[path["d_auroc"] >= -t]["n"].min())
                                      for t in (0.0005, 0.001, 0.002, 0.005, 0.01)}
        pts = sorted({int(path.iloc[(path["n"] - s).abs().argmin()]["n"]) for s in PATH_SIZES}, reverse=True)
        summary["path_points"] = pts
        dropped_at = {c: s["n"] for s in steps for c in s["dropped"]}
        last_imp = {c: v for s in steps for c, v in s["importance"].items()}  # importance in the last set holding c
        o = pd.DataFrame({"feature": cols})
        o["family"] = o["feature"].map(ab.family)
        o["meaning"] = o["feature"].map(describe)
        o["dropped_from_n"] = o["feature"].map(dropped_at)
        o["kept"] = o["feature"].isin(chosen)
        o["importance"] = o["feature"].map(steps[k]["importance"]).where(o["kept"], o["feature"].map(last_imp))
        o.sort_values(["dropped_from_n", "importance"], ascending=[True, False], na_position="last").to_csv(
            out / "rfe_order.csv", index=False, float_format="%.6f")
        fig_path(path, len(chosen), args.tol_auroc, FIG / "rfe_path.png")
        log(f"chosen step {k}: {len(chosen)} inputs (dAUROC {ch['d_auroc']:+.4f}, dAUPRC {ch['d_auprc']:+.4f})")
        save()

        # ------------------------------------------------------------ loo on the chosen set
        if "loo" in stages:
            lp = out / "loo_chosen.csv"
            singles = sorted(chosen, key=lambda c: -steps[k]["importance"][c])
            if args.loo_max:
                singles = singles[:args.loo_max]
            units = {f"nhóm {ab.FAMILY_VI[f]}": [c for c in chosen if ab.family(c) == f] for f in fams}
            units = {n: m for n, m in units.items() if 0 < len(m) < len(chosen)}
            units.update({c: [c] for c in singles})
            ref = {key: steps[k][key] for key in steps[k] if key.startswith(("auroc", "auprc"))}
            loo, _ = ab.leave_out(Xs, ys_s, folds_s, chosen, units, hold, W, SEED, ref,
                                  pd.read_csv(lp) if lp.exists() else None,
                                  lambda df: df.to_csv(lp, index=False, float_format="%.5f"), log)
            loo["family"] = loo["unit"].map(lambda u: ab.family(u) if u in chosen else "")
            loo.to_csv(lp, index=False, float_format="%.5f")
            fig_bars(loo, "unit", "family", "Bỏ ra khỏi bộ đã chọn (âm = cần giữ)", FIG / "loo_chosen.png")

    # ---------------------------------------------------------------- validation
    if "validate" in stages:
        if args.v8:
            vlab, VX = ab.load_v8(args.samples, args.labels, "validation", cols, all_main=True)
            vrows = pd.concat([vlab, VX[["map_current"]]], axis=1)
        else:
            vcols = list(dict.fromkeys(cols + ["map_current", "map_min_w30", "map_missing_w30"]))
            vlab, VX = fe.load_rows(args.samples, "validation", vcols, args.manifest, all_main=True)
            vrows = pd.concat([vlab.reset_index(drop=True),
                               VX[["map_current", "map_min_w30", "map_missing_w30"]].reset_index(drop=True)], axis=1)
        events = pd.read_parquet(args.events or args.samples / "events.parquet")
        events = events[events["split"] == "validation"]
        ctx = [c for c in numeric_cols(W) + list(CONTEXT_COLS) if c in set(cols)]
        base = {"MAP-only (lõi MAP)": core, f"lgbm_context ({len(ctx)})": ctx, FULL: cols}
        if "groups" in stages:
            models = {**base, **{f"{len(cols)} − {ab.FAMILY_VI[f]}": [c for c in cols if c not in set(fams[f])]
                                 for f in fams if f != "map"}}
            v = validate(models, X, ys, VX, vrows, events, W, args.boot)
            v.to_csv(out / "validation_groups.csv", index=False, float_format="%.4f")
        if "rfe" in stages:
            steps = [json.loads(s) for s in jp.read_text(encoding="utf-8").splitlines()]
            by_n = {s["n"]: s["columns"] for s in steps}
            models = {**base, **{f"RFE {n}": by_n[n] for n in summary["path_points"] if n != len(cols)}}
            models[f"RFE {summary['chosen']['n']} (chọn)"] = summary["chosen"]["columns"]
            models.pop(f"RFE {summary['chosen']['n']}", None)
            v = validate(models, X, ys, VX, vrows, events, W, args.boot)
            v.to_csv(out / "validation_path.csv", index=False, float_format="%.4f")
            fig_validation(v, FIG / "validation_path.png")
        summary["validation_rows"] = len(vrows)
        save()
    write_readme(out, summary)
    log("done")


if __name__ == "__main__":
    main()
