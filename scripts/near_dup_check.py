"""Week 15: does dropping the near-duplicate inputs (every pair |Spearman| >= 0.95, one kept per group) lose anything?
-> <out>/{near_dup_cv.csv, near_dup_validation.csv, summary.json, README.md}

  python scripts/near_dup_check.py --drop-list reports/weekly/week15/near_duplicates.csv

Heavy (LightGBM CV): run it on Kaggle, never on the laptop. Three pairs, each full set vs the same set without the
dropped inputs: the 214 inputs, lgbm_context (numeric + context, 97) and the MAP core (41).
  near_dup_cv.csv          train CV (5 subject folds, same row subsample / seed as scripts/ablation_214.py), paired by fold
  near_dup_validation.csv  validation (report only), every-time-point / Hatib / Lee protocols, subject-bootstrap CI of
                           AUROC and of the paired difference pruned - full
Test is never read.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uc04 import ablation as ab
from uc04 import beyond_map as bm
from uc04 import feature_eda as fe
from uc04 import peer_protocols as pp
from uc04.metrics import row_metrics

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data/samples_v3"
MANIFEST = ROOT / "data/vitaldb_full/meta/cohort_manifest.csv"
DROP_LIST = ROOT / "reports/weekly/week15/near_duplicates.csv"
OUT = ROOT / "eda/near_dup"
SEED = 20261021  # = scripts/ablation_214.py, so the CV rows and folds are the same
H = fe.HORIZONS


def log(msg: str, t0: float = time.time()) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def _f(x, nd=4, sign=False):
    if x is None or not np.isfinite(x):
        return "–"
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace(".", ",")


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def write_readme(out: Path, cv: pd.DataFrame | None, val: pd.DataFrame | None, summary: dict) -> None:
    L = ["# Bỏ đặc trưng gần trùng (tuần 15)", "",
         "Sinh tự động từ `scripts/near_dup_check.py` (Kaggle). Nhóm gần trùng: mọi cặp |ρ| ≥ 0,95, giữ một đặc trưng "
         "mỗi nhóm (`reports/weekly/week15/near_duplicates.csv`). Mỗi cặp so bộ đầy đủ với cùng bộ đó sau khi bỏ. "
         "Chọn trên train, validation chỉ để báo cáo, test không đọc.", "",
         "Bỏ: " + ", ".join(f"`{c}`" for c in summary["dropped"]) + ".", ""]
    if cv is not None:
        t = cv.assign(**{"Bộ": cv["pair"], "Số biến": cv["n_full"].astype(str) + " → " + cv["n_pruned"].astype(str),
                         "AUROC đầy đủ": cv["auroc_full"].map(_f), "AUROC sau khi bỏ": cv["auroc_pruned"].map(_f),
                         "ΔAUROC": cv["d_auroc"].map(lambda x: _f(x, sign=True)),
                         "ΔAUROC 5' / 10' / 15'": [" / ".join(_f(r[f"d_auroc_{h}"], sign=True) for h in H)
                                                    for _, r in cv.iterrows()],
                         "ΔAUPRC": cv["d_auprc"].map(lambda x: _f(x, sign=True)),
                         "Fold xấu hơn": cv["d_auroc_neg_folds"].astype(str) + "/" + str(summary["holdouts"])})
        L += ["## CV trên train (trung bình 3 mốc, ghép cặp theo fold)", "",
              _table(t[["Bộ", "Số biến", "AUROC đầy đủ", "AUROC sau khi bỏ", "ΔAUROC", "ΔAUROC 5' / 10' / 15'",
                        "ΔAUPRC", "Fold xấu hơn"]]), ""]
    if val is not None:
        L += ["## Validation (bootstrap theo bệnh nhân, ghép cặp)", ""]
        for proto in ("every", "lee", "hatib"):
            v = val[val["protocol"] == proto]
            rows = []
            for pair in dict.fromkeys(v["pair"]):
                r = {"Bộ": pair}
                for h in H:
                    x = v[(v["pair"] == pair) & (v["horizon"] == h)].iloc[0]
                    r[f"{h // 60}' đầy đủ → sau khi bỏ"] = f"{_f(x['auroc_full'], 3)} → {_f(x['auroc_pruned'], 3)}"
                    r[f"Δ {h // 60}' [CI]"] = f"{_f(x['diff'], 4, True)} [{_f(x['lo'], 4, True)}; {_f(x['hi'], 4, True)}]"
                rows.append(r)
            L += [f"Protocol `{proto}`:", "", _table(pd.DataFrame(rows)), ""]
    (out / "README.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=120)
    ap.add_argument("--train-cases", type=int, default=0, help="0 = every train case")
    ap.add_argument("--row-frac", type=float, default=0.4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--holdouts", type=int, default=5)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--drop-list", type=Path, default=DROP_LIST)
    ap.add_argument("--samples", type=Path, default=SAMPLES)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out, W = args.out, args.W
    out.mkdir(parents=True, exist_ok=True)

    cols, fams = ab.base_columns(args.samples, W)
    groups = fe.candidate_columns(args.samples, W)
    nd = pd.read_csv(args.drop_list)
    drop = set(nd.loc[~nd["keep"], "feature"])
    full = {"214 đặc trưng": cols, "Mô hình hiện tại (lgbm_context)": groups["numeric"] + groups["context"],
            "Lõi MAP": fams["map"]}
    pairs = {name: (fs, [c for c in fs if c not in drop]) for name, fs in full.items()}
    dropped = sorted(drop & set(cols))
    summary = {"W": W, "dropped": dropped, "n_dropped_from_214": len(dropped),
               "not_in_214": sorted(drop - set(cols)),
               "pairs": {n: [len(a), len(b)] for n, (a, b) in pairs.items()}}
    log(f"drop {len(dropped)} of {len(cols)}; pairs " + ", ".join(f"{n} {len(a)}->{len(b)}" for n, (a, b) in pairs.items()))

    rng = np.random.default_rng(SEED)
    cases = fe.sample_cases(args.samples / "labels/train.parquet", args.train_cases, SEED) if args.train_cases else None
    lab, X = fe.load_rows(args.samples, "train", cols, args.manifest, cases)
    ys = {h: lab[f"y_{h}"].to_numpy() for h in H}
    folds = bm.subject_folds(lab["subjectid"].to_numpy(), args.folds, SEED)
    sub = rng.random(len(lab)) < args.row_frac
    Xs, ys_s, folds_s = X[sub].reset_index(drop=True), {h: y[sub] for h, y in ys.items()}, folds[sub]
    hold = list(range(min(args.holdouts, args.folds)))
    summary.update({"train_rows": len(lab), "train_cases": int(lab["caseid"].nunique()), "cv_rows": int(sub.sum()),
                    "holdouts": len(hold)})
    log(f"train rows {len(lab):,} ({summary['train_cases']} cases); CV rows {sub.sum():,}")

    # ---------------------------------------------------------------- CV
    rows = []
    for name, (fs, ps) in pairs.items():
        rf = ab.cv_eval(Xs, ys_s, folds_s, fs, hold, W, SEED)
        rp = ab.cv_eval(Xs, ys_s, folds_s, ps, hold, W, SEED)
        rows.append({"pair": name, "n_full": len(fs), "n_pruned": len(ps), "auroc_full": rf["auroc"],
                     "auroc_pruned": rp["auroc"], "auprc_full": rf["auprc"], "auprc_pruned": rp["auprc"],
                     **{f"d_auroc_{h}": rp[f"auroc_{h}"] - rf[f"auroc_{h}"] for h in H},
                     **ab.paired_diff(rp, rf, H)})
        log(f"CV {name}: {rf['auroc']:.4f} -> {rp['auroc']:.4f} (d {rows[-1]['d_auroc']:+.4f})")
        pd.DataFrame(rows).to_csv(out / "near_dup_cv.csv", index=False, float_format="%.5f")
    cv = pd.DataFrame(rows)

    # ---------------------------------------------------------------- validation
    extra = ["map_current", "map_min_w30", "map_missing_w30"]
    vlab, VX = fe.load_rows(args.samples, "validation", list(dict.fromkeys(cols + extra)), args.manifest,
                            all_main=True)
    vrows = pd.concat([vlab.reset_index(drop=True), VX[extra].reset_index(drop=True)], axis=1)
    events = pd.read_parquet(args.samples / "events.parquet")
    events = events[events["split"] == "validation"]
    negs = {"hatib": pp.hatib_negatives(vrows, events), "lee": pp.lee_negatives(vrows, events)}
    preds = {h: {} for h in H}
    for name, (fs, ps) in pairs.items():
        for h in H:
            k = np.isin(ys[h], (0, 1))
            for tag, f in (("full", fs), ("pruned", ps)):
                preds[h][(name, tag)] = fe.fit_lgbm(X.loc[k, f], ys[h][k], W, SEED).predict(VX[f], raw_score=True)
        log(f"validation fit {name}")
    vout = []
    for proto in ("every", "hatib", "lee"):
        for h in H:
            idx, y = pp.protocol_rows(vrows, events, h, proto, negs)
            subj = vrows["subjectid"].to_numpy()[idx]
            for name in pairs:
                sc = {t: preds[h][(name, t)][idx] for t in ("full", "pruned")}
                ok = np.isfinite(sc["full"]) & np.isfinite(sc["pruned"])
                res = pp.paired_bootstrap(y[ok], {t: v[ok] for t, v in sc.items()}, subj[ok], "full", args.boot,
                                          SEED).set_index("model")
                pr = {t: row_metrics(y[ok], 1 / (1 + np.exp(-np.clip(sc[t][ok], -30, 30))))["auprc"] for t in sc}
                vout.append({"protocol": proto, "horizon": h, "pair": name, "auroc_full": res.loc["full", "auroc"],
                             "auroc_pruned": res.loc["pruned", "auroc"], "diff": res.loc["pruned", "diff_vs_full"],
                             "lo": res.loc["pruned", "diff_lo"], "hi": res.loc["pruned", "diff_hi"],
                             "auprc_full": pr["full"], "auprc_pruned": pr["pruned"],
                             "n_pos": int(res.loc["full", "n_pos"]), "n_neg": int(res.loc["full", "n_neg"])})
            log(f"{proto} h={h} done")
    val = pd.DataFrame(vout)
    val.to_csv(out / "near_dup_validation.csv", index=False, float_format="%.5f")
    summary["validation_rows"] = len(vrows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(out, cv, val, summary)
    log("done")


if __name__ == "__main__":
    main()
