"""Statistical significance of every candidate input, alone and once MAP is known -> <out>/l1_stats_h<h>.csv.

  python scripts/feature_significance.py --horizons 900 [--train-cases 0] [--boot 100]

One Kaggle kernel per horizon. Same train rows, screening subsample, subject folds and MAP-only offset as
scripts/beyond_map_eda.py (same seed). Rows of one case are strongly autocorrelated, so every p-value comes from a
case-level bootstrap SE or from cluster-robust (subject) standard errors, never from row counts; q-values are
Benjamini-Hochberg over all inputs of a horizon.

  auroc, auroc_lo/hi, auroc_p          the input alone (H0: AUROC = 0.5)
  rho, rho_lo/hi, rho_p                Spearman with the label (point-biserial on ranks)
  prho, prho_lo/hi, prho_p             Spearman with the MAP residual y - p(MAP only)
  or_iqr, or_lo/hi, or_p               logistic y ~ offset(MAP logit) + x/IQR + missing(x), cluster-robust by subject
  dauroc, dauroc_lo/hi, dauroc_p       paired AUROC(offset + x) - AUROC(offset), out-of-fold, case bootstrap
  *_q                                  BH q-values; sig_beyond_map = or_q < 0.05 and dauroc_q < 0.05 and dauroc > 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uc04 import beyond_map as bm
from uc04 import feature_eda as fe

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261017  # = scripts/beyond_map_eda.py


def log(msg: str, t0: float = time.time()) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=120)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(fe.HORIZONS))
    ap.add_argument("--train-cases", type=int, default=0)
    ap.add_argument("--row-frac", type=float, default=0.4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="first N candidates only (smoke)")
    ap.add_argument("--samples", type=Path, default=ROOT / "data/samples_v3")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/vitaldb_full/meta/cohort_manifest.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "eda")
    ap.add_argument("--extra", type=Path, help="dir with ecgppg_shard*.parquet (ECG/PPG features) to add")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    W, rng = args.W, np.random.default_rng(SEED)

    groups = fe.candidate_columns(args.samples, W)
    cols = [c for g in fe.GROUP_ORDER for c in groups[g]]
    cases = fe.sample_cases(args.samples / "labels/train.parquet", args.train_cases, SEED) if args.train_cases else None
    lab, X = fe.load_rows(args.samples, "train", cols, args.manifest, cases)
    X = bm.add_composites(X, W)
    X, epc = bm.merge_extra(X, bm.load_extra(args.extra))
    cols = cols + epc
    core = bm.map_core(cols)
    cand = [c for c in cols + list(bm.COMPOSITES) if c not in core]
    if args.limit:
        cand = cand[:args.limit]
    folds = bm.subject_folds(lab["subjectid"].to_numpy(), args.folds, SEED)
    sub = rng.random(len(lab)) < args.row_frac
    Xs, fs_, case_s = X[sub].reset_index(drop=True), folds[sub], lab["caseid"].to_numpy()[sub]
    subj_s = lab["subjectid"].to_numpy()[sub]
    log(f"train rows {len(lab):,}; screening {sub.sum():,}; candidates {len(cand)}")

    for h in args.horizons:
        y = lab[f"y_{h}"].to_numpy()
        off = bm.map_offset_oof(X[core], y, folds, dict(fe.LGB_PARAMS), fe.N_TREES, SEED)[sub]
        ys = y[sub]
        k = np.isin(ys, (0, 1))
        resid = np.where(k, (ys == 1) - 1 / (1 + np.exp(-off)), np.nan)
        ybin = np.where(k, (ys == 1).astype(float), np.nan)
        log(f"h={h}: MAP-only OOF AUROC {fe.auroc((ys[k] == 1).astype(np.int8), off[k]):.4f}")
        rows = []
        for i, c in enumerate(cand):
            x = Xs[c].to_numpy(np.float64)
            rec = {"feature": c, "family": "composite" if c.startswith("cx_") else
                   ("ecgppg" if c.startswith("ep_") else fe.feature_group(c)),
                   "missing": float(np.isnan(x).mean())}
            a = bm.auroc_boot(ys, x, case_s, args.boot, SEED)
            fold = a["est"] < 0.5  # direction: report AUROC >= 0.5 with its sign
            rec.update({"auroc": 1 - a["est"] if fold else a["est"], "auroc_lo": 1 - a["hi"] if fold else a["lo"],
                        "auroc_hi": 1 - a["lo"] if fold else a["hi"], "auroc_p": a["p"],
                        "direction": "thấp → nguy cơ" if fold else "cao → nguy cơ"})
            r = bm.spearman_boot(x, ybin, case_s, args.boot, SEED)
            rec.update({"rho": r["rho"], "rho_lo": r["lo"], "rho_hi": r["hi"], "rho_p": r["p"]})
            r = bm.spearman_boot(x, resid, case_s, args.boot, SEED)
            rec.update({"prho": r["rho"], "prho_lo": r["lo"], "prho_hi": r["hi"], "prho_p": r["p"]})
            g = bm.logit_offset_test(x, ys, off, subj_s)
            rec.update({"or_iqr": g["or_iqr"], "or_lo": g["or_lo"], "or_hi": g["or_hi"], "or_p": g["p"]})
            inc = bm.oof_incremental(Xs[[c]], ys, off, fs_, SEED)
            d = bm.auroc_boot(ys, inc, case_s, args.boot, SEED, s_ref=off)
            rec.update({"dauroc": d["est"], "dauroc_lo": d["lo"], "dauroc_hi": d["hi"], "dauroc_p": d["p"]})
            rows.append(rec)
            if i % 20 == 0:
                log(f"h={h} {i + 1}/{len(cand)} {c}: OR/IQR {rec['or_iqr']:.3f} p={rec['or_p']:.2g}; "
                    f"dAUROC {rec['dauroc']:+.4f} p={rec['dauroc_p']:.2g}")
        st = pd.DataFrame(rows)
        for t in ("auroc", "rho", "prho", "or", "dauroc"):
            st[f"{t}_q"] = bm.bh_qvalues(st[f"{t}_p"].to_numpy())
        st["sig_beyond_map"] = (st["or_q"] < 0.05) & (st["dauroc_q"] < 0.05) & (st["dauroc"] > 0)
        st.insert(0, "horizon", h)
        st.sort_values("dauroc", ascending=False).to_csv(args.out / f"l1_stats_h{h}.csv", index=False,
                                                          float_format="%.6g")
        log(f"h={h}: {int(st['sig_beyond_map'].sum())} inputs significant beyond MAP (BH q < 0.05)")


if __name__ == "__main__":
    main()
