"""Week 15 ablation of the 214 inputs (MAP core, other monitor, case context, arterial wave, anaesthesia /
ventilation, pre-operative clinical; no ECG/PPG, no composites): which groups and which single inputs can go
without losing accuracy.

  groups  leave one group out of the 214, and MAP core + one group, in grouped CV
  rfe     recursive elimination: fit grouped-CV models on the current set for every horizon, rank the inputs by
          mean |SHAP| on the held-out folds (normalised per horizon and fold, then averaged), drop the weakest,
          repeat down to `min_n` inputs
  choose  the smallest set on the path whose CV AUROC and AUPRC (means over horizons) stay within a tolerance
          of the full set
  loo     on the chosen set: CV change when one input (or one group of it) is left out

All folds are grouped by subjectid and every choice uses train only; validation only reports.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import beyond_map as bm
from . import feature_eda as fe

EPS = 1e-12
FAMILIES = ("map", "numeric", "context", "wave", "anaes", "vent", "hemo", "lab", "clinical")
FAMILY_VI = {"map": "Lõi MAP", "numeric": "Monitor khác", "context": "Ngữ cảnh ca", "wave": "Sóng ART",
             "anaes": "Thuốc mê / BIS", "vent": "Thở máy", "hemo": "Huyết động bổ sung",
             "lab": "Xét nghiệm trong mổ", "clinical": "Lâm sàng trước mổ"}
_VENT = ("f2_pip", "f2_pplat", "f2_tv", "f2_mv", "f2_compliance", "f1b_peep", "f1b_etco2_", "f1c_vent_")
_HEMO = ("f2_nibp_", "f2_ev_", "f2_vg_", "f2_pleth_hr", "f2_fem_", "f2_bt")


def family(col: str) -> str:
    """Ablation family. f2_* / f3_* are the columns added by lgbm_full_v4 (features_v7 / features_v8)."""
    if bm.is_map_core(col):
        return "map"
    if col.startswith(("f3_", "f2_st_")):
        return "clinical"
    if col.startswith("f2_lab_") or col == "f1c_lab_n_abg":
        return "lab"
    if col.startswith(_HEMO):
        return "hemo"
    if col.startswith(_VENT):
        return "vent"
    if col.startswith("f2_"):
        return "anaes"
    return fe.feature_group(col)


def families(cols: Sequence[str]) -> dict[str, list[str]]:
    fams = {f: [c for c in cols if family(c) == f] for f in FAMILIES}
    assert sum(map(len, fams.values())) == len(cols), "a column has no ablation family"
    return {f: c for f, c in fams.items() if c}


def base_columns(samples, W: int) -> tuple[list[str], dict[str, list[str]]]:
    """The 214 inputs (every candidate group of feature_eda, no ECG/PPG, no composites) and their families."""
    groups = fe.candidate_columns(samples, W)
    cols = [c for g in fe.GROUP_ORDER for c in groups[g]]
    return cols, families(cols)


def v8_columns(root) -> tuple[list[str], dict[str, list[str]]]:
    """The 349 cleaned inputs of lgbm_full_v4 (features_v7/clean_columns.json, same order as features_v8)."""
    import json
    from pathlib import Path
    cols = json.loads((Path(root) / "features_v7" / "clean_columns.json").read_text())["columns"]
    return cols, families(cols)


def load_v8(root, labels, split: str, columns: Sequence[str], all_main: bool = False
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(labels, float32 features) of one split from features_v8 (main-cohort rows only, in label order).
    `labels/<split>.parquet` holds the main-cohort label rows (plus map_min_w30 / map_missing_w30 for the peer
    protocols). Without all_main: only eligible rows with at least one known label."""
    from pathlib import Path
    lab = pd.read_parquet(Path(labels) / f"{split}.parquet")
    X = pd.read_parquet(Path(root) / "features_v8" / f"{split}.parquet", columns=["caseid", "time", *columns])
    if not (np.array_equal(X["caseid"].to_numpy(), lab["caseid"].to_numpy())
            and np.array_equal(X["time"].to_numpy(), lab["time"].to_numpy())):
        raise ValueError(f"labels and features_v8 of {split} are not row-aligned")
    X = X.astype({c: "float32" for c in columns})
    if not all_main:
        keep = lab["eligible"].to_numpy(bool)
        keep &= np.any([np.isin(lab[f"y_{h}"].to_numpy(), (0, 1)) for h in fe.HORIZONS], axis=0)
        lab, X = lab[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
    return lab, X


def cv_eval(X: pd.DataFrame, ys: dict[int, np.ndarray], folds: np.ndarray, cols: Sequence[str],
            holdouts: Sequence[int], W: int, seed: int, shap_rows: int = 0) -> dict:
    """Per horizon: mean over holdout folds of AUROC and AUPRC of a full-size LightGBM on `cols` (the per-fold
    values are kept for paired comparisons). With shap_rows > 0 also `importance`: mean |SHAP| on up to that many
    held-out rows, normalised to sum 1 per (horizon, fold) and averaged."""
    from sklearn.metrics import average_precision_score as ap
    cols = list(cols)
    out: dict = {"n": len(cols)}
    imp = pd.Series(0.0, index=cols)
    rng = np.random.default_rng(seed + len(cols))
    for h, y in ys.items():
        ok = np.isin(y, (0, 1))
        au, pr = [], []
        for k in holdouts:
            tr, te = ok & (folds != k), ok & (folds == k)
            b = fe.fit_lgbm(X.loc[tr, cols], y[tr], W, seed)
            Xte = X.loc[te, cols]
            raw = b.predict(Xte, raw_score=True)
            yy = (y[te] == 1).astype(np.int8)
            au.append(fe.auroc(yy, raw))
            pr.append(float(ap(yy, raw)))
            if shap_rows:
                pick = rng.choice(len(Xte), min(shap_rows, len(Xte)), replace=False)
                s = fe.shap_by_feature(b, Xte.iloc[pick])
                imp += s / max(float(s.sum()), EPS) / (len(ys) * len(holdouts))
        out[f"auroc_{h}"], out[f"auprc_{h}"] = float(np.mean(au)), float(np.mean(pr))
        out[f"auroc_folds_{h}"], out[f"auprc_folds_{h}"] = au, pr
    out["auroc"] = float(np.mean([out[f"auroc_{h}"] for h in ys]))
    out["auprc"] = float(np.mean([out[f"auprc_{h}"] for h in ys]))
    if shap_rows:
        out["importance"] = imp
    return out


def paired_diff(r: dict, ref: dict, horizons: Sequence[int]) -> dict:
    """r - ref: mean over horizons of AUROC / AUPRC, and per holdout fold (same folds, so paired)."""
    fa = np.mean([np.subtract(r[f"auroc_folds_{h}"], ref[f"auroc_folds_{h}"]) for h in horizons], axis=0)
    fp = np.mean([np.subtract(r[f"auprc_folds_{h}"], ref[f"auprc_folds_{h}"]) for h in horizons], axis=0)
    return {"d_auroc": r["auroc"] - ref["auroc"], "d_auprc": r["auprc"] - ref["auprc"],
            "d_auroc_worst_fold": float(fa.min()), "d_auroc_neg_folds": int((fa < 0).sum()),
            "d_auprc_worst_fold": float(fp.min()),
            **{f"d_auroc_{h}": r[f"auroc_{h}"] - ref[f"auroc_{h}"] for h in horizons}}


def n_drop(n: int, n_zero: int, min_n: int) -> int:
    """15 % of the set while it is above 60 inputs, then 5 % (at least one); never-used inputs always go."""
    d = max(1, int(n * (0.15 if n > 60 else 0.05)), n_zero)
    return max(0, min(d, n - min_n))


def rfe_path(X: pd.DataFrame, ys: dict[int, np.ndarray], folds: np.ndarray, cols: Sequence[str],
             holdouts: Sequence[int], W: int, seed: int, min_n: int = 5, shap_rows: int = 20000,
             start: list[dict] | None = None, on_step: Callable[[dict], None] | None = None,
             log: Callable[[str], None] = print) -> list[dict]:
    """Recursive elimination from `cols`. Each step: CV on the current set (its metrics and SHAP ranking), then drop
    the weakest. `start` = steps already done (resume from the last step's `next`)."""
    steps = list(start or [])
    cur = list(steps[-1]["next"]) if steps else list(cols)
    while steps == [] or steps[-1]["dropped"]:
        r = cv_eval(X, ys, folds, cur, holdouts, W, seed, shap_rows)
        imp = r.pop("importance").sort_values(kind="stable")
        drop = imp.index[:n_drop(len(cur), int((imp <= 0).sum()), min_n)].tolist()
        rec = {"step": len(steps), **r, "columns": cur, "importance": imp.round(6).to_dict(), "dropped": drop,
               "next": [c for c in cur if c not in drop]}
        steps.append(rec)
        if on_step:
            on_step(rec)
        log(f"RFE step {rec['step']}: {len(cur)} inputs, AUROC {r['auroc']:.4f} AUPRC {r['auprc']:.4f}; "
            f"drop {len(drop)}: {' '.join(drop[:6])}{' ...' if len(drop) > 6 else ''}")
        cur = rec["next"]
    return steps


def path_table(steps: list[dict], horizons: Sequence[int]) -> pd.DataFrame:
    ref = steps[0]
    rows = []
    for s in steps:
        rows.append({"step": s["step"], "n": s["n"], "auroc": s["auroc"], "auprc": s["auprc"],
                     **{f"auroc_{h}": s[f"auroc_{h}"] for h in horizons},
                     **{f"auprc_{h}": s[f"auprc_{h}"] for h in horizons},
                     **paired_diff(s, ref, horizons), "dropped_next": " ".join(s["dropped"])})
    return pd.DataFrame(rows)


def choose(path: pd.DataFrame, tol_auroc: float, tol_auprc: float) -> int:
    """Step of the smallest set whose mean AUROC and AUPRC are within the tolerances of the full set (step 0)."""
    ok = path[(path["d_auroc"] >= -tol_auroc) & (path["d_auprc"] >= -tol_auprc)]
    return int(ok.sort_values(["n", "step"]).iloc[0]["step"])


def leave_out(X: pd.DataFrame, ys: dict[int, np.ndarray], folds: np.ndarray, cols: Sequence[str],
              units: dict[str, list[str]], holdouts: Sequence[int], W: int, seed: int, ref: dict | None = None,
              done: pd.DataFrame | None = None, on_row: Callable[[pd.DataFrame], None] | None = None,
              log: Callable[[str], None] = print) -> tuple[pd.DataFrame, dict]:
    """CV change when each unit (one input or a group) is removed from `cols` (paired with the full `cols`)."""
    cols = list(cols)
    ref = ref or cv_eval(X, ys, folds, cols, holdouts, W, seed)
    rows = [] if done is None else done.to_dict("records")
    seen = {r["unit"] for r in rows}
    for name, members in units.items():
        if name in seen:
            continue
        keep = [c for c in cols if c not in set(members)]
        r = cv_eval(X, ys, folds, keep, holdouts, W, seed)
        rows.append({"unit": name, "n_removed": len(cols) - len(keep), "n_left": len(keep), "auroc": r["auroc"],
                     "auprc": r["auprc"], **paired_diff(r, ref, list(ys))})
        log(f"leave out {name} ({rows[-1]['n_removed']}): dAUROC {rows[-1]['d_auroc']:+.4f} "
            f"dAUPRC {rows[-1]['d_auprc']:+.4f}")
        if on_row:
            on_row(pd.DataFrame(rows))
    return pd.DataFrame(rows), ref
