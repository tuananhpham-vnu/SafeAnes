"""Baselines and LightGBM (plan section 6).

`TabularData` holds the labels and features of one window W for fit /
calibration / validation (main cohort). `run_combo` trains one (model, W, h),
calibrates it, picks the threshold on validation, evaluates with bootstrap CIs
and writes every output of that combination into its own files (plan 6.3):

  <work>/artifacts/tabular/<model>/W<W>/h<h>/   (map_threshold: .../map_threshold/h<h>/)
      model.joblib threshold.json val_predictions.parquet case_metrics.parquet provenance.json done.json
  <work>/reports/tabular_validation/<name>.csv   one row, schema 8.6
  <work>/reports/calibration_check/<name>.csv    one row
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .bootstrap import Bootstrap, ci
from .calibration import Platt, logit
from .columns import numeric_cols, window_cols
from .io import write_json, write_parquet
from .metrics import Evaluator, lead_quantiles, row_metrics, summarize_counts
from .thresholds import candidates, choose_threshold

BASE_MODELS = ("map_threshold", "map_logistic", "lgbm_numeric", "lgbm_wave")  # plan section 6 (NB02)
# Added for samples v3, outside the config so the plan's config digest stays unchanged:
# lgbm_context = lgbm_numeric + case-context columns of features_v5 (same LightGBM settings);
# catboost_numeric = the E08 CatBoost (scripts/e08) on the 66 numeric columns.
EXTRA_VERSIONS = {"lgbm_context": ("numeric", "context"), "catboost_numeric": ("numeric",)}
CATBOOST_PARAMS = {"iterations": 300, "depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 5}  # as E08
MODELS = (*BASE_MODELS, *EXTRA_VERSIONS)
SPLITS = ("train", "calibration", "validation")
RESULT_COLUMNS = ["model", "window", "horizon", "status", "n_rows", "n_cases", "prevalence", "auroc", "auroc_lo",
                  "auroc_hi", "auprc", "auprc_lo", "auprc_hi", "brier", "ece", "threshold", "budget_not_met",
                  "event_sensitivity", "event_sensitivity_lo", "event_sensitivity_hi", "events_detected",
                  "events_eligible", "event_sensitivity_all", "early_sensitivity", "false_alarms_per_hour",
                  "false_alarms_per_hour_lo", "false_alarms_per_hour_hi", "alarm_ppv", "lead_median_s", "lead_q25_s",
                  "lead_q75_s", "prediction_coverage", "alarm_time_fraction", "alarm_time_fraction_lo",
                  "alarm_time_fraction_hi", "alarm_rearm", "config_digest", "git_commit"]


def combo_name(model: str, W: int, h: int) -> str:
    return f"map_threshold_h{h}" if model == "map_threshold" else f"{model}_W{W}_h{h}"


def combo_dir(model: str, W: int, h: int) -> str:
    return f"tabular/map_threshold/h{h}" if model == "map_threshold" else f"tabular/{model}/W{W}/h{h}"


def model_columns(cfg, model: str, W: int) -> list[str]:
    if model == "map_threshold":
        return ["map_current"]
    if model == "map_logistic":
        return [c.replace("{W}", str(W)) for c in cfg.baselines.map_logistic["features"]]
    groups = EXTRA_VERSIONS[model] if model in EXTRA_VERSIONS else cfg.lightgbm.versions[model]
    return window_cols(W, groups)


@dataclass
class TabularData:
    W: int
    labels: dict[str, pd.DataFrame]
    features: dict[str, pd.DataFrame]
    events: pd.DataFrame
    evaluators: dict[int, Evaluator] = field(default_factory=dict)

    def evaluator(self, h: int, ev_cfg) -> Evaluator:
        if h not in self.evaluators:
            self.evaluators[h] = Evaluator(self.labels["validation"], self.events[self.events.split == "validation"],
                                           h, persistence=ev_cfg.alarm_persistence,
                                           cooldown=ev_cfg.alarm_cooldown_seconds, early_lead=ev_cfg.early_lead_seconds,
                                           rearm=ev_cfg.alarm_rearm)
        return self.evaluators[h]


# ------------------------------------------------------------------ models

class MapThreshold:
    """Score = -map_current (no fit)."""
    columns = ["map_current"]

    def fit(self, X, y):
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return -X["map_current"].to_numpy(float)


class MapLogistic:
    def __init__(self, columns, C=1.0, max_iter=2000):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.columns, self.scaler, self.lr = list(columns), StandardScaler(), LogisticRegression(C=C, max_iter=max_iter)
        self.medians: dict[str, float] = {}

    def _prep(self, X):
        return X[self.columns].astype(float).fillna(self.medians).to_numpy()

    def fit(self, X, y):
        self.medians = {c: float(X[c].median()) for c in self.columns}  # train medians
        self.lr.fit(self.scaler.fit_transform(self._prep(X)), y)
        return self

    def score(self, X):
        return logit(self.lr.predict_proba(self.scaler.transform(self._prep(X)))[:, 1])


class LGBM:
    def __init__(self, columns, params, monotone_decreasing, W, seed, extra=None):
        import lightgbm as lgb
        self.columns = list(columns)
        dec = {c.replace("{W}", str(W)) for c in monotone_decreasing}
        self.monotone = [-1 if c in dec else 0 for c in self.columns]
        self.model = lgb.LGBMClassifier(**dict(params), **dict(extra or {}), random_state=seed,
                                        monotone_constraints=self.monotone)

    def fit(self, X, y):
        self.model.fit(X[self.columns], y)
        return self

    def score(self, X):
        return logit(self.model.predict_proba(X[self.columns])[:, 1])


class CatBoost:
    def __init__(self, columns, seed, threads=4):
        from catboost import CatBoostClassifier
        self.columns = list(columns)
        self.model = CatBoostClassifier(**CATBOOST_PARAMS, random_seed=seed, thread_count=threads, verbose=False,
                                        allow_writing_files=False)  # no catboost_info/ in the working directory

    def fit(self, X, y):
        self.model.fit(X[self.columns], y)
        return self

    def score(self, X):
        return logit(self.model.predict_proba(X[self.columns])[:, 1])


def make_model(cfg, model: str, W: int, columns=None, seed=None, extra=None):
    cols = columns or model_columns(cfg, model, W)
    if model == "map_threshold":
        return MapThreshold()
    if model == "map_logistic":
        b = cfg.baselines.map_logistic
        return MapLogistic(cols, C=b["C"], max_iter=b["max_iter"])
    if model.startswith("catboost"):
        return CatBoost(cols, cfg.seed if seed is None else seed)
    return LGBM(cols, cfg.lightgbm.params, cfg.lightgbm.monotone_decreasing, W, cfg.seed if seed is None else seed,
                extra)


# ------------------------------------------------------------------ one combination

def fit_rows(labels: pd.DataFrame, h: int) -> np.ndarray:
    y = labels[f"y_{h}"].to_numpy()
    return labels["eligible"].to_numpy(bool) & ((y == 0) | (y == 1))


def fit_and_calibrate(cfg, model: str, data: TabularData, h: int, columns=None, seed=None, extra=None):
    """Returns (model, platt, status). status is 'ok' or 'skipped_no_positive'."""
    tr, fx = data.labels["train"], data.features["train"]
    m = fit_rows(tr, h)
    y = tr[f"y_{h}"].to_numpy()[m]
    if len(np.unique(y)) < 2:
        return None, None, "skipped_no_positive"
    est = make_model(cfg, model, data.W, columns, seed, extra).fit(fx[m], y)
    ca, fc = data.labels["calibration"], data.features["calibration"]
    mc = fit_rows(ca, h)
    s_cal = est.score(fc[mc])
    y_cal = ca[f"y_{h}"].to_numpy()[mc]
    ok = np.isfinite(s_cal)
    if len(np.unique(y_cal[ok])) < 2:
        return est, None, "skipped_no_positive"
    return est, Platt().fit(s_cal[ok], y_cal[ok]), "ok"


def predict_split(est, platt, labels: pd.DataFrame, feats: pd.DataFrame) -> np.ndarray:
    """Calibrated probability for every row; NaN where not eligible."""
    p = platt.predict(est.score(feats))
    return np.where(labels["eligible"].to_numpy(bool), p, np.nan)


def evaluate_probs(cfg, data: TabularData, h: int, prob_val: np.ndarray, bootstrap: int, seed: int):
    """Threshold on validation, metrics and bootstrap CIs. Returns (summary, threshold_info, evaluation, ev)."""
    E = cfg.evaluation
    ev = data.evaluator(h, E)
    p = ev.sort(prob_val)
    elig_p = p[ev.eligible & np.isfinite(p)]
    thr = choose_threshold(candidates(elig_p, E.threshold_candidates), lambda t: ev.quick(p, t),
                           budget=E.fa_per_hour_budget, curve_points=E.curve_points)
    res = ev.evaluate(p, thr["threshold"])
    rowsel = ev.eligible
    summary = {**row_metrics(ev.y[rowsel], p[rowsel], bins=E.ece_bins),
               **summarize_counts(res.case_table.sum(numeric_only=True)), **lead_quantiles(res),
               "threshold": thr["threshold"], "budget_not_met": thr["budget_not_met"],
               "n_cases": int(len(ev.cases))}
    if bootstrap:
        subj_rows = data.labels["validation"]["subjectid"].to_numpy()[ev.order][rowsel]
        bs = Bootstrap(subj_rows, res.case_table["subjectid"].to_numpy(), bootstrap, seed)
        summary.update(ci(bs.metrics(res.case_table, ev.y[rowsel], p[rowsel])))
    return summary, thr, res, ev


def run_combo(cfg, model: str, data: TabularData, h: int, work: Path, *, bootstrap: int, provenance: dict,
              log=print, fitted=None) -> tuple[dict, list[str]]:
    """Train, calibrate, choose threshold, evaluate, write. Returns (result row, written paths).
    `fitted` = (estimator with .score, Platt, status) skips training (models trained elsewhere, e.g. TabM)."""
    t0 = time.time()
    W = data.W
    name, rel = combo_name(model, W, h), combo_dir(model, W, h)
    out = Path(work) / "artifacts" / rel
    rep_v = Path(work) / "reports" / "tabular_validation" / f"{name}.csv"
    rep_c = Path(work) / "reports" / "calibration_check" / f"{name}.csv"
    row = {"model": model, "window": None if model == "map_threshold" else W, "horizon": h,
           "alarm_rearm": cfg.evaluation.alarm_rearm,
           "config_digest": provenance["config_digest"], "git_commit": provenance["git"]["commit"]}
    est, platt, status = fitted or fit_and_calibrate(cfg, model, data, h)
    if status != "ok":
        row["status"] = status
        log(f"  {name}: {status}")
    else:
        prob = predict_split(est, platt, data.labels["validation"], data.features["validation"])
        summary, thr, res, ev = evaluate_probs(cfg, data, h, prob, bootstrap, cfg.seed)
        row.update(summary, status="ok")
        row["n_rows"] = summary["n_rows"]
        cols = getattr(est, "columns", [])
        joblib.dump({"model": est, "calibrator": platt.to_dict(), "columns": cols, "window": W, "horizon": h,
                     "name": name, "medians": getattr(est, "medians", None),
                     "config_digest": provenance["config_digest"]}, _mk(out) / "model.joblib")
        write_json(out / "threshold.json", {k: v for k, v in thr.items()})
        lab = data.labels["validation"]
        write_parquet(out / "val_predictions.parquet", pd.DataFrame({
            "caseid": lab["caseid"], "subjectid": lab["subjectid"], "time": lab["time"], "eligible": lab["eligible"],
            "exposure_seconds": lab["exposure_seconds"], f"y_{h}": lab[f"y_{h}"], "probability": prob}))
        write_parquet(out / "case_metrics.parquet", res.case_table)
        # calibration check on validation (eligible, known label)
        y = ev.y
        p_sorted = ev.sort(prob)
        known = ev.eligible & (y != -1) & np.isfinite(p_sorted)
        _write_csv(rep_c, pd.DataFrame([{"model": model, "window": row["window"], "horizon": h,
                                         "mean_p_validation": float(p_sorted[known].mean()),
                                         "positive_rate_validation": float((y[known] == 1).mean()),
                                         "calibrator": str(platt.to_dict())}]))
    write_json(_mk(out) / "provenance.json", {**provenance, "combo": name})
    _write_csv(rep_v, pd.DataFrame([{c: row.get(c) for c in RESULT_COLUMNS}]))
    write_json(out / "done.json", {"combo": name, "status": row["status"], "seconds": round(time.time() - t0, 1)})
    written = [rel, f"reports/tabular_validation/{name}.csv"]
    if row["status"] == "ok":
        written.append(f"reports/calibration_check/{name}.csv")
    return row, written


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    from .io import atomic_path
    with atomic_path(path) as tmp:
        df.to_csv(tmp, index=False)
