"""E08 methods: shared preprocessing, per-method calibration, thresholds chosen on CALIBRATION."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Protocol
from .evaluation import evaluate_predictions, quality_gates
from .tabular_sota import log_odds

HORIZONS = (300, 600)
MODEL_SEED = 20260917
AUGMENT_SEED = 20260919
AUGMENT_COPIES = 3
JITTER_SCALE = .1
SMOTE_NEIGHBOURS = 5
MAP_FEATURES = ("map_current", "map_300_slope", "map_300_std")
MONOTONE_FEATURES = ("map_current", "map_60_mean", "map_300_mean")
LIGHTGBM_PARAMS = {"n_estimators": 500, "num_leaves": 7, "learning_rate": .03,
                   "min_child_samples": 150, "reg_lambda": 10}
CATBOOST_PARAMS = {"iterations": 300, "depth": 5, "learning_rate": .05, "l2_leaf_reg": 5}
LOGISTIC_PARAMS = {"C": 1., "max_iter": 2000}
CALIBRATOR_PARAMS = {"C": 1e6, "max_iter": 2000}
TABM_EXPECTED = {"k": 16, "width": 64, "epochs": 30, "patience": 6, "batch_size": 512}
TABM_FIXED = "n_blocks=2, dropout=0.1, PLE 8 bins x 4, AdamW lr=0.002 wd=3e-4"  # hardcoded in TabMRisk.fit
TABM_MODES = ("frozen", "retrain", "skip")
META = ["caseid", "subjectid", "time", "eligible", "exposure_seconds", "y_300", "y_600"]
ROLES = ("fit", "calibration", "validation")
DATASETS = ("development300", "full")
# Full VitalDB, grouped by subject in E07; the global test group is never read.
FULL_ROLES = {"development_seen": "fit", "unseen_train": "fit",
              "unseen_calibration": "calibration", "unseen_validation": "validation"}
LOCKED_GROUP = "unseen_test"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    description: str
    weighting: str = "none"
    augmentation: str = "none"
    tabm_seed: int = None


METHODS = (
    MethodSpec("map_logistic", "baseline", "Logistic trên 3 đặc trưng MAP"),
    MethodSpec("lgbm_plain", "lightgbm", "LightGBM gốc"),
    MethodSpec("lgbm_class_weight", "lightgbm", "LightGBM + trọng số lớp", weighting="balanced"),
    MethodSpec("lgbm_scale_pos_20x", "lightgbm", "LightGBM + scale_pos_weight", weighting="scale_pos_20x"),
    MethodSpec("lgbm_oversample_3x", "lightgbm", "LightGBM + nhân bản window dương", augmentation="oversample"),
    MethodSpec("lgbm_smote_3x", "lightgbm", "LightGBM + SMOTE", augmentation="smote"),
    MethodSpec("lgbm_jitter_3x", "lightgbm", "LightGBM + nhiễu Gaussian", augmentation="jitter"),
    MethodSpec("catboost_plain", "catboost", "CatBoost gốc"),
    MethodSpec("catboost_balanced", "catboost", "CatBoost + trọng số lớp", weighting="balanced"),
    MethodSpec("catboost_balanced_jitter", "catboost", "CatBoost + trọng số lớp + nhiễu Gaussian",
               weighting="balanced", augmentation="jitter"),
    *(MethodSpec(f"tabm_{seed}", "tabm", f"TabM seed {seed}", tabm_seed=seed)
      for seed in (20260917, 20260918, 20260919)),
)
ENSEMBLE_NAME = "ensemble_diverse"
ENSEMBLE_MEMBERS = ("map_logistic", "lgbm_jitter_3x", "catboost_balanced_jitter",
                    "tabm_20260917", "tabm_20260918", "tabm_20260919")


@dataclass
class Context:
    root: Path
    dataset: str
    protocol: Protocol
    events: pd.DataFrame
    features: list
    monotone: list
    map_index: np.ndarray
    fit_rows: pd.DataFrame
    cal_rows: pd.DataFrame
    val_rows: pd.DataFrame
    X_fit: np.ndarray
    X_cal: np.ndarray
    X_val: np.ndarray

    def fit_matrix(self, horizon):
        known = self.fit_rows[f"y_{horizon}"].ge(0).to_numpy()
        return self.X_fit[known], self.fit_rows.loc[known, f"y_{horizon}"].astype(int).to_numpy()

    def calibration_mask(self, horizon):
        return (self.cal_rows.eligible & self.cal_rows[f"y_{horizon}"].ge(0)).to_numpy()


def load_context(root, dataset="development300"):
    root = Path(root)
    meta = json.loads((root / "data/development300/dataset.json").read_text())
    protocol = Protocol(**{**meta["protocol"], "horizons_seconds": tuple(meta["protocol"]["horizons_seconds"])})
    features = [f for f in meta["features"] if not f.startswith("static_")]
    if dataset == "development300":
        rows, X, events = _development300(root, features)
    elif dataset == "full":
        rows, X, events = _full_vitaldb(root, features)
    else:
        raise ValueError(f"Unknown dataset {dataset}; expected one of {DATASETS}")

    medians = _fit_medians(X["fit"])
    for matrix in X.values():
        _fill_missing(matrix, medians)
    return Context(root=root, dataset=dataset, protocol=protocol, events=events, features=features,
                   monotone=[-1 if f in MONOTONE_FEATURES else 0 for f in features],
                   map_index=np.array([features.index(f) for f in MAP_FEATURES]),
                   fit_rows=rows["fit"], cal_rows=rows["calibration"], val_rows=rows["validation"],
                   X_fit=X["fit"], X_cal=X["calibration"], X_val=X["validation"])


def _usable_for_fit(frame):
    return frame[frame.eligible & frame[["y_300", "y_600"]].ge(0).any(axis=1)]


def _development300(root, features):
    data = root / "data/development300"
    roles = pd.read_csv(root / "artifacts/E05/roles.csv")
    frame = pd.read_csv(data / "windows.csv.gz").merge(roles, on=["caseid", "subjectid"], validate="many_to_one")
    rows, X = {}, {}
    for role in ROLES:
        part = frame[frame.role.eq(role)]
        if role == "fit":
            part = _usable_for_fit(part)
        rows[role], X[role] = part[META], part[features].to_numpy(dtype=float, copy=True)
    return rows, X, pd.read_csv(data / "events.csv")


def _full_vitaldb(root, features):
    """Read case by case into numpy so peak memory stays ~2x the FIT matrix, not ~5x."""
    import joblib
    manifest = pd.read_csv(root / "reports/E07/cohort_manifest.csv")
    manifest = manifest[manifest.eligible]
    meta_parts, x_parts, events = {r: [] for r in ROLES}, {r: [] for r in ROLES}, []
    for group, role in FULL_ROLES.items():
        for caseid in manifest.loc[manifest.evaluation_group.eq(group), "caseid"]:
            case = joblib.load(root / "data/vitaldb_full/csv_cases" / f"{caseid}.joblib")
            frame = _usable_for_fit(case["frame"]) if role == "fit" else case["frame"]
            meta_parts[role].append(frame[META])
            x_parts[role].append(frame[features].to_numpy(dtype=float))
            events += case["events"]
    rows, X = {}, {}
    for role in ROLES:
        rows[role] = pd.concat(meta_parts.pop(role), ignore_index=True)
        X[role] = np.concatenate(x_parts.pop(role))

    locked = set(manifest.loc[manifest.evaluation_group.eq(LOCKED_GROUP), "caseid"])
    for role, part in rows.items():
        if locked & set(part.caseid):
            raise RuntimeError(f"Global test cases leaked into {role}")
    subjects = {role: set(part.subjectid) for role, part in rows.items()}
    for a, b in (("fit", "calibration"), ("fit", "validation"), ("calibration", "validation")):
        if subjects[a] & subjects[b]:
            raise RuntimeError(f"Patients shared between {a} and {b}")
    return rows, X, pd.DataFrame(events)


def _fit_medians(X):
    """Column-wise median on FIT; all-missing columns become 0 like SimpleImputer(keep_empty_features)."""
    medians = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        present = X[:, j][~np.isnan(X[:, j])]
        if len(present):
            medians[j] = np.median(present)
    return medians


def _fill_missing(X, medians):
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = medians[j]


def augment(kind, X, y, rng):
    """Add AUGMENT_COPIES synthetic rows per positive window, whatever the kind."""
    if kind == "none":
        return X, y
    pos = np.where(y == 1)[0]
    if not len(pos):
        return X, y
    if kind == "oversample":
        extra = X[np.tile(pos, AUGMENT_COPIES)]
    elif kind == "jitter":
        std = X.std(axis=0)
        extra = np.concatenate([X[pos] + rng.normal(0, JITTER_SCALE * std, size=(len(pos), X.shape[1]))
                                for _ in range(AUGMENT_COPIES)])
    elif kind == "smote":
        extra = _smote(X, pos, rng)
    else:
        raise ValueError(f"Unknown augmentation {kind}")
    return np.concatenate([X, extra]), np.concatenate([y, np.ones(len(extra))])


def _smote(X, pos, rng):
    k = min(SMOTE_NEIGHBOURS, len(pos) - 1)
    if k < 1:
        return np.empty((0, X.shape[1]))
    scaler = StandardScaler().fit(X)
    pos_scaled = scaler.transform(X[pos])
    _, neighbours = NearestNeighbors(n_neighbors=k + 1).fit(pos_scaled).kneighbors(pos_scaled)
    synthetic = []
    for _ in range(AUGMENT_COPIES):
        picked = neighbours[np.arange(len(pos)), rng.integers(1, k + 1, len(pos))]
        alpha = rng.uniform(0, 1, (len(pos), 1))
        synthetic.append(scaler.inverse_transform(pos_scaled + alpha * (pos_scaled[picked] - pos_scaled)))
    return np.concatenate(synthetic)


def class_ratio(ctx, horizon):
    _, y = ctx.fit_matrix(horizon)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def fit_predict(ctx, spec, horizon, tabm_mode="frozen", model_dir=None):
    """Return raw scores on CALIBRATION and VALIDATION plus the parameters actually used."""
    if spec.family == "tabm":
        return _tabm(ctx, spec, horizon, tabm_mode, model_dir)
    X, y = ctx.fit_matrix(horizon)
    if spec.family == "baseline":
        model = make_pipeline(StandardScaler(), LogisticRegression(**LOGISTIC_PARAMS))
        model.fit(X[:, ctx.map_index], y)
        return (model.predict_proba(ctx.X_cal[:, ctx.map_index])[:, 1],
                model.predict_proba(ctx.X_val[:, ctx.map_index])[:, 1],
                {**LOGISTIC_PARAMS, "n_train": len(y), "n_positive": int(y.sum())})

    X, y = augment(spec.augmentation, X, y, np.random.default_rng(AUGMENT_SEED))
    if spec.family == "lightgbm":
        from lightgbm import LGBMClassifier
        params = dict(LIGHTGBM_PARAMS)
        if spec.weighting == "balanced":
            params["class_weight"] = "balanced"
        elif spec.weighting == "scale_pos_20x":
            params["scale_pos_weight"] = class_ratio(ctx, horizon) * 20
        model = LGBMClassifier(**params, monotone_constraints=ctx.monotone, random_state=MODEL_SEED,
                               n_jobs=2, verbosity=-1)
    elif spec.family == "catboost":
        from catboost import CatBoostClassifier
        params = {**CATBOOST_PARAMS,
                  "auto_class_weights": "Balanced" if spec.weighting == "balanced" else "None"}
        model = CatBoostClassifier(**params, random_seed=MODEL_SEED, thread_count=2, verbose=False,
                                   allow_writing_files=False)
    else:
        raise ValueError(f"Unknown family {spec.family}")
    model.fit(X, y)
    return (model.predict_proba(ctx.X_cal)[:, 1], model.predict_proba(ctx.X_val)[:, 1],
            {**params, "n_train": len(y), "n_positive": int(y.sum())})


def _tabm(ctx, spec, horizon, mode, model_dir):
    import joblib
    if mode == "frozen":
        bundle = joblib.load(ctx.root / "artifacts/E06" / f"tabm_{spec.tabm_seed}" / "bundle.joblib")
        if list(bundle["features"]) != ctx.features:
            raise ValueError(f"{spec.name}: frozen E06 features differ from the E08 feature list")
        model = bundle["model"]
    elif mode == "retrain":
        path = None if model_dir is None else Path(model_dir) / f"{spec.name}.joblib"
        if path is not None and path.exists():
            model = joblib.load(path)
        else:
            from .tabular_sota import TabMRisk
            labels = ctx.fit_rows[["y_300", "y_600"]].to_numpy(float)
            model = TabMRisk(seed=spec.tabm_seed, **TABM_EXPECTED).fit(
                ctx.X_fit, labels, ctx.fit_rows.subjectid.to_numpy())
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, path)
    else:
        raise ValueError(f"TabM mode {mode} cannot produce scores")
    actual = {key: getattr(model, key) for key in TABM_EXPECTED}
    if actual != TABM_EXPECTED:
        raise ValueError(f"{spec.name}: hyperparameters {actual} != {TABM_EXPECTED}")
    column = HORIZONS.index(horizon)
    return (model.predict_risk(ctx.X_cal)[:, column], model.predict_risk(ctx.X_val)[:, column],
            {**actual, "best_epoch": model.best_epoch, "n_train": None, "n_positive": None})


def calibrate_member(raw_cal, raw_val, y_cal, mask_cal):
    calibrator = make_pipeline(StandardScaler(), LogisticRegression(**CALIBRATOR_PARAMS))
    calibrator.fit(log_odds(raw_cal[mask_cal]).reshape(-1, 1), y_cal.astype(int))
    transform = lambda raw: calibrator.predict_proba(log_odds(raw).reshape(-1, 1))[:, 1]
    return transform(raw_cal), transform(raw_val)


def frame_with_probability(rows, probability):
    out = rows[META].copy().reset_index(drop=True)
    p = np.asarray(probability, float).copy()
    p[~out.eligible.to_numpy()] = np.nan
    out["probability"] = p
    return out


def threshold_grid(scores):
    finite = np.asarray(scores)[np.isfinite(scores)]
    fixed = np.linspace(.01, .99, 40)
    if not len(finite):
        return np.r_[fixed, 1.000001]
    return np.unique(np.r_[fixed, np.quantile(finite, np.linspace(0, 1, 201)), 1.000001])


def sweep(frame, events, horizon, protocol, grid):
    rows = []
    for threshold in grid:
        metric, _, _ = evaluate_predictions(frame, events, horizon, threshold, protocol)
        gates = quality_gates(metric, horizon)
        rows.append({"threshold": float(threshold), "gates_met": sum(gates["criteria"].values()),
                     "gates_total": len(gates["criteria"]),
                     **{k: metric[k] for k in ("auroc", "average_precision", "ece", "event_sensitivity",
                                               "alarm_ppv", "false_alarms_per_hour", "prediction_coverage",
                                               "events_detected", "events_eligible")}})
    return pd.DataFrame(rows)


def select_threshold(curve, policy, fa_budget=.5):
    usable = curve[curve.event_sensitivity.notna()]
    if not len(usable):
        raise ValueError("Empty threshold curve")
    if policy == "recall_first":
        ranked = usable.sort_values(by=["event_sensitivity", "alarm_ppv", "false_alarms_per_hour"],
                                    ascending=[False, False, True])
        return ranked.iloc[0], "recall_first"
    if policy == "fa_budget":
        under = usable[usable.false_alarms_per_hour.le(fa_budget)]
        if len(under):
            ranked = under.sort_values(by=["event_sensitivity", "alarm_ppv"], ascending=[False, False])
            return ranked.iloc[0], "max_recall_within_fa_budget"
        return usable.sort_values("false_alarms_per_hour").iloc[0], "no_threshold_under_budget"
    raise ValueError(f"Unknown policy {policy}")


def pareto_front(curve):
    """Rows not dominated on (recall up, PPV up, FA/hour down)."""
    r = curve.event_sensitivity.fillna(0).to_numpy()
    p = curve.alarm_ppv.fillna(0).to_numpy()
    f = curve.false_alarms_per_hour.fillna(np.inf).to_numpy()
    no_worse = (r[None, :] >= r[:, None]) & (p[None, :] >= p[:, None]) & (f[None, :] <= f[:, None])
    better = (r[None, :] > r[:, None]) | (p[None, :] > p[:, None]) | (f[None, :] < f[:, None])
    return curve[~(no_worse & better).any(axis=1)].sort_values("event_sensitivity", ascending=False)
