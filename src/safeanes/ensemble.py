"""Fixed-weight logit ensemble with independently held-out calibration patients."""

from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.special import expit
import torch

from .data import write_json
from .evaluation import bootstrap_ci, evaluate_predictions, quality_gates, select_threshold, window_metrics
from .experiment import pilot_roles
from .models import NumericModel
from .sequences import SequenceStore, WindowDataset, file_hash, load_dataset
from .training import TrainConfig, atomic_checkpoint, calibrate, fit_calibration, predict


def blend_logits(values, weights):
    values, weights = np.asarray(values), np.asarray(weights, dtype=float)
    if (values.ndim != 3 or values.shape[0] != len(weights) or values.shape[-1] != 2
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.isclose(weights.sum(), 1)):
        raise ValueError("Expected finite MxNx2 logits and nonnegative weights summing to one")
    if (values[:, :, 1] < values[:, :, 0]).any():
        raise ValueError("Every component must preserve horizon order")
    return np.tensordot(weights, values, axes=(0, 0))


def run_ensemble(dataset, sequences, runs, out, weights=None, repeats=200, threads=4):
    """No weight search: coefficients are supplied/frozen before test inference."""
    if len(runs) < 2 or len({str(Path(r).resolve()) for r in runs}) != len(runs):
        raise ValueError("At least two distinct component runs required")
    if repeats < 1 or threads < 1:
        raise ValueError("Positive bootstrap repeats and threads required")
    weights = [1 / len(runs)] * len(runs) if weights is None else list(weights)
    blend_logits(np.zeros((len(runs), 0, 2)), weights)
    dataset, out = Path(dataset), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Ensemble output must be new/empty")
    meta, protocol, manifest, windows = load_dataset(dataset)
    store = SequenceStore(sequences, dataset)
    roles = pilot_roles(manifest, protocol.seed)
    frames = windows.merge(roles, on=["caseid", "subjectid"], validate="many_to_one")
    fit_subjects = sorted(map(int, frames.loc[frames.role.eq("fit"), "subjectid"].unique()))
    parts = {role: frames[frames.role.eq(role)].copy() for role in ("calibration", "validation", "pilot_test")}
    eligible = {role: frame[frame.eligible] for role, frame in parts.items()}
    torch.set_num_threads(threads)
    logits = {role: [] for role in parts}
    components, component_bundles, inference = [], [], {}
    for run in map(Path, runs):
        tick = time.perf_counter()
        report = json.loads((run / "report.json").read_text(encoding="utf-8"))
        if report.get("errors") or report.get("scope") != "exploratory_pilot_not_final_test":
            raise ValueError(f"Incomplete or incompatible component: {run}")
        bundle = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
        identity = bundle["identity"]
        if (identity["dataset_hash"] != meta["windows_sha256"]
                or identity["protocol_hash"] != protocol.digest()
                or identity["sequence_hash"] != file_hash(Path(sequences) / "sequences.json")
                or identity["roles"] != roles.to_dict("list")
                or bundle["normalizer"]["fit_subjectids"] != fit_subjects):
            raise ValueError(f"Component data/split/normalizer mismatch: {run}")
        config = TrainConfig(**identity["config"])
        model = NumericModel(**bundle["model_args"])
        model.load_state_dict(bundle["model"])
        for role, frame in eligible.items():
            samples = WindowDataset(store, frame, bundle["normalizer"], config.feature_set, config.masks)
            logits[role].append(predict(model, samples, config, torch.device("cpu")))
        name = run.as_posix()
        inference[name] = time.perf_counter() - tick
        components.append({"run": name, "checkpoint_sha256": file_hash(run / "model.pt"),
                           "architecture": config.architecture, "seed": config.seed})
        component_bundles.append({"model": bundle["model"], "model_args": bundle["model_args"],
            "normalizer": bundle["normalizer"], "feature_set": config.feature_set, "masks": config.masks})
    blended = {role: blend_logits(value, weights) for role, value in logits.items()}
    calibration = fit_calibration(blended["calibration"], eligible["calibration"][["y_300", "y_600"]].to_numpy())
    for role, frame in parts.items():
        p = calibrate(blended[role], calibration)
        for i, h in enumerate(protocol.horizons_seconds):
            frame[f"p_{h}"] = np.nan
            frame[f"raw_p_{h}"] = np.nan
            frame.loc[frame.eligible, f"p_{h}"] = p[:, i]
            frame.loc[frame.eligible, f"raw_p_{h}"] = expit(blended[role][:, i])
    out.mkdir(parents=True, exist_ok=True)
    roles.to_csv(out / "pilot_roles.csv", index=False)
    write_json(out / "config.json", {"method": "fixed_logit_mean", "weights": weights, "components": components})
    write_json(out / "environment.json", {"dataset_hash": meta["windows_sha256"], "protocol": asdict(protocol),
        "device": "cpu", "device_name": platform.processor(), "torch": str(torch.__version__),
        "source_hashes": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
        "scope": "exploratory_pilot_not_final_test"})
    events = pd.read_csv(dataset / "events.csv")
    thresholds = {}
    result = {"scope": "exploratory_pilot_not_final_test", "models": {}, "errors": {},
        "calibration": calibration, "components": components, "weights": weights,
        "inference_seconds_by_component": inference, "monotonicity_violations": 0}
    for h in protocol.horizons_seconds:
        key = f"ensemble_{h}"
        val = parts["validation"].assign(probability=parts["validation"][f"p_{h}"])
        test = parts["pilot_test"].assign(probability=parts["pilot_test"][f"p_{h}"])
        threshold, selection = select_threshold(val, events, h, protocol)
        thresholds[str(h)] = threshold
        metrics, cases, alarms = evaluate_predictions(test, events, h, threshold, protocol)
        known = test.eligible & test[f"y_{h}"].ge(0)
        result["models"][key] = {"selection_on_validation": selection, "pilot_test": metrics,
            "gates": quality_gates(metrics, h), "ci95": bootstrap_ci(test, events, h, threshold, protocol, repeats),
            "raw_window_metrics": window_metrics(test.loc[known, f"y_{h}"], test.loc[known, f"raw_p_{h}"])}
        test.to_csv(out / f"{key}_predictions.csv.gz", index=False)
        val.to_csv(out / f"{key}_validation_predictions.csv.gz", index=False)
        cases.to_csv(out / f"{key}_case_metrics.csv", index=False)
        write_json(out / f"{key}_alarms.json", alarms)
    atomic_checkpoint(out / "ensemble.pt", {"components": component_bundles, "weights": weights,
        "calibration": calibration, "thresholds": thresholds, "protocol": asdict(protocol),
        "dataset_hash": meta["windows_sha256"], "component_provenance": components})
    write_json(out / "report.json", result)
    return result
