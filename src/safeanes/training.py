"""Reproducible sequence pilot: fit -> validation checkpoint -> calibration -> alarms.

Global test is never loaded. CUDA runs use FP16 AMP; CPU runs are FP32.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch.utils.data import DataLoader, Subset

from .data import write_json
from .evaluation import bootstrap_ci, evaluate_predictions, quality_gates, select_threshold, window_metrics
from .experiment import pilot_roles
from .models import NumericModel, masked_bce
from .sequences import SequenceStore, WindowDataset, file_hash, fit_normalizer, load_dataset


@dataclass(frozen=True)
class TrainConfig:
    architecture: str = "tcn"
    feature_set: str = "numeric_static"
    masks: bool = True
    width: int = 32
    dropout: float = .1
    patch: int = 10
    layers: int = 2
    batch_size: int = 64
    epochs: int = 30
    patience: int = 5
    learning_rate: float = .001
    weight_decay: float = .0001
    seed: int = 20260917
    max_windows: int | None = None
    device: str = "auto"
    threads: int = 4

    def __post_init__(self):
        if self.architecture not in ("tcn", "transformer", "inception", "timesnet"):
            raise ValueError("Unknown architecture")
        if self.feature_set not in ("map", "numeric", "numeric_static"):
            raise ValueError("Unknown feature set")
        if any(getattr(self, k) < 1 for k in ("width", "patch", "layers", "batch_size", "epochs", "patience", "threads")):
            raise ValueError("Training dimensions and limits must be positive")
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimization settings")
        if self.max_windows is not None and self.max_windows < 1:
            raise ValueError("max_windows must be positive")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")


def fit_calibration(logits, labels):
    """One positive temperature and common bias preserve ordered horizon risks."""
    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    known = labels >= 0
    if not all(len(np.unique(labels[known[:, h], h])) == 2 for h in range(labels.shape[1])):
        raise ValueError("Calibration needs both classes at each horizon; expand pilot without outcome-based reshuffling")
    def objective(theta):
        score = logits[known] / np.exp(theta[0]) + theta[1]
        return np.mean(np.logaddexp(0, score) - labels[known] * score)
    fit = minimize(objective, [0., 0.], method="L-BFGS-B", bounds=[(-4, 4), (-20, 20)])
    if not fit.success or not np.isfinite(fit.fun):
        raise ValueError(f"Calibration failed: {fit.message}")
    return {"temperature": float(np.exp(fit.x[0])), "bias": float(fit.x[1]),
            "method": "shared_positive_temperature_and_bias", "calibration_bce": float(fit.fun)}


def calibrate(logits, calibration):
    return expit(logits / calibration["temperature"] + calibration["bias"])


def predict(model, dataset, config, device):
    model.eval()
    values = []
    with torch.inference_mode():
        for x, static, _ in DataLoader(dataset, batch_size=config.batch_size, shuffle=False):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                values.append(model(x.to(device), static.to(device)).cpu().numpy())
    return np.concatenate(values) if values else np.empty((0, 2), np.float32)


def atomic_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_sequence(dataset, sequences, out, config=TrainConfig(), repeats=200, resume=False):
    dataset, out = Path(dataset), Path(out)
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError("Use a new output directory or --resume")
    if resume and not (out / "last.pt").is_file():
        raise FileNotFoundError("No last.pt checkpoint to resume")
    if resume and (out / "report.json").exists():
        raise ValueError("Completed pilot run is immutable; use a new output directory")
    meta, protocol, manifest, windows = load_dataset(dataset)
    if protocol.horizons_seconds != (300, 600):
        raise ValueError("These ordered heads require horizons (300, 600)")
    store = SequenceStore(sequences, dataset)
    roles = pilot_roles(manifest, protocol.seed)  # model seeds never change patient roles
    frames = windows.merge(roles, on=["caseid", "subjectid"], validate="many_to_one")
    labels = [f"y_{h}" for h in protocol.horizons_seconds]
    subsets = {role: frames[frames.role.eq(role) & frames.eligible & frames[labels].ge(0).any(axis=1)].copy()
               for role in ("fit", "validation", "calibration")}
    for role, subset in subsets.items():
        if not len(subset) or any(subset.loc[subset[c].ge(0), c].nunique() < 2 for c in labels):
            raise ValueError(f"{role} needs both classes for both horizons; expand pilot")
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available()
                          else "cpu" if config.device == "auto" else config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; choose CPU or install CUDA-enabled PyTorch")
    normalizer = fit_normalizer(store, frames[frames.role.eq("fit")])
    sets = {role: WindowDataset(store, subset, normalizer, config.feature_set, config.masks)
            for role, subset in subsets.items()}
    model_args = {"architecture": config.architecture, "inputs": sets["fit"].channels,
        "static_dim": sets["fit"].static_dim, "length": protocol.history_seconds // protocol.numeric_step_seconds,
        "width": config.width, "dropout": config.dropout, "patch": config.patch, "layers": config.layers}
    model = NumericModel(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    source_hashes = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    identity = {"config": {k: v for k, v in asdict(config).items() if k != "epochs"},
        "dataset_hash": meta["windows_sha256"], "sequence_hash": file_hash(Path(sequences) / "sequences.json"),
        "roles": roles.to_dict("list"), "protocol_hash": protocol.digest(), "source_hashes": source_hashes}
    start_epoch, stale, best_loss, history, best_state = 0, 0, float("inf"), [], None
    if resume:
        checkpoint = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["identity"] != identity:
            raise ValueError("Resume config, source code, patient roles or data changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_epoch, stale, best_loss = checkpoint["epoch"] + 1, checkpoint["stale"], checkpoint["best_loss"]
        history, best_state = checkpoint["history"], checkpoint["best_model"]
        if config.epochs < start_epoch:
            raise ValueError("epochs cannot be lower than checkpoint epoch")
    out.mkdir(parents=True, exist_ok=True)
    roles.to_csv(out / "pilot_roles.csv", index=False)
    write_json(out / "config.json", asdict(config))
    write_json(out / "normalizer.json", normalizer)
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(),
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "cuda": torch.version.cuda, "amp": device.type == "cuda", "dataset_hash": meta["windows_sha256"],
        "protocol": asdict(protocol), "source_hashes": source_hashes, "scope": "exploratory_pilot_not_final_test"}
    write_json(out / "environment.json", environment)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, config.epochs):
        if stale >= config.patience:
            break
        tick = time.perf_counter()
        generator = torch.Generator().manual_seed(config.seed + epoch)
        train_set = sets["fit"]
        if config.max_windows is not None and len(train_set) > config.max_windows:
            train_set = Subset(train_set, torch.randperm(len(train_set), generator=generator)[:config.max_windows].tolist())
        loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, generator=generator)
        model.train()
        total, count = 0., 0
        for x, static, y in loader:
            x, static, y = x.to(device), static.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = masked_bce(model(x, static), y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; inspect inputs/checkpoint")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            known = int((y >= 0).sum())
            total += float(loss.detach()) * known
            count += known
        validation_logits = predict(model, sets["validation"], config, device)
        validation_labels = subsets["validation"][labels].to_numpy(np.float32)
        valid_loss = float(masked_bce(torch.from_numpy(validation_logits), torch.from_numpy(validation_labels)))
        if not np.isfinite(valid_loss):
            raise ValueError("Non-finite validation loss")
        if valid_loss < best_loss:
            best_loss, stale = valid_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        item = {"epoch": epoch + 1, "train_bce": total / count, "validation_bce": valid_loss,
                "seconds": time.perf_counter() - tick, "train_windows": len(train_set)}
        history.append(item)
        atomic_checkpoint(out / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "stale": stale, "best_loss": best_loss,
            "history": history, "best_model": best_state, "identity": identity,
            "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        write_json(out / "history.json", history)
        print(json.dumps(item), flush=True)
    if best_state is None:
        raise ValueError("No valid checkpoint produced")
    model.load_state_dict(best_state)
    calibration_logits = predict(model, sets["calibration"], config, device)
    calibration = fit_calibration(calibration_logits, subsets["calibration"][labels].to_numpy())
    events = pd.read_csv(dataset / "events.csv")
    predictions, inference_seconds = {}, {}
    for role in ("validation", "pilot_test"):
        frame = frames[frames.role.eq(role)].copy()
        eligible = frame[frame.eligible]
        seq = WindowDataset(store, eligible, normalizer, config.feature_set, config.masks)
        tick = time.perf_counter()
        logits = predict(model, seq, config, device)
        inference_seconds[role] = time.perf_counter() - tick
        probabilities = calibrate(logits, calibration)
        if np.any(probabilities[:, 1] < probabilities[:, 0]):
            raise ValueError("Calibrated risks are not ordered")
        for i, h in enumerate(protocol.horizons_seconds):
            frame[f"p_{h}"] = np.nan
            frame[f"raw_p_{h}"] = np.nan
            frame.loc[frame.eligible, f"p_{h}"] = probabilities[:, i]
            frame.loc[frame.eligible, f"raw_p_{h}"] = expit(logits[:, i])
        predictions[role] = frame
    thresholds = {}
    report = {"scope": "exploratory_pilot_not_final_test", "models": {}, "errors": {},
        "calibration": calibration, "monotonicity_violations": 0,
        "resources": {"training_seconds": sum(item["seconds"] for item in history),
            "parameters": sum(p.numel() for p in model.parameters()), "epochs_completed": len(history),
            "best_epoch": min(history, key=lambda row: row["validation_bce"])["epoch"],
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "inference_seconds": inference_seconds, "device": str(device)}}
    for horizon in protocol.horizons_seconds:
        key = f"{config.architecture}_{horizon}"
        validation = predictions["validation"].assign(probability=predictions["validation"][f"p_{horizon}"])
        test = predictions["pilot_test"].assign(probability=predictions["pilot_test"][f"p_{horizon}"])
        threshold, selection = select_threshold(validation, events, horizon, protocol)
        thresholds[str(horizon)] = threshold
        metrics, cases, alarms = evaluate_predictions(test, events, horizon, threshold, protocol)
        known = test.eligible & test[f"y_{horizon}"].ge(0)
        report["models"][key] = {"selection_on_validation": selection, "pilot_test": metrics,
            "raw_window_metrics": window_metrics(test.loc[known, f"y_{horizon}"], test.loc[known, f"raw_p_{horizon}"]),
            "ci95": bootstrap_ci(test, events, horizon, threshold, protocol, repeats),
            "gates": quality_gates(metrics, horizon)}
        test.to_csv(out / f"{key}_predictions.csv.gz", index=False, compression="gzip")
        cases.to_csv(out / f"{key}_case_metrics.csv", index=False)
        write_json(out / f"{key}_alarms.json", alarms)
        validation.to_csv(out / f"{key}_validation_predictions.csv.gz", index=False, compression="gzip")
    atomic_checkpoint(out / "model.pt", {"model": best_state, "model_args": model_args,
        "normalizer": normalizer, "calibration": calibration, "thresholds": thresholds,
        "protocol": asdict(protocol), "identity": identity})
    write_json(out / "report.json", report)
    return report
