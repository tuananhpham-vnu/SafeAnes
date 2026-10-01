"""Training loop for one (W, seed) and logit prediction (plan sections 7.3, 7.4).

Per run, in <work>/artifacts/dl/W<W>/seed<s>/:
  best.pt, last.pt (resume state), train_log.csv, provenance.json,
  logits_calibration.parquet, logits_validation.parquet (from best.pt), done.json
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dl_data import HORIZONS, RowSet, WaveReader, WindowDataset
from .dl_model import ConvTransformer, masked_bce
from .io import write_json, write_parquet


@dataclass
class RunSettings:
    W: int
    seed: int
    max_epochs: int
    patience: int
    samples_per_epoch: int
    batch_size: int
    lr: float
    weight_decay: float
    amp: bool
    num_workers: int
    device: str
    max_val_rows: int | None = None
    warn_only: bool = False


def seed_everything(seed: int, warn_only: bool = False) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.benchmark = False


def _worker_init(worker_id: int) -> None:
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def _loader(ds, st: RunSettings, seed: int, shuffle: bool = False) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=st.batch_size, shuffle=shuffle, num_workers=st.num_workers,
                      worker_init_fn=_worker_init, generator=g, pin_memory=st.device.startswith("cuda"),
                      persistent_workers=False)


def peak_memory() -> dict:
    out = {"ram_peak_gb": None, "vram_peak_gb": None}
    try:
        import psutil
        out["ram_peak_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    except Exception:
        pass
    if torch.cuda.is_available():
        out["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return out


def save_atomic(obj, path: Path) -> None:
    """torch.save to a temporary file, then rename: a session stopped mid-write never leaves a broken
    checkpoint (plan 2.12). HubSync skips names containing ".tmp"."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state() -> dict:
    out = {"torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
    if torch.cuda.is_available():
        out["cuda_rng"] = torch.cuda.get_rng_state_all()  # dropout on the GPU draws from these
    return out


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        cuda = state["cuda_rng"]
        if len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda)


@torch.no_grad()
def predict(model, ds, st: RunSettings) -> np.ndarray:
    model.eval()
    out = []
    for wave, tab, _ in _loader(ds, st, 0):
        with torch.autocast(device_type=st.device.split(":")[0], dtype=torch.float16,
                            enabled=st.amp and st.device.startswith("cuda")):
            out.append(model(wave.to(st.device), tab.to(st.device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(HORIZONS)), np.float32)


@torch.no_grad()
def eval_loss(model, ds, st: RunSettings) -> float:
    model.eval()
    tot = torch.zeros((), dtype=torch.float64, device=st.device)  # no GPU sync per batch
    n = torch.zeros((), dtype=torch.int64, device=st.device)
    for wave, tab, y in _loader(ds, st, 0):
        y = y.to(st.device)
        with torch.autocast(device_type=st.device.split(":")[0], dtype=torch.float16,
                            enabled=st.amp and st.device.startswith("cuda")):
            logits = model(wave.to(st.device), tab.to(st.device))
        k = (y >= 0).sum()
        tot += masked_bce(logits.float(), y).double() * k
        n += k
    return float(tot) / max(int(n), 1)


def train_one(cfg, st: RunSettings, rows: dict[str, RowSet], reader: WaveReader, out: Path, *,
              on_epoch=None, log=print) -> dict:
    """Train (resuming from out/last.pt if present), then write logits from best.pt."""
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(st.seed, st.warn_only)
    dev = st.device
    model = ConvTransformer.from_config(cfg.dl, tab_dim=rows["train"].tab.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=st.lr, weight_decay=st.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=st.amp and dev.startswith("cuda"))
    start_epoch, best, bad, history = 0, float("inf"), 0, []
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best, bad, history = state["epoch"] + 1, state["best"], state["bad"], state["history"]
        set_rng_state(state)
        log(f"  resumed from epoch {state['epoch']} (best val loss {best:.4f})")

    val_idx = np.arange(len(rows["validation"]))
    if st.max_val_rows and len(val_idx) > st.max_val_rows:
        val_idx = np.sort(np.random.default_rng(st.seed).choice(val_idx, st.max_val_rows, replace=False))
    val_ds = WindowDataset(rows["validation"], reader, st.W, val_idx)
    n_train = len(rows["train"])
    timing = []
    for epoch in range(start_epoch, st.max_epochs):
        if bad >= st.patience:
            break
        t0 = time.time()
        g = np.random.default_rng(st.seed * 1000 + epoch)
        idx = g.choice(n_train, min(st.samples_per_epoch, n_train), replace=False)
        train_ds = WindowDataset(rows["train"], reader, st.W, idx)
        model.train()
        tot, nb = torch.zeros((), device=dev), 0  # summed on the device: no GPU sync per batch
        for wave, tab, y in _loader(train_ds, st, st.seed + epoch):
            wave, tab, y = wave.to(dev, non_blocking=True), tab.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.split(":")[0], dtype=torch.float16,
                                enabled=st.amp and dev.startswith("cuda")):
                logits = model(wave, tab)
            loss = masked_bce(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.detach()
            nb += 1
        tot = float(tot)
        t_train = time.time() - t0
        val = eval_loss(model, val_ds, st)
        t_epoch = time.time() - t0
        improved = val < best - 1e-6
        best, bad = (val, 0) if improved else (best, bad + 1)
        history.append({"epoch": epoch, "train_loss": tot / max(nb, 1), "val_loss": val, "improved": improved,
                        "train_seconds": round(t_train, 1), "epoch_seconds": round(t_epoch, 1),
                        "train_samples": len(idx), **peak_memory()})
        timing.append(history[-1])
        if improved:
            save_atomic({"model": model.state_dict(), "epoch": epoch, "val_loss": val}, out / "best.pt")
        save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "best": best, "bad": bad, "history": history, **rng_state()}, out / "last.pt")
        pd.DataFrame(history).to_csv(out / "train_log.csv", index=False)
        log(f"  W{st.W} seed {st.seed} epoch {epoch}: train {tot / max(nb, 1):.4f} val {val:.4f}"
            f"{' *' if improved else ''}  {t_epoch:.0f} s ({len(idx) / max(t_train, 1e-9):.0f} samples/s)")
        if on_epoch:
            on_epoch()

    # logits from the best checkpoint on calibration and validation (eligible rows)
    if not (out / "best.pt").exists():
        raise FileNotFoundError(f"{out / 'best.pt'} is missing (last.pt was restored without it?); "
                                "delete last.pt to retrain this seed from scratch")
    best_state = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(best_state["model"])
    t0 = time.time()
    for split in ("calibration", "validation"):
        r = rows[f"{split}_all"]
        logits = predict(model, WindowDataset(r, reader, st.W), st)
        df = pd.DataFrame({"caseid": r.caseid, "time": r.time})
        for k, h in enumerate(HORIZONS):
            df[f"logit_{h}"] = logits[:, k]
        write_parquet(out / f"logits_{split}.parquet", df)
    t_pred = time.time() - t0
    info = {"best_epoch": int(best_state["epoch"]), "best_val_loss": float(best_state["val_loss"]),
            "epochs_run": len(history), "predict_seconds": round(t_pred, 1),
            "predict_rows": int(sum(len(rows[f"{s}_all"]) for s in ("calibration", "validation")))}
    write_json(out / "done.json", info)
    return {**info, "timing": timing}
