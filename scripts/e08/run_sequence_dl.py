"""E08 DL family: sequence models on the full VitalDB cohort, scored like every other method.

  python scripts/e08/run_sequence_dl.py [--arch tcn inception ...] [--seeds 20260917 ...]
                                        [--max-cases N] [--epochs N] [--workers N]

Trains one model per architecture/seed on FIT (both horizons at once, ordered logits), then runs
the unchanged E08 tail: the CALIBRATION logistic recalibrator, the two threshold policies, and
bootstrap CIs on VALIDATION. Output lands in reports/E08/sequence_comparison.csv, which shares its
columns with method_comparison.csv so the two concatenate.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/e08"))
sys.path.insert(0, str(ROOT / "src"))

from run_comparison import evaluate  # noqa: E402
from safeanes.e08_methods import HORIZONS, calibrate_member  # noqa: E402
from safeanes.e08_sequence import (ARCHITECTURES, TRAIN, FastWindowDataset,  # noqa: E402
                                   load_sequence_context)
from safeanes.sequences import fit_normalizer  # noqa: E402
from safeanes.tabular_sota import inner_stop_mask  # noqa: E402

OUT = ROOT / "reports/E08"
MODELS = ROOT / "artifacts/E08/sequence"


def loader(dataset, batch_size, shuffle, workers):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=False, drop_last=False, persistent_workers=bool(workers))


def average_precision(y, score):
    from sklearn.metrics import average_precision_score
    known = y >= 0
    if not known.any() or len(np.unique(y[known])) < 2:
        return float("nan")
    return float(average_precision_score(y[known], score[known]))


def predict(model, dataset, device, batch_size, workers):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for x, static, _ in loader(dataset, batch_size, False, workers):
            logits = model(x.to(device), static.to(device))
            out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2))


def train_one(ctx, architecture, seed, args, device):
    import torch
    from safeanes.models import NumericModel, masked_bce

    torch.manual_seed(seed)
    np.random.seed(seed)
    normalizer = fit_normalizer(ctx.store, ctx.fit_rows)
    stop = inner_stop_mask(ctx.fit_rows.subjectid.to_numpy())
    inner_fit = ctx.fit_rows[~stop].reset_index(drop=True)
    inner_stop = ctx.fit_rows[stop].reset_index(drop=True)
    make = lambda frame: FastWindowDataset(ctx.store, frame, normalizer, HORIZONS)
    train_set, stop_set = make(inner_fit), make(inner_stop)
    print(f"    train={len(inner_fit)} stop={len(inner_stop)} channels={train_set.channels}", flush=True)

    model = NumericModel(architecture, train_set.channels, train_set.static_dim,
                         length=ctx.protocol.history_seconds // ctx.protocol.numeric_step_seconds,
                         width=TRAIN["width"], dropout=TRAIN["dropout"], layers=TRAIN["layers"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    y_stop = inner_stop[[f"y_{h}" for h in HORIZONS]].to_numpy(float)

    best, best_state, bad, history = -np.inf, None, 0, []
    for epoch in range(args.epochs):
        model.train()
        start, total, seen = time.monotonic(), 0.0, 0
        for x, static, y in loader(train_set, TRAIN["batch_size"], True, args.workers):
            x, static, y = x.to(device), static.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = masked_bce(model(x, static), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(x)
            seen += len(x)
        p = predict(model, stop_set, device, TRAIN["batch_size"] * 2, args.workers)
        score = float(np.nanmean([average_precision(y_stop[:, i], p[:, i]) for i in range(len(HORIZONS))]))
        history.append({"epoch": epoch, "loss": total / max(seen, 1), "stop_ap": score,
                        "seconds": time.monotonic() - start})
        print(f"    epoch {epoch:2d} loss={total/max(seen,1):.4f} stop_AP={score:.4f} "
              f"({time.monotonic()-start:.0f}s)", flush=True)
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= TRAIN["patience"]:
                print(f"    early stop tai epoch {epoch}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    raw_cal = predict(model, make(ctx.cal_rows), device, TRAIN["batch_size"] * 2, args.workers)
    raw_val = predict(model, make(ctx.val_rows), device, TRAIN["batch_size"] * 2, args.workers)
    params = {**TRAIN, "architecture": architecture, "seed": seed, "epochs_run": len(history),
              "best_stop_ap": best, "parameters": sum(p.numel() for p in model.parameters()),
              "n_train": len(inner_fit), "device": device.type}
    return raw_cal, raw_val, params, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", nargs="+", default=["tcn"], choices=list(ARCHITECTURES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260917])
    parser.add_argument("--max-cases", type=int, default=None, help="cases per split; smoke tests only")
    parser.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=TRAIN["width"])
    parser.add_argument("--batch-size", type=int, default=TRAIN["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN["lr"])
    parser.add_argument("--out", default="sequence_comparison.csv")
    args = parser.parse_args()

    TRAIN.update(width=args.width, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    print(f"device={device} arch={args.arch} seeds={args.seeds}", flush=True)

    ctx = load_sequence_context(ROOT, args.max_cases)
    print(f"FIT={len(ctx.fit_rows)} ({ctx.fit_rows.caseid.nunique()} ca) "
          f"CAL={len(ctx.cal_rows)} ({ctx.cal_rows.caseid.nunique()} ca) "
          f"VAL={len(ctx.val_rows)} ({ctx.val_rows.caseid.nunique()} ca)", flush=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    rows, params, curves = [], [], {}
    for architecture in args.arch:
        for seed in args.seeds:
            name = f"seq_{architecture}_{seed}"
            print(f"  {name}", flush=True)
            raw_cal, raw_val, used, history = train_one(ctx, architecture, seed, args, device)
            curves[name] = history
            params.append({"method": name, **used})
            for i, horizon in enumerate(HORIZONS):
                mask = ctx.calibration_mask(horizon)
                y_cal = ctx.cal_rows.loc[mask, f"y_{horizon}"].astype(int).to_numpy()
                p_cal, p_val = calibrate_member(raw_cal[:, i], raw_val[:, i], y_cal, mask)
                rows += evaluate(ctx, name, "sequence", p_cal, p_val, horizon)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / args.out, index=False)
    pd.DataFrame(params).to_csv(OUT / args.out.replace(".csv", "_parameters.csv"), index=False)
    (OUT / args.out.replace(".csv", "_curves.json")).write_text(json.dumps(curves, indent=2))
    print(f"wrote {OUT / args.out}", flush=True)


if __name__ == "__main__":
    main()
