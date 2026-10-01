"""NB03: train the Conv1D + Transformer for one window W and several seeds.

    python scripts/train_dl.py --window 60 --seeds 20260917 20260918 --prep <dir with wave100/>
    python scripts/train_dl.py --window 60 --smoke --prep <...>          # 03_dl_smoke: 1 epoch, timing only
    python scripts/train_dl.py --window 60 --subset dryrun --max-samples 5000 --epochs 1   # local dry run

Each (W, seed) writes only artifacts/dl/W<W>/seed<s>/ (plan 7.4); a run with
done.json locally or on HF is skipped, a run with last.pt resumes (locally, or
best.pt + last.pt pulled from HF). --smoke never pushes models: it only pushes
reports/dl_smoke/timing.json with the GPU-hour estimate.

With several GPUs (Kaggle "GPU T4 x2") the pending seeds run in parallel, one
child process per GPU (CUDA_VISIBLE_DEVICES), and the DataLoader workers are
split between them. A seed gives the same result on any GPU, so this only
changes the wall time. --gpus 1 turns it off.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

from uc04.dl_data import RowSet, WaveReader, load_case_meta, tab_columns, wave_stats
from uc04.dl_train import RunSettings, train_one
from uc04.hub import hub_from_config
from uc04.io import read_json, write_json
from uc04.provenance import make_provenance, write_provenance
from uc04.runtime import common_args, load_data, samples_inputs, setup

SMOKE_FILE = "reports/dl_smoke/timing.json"


def _stream(proc: subprocess.Popen, prefix: str) -> None:
    for line in proc.stdout:
        print(prefix + line, end="", flush=True)


def run_parallel(argv: list[str], seeds: list[int], gpus: int, workers: int, poll_s: float = 5.0) -> int:
    """Run one child per seed, at most one per GPU at a time; returns 1 if any seed failed."""
    queue, running, failed = list(seeds), {}, []
    print(f"parallel: seeds {seeds} on {gpus} GPUs, {workers} DataLoader workers each", flush=True)
    while queue or running:
        for g in range(gpus):
            if g in running or not queue:
                continue
            seed = queue.pop(0)
            # argparse keeps the last occurrence, so appending overrides --seeds / --num-workers / --gpus
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), *argv,
                   "--seeds", str(seed), "--num-workers", str(workers), "--gpus", "1"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace")
            th = threading.Thread(target=_stream, args=(proc, f"[gpu{g} seed {seed}] "), daemon=True)
            th.start()
            running[g] = (proc, seed, th)
        time.sleep(poll_s)
        for g, (proc, seed, th) in list(running.items()):
            if proc.poll() is not None:
                th.join()
                del running[g]
                if proc.returncode:
                    failed.append(seed)
                    print(f"seed {seed} on gpu{g} FAILED (exit code {proc.returncode})", flush=True)
    if failed:
        print(f"failed seeds: {failed}; rerun the notebook to resume them", flush=True)
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--prep", help="directory with wave100/ and normalization.json (default: paths.prep)")
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--epochs", type=int, help="max epochs (default dl.max_epochs)")
    ap.add_argument("--max-samples", type=int, help="train samples per epoch (default dl.train_samples_per_epoch)")
    ap.add_argument("--max-val-rows", type=int, help="validation rows for the early-stopping loss (default: all)")
    ap.add_argument("--smoke", action="store_true", help="1 epoch with the first seed; push only the timing file")
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gpus", type=int, help="GPUs for parallel seeds (default: all visible; 1 = sequential)")
    ap.add_argument("--warn-only", action="store_true", help="deterministic algorithms in warn-only mode (DEVIATIONS 13)")
    ap.add_argument("--final-push-only", action="store_true")
    args = ap.parse_args(argv)

    cfg, samples, work, push, subset = setup(args)
    D = cfg.dl
    limited = bool(args.max_samples or (args.epochs and args.epochs < D.max_epochs))
    push = push and (not limited or args.smoke)
    hub = hub_from_config(cfg, work, push=push, only=[SMOKE_FILE] if args.smoke else None)
    W = args.window
    seeds = args.seeds or list(D.seeds)
    if args.smoke:
        seeds = seeds[:1]
    if args.final_push_only:
        rels = [f"artifacts/dl/W{W}/seed{s}" for s in seeds if (work / f"artifacts/dl/W{W}/seed{s}").exists()]
        hub.register(*rels)
        hub.push(rels, message=f"dl W{W} final push")
        return 0

    remote_done = hub.done_set("dl")
    n_gpus = torch.cuda.device_count() if args.device == "auto" else 0
    gpus = min(args.gpus or n_gpus, n_gpus)
    if not args.smoke and gpus > 1:
        pending = [s for s in seeds if not (work / f"artifacts/dl/W{W}/seed{s}/done.json").exists()
                   and f"dl/W{W}/seed{s}" not in remote_done]
        if len(pending) > 1:
            workers = D.num_workers if args.num_workers is None else args.num_workers
            return run_parallel(list(sys.argv[1:] if argv is None else argv), pending, min(gpus, len(pending)),
                                max(1, workers // gpus))

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    prep = Path(args.prep) if args.prep else cfg.path("prep")
    st_base = dict(W=W, max_epochs=1 if args.smoke else (args.epochs or D.max_epochs), patience=D.patience,
                   samples_per_epoch=args.max_samples or D.train_samples_per_epoch, batch_size=D.batch_size,
                   lr=D.optimizer["lr"], weight_decay=D.optimizer["weight_decay"], amp=bool(D.amp),
                   num_workers=D.num_workers if args.num_workers is None else args.num_workers, device=device,
                   max_val_rows=args.max_val_rows, warn_only=args.warn_only)
    print(f"W={W} seeds={seeds} device={device} "
          f"{torch.cuda.get_device_name(0) if device.startswith('cuda') else ''} work={work}")

    t0 = time.time()
    labels, feats, _ = load_data(samples, tab_columns(W), subset)
    norm = read_json(samples / "norm_tabular.json")["columns"]
    starts, masks, base_missing = load_case_meta(samples)
    mean, std = wave_stats(prep)
    reader = WaveReader(prep / "wave100", starts, masks, mean, std, hz=D.wave_hz)
    rows = {"train": RowSet(labels["train"], feats["train"], W, norm, base_missing, train_rows=True),
            "validation": RowSet(labels["validation"], feats["validation"], W, norm, base_missing, train_rows=True)}
    for s in ("calibration", "validation"):
        rows[f"{s}_all"] = RowSet(labels[s], feats[s], W, norm, base_missing, train_rows=False)
    print(f"rows: train {len(rows['train']):,}, validation {len(rows['validation']):,}; "
          f"prediction rows cal {len(rows['calibration_all']):,} val {len(rows['validation_all']):,} "
          f"({time.time() - t0:.0f} s)", flush=True)

    prov = make_provenance(cfg, "train_dl", inputs=samples_inputs(samples),
                           extra={"window": W, "device": device, "subset": args.subset, "smoke": args.smoke})
    for seed in seeds:
        rel = f"artifacts/dl/W{W}/seed{seed}" if not args.smoke else f"artifacts/dl_smoke/W{W}/seed{seed}"
        out = work / rel
        if (out / "done.json").exists() or (not args.smoke and f"dl/W{W}/seed{seed}" in remote_done):
            print(f"seed {seed}: already done, skipping")
            continue
        if hub.enabled and not args.smoke and not (out / "last.pt").exists():
            # resume a run interrupted in an earlier Kaggle session: the whole seed directory, since
            # last.pt without best.pt would fail at prediction time if no later epoch improves
            try:
                src = hub.pull(f"dl/W{W}/seed{seed}/", local_dir=work / "artifacts" / "_pull" / f"seed{seed}")
                files = sorted(f for f in src.iterdir() if f.is_file()) if src.is_dir() else []
                if any(f.name == "last.pt" for f in files):
                    out.mkdir(parents=True, exist_ok=True)
                    for f in files:
                        f.replace(out / f.name)
                    print(f"seed {seed}: pulled {[f.name for f in files]} from HF")
            except Exception as exc:
                print(f"seed {seed}: could not pull an earlier run from HF ({exc!r}); starting from scratch")
        hub.register(rel)
        out.mkdir(parents=True, exist_ok=True)
        write_provenance(out, {**prov, "seed": seed})
        st = RunSettings(seed=seed, **st_base)
        res = train_one(cfg, st, rows, reader, out, on_epoch=lambda: hub.maybe_push([rel], f"dl W{W} seed {seed}"))
        hub.push([rel], message=f"dl W{W} seed {seed} done")
        print(f"seed {seed}: best epoch {res['best_epoch']}, val loss {res['best_val_loss']:.4f}, "
              f"{res['epochs_run']} epochs, predict {res['predict_seconds']} s")
        if args.smoke:
            ep = res["timing"][0]
            per_epoch = {w: ep["epoch_seconds"] * w / W for w in D.windows_seconds}  # time ~ proportional to W
            runs = len(D.seeds)
            par = max(1, torch.cuda.device_count())  # seeds run in parallel, one per GPU
            rounds = -(-runs // par)
            est = {"expected_epochs": {"typical": 12, "max": D.max_epochs},
                   "gpu_hours_typical": round(sum(per_epoch.values()) * runs * 12 / 3600, 1),
                   "gpu_hours_max": round(sum(per_epoch.values()) * runs * D.max_epochs / 3600, 1),
                   "gpu_count": par,
                   "session_hours_typical": round(sum(per_epoch.values()) * rounds * 12 / 3600, 1),
                   "session_hours_max": round(sum(per_epoch.values()) * rounds * D.max_epochs / 3600, 1)}
            timing = {"window": W, "device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
                      "epoch": ep, "predict_seconds": res["predict_seconds"], "predict_rows": res["predict_rows"],
                      "epoch_seconds_by_window": per_epoch, "runs_per_window": runs, **est}
            write_json(work / SMOKE_FILE, timing)
            hub.register(SMOKE_FILE)
            hub.push([SMOKE_FILE], message="dl smoke timing")
            print(f"SMOKE: {ep['epoch_seconds']:.0f} s/epoch at W={W} ({ep['train_samples']:,} samples), "
                  f"RAM {ep['ram_peak_gb']} GB, VRAM {ep['vram_peak_gb']} GB")
            print(f"estimated GPU hours for {len(D.windows_seconds)} windows x {runs} seeds: "
                  f"~{est['gpu_hours_typical']} h (12 epochs), up to {est['gpu_hours_max']} h ({D.max_epochs} epochs). "
                  "Compare with the remaining weekly Kaggle GPU quota.")
            print(f"with {par} GPU(s) running seeds in parallel, Kaggle sessions take ~{est['session_hours_typical']} h "
                  f"(up to {est['session_hours_max']} h); the quota counts session time. Assumes parallel runs "
                  "do not slow each other down (they share 4 vCPUs).")
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
