"""Measure DL inference speed on this machine's CPU and estimate NB04 time.

    python scripts/dl_cpu_speed.py [--rows 3000] [--threads 0]

NB04 runs on the local machine because test data never leaves it. This times the
model (random weights: speed does not depend on them) on real validation windows
for every W, then estimates the time to predict the test rows for 4 windows x the
configured seeds. The test row count comes from the parquet metadata of
features_test (no label is read); the eligible fraction is taken from validation.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from uc04.dl_data import RowSet, WaveReader, WindowDataset, load_case_meta, tab_columns, wave_stats
from uc04.dl_model import ConvTransformer
from uc04.dl_train import RunSettings, predict
from uc04.io import read_json, write_json
from uc04.runtime import REPO_CONFIG, load_data
from uc04.config import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_CONFIG))
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    samples, prep, sealed = cfg.path("samples"), cfg.path("prep"), cfg.path("sealed")
    norm = read_json(samples / "norm_tabular.json")["columns"]
    starts, masks, base_missing = load_case_meta(samples)
    mean, std = wave_stats(prep)
    reader = WaveReader(prep / "wave100", starts, masks, mean, std)
    test_rows = pq.read_metadata(sealed / "features_test.parquet").num_rows
    speeds = {}
    for W in cfg.dl.windows_seconds:
        labels, feats, _ = load_data(samples, tab_columns(W), None, splits=("validation",))
        rows = RowSet(labels["validation"], feats["validation"], W, norm, base_missing, train_rows=False)
        elig_frac = len(rows) / len(labels["validation"])
        idx = np.random.default_rng(0).choice(len(rows), min(args.rows, len(rows)), replace=False)
        model = ConvTransformer.from_config(cfg.dl).eval()
        st = RunSettings(W=W, seed=0, max_epochs=0, patience=0, samples_per_epoch=0, batch_size=cfg.dl.batch_size,
                         lr=0, weight_decay=0, amp=False, num_workers=args.num_workers, device="cpu")
        predict(model, WindowDataset(rows, reader, W, idx[:256]), st)  # warm-up
        t0 = time.time()
        predict(model, WindowDataset(rows, reader, W, idx), st)
        sps = len(idx) / (time.time() - t0)
        speeds[W] = {"samples_per_second": round(sps, 1), "eligible_fraction_validation": round(elig_frac, 3)}
        print(f"W={W:>3}: {sps:,.0f} samples/s on CPU ({torch.get_num_threads()} threads)", flush=True)
    seeds = len(cfg.dl.seeds)
    total = sum(test_rows * v["eligible_fraction_validation"] * seeds / v["samples_per_second"] for v in speeds.values())
    est = {"test_rows_all": test_rows, "seeds": seeds, "windows": list(cfg.dl.windows_seconds),
           "hours_all_windows_all_seeds": round(total / 3600, 2),
           "hours_one_window_all_seeds": round(total / len(speeds) / 3600, 2), "speeds": speeds}
    write_json(cfg.path("reports") / "dl_cpu_speed.json", est)
    print(f"test: {test_rows:,} rows; estimated NB04 DL inference on this CPU: "
          f"{est['hours_all_windows_all_seeds']} h for {len(speeds)} windows x {seeds} seeds "
          f"({est['hours_one_window_all_seeds']} h for one window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
