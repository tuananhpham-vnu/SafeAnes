"""Inputs of the Conv1D + Transformer model (plan section 7.1).

For a row (caseid, t) and window W:
- wave  [2, W*100]: samples k with t - W < start + k/100 <= t from wave100/<caseid>.npy;
  channel 0 = (x - mean) / std with the train statistics of normalization.json
  (NaN -> 0), channel 1 = mask (1 where NaN or its 1 s block has wave_mask != 0);
  samples before the start of the recording are padded and masked;
- tab   [67]: the 66 numeric columns of W, train-median imputed and z-scored with
  norm_tabular.json, plus `baseline_missing`;
- y     [5]: y_300 ... y_1800 (-1 = unknown, ignored in the loss).
Only data at or before t is read (tested in tests/test_dl.py).
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .columns import numeric_cols
from .io import read_json

HORIZONS = (300, 600, 900, 1200, 1800)


class WaveReader:
    """Reads causal waveform windows; keeps up to `max_open` memmaps open (LRU)."""

    def __init__(self, wave_dir: Path, starts: dict[int, float], masks: dict[int, np.ndarray], mean: float,
                 std: float, hz: int = 100, max_open: int = 256):
        self.wave_dir, self.starts, self.masks = Path(wave_dir), starts, masks
        self.mean, self.std, self.hz, self.max_open = float(mean), float(std), hz, max_open
        self._open: OrderedDict[int, np.ndarray] = OrderedDict()

    def __getstate__(self):  # never ship open memmaps to DataLoader workers
        state = dict(self.__dict__)
        state["_open"] = OrderedDict()
        return state

    def _wave(self, caseid: int) -> np.ndarray:
        w = self._open.pop(caseid, None)
        if w is None:
            w = np.load(self.wave_dir / f"{caseid}.npy", mmap_mode="r")
            if len(self._open) >= self.max_open:
                self._open.popitem(last=False)
        self._open[caseid] = w
        return w

    def window(self, caseid: int, t: float, W: int) -> np.ndarray:
        n = W * self.hz
        k_end = int(round((t - self.starts[caseid]) * self.hz))   # sample exactly at t
        k0 = k_end - n + 1                                        # first sample > t - W
        w = self._wave(caseid)
        lo, hi = max(k0, 0), min(k_end + 1, len(w))
        x = np.full(n, np.nan, np.float32)
        if hi > lo:
            x[lo - k0:hi - k0] = w[lo:hi]
        mask = np.isnan(x)
        blocks = self.masks.get(caseid)
        if blocks is not None:
            k = np.arange(k0, k0 + n)
            j = np.clip(k // self.hz, 0, len(blocks) - 1)
            mask |= (blocks[j] != 0) & (k >= 0)
        z = np.where(mask, 0.0, (np.nan_to_num(x) - self.mean) / self.std).astype(np.float32)
        return np.stack([z, mask.astype(np.float32)])


def tab_columns(W: int) -> list[str]:
    return numeric_cols(W)


def tab_matrix(feats: pd.DataFrame, W: int, norm: dict, baseline_missing: np.ndarray) -> np.ndarray:
    """[n, 67] float32: impute train median, z-score, append baseline_missing."""
    cols = tab_columns(W)
    out = np.empty((len(feats), len(cols) + 1), np.float32)
    for i, c in enumerate(cols):
        s = norm[c]
        x = feats[c].to_numpy(np.float64)
        med = s["median"] if s["median"] is not None else 0.0
        mean = s["mean"] if s["mean"] is not None else 0.0
        std = s["std"] if s["std"] else 1.0
        x = np.where(np.isfinite(x), x, med)
        out[:, i] = (x - mean) / std
    out[:, -1] = baseline_missing.astype(np.float32)
    return out


def load_case_meta(samples: Path, sealed: Path | None = None, include_test: bool = False):
    """(starts, masks, baseline_missing by caseid) from case_index / wave_mask."""
    idx = pd.read_parquet(Path(samples) / "case_index.parquet")
    masks = dict(np.load(Path(samples) / "wave_mask.npz"))
    if include_test:
        idx = pd.concat([idx, pd.read_parquet(Path(sealed) / "case_index_test.parquet")], ignore_index=True)
        masks.update(dict(np.load(Path(sealed) / "wave_mask_test.npz")))
    starts = dict(zip(idx.caseid.astype(int), idx.start.astype(float)))
    base_missing = dict(zip(idx.caseid.astype(int), (idx.baseline_source == 0)))
    return starts, {int(k): v for k, v in masks.items()}, base_missing


def wave_stats(prep: Path) -> tuple[float, float]:
    w = read_json(Path(prep) / "normalization.json")["wave100"]
    return float(w["mean"]), float(w["std"])


class RowSet:
    """Rows of one split for one window: identifiers, tab features, labels."""

    def __init__(self, labels: pd.DataFrame, feats: pd.DataFrame, W: int, norm: dict, base_missing: dict,
                 train_rows: bool):
        y = labels[[f"y_{h}" for h in HORIZONS]].to_numpy(np.int8)
        keep = labels["eligible"].to_numpy(bool)
        if train_rows:  # eligible and at least one known label
            keep = keep & (y != -1).any(1)
        self.index = np.flatnonzero(keep)  # positions in the split's labels table
        self.caseid = labels["caseid"].to_numpy(np.int64)[keep]
        self.time = labels["time"].to_numpy(np.float64)[keep]
        self.y = y[keep]
        bm = np.array([base_missing.get(int(c), False) for c in self.caseid])
        self.tab = tab_matrix(feats.iloc[self.index], W, norm, bm)

    def __len__(self) -> int:
        return len(self.caseid)


class WindowDataset(Dataset):
    """Picklable (spawn-safe) dataset over a RowSet; memmaps are reopened in each worker."""

    def __init__(self, rows: RowSet, reader: WaveReader, W: int, index: np.ndarray | None = None):
        self.rows, self.reader, self.W = rows, reader, W
        self.index = np.arange(len(rows)) if index is None else np.asarray(index)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i):
        j = int(self.index[i])
        wave = self.reader.window(int(self.rows.caseid[j]), float(self.rows.time[j]), self.W)
        return (torch.from_numpy(wave), torch.from_numpy(self.rows.tab[j]),
                torch.from_numpy(self.rows.y[j].astype(np.float32)))
