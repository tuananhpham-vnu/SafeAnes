"""Load sample tables for training and evaluation (plan principle 1, 2).

`load_split("test")` fails unless `allow_test=True` and a lock file exists
(the command-line flag `--final-test` sets allow_test in NB04 only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .columns import SIDE_PREFIXES

TEST = "test"


class SealedError(RuntimeError):
    """Raised when test data is requested before the models are locked."""


def _test_files(sealed: Path) -> tuple[Path, Path]:
    return Path(sealed) / "labels_test.parquet", Path(sealed) / "features_test.parquet"


def split_files(samples: Path, split: str, sealed: Path | None = None) -> tuple[Path, Path]:
    if split == TEST:
        if sealed is None:
            raise SealedError("test data lives in the sealed directory, pass sealed=")
        return _test_files(sealed)
    return Path(samples) / "labels" / f"{split}.parquet", Path(samples) / "features" / f"{split}.parquet"


def load_split(samples: Path, split: str, columns: Sequence[str] | None = None, *, main_cohort: bool = True,
               allow_test: bool = False, lock_path: Path | None = None, sealed: Path | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(labels, features) of one split, same rows in the same order.

    `columns=None` reads every feature column; pass a list to read only those.
    """
    if split == TEST:
        if not allow_test:
            raise SealedError("test split requested without --final-test")
        if lock_path is None or not Path(lock_path).exists():
            raise SealedError(f"test split requested but lock file is missing: {lock_path}")
    lab_path, feat_path = split_files(samples, split, sealed)
    labels = pd.read_parquet(lab_path)
    side = [] if columns is None else [c for c in columns if c.startswith(SIDE_PREFIXES)]
    cols = None if columns is None else ["caseid", "time", *[c for c in columns
                                                            if c not in ("caseid", "time") and c not in side]]
    feats = pq.read_table(feat_path, columns=cols).to_pandas()
    if len(labels) != len(feats) or not (
            np.array_equal(labels["caseid"].to_numpy(), feats["caseid"].to_numpy())
            and np.array_equal(labels["time"].to_numpy(), feats["time"].to_numpy())):
        raise ValueError(f"labels and features of {split} are not row-aligned")
    if side:  # samples v3 side features: main-cohort rows only, in another order, joined on (caseid, time)
        sp = Path(samples) / "features_v5" / f"{split}.parquet"
        if split == TEST or not sp.exists():
            raise FileNotFoundError(f"{sp} not found: columns {side[:3]}... need samples v3 features_v5")
        s = pq.read_table(sp, columns=["caseid", "time", *side]).to_pandas().set_index(["caseid", "time"])
        key = pd.MultiIndex.from_frame(feats[["caseid", "time"]])
        main = labels["in_main_cohort"].to_numpy(bool)
        if not s.index.is_unique or not key[main].isin(s.index).all():
            raise ValueError(f"features_v5/{split} misses main-cohort rows of features/{split}")
        feats = pd.concat([feats, s.reindex(key).reset_index(drop=True)], axis=1)  # NaN outside the main cohort
    if main_cohort:
        keep = labels["in_main_cohort"].to_numpy(bool)
        labels, feats = labels[keep].reset_index(drop=True), feats[keep].reset_index(drop=True)
    return labels, feats


def check_no_subject_overlap(index: pd.DataFrame) -> None:
    n = index.groupby("subjectid")["split"].nunique()
    bad = n[n > 1]
    if len(bad):
        raise ValueError(f"{len(bad)} subjects appear in more than one split, e.g. {bad.index[:5].tolist()}")
