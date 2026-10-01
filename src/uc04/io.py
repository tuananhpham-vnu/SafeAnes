"""Readers for prep_v1 and atomic writers (plan sections 1, 2.12).

prep_v1 is read-only. Readers only load the arrays or columns asked for.
`npz_shape` reads an array's shape from its .npy header inside the .npz without
loading values, which is how test-split `label_map` lengths are checked (plan 3.1).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SPLITS = ("train", "calibration", "validation", "test")


# ---------------------------------------------------------------- prep_v1 readers

def read_qc(prep: Path) -> pd.DataFrame:
    qc = pd.read_csv(Path(prep) / "qc.csv")
    qc["caseid"] = qc["caseid"].astype(int)
    qc["subjectid"] = qc["subjectid"].astype(int)
    qc["has_wave"] = qc["has_wave"].astype(bool)
    return qc


def case_path(prep: Path, caseid: int) -> Path:
    return Path(prep) / "cases" / f"{int(caseid)}.npz"


def features_path(prep: Path, caseid: int) -> Path:
    return Path(prep) / "features" / f"{int(caseid)}.parquet"


def wave_path(prep: Path, caseid: int) -> Path:
    return Path(prep) / "wave100" / f"{int(caseid)}.npy"


def load_case(prep: Path, caseid: int, keys: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """Load selected arrays of cases/<caseid>.npz (all if keys is None)."""
    with np.load(case_path(prep, caseid), allow_pickle=False) as z:
        names = z.files if keys is None else list(keys)
        return {k: z[k] for k in names}


def npz_shape(path: Path, key: str) -> tuple[tuple[int, ...], np.dtype]:
    """Shape and dtype of `key` in an .npz, read from the header only."""
    with zipfile.ZipFile(path) as zf, zf.open(f"{key}.npy") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
    return tuple(shape), np.dtype(dtype)


def feature_columns(prep: Path, caseid: int) -> list[str]:
    return pq.read_schema(features_path(prep, caseid)).names


def read_features(prep: Path, caseid: int, columns: Iterable[str] | None = None) -> pd.DataFrame:
    cols = None if columns is None else list(columns)
    return pq.read_table(features_path(prep, caseid), columns=cols).to_pandas()


def open_wave(prep: Path, caseid: int) -> np.ndarray:
    return np.load(wave_path(prep, caseid), mmap_mode="r")


# ---------------------------------------------------------------- writers and digests

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temporary path next to `path`; rename onto `path` only on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path: Path, obj: Any) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    with atomic_path(path) as tmp:
        df.to_parquet(tmp, index=False)
