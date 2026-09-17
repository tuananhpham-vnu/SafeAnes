"""Causal per-case arrays, memory-mapped windows, fit-patient normalization."""

from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TRACKS, Protocol
from .data import fetch_csv, read_numeric, write_json
from .dataset import sample_case


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(dataset):
    dataset = Path(dataset)
    meta = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    if meta["scope"] != "pilot_train_pool_only":
        raise ValueError("Only global TRAIN-pool pilot datasets are accepted")
    config = dict(meta["protocol"])
    config["horizons_seconds"] = tuple(config["horizons_seconds"])
    protocol = Protocol(**config)
    if meta["protocol_hash"] != protocol.digest():
        raise ValueError("Protocol hash mismatch")
    if file_hash(dataset / "windows.csv.gz") != meta["windows_sha256"]:
        raise ValueError("Dataset checksum mismatch")
    manifest = pd.read_csv(dataset / "manifest.csv")
    if not manifest.split.eq("train").all() or manifest.caseid.duplicated().any():
        raise ValueError("Unique cases from global TRAIN patients required")
    windows = pd.read_csv(dataset / "windows.csv.gz")
    joined = windows.merge(manifest[["caseid", "subjectid"]], on=["caseid", "subjectid"],
                           how="left", indicator=True, validate="many_to_one")
    if not joined._merge.eq("both").all() or windows.duplicated(["caseid", "time"]).any():
        raise ValueError("Window identifiers do not match manifest")
    return meta, protocol, manifest, windows


def build_sequences(dataset, root, out):
    """Build from verified local raw cache; never fetch additional patients."""
    dataset, root, out = Path(dataset), Path(root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Sequence output must be new/empty")
    meta, protocol, manifest, windows = load_dataset(dataset)
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for _, case in manifest.iterrows():
        raw, sources = {}, {}
        for name in TRACKS:
            tid = case.get(f"tid_{name}")
            if pd.isna(tid):
                continue
            path = root / "raw" / f"{tid}.csv.gz"
            if not path.is_file():
                raise FileNotFoundError(f"Raw cache missing: {path}; run fetch-pilot first")
            raw[name] = read_numeric(fetch_csv(str(tid), root / "raw"))
            sources[name] = file_hash(path)
        _, grid, _, sampled, ages = sample_case(case, raw, protocol)
        values = np.stack([sampled[k] for k in TRACKS], axis=1).astype("float32")
        age = np.stack([np.minimum(ages[k], protocol.history_seconds) for k in TRACKS], axis=1)
        array = np.concatenate([values, age.astype("float32")], axis=1)
        caseid = int(case.caseid)
        path = out / f"{caseid}.npy"
        np.save(path, array, allow_pickle=False)
        times = windows.loc[windows.caseid.eq(caseid), "time"].to_numpy()
        stops = np.searchsorted(grid, times, side="right")
        length = protocol.history_seconds // protocol.numeric_step_seconds
        if len(stops) and (stops.min() < length or stops.max() > len(grid)):
            raise ValueError(f"Incomplete sequence history for case {caseid}")
        records[str(caseid)] = {"start": float(grid[0]), "steps": len(grid),
            "subjectid": int(case.subjectid), "sha256": file_hash(path), "raw_hashes": sources}
    write_json(out / "sequences.json", {"version": "uc04-sequences-v1", "protocol": asdict(protocol),
        "protocol_hash": protocol.digest(), "windows_sha256": meta["windows_sha256"],
        "manifest_sha256": file_hash(dataset / "manifest.csv"), "tracks": list(TRACKS),
        "cases": records})
    return {"cases": len(records), "bytes": sum(p.stat().st_size for p in out.glob("*.npy"))}


class SequenceStore:
    def __init__(self, root, dataset):
        self.root = Path(root)
        self.meta = json.loads((self.root / "sequences.json").read_text(encoding="utf-8"))
        meta, self.protocol, manifest, _ = load_dataset(dataset)
        if (self.meta["protocol_hash"] != self.protocol.digest()
                or self.meta["windows_sha256"] != meta["windows_sha256"]
                or self.meta["manifest_sha256"] != file_hash(Path(dataset) / "manifest.csv")
                or self.meta["tracks"] != list(TRACKS)):
            raise ValueError("Sequence cache does not match the dataset")
        if set(self.meta["cases"]) != set(manifest.caseid.astype(str)):
            raise ValueError("Sequence case set does not match manifest")
        for caseid, record in self.meta["cases"].items():
            if file_hash(self.root / f"{caseid}.npy") != record["sha256"]:
                raise ValueError(f"Sequence checksum mismatch: {caseid}")
        self._cache = OrderedDict()

    def array(self, caseid):
        key = str(int(caseid))
        if key not in self._cache:
            self._cache[key] = np.load(self.root / f"{key}.npy", mmap_mode="r", allow_pickle=False)
        self._cache.move_to_end(key)
        if len(self._cache) > 8:
            self._cache.popitem(last=False)
        return self._cache[key]

    def window(self, caseid, time):
        record = self.meta["cases"][str(int(caseid))]
        stop = int(np.floor((time - record["start"]) / self.protocol.numeric_step_seconds)) + 1
        length = self.protocol.history_seconds // self.protocol.numeric_step_seconds
        if stop < length or stop > record["steps"]:
            raise ValueError("Window outside cached history")
        return np.asarray(self.array(caseid)[stop - length:stop], dtype=np.float32)


def fit_normalizer(store, fit_frame):
    """Each grid point counted once; no calibration/validation/test patients used."""
    n = len(TRACKS)
    count, total, squares = np.zeros(n), np.zeros(n), np.zeros(n)
    for caseid in fit_frame.caseid.unique():
        x = np.asarray(store.array(caseid)[:, :n], dtype=np.float64)
        good = np.isfinite(x)
        count += good.sum(axis=0)
        total += np.where(good, x, 0).sum(axis=0)
        squares += np.where(good, x * x, 0).sum(axis=0)
    mean = total / np.maximum(count, 1)
    scale = np.sqrt(np.maximum(squares / np.maximum(count, 1) - mean * mean, 0))
    scale[scale < 1e-6] = 1
    static = fit_frame.drop_duplicates("caseid")[["static_age", "static_bmi", "static_asa"]].to_numpy(float)
    good = np.isfinite(static)
    static_mean = np.where(good, static, 0).sum(0) / np.maximum(good.sum(0), 1)
    static_scale = np.sqrt(np.where(good, (static - static_mean)**2, 0).sum(0) / np.maximum(good.sum(0), 1))
    static_scale[static_scale < 1e-6] = 1
    return {"mean": mean.tolist(), "scale": scale.tolist(), "static_mean": static_mean.tolist(),
        "static_scale": static_scale.tolist(), "fit_subjectids": sorted(map(int, fit_frame.subjectid.unique()))}


class WindowDataset:
    """NumPy samples are collated to tensors by PyTorch; torch stays optional here."""
    def __init__(self, store, frame, normalizer, feature_set="numeric_static", masks=True):
        if feature_set not in ("map", "numeric", "numeric_static"):
            raise ValueError("Unknown feature set")
        self.store, self.frame, self.norm = store, frame.reset_index(drop=True), normalizer
        self.feature_set, self.masks = feature_set, masks
        self.indices = [0] if feature_set == "map" else list(range(len(TRACKS)))
        self.channels = len(self.indices) * (3 if masks else 1)
        self.static_dim = 6 if feature_set == "numeric_static" else 0

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        raw = self.store.window(row.caseid, row.time)
        values = raw[:, self.indices]
        good = np.isfinite(values)
        mean = np.asarray(self.norm["mean"])[self.indices]
        scale = np.asarray(self.norm["scale"])[self.indices]
        x = np.where(good, (values - mean) / scale, 0)
        if self.masks:
            age = raw[:, np.asarray(self.indices) + len(TRACKS)] / self.store.protocol.history_seconds
            x = np.concatenate([x, good, age], axis=1)
        static = np.empty(0, np.float32)
        if self.static_dim:
            v = row[["static_age", "static_bmi", "static_asa"]].to_numpy(float)
            good_static = np.isfinite(v)
            static = np.r_[np.where(good_static, (v - self.norm["static_mean"]) / self.norm["static_scale"], 0), good_static]
        y = row[[f"y_{h}" for h in self.store.protocol.horizons_seconds]].to_numpy(np.float32)
        return x.T.astype(np.float32), static.astype(np.float32), y
