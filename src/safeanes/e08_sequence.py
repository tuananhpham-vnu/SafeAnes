"""Sequence-model family for E08: same splits, same calibration, same evaluation as the tabular methods.

The tabular methods see 143 window statistics per decision row. These see the numeric grid those
statistics were computed from — 300 steps x (7 tracks x value/mask/age) plus the three static
fields — so any difference in the E08 table is a difference in what the model is allowed to look
at, not in how it is scored.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .e08_methods import FULL_ROLES, LOCKED_GROUP, META, ROLES, _usable_for_fit
from .sequences import FullSequenceStore

STATIC = ["static_age", "static_bmi", "static_asa"]
ARCHITECTURES = ("tcn", "inception", "timesnet", "transformer")
TRAIN = {"epochs": 20, "patience": 4, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4,
         "width": 32, "dropout": .1, "layers": 2}


@dataclass
class SequenceContext:
    root: Path
    protocol: object
    events: pd.DataFrame
    store: FullSequenceStore
    fit_rows: pd.DataFrame
    cal_rows: pd.DataFrame
    val_rows: pd.DataFrame

    def calibration_mask(self, horizon):
        return (self.cal_rows.eligible & self.cal_rows[f"y_{horizon}"].ge(0)).to_numpy()


def load_sequence_context(root, max_cases=None):
    """Rows in the exact order e08_methods._full_vitaldb produces, without the feature matrices."""
    import joblib
    root = Path(root)
    store = FullSequenceStore(root / "data/sequences_full")
    manifest = pd.read_csv(root / "reports/E07/cohort_manifest.csv")
    manifest = manifest[manifest.eligible]
    columns = META + STATIC
    parts, events, taken = {r: [] for r in ROLES}, [], {r: 0 for r in ROLES}
    for group, role in FULL_ROLES.items():
        for caseid in manifest.loc[manifest.evaluation_group.eq(group), "caseid"]:
            path = root / "data/vitaldb_full/cases" / f"{caseid}.joblib"
            if not path.exists() or caseid not in store:
                continue
            if max_cases and taken[role] >= max_cases:
                break
            case = joblib.load(path)
            frame = _usable_for_fit(case["frame"]) if role == "fit" else case["frame"]
            parts[role].append(frame[columns])
            events += case["events"]
            taken[role] += 1
    rows = {role: pd.concat(parts[role], ignore_index=True) for role in ROLES}

    locked = set(manifest.loc[manifest.evaluation_group.eq(LOCKED_GROUP), "caseid"])
    for role, part in rows.items():
        if locked & set(part.caseid):
            raise RuntimeError(f"Global test cases leaked into {role}")
    subjects = {role: set(part.subjectid) for role, part in rows.items()}
    for a, b in (("fit", "calibration"), ("fit", "validation"), ("calibration", "validation")):
        if subjects[a] & subjects[b]:
            raise RuntimeError(f"Patients shared between {a} and {b}")
    return SequenceContext(root=root, protocol=store.protocol, events=pd.DataFrame(events),
                           store=store, fit_rows=rows["fit"], cal_rows=rows["calibration"],
                           val_rows=rows["validation"])


class FastWindowDataset:
    """WindowDataset's samples, without a DataFrame lookup per item.

    Identical output to sequences.WindowDataset(feature_set="numeric_static", masks=True); the
    per-row fields are hoisted into numpy up front because at ~600k windows per epoch the .iloc
    call dominates GPU time. Verified element-wise against WindowDataset in tests.
    """
    def __init__(self, store, frame, normalizer, horizons):
        from .config import TRACKS
        self.store, self.n = store, len(TRACKS)
        self.caseid = frame.caseid.to_numpy(np.int64)
        self.time = frame.time.to_numpy(float)
        self.y = frame[[f"y_{h}" for h in horizons]].to_numpy(np.float32)
        # float64 like WindowDataset: normalizing in float32 shifts samples by ~2e-6.
        self.mean = np.asarray(normalizer["mean"])
        self.scale = np.asarray(normalizer["scale"])
        self.age_scale = np.float32(store.protocol.history_seconds)
        static = frame[STATIC].to_numpy(float)
        good = np.isfinite(static)
        centred = (static - np.asarray(normalizer["static_mean"])) / np.asarray(normalizer["static_scale"])
        self.static = np.concatenate([np.where(good, centred, 0), good], axis=1).astype(np.float32)
        self.channels, self.static_dim = self.n * 3, 2 * len(STATIC)

    def __len__(self):
        return len(self.caseid)

    def __getitem__(self, index):
        raw = self.store.window(self.caseid[index], self.time[index])
        values, age = raw[:, :self.n], raw[:, self.n:]
        good = np.isfinite(values)
        x = np.concatenate([np.where(good, (values - self.mean) / self.scale, 0),
                            good, age / self.age_scale], axis=1)
        return x.T.astype(np.float32), self.static[index], self.y[index]
