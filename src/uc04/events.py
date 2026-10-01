"""Hypotension events from label_map (plan section 3.4).

An event is MAP < 65 for >= 60 consecutive known seconds. Two events less than
120 s apart are merged, but only when every second between them is known.
Times are absolute seconds on the case clock; `end` is exclusive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["onset", "end", "duration", "min_map", "suspect_artefact"]


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(i0, i1) of consecutive True runs, i1 exclusive."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def longest_true_run(mask: np.ndarray) -> int:
    r = runs(np.asarray(mask, bool))
    return max((b - a for a, b in r), default=0)


def detect_events(label_map: np.ndarray, start: float, *, threshold: float = 65.0, min_seconds: int = 60,
                  merge_gap: int = 120, suspect_min_map: float = 30.0) -> pd.DataFrame:
    lm = np.asarray(label_map, dtype=np.float64)
    finite = np.isfinite(lm)
    below = finite & (lm < threshold)
    events: list[list[float]] = []  # [onset_idx, end_idx, min_map]
    for i0, i1 in runs(below):
        if i1 - i0 < min_seconds:
            continue
        m = float(lm[i0:i1].min())
        if events and i0 - events[-1][1] < merge_gap and finite[int(events[-1][1]):i0].all():
            events[-1][1] = i1
            events[-1][2] = min(events[-1][2], m)
        else:
            events.append([i0, i1, m])
    df = pd.DataFrame(events, columns=["i0", "i1", "min_map"])
    out = pd.DataFrame({
        "onset": start + df["i0"].astype(float),
        "end": start + df["i1"].astype(float),
    })
    out["duration"] = out["end"] - out["onset"]
    out["min_map"] = df["min_map"].astype(float)
    out["suspect_artefact"] = out["min_map"] <= suspect_min_map
    return out[EVENT_COLUMNS]
