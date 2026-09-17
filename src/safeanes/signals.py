"""Causal inputs and separately reconstructed retrospective labels (S01/S03).

Engineering choices including gap limits and recovery are defined in PROTOCOL.md.
"""

from dataclasses import dataclass
import numpy as np

from .config import Protocol


def validate_track(times, values):
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    if times.ndim != 1 or times.shape != values.shape:
        raise ValueError("Track arrays must be matching vectors")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Track timestamps must be finite, strictly increasing")
    return times, values


def causal_sample(times, values, grid, max_age):
    """No interpolation/backfill. Return values and age using only observations <= t."""
    times, values = validate_track(times, values)
    grid = np.asarray(grid, dtype=float)
    if len(times) == 0:
        return np.full(grid.shape, np.nan), np.full(grid.shape, np.inf)
    idx = np.searchsorted(times, grid, side="right") - 1
    safe = np.maximum(idx, 0)
    age = np.where(idx >= 0, grid - times[safe], np.inf)
    result = values[safe].copy()
    result[(idx < 0) | (age > max_age)] = np.nan
    return result, age


def label_grid(times, values, grid, max_gap):
    """Retrospective 1s intervals: supported only by successive finite measurements.

    A long gap invalidates the ENTIRE interval. Never extrapolate the final reading.
    Uses the next observation for quality support, hence must NEVER feed predictors.
    """
    times, values = validate_track(times, values)
    grid = np.asarray(grid, dtype=float)
    if len(times) < 2:
        return np.full(grid.shape, np.nan)
    first = np.searchsorted(times, grid, side="right") - 1
    last = np.searchsorted(times, grid + 1, side="left") - 1
    a, b = np.clip(first, 0, len(times) - 2), np.clip(last, 0, len(times) - 2)
    bad = ((np.diff(times) > max_gap) | ~np.isfinite(values[:-1]) | ~np.isfinite(values[1:]))
    prefix = np.r_[0, np.cumsum(bad)]
    valid = ((first >= 0) & (last < len(times) - 1) & (grid + 1 <= times[-1])
             & (prefix[b + 1] - prefix[a] == 0))
    result = values[a].copy()
    # Max means the ENTIRE 1s interval must be below threshold. This avoids
    # dropping every other cell when native timestamps have fractional seconds.
    for pos in np.flatnonzero(valid & (b > a)):
        result[pos] = np.max(values[a[pos]:b[pos] + 1])
    return np.where(valid, result, np.nan)



@dataclass(frozen=True)
class Event:
    onset: float
    end: float


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def detect_events(grid, label_map, protocol=Protocol()):
    grid, label_map = np.asarray(grid), np.asarray(label_map)
    if len(grid) != len(label_map) or (len(grid) > 1 and not np.allclose(np.diff(grid), 1)):
        raise ValueError("Label grid must have uniform 1-second steps")
    events = []
    for start, stop in runs(np.isfinite(label_map) & (label_map < protocol.map_threshold)):
        if stop - start < protocol.event_seconds:
            continue
        event = Event(float(grid[start]), float(grid[stop - 1] + 1))
        if events and event.onset - events[-1].end < protocol.recovery_seconds:
            prev = events[-1]
            between = (grid >= prev.end) & (grid < event.onset)
            if np.isfinite(label_map[between]).all():
                events[-1] = Event(prev.onset, event.end)
                continue
        events.append(event)
    return events


def future_label(t, horizon, grid, label_map, events, protocol=Protocol()):
    """-1 means censored. Require complete horizon + duration support for both classes."""
    left = np.searchsorted(grid, t, side="right")
    right = np.searchsorted(grid, t + horizon + protocol.event_seconds, side="right")
    future = label_map[left:right]
    expected = horizon + protocol.event_seconds
    if len(future) != expected or not np.isfinite(future).all():
        return -1
    return int(any(t < event.onset <= t + horizon for event in events))
