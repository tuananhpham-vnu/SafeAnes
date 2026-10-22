"""Sampling protocols of published hypotension predictors, so our scores can be put next to theirs.

`rows` is one frame of prediction times (main-cohort rows of a split, any order) with caseid, time, map_current,
map_min_w30, map_missing_w30; `events` the events table of that split (suspect artefacts are dropped here).
Each protocol returns positions into `rows` of its positives and negatives for a horizon h (seconds).

  every     all eligible rows with a known label at h (our own protocol)
  hatib     Hatib 2018 (HPI): positive = the time h before an event onset; negative = the centre of each
            non-overlapping 30-min stretch with MAP > 75 throughout, >= 20 min from any event
  lee       Lee 2021 / HypoBridCast 2026: positive = the latest time in (onset - h - 30 s, onset - h];
            negative = any time inside a >= 20-min run with MAP > 65 (30-s minimum, < 50 % missing) and no
            event onset in the next 20 min

These protocols drop the hard negatives (MAP low but no event), which is why every model, MAP itself included,
scores higher under them (week 13).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_eda import auroc

STEP = 30


def _events(events: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    ev = events[~events["suspect_artefact"].astype(bool)]
    return ev[ev["caseid"].isin(rows["caseid"].unique())]


def positives(rows: pd.DataFrame, events: pd.DataFrame, h: int) -> np.ndarray:
    """Per event: the latest row with time in (onset - h - 30, onset - h] (none when that row is missing)."""
    ev = _events(events, rows)
    out = []
    for c, g in rows.groupby("caseid", sort=False):
        e = ev.loc[ev["caseid"] == c, "onset"].to_numpy(float)
        if not len(e):
            continue
        order = np.argsort(g["time"].to_numpy(float), kind="stable")
        t = g["time"].to_numpy(float)[order]
        idx = g.index.to_numpy()[order]
        for target in e - h:
            k = np.searchsorted(t, target, side="right") - 1
            if k >= 0 and t[k] > target - STEP:
                out.append(idx[k])
    return np.asarray(out, np.int64)


def _runs(ok: np.ndarray, t: np.ndarray, max_gap: float):
    """(start, end) positions of runs of ok rows with consecutive time steps <= max_gap."""
    i, n = 0, len(t)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1] and t[j + 1] - t[j] <= max_gap:
            j += 1
        yield i, j
        i = j + 1


def hatib_negatives(rows: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    ev = _events(events, rows)
    evc = {c: g[["onset", "end"]].to_numpy(float) for c, g in ev.groupby("caseid")}
    out = []
    for c, g in rows.sort_values(["caseid", "time"]).groupby("caseid", sort=False):
        t = g["time"].to_numpy(float)
        ok = (g["map_min_w30"].to_numpy() > 75) & (g["map_current"].to_numpy() > 75)
        for on, en in evc.get(c, np.empty((0, 2))):
            ok &= (t < on - 1200) | (t > en + 1200)
        for i, j in _runs(ok, t, STEP + 1):
            k = i
            while k + 59 <= j:  # 30-min windows of 60 rows; the centre row
                out.append(g.index[k + 30])
                k += 60
    return np.asarray(out, np.int64)


def lee_negatives(rows: pd.DataFrame, events: pd.DataFrame, stable_s: int = 1200, excl_s: int = 1200) -> np.ndarray:
    ev = _events(events, rows)
    onsets = {c: g["onset"].to_numpy(float) for c, g in ev.groupby("caseid")}
    out = []
    for c, g in rows.sort_values(["caseid", "time"]).groupby("caseid", sort=False):
        t = g["time"].to_numpy(float)
        ok = (g["map_min_w30"].to_numpy() > 65) & (g["map_missing_w30"].to_numpy() < 0.5)
        on = onsets.get(c, np.array([]))
        for i, j in _runs(ok, t, STEP):
            if t[j] - t[i] + STEP < stable_s:
                continue
            for k in range(i, j + 1):
                if not ((on > t[k]) & (on <= t[k] + excl_s)).any():
                    out.append(g.index[k])
    return np.asarray(out, np.int64)


def protocol_rows(rows: pd.DataFrame, events: pd.DataFrame, h: int, name: str,
                  negatives: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(row positions, binary labels) of one protocol. `negatives` caches the h-independent negatives."""
    if name == "every":
        y = rows[f"y_{h}"].to_numpy()
        idx = np.flatnonzero(rows["eligible"].to_numpy(bool) & np.isin(y, (0, 1)))
        return rows.index.to_numpy()[idx], (y[idx] == 1).astype(np.int8)
    neg = (negatives or {}).get(name)
    if neg is None:
        neg = hatib_negatives(rows, events) if name == "hatib" else lee_negatives(rows, events)
    pos = positives(rows, events, h)
    return np.r_[pos, neg], np.r_[np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)]


def paired_bootstrap(y: np.ndarray, scores: dict[str, np.ndarray], subject: np.ndarray, ref: str,
                     n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """AUROC of every score with a subject-bootstrap CI, and its difference to `ref` (paired: same resamples)."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(subject, return_inverse=True)
    rows_of = [np.flatnonzero(inv == i) for i in range(len(u))]
    names = list(scores)
    point = {n: auroc(y, scores[n]) for n in names}
    bs = {n: [] for n in names}
    for _ in range(n_boot):
        w = np.concatenate([rows_of[i] for i in rng.integers(0, len(u), len(u))])
        if y[w].min() == y[w].max():
            continue
        for n in names:
            bs[n].append(auroc(y[w], scores[n][w]))
    out = []
    for n in names:
        a, r = np.array(bs[n]), np.array(bs[ref])
        d = a - r
        out.append({"model": n, "auroc": point[n], "lo": np.percentile(a, 2.5), "hi": np.percentile(a, 97.5),
                    f"diff_vs_{ref}": point[n] - point[ref], "diff_lo": np.percentile(d, 2.5),
                    "diff_hi": np.percentile(d, 97.5), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())})
    return pd.DataFrame(out)
