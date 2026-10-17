"""Personalised hypotension label (week 14, plan C1): MAP below the patient's own threshold
thr = max(65, 0.8 x baseline MAP) for >= 60 s, i.e. "MAP < 65 or a fall of >= 20 % from baseline".

Samples v3 have no 1-s MAP, only 30-s rows with the min / max / missing share of MAP over (t - 30, t]. Events are
therefore built from 30-s blocks:
  full  block: map_max_w30 < thr and no missing second -> MAP below thr the whole 30 s
  part  block: map_min_w30 < thr (some second below), or MAP missing in the block
An event is a run of >= 2 consecutive full blocks (>= 60 s below), onset = start of its first full block; runs that
end and restart within 120 s merge as in uc04.events. A single full block next to a part block could hide a 60-s run,
so such rows cannot be negatives (label -1), like the `possible_event` rule of uc04.labels.
`check_against_abs` applies the same procedure with thr = 65 and compares with the shipped y_h labels.

Baseline MAP: the case's own baseline used by `map_drop_pct` (pre-induction NIBP, first NIBP or first 5 min of the
arterial line; INTERPRETATION_NOTES 4), recovered as map_current / (1 - map_drop_pct / 100). Alternative
`running`: the running case median up to t (causal), from f1a_map_rel_case_median.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BLOCK = 30.0
MERGE_GAP = 120.0
POST_EVENT = 120.0
CONFIRM = 60.0
COLUMNS = ["caseid", "time", "map_current", "map_min_w30", "map_max_w30", "map_missing_w30", "map_missing_w120",
           "map_drop_pct"]


def baseline_map(f: pd.DataFrame, kind: str = "case") -> np.ndarray:
    """Per-row baseline MAP. `case`: constant per case (median over rows); `running`: causal running median."""
    with np.errstate(all="ignore"):
        if kind == "case":
            b = f["map_current"] / (1 - f["map_drop_pct"] / 100)
            return f.assign(_b=b).groupby("caseid")["_b"].transform("median").to_numpy(float)
        if kind == "running":
            return (f["map_current"] / f["f1a_map_rel_case_median"]).to_numpy(float)
    raise ValueError(kind)


def personal_threshold(f: pd.DataFrame, kind: str = "case", fall: float = 0.2, floor: float = 65.0) -> np.ndarray:
    b = baseline_map(f, kind)
    return np.where(np.isfinite(b), np.maximum(floor, (1 - fall) * b), np.nan)


def case_events(t: np.ndarray, mx: np.ndarray, mn: np.ndarray, miss: np.ndarray, thr: np.ndarray
                ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Events of one case (rows sorted by time, 30-s apart) and per-row full / part block flags."""
    full = (mx < thr) & (miss <= 0) & np.isfinite(thr)
    part = ~full & (((mn < thr) & np.isfinite(thr)) | (miss > 0) | ~np.isfinite(mx))
    ev: list[list[float]] = []
    i, n = 0, len(t)
    while i < n:
        if not full[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and full[j + 1] and t[j + 1] - t[j] <= BLOCK + 1:
            j += 1
        if j - i + 1 >= 2:  # >= 60 s below
            onset, end = t[i] - BLOCK, t[j]
            if ev and onset - ev[-1][1] < MERGE_GAP:
                ev[-1][1] = end
            else:
                ev.append([onset, end])
        i = j + 1
    return pd.DataFrame(ev, columns=["onset", "end"]), full, part


def case_labels(t: np.ndarray, mx, mn, miss, thr, map_current, miss120, horizons) -> tuple[dict, pd.DataFrame]:
    """y_h (1 / 0 / -1) and eligibility for one case."""
    events, full, part = case_events(t, mx, mn, miss, thr)
    on, en = events["onset"].to_numpy(), events["end"].to_numpy()
    end_case = t[-1]
    # blocks that could belong to a hidden >= 60 s run: a full or part block next to a part block, or a part block
    # next to a full one (a lone full block between blocks that are above threshold cannot reach 60 s)
    prev_full, next_full = np.r_[False, full[:-1]], np.r_[full[1:], False]
    prev_part, next_part = np.r_[False, part[:-1]], np.r_[part[1:], False]
    possible = ((full | part) & (prev_part | next_part)) | (part & (prev_full | next_full))
    cum = np.r_[0, np.cumsum(possible)]
    out = {}
    for h in horizons:
        pos = np.searchsorted(on, t, "right") < np.searchsorted(on, t + h, "right")  # an onset in (t, t + h]
        lo, hi = np.searchsorted(t, t, "right"), np.searchsorted(t, t + h + CONFIRM, "right")
        hidden = cum[hi] - cum[lo] > 0  # a possible block in (t, t + h + 60]
        y = np.where(pos, 1, np.where((t + h + CONFIRM > end_case) | hidden, -1, 0)).astype(np.int8)
        out[h] = y
    in_ev = ((on[None, :] <= t[:, None]) & (t[:, None] < en[None, :] + POST_EVENT)).any(1) if len(on) else \
        np.zeros(len(t), bool)
    with np.errstate(invalid="ignore"):
        elig = np.isfinite(map_current) & np.isfinite(thr) & (map_current >= thr) & ~in_ev & (miss120 <= 0.5)
    out["eligible"] = elig
    return out, events


def build(f: pd.DataFrame, thr: np.ndarray, horizons=(300, 600, 900)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Labels for every row of `f` (all main-cohort rows, any order) and the events table."""
    f = f.assign(_thr=thr, _row=np.arange(len(f))).sort_values(["caseid", "time"])
    labs, evs = [], []
    for cid, g in f.groupby("caseid", sort=False):
        res, ev = case_labels(g["time"].to_numpy(float), g["map_max_w30"].to_numpy(float),
                              g["map_min_w30"].to_numpy(float), g["map_missing_w30"].to_numpy(float),
                              g["_thr"].to_numpy(float), g["map_current"].to_numpy(float),
                              g["map_missing_w120"].to_numpy(float), horizons)
        d = pd.DataFrame({"_row": g["_row"].to_numpy(), "eligible": res["eligible"],
                          **{f"y_{h}": res[h] for h in horizons}})
        labs.append(d)
        evs.append(ev.assign(caseid=cid))
    lab = pd.concat(labs).sort_values("_row").drop(columns="_row").reset_index(drop=True)
    ev = pd.concat(evs, ignore_index=True)
    ev["duration"] = ev["end"] - ev["onset"]
    return lab, ev


def event_eligibility(ev: pd.DataFrame, f: pd.DataFrame, lab: pd.DataFrame, h: int) -> np.ndarray:
    """eligible_h of each event (uc04.labels.event_eligible_h): an eligible row with y_h == 1 in [onset - h, onset)."""
    ok = lab["eligible"].to_numpy(bool) & (lab[f"y_{h}"].to_numpy() == 1)
    rows = pd.DataFrame({"caseid": f["caseid"].to_numpy(), "time": f["time"].to_numpy(float)})[ok]
    out = np.zeros(len(ev), bool)
    by = {c: g["time"].to_numpy() for c, g in rows.groupby("caseid")}
    for i, (c, o) in enumerate(zip(ev["caseid"], ev["onset"])):
        t = by.get(c)
        out[i] = t is not None and bool(((t >= o - h) & (t < o)).any())
    return out


def check_against_abs(f: pd.DataFrame, shipped: pd.DataFrame, horizons=(300, 600, 900)) -> pd.DataFrame:
    """Rebuild the MAP < 65 labels with the block procedure and compare with the shipped y_h on rows where the
    shipped label is known and the shipped row is eligible."""
    lab, _ = build(f, np.full(len(f), 65.0), horizons)
    rows = []
    for h in horizons:
        s = shipped[f"y_{h}"].to_numpy()
        r = lab[f"y_{h}"].to_numpy()
        m = shipped["eligible"].to_numpy(bool) & np.isin(s, (0, 1))
        both = m & np.isin(r, (0, 1))
        rows.append({"horizon": h, "rows": int(m.sum()), "rebuilt_known": float(both.sum() / m.sum()),
                     "agreement": float((s[both] == r[both]).mean()),
                     "pos_shipped": int((s[both] == 1).sum()), "pos_rebuilt": int((r[both] == 1).sum()),
                     "sens_vs_shipped": float(((r == 1) & (s == 1) & both).sum() / max(((s == 1) & both).sum(), 1)),
                     "spec_vs_shipped": float(((r == 0) & (s == 0) & both).sum() / max(((s == 0) & both).sum(), 1))})
    return pd.DataFrame(rows)
