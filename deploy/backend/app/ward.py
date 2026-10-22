"""The simulated ward: 50 recorded test cases replayed as if live, with the models run as the clock reaches them.

Clock. A virtual time V (seconds) runs at 1x from a fixed epoch, so every viewer at 1x sees the same ward; a viewer
may run faster (V advances `speed` times wall time from where it joined). Bed i plays its case in a loop, shifted by a
fixed phase so the beds are at different points of their cases: case time t = start + ((V + phase) mod duration).

Models. One prediction row per 30 s of case time (the rows of features_v8: inputs computed from data up to that
time). When a bed's clock passes a row, the four models are run on it (batched over beds); probabilities, alarm
states and alarms are cached per (bed, model) and shared by every viewer. Nothing after the current time is sent.
Alarms: uc04 evaluation rule (2 consecutive eligible predictions at or above the threshold, at most one per 300 s,
re-armed only after the probability drops below the threshold).

SAFEANES_ENGINE=replay reads the offline test probabilities (reference/) instead of running the models, for working
on the UI on a machine where the models must not run.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .models import HORIZONS, Adapter, explain, load_models

PERSISTENCE, COOLDOWN, CADENCE = 2, 300.0, 30.0
EVENT_KNOWN_S = 60.0          # an event (MAP < 65 mmHg for >= 60 s) is known 60 s after its onset
EPOCH = 1_767_225_600.0       # 2026-01-01 00:00 UTC: V = 0
VITALS = ("map", "sbp", "dbp", "hr", "spo2")
CONTEXT = ("f1a_min_since_anestart", "f1a_min_since_opstart", "f1a_min_since_last_event", "map_drop_pct",
           "f1a_map_min_m15", "f1b_ppf", "f1b_rftn_d5", "f2_exp_sevo")
GOLDEN = 0.6180339887498949


class AlarmState:
    """uc04.alarms.replay_reference for one case and horizon, one prediction at a time (rearm='drop_below')."""

    def __init__(self, threshold: float):
        self.thr, self.streak, self.active = threshold, 0, False
        self.next_allowed, self.prev_t = -np.inf, -np.inf

    def step(self, t: float, eligible: bool, p: float) -> bool:
        """Advance to the prediction at t; True when an alarm fires here."""
        if t - self.prev_t > CADENCE:
            self.streak, self.active = 0, False
        self.prev_t = t
        if not eligible or not np.isfinite(p) or p < self.thr:
            self.streak, self.active = 0, False
            return False
        self.streak += 1
        if not self.active and self.streak >= PERSISTENCE and t >= self.next_allowed:
            self.active, self.next_allowed = True, t + COOLDOWN
            return True
        return False


def replay_alarms(time: np.ndarray, eligible: np.ndarray, prob: np.ndarray, thr: float) -> list[float]:
    st = AlarmState(thr)
    return [float(t) for t, e, p in zip(time, eligible, prob) if st.step(float(t), bool(e), float(p))]


@dataclass
class Track:
    """Per (bed, model): probabilities and alarm states of the rows computed so far (rows 0 .. n-1)."""
    prob: np.ndarray                                        # [rows, 5], NaN where not eligible / not computed
    n: int = 0
    states: list = field(default_factory=list)              # AlarmState per horizon
    alarm_on: np.ndarray | None = None                      # [rows, 5] alarm active after the row
    fired: list = field(default_factory=list)               # per horizon: row indices where an alarm fired


class Bed:
    def __init__(self, info: dict, data: Path, inputs: list[str]):
        self.info = info
        self.n_bed, self.caseid = info["bed"], info["caseid"]
        df = pd.read_parquet(data / "cases" / f"{self.caseid}.parquet")
        self.frame = df
        self.time = df["time"].to_numpy(float)
        self.eligible = df["eligible"].to_numpy(bool)
        self.y = {h: df[f"y_{h}"].to_numpy() for h in HORIZONS}
        self.inputs = df[inputs]
        w = pd.read_parquet(data / "wave" / f"{self.caseid}.parquet")
        self.wt = w["time"].to_numpy(float)
        self.wave = {k: w[k].to_numpy(float) for k in VITALS}
        self.start, self.end = float(self.time[0]), float(self.time[-1]) + CADENCE
        self.duration = self.end - self.start
        self.phase = ((self.n_bed * GOLDEN) % 1.0) * self.duration
        self.surgery_start = float(info["surgery_start_s"])
        self.events = info["events"]

    def clock(self, v: float) -> tuple[float, int]:
        """(case time, loop number) at virtual time v."""
        k, r = divmod(v + self.phase, self.duration)
        return self.start + r, int(k)

    def row(self, t: float) -> int:
        """Index of the last prediction row at or before t (-1 before the first)."""
        return int(np.searchsorted(self.time, t, side="right")) - 1

    def vitals_at(self, t: float) -> dict:
        i = int(np.searchsorted(self.wt, t, side="right")) - 1
        return {k: (None if i < 0 or not np.isfinite(v[i]) else round(float(v[i]), 1)) for k, v in self.wave.items()}

    def wave_between(self, t0: float, t1: float, keys=VITALS) -> dict:
        """2 s samples with t0 < time <= t1."""
        a, b = np.searchsorted(self.wt, [t0, t1], side="right")
        return {"t": self.wt[a:b].tolist(),
                **{k: [None if not np.isfinite(x) else round(float(x), 1) for x in self.wave[k][a:b]] for k in keys}}

    def known_events(self, t: float) -> list[dict]:
        """Events known at t (60 s after onset), the end clipped to t while still running."""
        out = []
        for e in self.events:
            if e["onset_s"] + EVENT_KNOWN_S <= t:
                out.append({**e, "end_s": min(e["end_s"], t), "ongoing": e["end_s"] > t})
        return out


class Ward:
    def __init__(self, data: Path, mode: str = "live"):
        self.data, self.mode = Path(data), mode
        meta = json.loads((self.data / "ward.json").read_text(encoding="utf-8"))
        self.meta, self.inputs = meta, meta["inputs"]
        self.beds = {b["bed"]: Bed(b, self.data, self.inputs) for b in meta["beds"]}
        names = meta["models"]
        if mode == "live":
            self.models: dict[str, Adapter] = load_models(self.data / "models", names)
        else:  # meta only; probabilities from the offline reference
            self.models = {n: Adapter(self.data / "models" / n) for n in names}
            self.reference = {n: pd.read_parquet(self.data / "reference" / f"{n}.parquet").set_index(["caseid", "time"])
                              for n in names}
        self.tracks: dict[tuple[int, str], Track] = {}
        for b in self.beds.values():
            for n, m in self.models.items():
                self.tracks[(b.n_bed, n)] = Track(np.full((len(b.time), len(HORIZONS)), np.nan), 0,
                                                   [AlarmState(m.threshold[h]) for h in HORIZONS],
                                                   np.zeros((len(b.time), len(HORIZONS)), bool),
                                                   [[] for _ in HORIZONS])
        self.lock = threading.Lock()
        self.rows_scored = 0

    # ------------------------------------------------------------------ models, run as the clock reaches a row
    def ensure(self, upto: dict[int, int]) -> None:
        """Compute every row up to upto[bed] (inclusive) for every model, batched over beds."""
        with self.lock:
            for name, m in self.models.items():
                need = [(b, self.tracks[(b, name)].n, i + 1) for b, i in upto.items() if self.tracks[(b, name)].n <= i]
                if not need:
                    continue
                if self.mode == "live":
                    X = np.concatenate([self.beds[b].inputs.iloc[a:z].to_numpy(np.float64) for b, a, z in need])
                    P = m.prob(X)
                    self.rows_scored += len(X)
                else:
                    ref = self.reference[name]
                    P = np.concatenate([ref.reindex(pd.MultiIndex.from_arrays(
                        [np.full(z - a, self.beds[b].caseid), self.beds[b].time[a:z]]))[[f"p_{h}" for h in HORIZONS]]
                        .to_numpy(float) for b, a, z in need])
                k = 0
                for b, a, z in need:
                    bed, tr = self.beds[b], self.tracks[(b, name)]
                    p = np.where(bed.eligible[a:z, None], P[k:k + z - a], np.nan)
                    k += z - a
                    tr.prob[a:z] = p
                    for i in range(a, z):
                        for j, st in enumerate(tr.states):
                            if st.step(float(bed.time[i]), bool(bed.eligible[i]), float(tr.prob[i, j])):
                                tr.fired[j].append(i)
                            tr.alarm_on[i, j] = st.active
                    tr.n = z

    # ------------------------------------------------------------------ what a viewer sees at virtual time v
    def snapshot(self, v: float) -> list[dict]:
        pos = {b: bed.clock(v) for b, bed in self.beds.items()}
        idx = {b: self.beds[b].row(t) for b, (t, _) in pos.items()}
        self.ensure({b: i for b, i in idx.items() if i >= 0})
        out = []
        for b, bed in self.beds.items():
            t, loop = pos[b]
            i = idx[b]
            risk, alarm, n_alarms = {}, {}, {}
            for name in self.models:
                tr = self.tracks[(b, name)]
                risk[name] = None if i < 0 or not bed.eligible[i] else [round(float(x), 4) for x in tr.prob[i]]
                alarm[name] = [False] * len(HORIZONS) if i < 0 else tr.alarm_on[i].tolist()
                n_alarms[name] = [sum(r <= i for r in f) for f in tr.fired]
            ev = bed.known_events(t)
            out.append({"bed": b, "t": round(t, 1), "loop": loop, "elapsed_s": round(t - bed.start, 1),
                        "phase": "pre_incision" if t <= bed.surgery_start else "surgery",
                        "eligible": bool(i >= 0 and bed.eligible[i]), "row_time": None if i < 0 else bed.time[i],
                        "vitals": bed.vitals_at(t), "risk": risk, "alarm": alarm, "n_alarms": n_alarms,
                        "in_event": any(e["ongoing"] for e in ev), "n_events": len(ev)})
        return out

    def history(self, b: int, v: float, wave_s: float = 1800.0) -> dict:
        bed = self.beds[b]
        t, loop = bed.clock(v)
        i = bed.row(t)
        if i >= 0:
            self.ensure({b: i})
        n = i + 1
        onsets = np.array([e["onset_s"] for e in bed.events], float)
        risk, alarms, thresholds = {}, {}, {}
        for name, m in self.models.items():
            tr = self.tracks[(b, name)]
            risk[name] = {str(h): [None if not np.isfinite(x) else round(float(x), 4) for x in tr.prob[:n, k]]
                          for k, h in enumerate(HORIZONS)}
            alarms[name] = {str(h): [classify(float(bed.time[r]), h, t, onsets, bed.y[h][r])
                                     for r in tr.fired[k] if r < n] for k, h in enumerate(HORIZONS)}
            thresholds[name] = {str(h): m.threshold[h] for h in HORIZONS}
        row = bed.inputs.iloc[i] if i >= 0 else None
        ctx = {c: (None if row is None or c not in row or pd.isna(row[c]) else float(row[c])) for c in CONTEXT}
        return {"bed": b, "info": bed.info, "t": t, "loop": loop, "start_s": bed.start, "end_s": bed.end,
                "surgery_start_s": bed.surgery_start, "time_s": bed.time[:n].tolist(),
                "eligible": bed.eligible[:n].tolist(), "risk": risk, "alarms": alarms, "thresholds": thresholds,
                "events": bed.known_events(t), "wave": bed.wave_between(t - wave_s, t), "vitals": bed.vitals_at(t),
                "context": ctx}

    def explain(self, b: int, v: float, model: str, h: int) -> dict:
        bed = self.beds[b]
        t, _ = bed.clock(v)
        i = bed.row(t)
        if i < 0:
            return {"available": False, "reason": "chưa có dự báo"}
        x = bed.inputs.iloc[i].to_numpy(np.float64)
        m = self.models[model]
        proxy = None
        if self.mode != "live":
            return {"available": False, "reason": "chế độ replay không chạy mô hình"}
        if not m.explains:
            proxy, m = m.name, self.models["lgbm_v8"]
        ex = explain(m, x, h)
        return {"available": True, "time_s": float(bed.time[i]), "eligible": bool(bed.eligible[i]),
                "model": m.name, "proxy_for": proxy, **ex}


def classify(ta: float, h: int, now: float, onsets: np.ndarray, y: int) -> dict:
    """Outcome of an alarm at ta as known at `now`: true if an event starts in (ta, ta + h] and is already known,
    pending while ta + h is still ahead, censored if the label at ta is unknown, false otherwise."""
    nxt = onsets[(onsets > ta) & (onsets <= ta + h)]
    if len(nxt) and nxt[0] + EVENT_KNOWN_S <= now:
        return {"time_s": ta, "kind": "true", "lead_s": float(nxt[0] - ta)}
    if ta + h > now:
        return {"time_s": ta, "kind": "pending", "lead_s": None}
    return {"time_s": ta, "kind": "censored" if y == -1 else "false", "lead_s": None}
