"""Metrics by phase of the case: pre-incision vs surgery (samples v3, DEVIATIONS 46).

Samples v3 start at the beginning of the recording; `surgery_start.csv` gives the incision
time per case. A prediction time, an alarm or an event onset at or before surgery_start is
"pre_incision", later ones are "surgery". Samples v2 start at incision, so all of their rows
are "surgery" and a v3 model scored on the "surgery" phase is the closest like-for-like
comparison with a v2 model (labels still differ slightly: events before incision now count
for merging and post-event exclusion).

Alarms are replayed on the whole case (an alarm raised just before incision can detect an
event just after); counts are then attributed by time: events by onset, alarms by alarm
time, exposure by prediction time.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json
from .loaders import load_split
from .metrics import Evaluator, row_metrics

PHASES = ("all", "pre_incision", "surgery")
DL_DIRS = {"dl": "dl_conv_tf", "dl_context": "dl_context"}  # artifacts/<dir>/W<W>/ -> model name


def read_surgery_start(samples: Path) -> dict[int, float] | None:
    p = Path(samples) / "surgery_start.csv"
    if not p.exists():
        return None
    ss = pd.read_csv(p)
    return dict(zip(ss.caseid.astype(int), ss.surgery_start.astype(float)))


def _pre(caseids: np.ndarray, times: np.ndarray, surgery_start: dict[int, float] | None) -> np.ndarray:
    if surgery_start is None:  # v2: every row is after incision
        return np.zeros(len(times), bool)
    s0 = np.array([surgery_start[int(c)] for c in caseids], float)
    return np.asarray(times, float) <= s0


def phase_metrics(ev: Evaluator, prob_sorted: np.ndarray, thr: float,
                  surgery_start: dict[int, float] | None, bins: int = 10) -> list[dict]:
    """One dict per phase with row metrics, event sensitivity, FA/h, PPV and lead time at `thr`."""
    p = np.asarray(prob_sorted, float)
    res = ev.evaluate(p, thr)
    row_case = ev.cases[ev.case_code]
    row_pre = _pre(row_case, ev.time, surgery_start)
    ev_pre = _pre(ev.cases[ev.ev_case], ev.ev_onset, surgery_start)
    al = res.alarms
    al_pre = _pre(al["caseid"].to_numpy(), al["time"].to_numpy(), surgery_start)
    evaluable = ev.eligible & (ev.y != -1) & np.isfinite(p)
    detected = np.isfinite(res.event_lead)
    out = []
    for phase in PHASES:
        rm = np.ones(len(p), bool) if phase == "all" else (row_pre if phase == "pre_incision" else ~row_pre)
        em = np.ones(len(ev_pre), bool) if phase == "all" else (ev_pre if phase == "pre_incision" else ~ev_pre)
        am = np.ones(len(al), bool) if phase == "all" else (al_pre if phase == "pre_incision" else ~al_pre)
        elig_ev = em & ev.ev_eligible
        hours = float((ev.exposure * (evaluable & rm)).sum()) / 3600.0
        n_true = int((am & (al["kind"] == "true").to_numpy()).sum())
        n_false = int((am & (al["kind"] == "false").to_numpy()).sum())
        sel = ev.eligible & rm
        lead = res.event_lead[elig_ev & detected]
        out.append({
            "phase": phase, **row_metrics(ev.y[sel], p[sel], bins=bins),
            "events_eligible": int(elig_ev.sum()), "events_detected": int((elig_ev & detected).sum()),
            "event_sensitivity": float((elig_ev & detected).sum() / elig_ev.sum()) if elig_ev.sum() else np.nan,
            "true_alarms": n_true, "false_alarms": n_false, "evaluable_hours": round(hours, 1),
            "false_alarms_per_hour": n_false / hours if hours > 0 else np.nan,
            "alarm_ppv": n_true / (n_true + n_false) if n_true + n_false else np.nan,
            "lead_median_s": float(np.median(lead)) if len(lead) else np.nan,
        })
    return out


def evaluator_for(labels: pd.DataFrame, events: pd.DataFrame, h: int, ev_cfg) -> Evaluator:
    return Evaluator(labels, events[events.split == "validation"], h, persistence=ev_cfg.alarm_persistence,
                     cooldown=ev_cfg.alarm_cooldown_seconds, early_lead=ev_cfg.early_lead_seconds,
                     rearm=ev_cfg.alarm_rearm)


def align_probability(labels: pd.DataFrame, preds: pd.DataFrame, col: str) -> np.ndarray:
    """`preds[col]` in the row order of `labels` (NaN where a (caseid, time) row has no prediction)."""
    s = pd.Series(preds[col].to_numpy(float), index=pd.MultiIndex.from_frame(preds[["caseid", "time"]]))
    return s.reindex(pd.MultiIndex.from_frame(labels[["caseid", "time"]])).to_numpy(float)


def find_predictions(work_dirs, kinds=("tabular", "dl")) -> list[dict]:
    """[{model, window, horizon, path, column, threshold}] for every validation prediction set under
    `work_dirs`: tabular `artifacts/tabular/**/val_predictions.parquet` (column `probability`) and DL
    `**/dl/W<W>/` or `**/dl_context/W<W>/val_predictions.parquet` (columns `probability_<h>`), each with its threshold.json."""
    out, seen = [], set()
    for w in map(Path, work_dirs):
        if "tabular" in kinds:
            for p in sorted(w.glob("artifacts/tabular/**/val_predictions.parquet")):
                parts = p.parent.relative_to(w / "artifacts" / "tabular").parts
                key = (parts[0], int(parts[1][1:]) if len(parts) == 3 else None, int(parts[-1][1:]))
                if key not in seen and (p.parent / "threshold.json").exists():
                    seen.add(key)
                    out.append(dict(model=key[0], window=key[1], horizon=key[2], path=p, column="probability",
                                    threshold=read_json(p.parent / "threshold.json")["threshold"]))
        if "dl" in kinds:
            for p in sorted(w.glob("**/dl*/W*/val_predictions.parquet")):
                model = DL_DIRS.get(p.parent.parent.name)
                if model is None or not re.fullmatch(r"W\d+", p.parent.name) or not (p.parent / "threshold.json").exists():
                    continue
                W = int(p.parent.name[1:])
                for h, t in read_json(p.parent / "threshold.json").items():
                    key = (model, W, int(h))
                    if key not in seen:
                        seen.add(key)
                        out.append(dict(model=model, window=W, horizon=int(h), path=p,
                                        column=f"probability_{h}", threshold=t["threshold"]))
    return out


def evaluate_run(name: str, samples: Path, work_dirs, ev_cfg, kinds=("tabular", "dl"), subset: dict | None = None,
                 log=print) -> list[dict]:
    """Phase metrics of every validation prediction set of one run (version) at its own threshold.
    `subset` ({split: caseids}, as --subset) restricts the validation cases."""
    samples = Path(samples)
    labels, _ = load_split(samples, "validation", [])
    if subset is not None:
        labels = labels[labels["caseid"].isin(subset.get("validation", []))].reset_index(drop=True)
    events = pd.read_parquet(samples / "events.parquet")
    ss = read_surgery_start(samples)
    preds = find_predictions(work_dirs, kinds)
    log(f"{name}: {len(preds)} prediction sets, {len(labels):,} validation rows, "
        f"phases {'pre_incision/surgery' if ss else 'surgery only'}")
    rows, evaluators, cache = [], {}, {}
    for d in preds:
        h = d["horizon"]
        if h not in evaluators:
            evaluators[h] = evaluator_for(labels, events, h, ev_cfg)
        ev = evaluators[h]
        if d["path"] not in cache:
            cache = {d["path"]: pd.read_parquet(d["path"])}
        prob = align_probability(labels, cache[d["path"]], d["column"])
        for r in phase_metrics(ev, ev.sort(prob), d["threshold"], ss, bins=ev_cfg.ece_bins):
            if ss is None and r["phase"] == "pre_incision":
                continue
            rows.append({"version": name, "model": d["model"], "window": d["window"], "horizon": h,
                         "threshold": d["threshold"], **r})
    return rows
