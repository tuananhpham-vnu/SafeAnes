"""Build the sample tables of NB01 (plan sections 3.2, 3.5 to 3.8).

One worker call per case (`build_case`) returns labels, features, events,
wave_mask and aggregated report counts. The caller streams them into one parquet
file per split. Test outputs go to the separate `sealed` directory and are never
summarised (principle 1); report counts are only computed for non-test cases.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .columns import PPV_MODEL, PPV_PREP, all_model_cols, ppv_prep_to_model
from .events import detect_events
from .io import load_case, read_features
from .labels import LabelRules, event_eligible_h, label_case
from .ppv import beat_table, ppv30_features

TEST = "test"
SAMPLES_SECTIONS = ("input", "features", "labels")  # config sections the samples depend on
RULES = ("event", "any_low")
POLICIES = ("lenient", "strict")  # label tables: config negative rule for lenient
REPORT_POLICIES = ("lenient_possible", "lenient_fraction", "strict")  # label_report: new vs old negative rule


@dataclass(frozen=True)
class BuildParams:
    """Plain, picklable parameters for `build_case` (taken from the config)."""
    rules: LabelRules
    exclusion_reset: str
    label_policy: str
    windows: tuple[int, ...]
    recompute_ppv: bool
    ppv: dict = field(default_factory=dict)  # sub, min_beats, max_period_cv, rr_range, min_etco2
    suspect_min_map: float = 30.0
    min_map_coverage: float = 0.10

    @classmethod
    def from_config(cls, cfg) -> "BuildParams":
        F, L = cfg.features, cfg.labels
        return cls(rules=LabelRules.from_config(L), exclusion_reset=L.exclusion_reset, label_policy=L.label_policy,
                   windows=tuple(F.windows_seconds), recompute_ppv=bool(F.recompute_ppv),
                   ppv={"sub": int(F.ppv_subwindow_seconds), "min_beats": int(F.ppv_min_beats),
                        "max_period_cv": float(F.ppv_max_period_cv), "rr_range": tuple(F.ventilated_rr),
                        "min_etco2": float(F.ventilated_min_etco2)},
                   suspect_min_map=float(L.suspect_artefact_min_map),
                   min_map_coverage=float(cfg.input.min_map_coverage))


@dataclass(frozen=True)
class CaseSpec:
    caseid: int
    subjectid: int
    split: str
    has_wave: bool
    map_coverage: float
    baseline_source: int

    def in_main_cohort(self, min_map_coverage: float) -> bool:
        return bool(self.has_wave and self.map_coverage >= min_map_coverage)


def case_specs(qc: pd.DataFrame) -> list[CaseSpec]:
    return [CaseSpec(int(r.caseid), int(r.subjectid), str(r.split), bool(r.has_wave), float(r.map_coverage),
                     int(r.baseline_source) if pd.notna(r.baseline_source) else 0) for r in qc.itertuples()]


def model_columns(params: BuildParams) -> list[str]:
    return all_model_cols(params.windows, PPV_MODEL if params.recompute_ppv else PPV_PREP)


def build_case(prep, spec: CaseSpec, params: BuildParams) -> dict:
    """All NB01 outputs for one case."""
    z = load_case(prep, spec.caseid, ["start", "label_map", "wave_mask", "values", "channels", "beats",
                                      "beat_table_columns"])
    start = float(z["start"])
    lm = z["label_map"]
    prep_cols = all_model_cols(params.windows, PPV_PREP)
    f = read_features(prep, spec.caseid, ["caseid", "time", *prep_cols])
    times = f["time"].to_numpy(float)
    r = params.rules
    events = detect_events(lm, start, threshold=r.threshold, min_seconds=r.confirm_seconds,
                           merge_gap=r.merge_gap, suspect_min_map=params.suspect_min_map)
    channels = [str(c) for c in z["channels"]]
    grid_map = z["values"][:, channels.index("map")]
    lab = label_case(times, f["map_current"].to_numpy(), f["map_missing_w120"].to_numpy(), lm, start, events,
                     rules=r, grid_map=grid_map, policies=("lenient", *REPORT_POLICIES), exclusion_rules=RULES)

    # ---- features: 342 model columns, PPV recomputed
    feats = f[["caseid", "time"]].copy()
    ppv_nan = {}
    if params.recompute_ppv:
        beats = beat_table(z["beats"], z["beat_table_columns"])
        p = params.ppv
        ppv = ppv30_features(times, start, beats, f["rr_current"].to_numpy(), f["etco2_current"].to_numpy(),
                             params.windows, sub=p["sub"], min_beats=p["min_beats"],
                             max_period_cv=p["max_period_cv"], rr_range=p["rr_range"], min_etco2=p["min_etco2"])
        for W in params.windows:
            ppv_nan[W] = (int(f[f"{PPV_PREP}_w{W}"].isna().sum()), int(ppv[f"{PPV_MODEL}_w{W}"].isna().sum()))
        body = f[prep_cols].rename(columns=ppv_prep_to_model)
        for W in params.windows:
            body[f"{PPV_MODEL}_w{W}"] = ppv[f"{PPV_MODEL}_w{W}"].to_numpy()
    else:
        body = f[prep_cols]
    feats = pd.concat([feats, body[model_columns(params)].astype(np.float32)], axis=1)

    # ---- labels table (config rule and policy)
    in_main = spec.in_main_cohort(params.min_map_coverage)
    rule, pol = params.exclusion_reset, params.label_policy
    labels = pd.DataFrame({
        "caseid": np.int64(spec.caseid), "subjectid": np.int64(spec.subjectid), "split": spec.split,
        "time": times, "exposure_seconds": lab["exposure_seconds"].to_numpy(float),
        "eligible": lab[f"eligible_{rule}"].to_numpy(bool),
        "abstain_reason": lab[f"abstain_reason_{rule}"].astype(str).to_numpy(),
        "has_wave": spec.has_wave, "in_main_cohort": in_main,
    })
    for h in r.horizons:
        labels[f"y_{h}"] = lab[f"y_{h}_{pol}"].to_numpy(np.int8)
    for h in r.horizons:  # why y_h is -1: end_of_case / possible_event / ... ("" when known)
        labels[f"unknown_reason_{h}"] = lab[f"unknown_reason_{h}_{pol}"].astype(str).to_numpy()

    ev = events.copy()
    ev.insert(0, "split", spec.split)
    ev.insert(0, "subjectid", np.int64(spec.subjectid))
    ev.insert(0, "caseid", np.int64(spec.caseid))
    for h in r.horizons:
        ev[f"eligible_{h}"] = event_eligible_h(events, times, labels["eligible"].to_numpy(), labels[f"y_{h}"], h)

    out = {
        "spec": spec, "start": start, "surgery_seconds": len(lm), "labels": labels, "features": feats,
        "events": ev, "wave_mask": np.asarray(z["wave_mask"], np.uint8),
    }
    if spec.split != TEST:  # nothing about test labels is ever aggregated
        out["report"] = _report_counts(spec, in_main, times, lab, events, r.horizons)
        out["abstain"] = labels.loc[~labels.eligible, "abstain_reason"].value_counts().to_dict()
        out["ppv_nan"] = ppv_nan
        out["rows"] = len(labels)
    return out


def _report_counts(spec, in_main, times, lab, events, horizons) -> list[dict]:
    rows = []
    for rule in RULES:
        el = lab[f"eligible_{rule}"].to_numpy(bool)
        for pol in REPORT_POLICIES:
            for h in horizons:
                y = lab[f"y_{h}_{pol}"].to_numpy()
                ye = y[el]
                why = lab[f"unknown_reason_{h}_{pol}"].to_numpy()[el]
                ev_ok = event_eligible_h(events, times, el, y, h)
                rows.append({"split": spec.split, "in_main_cohort": in_main, "exclusion_reset": rule,
                             "label_policy": pol, "horizon": h, "rows": len(y), "eligible": int(el.sum()),
                             "positive": int((ye == 1).sum()), "negative": int((ye == 0).sum()),
                             "unknown": int((ye == -1).sum()),
                             "unknown_end_of_case": int((why == "end_of_case").sum()),
                             "unknown_other": int(((ye == -1) & (why != "end_of_case")).sum()),
                             "events": len(events),
                             "events_eligible": int(ev_ok.sum())})
    return rows


# ------------------------------------------------------------------ summaries (non-test only)

def label_report(report_rows: Sequence[dict]) -> pd.DataFrame:
    df = pd.DataFrame(report_rows)
    df = df[df["in_main_cohort"]]
    keys = ["split", "horizon", "label_policy", "exclusion_reset"]
    g = df.groupby(keys, sort=False)[["rows", "eligible", "positive", "negative", "unknown", "unknown_end_of_case",
                                      "unknown_other", "events", "events_eligible"]].sum().reset_index()
    known = g["positive"] + g["negative"]
    g["positive_rate"] = (g["positive"] / known.where(known > 0)).round(5)
    g["unknown_rate"] = (g["unknown"] / g["eligible"].where(g["eligible"] > 0)).round(5)
    g["unknown_end_of_case_rate"] = (g["unknown_end_of_case"] / g["eligible"].where(g["eligible"] > 0)).round(5)
    g["unknown_other_rate"] = (g["unknown_other"] / g["eligible"].where(g["eligible"] > 0)).round(5)
    g.insert(0, "cohort", "D")
    return g.sort_values(keys).reset_index(drop=True)


def dryrun_subset(events: pd.DataFrame, index: pd.DataFrame, sizes: dict[str, int], seed: int) -> dict:
    """Cases of the main cohort with at least one event that has an eligible row at 5 min."""
    ok = events[events["eligible_300"]].caseid.unique()
    pool = index[index.in_main_cohort & index.caseid.isin(ok)]
    out = {"seed": seed, "rule": "main cohort, >= 1 event with eligible_300", "cases": {}}
    for split, n in sizes.items():
        ids = pool[pool.split == split].caseid.sort_values()
        take = ids.sample(min(n, len(ids)), random_state=seed) if len(ids) else ids
        out["cases"][split] = sorted(int(c) for c in take)
    return out


def column_stats(read_columns, columns: Sequence[str], mask: np.ndarray, batch: int = 40) -> dict:
    """median / mean / std of `columns` over rows where `mask`, reading `batch` columns at a time."""
    stats = {}
    for i in range(0, len(columns), batch):
        cols = list(columns[i:i + batch])
        block = read_columns(cols)
        for c in cols:
            x = block[c].to_numpy(np.float64)[mask]
            x = x[np.isfinite(x)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                stats[c] = {"median": float(np.median(x)) if len(x) else None,
                            "mean": float(x.mean()) if len(x) else None,
                            "std": float(x.std()) if len(x) else None, "n": int(len(x))}
    return stats
