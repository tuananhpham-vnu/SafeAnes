"""Patient-level bootstrap (plan section 8.5).

Subjects are drawn with replacement (same number of subjects). Alarms are
replayed once on the original data; each draw re-weights the per-case counts
(`case_metrics`) and the rows (sample weights) by how often their subject was
drawn. `paired` uses the same draws for two models and reports the CI of the
difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import CASE_COLUMNS, row_metrics, summarize_counts

MAIN = ("auroc", "auprc", "event_sensitivity", "false_alarms_per_hour", "alarm_time_fraction")


class Bootstrap:
    def __init__(self, subjects_rows: np.ndarray, subjects_cases: np.ndarray, repeats: int, seed: int):
        subj = np.unique(np.concatenate([subjects_rows, subjects_cases]))
        self.row_subj = np.searchsorted(subj, subjects_rows)
        self.case_subj = np.searchsorted(subj, subjects_cases)
        rng = np.random.default_rng(seed)
        self.draws = [np.bincount(rng.integers(0, len(subj), len(subj)), minlength=len(subj))
                      for _ in range(repeats)]

    def metrics(self, case_table: pd.DataFrame, y: np.ndarray, p: np.ndarray, with_rows: bool = True) -> list[dict]:
        counts = case_table[list(CASE_COLUMNS)].to_numpy(float)
        out = []
        for w in self.draws:
            m = summarize_counts(dict(zip(CASE_COLUMNS, w[self.case_subj] @ counts)))
            if with_rows:
                m.update(row_metrics(y, p, w[self.row_subj].astype(float)))
            out.append(m)
        return out


def ci(draws: list[dict], keys=MAIN, level: float = 0.95) -> dict:
    a = (1 - level) / 2
    out = {}
    for k in keys:
        v = np.array([d.get(k, np.nan) for d in draws], float)
        v = v[np.isfinite(v)]
        out[f"{k}_lo"] = float(np.quantile(v, a)) if len(v) else np.nan
        out[f"{k}_hi"] = float(np.quantile(v, 1 - a)) if len(v) else np.nan
    return out


def paired_ci(draws_a: list[dict], draws_b: list[dict], keys=MAIN, level: float = 0.95) -> dict:
    """CI of (a - b) with the same draws for both models."""
    diff = [{k: d1.get(k, np.nan) - d2.get(k, np.nan) for k in keys} for d1, d2 in zip(draws_a, draws_b)]
    out = ci(diff, keys, level)
    out.update({f"{k}_diff_mean": float(np.nanmean([d[k] for d in diff])) for k in keys})
    return out
