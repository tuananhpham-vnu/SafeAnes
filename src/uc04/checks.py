"""Data checks on prep_v1 (plan section 3.1).

Reads inputs only. `label_map` / `wave_mask` lengths come from the .npy header
inside each .npz, so no label values are loaded for any split (principle 1).
The report lists every check by name with its result, so a reviewer can see
that it ran, not only that nothing failed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .columns import PPV_PREP, all_model_cols
from .io import SPLITS, case_path, features_path, npz_shape, read_json, wave_path

# Plan 3.1: expected counts after dropping the 4 known failed cases.
EXPECTED = {
    #              cohort, processed, subjects, has_wave, cohort_D, subjects_D
    "train":       (2559, 2557, 2458, 2503, 2393, 2302),
    "calibration": (280, 280, 266, 273, 258, 244),
    "validation":  (271, 271, 266, 268, 248, 245),
    "test":        (516, 514, 500, 500, 464, 451),
}
COUNT_KEYS = ("cohort", "processed", "subjects", "has_wave", "cohort_D", "subjects_D")
WAVE_TOLERANCE_SAMPLES = 200  # |len(wave100) - 100 * surgery_seconds|

# Per-case checks, in report order.
CASE_CHECKS = (
    "label_map_length_header_only",
    "wave_mask_length_header_only",
    "case_start_integer",
    "wave100_dtype_float16",
    "wave100_length",
    "features_caseid",
    "features_model_columns_342",
    "features_duplicate_rows",
    "features_time_grid",
    "files_readable",
)


@dataclass(frozen=True)
class CaseRow:
    caseid: int
    surgery_seconds: int
    has_wave: bool


def check_case(prep: Path, row: CaseRow, required_cols: tuple[str, ...]) -> dict:
    """All per-case checks. `problems` is a list of (check_name, message)."""
    prep = Path(prep)
    cid, n = row.caseid, int(row.surgery_seconds)
    out: dict = {"caseid": cid, "problems": [], "missing": [], "duplicate_rows": 0, "rows": 0}
    probs = out["problems"]

    cpath = case_path(prep, cid)
    start = None
    if not cpath.exists():
        out["missing"].append("cases")
    else:
        try:
            for key in ("label_map", "wave_mask"):
                shape, _ = npz_shape(cpath, key)  # header only, values never loaded
                if shape != (n,):
                    probs.append((f"{key}_length_header_only", f"{key} shape {shape} != ({n},)"))
            with np.load(cpath, allow_pickle=False) as z:
                start = float(z["start"])  # input metadata, not a label
            if start != round(start):
                probs.append(("case_start_integer", f"start {start} is not an integer"))
        except Exception as exc:  # corrupt file
            probs.append(("files_readable", f"cases unreadable: {exc!r}"))

    if row.has_wave:
        wpath = wave_path(prep, cid)
        if not wpath.exists():
            out["missing"].append("wave100")
        else:
            try:
                w = np.load(wpath, mmap_mode="r")
                out["wave_len_diff"] = int(len(w) - 100 * n)
                if w.dtype != np.float16:
                    probs.append(("wave100_dtype_float16", f"wave100 dtype {w.dtype}"))
                if w.ndim != 1 or abs(out["wave_len_diff"]) > WAVE_TOLERANCE_SAMPLES:
                    probs.append(("wave100_length", f"wave100 length {len(w)} vs {100 * n}"))
            except Exception as exc:
                probs.append(("files_readable", f"wave100 unreadable: {exc!r}"))

    fpath = features_path(prep, cid)
    if not fpath.exists():
        out["missing"].append("features")
    else:
        try:
            names = set(pq.read_schema(fpath).names)
            absent = [c for c in required_cols if c not in names]
            if absent:
                probs.append(("features_model_columns_342", f"missing {len(absent)} columns, e.g. {absent[:3]}"))
            t = pq.read_table(fpath, columns=["caseid", "time"]).to_pandas()
            out["rows"] = len(t)
            if (t["caseid"] != cid).any():
                probs.append(("features_caseid", "features.caseid differs from file name"))
            out["duplicate_rows"] = int(t.duplicated(["caseid", "time"]).sum())
            if out["duplicate_rows"]:
                probs.append(("features_duplicate_rows", f"{out['duplicate_rows']} duplicate (caseid, time) rows"))
            if start is not None and len(t):
                k = (t["time"].to_numpy() - start) / 30.0
                if not np.allclose(k, np.round(k)) or k.min() < 1:
                    probs.append(("features_time_grid", "features.time is not start + 30k with k >= 1"))
        except Exception as exc:
            probs.append(("files_readable", f"features unreadable: {exc!r}"))
    return out


def split_counts(qc: pd.DataFrame, min_map_coverage: float) -> dict[str, dict[str, int]]:
    """Counts from qc.csv only (allowed for test, principle 1)."""
    out = {}
    for s in SPLITS:
        d = qc[qc["split"] == s]
        dd = d[d["has_wave"] & (d["map_coverage"] >= min_map_coverage)]
        out[s] = {"processed": len(d), "subjects": int(d["subjectid"].nunique()),
                  "has_wave": int(d["has_wave"].sum()), "cohort_D": len(dd),
                  "subjects_D": int(dd["subjectid"].nunique())}
    return out


def run_checks(prep: Path, qc: pd.DataFrame, case_results: Iterable[dict], *, min_map_coverage: float,
               known_failed: Iterable[int], max_missing_fraction: float,
               expected: Mapping[str, tuple] = EXPECTED, qc_sha256: str | None = None,
               expected_qc_sha256: str | None = None) -> dict:
    """Combine qc-level and per-case results into the data_check report."""
    prep = Path(prep)
    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, detail) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    if expected_qc_sha256:
        record("qc_sha256_matches_config", qc_sha256 == expected_qc_sha256,
               {"actual": qc_sha256, "config": expected_qc_sha256})

    invalid_split = sorted(str(v) for v in set(qc["split"].unique()) - set(SPLITS))
    record("split_values_valid", not invalid_split, {"invalid_split_values": invalid_split})

    per_subject = qc.groupby("subjectid")["split"].nunique()
    overlap = sorted(int(s) for s in per_subject[per_subject > 1].index)
    record("no_subject_in_two_splits", not overlap, {"subjects_in_two_splits": overlap})

    dup_case = int(qc["caseid"].duplicated().sum())
    record("qc_caseid_unique", dup_case == 0, {"duplicate_caseids": dup_case})

    errors = {int(k): v for k, v in read_json(prep / "errors.json").items()}
    known = set(int(c) for c in known_failed)
    record("failed_cases_equal_known", set(errors) == known and not (known & set(qc["caseid"])),
           {"errors_json": sorted(errors), "known_failed_cases": sorted(known)})
    processed = read_json(prep / "prep.json").get("processed_cases")
    record("prep_json_processed_matches_qc", processed == len(qc),
           {"prep_json_processed": processed, "qc_rows": len(qc)})

    counts = split_counts(qc, min_map_coverage)
    table = {}
    for s in SPLITS:
        exp = dict(zip(COUNT_KEYS, expected[s]))
        got = {"cohort": exp["cohort"], **counts[s]}
        miss_frac = 1 - got["processed"] / exp["cohort"]
        table[s] = {"expected": exp, "actual": got, "missing_fraction": round(miss_frac, 5),
                    "match": all(got[k] == exp[k] for k in COUNT_KEYS)}
    record("split_counts_match_plan", all(t["match"] for t in table.values()),
           {s: t["match"] for s, t in table.items()})
    record("missing_fraction_within_limit",
           all(t["missing_fraction"] <= max_missing_fraction for t in table.values()),
           {"limit": max_missing_fraction, **{s: t["missing_fraction"] for s, t in table.items()}})

    missing = {"cases": [], "features": [], "wave100": []}
    per_check: dict[str, list] = {c: [] for c in CASE_CHECKS}
    rows = dup_rows = n_cases = 0
    wave_diffs: list[int] = []
    for r in case_results:
        n_cases += 1
        for d in r["missing"]:
            missing[d].append(r["caseid"])
        for name, msg in r["problems"]:
            per_check[name].append({"caseid": r["caseid"], "problem": msg})
        rows += r.get("rows", 0)
        dup_rows += r.get("duplicate_rows", 0)
        if "wave_len_diff" in r:
            wave_diffs.append(r["wave_len_diff"])
    record("no_missing_files", not any(missing.values()), {k: sorted(v) for k, v in missing.items()})
    for name in CASE_CHECKS:
        record(name, not per_check[name], {"cases_checked": n_cases, "failures": per_check[name][:50],
                                           "n_failures": len(per_check[name])})

    problems = [f"{name}: {c['detail']}" for name, c in checks.items() if not c["passed"]]
    return {
        "ok": not problems,
        "problems": problems,
        "checks": checks,
        "split_counts": table,
        "invalid_split_values": invalid_split,
        "subjects_in_two_splits": overlap,
        "duplicate_rows": dup_rows,
        "known_failed_cases": sorted(errors),
        "missing_files": {k: sorted(v) for k, v in missing.items()},
        "feature_rows": rows,
        "wave_len_diff": ({"min": int(min(wave_diffs)), "max": int(max(wave_diffs))} if wave_diffs else None),
        "model_columns_checked": len(all_model_cols(ppv=PPV_PREP)),
        "test_label_values_read": False,
        "label_map_read_mode": "npy_header_only_all_splits",
    }


def case_rows(qc: pd.DataFrame) -> list[CaseRow]:
    return [CaseRow(int(r.caseid), int(r.surgery_seconds), bool(r.has_wave)) for r in qc.itertuples()]
