"""Cohort audit, causal features, and all-window development dataset.

S01/S02 define input semantics; S03 motivates keeping natural negative windows.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BOUNDS, TRACKS, Protocol
from .data import fetch_csv, read_numeric, write_json
from .signals import causal_sample, detect_events, future_label, label_grid


def feature_row(sampled, ages, grid, t, protocol):
    result = {}
    for name, series in sampled.items():
        for seconds in (60, 300, protocol.history_seconds):
            sel = (grid > t - seconds) & (grid <= t)
            x, dt = series[sel], grid[sel] - t
            finite = np.isfinite(x)
            prefix = f"{name}_{seconds}"
            result[f"{prefix}_missing"] = float(1 - finite.mean()) if len(x) else 1.0
            for key, func in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
                result[f"{prefix}_{key}"] = float(func(x[finite])) if finite.any() else np.nan
            xx, yy = dt[finite], x[finite]
            if len(xx) >= 2:
                xx = xx - xx.mean()
                result[f"{prefix}_slope"] = float(np.dot(xx, yy - yy.mean()) / np.dot(xx, xx))
            else:
                result[f"{prefix}_slope"] = np.nan
        i = np.searchsorted(grid, t, side="right") - 1
        result[f"{name}_current"] = float(series[i]) if i >= 0 else np.nan
        result[f"{name}_age"] = float(min(ages[name][i], 600)) if i >= 0 else 600.0
    return result


def build_case(meta, raw_tracks, protocol=Protocol()):
    """Returns all decision rows (including abstentions), labels, events and audit."""
    start, end = int(np.ceil(meta["opstart"])), int(np.floor(meta["opend"]))
    if end <= start:
        raise ValueError("Empty surgery interval")
    grid = np.arange(start, end, dtype=float)
    numeric_grid = grid[::protocol.numeric_step_seconds]
    cleaned = {}
    for key in TRACKS:
        times, values = raw_tracks.get(key, (np.array([]), np.array([])))
        values = np.asarray(values, float).copy()
        low, high = BOUNDS[key]
        # Zero EtCO2/RR are kept; zero BP/HR are invalid as sensor observations.
        invalid = (values < low) | (values > high) | ~np.isfinite(values)
        if key in ("map", "sbp", "dbp", "hr"):
            invalid |= values == 0
        values[invalid] = np.nan
        cleaned[key] = (np.asarray(times), values)
    sampled, ages = {}, {}
    for name, (times, values) in cleaned.items():
        sampled[name], ages[name] = causal_sample(times, values, numeric_grid, protocol.feature_max_age_seconds)
    mt, mv = cleaned["map"]
    truth = label_grid(mt, mv, grid, protocol.label_max_gap_seconds)
    events = detect_events(grid, truth, protocol)
    online_map, _ = causal_sample(mt, mv, grid, protocol.label_max_gap_seconds)
    normal = np.isfinite(online_map) & (online_map >= protocol.map_threshold)
    # Conservative causal recovery: require 120s stable after any low/unknown.
    stable = np.zeros(len(grid), int)
    for i, good in enumerate(normal):
        stable[i] = (stable[i - 1] if i else 0) + 1 if good else 0
    rows = []
    for offset in range(protocol.history_seconds, len(grid), protocol.cadence_seconds):
        t = grid[offset]
        hist = (numeric_grid > t - protocol.history_seconds) & (numeric_grid <= t)
        coverage = float(np.isfinite(sampled["map"][hist]).mean())
        eligible = bool(stable[offset] >= protocol.recovery_seconds
                        and coverage >= protocol.min_map_history_coverage)
        row = {"caseid": int(meta["caseid"]), "subjectid": int(meta["subjectid"]),
               "time": float(t), "eligible": eligible, "history_coverage": coverage,
               "exposure_seconds": min(protocol.cadence_seconds, end - t)}
        row.update(feature_row(sampled, ages, numeric_grid, t, protocol))
        for name in ("age", "bmi", "asa"):
            row[f"static_{name}"] = float(meta.get(name, np.nan))
        for horizon in protocol.horizons_seconds:
            row[f"y_{horizon}"] = future_label(t, horizon, grid, truth, events, protocol)
        rows.append(row)
    frame = pd.DataFrame(rows)
    event_rows = []
    for event in events:
        record = {"caseid": int(meta["caseid"]), "subjectid": int(meta["subjectid"]), **asdict(event)}
        for horizon in protocol.horizons_seconds:
            opportunity = (frame.time < event.onset) & (frame.time >= event.onset - horizon)
            record[f"eligible_{horizon}"] = bool((opportunity & frame.eligible & frame[f"y_{horizon}"].eq(1)).any())
        event_rows.append(record)
    audit = {"caseid": int(meta["caseid"]), "subjectid": int(meta["subjectid"]),
             "surgery_seconds": end - start, "label_coverage": float(np.isfinite(truth).mean()),
             "events": len(events), "decision_rows": len(rows),
             "eligible_rows": int(frame.eligible.sum()) if len(frame) else 0,
             "median_map_update_seconds": float(np.median(np.diff(mt))) if len(mt) > 1 else None}
    return frame, event_rows, audit


def feature_columns(frame):
    # Allowlist prefixes: identifiers, future labels and audit metadata never enter X.
    return [c for c in frame.columns if c.startswith(tuple(f"{k}_" for k in TRACKS)) or c.startswith("static_")]


def build_pilot(root, out, protocol=Protocol()):
    root, out = Path(root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Dataset output must be new/empty to preserve provenance")
    manifest = pd.read_csv(root / "pilot_manifest.csv")
    if not manifest.split.eq("train").all():
        raise ValueError("Pilot must use only global training patients")
    fetch_report = json.loads((root / "fetch_report.json").read_text(encoding="utf-8"))
    if fetch_report["errors"] or fetch_report["protocol_hash"] != protocol.digest():
        raise ValueError("Fetch errors or protocol mismatch; audit data first")
    out.mkdir(parents=True, exist_ok=True)
    frames, event_rows, audits = [], [], []
    for _, meta in manifest.iterrows():
        raw = {name: read_numeric(fetch_csv(str(meta[f"tid_{name}"]), root / "raw"))
               for name in TRACKS if pd.notna(meta[f"tid_{name}"])}
        frame, events, audit = build_case(meta, raw, protocol)
        frames.append(frame)
        event_rows.extend(events)
        audits.append(audit)
    if not frames or sum(len(f) for f in frames) == 0:
        raise ValueError("No windows; inspect cohort intervals")
    windows = pd.concat(frames, ignore_index=True)
    windows.to_csv(out / "windows.csv.gz", index=False, compression="gzip")
    columns = ["caseid", "subjectid", "onset", "end"] + [f"eligible_{h}" for h in protocol.horizons_seconds]
    pd.DataFrame(event_rows, columns=columns).to_csv(out / "events.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "quality.csv", index=False)
    manifest.to_csv(out / "manifest.csv", index=False)
    write_json(out / "dataset.json", {"scope": "pilot_train_pool_only", "protocol": asdict(protocol),
        "protocol_hash": protocol.digest(), "features": feature_columns(windows),
        "subjects": int(manifest.subjectid.nunique()), "cases": len(manifest),
        "windows": len(windows), "events": len(event_rows),
        "manifest_sha256": hashlib.sha256((root / "pilot_manifest.csv").read_bytes()).hexdigest(),
        "windows_sha256": hashlib.sha256((out / "windows.csv.gz").read_bytes()).hexdigest()})
    return {"cases": len(manifest), "windows": len(windows), "events": len(event_rows)}

