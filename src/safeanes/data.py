"""Public VitalDB numeric API. Sources S01/S02 in docs/SOURCES.md.

No waveforms are decoded here: their CSV timestamp encoding is different.
Raw responses, source URLs, and hashes are preserved for auditability.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .config import TRACKS, Protocol

API = "https://api.vitaldb.net"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def decode_csv(payload):
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    return pd.read_csv(io.BytesIO(payload))


def fetch_csv(endpoint, cache):
    """GET only, fixed API host, atomic cached raw payload + metadata."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", endpoint):
        raise ValueError("Invalid API endpoint")
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{endpoint}.csv.gz"
    meta_path = cache / f"{endpoint}.source.json"
    if path.exists():
        payload = path.read_bytes()
        if not meta_path.exists():
            raise ValueError(f"Missing provenance: {meta_path}")
        expected = json.loads(meta_path.read_text(encoding="utf-8"))["sha256"]
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Cache checksum mismatch: {path}")
        return decode_csv(payload)
    url = f"{API}/{endpoint}"
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "SafeAnes-UC04-research/0.1"})
            with urlopen(request, timeout=45) as response:
                payload = response.read()
            frame = decode_csv(payload)  # fail before committing an HTML/error response
            break
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    temp = path.with_suffix(".tmp")
    temp.write_bytes(payload)
    temp.replace(path)
    write_json(meta_path, {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                           "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
    return frame


def subject_bucket(subjectid, seed=20260917):
    """Stable subject-level assignment; adding cases cannot move existing subjects."""
    if pd.isna(subjectid) or float(subjectid) != int(subjectid):
        raise ValueError("Missing/non-integer subjectid cannot be split safely")
    digest = hashlib.sha256(f"{seed}:{int(subjectid)}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if u < 0.70 else "calibration" if u < 0.775 else "validation" if u < 0.85 else "test"


def cohort_manifest(cases, tracks, protocol=Protocol()):
    required = {"caseid", "subjectid", "age", "ane_type", "opstart", "opend"}
    if not required.issubset(cases.columns) or not {"caseid", "tname", "tid"}.issubset(tracks.columns):
        raise ValueError("VitalDB metadata schema changed; inspect raw metadata")
    if cases.caseid.duplicated().any():
        raise ValueError("Duplicate case IDs")
    out = cases.copy()
    out["exclusion_reason"] = ""
    checks = [
        (out.subjectid.isna(), "missing_subjectid"),
        (out.age.isna() | (out.age < 18), "not_known_adult"),
        (out.ane_type != "General", "not_general_anesthesia"),
        (~np.isfinite(out.opstart) | ~np.isfinite(out.opend)
         | (out.opend <= out.opstart), "invalid_surgery_interval"),
        ((out.opend - out.opstart) < protocol.history_seconds + max(protocol.horizons_seconds)
         + protocol.event_seconds, "insufficient_duration"),
    ]
    for bad, reason in checks:
        out.loc[bad & out.exclusion_reason.eq(""), "exclusion_reason"] = reason
    for name, track in TRACKS.items():
        match = tracks[tracks.tname.eq(track)]
        if match.caseid.duplicated().any():
            raise ValueError(f"Ambiguous tracks for {track}; resolve explicitly")
        out[f"tid_{name}"] = out.caseid.map(match.set_index("caseid").tid)
    out.loc[out.tid_map.isna() & out.exclusion_reason.eq(""), "exclusion_reason"] = "missing_arterial_map"
    out["eligible"] = out.exclusion_reason.eq("")
    out["split"] = out.subjectid.map(lambda s: subject_bucket(s, protocol.seed) if pd.notna(s) else "excluded")
    return out


def read_numeric(frame):
    if frame.shape[1] != 2:
        raise ValueError("Numeric track must have exactly two columns")
    times = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(float)
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(float)
    if not np.isfinite(times).all():
        raise ValueError("Invalid numeric timestamps (possibly a waveform track)")
    # Preserve last record at duplicate timestamps, including explicit invalid values.
    clean = pd.DataFrame({"time": times, "value": values}).sort_values("time", kind="stable")
    clean = clean.drop_duplicates("time", keep="last")
    return clean.time.to_numpy(), clean.value.to_numpy()


def fetch_pilot(root, limit=60, workers=4, protocol=Protocol()):
    """Only TRAIN-pool patients are downloaded by this pilot command."""
    if limit <= 0 or not 1 <= workers <= 8:
        raise ValueError("Positive case limit and 1..8 workers required")
    root = Path(root)
    raw = root / "raw"
    cases, tracks = fetch_csv("cases", raw), fetch_csv("trks", raw)
    manifest = cohort_manifest(cases, tracks, protocol)
    manifest.to_csv(root / "cohort_manifest.csv", index=False)
    pool = manifest[manifest.eligible & manifest.split.eq("train")]
    selected = pool.sample(frac=1, random_state=protocol.seed).head(limit).sort_values("caseid")
    selected.to_csv(root / "pilot_manifest.csv", index=False)
    tids = sorted({str(row[f"tid_{key}"]) for _, row in selected.iterrows()
                   for key in TRACKS if pd.notna(row[f"tid_{key}"])})
    errors = []
    def download(tid):
        try:
            read_numeric(fetch_csv(tid, raw))
            return None
        except Exception as exc:
            return {"tid": tid, "error": str(exc)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(download, tids):
            if result:
                errors.append(result)
    write_json(root / "fetch_report.json", {
        "scope": "pilot_train_pool_only", "selected_cases": len(selected), "tracks": len(tids),
        "eligible_total": int(manifest.eligible.sum()), "errors": errors,
        "protocol_hash": protocol.digest(),
    })
    if errors:
        raise RuntimeError(f"{len(errors)} downloads failed; see fetch_report.json. Rerun to resume.")
    return selected

