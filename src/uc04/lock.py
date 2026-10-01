"""Lock models, calibrators and thresholds before the test set is opened (plan section 9).

`build_lock` refuses to lock any result whose provenance shows uncommitted changes in
src/, scripts/, configs/ or templates/ (or records no dirty information at all), and
records the sha256 of every file so final_test.py can check nothing changed.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

from .io import read_json, sha256_file

LOCKED_FILES = ("model.joblib", "threshold.json", "calibration.json", "best.pt", "provenance.json")


class LockRefused(RuntimeError):
    pass


def provenance_problem(rec: dict) -> str | None:
    """Why a provenance record cannot be locked, or None if it can."""
    git = rec.get("git") or {}
    if not git.get("commit"):
        return "no commit recorded"
    if "dirty_paths" not in git:
        return "old provenance format without dirty_paths (rerun with the current code)"
    if git.get("dirty") is not False or git.get("dirty_paths"):
        return f"uncommitted changes in code: {git.get('dirty_paths')}"
    return None


def build_lock(dirs: Iterable[Path], root: Path, *, config_digest: str, commit: str, extra: dict | None = None) -> dict:
    """Lock record for result directories `dirs` (each holding a provenance.json)."""
    root = Path(root)
    problems, files = {}, {}
    for d in sorted(Path(x) for x in dirs):
        prov = d / "provenance.json"
        rel_dir = d.relative_to(root).as_posix()
        if not prov.exists():
            problems[rel_dir] = "missing provenance.json"
            continue
        why = provenance_problem(read_json(prov))
        if why:
            problems[rel_dir] = why
            continue
        for f in sorted(p for p in d.iterdir() if p.is_file() and p.name in LOCKED_FILES):
            files[f.relative_to(root).as_posix()] = sha256_file(f)
    if problems:
        raise LockRefused("refusing to lock: " + "; ".join(f"{k}: {v}" for k, v in list(problems.items())[:10])
                          + (f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""))
    if not files:
        raise LockRefused("nothing to lock")
    return {"locked_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "config_digest": config_digest, "code_commit": commit, "files": files, **(extra or {})}


def verify_lock(lock: dict, root: Path) -> list[str]:
    """Files whose sha256 no longer matches the lock (empty list = all good)."""
    return [rel for rel, digest in lock["files"].items()
            if not (Path(root) / rel).exists() or sha256_file(Path(root) / rel) != digest]
