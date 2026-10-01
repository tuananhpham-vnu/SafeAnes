"""Provenance records (plan principle 11).

Every output directory gets a `provenance.json` with the config digest, git
commit, package versions and input digests. A later step calls
`check_provenance` to make sure it builds on the output it expects.
"""
from __future__ import annotations

import datetime as dt
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Mapping

from .config import Config
from .envcheck import LOCKED
from .io import read_json, write_json

PACKAGES = (*LOCKED, "uc04")  # NB04 compares these against the local venv before loading a model


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


CODE_DIRS = ("src", "scripts", "configs", "templates")  # changes here can change results


def git_commit(repo: str | Path = ".") -> dict:
    """HEAD commit and whether code that affects results has uncommitted changes.

    `dirty` is True only for uncommitted (modified, staged or untracked) files under
    CODE_DIRS; changes to docs or reports do not count. `dirty_paths` lists them.
    """
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                                  check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    if commit is None:  # code from the uc04-code Kaggle Dataset: git archive of a clean commit
        f = Path(repo) / "CODE_COMMIT"
        commit = os.environ.get("UC04_CODE_COMMIT") or (f.read_text().strip() if f.exists() else None)
        return {"commit": commit, "dirty": False, "dirty_paths": [], "source": "CODE_COMMIT"}
    status = run("status", "--porcelain", "--untracked-files=all", "--", *CODE_DIRS)
    if status is None:
        return {"commit": commit.strip(), "dirty": None, "dirty_paths": None}
    paths = sorted(line[3:].strip().strip('"') for line in status.splitlines() if line.strip())
    return {"commit": commit.strip(), "dirty": bool(paths), "dirty_paths": paths}


def make_provenance(config: Config, step: str, inputs: Mapping[str, str] | None = None,
                    sections: tuple[str, ...] = (), extra: Mapping | None = None) -> dict:
    """`inputs` maps a name to a digest (e.g. sha256 of qc.csv, HF revision)."""
    return {
        "step": step,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_source": config.source,
        "config_digest": config.digest(*sections),
        "config_sections": list(sections) or "all_result_sections",
        "git": git_commit(config.root),
        "packages": package_versions(),
        "inputs": dict(inputs or {}),
        **(dict(extra) if extra else {}),
    }


def write_provenance(directory: Path, record: dict) -> Path:
    path = Path(directory) / "provenance.json"
    write_json(path, record)
    return path


class ProvenanceError(RuntimeError):
    pass


def check_provenance(path: Path, *, config_digest: str | None = None,
                     inputs: Mapping[str, str] | None = None) -> dict:
    """Raise if the recorded config digest or any named input digest differs."""
    path = Path(path)
    if not path.exists():
        raise ProvenanceError(f"missing provenance file: {path}")
    rec = read_json(path)
    if config_digest is not None and rec.get("config_digest") != config_digest:
        raise ProvenanceError(f"{path}: config digest {rec.get('config_digest')} != expected {config_digest}")
    for name, digest in (inputs or {}).items():
        got = rec.get("inputs", {}).get(name)
        if got != digest:
            raise ProvenanceError(f"{path}: input {name!r} digest {got} != expected {digest}")
    return rec
