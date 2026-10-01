"""Library versions: lock on the local machine, enforce on Kaggle and in NB04.

Models trained on Kaggle (.joblib, .pt) are loaded back on the local machine in
NB04. A different scikit-learn / LightGBM / torch version can fail to load them or
silently change predictions, so:
- `requirements-lock.json` records the local versions (scripts/lock_requirements.py);
- Kaggle notebooks install exactly those versions (torch: same version, CUDA build)
  and `00_env_check` compares them;
- NB04 compares the versions in each model's provenance with the local ones and
  stops on any difference in `MODEL_CRITICAL`.
"""
from __future__ import annotations

import platform
from importlib import metadata
from typing import Mapping

LOCKED = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "joblib", "lightgbm", "shap",
          "huggingface_hub", "numba", "torch")
# A difference here can break loading a model or change its predictions.
MODEL_CRITICAL = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "lightgbm", "torch")


def base_version(v: str | None) -> str | None:
    """'2.14.0+cpu' -> '2.14.0' (CPU and CUDA builds of torch share the version)."""
    return None if v is None else v.split("+", 1)[0]


def installed_versions(packages=LOCKED) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in packages:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def lock_record() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": installed_versions()}


def compare(expected: Mapping[str, str | None], actual: Mapping[str, str | None],
            critical=MODEL_CRITICAL) -> list[dict]:
    """Differences between two {package: version} maps (torch compared without the +build tag)."""
    diffs = []
    for p, exp in expected.items():
        got = actual.get(p)
        if base_version(exp) != base_version(got):
            diffs.append({"package": p, "expected": exp, "actual": got, "critical": p in critical})
    return diffs


class EnvMismatch(RuntimeError):
    pass


def assert_env_matches(expected: Mapping[str, str | None], actual: Mapping[str, str | None] | None = None,
                       critical=MODEL_CRITICAL) -> list[dict]:
    """Raise on any difference in a critical package; return the non-critical ones."""
    diffs = compare(expected, actual if actual is not None else installed_versions(expected.keys()), critical)
    bad = [d for d in diffs if d["critical"]]
    if bad:
        lines = ", ".join(f"{d['package']} {d['expected']} != {d['actual']}" for d in bad)
        raise EnvMismatch(f"library versions differ from the lock/model: {lines}")
    return diffs


def pip_requirements(lock: Mapping, exclude=("torch",)) -> str:
    """pip -r content pinning every locked package except torch (installed separately)."""
    lines = [f"{p}=={v}" for p, v in lock["packages"].items() if v and p not in exclude]
    return "\n".join(lines) + "\n"
