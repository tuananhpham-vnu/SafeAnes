"""Library versions: recorded with every model, enforced in NB04.

requirements.txt only sets minimum versions (DEVIATIONS 40), so Kaggle runs use the
image's versions. Models trained on Kaggle (.pt) are loaded back on the local machine
in NB04, and a different scikit-learn / LightGBM / torch version can fail to load them
or silently change predictions, so:
- each run's provenance.json records the installed versions of `LOCKED`;
- NB04 compares those with the local ones (scripts/env_check.py --provenance) and
  stops on any difference in `MODEL_CRITICAL`: install the recorded versions first.
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
