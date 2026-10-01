"""Record the local library versions so Kaggle installs the same ones.

    python scripts/lock_requirements.py

Writes, at the repo root:
- requirements-lock.json : python version and the locked packages (read by env_check.py)
- requirements-lock.txt  : `pip install -r` pins for every locked package except torch
- requirements-freeze-local.txt : full `pip freeze` of the local venv, for the record only
  (contains Windows-only packages, do not install it on Kaggle)

Torch is installed on Kaggle separately with the same version from the CUDA index,
for example `pip install torch==<version> --index-url https://download.pytorch.org/whl/<cuda>`.
Re-run this script (and commit) whenever the local venv changes.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from uc04.envcheck import base_version, lock_record, pip_requirements
from uc04.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    lock = lock_record()
    missing = [p for p, v in lock["packages"].items() if v is None]
    if missing:
        print(f"not installed, cannot lock: {missing}", file=sys.stderr)
        return 1
    lock["torch_base_version"] = base_version(lock["packages"]["torch"])
    write_json(ROOT / "requirements-lock.json", lock)
    (ROOT / "requirements-lock.txt").write_text(pip_requirements(lock), encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
                            capture_output=True, text=True, check=True).stdout
    (ROOT / "requirements-freeze-local.txt").write_text(freeze, encoding="utf-8")
    for p, v in lock["packages"].items():
        print(f"  {p:<16} {v}")
    print(f"python {lock['python']}; wrote requirements-lock.json / .txt / freeze")
    return 0


if __name__ == "__main__":
    sys.exit(main())
