"""Compare installed library versions with requirements-lock.json.

    python scripts/env_check.py [--lock requirements-lock.json] [--provenance path/to/provenance.json]

Without --provenance: compares with the lock (used by every Kaggle notebook
after installing, and by 00_env_check). With --provenance: compares with the
versions recorded when a model was trained (used by NB04 before loading it).
Exit code 1 if a model-critical package differs.
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from uc04.envcheck import EnvMismatch, LOCKED, assert_env_matches, installed_versions
from uc04.io import read_json

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lock", default=str(ROOT / "requirements-lock.json"))
    ap.add_argument("--provenance", help="provenance.json of a trained model")
    ap.add_argument("--ignore", nargs="*", default=[], help="packages not used here (e.g. torch on CPU notebooks)")
    args = ap.parse_args(argv)

    if args.provenance:
        rec = read_json(args.provenance)["packages"]
        expected = {p: rec.get(p) for p in LOCKED if p in rec}
        source = args.provenance
    else:
        lock = read_json(args.lock)
        expected, source = lock["packages"], args.lock
        if lock.get("python", "").rsplit(".", 1)[0] != platform.python_version().rsplit(".", 1)[0]:
            print(f"WARNING: python {platform.python_version()} vs locked {lock.get('python')}")
    expected = {p: v for p, v in expected.items() if p not in set(args.ignore)}
    actual = installed_versions(expected.keys())
    print(f"{'package':<16}{'expected':<16}{'installed':<16}")
    for p, v in expected.items():
        flag = "" if (v or "").split("+")[0] == (actual.get(p) or "").split("+")[0] else "  <-- differs"
        print(f"{p:<16}{str(v):<16}{str(actual.get(p)):<16}{flag}")
    try:
        other = assert_env_matches(expected, actual)
    except EnvMismatch as exc:
        print(f"FAIL vs {source}: {exc}", file=sys.stderr)
        return 1
    if other:
        print(f"WARNING: non-critical differences: {[d['package'] for d in other]}")
    print(f"OK: model-critical versions match {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
