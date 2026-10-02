"""Check installed library versions.

    python scripts/env_check.py [--requirements requirements.txt] [--ignore torch]
    python scripts/env_check.py --provenance path/to/provenance.json

Without --provenance: every package in requirements.txt is installed at or above its
minimum version (used by every Kaggle run after installing, and by 00_env_check);
the installed versions are printed and recorded in each run's provenance.json.
With --provenance: compares with the versions recorded when a model was trained
(used by NB04 before loading it). Exit code 1 if a package is missing or too old, or,
with --provenance, if a model-critical package differs.
"""
from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

from uc04.envcheck import EnvMismatch, LOCKED, assert_env_matches, installed_versions
from uc04.io import read_json

ROOT = Path(__file__).resolve().parents[1]


def requirements(path: Path) -> list[Requirement]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            out.append(Requirement(line))
    return out


def check_minimums(reqs: list[Requirement], ignore=()) -> list[str]:
    """Print each requirement with its installed version; return the problems."""
    problems = []
    print(f"{'package':<22}{'required':<16}{'installed':<16}")
    for r in reqs:
        if r.name in ignore or re.sub(r"[-_.]+", "-", r.name.lower()) in ignore:
            continue
        try:
            got = metadata.version(r.name)
        except metadata.PackageNotFoundError:
            got = None
        ok = got is not None and r.specifier.contains(Version(got).base_version, prereleases=True)
        print(f"{r.name:<22}{str(r.specifier) or 'any':<16}{str(got):<16}{'' if ok else '  <-- missing or too old'}")
        if not ok:
            problems.append(f"{r.name} {got} does not satisfy {r.specifier}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requirements", default=str(ROOT / "requirements.txt"))
    ap.add_argument("--provenance", help="provenance.json of a trained model")
    ap.add_argument("--ignore", nargs="*", default=[], help="packages not used here (e.g. torch on CPU notebooks)")
    args = ap.parse_args(argv)

    if not args.provenance:
        problems = check_minimums(requirements(Path(args.requirements)), set(args.ignore))
        if problems:
            print(f"FAIL vs {args.requirements}: {'; '.join(problems)}", file=sys.stderr)
            return 1
        print(f"OK: installed versions satisfy {args.requirements}")
        return 0

    rec = read_json(args.provenance)["packages"]
    expected = {p: rec.get(p) for p in LOCKED if p in rec and p not in set(args.ignore)}
    actual = installed_versions(expected.keys())
    print(f"{'package':<16}{'trained with':<16}{'installed':<16}")
    for p, v in expected.items():
        flag = "" if (v or "").split("+")[0] == (actual.get(p) or "").split("+")[0] else "  <-- differs"
        print(f"{p:<16}{str(v):<16}{str(actual.get(p)):<16}{flag}")
    try:
        other = assert_env_matches(expected, actual)
    except EnvMismatch as exc:
        print(f"FAIL vs {args.provenance}: {exc}", file=sys.stderr)
        return 1
    if other:
        print(f"WARNING: non-critical differences: {[d['package'] for d in other]}")
    print(f"OK: model-critical versions match {args.provenance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
