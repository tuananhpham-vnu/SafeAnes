"""NB04 step 2: write artifacts/lock.json for the chosen results (plan section 9).

    python scripts/lock_models.py [--include tabular dl] [--dry-run]

Refuses to lock if the working tree has uncommitted code changes, if any result's
provenance records such changes (or predates the dirty_paths field), or if a lock
already exists. --dry-run only reports what would be locked or why it is refused.
Writing lock.json is what makes the test split loadable: do it only when the team
has fixed the configurations to report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from uc04.config import load_config
from uc04.io import write_json
from uc04.lock import LockRefused, build_lock
from uc04.provenance import git_commit
from uc04.runtime import REPO_CONFIG


def result_dirs(artifacts: Path, include) -> list[Path]:
    out = []
    if "tabular" in include:
        out += [p.parent for p in (artifacts / "tabular").glob("**/done.json")]
    if "dl" in include:
        out += [p.parent for p in (artifacts / "dl").glob("W*/calibration.json")]
        out += [p.parent for p in (artifacts / "dl").glob("W*/seed*/done.json")]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_CONFIG))
    ap.add_argument("--include", nargs="+", default=["tabular", "dl"], choices=["tabular", "dl"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    artifacts = cfg.path("artifacts")
    lock_path = artifacts / "lock.json"
    if lock_path.exists():
        print(f"{lock_path} already exists; never overwritten", file=sys.stderr)
        return 1
    git = git_commit(cfg.root)
    if git.get("dirty") is not False:
        print(f"refusing to lock: uncommitted code changes {git.get('dirty_paths')}", file=sys.stderr)
        return 1
    try:
        lock = build_lock(result_dirs(artifacts, args.include), artifacts, config_digest=cfg.digest(),
                          commit=git["commit"], extra={"include": args.include})
    except LockRefused as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{len(lock['files'])} files would be locked" if args.dry_run else f"locking {len(lock['files'])} files")
    if not args.dry_run:
        write_json(lock_path, lock)
        print(f"wrote {lock_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
