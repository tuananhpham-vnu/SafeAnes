"""Push the tabular results to the private Hugging Face runs repo (hub.runs_repo).

    python scripts/push_results_hf.py [--dry-run]

Pushes, in one commit:
  artifacts/tabular/<model>/[W<W>/]h<h>/  -> tabular/...   (models, thresholds, validation predictions,
                                                             per-case counts, provenance, done.json)
  results/                                 -> results/...   (the curated tables, as on GitHub)
Before pushing it checks that the repo is PRIVATE (refuses otherwise), and every
file goes through hub.assert_no_test (no sealed/test files, no parquet/csv with a
test split). Nothing from data/, sealed_v2 or .env is ever included. The token is
read from the environment or the git-ignored .env and never printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from uc04.config import load_config
from uc04.hub import hub_from_config
from uc04.runtime import REPO_CONFIG

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_CONFIG))
    ap.add_argument("--dry-run", action="store_true", help="check and list, push nothing")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    hub = hub_from_config(cfg, ROOT, push=not args.dry_run)  # raises if the repo is public
    paths = ["artifacts/tabular", "results"]
    for p in paths:
        if not (ROOT / p).is_dir():
            print(f"missing {p}; run the tabular pipeline and export_results.py first", file=sys.stderr)
            return 1
    hub.register(*paths)
    files = hub._files(paths)
    size = sum((ROOT / f).stat().st_size for f in files)
    print(f"{len(files)} files, {size / 1e6:.1f} MB -> {cfg.hub.runs_repo} ({'dry run' if args.dry_run else 'pushing'})")
    if args.dry_run:
        from uc04.hub import assert_no_test
        assert_no_test(files, ROOT)
        print("assert_no_test: OK")
        return 0
    ok = hub.push(paths, message="tabular results (65 combinations) and curated results/")
    print("pushed" if ok else "push failed (see messages above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
