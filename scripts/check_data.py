"""Stage 1.3: check prep_v1 before building samples (plan section 3.1).

    python scripts/check_data.py --config configs/uc04_v2.json [--prep D:/.../prep_v1] [--workers 8]

Writes <reports>/data_check.json (with sha256 of qc.csv and provenance) and exits
with code 1 if any check fails. Send data_check.json to the team before stage 1.4.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from uc04.checks import case_rows, check_case, run_checks
from uc04.columns import PPV_PREP, all_model_cols
from uc04.config import load_config
from uc04.io import read_qc, sha256_file, write_json
from uc04.provenance import make_provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/uc04_v2.json")
    ap.add_argument("--prep", help="override paths.prep")
    ap.add_argument("--reports", help="override paths.reports")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    cfg = load_config(args.config).with_paths(prep=args.prep, reports=args.reports)
    prep = cfg.path("prep")
    for name in cfg.input.required:
        if not (prep / name).exists():
            print(f"missing required input: {prep / name}", file=sys.stderr)
            return 1

    t0 = time.time()
    qc = read_qc(prep)
    qc_sha = sha256_file(prep / "qc.csv")
    rows = case_rows(qc)
    required = tuple(all_model_cols(ppv=PPV_PREP))
    check = partial(check_case, prep, required_cols=required)
    print(f"checking {len(rows)} cases in {prep} with {args.workers} workers ...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(check, rows, chunksize=32))

    report = run_checks(prep, qc, results, min_map_coverage=cfg.input.min_map_coverage,
                        known_failed=cfg.input.known_failed_cases,
                        max_missing_fraction=cfg.input.max_missing_case_fraction,
                        qc_sha256=qc_sha, expected_qc_sha256=cfg.input.qc_sha256)
    report["qc_sha256"] = qc_sha
    report["prep"] = str(prep)
    report["seconds"] = round(time.time() - t0, 1)
    report["provenance"] = make_provenance(cfg, "check_data", inputs={"qc.csv": qc_sha},
                                           sections=("input",))
    out = cfg.path("reports") / "data_check.json"
    write_json(out, report)

    print(f"{'split':<12}{'processed':>10}{'subjects':>10}{'has_wave':>10}{'cohort_D':>10}{'subj_D':>8}  match")
    for s, t in report["split_counts"].items():
        a = t["actual"]
        print(f"{s:<12}{a['processed']:>10}{a['subjects']:>10}{a['has_wave']:>10}{a['cohort_D']:>10}"
              f"{a['subjects_D']:>8}  {t['match']}")
    print(f"feature rows: {report['feature_rows']:,}; duplicate rows: {report['duplicate_rows']}; "
          f"wave length diff: {report['wave_len_diff']}")
    for name, c in report["checks"].items():
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name}")
    print(f"qc.csv sha256: {qc_sha}")
    if report["ok"]:
        print(f"ALL CHECKS PASSED -> {out}")
        return 0
    print("CHECKS FAILED:", *report["problems"], sep="\n  - ", file=sys.stderr)
    print(f"details -> {out}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
