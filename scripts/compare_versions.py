"""Compare model results across sample versions (v2 vs v3), overall and by phase of the case.

    python scripts/compare_versions.py \
        --run v2 data/samples_v2 artifacts/v2 artifacts/v2/artifacts_hf \
        --run v3 data/samples_v3 artifacts/v3 artifacts/v3/dl_final \
        --out reports/v3/version_comparison.csv

Each --run is: name, samples directory, then one or more work directories. Validation
predictions are found under them (uc04.phases.find_predictions): tabular
`artifacts/tabular/**/val_predictions.parquet` and DL `**/dl/W<W>/val_predictions.parquet`,
each with the threshold.json written next to it (chosen on the full validation split of that
version). Every prediction set is re-evaluated with the same evaluator at that threshold and
split by phase: v3 rows before incision are "pre_incision"; v2 rows are all "surgery". No
bootstrap: this is a descriptive comparison; the per-version CIs are in each run's
tabular_validation / dl_validation reports.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from uc04.config import load_config
from uc04.phases import evaluate_run

ROOT = Path(__file__).resolve().parents[1]
SHOW = ["version", "model", "window", "phase", "auroc", "event_sensitivity", "events_eligible",
        "false_alarms_per_hour", "alarm_ppv", "lead_median_s"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", action="append", required=True, metavar="NAME SAMPLES WORK",
                    help="name, samples dir, work dir(s)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "uc04_v2.json"), help="evaluation settings")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    E = load_config(args.config).evaluation
    t0 = time.time()
    rows = [r for name, samples, *works in args.run for r in evaluate_run(name, samples, works, E)]
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    show = df[(df.horizon == 300) & (df.window.isna() | (df.window == 120))]
    print(show[SHOW].round(3).to_string(index=False))
    print(f"{len(df)} rows -> {out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
