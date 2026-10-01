"""Export the curated, shareable results into results/, organised by model.

    python scripts/export_results.py

reports/ and artifacts/ are working directories (not committed). This copies what
can be shared publicly into results/ (committed):

  results/data/                        data checks and label summaries (aggregates only)
  results/tabular/summary/             combined tables (validation), model comparison, W*
  results/tabular/<model>/[W<W>/]h<h>/ metrics.csv, calibration_check.csv, threshold.json, provenance.json
  results/tabular/analysis/            SHAP, ablation, confidence check
  results/tabular/exploratory/         long windows (post-hoc, reported separately)
  results/dl/                          filled after the Kaggle runs

Never exported: trained models (*.joblib, *.pt), per-case predictions
(val_predictions.parquet), per-case counts (case_metrics.parquet), anything from
data/ or sealed_v2. The script checks this before writing.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from uc04.config import load_config
from uc04.hub import assert_no_test
from uc04.io import write_json
from uc04.runtime import REPO_CONFIG
from uc04.tabular import MODELS, combo_dir, combo_name

ROOT = Path(__file__).resolve().parents[1]
NEVER = (".joblib", ".pt", ".parquet", ".npz", ".npy")


def copy(src: Path, dst: Path, copied: list[str]) -> None:
    if not src.exists():
        raise SystemExit(f"missing expected input: {src}")
    if src.is_dir():
        for f in sorted(src.rglob("*")):
            if f.is_file():
                copy(f, dst / f.relative_to(src), copied)
        return
    if src.suffix in NEVER:
        raise SystemExit(f"refusing to export model or per-case file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(dst.relative_to(ROOT).as_posix())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_CONFIG))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    rep, art, samples = ROOT / "reports", ROOT / "artifacts", cfg.path("samples")
    out = ROOT / "results"
    for sub in ("data", "tabular"):
        shutil.rmtree(out / sub, ignore_errors=True)
    copied: list[str] = []

    # ---- data
    for f in ("data_check.json", "dl_cpu_speed.json"):
        copy(rep / f, out / "data" / f, copied)
    copy(samples / "cohort_flow.csv", out / "data" / "cohort_flow.csv", copied)  # version without the test row
    excl = pd.read_csv(rep / "cohort_excluded.csv")
    excl = excl[excl["split"].fillna("") != "test"]                             # no test-split rows in public files
    (out / "data").mkdir(parents=True, exist_ok=True)
    excl.to_csv(out / "data" / "cohort_excluded_train_cal_val.csv", index=False)
    copied.append("results/data/cohort_excluded_train_cal_val.csv")
    for f in ("label_report.csv", "abstain_report.csv", "samples.json", "provenance.json"):
        name = {"samples.json": "samples_summary.json", "provenance.json": "samples_provenance.json"}.get(f, f)
        copy(samples / f, out / "data" / name, copied)

    # ---- tabular summary
    t = out / "tabular"
    for f in ("tabular_validation.csv", "tabular_validation_by_policy.csv", "calibration_check_tabular.csv",
              "model_comparison.csv", "model_comparison_pairs.csv", "w_star.json"):
        copy(rep / f, t / "summary" / f, copied)

    # ---- per model / window / horizon
    for m in MODELS:
        for W in (cfg.features.windows_seconds if m != "map_threshold" else [None]):
            for h in cfg.labels.horizons_seconds:
                name, rel = combo_name(m, W, h), combo_dir(m, W, h)
                dst = t / rel.replace("tabular/", "", 1)
                copy(rep / "tabular_validation" / f"{name}.csv", dst / "metrics.csv", copied)
                cc = rep / "calibration_check" / f"{name}.csv"
                if cc.exists():  # absent when the combination was skipped (no positives)
                    copy(cc, dst / "calibration_check.csv", copied)
                for f in ("threshold.json", "provenance.json"):
                    copy(art / rel / f, dst / f, copied)

    # ---- analysis and exploratory
    copy(rep / "ablation.csv", t / "analysis" / "ablation.csv", copied)
    copy(rep / "shap_groups.csv", t / "analysis" / "shap" / "shap_groups.csv", copied)
    for f in sorted(rep.glob("shap_W*_h*.csv")):
        copy(f, t / "analysis" / "shap" / f.name, copied)
    copy(rep / "tabular_confidence", t / "analysis" / "confidence", copied)
    copy(rep / "tabular_analysis" / "provenance.json", t / "analysis" / "provenance.json", copied)
    copy(rep / "exploratory", t / "exploratory", copied)

    assert_no_test([c.replace("results/", "", 1) for c in copied], out)
    write_json(out / "index.json", {"files": copied, "note": "generated by scripts/export_results.py"})
    print(f"exported {len(copied)} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
