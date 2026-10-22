"""Pack the samples v3 inputs of the week-15 214-input models (W = 120) into a small directory for a Kaggle dataset.

    python scripts/data/pack_v3_w120.py <out_dir> [--samples data/samples_v3]

The Kaggle dataset datnguyen31112/safeanes-v3 now holds only features_v7 / features_v8; the tabular pipeline
(uc04.loaders, scripts/train_tabular.py, scripts/train_tabm.py) needs the samples v3 layout. This writes that layout
with only what configs/features_214.json needs, for train / calibration / validation (never test):

  samples.json, events.parquet, subsets/*.json, labels/<split>.parquet     copied unchanged
  features/<split>.parquet      caseid, time + the W = 120 monitor and arterial-wave columns (same rows as labels)
  features_v5/<split>.parquet   caseid, time + the f1a/f1b/f1c columns of the 214 + st_* from the cohort manifest
  features_v5/columns.json      copied
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from uc04.columns import SIDE_PREFIXES
from uc04.feature_eda import static_from_manifest
from uc04.tabular import features_214

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "calibration", "validation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--samples", type=Path, default=ROOT / "data/samples_v3")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/vitaldb_full/meta/cohort_manifest.csv")
    args = ap.parse_args()
    src, out = args.samples, args.out
    cols = features_214(120)
    side = [c for c in cols if c.startswith(SIDE_PREFIXES)]
    main_cols = [c for c in cols if c not in side]
    st = [c for c in side if c.startswith("st_")]
    for d in ("labels", "features", "features_v5", "subsets"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for f in ("samples.json", "events.parquet", "features_v5/columns.json"):
        shutil.copy2(src / f, out / f)
    for f in (src / "subsets").glob("*.json"):
        shutil.copy2(f, out / "subsets" / f.name)
    static = static_from_manifest(args.manifest)[["caseid", *st]]
    for s in SPLITS:
        shutil.copy2(src / "labels" / f"{s}.parquet", out / "labels" / f"{s}.parquet")
        pq.write_table(pq.read_table(src / "features" / f"{s}.parquet", columns=["caseid", "time", *main_cols]),
                       out / "features" / f"{s}.parquet", compression="zstd")
        v5 = pq.read_table(src / "features_v5" / f"{s}.parquet",
                           columns=["caseid", "time", *[c for c in side if c not in st]]).to_pandas()
        n = len(v5)
        v5 = v5.merge(static, on="caseid", how="left")
        assert len(v5) == n
        v5.to_parquet(out / "features_v5" / f"{s}.parquet", index=False, compression="zstd")
        print(f"{s}: features {len(main_cols)} columns, features_v5 {len(side)} columns ({n:,} rows)")
    (out / "README.md").write_text(
        "# UC04 samples v3, W = 120 inputs of the 214-input models\n\nTrain, calibration and validation only, no test "
        "data. Private, internal use. Built by `scripts/data/pack_v3_w120.py` from samples v3 (same rows, labels and "
        "events); the columns are those of `configs/features_214.json`. Source data: VitalDB (Lee HC et al., "
        "Scientific Data 2022), used under its data use terms.\n", encoding="utf-8")
    print("total", sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6, "MB")


if __name__ == "__main__":
    main()
