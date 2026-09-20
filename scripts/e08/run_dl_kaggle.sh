#!/usr/bin/env bash
# E08 DL family end to end from a fresh clone, sized for a Kaggle GPU session.
#
#   !bash scripts/e08/run_dl_kaggle.sh                 # tcn, 1 seed
#   ARCH="tcn inception" SEEDS="20260917 20260918" !bash scripts/e08/run_dl_kaggle.sh
#
# Steps 1 and 2 rebuild what git does not carry (only part of data/vitaldb_full/cases is tracked,
# and the ~1 GB sequence cache is not tracked at all). On a fresh Kaggle session both pull raw
# tracks from the VitalDB API -- about 25k files, ~30 minutes in total, ~3 GB on disk. Both resume
# from what is already on disk, so a rerun in the same session skips straight to training.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8

ARCH="${ARCH:-tcn}"
SEEDS="${SEEDS:-20260917}"
EPOCHS="${EPOCHS:-20}"
WIDTH="${WIDTH:-32}"
WORKERS="${WORKERS:-2}"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p reports/E08
LOG="reports/E08/dl_${STAMP}.log"

echo "E08 DL | arch=${ARCH} seeds=${SEEDS} epochs=${EPOCHS} width=${WIDTH} | log: ${LOG}"
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY

{
  echo "--- 1/3 preprocessed cases ---"
  python scripts/e08/build_missing_cases.py --workers 8

  echo "--- 2/3 sequence cache ---"
  python scripts/e08/build_sequences_full.py --workers 8

  echo "--- 3/3 train + evaluate ---"
  python scripts/e08/run_sequence_dl.py \
    --arch ${ARCH} --seeds ${SEEDS} --epochs "${EPOCHS}" \
    --width "${WIDTH}" --workers "${WORKERS}"
} 2>&1 | tee "${LOG}"

echo "kết quả: reports/E08/sequence_comparison.csv"
