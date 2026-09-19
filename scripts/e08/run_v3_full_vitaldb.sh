#!/usr/bin/env bash
# E08 v3: method comparison on every eligible VitalDB case except the locked global test.
#   FIT         = development_seen + unseen_train  (2.559 ca, 3.069/3.326 biến cố)
#   CALIBRATION = unseen_calibration               (280 ca, 293/308 biến cố)
#   VALIDATION  = unseen_validation                (271 ca, 307/333 biến cố)
#   unseen_test (516 ca) is never read.
#
# Usage (Git Bash on Windows, or any bash):
#   bash scripts/e08/run_v3_full_vitaldb.sh                  # TabM skipped — fastest
#   TABM=frozen  bash scripts/e08/run_v3_full_vitaldb.sh     # reuse the E06 TabM backbone, recalibrated
#   TABM=retrain bash scripts/e08/run_v3_full_vitaldb.sh     # retrain TabM on FIT — many extra CPU hours
#   PYTHON=/path/to/python bash scripts/e08/run_v3_full_vitaldb.sh
#
# Interrupted or crashed? Run the same command again: finished methods reload from artifacts/E08/cache/.
# When it finishes, the v2 results move to reports/E08/version/ and README.md shows v3.
set -euo pipefail
cd "$(dirname "$0")/../.."

TABM="${TABM:-skip}"
PYTHON="${PYTHON:-python}"
case "$TABM" in
  skip|frozen|retrain) ;;
  *) echo "TABM phải là skip, frozen hoặc retrain (đang là '$TABM')" >&2; exit 2 ;;
esac

# Windows Python separates PYTHONPATH entries with ';', POSIX Python with ':'.
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) SEP=";" ;; *) SEP=":" ;; esac
export PYTHONPATH="src${SEP}.local_deps${PYTHONPATH:+${SEP}${PYTHONPATH}}"
export PYTHONIOENCODING=utf-8

for path in data/vitaldb_full/csv_cases reports/E07/cohort_manifest.csv data/development300/dataset.json; do
  [ -e "$path" ] || { echo "Thiếu $path — cần dữ liệu đã tiền xử lý của E07." >&2; exit 1; }
done
if [ "$TABM" = "frozen" ]; then
  [ -d artifacts/E06/tabm_20260917 ] || { echo "TABM=frozen cần artifacts/E06/tabm_*/bundle.joblib." >&2; exit 1; }
fi

modules="lightgbm, catboost, sklearn, pandas"
[ "$TABM" = "skip" ] || modules="$modules, torch, tabm, rtdl_num_embeddings"
"$PYTHON" -c "import $modules" 2>/dev/null || {
  echo "Thiếu thư viện ($modules). Cài: $PYTHON -m pip install -e \".[tabular,catboost]\"" >&2; exit 1; }

mkdir -p reports/E08
log="reports/E08/run_v3_tabm-${TABM}_$(date +%Y%m%d-%H%M%S).log"
echo "E08 v3 | TabM=$TABM | log: $log"
echo "Ước tính 4–8 giờ trên CPU (chưa đo; TabM=retrain lâu hơn nhiều). Cần ~3–4 GB RAM trống."
"$PYTHON" scripts/e08/run_comparison.py --dataset full --tabm "$TABM" 2>&1 | tee "$log"
