#!/usr/bin/env bash
# E08 v3: method comparison on every eligible VitalDB case except the locked global test.
#   FIT         = development_seen + unseen_train  (2.559 ca nếu có đủ dữ liệu tiền xử lý)
#   CALIBRATION = unseen_calibration               (280 ca)
#   VALIDATION  = unseen_validation                (271 ca)
#   unseen_test (516 ca) is never read.
# Ca nào thiếu file tiền xử lý sẽ bị bỏ qua, có cảnh báo, và số ca thật được ghi vào báo cáo.
#
# Run with bash, not python:
#   bash scripts/e08/run_v3_full_vitaldb.sh                  # TabM skipped — fastest
#   TABM=frozen  bash scripts/e08/run_v3_full_vitaldb.sh     # reuse the E06 TabM backbone, recalibrated
#   TABM=retrain bash scripts/e08/run_v3_full_vitaldb.sh     # retrain TabM on FIT — many extra CPU hours
#   Kaggle notebook cell:  !bash scripts/e08/run_v3_full_vitaldb.sh
#
# data/, reports/ and artifacts/ are git-ignored, so a fresh clone lacks the E07 inputs. Upload them as a
# dataset whose layout mirrors the repo (data/vitaldb_full/csv_cases, reports/E07/cohort_manifest.csv,
# data/development300/dataset.json, and artifacts/E06 for TABM=frozen); the script finds it under
# /kaggle/input automatically, or set E08_INPUT=/path/to/that/folder.
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

link_inputs() {
  for rel in data/vitaldb_full/csv_cases data/vitaldb_full/cases reports/E07/cohort_manifest.csv data/development300/dataset.json artifacts/E06; do
    if [ ! -e "$rel" ] && [ -e "$1/$rel" ]; then
      mkdir -p "$(dirname "$rel")" && ln -s "$1/$rel" "$rel" && echo "liên kết $rel -> $1/$rel"
    fi
  done
}
if [ -n "${E08_INPUT:-}" ]; then
  link_inputs "$E08_INPUT"
elif [ ! -e data/vitaldb_full/csv_cases ] && [ -d /kaggle/input ]; then
  found="$(find /kaggle/input -maxdepth 8 -type d -path '*/data/vitaldb_full/csv_cases' -print -quit)"
  [ -z "$found" ] || link_inputs "${found%/data/vitaldb_full/csv_cases}"
fi

for path in reports/E07/cohort_manifest.csv data/development300/dataset.json; do
  [ -e "$path" ] || { echo "Thiếu $path — upload dữ liệu E07 (xem đầu file) hoặc đặt E08_INPUT." >&2; exit 1; }
done
[ -d data/vitaldb_full/csv_cases ] || [ -d data/vitaldb_full/cases ] || {
  echo "Thiếu data/vitaldb_full/csv_cases (hoặc cases) — không có dữ liệu tiền xử lý để chạy." >&2; exit 1; }
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
echo "Kaggle: phiên tối đa 12 giờ — nên dùng 'Save Version > Save & Run All' để chạy nền."
"$PYTHON" scripts/e08/run_comparison.py --dataset full --tabm "$TABM" 2>&1 | tee "$log"
