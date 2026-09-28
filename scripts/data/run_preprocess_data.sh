#!/usr/bin/env bash
# Tiền xử lý VitalDB (research plan V.2) từ một bản clone mới, dùng được trên Kaggle (bật Internet).
#
#   !bash scripts/data/run_preprocess_data.sh                      # thử 1 ca, rồi chạy toàn bộ 3.626 ca
#   CASEIDS="1 4 7" !bash scripts/data/run_preprocess_data.sh      # chỉ các ca này (demo)
#   OUT=/kaggle/working/prep_v1 WORKERS=2 !bash scripts/data/run_preprocess_data.sh
#
# Biến môi trường:
#   OUT      thư mục output          (mặc định data/prep_v1)
#   WORKERS  số process song song    (mặc định 4; giảm xuống 2 nếu hết RAM, ~1 GB/worker)
#   CASEIDS  danh sách caseid         (mặc định rỗng = toàn bộ cohort eligible)
#   SMOKE    1 = thử ca SMOKE_CASE trước khi chạy toàn bộ (mặc định 1)
#
# Track số và danh sách track (trks) được tải từ VitalDB API khi thiếu, lưu ở data/vitaldb_full;
# sóng SNUADC/ART được stream, chỉ giữ bản 100 Hz. Chạy lại lệnh sẽ bỏ qua các ca đã xong.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8

OUT="${OUT:-data/prep_v1}"
WORKERS="${WORKERS:-4}"
CASEIDS="${CASEIDS:-}"
SMOKE="${SMOKE:-1}"
SMOKE_CASE="${SMOKE_CASE:-1}"
mkdir -p "$OUT"
LOG="$OUT/run_$(date +%Y%m%d-%H%M%S).log"

echo "Tiền xử lý VitalDB | out=${OUT} workers=${WORKERS} caseids=${CASEIDS:-all} | log: ${LOG}"
python -c "import numpy, pandas, scipy; print('numpy', numpy.__version__, '| pandas', pandas.__version__, '| scipy', scipy.__version__)"
python -c "import urllib.request; urllib.request.urlopen('https://api.vitaldb.net/cases', timeout=30); print('VitalDB API: OK')" \
  || { echo "Không kết nối được api.vitaldb.net — bật Internet cho notebook" >&2; exit 1; }

run() {
  python -W ignore scripts/data/preprocess_vitaldb.py --out "$OUT" --workers "$WORKERS" "$@" 2>&1 | tee -a "$LOG"
}

if [[ -n "$CASEIDS" ]]; then
  # shellcheck disable=SC2086
  run --caseids $CASEIDS
else
  if [[ "$SMOKE" == "1" ]]; then
    echo "== Thử 1 ca (${SMOKE_CASE}) =="
    run --caseids "$SMOKE_CASE"
  fi
  echo "== Toàn bộ cohort (ca đã xong được bỏ qua) =="
  run
fi

echo "== Đặc trưng cửa sổ W = 30/60/90/120 s (mục 2.1.4, 2.2.2) =="
python -W ignore scripts/data/build_features.py --prep "$OUT" --workers "$WORKERS" 2>&1 | tee -a "$LOG"

echo "== Kiểm chứng SV/CO/SVR/PPV ước lượng với EV1000 =="
python -W ignore scripts/data/validate_proxies_ev1000.py --prep "$OUT" 2>&1 | tee -a "$LOG"

echo "== Ví dụ output =="
PREP="$OUT" bash scripts/inspect_prep_v1.sh --rows 3 | tee -a "$LOG"
PREP="$OUT" bash scripts/data/describe_prep_data.sh | tee -a "$LOG"
du -sh "$OUT"/cases "$OUT"/wave100 "$OUT"/features 2>/dev/null || true
