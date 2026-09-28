#!/usr/bin/env bash
# Tiền xử lý VitalDB (research plan V.2 + đặc trưng 2.1.4 / 2.2.2) từ một bản clone mới.
# Dùng được trên Kaggle (bật Internet trong Settings của notebook).
#
#   !bash scripts/data/run_preprocess_data.sh                   # bước 0 -> 4
#   !DEMO_ONLY=1 bash scripts/data/run_preprocess_data.sh       # chỉ bước 0 -> 3 (xem demo, vài phút)
#   !DEMO_CASES="1 4 7" WORKERS=2 bash scripts/data/run_preprocess_data.sh
#
# Các bước:
#   0. Kiểm tra môi trường: thư viện, Internet tới VitalDB, CPU/RAM, device (cuda/cpu)
#   1. Demo: tiền xử lý + đặc trưng cho DEMO_CASES
#   2. Xem output: format, kiểu dữ liệu, ví dụ           (scripts/inspect_prep_v1.sh)
#   3. Mô tả: đối chiếu yêu cầu 2.1 / 2.2 kèm số liệu     (scripts/data/describe_prep_data.sh)
#   4. Toàn bộ 3.626 ca + đặc trưng + kiểm chứng EV1000 (bỏ qua ca đã xong, chạy lại được)
#
# Biến môi trường:
#   OUT         thư mục output                 (mặc định data/prep_v1)
#   WORKERS     số process song song            (mặc định: tự chọn theo CPU và RAM, ~1,5 GB/worker)
#   DEMO_CASES  ca dùng cho demo                (mặc định "1 4 7")
#   DEMO_ONLY   1 = dừng sau bước 3             (mặc định 0)
#
# Tiền xử lý chạy trên CPU (numpy/scipy): thời gian nằm ở tải sóng từ API và phát hiện từng nhịp,
# GPU không tăng tốc bước này. Device được kiểm tra và ghi vào log cho bước huấn luyện DL
# (batching trên cuda).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8

OUT="${OUT:-data/prep_v1}"
DEMO_CASES="${DEMO_CASES:-1 4 7}"
DEMO_ONLY="${DEMO_ONLY:-0}"
mkdir -p "$OUT"
LOG="$OUT/run_$(date +%Y%m%d-%H%M%S).log"
FIRST_DEMO="${DEMO_CASES%% *}"

step() { echo; echo "######## $* ########" | tee -a "$LOG"; }

# --------------------------------------------------------------------------- 0
step "0. Kiểm tra môi trường"
python - <<'PY' 2>&1 | tee -a "$LOG"
import os, platform, urllib.request
import numpy, pandas, scipy
print(f"python {platform.python_version()} | numpy {numpy.__version__} | pandas {pandas.__version__} | scipy {scipy.__version__}")
try:
    import pyarrow
    print("pyarrow", pyarrow.__version__)
except ImportError:
    raise SystemExit("Thiếu pyarrow (cần để ghi features/*.parquet): pip install pyarrow")
cpus = os.cpu_count() or 1
try:
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
except (ValueError, AttributeError, OSError):
    ram = float("nan")
print(f"CPU: {cpus} | RAM: {ram:.1f} GB")
try:
    import torch
    if torch.cuda.is_available():
        names = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
        print(f"device = cuda ({torch.cuda.device_count()} GPU: {names}) — dùng cho batching khi huấn luyện DL")
    else:
        print("device = cpu (không có GPU) — tiền xử lý không cần GPU; bật GPU trước bước huấn luyện DL")
except ImportError:
    print("device = cpu (chưa cài torch) — tiền xử lý không cần torch")
urllib.request.urlopen("https://api.vitaldb.net/cases", timeout=30)
print("VitalDB API: OK")
PY
if [[ -z "${WORKERS:-}" ]]; then
  WORKERS=$(python -c "
import os
cpus = os.cpu_count() or 1
try:
    ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30
except (ValueError, AttributeError, OSError):
    ram = 8
print(max(1, min(cpus, int(ram // 1.5), 8)))")
fi
echo "out=${OUT} workers=${WORKERS} demo=${DEMO_CASES} | log: ${LOG}" | tee -a "$LOG"

prep() { python -W ignore scripts/data/preprocess_vitaldb.py --out "$OUT" --workers "$WORKERS" "$@" 2>&1 | tee -a "$LOG"; }
feats() { python -W ignore scripts/data/build_features.py --prep "$OUT" --workers "$WORKERS" "$@" 2>&1 | tee -a "$LOG"; }

# --------------------------------------------------------------------------- 1
step "1. Demo: tiền xử lý + đặc trưng cho ca ${DEMO_CASES}"
# shellcheck disable=SC2086
prep --caseids $DEMO_CASES
# shellcheck disable=SC2086
feats --caseids $DEMO_CASES

# --------------------------------------------------------------------------- 2
step "2. Xem output — format, kiểu dữ liệu, ví dụ (ca ${FIRST_DEMO})"
PREP="$OUT" bash scripts/inspect_prep_v1.sh "$FIRST_DEMO" --rows 3 2>&1 | tee -a "$LOG"

# --------------------------------------------------------------------------- 3
step "3. Mô tả — đối chiếu yêu cầu 2.1 / 2.2 (ca ${FIRST_DEMO})"
PREP="$OUT" bash scripts/data/describe_prep_data.sh "$FIRST_DEMO" 2>&1 | tee -a "$LOG"

if [[ "$DEMO_ONLY" == "1" ]]; then
  echo; echo "DEMO_ONLY=1: dừng sau bước 3. Chạy toàn bộ: bash scripts/data/run_preprocess_data.sh" | tee -a "$LOG"
  exit 0
fi

# --------------------------------------------------------------------------- 4
step "4. Toàn bộ cohort (ca đã xong được bỏ qua)"
prep
feats
python -W ignore scripts/data/validate_proxies_ev1000.py --prep "$OUT" 2>&1 | tee -a "$LOG"
du -sh "$OUT"/cases "$OUT"/wave100 "$OUT"/features 2>/dev/null | tee -a "$LOG" || true
echo "Xong. Xem lại: PREP=$OUT bash scripts/data/describe_prep_data.sh <caseid>" | tee -a "$LOG"
