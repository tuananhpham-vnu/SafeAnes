#!/usr/bin/env bash
# In ví dụ output của bước tiền xử lý (data/prep_v1) để người khác dễ hiểu:
#   1. FORMAT  – cấu trúc thư mục và các key trong mỗi file
#   2. TYPE    – kiểu dữ liệu, shape, đơn vị, ý nghĩa, bảng mã cờ nhiễu
#   3. SAMPLE  – vài dòng thật của một ca
#
#   bash scripts/inspect_prep_v1.sh [caseid] [--rows N]
#
# Mặc định lấy ca đầu tiên đã xử lý xong và in 5 dòng mỗi bảng.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREP="${PREP:-$ROOT/data/prep_v1}"   # PREP=/kaggle/working/prep_v1 nếu đã chạy với --out khác
CASEID=""
ROWS=5
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rows) ROWS="$2"; shift 2 ;;
    *) CASEID="$1"; shift ;;
  esac
done

if [[ ! -d "$PREP/cases" ]]; then
  echo "Chưa có $PREP/cases — chạy trước: python scripts/data/preprocess_vitaldb.py" >&2
  exit 1
fi

cd "$ROOT"
export PYTHONPATH=src   # relative: PYTHONPATH separators differ between Linux (:) and Windows (;)
export PYTHONIOENCODING=utf-8
python - "$PREP" "$CASEID" "$ROWS" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from safeanes.preprocess import (BEAT_COLUMNS, CHANNELS, FLAG_NAMES, B_ARTIFACT, B_DAMPED,
                                 B_IRREGULAR, B_SHAPE, W_DAMPED, W_FLAT, W_FLUSH, W_MISSING)

prep, caseid, rows = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
done = sorted(int(p.stem) for p in (prep / "cases").glob("*.npz") if p.stem.isdigit())
if not done:
    sys.exit("Chưa có ca nào trong data/prep_v1/cases")
caseid = int(caseid) if caseid else done[0]
z = np.load(prep / "cases" / f"{caseid}.npz")
wave_path = prep / "wave100" / f"{caseid}.npy"

def title(text):
    print("\n" + "=" * 90 + f"\n{text}\n" + "=" * 90)

# ----------------------------------------------------------------------------- 1. FORMAT
title("1. FORMAT — cấu trúc output")
print(f"""data/prep_v1/
├── cases/<caseid>.npz     1 file/ca (np.load), {len(done)} ca đã xong
├── wave100/<caseid>.npy   sóng ART 100 Hz của ca (np.load(..., mmap_mode='r'))
├── qc.csv                 1 dòng/ca: độ phủ, tỷ lệ nhiễu, nhịp hợp lệ, độ trễ monitor, baseline
├── errors.json            ca lỗi và exception
├── normalization.json     mean/std tính trên split == train (dùng z-score cho DL)
└── prep.json              version, cấu hình, hash cấu hình

Trục thời gian: giây tính từ lúc bắt đầu ghi ca (giống VitalDB), chỉ giữ [ceil(opstart), floor(opend)).
  lưới 2 s  : t = start + 2*k     -> values / ages / flags
  lưới 1 s  : t = start + k       -> label_map / label_flags / wave_mask
  lưới 100 Hz: t = start + k/100  -> wave100

Ví dụ đọc:
  z = np.load('data/prep_v1/cases/{caseid}.npz')
  t2 = z['start'] + z['step'] * np.arange(len(z['values']))
  map_2s = z['values'][:, list(z['channels']).index('map')]
  wave = np.load('data/prep_v1/wave100/{caseid}.npy', mmap_mode='r')""")

# ----------------------------------------------------------------------------- 2. TYPE
title(f"2. TYPE — kiểu dữ liệu từng key trong cases/{caseid}.npz")
meaning = {
    "start": "thời điểm bắt đầu (s) = ceil(opstart)",
    "step": "bước lưới chỉ số (s) = 2",
    "channels": "tên 14 kênh chỉ số (cột của values/ages/flags)",
    "values": "giá trị hợp lệ gần nhất <= t, tối đa max_age giây; NaN nếu nhiễu/thiếu (không nội suy)",
    "ages": "số giây từ lần đo hợp lệ gần nhất (chặn 600)",
    "flags": "mã cờ (bit) của lần đo gần nhất <= t; != 0 nghĩa là lần đo đó bị loại (values giữ giá trị hợp lệ trước đó nếu còn <= max_age) — dùng làm kênh mask",
    "beat_table_columns": "tên cột của beats",
    "beats": "1 dòng/nhịp tim phát hiện từ sóng 500 Hz",
    "label_map": "MAP 1 s dùng GÁN NHÃN; NaN = không xác định (nhiễu hoặc khoảng trống > 10 s)",
    "label_flags": "mã cờ lý do nhiễu của label_map",
    "wave_mask": "mã cờ nhiễu sóng theo từng giây (0 = sạch)",
    "baseline": "[MAP nền, SBP nền, nguồn] — nguồn: 1 NIBP trước khởi mê, 2 NIBP đầu trước rạch da, 3 ART trước rạch da, 0 thiếu",
}
table = [{"key": k, "dtype": str(z[k].dtype), "shape": str(z[k].shape), "ý nghĩa": meaning.get(k, "")}
         for k in z.files]
if wave_path.exists():
    w = np.load(wave_path, mmap_mode="r")
    table.append({"key": "wave100 (.npy)", "dtype": str(w.dtype), "shape": str(w.shape),
                  "ý nghĩa": "sóng ART mmHg 100 Hz, lọc chống răng cưa nhân quả; NaN = mất tín hiệu"})
print(pd.DataFrame(table).to_string(index=False))

print("\nKênh chỉ số (cột của values):")
print(pd.DataFrame([{"kênh": k, "track VitalDB": c.track, "khoảng hợp lệ": f"{c.lo}–{c.hi}",
                     "max_age (s)": c.max_age, "kiểm tra thêm": ", ".join(c.checks) or "-"}
                    for k, c in CHANNELS.items()]).to_string(index=False))

print("\nMã cờ flags / label_flags (cộng bit, 0 = sạch):")
print("  " + "  ".join(f"{b}={n}" for b, n in FLAG_NAMES.items()))
print("Mã cờ wave_mask:", f"{W_MISSING}=missing  {W_FLUSH}=flush (bơm rửa dây)  {W_FLAT}=flat (zero/lấy máu)  {W_DAMPED}=damped")
print("Cột reject của beats:", f"{B_SHAPE}=hình dạng bất thường  {B_ARTIFACT}=nằm trong đoạn nhiễu  "
      f"{B_IRREGULAR}=nhịp không đều  {B_DAMPED}=damping")

# ----------------------------------------------------------------------------- 3. SAMPLE
title(f"3. SAMPLE — ca {caseid}")
start, step = float(z["start"]), float(z["step"])
names = [str(c) for c in z["channels"]]
b = z["baseline"]
print(f"start={start:.0f}s  thời gian mổ={len(z['label_map']) / 3600:.2f} h  "
      f"baseline MAP={b[0]:.1f}  SBP={b[1]:.1f}  nguồn={int(b[2])}")

values = pd.DataFrame(z["values"], columns=names)
values.insert(0, "time", start + step * np.arange(len(values)))
mid = len(values) // 2
print(f"\nvalues (lưới 2 s), {rows} dòng giữa ca:")
print(values.iloc[mid:mid + rows].round(2).to_string(index=False))

flags = pd.DataFrame(z["flags"], columns=names)
bad = np.flatnonzero((z["flags"][:, names.index("map")] & ~np.uint8(64 | 128)) > 0)
if len(bad):
    i = int(bad[0])
    print(f"\nVí dụ MAP bị loại ở t={start + step * i:.0f}s. map_flag là cờ của lần đo mới nhất; "
          "cột map vẫn là giá trị hợp lệ trước đó (tối đa 30 s), quá 30 s thì thành NaN:")
    sl = slice(max(i - 2, 0), i + 3)
    show = pd.DataFrame({"time": values.time.iloc[sl], "map": values["map"].iloc[sl],
                         "map_flag": flags["map"].iloc[sl],
                         "lý do": [", ".join(n for bit, n in FLAG_NAMES.items() if f & bit) or "-"
                                   for f in flags["map"].iloc[sl]]})
    print(show.to_string(index=False))

label = pd.DataFrame({"time": start + np.arange(len(z["label_map"])), "label_map": z["label_map"],
                      "label_flags": z["label_flags"], "wave_mask": z["wave_mask"]})
print(f"\nlabel_map (lưới 1 s), {rows} dòng giữa ca:")
print(label.iloc[len(label) // 2:len(label) // 2 + rows].to_string(index=False))
known = np.isfinite(z["label_map"])
print(f"  -> {known.mean() * 100:.1f}% giây xác định được; {np.mean(z['label_map'][known] < 65) * 100:.1f}% giây MAP < 65")

beats = pd.DataFrame(z["beats"], columns=[str(c) for c in z["beat_table_columns"]])
print(f"\nbeats: {len(beats)} nhịp, {np.mean(beats.reject == 0) * 100:.1f}% hợp lệ; {rows} nhịp giữa ca:")
print(beats.iloc[len(beats) // 2:len(beats) // 2 + rows].round(3).to_string(index=False))

if wave_path.exists():
    w = np.load(wave_path, mmap_mode="r")
    k = len(w) // 2
    print(f"\nwave100: {len(w)} mẫu; 1 giây giữa ca (mỗi 10 mẫu = 0,1 s):")
    print(np.round(np.asarray(w[k:k + 100:10], dtype=float), 1))

qc_path = prep / "qc.csv"
if qc_path.exists():
    qc = pd.read_csv(qc_path)
    row = qc[qc.caseid.eq(caseid)]
    if len(row):
        cols = ["caseid", "split", "has_wave", "map_coverage", "map_input_artefact_frac",
                "map_label_artefact_frac", "label_known_frac", "beat_valid_frac",
                "wave_lag_seconds", "wave_lag_corr", "baseline_source"]
        print("\nqc.csv (dòng của ca này):")
        print(row[[c for c in cols if c in row]].round(4).T.to_string(header=False))
    print(f"\nqc.csv tổng: {len(qc)} ca; split = {qc.split.value_counts().to_dict()}")

norm = prep / "normalization.json"
if norm.exists():
    n = json.loads(norm.read_text(encoding="utf-8"))
    print(f"\nnormalization.json ({n['scope']}): MAP mean={n['numeric']['map']['mean']:.2f} "
          f"std={n['numeric']['map']['std']:.2f}; wave100 mean={n['wave100']['mean']:.2f} std={n['wave100']['std']:.2f}")
else:
    print("\nnormalization.json: chưa có (tạo khi chạy xong toàn cohort)")
PY
