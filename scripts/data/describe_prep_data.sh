#!/usr/bin/env bash
# Đối chiếu output tiền xử lý (data/prep_v1) với yêu cầu mục 2.1 / 2.2 của kế hoạch nghiên cứu,
# kèm số liệu demo của một ca: mỗi yêu cầu -> trạng thái, nằm ở đâu trong output, giá trị ví dụ.
#
#   bash scripts/data/describe_prep_data.sh [caseid]
#   PREP=/kaggle/working/prep_v1 bash scripts/data/describe_prep_data.sh 7
#
# Trạng thái: [ĐÃ CÓ] có trong output; [CHƯA CHẠY] chưa chạy scripts/data/build_features.py.
# Nguồn: cases/<caseid>.npz (tiền xử lý), features/<caseid>.parquet (đặc trưng cửa sổ W),
# reports/PREP/proxy_vs_ev1000_summary.csv (kiểm chứng SV/CO/SVR/PPV với EV1000).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8
PREP="${PREP:-data/prep_v1}"
if [[ ! -d "$PREP/cases" ]]; then
  echo "Chưa có $PREP/cases — chạy trước: bash scripts/data/run_preprocess_data.sh" >&2
  exit 1
fi

python - "$PREP" "${1:-}" <<'PY'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from safeanes.preprocess import (B_ARTIFACT, B_DAMPED, B_IRREGULAR, B_SHAPE, CHANNELS, FLAT, JUMP, PP,
                                 RANGE, WAVE, W_FLAT, W_FLUSH)

prep = Path(sys.argv[1])
done = sorted(int(p.stem) for p in (prep / "cases").glob("*.npz") if p.stem.isdigit())
caseid = int(sys.argv[2]) if sys.argv[2] else done[0]
z = np.load(prep / "cases" / f"{caseid}.npz")
qc = pd.read_csv(prep / "qc.csv") if (prep / "qc.csv").exists() else pd.DataFrame(columns=["caseid"])
q = qc[qc.caseid.eq(caseid)].iloc[0] if qc.caseid.eq(caseid).any() else None
names = [str(c) for c in z["channels"]]
start, step = float(z["start"]), float(z["step"])
V, F, A = z["values"], z["flags"], z["ages"]
t2 = start + step * np.arange(len(V))
beats = pd.DataFrame(z["beats"], columns=[str(c) for c in z["beat_table_columns"]])
good = beats[beats.reject.eq(0)]
mid = len(V) // 2
col = lambda n: V[:, names.index(n)]


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def item(status, need, where, demo):
    print(f"\n[{status}] {need}\n    ở đâu : {where}\n    demo  : {demo}")


def pct(x):
    return f"{100 * x:.2f}%"


print(f"Ca {caseid}: {len(z['label_map']) / 3600:.2f} h mổ, start={start:.0f}s; {len(done)} ca trong {prep}")

# ----------------------------------------------------------------------------- 2.1.1
section("2.1.1 Chỉ số monitor được chọn")
spec = [("map", "MAP — huyết áp trung bình, dùng gán nhãn"), ("sbp", "SBP — tâm thu"),
        ("dbp", "DBP — tâm trương"), ("hr", "Nhịp tim"), ("spo2", "SpO2"),
        ("etco2", "EtCO2 (chỉ dùng nếu MOVER có)"), ("rr", "Tần số thở (chỉ dùng nếu MOVER có)")]
rows = []
for n, desc in spec:
    x = col(n)
    rows.append({"chỉ số": desc, "track": CHANNELS[n].track, "kênh": n,
                 "độ phủ": pct(np.isfinite(x).mean()), f"giá trị t={t2[mid]:.0f}s": x[mid]})
print(pd.DataFrame(rows).to_string(index=False))
print("  Kênh bổ sung cho mục IV (độ mê, thở máy, CVP):",
      ", ".join(f"{n} {pct(np.isfinite(col(n)).mean())}" for n in names if n not in dict(spec)))
b = z["baseline"]
src = {0: "không có", 1: "NIBP trước khởi mê (đúng yêu cầu)", 2: "NIBP đầu tiên trước rạch da (thay thế)",
       3: "ART sạch trước rạch da (thay thế)"}[int(b[2])]
item("ĐÃ CÓ*", "MAP nền = Solar8000/NIBP_MBP trước khởi mê", "baseline = [MAP, SBP, nguồn]",
     f"MAP nền {b[0]:.1f}, SBP nền {b[1]:.1f}, nguồn: {src}")
if len(qc):
    vc = qc.baseline_source.value_counts(normalize=True)
    print("    *toàn bộ ca đã xử lý: " + ", ".join(f"nguồn {int(k)} {pct(v)}" for k, v in vc.sort_index().items())
          + " — VitalDB hiếm khi ghi trước khởi mê nên phần lớn dùng mức thay thế")

# ----------------------------------------------------------------------------- 2.1.3
section("2.1.3 Tiền xử lý chỉ số")
m, fm = col("map"), F[:, names.index("map")]
count = lambda bit: int(((fm & bit) > 0).sum())
first = lambda bit: f"t={t2[np.flatnonzero((fm & bit) > 0)[0]]:.0f}s" if count(bit) else "-"
item("ĐÃ CÓ", "Loại giá trị ngoài khoảng hợp lý (MAP 20–200), giữ MAP thấp có thật", "flags bit 1 (range)",
     f"{count(RANGE)} ô 2 s bị loại (đầu tiên {first(RANGE)}); MAP thấp nhất còn giữ = {np.nanmin(m):.0f} mmHg; "
     f"{int(np.sum(m < 65))} ô 2 s có MAP < 65 được giữ")
item("ĐÃ CÓ", "Đánh dấu hiệu áp nhỏ bất thường (SBP − DBP < 10)", "flags bit 2 (pulse_pressure)",
     f"{count(PP)} ô (đầu tiên {first(PP)})")
item("ĐÃ CÓ", "Đánh dấu MAP nhảy đột ngột (> 30 mmHg so với median 30 s trước)", "flags bit 4 (jump)",
     f"{count(JUMP)} ô (đầu tiên {first(JUMP)}); khi gán nhãn chỉ loại đợt nhảy rồi quay lại, đợt giảm kéo dài được giữ")
item("ĐÃ CÓ", "Đánh dấu giá trị đứng yên quá lâu (≥ 60 s)", "flags bit 8 (flat)", f"{count(FLAT)} ô (đầu tiên {first(FLAT)})")
ages = A[:, names.index("map")]
item("ĐÃ CÓ", "Chỉ dùng giá trị đo trước t, tối đa 30 s, không nội suy", "values + ages (lưới 2 s)",
     f"tuổi dữ liệu MAP: median {np.median(ages[np.isfinite(m)]):.1f} s, max {ages[np.isfinite(m)].max():.1f} s "
     f"(≤ 30); {int(np.isnan(m).sum())} ô NaN vì không có đo hợp lệ trong 30 s")

# ----------------------------------------------------------------------------- 2.1.4
fpath = prep / "features" / f"{caseid}.parquet"
feat = pd.read_parquet(fpath) if fpath.exists() else None
FEAT_STATUS = "ĐÃ CÓ" if feat is not None else "CHƯA CHẠY"
section(f"2.1.4 Đặc trưng chỉ số — features/{caseid}.parquet, 1 dòng mỗi 30 s, W = 30/60/90/120 s")
if feat is None:
    print("\nChưa có features — chạy: python scripts/data/build_features.py")
else:
    r = feat.iloc[len(feat) // 2]
    print(f"{len(feat)} mốc dự đoán × {feat.shape[1] - 2} đặc trưng; ví dụ tại t={r.time:.0f}s:")
    W = 60
    fmt = lambda *cols: ", ".join(f"{c}={r[c]:.2f}" for c in cols)
    item(FEAT_STATUS, "Trung bình, độ lệch chuẩn, min, max, độ dốc mỗi chỉ số", "<kênh>_{mean,std,min,max,slope}_w<W> (dốc: /phút)",
         fmt(*(f"map_{s}_w{W}" for s in ("mean", "std", "min", "max", "slope"))))
    item(FEAT_STATUS, "Giá trị hiện tại, tỷ lệ thiếu dữ liệu", "<kênh>_current, <kênh>_missing_w<W>",
         fmt("map_current", "hr_current", f"map_missing_w{W}", f"bis_missing_w{W}"))
    item(FEAT_STATUS, "Hiệu áp (SBP − DBP), shock index (HR / SBP)", "pp_*, shock_index_* (đủ các thống kê như kênh khác)",
         fmt("pp_current", f"pp_mean_w{W}", "shock_index_current"))
    item(FEAT_STATUS, "% MAP giảm so với nền, thời gian MAP ở 65–75 mmHg", "map_drop_pct, map_time_65_75_w<W> (giây)",
         fmt("map_drop_pct", "map_time_65_75_w120"))
    item(FEAT_STATUS, "MAP ngoại suy tối đa 5 phút, giới hạn 30–150", "map_extrap_w<W> = hiện tại + dốc_W × 5 phút",
         fmt("map_extrap_w30", "map_extrap_w120"))

# ----------------------------------------------------------------------------- 2.2.1
section("2.2.1 Tiền xử lý sóng huyết áp động mạch (SNUADC/ART 500 Hz)")
if beats.empty:
    print("\nCa này không có sóng ART — chọn ca khác.")
else:
    item("ĐÃ CÓ", "Phát hiện từng nhịp: đỉnh, khấc sóng, đáy", "beats: onset (đáy), sbp (đỉnh), dbp (đáy), notch_rel, notch_found",
         f"{len(beats)} nhịp; tìm được khấc sóng thật ở {pct(beats.notch_found.mean())} nhịp "
         "(còn lại lấy điểm uốn)")
    wm = z["wave_mask"]
    item("ĐÃ CÓ", "Loại bơm rửa dây (sóng vuông) / chỉnh zero, lấy máu (sóng phẳng)",
         "wave_mask theo giây: bit 2 flush, bit 4 flat; beats.reject bit 2",
         f"{int(((wm & W_FLUSH) > 0).sum())} s flush, {int(((wm & W_FLAT) > 0).sum())} s flat; "
         f"{int(((beats.reject.astype(int) & B_ARTIFACT) > 0).sum())} nhịp bị loại vì nằm trong đoạn này")
    rej = beats.reject.astype(int)
    item("ĐÃ CÓ", "Loại damping (sóng bị làm tròn) và nhịp bất thường", "beats.reject bit 8 damped, bit 4 không đều, bit 1 hình dạng",
         f"damped {int(((rej & B_DAMPED) > 0).sum())}, không đều {int(((rej & B_IRREGULAR) > 0).sum())}, "
         f"hình dạng {int(((rej & B_SHAPE) > 0).sum())}; hợp lệ {pct((rej == 0).mean())}")
    lf = z["label_flags"]
    lag = f"{q.wave_lag_seconds:.0f}s (r={q.wave_lag_corr:.3f}), lệch hệ thống {q.wave_map_bias:.1f} mmHg" if q is not None else "-"
    item("ĐÃ CÓ", "So MAP từ sóng với Solar8000/ART_MBP, đoạn lệch nhiều không dùng gán nhãn",
         "label_flags bit 16 (waveform) -> label_map = NaN; qc.csv wave_lag_*",
         f"monitor trễ so với sóng {lag}; {int(((lf & WAVE) > 0).sum())} s bị loại khỏi nhãn vì lệch/nhiễu sóng")
    w = prep / "wave100" / f"{caseid}.npy"
    if w.exists():
        n = len(np.load(w, mmap_mode="r"))
        item("ĐÃ CÓ", "Sóng 100 Hz sau lọc chống răng cưa cho deep learning", "wave100/<caseid>.npy (float16)",
             f"{n} mẫu = {n / 100 / 3600:.2f} h")

# ----------------------------------------------------------------------------- 2.2.2
section("2.2.2 Đặc trưng sóng — giá trị từng nhịp (beats) + thống kê theo cửa sổ (features, cột bt_*)")
if feat is not None and not beats.empty:
    W = 60
    fmt = lambda *cols: ", ".join(f"{c}={r[c]:.3f}" for c in cols)
    item(FEAT_STATUS, "dP/dt max (sức co bóp)", "beats.dpdt_max; bt_dpdt_max_{mean,std,slope}_w<W>",
         fmt(*(f"bt_dpdt_max_{s}_w{W}" for s in ("mean", "std", "slope"))))
    item(FEAT_STATUS, "Diện tích tâm thu, thời gian tống máu", "bt_sys_area_*, bt_ejection_time_*",
         fmt(f"bt_sys_area_mean_w{W}", f"bt_ejection_time_mean_w{W}"))
    item(FEAT_STATUS, "Độ cao khấc sóng, tốc độ giảm áp tâm trương", "bt_notch_rel_* ((P_khấc − DBP)/PP), bt_decay_tau_* (s)",
         fmt(f"bt_notch_rel_mean_w{W}", f"bt_decay_tau_mean_w{W}"))
    item(FEAT_STATUS, "PPV — cửa sổ ≥ 30 s, thở máy, nhịp đều", "bt_ppv_w<W>: NaN nếu < 5 nhịp, có nhịp bị loại, CV chu kỳ > 10%, "
         "hoặc không thở máy (RR ngoài 6–40 hoặc EtCO2 < 10)",
         fmt(f"bt_ppv_w30", f"bt_ppv_w{W}") + f"; có PPV ở {pct(feat[f'bt_ppv_w{W}'].notna().mean())} mốc của ca")
    summary = Path("reports/PREP/proxy_vs_ev1000_summary.csv")
    check = ""
    if summary.exists():
        s = pd.read_csv(summary).set_index("proxy")
        check = "; kiểm chứng EV1000 (Spearman trong ca, median): " + ", ".join(
            f"{k} {s.loc[k, 'rho_level_median']:.2f} ({int(s.loc[k, 'ca'])} ca)" for k in s.index)
    item(FEAT_STATUS, "SV, CO, SVR ước lượng (tương đối), kiểm chứng EV1000", "bt_sv_*, bt_co_*, bt_svr_* (Liljestrand-Zander); "
         "reports/PREP/proxy_vs_ev1000*.csv",
         fmt(f"bt_sv_mean_w{W}", f"bt_co_mean_w{W}", f"bt_svr_mean_w{W}") + check)
    item(FEAT_STATUS, "Tỷ lệ nhịp hợp lệ trong cửa sổ", "bt_valid_frac_w<W>, bt_n_beats_w<W>",
         fmt(f"bt_valid_frac_w{W}", f"bt_n_beats_w{W}"))
PY
