# UC04 v2: Kế hoạch huấn luyện (code độc lập)

> **Dành cho Claude Code và nhóm UC04.** Đọc hết file này trước khi viết code.
>
> - Dự án viết **code mới**, không import và không copy code từ repo SafeAnes.
> - **Đầu vào duy nhất** là thư mục dữ liệu đã tiền xử lý (`prep_v1`), tải về từ output của notebook Kaggle `dainn98s/safeanes`. Ngoại lệ duy nhất: NB06 (làm sau) cần thêm track EV1000 (mục 5.2).
> - **Bản 2.1 (29/09):** cập nhật theo kết quả kiểm tra dữ liệu thật. Thay đổi ở mục 1.1, 1.5, 1.6, 2, 3.1, 3.6, 3.7, 3.8, 5.2, 7.1, 13 và mục 14 mới.
> - **Bản 2.5 (29/09):** sửa các chỗ còn mâu thuẫn của bản 2.4. Notebook chỉ đẩy thư mục của riêng mình, kể cả ở cell cuối (mục 15.3). `calibration_check` ghi riêng từng tổ hợp (mục 8.1). `03_dl_smoke` được đẩy riêng file đo thời gian. `map_threshold` chỉ có 5 tổ hợp, tổng tabular 65 (mục 6.2). Output mỗi tổ hợp có thêm `case_metrics.parquet` (mục 6.3). Chốt dùng Kaggle Dataset (`dataset_sources`) cho sóng thay cho `kernel_sources` (mục 1.1, 4, 15.2, 15.4). Cập nhật cấu trúc `runs_repo` và ví dụ notebook (mục 15.3, 15.4).
> - **Bản 2.4 (29/09):** mục 11 và 12 chia theo 5 giai đoạn, cho phép làm chồng tabular và DL. Sửa lỗi nhiều notebook cùng ghi một file kết quả (mục 6.3), logit theo seed để `03_dl_finalize` chạy trên CPU (mục 7.4), luật `assert_no_test` theo thành phần đường dẫn, chế độ `local` cho notebook, chặn đẩy kết quả chạy thử, và lưu ý về `kernel_sources` (mục 15).
> - **Bản 2.3 (29/09):** đổi nơi chạy. Xử lý dữ liệu (NB01) và khóa, chạy test (NB04) làm trên **máy cá nhân**. Notebook Kaggle được **tạo sẵn ở máy** rồi đẩy lên bằng Kaggle CLI. Train (NB02, NB03) chạy trên Kaggle, kết quả **tự động lưu lên Hugging Face**. Dữ liệu test không bao giờ rời máy cá nhân. Chi tiết ở mục 15. Các mục 1.1, 2, 3.7, 4, 5, 6.3, 7.1, 7.3, 7.4, 9, 11 đã sửa theo.
> - **Bản 2.2 (29/09):** đưa Task 0 của danh sách việc vào kế hoạch (niêm phong nhãn test, số cohort D, θ đã chốt, EV1000, `wave_mask`). Thêm: môi trường chạy test, đưa code lên Kaggle, file provenance, báo cáo cả hai luật loại mẫu, cách tính PPV nhanh, test nhân quả cho DL. Bỏ phần đánh giá hiệu năng riêng trên ca không có sóng.
> - Mọi định dạng dữ liệu và thuật toán cần dùng đều được mô tả trong file này.
> - Nếu proposal và file này khác nhau, làm theo file này và ghi chỗ khác vào `reports/DEVIATIONS.md`.

---

## 0. Tóm tắt bài toán

- Tại mỗi thời điểm t (cứ 30 giây một lần), mô hình nhìn lại W giây dữ liệu monitor (W = 30, 60, 90, 120) và trả lời: *"trong h phút tới có đợt tụt huyết áp **mới** bắt đầu không?"*, với h = 5, 10, 15, 20, 30 phút.
- **Tụt huyết áp** là MAP < 65 mmHg kéo dài ít nhất 60 giây. Hai đợt cách nhau dưới 120 giây được gộp thành một.
- **Input** chỉ lấy từ monitor: chỉ số `Solar8000` và sóng `SNUADC/ART`. Không dùng thuốc, BIS, máy gây mê hay thông tin trước mổ.
- **Dữ liệu:** VitalDB để train và kiểm tra nội bộ. MOVER-SIS chỉ dùng cho kiểm tra ngoài, làm sau.
- **Các mô hình:**

| Tên | Mô tả |
|---|---|
| `map_threshold` | Baseline 1: điểm nguy cơ = −MAP hiện tại |
| `map_logistic` | Baseline 2: logistic với MAP hiện tại, độ dốc MAP, độ lệch chuẩn MAP |
| `lgbm_numeric` | LightGBM, 66 đặc trưng từ chỉ số |
| `lgbm_wave` | LightGBM, 66 đặc trưng từ chỉ số và 27 đặc trưng từ sóng |
| `dl_conv_tf` | Conv1D + Transformer: sóng 100 Hz thô, 66 đặc trưng chỉ số và cờ `baseline_missing` |

- **Tiêu chí chính:** tỷ lệ đợt tụt được cảnh báo trước (event sensitivity) khi chỉ cho phép **1 cảnh báo sai mỗi giờ**.

---

## 1. Dữ liệu đầu vào (output của notebook)

### 1.1 Thư mục

```
<PREP>/                      # ví dụ trên Kaggle: /kaggle/input/safeanes/SafeAnes/data/prep_v1 (kiểm tra bằng ls)
  qc.csv                     # bắt buộc
  cases/<caseid>.npz         # bắt buộc
  features/<caseid>.parquet  # bắt buộc
  features/columns.json
  wave100/<caseid>.npy       # bắt buộc cho DL
  normalization.json         # dùng phần "wave100" cho DL
  prep.json, errors.json     # chỉ để đối chiếu
```

- Code chỉ đọc qua đường dẫn trong config (`paths.prep`). Không hard-code đường dẫn.
- Dữ liệu là **chỉ đọc**. Mọi output ghi ra thư mục khác.
- **Nếu thiếu `wave100/`:** vẫn làm được baseline và LightGBM. DL phải đợi có dữ liệu.
- **Nếu thiếu ca:** ghi danh sách ca thiếu. Không tự tải lại từ VitalDB.
- **Ca lỗi đã biết:** 4 ca `4458, 6119, 6133, 6375` không có trong `prep_v1`. Sóng ART của chúng lấy mẫu ở 100 Hz, trong khi code tiền xử lý yêu cầu 500 Hz. Bỏ 4 ca này và ghi vào `DEVIATIONS.md`.
- **Nơi chạy** (chi tiết ở mục 15):
  - Trên máy cá nhân, `prep_v1` nằm ở `D:\SafeAnes\SafeAnes\data\prep_v1`. NB01 đọc trực tiếp từ đây. Cần cài môi trường Python trước (mục 15.1).
  - Trên Kaggle, chỉ NB03 (và `00_env_check`) cần `wave100/`. Lưu output hiện tại của notebook `safeanes` thành một **Kaggle Dataset private** (*Output → New Dataset*), rồi gắn vào notebook qua `dataset_sources` trong `kernel-metadata.json`. Không phải tải lên lại khoảng 7 GB. Dataset chỉ đổi khi chính nhóm chủ động tạo phiên bản mới, khác với output notebook. Không dùng `kernel_sources`, vì nó luôn lấy output mới nhất. Nếu ai đó chạy lại notebook `safeanes` giữa chừng, dữ liệu sóng sẽ đổi. Các file còn lại (bảng mẫu, `wave_mask`) lấy từ Hugging Face.
  - Để chắc hai nơi dùng cùng một bản dữ liệu, so sha256 của `qc.csv` trên Kaggle với giá trị ghi trong `samples.json`. Khác thì dừng.

### 1.2 `qc.csv`: một dòng mỗi ca

Các cột cần dùng:

| Cột | Ý nghĩa |
|---|---|
| `caseid`, `subjectid` | Mã ca, mã bệnh nhân |
| `split` | `train`, `calibration`, `validation` hoặc `test`. Đã chia theo bệnh nhân, **không chia lại** |
| `has_wave` | Ca có sóng `SNUADC/ART` hay không |
| `surgery_seconds` | Độ dài ca mổ (giây) = độ dài `label_map` |
| `label_known_frac`, `beat_valid_frac`, `n_beats` | Chất lượng dữ liệu, để báo cáo |
| `baseline_source` | Nguồn MAP nền: 1 NIBP trước khởi mê, 2 NIBP đầu tiên trước rạch da, 3 MAP arterial line 5 phút đầu, 0 không có |

### 1.3 `cases/<caseid>.npz`

Đọc bằng `np.load(path)`, không cần `allow_pickle`.

| Khóa | Kiểu, shape | Ý nghĩa |
|---|---|---|
| `start` | float64 | Thời điểm bắt đầu mổ (giây trên đồng hồ của ca VitalDB), là số nguyên |
| `step` | float64 | 2.0, bước của lưới chỉ số |
| `channels` | str [14] | `map, sbp, dbp, hr, spo2, etco2, rr, nibp_mbp, cvp, bis, mac, ppf_ce, rftn_ce, peep` |
| `values` | float32 [n, 14] | Lưới 2 giây, ô i ứng với `start + 2i`. Giá trị là lần đo hợp lệ gần nhất tại hoặc trước thời điểm đó, cũ tối đa 30 giây với 7 chỉ số đầu. **Nhân quả**, dùng được cho input |
| `ages`, `flags` | float32, uint8 [n, 14] | Số giây từ lần đo gần nhất, và bit cờ nhiễu |
| `beat_table_columns`, `beats` | str [15], float64 [m, 15] | Bảng nhịp tim từ sóng 500 Hz (mục 1.4) |
| `label_map` | float32 [surgery_seconds] | Ô j ứng với giây `[start + j, start + j + 1)`. Giá trị là MAP cao nhất trong giây đó. NaN nghĩa là không biết (nhiễu, hoặc khoảng trống > 10 giây). **Có nhìn về sau, chỉ dùng để gán nhãn** |
| `label_flags` | uint8 [surgery_seconds] | Cờ nhiễu của `label_map` |
| `wave_mask` | uint8 [surgery_seconds] | Nhiễu sóng theo khối 1 giây: bit 1 mất sóng, 2 flush, 4 sóng phẳng. Nhân quả |
| `baseline` | float64 [3] | `[MAP nền, SBP nền, baseline_source]` |

### 1.4 Bảng nhịp (`beats`)

Lấy chỉ số cột theo `beat_table_columns`, không giả định thứ tự.

| Cột | Ý nghĩa |
|---|---|
| `onset` | Thời điểm chân sóng của nhịp (giây) |
| `avail` | Thời điểm nhịp **dùng được** = chân sóng của nhịp sau + 1 giây. Chỉ dùng nhịp có `avail ≤ t` |
| `sbp, dbp, map, pp, period, hr` | Áp lực (mmHg), chu kỳ (giây), nhịp tim |
| `dpdt_max, sys_area, ejection_time, notch_rel, decay_tau, notch_found` | Đặc trưng hình dạng nhịp |
| `reject` | 0 là nhịp hợp lệ. Bit 1 hình dạng sai, 2 nằm trong đoạn nhiễu, 4 nhịp không đều, 8 sóng bị damping |

### 1.5 `wave100/<caseid>.npy`

float16, 1 chiều, 100 Hz. Mẫu k ứng với thời điểm `start + k/100`. NaN là mất sóng. Độ dài khoảng `surgery_seconds × 100`. Sóng đã qua lọc chống răng cưa nhân quả rồi lấy 1 trên 5 mẫu từ 500 Hz. Tổng dung lượng 6,8 GB (đã đo).

### 1.6 `features/<caseid>.parquet`

- Mỗi dòng là một thời điểm dự đoán: `caseid` (int), `time` (float64, = `start + 30k` với k ≥ 1, luôn là số nguyên), và 517 cột float32.
- Mọi đặc trưng tại t chỉ dùng dữ liệu trong (t − W, t].
- **Chỉ dùng 342 cột** (93 cột mỗi cửa sổ, phụ lục A). 175 cột còn lại tính từ `nibp_mbp, cvp, bis, mac, ppf_ce, rftn_ce, peep` và **không bao giờ** được đưa vào mô hình.
- Tổng 1.240.755 dòng, train 875.552 dòng (đã đo). Đọc bằng pyarrow và chỉ lấy các cột cần (`columns=`).

### 1.7 `normalization.json`

`{"numeric": {...}, "wave100": {"mean": ..., "std": ..., "n": ...}}`, đã tính **chỉ trên train**. Chỉ dùng phần `wave100` để z-score sóng cho DL. Nếu không có, tự tính trên một mẫu ngẫu nhiên từ ca train, bỏ các điểm bị mask.

---

## 2. Nguyên tắc bắt buộc

1. **Không bao giờ đọc tập test** trước khi có `artifacts/lock.json` (mục 9). Hàm nạp dữ liệu phải báo lỗi nếu gặp `split == "test"` mà không có cờ `--final-test` và file khóa hợp lệ.
   - **Được phép** trước khi khóa: kiểm tra chất lượng input của ca test ở mức `qc.csv` (có file không, có sóng không, `map_coverage`), vì các thông tin này không chứa kết quả.
   - **Không được phép** trước khi khóa: đọc `label_map`, đếm đợt tụt, tính tỷ lệ dương hay bất kỳ thống kê nào về nhãn của test, **kể cả chỉ để lấy số tổng**.
2. **Không chia lại tập.** Chỉ dùng cột `split` trong `qc.csv`. Kiểm tra không bệnh nhân nào nằm ở 2 tập.
3. **Input chỉ dùng dữ liệu tại hoặc trước t.** `label_map` và `label_flags` chỉ dùng để gán nhãn.
4. **Mọi mô hình dùng chung cột `eligible`** (có dự đoán hay không). Chất lượng sóng là input, không phải lý do để một mô hình bỏ dự đoán. Nhờ vậy coverage giống nhau và so sánh công bằng.
5. **Hai phiên bản LightGBM chạy trên đúng cùng các dòng.** Chỉ khác tập cột.
6. **Thống kê dùng để điền thiếu và chuẩn hóa chỉ tính trên train**, rồi áp dụng y hệt cho các tập khác.
7. **Mọi mô hình được calibrate trên tập calibration.** Kiểm tra trung bình xác suất sau calibrate xấp xỉ tỷ lệ dương.
8. **Chọn ngưỡng và cấu hình trên validation. Báo cáo cuối trên test, chạy một lần.**
9. **Không augmentation, không oversample, không trọng số lớp** ở bản chính.
10. **Không chuẩn hóa riêng từng đoạn sóng.** Chỉ z-score theo thống kê train, để giữ mức huyết áp tuyệt đối.
11. **Mọi output ghi kèm** config digest (sha256 của phần config liên quan), git commit, phiên bản thư viện và digest của input. Bước sau kiểm tra digest của bước trước.
12. **Script chạy lại được giữa chừng:** bỏ qua phần đã xong, ghi file tạm rồi đổi tên.
13. **Dữ liệu test không rời máy cá nhân.** Không đẩy lên Hugging Face, Kaggle hay GitHub bất kỳ file nào chứa dòng, nhãn hay đợt tụt của test, kể cả trong repo private. `hub.py` phải kiểm tra trước khi đẩy (mục 15.3).
14. **Token không bao giờ nằm trong code hay notebook.** Trên máy dùng `hf auth login` và `~/.kaggle/kaggle.json`. Trên Kaggle dùng Kaggle Secrets.

---

## 3. Xử lý lại dữ liệu (NB01, trên máy cá nhân)

Không cần tiền xử lý lại tín hiệu. Bước này chỉ gồm: kiểm tra dữ liệu, chọn cột, tính lại PPV, gán nhãn và gom thành bảng theo tập.

### 3.1 Kiểm tra dữ liệu (`scripts/check_data.py`)

- Đếm số file trong `cases/`, `features/`, `wave100/`, rồi so với `qc.csv`. Ghi danh sách ca thiếu cho từng thư mục.
- So số ca thực tế với số mong đợi. Bảng này đã được đối chiếu với dữ liệu thật (sau khi bỏ 4 ca lỗi ở mục 1.1). Nếu lệch bảng, dừng và báo nhóm.

| Tập | Ca (cohort gốc) | Ca đã xử lý | Bệnh nhân đã xử lý | Ca có sóng | **Cohort D** |
|---|--:|--:|--:|--:|--:|
| train | 2.559 | 2.557 | 2.458 | 2.503 | 2.393 |
| calibration | 280 | 280 | 266 | 273 | 258 |
| validation | 271 | 271 | 266 | 268 | 248 |
| test | 516 | 514 | 500 | 500 | 464 |
| **Tổng** | **3.626** | **3.622** | **3.490** | **3.544** | **3.363** (3.242 bệnh nhân) |

- Kiểm tra không `subjectid` nào xuất hiện ở 2 tập.
- Kiểm tra trên **mọi ca** (không chỉ vài ca mẫu):
  - `len(label_map) == len(wave_mask) == surgery_seconds`;
  - `len(wave100)` ≈ `surgery_seconds × 100`, kiểu float16;
  - `features.time` đúng bằng `start + 30k`;
  - có đủ 342 cột ở phụ lục A.
- Các kiểm tra này chỉ đọc input, **không** đọc `label_map` để đếm đợt tụt ở test (nguyên tắc 1). Với ca test, lấy độ dài `label_map` từ header của mảng trong file npz (`np.lib.format.read_array_header_*`), không nạp giá trị.
- Kiểm tra thêm:
  - 4 ca thiếu đúng bằng `input.known_failed_cases`, không hơn, không kém;
  - không có dòng trùng `(caseid, time)` trong `features`;
  - `split` chỉ có 4 giá trị hợp lệ.
- Ghi sha256 của `qc.csv` vào `data_check.json`. Các bước sau kiểm tra lại giá trị này.
- Ghi `reports/data_check.json`.

### 3.2 Chọn cột

- Chỉ đọc `caseid`, `time` và 342 cột ở phụ lục A.
- Nếu tính lại PPV (mục 3.3), thay 4 cột `bt_ppv_w{W}` bằng `bt_ppv30_w{W}`. Số cột không đổi.
- Test: không cột nào trong danh sách đặc trưng mô hình bắt đầu bằng `nibp, cvp, bis, mac, ppf, rftn, peep`.

### 3.3 Tính lại PPV (mặc định bật: `features.recompute_ppv = true`)

**Lý do:** `bt_ppv_w{W}` trong notebook là max − min của hiệu áp trên cả cửa sổ W. Với W = 120 giây, giá trị này lẫn cả xu hướng huyết áp, không chỉ dao động theo nhịp thở.

**Cách tính mới:**

```python
def ppv_subwindow(beats, t_end, sub=30):
    b = beats[(beats.avail > t_end - sub) & (beats.avail <= t_end)]
    if len(b) < 5 or (b.reject != 0).any():
        return nan
    if b.period.std(ddof=0) / b.period.mean() > 0.10:     # nhịp không đều
        return nan
    return (b.pp.max() - b.pp.min()) / ((b.pp.max() + b.pp.min()) / 2) * 100

ventilated(t) = 6 <= rr_current(t) <= 40 and etco2_current(t) >= 10   # lấy từ features parquet
bt_ppv30_w{W}(t) = nanmedian([ppv_subwindow(beats, t - 30*j) for j in range(W // 30)]) if ventilated(t) else nan
```

Ghi vào `samples/ppv30/<caseid>.parquet` (`caseid, time, bt_ppv30_w30 … bt_ppv30_w120`), rồi join vào bảng đặc trưng theo `(caseid, time)`.

**Cách tính nhanh:** vì các thời điểm dự đoán cách nhau đúng 30 giây, đoạn 30 giây kết thúc tại t − 30j chính là đoạn của thời điểm dự đoán t − 30j. Vì vậy:
1. Mỗi ca, tính `ppv30(t)` **một lần** cho mọi thời điểm dự đoán, dùng `np.searchsorted` trên `avail` và cumsum để đếm nhịp bị loại, không lặp bằng pandas.
2. Với các đoạn nằm trước thời điểm dự đoán đầu tiên (t − 30j < start + 30), tính thêm cho các mốc `start + 30 − 30j`.
3. `bt_ppv30_w{W}(t)` = `nanmedian` của `W/30` giá trị `ppv30` gần nhất, rồi đặt NaN nếu không thở máy tại t.

Cách này nhanh hơn khoảng 10 lần so với tính lại từng đoạn cho từng W.

### 3.4 Phát hiện đợt tụt (`events.py`)

```python
grid1 = start + arange(len(label_map))                 # giây
below = isfinite(label_map) & (label_map < 65)
events = []
for i0, i1 in runs(below):                             # đoạn liên tiếp True, i1 không tính
    if i1 - i0 < 60:
        continue
    ev = Event(onset=grid1[i0], end=grid1[i1 - 1] + 1, min_map=label_map[i0:i1].min())
    if events and ev.onset - events[-1].end < 120 \
            and isfinite(label_map[int(events[-1].end - start):i0]).all():
        events[-1].end = ev.end                         # gộp với đợt trước
        events[-1].min_map = min(events[-1].min_map, ev.min_map)
    else:
        events.append(ev)
```

- Mỗi đợt ghi thêm `duration = end − onset` và `suspect_artefact = min_map ≤ 30`.
- Đợt nghi nhiễu vẫn được giữ, nhưng phải báo cáo số lượng. Nếu tỷ lệ > 2%, báo nhóm để xem lại.

### 3.5 Gán nhãn và `eligible` (`labels.py`)

**Thời điểm dự đoán:** đúng các giá trị `time` trong `features/<caseid>.parquet`.

**`eligible`** (nhân quả, dùng chung cho mọi mô hình). Ghi lý do đầu tiên gặp phải vào `abstain_reason`:

| Lý do | Điều kiện |
|---|---|
| `map_unknown` | `map_current` là NaN |
| `in_hypotension` | `map_current < 65` |
| `in_event` | Có đợt tụt với `onset ≤ t < end` |
| `post_event` | Có đợt tụt với `end ≤ t < end + 120` |
| `map_sparse` | `map_missing_w120 > 0.5` |
| `recent_low` | Chỉ khi `exclusion_reset = "any_low"`: có ô `values[:, map] < 65` trong (t − 120, t] |

- Mặc định `exclusion_reset = "event"` (theo proposal), nên không dùng `recent_low`.
- Lưu ý: `in_event` và `post_event` dùng các đợt tụt phát hiện từ `label_map`. `label_map` được làm sạch với cờ nhìn về sau tối đa khoảng 60 giây, nên hai luật này có thể dùng một chút thông tin sau t. Ảnh hưởng nhỏ, chấp nhận ở bản chính, và ghi vào `DEVIATIONS.md`.

**Nhãn cho mỗi mốc h ∈ {300, 600, 900, 1200, 1800}** (giá trị 1, 0 hoặc −1 là không xác định):

```python
end = start + len(label_map)
j0, j1 = int(t - start) + 1, int(t - start) + h + 61     # các giây trong (t, t + h + 60]
future = label_map[j0:j1]
if any(t < ev.onset <= t + h for ev in events):
    y = 1                                                  # lenient: không cần dữ liệu sau đợt tụt
elif t + h + 60 > end:
    y = -1
else:
    unknown = ~isfinite(future)
    y = -1 if unknown.mean() > 0.05 or longest_true_run(unknown) >= 30 else 0
```

- **Vì sao +60 giây:** một đợt tụt bắt đầu ngay tại t + h cần 60 giây dữ liệu sau đó mới xác nhận được. Mẫu âm vì vậy cần biết dữ liệu đến t + h + 60.
- **`label_policy = "strict"`:** mọi mẫu, kể cả mẫu dương, cần `future` đủ 100%, nếu không thì bằng −1. Chỉ dùng để báo cáo số mẫu, bản chính dùng `lenient`.

**Biến cố có thể cảnh báo trước:** `eligible_h(ev) = True` nếu có ít nhất một dòng `eligible` với `y_h = 1` và `ev.onset − h ≤ t < ev.onset`.

**Thời gian phơi nhiễm:** `exposure_seconds = min(30, end − t)`.

### 3.6 Cohort chính

**Vấn đề:** 183 ca gần như không có MAP dùng được (`map_coverage < 1%`), trong đó 180 ca có `has_wave == True`. Ở các ca này, MAP của Solar8000 luôn nằm ngoài [20, 200] và sóng ART phẳng khoảng 99% thời gian. Nhiều khả năng catheter động mạch không được nối vào máy đo. Cờ `has_wave` chỉ cho biết ca có track sóng, không cho biết sóng dùng được.

Các ca này không có dòng eligible và không có đợt tụt, nên **không làm lệch** sensitivity hay FA/giờ. Tuy vậy, chúng làm phồng số ca báo cáo, kéo `prediction_coverage` xuống, và làm nhiễu bootstrap theo bệnh nhân. Vì vậy cần loại khỏi cohort chính.

**Định nghĩa cohort chính** (tiêu chí chỉ dựa trên chất lượng input, không dựa trên nhãn):

| Bước | Điều kiện | Số ca (ước tính) |
|---|---|--:|
| A | Cohort gốc (`reports/E07`, đủ điều kiện) | 3.626 |
| B | Đã xử lý thành công (bỏ 4 ca lỗi 100 Hz) | 3.622 |
| C | `has_wave == True` | 3.544 |
| D | **Có arterial line hoạt động:** `map_coverage ≥ θ` | 3.363 (3.242 bệnh nhân) |

- **θ = 0,10 (đã chốt).** Trên train, không có ca nào có `map_coverage` trong khoảng (0,049; 0,219], nên mọi θ trong khoảng này cho cùng kết quả. θ được chọn chỉ từ train, trước khi có kết quả mô hình nào, và đã ghi vào config (`input.min_map_coverage`).
- `n_beats > 0` **không** dùng làm tiêu chí. Ca có MAP tốt nhưng sóng hỏng vẫn được giữ, vì trong thực tế mô hình sẽ gặp những ca như vậy. LightGBM và DL tự xử lý qua NaN và mask.
- Ghi flow A → D theo từng tập vào `reports/cohort_flow.csv`, kèm danh sách ca bị loại ở mỗi bước và lý do (`prep_error`, `no_wave`, `no_usable_arterial_line`).
- **Mọi so sánh mô hình dùng cohort D.**
- **Báo cáo riêng:**
  - 78 ca không có sóng (bước C): **chỉ báo cáo số ca** (train 54, calibration 7, validation 3, test 14). Không đánh giá hiệu năng riêng trên nhóm này: validation chỉ có 3 ca và test có 14 ca, quá ít để kết quả có ý nghĩa.
  - Các ca bị loại ở bước D: chỉ báo cáo số ca.

### 3.7 Output của NB01 (`<SAMPLES>/`)

| File | Nội dung |
|---|---|
| `labels/{split}.parquet` | **Mọi** thời điểm dự đoán (kể cả không eligible, cần cho phát lại cảnh báo): `caseid, subjectid, split, time, exposure_seconds, eligible, abstain_reason, has_wave, in_main_cohort, y_300, y_600, y_900, y_1200, y_1800`. **Toàn bộ bảng của test** (gồm cả `eligible` và `abstain_reason`, vì `in_event` và `post_event` suy ra từ đợt tụt) được ghi vào thư mục riêng `sealed/` (`sealed/labels_test.parquet`, `sealed/events_test.parquet`). Script chỉ ghi, không in hay tóm tắt gì từ các file này |
| `features/{split}.parquet` | `caseid, time` và 342 cột mô hình, cùng thứ tự dòng với `labels` |
| `events.parquet` | `caseid, subjectid, split, onset, end, duration, min_map, suspect_artefact, eligible_300 … eligible_1800` |
| `norm_tabular.json` | Trung vị, trung bình, độ lệch chuẩn của 342 cột, **tính trên train** (dòng eligible) |
| `label_report.csv` | Theo tập × mốc × chính sách nhãn (`lenient`, `strict`) × luật loại mẫu (`event`, `any_low`), **chỉ train, calibration, validation**. Phải có cả hai luật loại mẫu thì nhóm mới chốt được `exclusion_reset`. Các cột: số dòng, số eligible, dương, âm, không xác định, tỷ lệ dương, số đợt tụt, số đợt eligible. Số liệu nhãn của test chỉ được in ở NB04, sau khi khóa |
| `abstain_report.csv` | Số dòng theo `abstain_reason` và tập, **không có test** |
| `samples.json` | Config digest, digest của `qc.csv`, số liệu tổng (**không có số liệu nhãn của test**), danh sách ca bị bỏ và lý do |

`events.parquet` (bản không niêm phong) cũng chỉ chứa train, calibration và validation.

Thêm 2 file để NB03 trên Kaggle không cần đọc `cases/`:

| File | Nội dung |
|---|---|
| `case_index.parquet` | Một dòng mỗi ca **không thuộc test**: `caseid, subjectid, split, start, surgery_seconds, has_wave, in_main_cohort, baseline_source` |
| `wave_mask.npz` | `wave_mask` của các ca không thuộc test, khóa là `str(caseid)` (khoảng 30 MB) |

Bản của test (`case_index_test.parquet`, `wave_mask_test.npz`) ghi vào `sealed/`.

### 3.8 Số liệu phải in ra cuối NB01

- Flow cohort A → D theo từng tập (số ca và số bệnh nhân), cùng histogram `map_coverage` trên train và giá trị θ đã chốt.
- Phân bố `baseline_source`. Trung vị `label_known_frac` và `beat_valid_frac` trong cohort D.
- Với mỗi mốc h (chỉ train, calibration, validation): số dòng eligible, tỷ lệ dương, số đợt tụt, tỷ lệ nhãn −1, theo cả `lenient` và `strict`.
- Số và tỷ lệ đợt tụt `suspect_artefact` (chỉ train, calibration, validation).
- Tỷ lệ NaN của `bt_ppv_w{W}` (bản cũ) và `bt_ppv30_w{W}` (bản mới) trên train. Bản cũ khoảng 26% ở W = 60 (đo trên 200 ca train).
- Số cột đặc trưng: đúng 66 + 27 mỗi cửa sổ.

---

## 4. Config: `configs/uc04_v2.json`

Nạp vào các dataclass bất biến trong `src/uc04/config.py`. Mỗi phần có hàm `digest()`. Notebook chỉ ghi đè đường dẫn qua tham số dòng lệnh, không sửa giá trị khác. Muốn thử cấu hình khác thì tạo file `configs/uc04_v2_<tên>.json`.

```json
{
  "version": "uc04-v2",
  "seed": 20260917,
  "paths": {
    "prep": "D:/SafeAnes/SafeAnes/data/prep_v1",
    "samples": "data/samples_v2",
    "artifacts": "artifacts",
    "reports": "reports"
  },
  "input": {
    "required": ["qc.csv", "cases", "features"],
    "required_for_dl": ["wave100", "normalization.json"],
    "forbidden_feature_prefixes": ["nibp", "cvp", "bis", "mac", "ppf", "rftn", "peep"],
    "max_missing_case_fraction": 0.05,
    "known_failed_cases": [4458, 6119, 6133, 6375],
    "require_wave": true,
    "min_map_coverage": 0.10
  },
  "features": {
    "windows_seconds": [30, 60, 90, 120],
    "cadence_seconds": 30,
    "numeric_signals": ["map", "sbp", "dbp", "hr", "spo2", "etco2", "rr", "pp", "shock_index"],
    "numeric_stats": ["mean", "std", "min", "max", "slope", "missing"],
    "map_extra": ["map_drop_pct", "map_time_65_75_w{W}", "map_extrap_w{W}"],
    "beat_signals": ["dpdt_max", "sys_area", "ejection_time", "notch_rel", "decay_tau", "sv", "co", "svr"],
    "beat_stats": ["mean", "std", "slope"],
    "beat_extra": ["ppv", "valid_frac", "n_beats"],
    "recompute_ppv": true,
    "ppv_subwindow_seconds": 30,
    "ppv_min_beats": 5,
    "ppv_max_period_cv": 0.10,
    "ventilated_rr": [6, 40],
    "ventilated_min_etco2": 10
  },
  "labels": {
    "map_threshold": 65,
    "event_seconds": 60,
    "merge_gap_seconds": 120,
    "horizons_seconds": [300, 600, 900, 1200, 1800],
    "post_event_exclusion_seconds": 120,
    "exclusion_reset": "event",
    "max_map_missing_w120": 0.5,
    "label_policy": "lenient",
    "negative_max_unknown_fraction": 0.05,
    "negative_max_unknown_run_seconds": 30,
    "suspect_artefact_min_map": 30
  },
  "splits": {
    "fit": "train", "calibration": "calibration", "validation": "validation", "test": "test",
    "test_requires_lock": true
  },
  "hub": {
    "samples_repo": "<hf_user>/uc04-v2-samples",
    "samples_repo_type": "dataset",
    "runs_repo": "<hf_user>/uc04-v2-runs",
    "runs_repo_type": "model",
    "private": true,
    "push_min_interval_seconds": 600,
    "forbidden_path_parts": ["sealed", "test"],
    "token_env": "HF_TOKEN"
  },
  "kaggle": {
    "user": "<kaggle_user>",
    "prep_kernel_source": "dainn98s/safeanes",
    "wave_dataset": "<kaggle_user>/uc04-prep-v1",
    "wave_dataset_version": 1,
    "code_repo": "https://github.com/<org>/uc04.git",
    "tabular_machine": {"enable_gpu": false, "enable_internet": true},
    "dl_machine": {"enable_gpu": true, "machine_shape": "NvidiaTeslaT4", "enable_internet": true}
  },
  "baselines": {
    "map_threshold": {"score": "neg_map_current"},
    "map_logistic": {"features": ["map_current", "map_slope_w{W}", "map_std_w{W}"], "C": 1.0,
                     "max_iter": 2000, "impute": "train_median", "scale": true}
  },
  "lightgbm": {
    "versions": {"lgbm_numeric": ["numeric"], "lgbm_wave": ["numeric", "waveform"]},
    "params": {"n_estimators": 500, "num_leaves": 7, "learning_rate": 0.03,
               "min_child_samples": 150, "reg_lambda": 10, "n_jobs": 4, "verbosity": -1},
    "missing": "native_nan",
    "monotone_decreasing": ["map_current", "map_mean_w{W}", "map_min_w{W}"],
    "shap_rows": 50000,
    "confidence_seeds": [20260917, 20260918, 20260919, 20260920, 20260921],
    "confidence_params": {"subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8}
  },
  "dl": {
    "windows_seconds": [60, 30, 120, 90],
    "seeds": [20260917, 20260918, 20260919, 20260920, 20260921],
    "wave_hz": 100,
    "tab_features": "numeric_w{W}",
    "tab_extra_flags": ["baseline_missing"],
    "conv": {"channels": [32, 64, 64], "kernel": 7, "strides": [2, 2, 5]},
    "transformer": {"d_model": 64, "heads": 4, "layers": 2, "ff": 128, "dropout": 0.1, "positional": "sinusoidal"},
    "tab_mlp": [64, 64],
    "head_hidden": 64,
    "ordered_horizons": true,
    "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001},
    "batch_size": 256,
    "max_epochs": 20,
    "patience": 4,
    "train_samples_per_epoch": 300000,
    "amp": true,
    "num_workers": 4
  },
  "calibration": {
    "tabular": "platt_on_score",
    "dl": "shared_temperature_and_bias"
  },
  "evaluation": {
    "alarm_persistence": 2,
    "alarm_cooldown_seconds": 300,
    "fa_per_hour_budget": 1.0,
    "threshold_candidates": 200,
    "curve_points": 40,
    "ece_bins": 10,
    "early_lead_seconds": 300,
    "bootstrap_unit": "subjectid",
    "bootstrap_repeats_validation": 200,
    "bootstrap_repeats_test": 1000
  }
}
```

`{W}` được thay bằng độ dài cửa sổ lúc chạy. Đổi bất kỳ giá trị nào thì digest đổi, và các bước sau phải chạy lại.

Hai phần `hub` và `kaggle` không tính vào digest của các bước dữ liệu và mô hình, vì chúng chỉ quyết định nơi lưu, không ảnh hưởng kết quả. `paths` cũng vậy: trên Kaggle, script ghi đè `paths` qua tham số dòng lệnh.

---

## 5. Cấu trúc code và notebook

### 5.1 Repo mới

```
uc04/
  pyproject.toml                # numpy, pandas, pyarrow, scipy, scikit-learn, lightgbm, shap, torch, joblib, pytest
  configs/uc04_v2.json
  src/uc04/
    config.py        # load_config(path) -> dataclass bất biến; digest() từng phần
    io.py            # đọc qc.csv, npz, parquet (chỉ lấy cột cần), wave100 (memmap); ghi file tạm rồi đổi tên
    columns.py       # sinh danh sách 66 + 27 cột mỗi W (phụ lục A); nhóm cột cho ablation
    ppv.py           # mục 3.3
    events.py        # mục 3.4
    labels.py        # mục 3.5
    samples.py       # ghép tất cả thành output mục 3.7
    loaders.py       # load_split(split, columns, allow_test=False): chặn test, kiểm tra trùng bệnh nhân
    calibration.py   # Platt; nhiệt độ + bias chung (mục 8.1)
    alarms.py        # phát lại cảnh báo (mục 8.2)
    metrics.py       # chỉ số theo dòng và theo biến cố (mục 8.3)
    thresholds.py    # chọn ngưỡng theo ngân sách FA (mục 8.4)
    bootstrap.py     # bootstrap theo bệnh nhân, bootstrap ghép cặp (mục 8.5)
    tabular.py       # baseline, LightGBM, SHAP, ablation (mục 6)
    dl_data.py  dl_model.py  dl_train.py   # mục 7
    lock.py          # mục 9
    hub.py           # đẩy và kéo file với Hugging Face, chặn dữ liệu test (mục 15.3)
  scripts/
    check_data.py  build_samples.py  train_tabular.py  train_dl.py  lock_models.py  final_test.py
    push_samples.py          # máy cá nhân → HF samples_repo (bỏ sealed/)
    pull_samples.py          # HF samples_repo → Kaggle, theo đúng revision
    make_kaggle_notebooks.py # sinh notebook Kaggle và kernel-metadata.json, điền sẵn commit và revision
  notebooks/
    01_samples.ipynb             # chạy trên máy cá nhân
    04_lock_test.ipynb           # chạy trên máy cá nhân
    kaggle/
      <tên>/  <tên>.ipynb  kernel-metadata.json   # 11 notebook ở mục 15.4, sinh tự động, đẩy bằng kaggle kernels push
  templates/                     # mẫu notebook Kaggle dùng cho make_kaggle_notebooks.py
  tests/
  reports/  artifacts/  data/    # data/ và artifacts/ không commit; .gitignore có thêm sealed/, *.pt, *.joblib
```

### 5.2 Notebook: 4 notebook chính, 2 notebook sau

Notebook chỉ làm 3 việc: cài thư viện, gọi script, in bảng tóm tắt. Logic nằm trong `src/` và `scripts/`.

| # | Notebook | Chạy ở | Input | Output | Thời gian ước tính |
|---|---|---|---|---|---|
| NB01 | `01_samples` | **Máy cá nhân** (CPU) | `prep_v1` trên máy | `samples_v2/` trên máy, rồi đẩy phần không phải test lên HF `samples_repo` | 1–2 giờ, tùy số nhân CPU |
| NB02 | `00_env_check`, `02a`–`02d` | **Kaggle** (CPU, bật Internet) | HF `samples_repo` | HF `runs_repo/tabular/...`, `reports/tabular_validation.csv`, SHAP, ablation | 1–3 giờ |
| NB03 | `03_dl_smoke`, `03_dl_W60`/`W30`/`W120`/`W90` (GPU T4), `03_dl_finalize` (CPU) | **Kaggle** (bật Internet) | HF `samples_repo` (có `wave_mask.npz`), `wave100/` từ Kaggle Dataset `kaggle.wave_dataset` | HF `runs_repo/dl/...`, `reports/dl_validation.csv` | 4–10 giờ GPU |
| NB04 | `04_lock_test` | **Máy cá nhân** (CPU) | HF `runs_repo` tại một revision cố định, `samples_v2/sealed/` và `prep_v1/wave100` trên máy | `artifacts/lock.json`, `reports/test_results.csv` | Tabular < 1 giờ. DL trên CPU chưa đo, có thể 1–2 giờ |
| NB05 | `05_mover` (sau) | Máy cá nhân | Dữ liệu MOVER (cần DUA), `lock.json` | Kiểm tra ngoài. **Không đẩy dữ liệu MOVER lên HF** trừ khi DUA cho phép | chưa ước tính |
| NB06 | `06_cause` (sau) | CPU | `prep_v1/cases` (MAC, Ce, BIS), SHAP, `events.parquet`, **và `ev1000/`** (ngoại lệ, xem bên dưới) | Bảng cơ chế | chưa ước tính |

- Thời gian là ước tính, chưa đo. Với NB03, hãy đo 1 epoch trước rồi mới quyết định số seed, vì tốc độ phụ thuộc nhiều vào việc đọc ngẫu nhiên từ `wave100`.
- Cách tạo, đẩy và chạy notebook Kaggle, cùng cách lưu kết quả lên HF: xem mục 15.
- **Ngoại lệ cho NB06: dữ liệu EV1000.** `prep_v1` không có track EV1000, nhưng NB06 cần chúng để kiểm tra SV/CO/SVR proxy trong 15 phút trước đợt tụt.
  - Viết riêng `scripts/fetch_ev1000.py` để lấy `EV1000/SV`, `EV1000/CO`, `EV1000/SVR`, `EV1000/SVV` cho các ca có EV1000 trong cohort, **không tính test**: 514 ca (SVR chỉ có ở 207 ca). 1.739 trên 1.749 track đã có trong `vitaldb_full/raw` trên máy. 10 track còn lại lấy từ API VitalDB.
  - Ghi ra `ev1000/<caseid>.parquet` (`time, sv, co, svr, svv`), dùng cùng đồng hồ giây với `start` trong npz.
  - Việc này không ảnh hưởng NB01 đến NB04, nên làm sau.
- **NB05 (MOVER)** vẫn chờ DUA.

Khung notebook Kaggle ở mục 15.4.

---

## 6. Baseline và LightGBM (NB02, trên Kaggle)

### 6.1 Dữ liệu cho mỗi tổ hợp (mô hình, W, h)

- **Dòng để fit và calibrate:** `eligible == True`, `y_h ∈ {0, 1}`, thuộc cohort chính.
- **Dòng để đánh giá:** mọi dòng của các ca trong tập (kể cả dòng không eligible, với `probability = NaN`), vì phát lại cảnh báo cần chuỗi thời gian đầy đủ.
- **Tập:** fit trên `train`, calibrate trên `calibration`, chọn ngưỡng và báo cáo trên `validation`.
- **Cột:**
  - `map_threshold`: `map_current`.
  - `map_logistic`: `map_current`, `map_slope_w{W}`, `map_std_w{W}`.
  - `lgbm_numeric`: 66 cột chỉ số của cửa sổ W.
  - `lgbm_wave`: 93 cột (66 chỉ số và 27 sóng) của cửa sổ W.

### 6.2 Cách train

| Mô hình | Cách fit | Điểm thô `s` |
|---|---|---|
| `map_threshold` | Không fit | `−map_current` |
| `map_logistic` | Điền thiếu bằng trung vị train → StandardScaler → LogisticRegression(C=1) | log-odds |
| `lgbm_*` | `LGBMClassifier(**params, monotone_constraints=m)`, giữ NaN nguyên bản | log-odds |

- `m` gán −1 cho `map_current`, `map_mean_w{W}`, `map_min_w{W}`, và 0 cho các cột còn lại. Ý nghĩa: MAP cao hơn thì nguy cơ không tăng.
- Log-odds = `log(p / (1 − p))`, với p kẹp trong [1e-6, 1 − 1e-6].
- **Mỗi mốc h một mô hình riêng.** Tổng **65 tổ hợp**: `map_logistic`, `lgbm_numeric`, `lgbm_wave` mỗi mô hình 4 W × 5 h = 60, cộng `map_threshold` 5 tổ hợp. `map_threshold` chỉ dùng `map_current`, không phụ thuộc W, nên chỉ chạy một lần cho mỗi h. Mô hình này không cần fit, nhưng vẫn được calibrate và chọn ngưỡng. Khi gộp bảng, `02d` lặp lại dòng của `map_threshold` cho mỗi W (cột `window` ghi `W` tương ứng) để dễ so sánh.
- Sau khi fit: calibrate (mục 8.1), chọn ngưỡng (mục 8.4), đánh giá (mục 8.3).

### 6.3 Output mỗi tổ hợp: `artifacts/tabular/{model}/W{W}/h{h}/`

Với `map_threshold`, thư mục là `artifacts/tabular/map_threshold/h{h}/` (không có W).

- `model.joblib`: mô hình, calibrator, danh sách cột, trung vị train, config digest.
- `threshold.json`: ngưỡng, và đường cong ngưỡng trên validation.
- `val_predictions.parquet`: `caseid, subjectid, time, eligible, exposure_seconds, y_h, probability`.
- `case_metrics.parquet`: bảng theo từng ca trên validation (mục 8.3). `02d` dùng file này cho bootstrap ghép cặp, không phải phát lại cảnh báo.
- `provenance.json`, `done.json`.

Mỗi tổ hợp còn ghi 2 file kết quả của riêng nó, mỗi file một dòng:
- `reports/tabular_validation/{model}_W{W}_h{h}.csv` (schema ở mục 8.6);
- `reports/calibration_check/{model}_W{W}_h{h}.csv` (mục 8.1).

Với `map_threshold`, tên file là `map_threshold_h{h}.csv`. Bảng tổng `reports/tabular_validation.csv` và `reports/calibration_check_tabular.csv` chỉ do `02d_tabular_analysis` gộp lại.

**Không notebook nào ghi chung một file**, vì các notebook 02a, 02b, 02c chạy song song và cùng đẩy lên một repo HF. Nếu cùng ghi một file, lần đẩy sau sẽ ghi đè kết quả của notebook kia.

**Lưu lên HF:** thư mục `artifacts/tabular/` và `reports/` được đồng bộ lên `runs_repo`, cùng đường dẫn (mục 15.3). Mỗi tổ hợp xong thì ghi `done.json`. Khi chạy lại (ví dụ phiên Kaggle bị dừng), script kéo danh sách `done.json` từ HF và bỏ qua các tổ hợp đã xong.

### 6.4 Sau khi xong 65 tổ hợp (notebook `02d_tabular_analysis`)

- Kéo mọi file trong `reports/tabular_validation/` và `reports/calibration_check/` từ HF, gộp thành bảng tổng. Nếu thiếu file của tổ hợp nào, in danh sách và dừng.
- Mỗi phần phân tích dưới đây ghi file riêng: `reports/shap_W{W}_h{h}.csv`, `reports/ablation.csv`, `reports/tabular_confidence/`.

- **Chọn W\*:** cửa sổ của `lgbm_wave` có event sensitivity trung bình cao nhất ở mốc 5 và 10 phút trên validation.
- **SHAP:** TreeSHAP cho `lgbm_wave` trên 50.000 dòng validation, cho mọi W và h. Ghi tỷ lệ |SHAP| theo nhóm cột.
- **Ablation** (proposal mục 5.3), chỉ ở W\* và mốc 5, 10 phút. Bỏ lần lượt từng nhóm: MAP, SBP/DBP, HR, SpO2, EtCO2/RR, hình dạng sóng, PPV, SV/CO/SVR proxy (nhóm cột ở phụ lục A).
- **Mức tin cậy** (chỉ cho cấu hình được chọn): train thêm 5 seed với `confidence_params`. Tính độ lệch chuẩn xác suất giữa các seed, rồi chia 3 mức (cao, trung bình, thấp) theo tam phân vị trên validation.

---

## 7. Conv1D + Transformer (NB03, trên Kaggle GPU)

### 7.1 Input cho một mẫu (caseid, t, W)

- **Sóng:** đọc `wave100/<caseid>.npy` bằng `np.load(mmap_mode="r")`. Lấy mẫu k thỏa `t − W < start + k/100 ≤ t`, được `W × 100` điểm.
  - Kênh 1 = `(x − mean) / std` theo `normalization.json["wave100"]`. NaN đổi thành 0.
  - Kênh 2 = mask: 1 tại điểm NaN, hoặc thuộc khối 1 giây có `wave_mask != 0` (mỗi khối lặp thành 100 điểm).
  - Tensor `[2, W × 100]`, float32.
  - Đọc `wave_mask` từ `samples_v2/wave_mask.npz` (tạo ở NB01, lấy từ HF) một lần vào RAM, không đọc `cases/`. `start` của mỗi ca lấy từ `case_index.parquet`.
  - Giữ memmap `wave100` đã mở trong một cache LRU (ví dụ 256 file mỗi worker), để không mở lại file cho mỗi mẫu và không vượt giới hạn số file mở.
- **Bảng chỉ số:** 66 cột chỉ số của cửa sổ W. Điền thiếu bằng trung vị train, rồi z-score theo `norm_tabular.json`.
  - Thêm 1 cờ `baseline_missing` (bằng 1 khi `baseline_source == 0`, có ở 77 ca). Lý do: ở các ca này `map_drop_pct` luôn NaN, và nếu chỉ điền trung vị thì mô hình không phân biệt được "MAP giảm trung bình" với "không có MAP nền". Input bảng của DL vì vậy có 67 cột.
  - LightGBM không cần cờ này, vì LightGBM tự xử lý NaN.
- **Nhãn:** `[y_300, y_600, y_900, y_1200, y_1800]`, giá trị −1 được bỏ qua trong loss.
- **Dòng để train:** `eligible` và có ít nhất một nhãn khác −1.

### 7.2 Kiến trúc (`dl_model.py`)

```
wave [B, 2, W*100]
 → Conv1D(2→32, k7, stride 2) → BatchNorm → GELU
 → Conv1D(32→64, k7, stride 2) → BatchNorm → GELU
 → Conv1D(64→64, k7, stride 5) → BatchNorm → GELU        # tổng stride 20: 5 token/giây
 → + positional encoding (sinusoidal)                      # W=30: 150 token, W=120: 600 token
 → TransformerEncoder(d=64, heads=4, layers=2, ff=128, dropout=0.1, norm_first=True)
 → mean pooling theo thời gian → z_wave [B, 64]
tab [B, 67] → Linear(67→64) → GELU → Linear(64→64) → z_tab [B, 64]
concat [B, 128] → Linear(128→64) → GELU → Linear(64→5) = a
logit_1 = a_1 ;  logit_k = logit_{k-1} + softplus(a_k)      # xác suất tăng dần theo mốc
```

### 7.3 Cách train

- **Loss:** BCE với logit, chỉ tính trên các ô nhãn khác −1, lấy trung bình. Không dùng `pos_weight`, vì xác suất sẽ được calibrate sau.
- **Tối ưu:** AdamW (lr 1e-3, weight decay 1e-4), batch 256, AMP fp16 trên GPU.
- **Mỗi epoch** rút ngẫu nhiên 300.000 mẫu train, theo seed. Dừng sớm theo loss trên validation, `patience` = 4, tối đa 20 epoch.
- **Số lần train:** 4 cửa sổ × 5 seed = 20 lần. Thứ tự: W = 60 đủ 5 seed trước, rồi 30, 120, 90.
- Mỗi lần train lưu `best.pt`, `last.pt` và `train_log.csv` (loss train và validation theo epoch), có `--resume`.
- **Lưu lên HF và chạy tiếp giữa các phiên Kaggle:**
  - Cuối mỗi epoch, đẩy `artifacts/dl/W{W}/seed{s}/` lên `runs_repo`, nhưng không dày hơn 10 phút một lần (`push_min_interval_seconds`). Khi xong một lần train thì luôn đẩy, và ghi `done.json`.
  - Khi bắt đầu, script kiểm tra HF: có `done.json` thì bỏ qua; có `last.pt` mà chưa có `done.json` thì tải về và chạy tiếp từ epoch đó.
  - Nhờ vậy, nếu phiên Kaggle (khoảng 12 giờ) bị dừng, chỉ mất tối đa 10 phút train.
- **Tính tất định:** đặt seed cho torch, numpy, random và từng DataLoader worker. Đặt `CUBLAS_WORKSPACE_CONFIG=":4096:8"` và bật `torch.use_deterministic_algorithms(True)`. Nếu `03_dl_smoke` báo lỗi vì một phép tính trên CUDA không có bản tất định, chuyển sang `warn_only=True` và ghi vào `DEVIATIONS.md` (mục 14, dòng 13).

### 7.4 Sau khi train

- **Ensemble 5 seed:** lấy trung bình logit, rồi calibrate bằng một nhiệt độ và một bias chung cho cả 5 mốc (mục 8.1), để giữ thứ tự tăng dần.
- **Mức tin cậy:** độ lệch chuẩn xác suất giữa các seed, chia 3 mức theo tam phân vị trên validation.
- **Chọn ngưỡng** cho từng mốc trên validation (mục 8.4).
- **Output:**
  - `artifacts/dl/W{W}/seed{s}/best.pt`
  - `artifacts/dl/W{W}/seed{s}/logits_{calibration,validation}.parquet`: mỗi lần train tự dự đoán trên calibration và validation bằng `best.pt` rồi ghi file này. Nhờ vậy `03_dl_finalize` chạy được trên CPU, không cần GPU.
  - `artifacts/dl/W{W}/logits_{calibration,validation}.parquet`: logit trung bình của các seed, do `03_dl_finalize` ghi
  - `artifacts/dl/W{W}/calibration.json`, `threshold.json`, `case_metrics_h{h}.parquet`
  - `reports/dl_validation.csv` và `reports/calibration_check_dl.csv`. Cả hai chỉ do `03_dl_finalize` ghi, vì đây là notebook duy nhất tính trên mọi W.
  - Tất cả được đồng bộ lên `runs_repo`, cùng đường dẫn.

---

## 8. Hiệu chỉnh, cảnh báo, chỉ số

### 8.1 Calibration (`calibration.py`)

- **Tabular (Platt):** trên các dòng calibration (eligible, `y_h` khác −1), chuẩn hóa điểm thô `s` rồi fit `LogisticRegression(C=1e6)` với input là `s`. Xác suất cuối = `sigmoid(a · s_chuẩn_hóa + b)`.
- **DL (nhiệt độ và bias chung):** tìm T > 0 và b để giảm BCE trên mọi ô nhãn đã biết của 5 mốc, với `p = sigmoid(logit / T + b)`. Dùng `scipy.optimize.minimize` (L-BFGS-B), tham số `log T` trong [−4, 4] và b trong [−20, 20].
- **Kiểm tra:** ghi `mean(p)` và tỷ lệ dương trên validation.
  - Tabular: mỗi tổ hợp ghi file riêng `reports/calibration_check/{model}_W{W}_h{h}.csv`. `02d` gộp thành `reports/calibration_check_tabular.csv`.
  - DL: `03_dl_finalize` ghi `reports/calibration_check_dl.csv`.
  - Không dùng một file `calibration_check.csv` chung, vì các notebook tabular chạy song song.

### 8.2 Phát lại cảnh báo (`alarms.py`)

Chạy trên **mọi dòng** của một ca, sắp theo `time`. Pseudo-code dưới đây là định nghĩa. Bản cài đặt nên chạy trên mảng numpy của từng ca (không dùng `itertuples`), hoặc dùng numba, vì mỗi tổ hợp cần khoảng 100 lần phát lại trên khoảng 90 nghìn dòng validation. Test phải khẳng định bản nhanh cho cùng kết quả với bản pseudo-code.

```python
alarms, streak, active = [], 0, False
next_allowed, prev_t = -inf, -inf
for row in case.sort_values("time").itertuples():
    if row.time - prev_t > 30:            # có khoảng trống trong lưới 30 giây
        streak, active = 0, False
    prev_t = row.time
    if not row.eligible or isnan(row.probability) or row.probability < thr:
        streak, active = 0, False
        continue
    streak += 1
    if not active and streak >= 2 and row.time >= next_allowed:
        alarms.append(row.time)
        active = True                     # phải xuống dưới ngưỡng rồi mới cảnh báo lại được
        next_allowed = row.time + 300
```

**Phân loại từng cảnh báo tại thời điểm a, với mốc h:**
- Nếu `y_h(a) == −1`: **censored**, không tính đúng hay sai.
- Nếu có đợt tụt với `a < onset ≤ a + h`: **đúng**. Ghép với đợt tụt đầu tiên như vậy, `lead = onset − a`.
- Còn lại: **sai**.

### 8.3 Chỉ số (`metrics.py`)

**Theo dòng** (các dòng eligible, `y_h` khác −1, có xác suất):
- AUROC, AUPRC (average precision), Brier;
- ECE với 10 khoảng đều nhau trên [0, 1];
- `prevalence` (tỷ lệ dương).

**Theo biến cố và cảnh báo:**

| Chỉ số | Cách tính |
|---|---|
| `event_sensitivity` | Số đợt tụt `eligible_h` được ghép ít nhất một cảnh báo đúng / số đợt tụt `eligible_h` |
| `event_sensitivity_all` | Như trên, nhưng mẫu số là mọi đợt tụt |
| `early_sensitivity` | Số đợt tụt `eligible_h` có lead ≥ 300 giây / số đợt tụt `eligible_h` |
| `lead_median_s`, `lead_q25_s`, `lead_q75_s` | Với mỗi đợt tụt được phát hiện, lấy lead lớn nhất trong các cảnh báo đúng ghép với nó |
| `false_alarms_per_hour` | Số cảnh báo sai / (tổng `exposure_seconds` của các dòng eligible, `y_h` khác −1, có xác suất / 3600) |
| `alarm_ppv` | Đúng / (đúng + sai) |
| `prediction_coverage` | Tổng `exposure_seconds` của dòng eligible / tổng `exposure_seconds` mọi dòng |

Ghi thêm bảng theo từng ca (`case_metrics`): `caseid, subjectid, events_all, events_eligible, events_detected, events_early, true_alarms, false_alarms, censored_alarms, evaluable_seconds, eligible_seconds, scheduled_seconds`. Bootstrap dùng bảng này.

### 8.4 Chọn ngưỡng (`thresholds.py`)

- **Mục tiêu:** event sensitivity cao nhất với FA/giờ ≤ 1,0 trên validation.
- **Ứng viên:** 200 phân vị của xác suất (dòng eligible trên validation), cộng lưới 0,01 đến 0,99.
- FA/giờ giảm gần như đơn điệu khi ngưỡng tăng. Vì vậy:
  1. chia đôi trên danh sách ứng viên đã sắp xếp để tìm ngưỡng nhỏ nhất có FA/giờ ≤ 1;
  2. đánh giá thêm 20 ứng viên hai bên ngưỡng đó;
  3. chọn ngưỡng có event sensitivity cao nhất trong những ngưỡng đạt FA ≤ 1. Nếu bằng nhau, chọn ngưỡng có alarm PPV cao hơn.
- Nếu không có ngưỡng nào đạt, lấy ngưỡng có FA/giờ thấp nhất và ghi `budget_not_met = true`.
- Ghi thêm đường cong 40 điểm (sensitivity theo FA/giờ) để vẽ hình.

### 8.5 Khoảng tin cậy (`bootstrap.py`)

- **Bootstrap theo bệnh nhân:** rút có hoàn lại `subjectid`, giữ nguyên số bệnh nhân. Phát lại cảnh báo **chỉ một lần** trên dữ liệu gốc, rồi cộng lại bảng `case_metrics` theo mẫu rút. Chỉ số theo dòng được tính lại trên các dòng của mẫu rút. Lấy phân vị 2,5% và 97,5%.
- **Bootstrap ghép cặp** (so sánh 2 mô hình, ví dụ `lgbm_wave` với `lgbm_numeric`): dùng cùng một mẫu rút cho cả 2 mô hình, lấy chênh lệch của từng chỉ số, rồi báo cáo CI của chênh lệch.
- Số lần rút: 200 trên validation, 1000 trên test.

### 8.6 Schema bảng kết quả (`*_validation.csv`, `test_results.csv`)

`model, window, horizon, n_rows, n_cases, prevalence, auroc, auprc, brier, ece, threshold, budget_not_met, event_sensitivity, events_detected, events_eligible, event_sensitivity_all, early_sensitivity, false_alarms_per_hour, alarm_ppv, lead_median_s, lead_q25_s, lead_q75_s, prediction_coverage, config_digest, git_commit`

Mỗi chỉ số chính (`auroc`, `auprc`, `event_sensitivity`, `false_alarms_per_hour`) có thêm 2 cột `_lo`, `_hi` cho CI 95%. Luôn báo cáo `prevalence` bên cạnh `auprc`.

---

## 9. Khóa mô hình và chạy test (NB04, trên máy cá nhân)

1. Trên validation, nhóm chốt: danh sách tổ hợp sẽ báo cáo (mặc định: mọi tổ hợp đã chạy), và cấu hình chính để nêu trong kết luận (mặc định: W\*).
2. `lock_models.py` ghi `artifacts/lock.json`:
   - `runs_repo` và **commit sha của HF** tại thời điểm khóa;
   - đường dẫn và sha256 của mọi file mô hình, calibrator và ngưỡng;
   - config digest, git commit của code, ngày khóa.
   `lock.json` được đẩy lên `runs_repo` (file này không chứa dữ liệu test).
3. `final_test.py --final-test` (chạy trên máy cá nhân, vì nhãn test chỉ có ở `samples_v2/sealed/`):
   - tải mô hình từ HF đúng commit sha trong lock (`snapshot_download(revision=...)`);
   - kiểm tra sha256 khớp với lock, sai thì dừng;
   - dự đoán trên test, tính chỉ số và bootstrap 1000 lần;
   - ghi `reports/test_results.csv`;
   - **nếu file kết quả đã tồn tại thì dừng**, không ghi đè.
4. Sau khi xem kết quả test, không chỉnh mô hình nữa. Mọi phân tích thêm được ghi là post-hoc.
5. Sau khi có `test_results.csv`, có thể đẩy **bảng kết quả tổng** (không có dòng hay nhãn của test) lên `runs_repo/reports/`.

---

## 10. Test bắt buộc (`pytest tests/` trước mỗi lần push)

Dùng dữ liệu giả nhỏ, tạo trong test, không cần `prep_v1`.

| Test | Khẳng định |
|---|---|
| Đợt tụt | `label_map` < 65 trong 59 giây: không có đợt tụt. Trong 60 giây: 1 đợt. Hai đợt cách 100 giây (khoảng giữa đủ dữ liệu): gộp thành 1. Cách 100 giây nhưng khoảng giữa có NaN: không gộp |
| Nhãn dương | Đợt tụt bắt đầu ở t + 180 giây: `y_300 = 1` và mọi mốc lớn hơn đều bằng 1 |
| Nhãn âm | Không có tụt, dữ liệu đủ: `y_h = 0`. Có 60 giây NaN trong (t, t + h + 60]: `y_h = −1` |
| lenient và strict | Đợt tụt từ t + 60 đến t + 150 giây, NaN từ t + 290 đến t + 320 giây: `y_300 = 1` với lenient, `−1` với strict |
| Cuối ca | t + h + 60 > end và không có tụt: `y_h = −1` |
| eligible | MAP hiện tại < 65: `in_hypotension`. Trong 120 giây sau khi đợt tụt kết thúc: `post_event` |
| eligible_h | Đợt tụt không có dòng eligible nào trong [onset − h, onset): `eligible_h = False` |
| PPV | Hiệu áp giảm đều trong 120 giây, không dao động theo nhịp thở: `bt_ppv30_w120` nhỏ hơn rõ so với max − min trên cả 120 giây. Có 1 nhịp `reject != 0` trong đoạn: đoạn đó bị bỏ |
| Cột | Có đúng 66 + 27 cột mỗi W. Không cột nào có tiền tố bị cấm |
| Tập | `load_split("test")` báo lỗi khi chưa có lock. Không bệnh nhân nào ở 2 tập |
| Cảnh báo | Xác suất vượt ngưỡng 1 lần: không cảnh báo. 2 lần liên tiếp: 1 cảnh báo. Vẫn trên ngưỡng sau 300 giây: không cảnh báo thêm (chưa xuống dưới ngưỡng). Xuống rồi lên lại trước 300 giây: không cảnh báo |
| Chỉ số | Ví dụ tay: 2 đợt tụt, 3 cảnh báo (2 đúng, 1 sai), 1 giờ phơi nhiễm → sensitivity 1,0, FA/giờ 1,0, PPV 2/3 |
| Ngưỡng | Trên dữ liệu giả, cách chia đôi cho cùng kết quả với quét toàn bộ ứng viên |
| Calibration | Sau Platt, `mean(p)` trên tập calibration bằng tỷ lệ dương (sai số < 1e-3) |
| DL | Output `[B, 5]`, xác suất tăng dần theo mốc, chạy được với W = 30 và W = 120 |
| DL nhân quả | Với một mẫu (caseid, t, W): mẫu sóng cuối cùng có thời điểm ≤ t và mẫu đầu tiên > t − W. Thay đổi `wave100` sau t không làm đổi tensor input |
| Khớp dòng | `labels/{split}` và `features/{split}` có cùng `(caseid, time)` theo đúng thứ tự |
| Luật loại mẫu | Một lần MAP < 65 kéo dài 20 giây (chưa thành đợt) trong (t − 120, t]: eligible với `event`, không eligible (`recent_low`) với `any_low` |
| `assert_no_test` | Chặn `sealed/x`, `labels/test.parquet`, `wave_mask_test.npz`, parquet có `split == "test"`. Không chặn `latest/x`, `tests/x`, `runs/healthcheck_1.json`. `reports/test_results.csv` chỉ được phép khi đã có `lock.json` |
| Đẩy file | `HubSync.push()` báo lỗi khi nhận đường dẫn không do notebook này tạo. Chạy với `--subset` thì không đẩy gì; `--smoke` chỉ đẩy `reports/dl_smoke/timing.json` |
| Bảng tổng | `02d` gộp đúng 65 file kết quả; thiếu một file thì dừng và in tên tổ hợp thiếu |

---

## 11. Thứ tự làm việc: 5 giai đoạn

Danh sách việc chi tiết nằm ở `task/task1.md`. Mục này chỉ ghi khung và các phụ thuộc bắt buộc.

| Giai đoạn | Nơi chạy | Việc | Xong khi |
|---|---|---|---|
| 0 | | Chốt kế hoạch | Đã xong (bản 2.2) |
| 1. Xử lý lại dữ liệu | Máy | Môi trường, khung repo, `hub.py`, tạo Kaggle Dataset cho sóng, `check_data.py`, nhãn, PPV, NB01, đẩy samples lên HF | Có `hub_revision.txt`. Nhóm đã chốt `exclusion_reset` và `label_policy` |
| 2. Tạo sẵn notebook | Máy | Code đánh giá, code train, `make_kaggle_notebooks.py`, sinh 11 notebook Kaggle, chạy thử từng notebook trên máy ở chế độ `local` với tập con | Mọi notebook chạy hết trên máy. `assert_no_test` qua với mọi file sẽ đẩy |
| 3. Train | Kaggle, HF | `00_env_check`, `02a`–`02d`, `03_dl_smoke`, `03_dl_W*`, `03_dl_finalize` | Có `tabular_validation.csv` và `dl_validation.csv` trên HF |
| 4. Khóa và test | Máy | `lock_models.py`, `final_test.py` | Có `test_results.csv` |
| 5. Tài liệu | Máy | `README.md`, hoàn thiện `DEVIATIONS.md` | |

**Phụ thuộc bắt buộc** (không được làm trước):
- Mọi notebook Kaggle cần samples trên HF (cuối giai đoạn 1) và code đã commit.
- `02d` cần 02a, 02b, 02c xong. `03_dl_W*` cần `03_dl_smoke`. `03_dl_finalize` cần các `03_dl_W*` đã chạy.
- Giai đoạn 4 cần nhóm chốt danh sách tổ hợp báo cáo.

**Được làm chồng lên nhau** (để kịp lịch):
- Viết code và test trên dữ liệu giả của giai đoạn 1.3 trong lúc chờ nhóm duyệt `data_check.json`.
- Viết code đánh giá (2.1) trong lúc NB01 chạy trên máy.
- **Chia giai đoạn 2 và 3 thành 2 đợt:**
  - **Đợt A (tabular):** code đánh giá, code tabular, notebook `00`, `02a`–`02d`. Chạy thử trên máy xong thì đẩy ngay lên Kaggle.
  - **Đợt B (DL):** viết code DL và các notebook `03_*` trong lúc Kaggle đang train tabular.
  - Cách này sớm hơn khoảng nửa ngày so với đợi viết xong mọi notebook rồi mới train.

**Chạy thử trên máy (giai đoạn 2.4):**
- Tập con phải có đủ hai lớp ở mọi mốc. Chọn ca train, calibration, validation có ít nhất một đợt tụt (được phép vì không dùng test). Nếu vẫn thiếu lớp dương ở một mốc, code phải bỏ qua mốc đó kèm cảnh báo, không được dừng.
- Đo tốc độ suy luận DL trên CPU của máy, dùng validation làm đại diện, để biết NB04 có chạy kịp trên máy không. Test khoảng 178 nghìn dòng. Nếu quá chậm, NB04 chỉ chạy DL cho các cấu hình đã chốt.

**Kaggle:**
- Kaggle giới hạn số phiên chạy đồng thời và quota GPU mỗi tuần. Xem quota trong trang tài khoản trước khi đẩy nhiều notebook cùng lúc.
- Sau `03_dl_smoke`, tính tổng giờ GPU cần (thời gian mỗi epoch × số epoch dự kiến × 20 lần train) và so với quota còn lại. Không đủ thì giảm seed theo mục 12.

**Tiêu chí xong mỗi notebook:** chạy từ đầu đến cuối không lỗi, output có file `.json` chứa digest, và bảng tóm tắt in ra đúng các mục ở 3.8, 6.3 hoặc 7.4.

---

## 12. Kế hoạch thời gian

| Thời gian | Việc | Kết quả |
|---|---|---|
| 29/09 | Giai đoạn 1.1–1.3. Tạo repo HF, Kaggle token | `data_check.json`, code nhãn và test |
| 30/09 sáng | Giai đoạn 1.4 (NB01 trên máy, đẩy samples). Song song: viết code đánh giá (2.1) | `hub_revision.txt`, `label_report.csv` |
| 30/09 chiều | Đợt A: code tabular, notebook `00`, `02a`–`02d`, chạy thử trên máy, đẩy lên Kaggle | Tabular bắt đầu train trên Kaggle |
| 30/09 tối – 01/10 sáng | Đợt B: code DL, notebook `03_*`, chạy thử trên máy. Đẩy `03_dl_smoke` | Thời gian mỗi epoch, số seed |
| 01/10 | `03_dl_W60`, `W30`, `W120`, `W90`. `02d` khi tabular xong | `tabular_validation.csv`, SHAP, ablation |
| 02/10 sáng | `03_dl_finalize`. Nhóm chốt cấu hình | `dl_validation.csv` |
| 02/10 chiều | Giai đoạn 4 (NB04 trên máy), tổng hợp bảng | `lock.json`, `test_results.csv` |

**Nếu chậm, cắt theo thứ tự:**
1. DL chỉ chạy W = 60 với 3 seed.
2. Ablation chỉ ở mốc 5 phút.
3. Bỏ phần mức tin cậy của LightGBM.
4. Bỏ hẳn DL. Hai baseline cùng `lgbm_numeric` và `lgbm_wave` đã đủ trả lời câu hỏi nghiên cứu chính.

---

## 13. Quyết định còn mở (đã chọn sẵn mặc định)

| Câu hỏi | Mặc định | Ai chốt |
|---|---|---|
| Luật loại mẫu sau khi MAP thấp | `exclusion_reset = "event"` (theo proposal) | Nhóm, sau khi xem `label_report` |
| Chính sách nhãn | `lenient`. `strict` chỉ để báo cáo | Nhóm |
| Tính lại PPV | Có (`bt_ppv30`) | Cố định |
| Giữ SpO2 | Giữ. Ablation sẽ cho biết có cần không | Theo ablation |
| Cửa sổ chính W\* | Chọn trên validation (mục 6.4) | Tự động |
| Số seed DL | 5, giảm còn 3 nếu thiếu GPU | Theo thời gian |
| Ca không có sóng | Loại khỏi cohort chính, báo cáo riêng | Cố định |
| Ngưỡng `map_coverage` θ cho cohort chính | **Đã chốt 0,10** (khoảng trống trên train (0,049; 0,219]) | Xong |

---

## Phụ lục A: Cột đặc trưng dùng cho mô hình (một cửa sổ W)

```python
SIG = ["map", "sbp", "dbp", "hr", "spo2", "etco2", "rr", "pp", "shock_index"]
STATS = ["mean", "std", "min", "max", "slope", "missing"]
BEAT = ["dpdt_max", "sys_area", "ejection_time", "notch_rel", "decay_tau", "sv", "co", "svr"]

def numeric_cols(W):                                     # 66 cột
    return ([f"{s}_{st}_w{W}" for s in SIG for st in STATS]          # 54
            + [f"{s}_current" for s in SIG]                           # 9, không phụ thuộc W
            + ["map_drop_pct", f"map_time_65_75_w{W}", f"map_extrap_w{W}"])   # 3

def wave_cols(W, ppv="bt_ppv30"):                        # 27 cột
    return ([f"bt_{b}_{st}_w{W}" for b in BEAT for st in ["mean", "std", "slope"]]   # 24
            + [f"{ppv}_w{W}", f"bt_valid_frac_w{W}", f"bt_n_beats_w{W}"])            # 3
```

- Tổng cột cần đọc cho 4 cửa sổ: 234 cột chỉ số (vì 10 cột không phụ thuộc W) và 108 cột sóng, tổng **342**.
- Đơn vị: độ dốc tính theo mỗi phút. `missing` là tỷ lệ ô lưới 2 giây bị thiếu trong cửa sổ. `map_drop_pct` là % MAP giảm so với MAP nền. `map_time_65_75` là số giây MAP ở vùng 65–75 mmHg. `map_extrap` là MAP hiện tại cộng độ dốc × 5 phút, kẹp trong [30, 150].

**Nhóm cột cho ablation:**

| Nhóm | Cột |
|---|---|
| MAP | `map_*` |
| SBP/DBP | `sbp_*`, `dbp_*`, `pp_*` |
| HR | `hr_*`, `shock_index_*` |
| SpO2 | `spo2_*` |
| EtCO2/RR | `etco2_*`, `rr_*` |
| Hình dạng sóng | `bt_dpdt_max_*`, `bt_sys_area_*`, `bt_ejection_time_*`, `bt_notch_rel_*`, `bt_decay_tau_*` |
| PPV | `bt_ppv30_*` |
| SV/CO/SVR proxy | `bt_sv_*`, `bt_co_*`, `bt_svr_*` |

**Lưu ý về SV/CO/SVR:** đây là proxy Liljestrand-Zander, SV = PP / (SBP + DBP), chỉ có ý nghĩa tương đối trong cùng một ca. Khi mọi áp lực giảm cùng tỷ lệ, SV proxy gần như không đổi, còn SVR proxy giảm theo MAP. Phần nguyên nhân (NB06) phải kiểm tra điều này bằng EV1000 trong 15 phút trước đợt tụt.

---

## 14. Nội dung bắt buộc của `reports/DEVIATIONS.md`

Mỗi mục ghi: khác gì so với proposal, lý do, và ảnh hưởng dự kiến.

| # | Nội dung | Lý do |
|---|---|---|
| 1 | Bỏ 4 ca `4458, 6119, 6133, 6375` (sóng ART 100 Hz, tiền xử lý lỗi) | Không tiền xử lý lại. Ảnh hưởng không đáng kể (4/3.626 ca) |
| 2 | Cohort chính yêu cầu `has_wave` và `map_coverage ≥ θ` (mục 3.6) | 180 ca có track sóng nhưng không có arterial line hoạt động |
| 3 | Nhãn dùng chính sách `lenient`: mẫu dương không cần dữ liệu sau đợt tụt | Tránh mất mẫu dương ở mốc dài chỉ vì nhiễu xảy ra sau đợt tụt |
| 4 | PPV tính lại theo đoạn 30 giây, lấy trung vị trong W | PPV trên cả cửa sổ lẫn xu hướng huyết áp |
| 5 | `in_event` và `post_event` dùng đợt tụt phát hiện từ `label_map`, có thể dùng tối đa khoảng 60 giây thông tin sau t | `label_map` được làm sạch với cờ nhìn về sau |
| 6 | DL có thêm cờ `baseline_missing` | 77 ca không có MAP nền |
| 7 | NB06 dùng thêm dữ liệu EV1000 ngoài `prep_v1` | `prep_v1` không có track EV1000 |
| 8 | Trong lần kiểm tra dữ liệu ngày 29/09, agent đã đọc `label_map` của test để đếm tổng số đợt tụt (gộp cả các tập, không tách theo tập). Không có mô hình, ngưỡng hay tiêu chí cohort nào được chọn dựa trên số này | Ghi lại để minh bạch. Từ nay áp dụng nguyên tắc 1 |
| 9 | `eligible` và `abstain_reason` của test cũng được niêm phong | Hai luật `in_event` và `post_event` suy ra từ đợt tụt |
| 10 | Không đánh giá hiệu năng riêng trên 78 ca không có sóng, chỉ báo cáo số ca | Validation chỉ có 3 ca, test 14 ca |
| 11 | `map_threshold` chỉ có 5 tổ hợp (một cho mỗi h), tổng tabular 65 thay vì 80 | Mô hình chỉ dùng `map_current`, không phụ thuộc W |
| 12 | Dữ liệu sóng trên Kaggle lấy từ Kaggle Dataset (`dataset_sources`), không từ output notebook (`kernel_sources`) | Tránh dữ liệu đổi giữa chừng khi notebook `safeanes` chạy lại |
| 13 | (Chỉ khi xảy ra) `torch.use_deterministic_algorithms(True, warn_only=True)` | Một số phép backward của attention trên CUDA không có bản tất định |

Ghi `DEVIATIONS.md` **dần theo từng task**, không đợi đến cuối.

---

## 15. Chạy ở đâu: máy cá nhân, Kaggle, Hugging Face

### 15.0 Tổng quan

```
MÁY CÁ NHÂN (Windows, ổ D)                  HUGGING FACE (private)            KAGGLE
───────────────────────────                 ──────────────────────            ──────
prep_v1/ ─► NB01 ─► samples_v2/ ──push────► <user>/uc04-v2-samples ──pull──► NB02 (CPU) ─┐
                     └─ sealed/ (test, giữ lại trên máy)            │                  │
                                                                    └──pull──► NB03 (GPU) ─┤  + wave100 từ output
make_kaggle_notebooks.py ─► kaggle kernels push ───────────────────────────────► NB02, NB03   notebook safeanes
                                            <user>/uc04-v2-runs ◄──push tự động─────────────┘
NB04 ◄──pull (revision trong lock.json)──── <user>/uc04-v2-runs
 └─ dùng sealed/ và wave100 trên máy ─► reports/test_results.csv
```

### 15.1 Chuẩn bị máy cá nhân (một lần)

1. Cài Python 3.11 và tạo venv trong thư mục repo:
   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   pip install -e ".[dev,boosting,deep,hub]"
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
   Nhóm `hub` trong `pyproject.toml` gồm `huggingface_hub` và `kaggle`.
2. Đăng nhập Hugging Face: `hf auth login` (bản cũ: `huggingface-cli login`). Dùng token có quyền ghi.
3. Kaggle CLI: tải `kaggle.json` từ trang Settings của Kaggle, đặt vào `%USERPROFILE%\.kaggle\kaggle.json`.
4. Trên Kaggle, tạo 2 Secret: `HF_TOKEN` (token HF có quyền ghi) và `GITHUB_TOKEN` (chỉ khi repo code là private).
5. Tạo 2 repo **private** trên HF: `samples_repo` (loại dataset) và `runs_repo` (loại model). `hub.py` có thể tự tạo bằng `create_repo(..., private=True, exist_ok=True)`.

**Lưu ý cho Windows:**
- Dùng `pathlib` cho mọi đường dẫn.
- Code chạy song song (`ProcessPoolExecutor`) phải nằm trong package `uc04` và được gọi qua `scripts/*.py` có `if __name__ == "__main__":`. Trong notebook, gọi `!python scripts/...` thay vì tạo pool trực tiếp trong cell, vì Windows không chạy được hàm định nghĩa trong notebook ở process con.
- NB01 ghi bảng `features/{split}.parquet` theo kiểu nối dần từng ca (`pyarrow.parquet.ParquetWriter`), không gom toàn bộ vào RAM rồi mới ghi.

### 15.2 Dữ liệu: máy cá nhân → Hugging Face → Kaggle

- `push_samples.py` đẩy `samples_v2/` lên `samples_repo` bằng `HfApi.upload_folder(..., ignore_patterns=["sealed/*", "*test*"])`. Hàm này tự chia commit và chạy lại được nếu đứt mạng. Trước khi đẩy, `hub.py` kiểm tra lần cuối là không file nào chứa dữ liệu test.
- Sau khi đẩy xong, ghi commit sha của HF vào `samples_v2/hub_revision.txt`. Mọi notebook Kaggle tải đúng revision này: `snapshot_download(repo_id, repo_type="dataset", revision=<sha>)`.
- Kích thước dự kiến khoảng 1–2 GB (chủ yếu là `features/train.parquet`).
- `wave100/` **không** đẩy lên HF. NB03 lấy từ Kaggle Dataset `kaggle.wave_dataset` (tạo từ output của notebook `safeanes`, mục 1.1), gắn qua `dataset_sources`. `kaggle.wave_dataset_version` ghi phiên bản đang dùng, để đối chiếu. NB03 và `00_env_check` so sha256 của `qc.csv` trong dataset với giá trị trong `samples.json`. Khác thì dừng.

### 15.3 Kết quả: Kaggle → Hugging Face (tự động)

**Cấu trúc `runs_repo`** (giống hệt thư mục `artifacts/` và `reports/` trên Kaggle). Mỗi đường dẫn chỉ có **đúng một notebook ghi**:

| Đường dẫn | File | Notebook ghi |
|---|---|---|
| `tabular/map_threshold/h{h}/` | `model.joblib  threshold.json  val_predictions.parquet  case_metrics.parquet  provenance.json  done.json` | `02a` |
| `tabular/map_logistic/W{W}/h{h}/` | như trên | `02a` |
| `tabular/lgbm_numeric/W{W}/h{h}/` | như trên | `02b` |
| `tabular/lgbm_wave/W{W}/h{h}/` | như trên | `02c` |
| `reports/tabular_validation/`, `reports/calibration_check/` | Một file một dòng cho mỗi tổ hợp, tên `{model}_W{W}_h{h}.csv` hoặc `map_threshold_h{h}.csv` | `02a`, `02b`, `02c` (mỗi notebook chỉ ghi file của mô hình mình) |
| `reports/` | `tabular_validation.csv  calibration_check_tabular.csv  shap_W{W}_h{h}.csv  ablation.csv` | `02d` |
| `reports/tabular_confidence/` | Mức tin cậy từ 5 seed | `02d` |
| `reports/dl_smoke/` | `timing.json` | `03_dl_smoke` |
| `dl/W{W}/seed{s}/` | `best.pt  last.pt  train_log.csv  logits_calibration.parquet  logits_validation.parquet  provenance.json  done.json` | `03_dl_W{W}` |
| `dl/W{W}/` | `logits_calibration.parquet  logits_validation.parquet  calibration.json  threshold.json  case_metrics_h{h}.parquet` | `03_dl_finalize` |
| `reports/` | `dl_validation.csv  calibration_check_dl.csv` | `03_dl_finalize` |
| `runs/` | `{run_id}.json` (một file cho mỗi lần chạy), `healthcheck_*.json` | Notebook tương ứng, `00_env_check` |
| `lock.json` | Sau NB04 bước 2 | NB04 (máy) |
| `reports/test_results.csv` | Bảng kết quả tổng của test, sau khi khóa | NB04 (máy) |

- `run_id` = `{notebook}_{UTC time}_{git commit 7 ký tự}`. File `runs/{run_id}.json` ghi: commit code, revision của `samples_repo`, config digest, thời gian bắt đầu và kết thúc, trạng thái (`running`, `done`, `failed`), danh sách tổ hợp đã xong.

**`hub.py` cần có:**

| Hàm | Việc |
|---|---|
| `HubSync(repo_id, local_dir, min_interval_s)` | Giữ thời điểm đẩy gần nhất |
| `.maybe_push(paths, message)` | Đẩy nếu đã qua `min_interval_s` kể từ lần trước, ngược lại bỏ qua |
| `.push(paths, message)` | Luôn đẩy. Dùng khi xong một tổ hợp hoặc một lần train, và ở cuối notebook |
| `.done_set(prefix)` | Danh sách thư mục đã có `done.json` trên HF, để bỏ qua khi chạy lại |
| `.pull(path, revision=None)` | Tải một file hoặc thư mục về (`hf_hub_download` hoặc `snapshot_download`) |
| `assert_no_test(paths)` | Báo lỗi nếu (a) một **thành phần** của đường dẫn bằng `sealed`, hoặc tên file khớp `test_*`, `*_test.*`, `*_test`; (b) file parquet hoặc csv có cột `split` chứa giá trị `test`. So theo từng thành phần, không so chuỗi con, để `latest`, `tests/` hay `smoke` không bị chặn nhầm. Ngoại lệ duy nhất là `reports/test_results.csv`, và chỉ khi đã có `lock.json` |

- Đẩy bằng `HfApi.upload_folder(folder_path, path_in_repo, allow_patterns=...)`. **Không dùng `CommitScheduler`**, vì công cụ này giả định file chỉ được thêm mới, trong khi `last.pt` bị ghi đè sau mỗi epoch.
- **Nhiều notebook đẩy cùng lúc:** mỗi notebook chỉ đẩy thư mục của riêng mình (`path_in_repo` = đúng thư mục tổ hợp hoặc seed), không bao giờ đẩy cả `artifacts/`. Nếu HF báo xung đột commit, cơ chế thử lại ở trên xử lý.
- **Chạy trên tập con thì không được đẩy:** khi có bất kỳ tham số giới hạn dữ liệu nào (`--subset`, `--max-cases`, `--max-samples`, `--epochs` nhỏ hơn config), script tự bật `--no-push` và ghi vào `artifacts_dryrun/`, để kết quả chạy thử không lẫn vào `runs_repo`.
  - **Ngoại lệ duy nhất: `03_dl_smoke`** (chạy với `--smoke`). Script vẫn không đẩy mô hình hay logit, nhưng đẩy đúng một file `reports/dl_smoke/timing.json`: thời gian mỗi epoch, RAM, VRAM, số mẫu mỗi giây, GPU. Như vậy có thể tính giờ GPU cần mà không phải tải output của notebook về.
- **Giãn cách đẩy:** không đẩy dày hơn 10 phút một lần, trừ lúc xong một tổ hợp hay một lần train. HF khuyên không commit quá dày.
- **Lỗi mạng khi đẩy không được làm dừng train:** thử lại 3 lần (chờ 30, 60, 120 giây). Nếu vẫn lỗi thì ghi cảnh báo vào log rồi train tiếp. Lần đẩy sau sẽ gửi cả phần còn thiếu.
- **Cell cuối của mỗi notebook Kaggle** luôn gọi `push()` cho **các thư mục và file do chính notebook này tạo ra** (theo bảng ở trên), rồi cập nhật `runs/{run_id}.json`. **Không** đẩy cả `artifacts/` hay `reports/`: thư mục cục bộ có thể chứa bản cũ của file do notebook khác ghi (ví dụ file đã kéo về để kiểm tra `done.json`), và nếu đẩy cả thư mục thì bản cũ sẽ ghi đè bản mới trên HF.
- `HubSync` giữ danh sách đường dẫn mà notebook đã tạo trong phiên này, và `push()` chỉ nhận đường dẫn nằm trong danh sách đó. Nếu nhận đường dẫn khác, báo lỗi.
- Script có tham số `--no-push` để chạy thử trên máy mà không đẩy gì.

### 15.4 Notebook Kaggle: tạo sẵn trên máy, đẩy bằng CLI

`scripts/make_kaggle_notebooks.py` đọc `templates/` và sinh cho mỗi notebook một thư mục gồm file `.ipynb` và `kernel-metadata.json`. Script tự điền sẵn commit của code và revision của `samples_repo`, để không phải sửa tay.

**11 notebook Kaggle** (mỗi notebook một thư mục trong `notebooks/kaggle/`, id dạng `<kaggle_user>/uc04-v2-<tên>`):

| Notebook | Máy | `dataset_sources` | Nội dung |
|---|---|---|---|
| `00_env_check` | CPU | có | Đường dẫn, sha256 `qc.csv`, Secret, ghi thử `runs/healthcheck_*.json` |
| `02a_baselines` | CPU | không | `map_threshold` (5) và `map_logistic` (20) |
| `02b_lgbm_numeric` | CPU | không | 20 tổ hợp |
| `02c_lgbm_wave` | CPU | không | 20 tổ hợp |
| `02d_tabular_analysis` | CPU | không | Gộp bảng, W\*, SHAP, ablation, mức tin cậy |
| `03_dl_smoke` | GPU | có | `nvidia-smi`, 1 epoch W = 60, `timing.json` |
| `03_dl_W60`, `03_dl_W30`, `03_dl_W120`, `03_dl_W90` | GPU | có | Mỗi notebook một W, các seed đã chốt. Nếu quá 12 giờ, tách seed qua `RUN_ARGS` |
| `03_dl_finalize` | CPU | không | Ensemble, calibrate, chọn ngưỡng cho mọi W |

**`kernel-metadata.json` của `03_dl_W60`** (notebook CPU: `enable_gpu: false`, không có `machine_shape`; notebook không cần sóng: `dataset_sources` rỗng):

```json
{
  "id": "<kaggle_user>/uc04-v2-03-dl-w60",
  "title": "uc04-v2-03-dl-w60",
  "code_file": "03_dl_W60.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "machine_shape": "NvidiaTeslaT4",
  "enable_internet": true,
  "kernel_sources": [],
  "dataset_sources": ["<kaggle_user>/uc04-prep-v1"],
  "competition_sources": []
}
```

**Các cell của notebook Kaggle:**

```python
# Cell 1: tham số (make_kaggle_notebooks.py điền sẵn)
import os
RUNTIME = "kaggle" if os.path.exists("/kaggle/input") else "local"   # cùng một file chạy được ở cả hai nơi
CODE_COMMIT = "<sha>"; SAMPLES_REVISION = "<sha>"
RUN_ARGS = "--window 60 --seeds 20260917 20260918 20260919 20260920 20260921"
LOCAL_ARGS = "--subset dryrun --max-samples 5000 --epochs 1"   # chỉ dùng khi RUNTIME == "local"

# Cell 2: token và đường dẫn theo RUNTIME, không in token
if RUNTIME == "kaggle":
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    REPO, WORK = "uc04", "/kaggle/working"
    WAVE100 = "/kaggle/input/uc04-prep-v1/<...>/wave100"   # kiểm tra bằng 00_env_check
    EXTRA = ""
else:                          # token lấy từ `hf auth login`
    REPO, WORK = ".", "artifacts_dryrun"
    WAVE100 = "D:/SafeAnes/SafeAnes/data/prep_v1/wave100"
    EXTRA = LOCAL_ARGS         # tập con, tự tắt việc đẩy lên HF

# Cell 3: lấy code đúng commit và cài đặt (chỉ trên Kaggle)
if RUNTIME == "kaggle":
    !git clone -q https://github.com/<org>/uc04.git && cd uc04 && git checkout -q {CODE_COMMIT}
    !pip install -q -e "uc04[boosting,deep,hub]"

# Cell 4: tải samples từ HF đúng revision
!python {REPO}/scripts/pull_samples.py --revision {SAMPLES_REVISION} --out {WORK}/samples_v2

# Cell 5: train; script tự đẩy kết quả lên HF và chạy tiếp phần dở dang
!python {REPO}/scripts/train_dl.py --config {REPO}/configs/uc04_v2.json \
    --samples {WORK}/samples_v2 --wave100 {WAVE100} --out {WORK}/artifacts {RUN_ARGS} {EXTRA}

# Cell 6: đẩy lần cuối (chỉ thư mục của notebook này) và in tóm tắt
!python {REPO}/scripts/train_dl.py --final-push-only --out {WORK}/artifacts {RUN_ARGS} {EXTRA}
```

- Đường dẫn `wave100` trong Cell 2 phải được kiểm tra một lần bằng `!ls /kaggle/input` (notebook `00_env_check`) rồi điền vào `templates/`. Đường dẫn này phụ thuộc cấu trúc thư mục bên trong Kaggle Dataset.
- **Dùng `dataset_sources`, không dùng `kernel_sources`.** `kernel_sources` luôn lấy output của phiên bản mới nhất của notebook `safeanes`, nên dữ liệu sóng có thể đổi giữa chừng. Kaggle Dataset `kaggle.wave_dataset` chỉ đổi khi nhóm chủ động tạo phiên bản mới. **Không tạo phiên bản mới trong lúc train.** `dataset_sources` có thể luôn gắn phiên bản mới nhất của dataset, nên bước so sha256 của `qc.csv` (với giá trị trong `samples.json`) là hàng rào bắt buộc. Nếu buộc phải tạo phiên bản mới, tăng `kaggle.wave_dataset_version` và sinh lại notebook.
- Tài khoản Kaggle dùng để chạy phải đọc được dataset này. Tạo dataset bằng chính tài khoản đó thì không cần chia sẻ.
- **Template có chế độ `RUNTIME = "local"` hoặc `"kaggle"`.** Ở chế độ local, notebook bỏ qua `kaggle_secrets` và `git clone`, lấy token từ `hf auth login`, lấy đường dẫn từ config, và thêm tham số tập con. Nhờ vậy, chính file `.ipynb` sẽ chạy trên Kaggle cũng chạy thử được trên máy (bằng `jupyter nbconvert --execute` hoặc papermill).
- `00_env_check` chạy trên **CPU** (kiểm tra đường dẫn, sha256, Secret, ghi thử file `runs/healthcheck_*.json` lên HF). Phần kiểm tra GPU gộp vào `03_dl_smoke`, để không tốn quota GPU.
- Lần đầu chạy notebook trên Kaggle, mở giao diện một lần để **bật Secret `HF_TOKEN` cho notebook đó** (Add-ons → Secrets). Những lần đẩy sau bằng CLI sẽ giữ cài đặt này.

**Lệnh trên máy:**

```bat
python scripts\make_kaggle_notebooks.py --config configs\uc04_v2.json
jupyter nbconvert --execute --to notebook --output-dir artifacts_dryrun notebooks\kaggle\02b_lgbm_numeric\02b_lgbm_numeric.ipynb   # chạy thử trên máy, RUNTIME tự thành "local"
kaggle kernels push -p notebooks\kaggle\02b_lgbm_numeric
kaggle kernels status <kaggle_user>/uc04-v2-02b-lgbm-numeric
kaggle kernels output <kaggle_user>/uc04-v2-02b-lgbm-numeric -p logs\02b_lgbm_numeric   # lấy log nếu cần
```

- Chia NB03 thành nhiều lần chạy nếu cần: mỗi lần một W, truyền qua `RUN_ARGS`. Vì kết quả đã lưu trên HF, lần chạy sau tự bỏ qua phần đã xong.

### 15.5 Những gì không bao giờ được đẩy lên

| Không đẩy | Lý do |
|---|---|
| `samples_v2/sealed/`, mọi file có dòng hoặc nhãn của test | Nguyên tắc 1 và 13 |
| `prep_v1/` | Đã có trên Kaggle qua Kaggle Dataset `kaggle.wave_dataset` (tạo từ output notebook `safeanes`). Không đẩy lên HF |
| Dữ liệu và kết quả từ MOVER | Phụ thuộc DUA của MOVER. Kiểm tra DUA trước |
| Token, `kaggle.json` | Nguyên tắc 14 |
