# Chạy DL (Conv1D + Transformer) trên Kaggle

Code lấy thẳng từ GitHub (`git clone`), dữ liệu lấy từ Kaggle Dataset, kết quả tự lưu lên Hugging Face.
Mỗi bước chỉ cần 2 ô lệnh trong một notebook Kaggle.

| Thứ tự | Bước | Máy | Việc |
|---|---|---|---|
| 1 | `00_env_check` | CPU | Kiểm tra dữ liệu, thư viện, Hugging Face. Chạy một lần |
| 2 | `03_dl_smoke` | GPU T4 x2 | 1 epoch ở W = 60: đo thời gian, ước tính giờ GPU. Chạy một lần |
| 3 | `03_dl_W60`, `03_dl_W30`, `03_dl_W120`, `03_dl_W90` | GPU T4 x2 | Train 5 seed cho mỗi cửa sổ W |
| 4 | `03_dl_finalize` | CPU | Gộp seed, hiệu chỉnh xác suất, chọn ngưỡng, bảng kết quả DL. Chạy sau khi đủ 4 W |

## 0. Chuẩn bị (một lần)

1. **Code trên GitHub:** commit và push mọi thay đổi trước khi chạy. Kaggle clone `main` của
   `https://github.com/tuananhpham-vnu/SafeAnes`.
2. **Repo kết quả trên Hugging Face:** `tuananhpham-vnu/uc04-v2-runs`, phải là **private**
   (code từ chối đẩy lên repo public). Đặt ở `hub.runs_repo` trong `configs/uc04_v2.json`.
3. **Secret trên Kaggle:** *Settings → Secrets*, thêm `HF_TOKEN` (token HF loại **Write**).
4. **Quyền đọc dữ liệu:** tài khoản Kaggle phải mở được
   - `datnguyen31112/uc04-samples-v2` (bảng mẫu, nhãn, đặc trưng),
   - `datnguyen31112/uc04-prep-v1` (sóng 100 Hz, khoảng 7 GB).

## 1. Tạo notebook cho một bước

*Create → New Notebook*, rồi ở cột bên phải:

| Mục | Giá trị |
|---|---|
| **Accelerator** | `03_dl_smoke`, `03_dl_W*`: **GPU T4 x2**. `00_env_check`, `03_dl_finalize`: **None** |
| **Internet** | **On** |
| **Add Input** | `datnguyen31112/uc04-samples-v2`, và `datnguyen31112/uc04-prep-v1` (trừ `03_dl_finalize`). Mỗi dataset gắn đúng một lần |
| **Add-ons → Secrets** | Bật **`HF_TOKEN`** (phải bật riêng cho từng notebook) |

**Ô 1: lấy code** (khoảng 4 giây, chỉ tải phần code, không tải `data/`)

```python
!rm -rf SafeAnes
!git clone --depth 1 --filter=blob:none --sparse https://github.com/tuananhpham-vnu/SafeAnes.git
!cd SafeAnes && git sparse-checkout set src/uc04 scripts templates configs && git log --oneline -1
```

**Ô 2: chạy bước** (đổi tên bước)

```python
!python SafeAnes/scripts/kaggle_run.py 03_dl_smoke
```

Rồi **Save Version → Save & Run All (Commit)**. Notebook chạy ngầm (tối đa 12 giờ), tắt trình duyệt vẫn chạy.
Nên tạo **một notebook cho mỗi bước** (ví dụ `uc04-dl-W60`), để chạy lại bước nào thì mở đúng notebook đó.

`kaggle_run.py` tự làm theo thứ tự: tìm input → kiểm tra sha256 của `samples.json` và `qc.csv` →
lấy `HF_TOKEN` → cài những thư viện còn thiếu hoặc cũ hơn `requirements.txt` (giữ nguyên bản có sẵn trên
Kaggle, kể cả torch bản CUDA) → kiểm tra phiên bản → chạy bước. Phiên bản thực tế được ghi vào
`provenance.json` của mỗi lần train. Sai một thứ là dừng ngay, trước khi tốn GPU.

## 2. `00_env_check`

```python
!python SafeAnes/scripts/kaggle_run.py 00_env_check
```

Mọi dòng phải là `[PASS]`, gồm cả `HF push (healthcheck)`. Trên HF sẽ có thêm file `runs/healthcheck_<thời điểm>.json`.

## 3. `03_dl_smoke`

```python
!python SafeAnes/scripts/kaggle_run.py 03_dl_smoke
```

Train 1 epoch ở W = 60 với 1 seed (chỉ đẩy file đo thời gian lên HF, không đẩy mô hình). Cuối log có:

```
SMOKE: ... s/epoch at W=60 (300,000 samples), RAM ... GB, VRAM ... GB
estimated GPU hours for 4 windows x 5 seeds: ~... h (12 epochs), up to ... h (20 epochs)
with 2 GPU(s) running seeds in parallel, Kaggle sessions take ~... h (up to ... h)
```

- Số cần xem là **`session_hours`** (dòng cuối): tổng giờ phiên Kaggle cho cả 4 W. So với quota GPU còn lại
  trong tuần (*Settings → Quotas*).
- Kết quả cũng được lưu ở HF: `reports/dl_smoke/timing.json`.
- Ước tính giả định 2 GPU chạy song song không làm chậm nhau (Kaggle chỉ có 4 vCPU). Nếu số
  `samples/s` của từng seed khi train thật thấp hơn nhiều so với lúc smoke, dùng `--gpus 1` (mục 6).

## 4. Train 4 cửa sổ: `03_dl_W60`, `03_dl_W30`, `03_dl_W120`, `03_dl_W90`

Mỗi cửa sổ một notebook, cùng 2 ô lệnh, chỉ đổi tên bước:

```python
!python SafeAnes/scripts/kaggle_run.py 03_dl_W60
```

```python
!python SafeAnes/scripts/kaggle_run.py 03_dl_W30
```

```python
!python SafeAnes/scripts/kaggle_run.py 03_dl_W120
```

```python
!python SafeAnes/scripts/kaggle_run.py 03_dl_W90
```

- Thứ tự theo kế hoạch: **W60 → W30 → W120 → W90** (W60 đủ 5 seed trước). Các notebook ghi vào thư mục
  khác nhau trên HF, nên chạy song song được nếu quota và số phiên GPU đồng thời cho phép.
- Mỗi W train 5 seed (`20260917` … `20260921`), mỗi seed tối đa 20 epoch, dừng sớm sau 4 epoch không cải thiện.
  Với 2 GPU, 2 seed chạy cùng lúc (3 lượt cho 5 seed).
- W120 lâu nhất (chuỗi dài nhất), W30 nhanh nhất.
- Log có dạng:

  ```
  [gpu0 seed 20260917]   W60 seed 20260917 epoch 3: train 0.1234 val 0.1301 *  412 s (850 samples/s)
  ```

  Dấu `*` là epoch tốt hơn epoch tốt nhất trước đó (lưu thành `best.pt`).

## 5. Theo dõi, bị ngắt thì chạy tiếp

- **Tiến độ** trên https://huggingface.co/tuananhpham-vnu/uc04-v2-runs (private):
  `dl/W<W>/seed<s>/` có `train_log.csv`, `best.pt`, `last.pt`. Seed nào có **`done.json`** là xong.
- Trong lúc train, thư mục của seed được đẩy lên HF khoảng 10 phút một lần.
- **Phiên bị dừng** (hết 12 giờ, lỗi, hết quota): mở lại đúng notebook đó và *Save & Run All* lại. Code tự
  bỏ qua seed đã có `done.json`, tải `best.pt` + `last.pt` của seed đang dở và chạy tiếp từ epoch đó
  (mất tối đa khoảng 10 phút train).
- Một W xong khi cả 5 seed có `done.json` (log in `seed ...: best epoch ...` cho từng seed).

## 6. `03_dl_finalize` (sau khi đủ 4 W)

Notebook CPU, chỉ gắn `datnguyen31112/uc04-samples-v2`:

```python
!python SafeAnes/scripts/kaggle_run.py 03_dl_finalize
```

Đọc logit của từng seed từ HF, gộp 5 seed, hiệu chỉnh (một nhiệt độ và một bias chung), chọn ngưỡng trên
validation. Kết quả trên HF: `reports/dl_validation.csv`, `reports/calibration_check_dl.csv`,
`dl/W<W>/calibration.json`, `dl/W<W>/threshold.json`.

## Tùy chọn

Tham số **sau** tên bước được chuyển cho script train; tham số của `kaggle_run.py` đặt **trước** tên bước.

| Lệnh | Khi nào |
|---|---|
| `kaggle_run.py 03_dl_W60 --gpus 1` | Chạy tuần tự trên 1 GPU (máy 1 GPU, hoặc 2 seed song song bị nghẽn CPU) |
| `kaggle_run.py 03_dl_W60 --seeds 20260917 20260918` | Chỉ train một số seed |
| `kaggle_run.py --skip-install 03_dl_W30` | Chạy bước thứ hai trong cùng một phiên (thư viện đã cài) |
| `git checkout <commit>` sau ô clone | Cố định đúng một phiên bản code (commit được ghi vào `provenance.json` của mỗi lần train) |

## Lỗi thường gặp

| Thông báo | Cách xử lý |
|---|---|
| `Kaggle Secret HF_TOKEN is missing or not enabled for this notebook` | *Add-ons → Secrets*, bật `HF_TOKEN` cho notebook này |
| `expected exactly one uc04-samples-v2 dataset ...` / `... wave dataset (uc04-prep-v1)` | Thiếu hoặc gắn trùng dataset trong *Add Input* |
| `samples.json sha256 ... wrong uc04-samples-v2 version` | Dataset samples khác bản đã kiểm tra (`kaggle.samples_json_sha256` trong config) |
| `qc.csv sha256 ... wrong uc04-prep-v1 dataset` | Dataset sóng khác bản đã kiểm tra (`input.qc_sha256`) |
| `FAIL vs .../requirements.txt` | Thư viện thiếu hoặc cũ hơn mức tối thiểu; chạy lại không có `--skip-install` |
| `... is PUBLIC but hub.private is true` | Repo HF đang public: *Settings → Change visibility* thành private |
| `seed ... on gpu1 FAILED` | Xem log của seed đó (dòng có `[gpu1 seed ...]`), sửa rồi chạy lại notebook; các seed khác không bị ảnh hưởng |
| `CUDA out of memory` | Thêm `--gpus 1`; nếu vẫn lỗi, báo nhóm (không tự giảm batch size: kế hoạch cố định 256) |

## Quy tắc không được phá

- Không đọc hay đẩy dữ liệu **test** lên đâu cả: Kaggle chỉ có train/calibration/validation, `hub.py` chặn mọi
  file có `test`/`sealed` trước khi đẩy.
- Không sửa siêu tham số trong `configs/uc04_v2.json` (`dl.*`) để "chạy nhanh hơn": kế hoạch đã chốt; mọi khác
  biệt phải ghi vào `docs/DEVIATIONS.md`.
- Token chỉ để trong Kaggle Secrets và `.env` (đã git-ignore), không đưa vào notebook hay code.
