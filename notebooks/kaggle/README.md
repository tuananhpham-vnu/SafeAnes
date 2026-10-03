# Hướng dẫn chạy DL (Conv1D + Transformer) trên Kaggle

Code lấy từ GitHub, dữ liệu lấy từ Kaggle Dataset, kết quả tự lưu lên Hugging Face (HF).
Mỗi bước là **một notebook Kaggle với một ô lệnh**, chỉ khác dòng `STEP`.

```
00_env_check  →  03_dl_smoke  →  03_dl_W60 → 03_dl_W30 → 03_dl_W120 → 03_dl_W90  →  03_dl_finalize
   (CPU)          (GPU)                       (GPU, mỗi W một notebook)                  (CPU)
```

---

## Bước 0. Chuẩn bị (làm một lần)

- [ ] **Push code lên GitHub.** Kaggle clone nhánh `main` của https://github.com/tuananhpham-vnu/SafeAnes,
      nên thay đổi nào chưa push thì Kaggle không thấy.
- [ ] **Repo kết quả trên HF:** https://huggingface.co/tuananhpham-vnu/uc04-v2-runs phải là **private**
      (code từ chối đẩy lên repo public). Tên repo nằm ở `hub.runs_repo` trong `configs/uc04_v2.json`.
- [ ] **Secret trên Kaggle:** *Settings → Secrets → Add*, tên `HF_TOKEN`, giá trị là token HF loại **Write**.
- [ ] **Quyền đọc dữ liệu:** tài khoản Kaggle mở được hai dataset
  - `datnguyen31112/uc04-samples-v2`: bảng mẫu, nhãn, đặc trưng;
  - `datnguyen31112/uc04-prep-v1`: sóng 100 Hz (khoảng 7 GB).

---

## Bước 1. Tạo notebook (làm giống nhau cho mọi bước)

1. Kaggle → **Create → New Notebook**. Đặt tên theo bước, ví dụ `uc04-dl-W60`.
2. Cột bên phải (*Notebook options* và *Input*):

   | Mục | `00_env_check` | `03_dl_smoke`, `03_dl_W*` | `03_dl_finalize` |
   |---|---|---|---|
   | **Accelerator** | None | **GPU T4 x2** | None |
   | **Internet** | On | On | On |
   | **Add Input** `uc04-samples-v2` | ✔ | ✔ | ✔ |
   | **Add Input** `uc04-prep-v1` | ✔ | ✔ | — |
   | **Add-ons → Secrets** `HF_TOKEN` | bật | bật | bật |

   Secret phải **bật riêng trong từng notebook**. Mỗi dataset chỉ gắn một lần.

3. Xóa ô mẫu, dán **một ô** sau và sửa dòng `STEP`:

   ```python
   STEP = "00_env_check"   # 00_env_check | 03_dl_smoke | 03_dl_W60 | 03_dl_W30 | 03_dl_W120 | 03_dl_W90 | 03_dl_finalize

   !rm -rf SafeAnes
   !git clone -q --depth 1 --filter=blob:none --sparse https://github.com/tuananhpham-vnu/SafeAnes.git
   !cd SafeAnes && git sparse-checkout set src/uc04 scripts templates configs && git log --oneline -1
   !python SafeAnes/scripts/kaggle_run.py {STEP}
   ```

   Lệnh clone chỉ tải phần code (khoảng 1 MB, vài giây), không tải `data/` của repo.

4. Bấm **Save Version → Save & Run All (Commit) → Save**. Notebook chạy ngầm tối đa 12 giờ;
   tắt trình duyệt hay máy vẫn chạy. Xem log ở trang notebook → *Version → Logs*.

`kaggle_run.py` lần lượt: tìm input → kiểm tra sha256 của `samples.json` và `qc.csv` → lấy `HF_TOKEN` →
cài thư viện còn thiếu hoặc cũ hơn `requirements.txt` (giữ bản có sẵn của Kaggle, kể cả torch CUDA) →
kiểm tra phiên bản → chạy bước. Sai một thứ là dừng ngay, trước khi tốn GPU.

---

## Bước 2. `00_env_check` (CPU, vài phút, một lần)

`STEP = "00_env_check"`

**Xong khi:** mọi dòng là `[PASS]`, gồm cả `[PASS] HF push (healthcheck)`, và trên HF có thêm
`runs/healthcheck_<thời điểm>.json`. Có dòng `[FAIL]` thì xem mục *Lỗi thường gặp*.

Lần đầu gắn `uc04-prep-v1`, Kaggle có thể mất 10–15 phút chuẩn bị dữ liệu trước khi chạy.

---

## Bước 3. `03_dl_smoke` (GPU, khoảng 30 phút, một lần)

`STEP = "03_dl_smoke"`

Train thử **1 epoch, W = 60, 1 seed**, rồi dự đoán trên calibration và validation. Mục đích: bắt lỗi GPU,
lỗi bước dự đoán trước khi tốn nhiều giờ, và đo thời gian để ước tính quota. Không đẩy mô hình lên HF.

**Xong khi** cuối log có ba dòng:

```
SMOKE: ... s/epoch at W=60 (300,000 samples), RAM ... GB, VRAM ... GB
estimated GPU hours for 4 windows x 5 seeds: ~... h (12 epochs), up to ... h (20 epochs)
with 2 GPU(s) running seeds in parallel, Kaggle sessions take ~... h (up to ... h)
```

- Số quan trọng là **`session_hours`** (dòng cuối): tổng giờ phiên Kaggle cho cả 4 W.
  So với quota GPU còn lại trong tuần (*Settings → Quotas*) để biết cần mấy tuần.
- File `reports/dl_smoke/timing.json` trên HF lưu các số này.
- Số này lạc quan: smoke chạy 1 GPU, còn khi train thật 2 seed chạy song song và dùng chung 4 vCPU.

---

## Bước 4. Train 4 cửa sổ (GPU, mỗi W một notebook)

| Notebook | `STEP` | Ghi chú |
|---|---|---|
| `uc04-dl-W60` | `"03_dl_W60"` | Chạy **đầu tiên** (theo kế hoạch) |
| `uc04-dl-W30` | `"03_dl_W30"` | Nhanh nhất |
| `uc04-dl-W120` | `"03_dl_W120"` | Lâu nhất (chuỗi dài nhất) |
| `uc04-dl-W90` | `"03_dl_W90"` | |

- Mỗi W train **5 seed** (`20260917` … `20260921`), mỗi seed tối đa 20 epoch, dừng sớm sau 4 epoch không cải thiện.
  Với GPU T4 x2, 2 seed chạy cùng lúc, nên 5 seed là 3 lượt.
- Các W ghi vào thư mục khác nhau trên HF, nên có thể chạy nhiều notebook cùng lúc nếu quota cho phép.
- Log mỗi epoch:

  ```
  [gpu0 seed 20260917]   W60 seed 20260917 epoch 3: train 0.1234 val 0.1301 *  412 s (850 samples/s)
  ```

  Dấu `*` nghĩa là epoch này tốt nhất từ trước đến giờ (được lưu thành `best.pt`).
  So `samples/s` với lúc smoke: nếu thấp hơn rất nhiều thì 2 seed đang nghẽn CPU, chạy lại với `--gpus 1` (mục *Tùy chọn*).

**Xong khi:** log in `seed ...: best epoch ..., val loss ...` cho cả 5 seed và kết thúc bằng `03_dl_W60 done`.
Trên HF, cả 5 thư mục `dl/W60/seed2026091*/` đều có `done.json`.

### Theo dõi tiến độ

Trên https://huggingface.co/tuananhpham-vnu/uc04-v2-runs (thư mục `dl/`):

| W | seed 20260917 | 20260918 | 20260919 | 20260920 | 20260921 |
|---|---|---|---|---|---|
| 60 | `dl/W60/seed20260917/done.json` | … | … | … | … |
| 30 | … | | | | |
| 120 | … | | | | |
| 90 | … | | | | |

Mỗi thư mục seed có `train_log.csv` (loss theo epoch), `best.pt`, `last.pt`; được đẩy lên khoảng 10 phút một lần.

### Bị ngắt giữa chừng? Chạy lại đúng notebook đó

Hết 12 giờ, lỗi mạng, hết quota… thì mở lại notebook và **Save & Run All** lần nữa, không sửa gì.
Code tự bỏ qua seed đã có `done.json`, tải `best.pt` + `last.pt` của seed đang dở từ HF và chạy tiếp
từ epoch đó (mất tối đa khoảng 10 phút train).

---

## Bước 5. `03_dl_finalize` (CPU, sau khi đủ 4 W × 5 seed)

`STEP = "03_dl_finalize"`, chỉ gắn `uc04-samples-v2`.

Đọc logit của từng seed từ HF, gộp 5 seed, hiệu chỉnh xác suất (một nhiệt độ và một bias chung), chọn ngưỡng
trên validation. **Xong khi** trên HF có `reports/dl_validation.csv` (bảng kết quả DL),
`reports/calibration_check_dl.csv`, `dl/W<W>/calibration.json` và `dl/W<W>/threshold.json`.

---

## Tùy chọn

Thêm vào sau `{STEP}` trong dòng `!python ...` (truyền cho script train); tham số của chính `kaggle_run.py`
(`--skip-install`, `--work`) đặt **trước** `{STEP}`.

| Dòng lệnh | Khi nào |
|---|---|
| `!python SafeAnes/scripts/kaggle_run.py {STEP} --gpus 1` | Chạy từng seed một trên 1 GPU (2 seed song song bị nghẽn CPU, hoặc máy 1 GPU) |
| `!python SafeAnes/scripts/kaggle_run.py {STEP} --seeds 20260917 20260918` | Chỉ train một số seed |
| `!python SafeAnes/scripts/kaggle_run.py --skip-install {STEP}` | Chạy bước thứ hai trong cùng một phiên (thư viện đã cài) |
| thêm `!cd SafeAnes && git fetch --depth 1 origin <commit> && git checkout FETCH_HEAD` sau lệnh clone | Chạy đúng một phiên bản code (commit được ghi vào `provenance.json` của mỗi lần train) |

---

## Lỗi thường gặp

| Thông báo trong log | Cách xử lý |
|---|---|
| `Kaggle Secret HF_TOKEN is missing or not enabled for this notebook` | *Add-ons → Secrets*, bật `HF_TOKEN` cho notebook này, chạy lại |
| `expected exactly one uc04-samples-v2 dataset ...` hoặc `... wave dataset (uc04-prep-v1)` | Thiếu hoặc gắn trùng dataset ở *Add Input* |
| `samples.json sha256 ... wrong uc04-samples-v2 version` | Gắn sai phiên bản dataset samples (so với `kaggle.samples_json_sha256` trong config) |
| `qc.csv sha256 ... wrong uc04-prep-v1 dataset` | Gắn sai phiên bản dataset sóng (so với `input.qc_sha256`) |
| `FAIL vs .../requirements.txt` | Thư viện thiếu hoặc cũ hơn mức tối thiểu; chạy lại, không dùng `--skip-install` |
| `... is PUBLIC but hub.private is true` | Repo HF đang public: *Settings → Change visibility* sang private |
| `/kaggle/input not found` | Đang chạy ngoài Kaggle; trên máy cá nhân gọi thẳng `scripts/train_dl.py` |
| `seed ... on gpu1 FAILED` | Xem các dòng `[gpu1 seed ...]` phía trên để biết lỗi; các seed khác không bị ảnh hưởng. Sửa rồi chạy lại notebook |
| `CUDA out of memory` | Chạy lại với `--gpus 1`; vẫn lỗi thì báo nhóm (không tự giảm batch size, kế hoạch cố định 256) |
| Notebook đứng lâu ở đầu, chưa có log | Kaggle đang chuẩn bị dataset sóng 7 GB (lần đầu có thể 10–15 phút) |

---

## Không được làm

- Không đưa dữ liệu **test** lên Kaggle, HF hay GitHub. Kaggle chỉ có train/calibration/validation;
  `hub.py` từ chối đẩy mọi file có `test`/`sealed`.
- Không sửa siêu tham số `dl.*` trong `configs/uc04_v2.json` để chạy nhanh hơn: kế hoạch đã chốt; mọi khác biệt
  ghi vào `docs/DEVIATIONS.md`.
- Không đưa token vào notebook hay code: chỉ để trong Kaggle Secrets và `.env` (đã git-ignore).

Các thư mục `00_env_check/`, `03_dl_*/` bên cạnh là notebook sinh tự động cho cách chạy cũ (code từ dataset
`uc04-code`); cách chạy trong file này không cần chúng.
