# Notebooks

Notebook chỉ gọi script trong `scripts/`; mọi logic nằm trong `src/uc04/`. Có hai nhóm.

## `local/`: chạy trên máy cá nhân

Mở bằng Jupyter với venv của repo (`.venv`).

| Notebook | Việc | Ghi chú |
|---|---|---|
| `01_data_pipeline.ipynb` | Kiểm tra `prep_v1`, thử nhãn, (dựng samples), đóng gói cho Kaggle, kiểm tra độc lập | Bước dựng lại `samples_v2` **khóa mặc định** (`REBUILD_SAMPLES = False`) |
| `02_tabular.ipynb` | Train 65 tổ hợp tabular, hai chính sách cảnh báo, phân tích (W\*, SHAP, ablation, mức tin cậy), so sánh ghép cặp, cửa sổ dài (thăm dò), xuất kết quả sang `results/tabular/` | Khoảng 45 phút |
| `04_lock_and_test.ipynb` | Khóa mô hình và chạy test (NB04) | **Chưa chạy.** Chỉ chạy khi nhóm đã chốt cấu hình |

## `kaggle/`: chạy trên Kaggle (phần DL)

Sinh bằng `scripts/make_kaggle_notebooks.py` từ `templates/` cho đúng commit của dataset `uc04-code` (ghi trong `CODE_COMMIT` của mỗi notebook). Mỗi thư mục có file `.ipynb` và `kernel-metadata.json`.

| Thứ tự | Notebook | Máy | Việc |
|---|---|---|---|
| 1 | `00_env_check` | CPU | Kiểm tra input đúng phiên bản (commit code, sha256 `samples.json` và `qc.csv`), thư viện, Hugging Face |
| 2 | `03_dl_smoke` | GPU | 1 epoch ở W = 60: thời gian, RAM, VRAM, ước tính giờ GPU |
| 3–6 | `03_dl_W60`, `03_dl_W30`, `03_dl_W120`, `03_dl_W90` | GPU | Conv1D + Transformer, 5 seed mỗi cửa sổ; tự đẩy kết quả lên HF và tự chạy tiếp nếu phiên bị dừng |
| 7 | `03_dl_finalize` | CPU | Ensemble seed, hiệu chỉnh xác suất, chọn ngưỡng, bảng kết quả DL |

Input mỗi notebook (Add Input): `datnguyen31112/uc04-code`, `datnguyen31112/uc04-samples-v2`, và với notebook cần sóng, `datnguyen31112/uc04-prep-v1`. Cần Secret `HF_TOKEN` (Add-ons → Secrets) và bật Internet. Thiếu một thứ là ô setup dừng ngay.

Chạy thử trên máy (tập con, không đẩy): `python scripts/make_kaggle_notebooks.py --run-local`.
