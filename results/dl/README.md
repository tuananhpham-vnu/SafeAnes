# Kết quả DL (Conv1D + Transformer)

**Chưa có.** Mô hình DL được train trên Kaggle bằng các notebook trong `notebooks/kaggle/` (`03_dl_smoke`, `03_dl_W60/W30/W120/W90`, `03_dl_finalize`). Kết quả tự đẩy lên repo Hugging Face private `yungdyo3112/uc04-v2-runs`:

- `dl/W<W>/seed<s>/`: `best.pt`, `last.pt`, `train_log.csv`, logit của calibration và validation;
- `dl/W<W>/`: logit trung bình của các seed, `calibration.json`, `threshold.json`;
- `reports/dl_validation.csv`, `reports/calibration_check_dl.csv`, `reports/dl_smoke/timing.json`.

Sau khi train xong, bảng tổng (không có mô hình hay dự đoán theo từng ca) sẽ được chép vào đây.
