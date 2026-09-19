# Chạy E04 (v0.2)

Đầu vào: `data/pilot_v1`, predictions/roles/environment của `artifacts/tcn_v1` và `artifacts/v0_3/ensemble`. Cài dependencies của project và extra `catboost`; version đã đo là CatBoost 1.2.10. Cấu hình CPU, 2 threads.

```powershell
$env:PYTHONPATH = "$PWD/src;$PWD/.local_deps"
python scripts/run_version04.py
python scripts/summarize_version04.py
python scripts/verify_version04.py
python -m pytest -q
```

Runner đăng ký source/data/input hashes trước thí nghiệm, kiểm tra roles với v0.2, không ghi đè kết quả từng horizon đã hoàn thành. Nếu code/plan/input thay đổi thì dùng version mới hoặc tái lập trong bản source snapshot tương ứng. Không xóa registration để lách kiểm tra.

- `artifacts/v0_4/registration.json`, `source/`: thông số, dependencies, thời gian UTC và source trước chạy.
- `artifacts/v0_4/<run>/<horizon>_model.joblib`: classifier, fit-only imputer, calibration-patient sigmoid, feature names, protocol/data hashes.
- `*_validation.csv.gz`, `*_test.csv.gz`: predictions đủ decision grid, bao gồm ineligible để reset alarm.
- `*_validation_curve.json`: mọi ngưỡng và metric validation; không chọn ngưỡng bằng test.
- `*_results.json`: ngưỡng từng policy, metric test, CI bootstrap bệnh nhân và gates.
- `reports/v0_4/`: báo cáo tổng hợp, khoảng dao động ba seed, kiểm tra tái tạo xác suất từ model đã lưu.

Hai horizon CatBoost được train/calibrate độc lập như baseline tabular, chưa áp đặt thứ tự p10 >= p5 như sequence head. `verification.json` đếm vi phạm; không mô tả đầu ra là risk có thứ tự. Không lấy checkpoint có điểm test cao nhất để triển khai.

[Thiết kế khóa trước chạy](V0_4_PLAN.md) · [Báo cáo](../../reports/v0_4/REPORT.md) · [Nhận xét](../../reports/v0_4/REPORT.md#nhan-xet)
