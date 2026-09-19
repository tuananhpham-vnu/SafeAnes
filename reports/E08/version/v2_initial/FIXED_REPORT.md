> **Bản v2 đầu tiên — đã thay thế.** Sinh bởi `run_pipeline.py` và `run_method_comparison.py`, hai script này đã được gộp vào [`scripts/e08/run_comparison.py`](../../../../scripts/e08/run_comparison.py). Bản này còn một lỗi: `lgbm_oversample_3x` chỉ thêm 2 bản sao mỗi window dương trong khi jitter/SMOTE thêm 3. Số liệu hiện hành: [METHOD_COMPARISON.md](../../METHOD_COMPARISON.md).

# E08 — pipeline đã sửa lỗi (thay cho chuỗi Giai đoạn 1-4 cũ)

Audit lại các script E08 trước đó tìm được 5 lỗi làm sai lệch kết quả. File này chạy lại toàn bộ sau khi sửa cả 5. **Các bảng ở REPORT.md cũ (Giai đoạn 1-4) được tạo trước khi sửa, không dùng để kết luận nữa.**

## Năm lỗi đã sửa

| Lỗi | Ảnh hưởng đo được | Cách sửa |
|---|---|---|
| F1 — train/serve skew | Model augment train trên dữ liệu đã impute nhưng chấm điểm validation bằng feature thô còn NaN (3,7% dòng validation có NaN) | Một imputer fit trên FIT, áp dụng y hệt cho fit/calibration/validation |
| F2 — trộn thang xác suất | Ensemble trộn xác suất đã calibrate (mean≈0,018-0,020, khớp prevalence 0,0244) với predict_proba thô (catboost balanced mean=0,0871, cao gấp ~3,5 lần) | Calibrate mọi thành viên bằng cùng sigmoid-trên-log-odds, fit trên role calibration |
| F3 — gate tính trên vote | Vote fraction không phải xác suất: ECE=0,210 ở mọi mức vote trong khi gate yêu cầu <=0,05, nên gate ECE không bao giờ đạt được | Gate tính trên điểm ensemble liên tục đã calibrate |
| F4 — so sánh bị nhiễu | Baseline train dữ liệu thô có NaN còn jitter/SMOTE train dữ liệu đã impute — lẫn hai thay đổi vào một so sánh | Mọi biến thể dùng chung preprocessing |
| F5 — chọn ngưỡng vòng lặp | Ngưỡng chọn trên validation rồi báo cáo cũng trên validation, vốn chỉ có 23-24 biến cố (mỗi biến cố đổi recall 4,3 điểm) | Chọn ngưỡng trên role calibration (58 ca, 65-73 biến cố), báo cáo trên validation chưa dùng để chọn |

## Kiểm chứng F2 — mọi thành viên giờ cùng một thang xác suất

| Phút | Thành viên | Mean thô | Mean sau calibrate | Prevalence thật |
|---:|---|---:|---:|---:|
| 5 | lgbm_plain | 0.0329 | 0.0252 | 0.0251 |
| 5 | lgbm_jitter | 0.0347 | 0.0251 | 0.0251 |
| 5 | catboost_balanced_jitter | 0.1116 | 0.0252 | 0.0251 |
| 5 | logistic_map | 0.0345 | 0.0251 | 0.0251 |
| 10 | lgbm_plain | 0.0646 | 0.0553 | 0.0553 |
| 10 | lgbm_jitter | 0.0682 | 0.0553 | 0.0553 |
| 10 | catboost_balanced_jitter | 0.1414 | 0.0553 | 0.0553 |
| 10 | logistic_map | 0.0631 | 0.0553 | 0.0553 |

Mean sau calibrate bám sát prevalence ở mọi thành viên — điều kiện để phép trung bình cộng xác suất có nghĩa. Bản cũ không có bước này.

## Kết quả trên validation (ngưỡng đã khóa bằng calibration, không nhìn validation)

| Phút | Ngưỡng | Recall (CI95) | PPV (CI95) | FA/giờ (CI95) | AUROC | ECE | Gate đạt | Gate còn thiếu |
|---:|---:|---|---|---|---:|---:|---|---|
| 5 | 0.136 | 0.478 (0.200–0.723) | 0.393 (0.226–0.632) | 0.297 (0.088–0.577) | 0.885 | 0.004 | 2/6 | auroc,event_sensitivity,alarm_ppv,prediction_coverage |
| 10 | 0.175 | 0.417 (0.117–0.696) | 0.407 (0.234–0.601) | 0.296 (0.066–0.602) | 0.821 | 0.015 | 2/7 | auroc,event_sensitivity,alarm_ppv,prediction_coverage,early_5min_sensitivity |

Lưu ý đọc CI: FA/giờ đạt ngân sách 0,5 ở ước lượng điểm nhưng **cận trên CI95 vượt 0,5** ở cả hai horizon — chưa thỏa tiêu chí risk-controlling (đòi hỏi cả cận trên nằm trong ngân sách). Với 23-24 biến cố trên validation, CI còn rất rộng.

## Mặt Pareto trên validation (các điểm không bị điểm nào trội hơn về cả 3 chỉ số)

| Phút | Ngưỡng | Recall | PPV | FA/giờ | Gate đạt |
|---:|---:|---:|---:|---:|---|
| 5 | 0.046 | 0.826 | 0.160 | 1.745 | 1/6 |
| 5 | 0.058 | 0.783 | 0.209 | 1.186 | 1/6 |
| 5 | 0.063 | 0.739 | 0.239 | 0.942 | 1/6 |
| 5 | 0.065 | 0.696 | 0.235 | 0.907 | 1/6 |
| 5 | 0.070 | 0.609 | 0.222 | 0.855 | 1/6 |
| 5 | 0.073 | 0.609 | 0.222 | 0.855 | 1/6 |
| 5 | 0.099 | 0.565 | 0.295 | 0.541 | 1/6 |
| 5 | 0.119 | 0.522 | 0.375 | 0.349 | 2/6 |
| 5 | 0.136 | 0.478 | 0.393 | 0.297 | 2/6 |
| 5 | 0.139 | 0.478 | 0.393 | 0.297 | 2/6 |
| 10 | 0.054 | 1.000 | 0.170 | 2.798 | 2/7 |
| 10 | 0.056 | 0.958 | 0.167 | 2.779 | 2/7 |
| 10 | 0.077 | 0.917 | 0.200 | 2.075 | 2/7 |
| 10 | 0.079 | 0.875 | 0.203 | 1.964 | 2/7 |
| 10 | 0.089 | 0.833 | 0.228 | 1.630 | 1/7 |
| 10 | 0.090 | 0.833 | 0.220 | 1.575 | 1/7 |
| 10 | 0.096 | 0.792 | 0.230 | 1.427 | 1/7 |
| 10 | 0.104 | 0.750 | 0.261 | 1.204 | 1/7 |
| 10 | 0.112 | 0.708 | 0.288 | 0.963 | 1/7 |
| 10 | 0.118 | 0.667 | 0.308 | 0.834 | 1/7 |

Mặt Pareto thay cho việc tranh luận một điểm 'tốt nhất': mỗi dòng là một đánh đổi không bị dominated, chọn dòng nào là quyết định lâm sàng chứ không phải quyết định kỹ thuật.

[Tạo bởi scripts/e08/run_comparison.py](../../../../scripts/e08/run_comparison.py) — development300, không mở pilot_test/global test, không đổi artifact E05/E06 đã khóa.
