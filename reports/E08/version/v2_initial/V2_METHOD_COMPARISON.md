> **Bản v2 đầu tiên — đã thay thế.** Sinh bởi `run_pipeline.py` và `run_method_comparison.py`, hai script này đã được gộp vào [`scripts/e08/run_comparison.py`](../../../../scripts/e08/run_comparison.py). Bản này còn một lỗi: `lgbm_oversample_3x` chỉ thêm 2 bản sao mỗi window dương trong khi jitter/SMOTE thêm 3. Số liệu hiện hành: [METHOD_COMPARISON.md](../../METHOD_COMPARISON.md).

# E08 v2 — so sánh phương pháp trên cùng một quy trình

Mọi phương pháp dùng chung: một imputer fit trên FIT, một thủ tục calibration (sigmoid trên log-odds, fit trên CALIBRATION), cùng lưới ngưỡng. **Ngưỡng chọn trên CALIBRATION** (58 ca, 65-73 biến cố) và **báo cáo trên VALIDATION** (24 ca, 23-24 biến cố) — validation không tham gia chọn ngưỡng. Khác biệt duy nhất giữa hai dòng là bản thân phương pháp.

Chạy hết 2334 giây. Vẫn là development300; chưa mở pilot_test/global test.

## Chính sách recall_first (đã chấp nhận mức FA đi kèm)

### 5 phút — 23 biến cố eligible trên validation

| Phương pháp | Nhóm | Recall (CI95) | Bắt được | PPV | FA/giờ | AUROC | AP | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `map_logistic` | baseline | 0.826 (0.641–0.953) | 19/23 | 0.216 | 1.204 | 0.881 | 0.195 | 0.004 |
| `catboost_balanced` | catboost | 0.826 (0.625–1.000) | 19/23 | 0.135 | 2.129 | 0.855 | 0.213 | 0.006 |
| `lgbm_oversample_3x` | lightgbm | 0.783 (0.588–0.913) | 18/23 | 0.129 | 2.129 | 0.859 | 0.200 | 0.005 |
| `catboost_balanced_jitter` | catboost | 0.783 (0.588–0.913) | 18/23 | 0.129 | 2.129 | 0.874 | 0.205 | 0.005 |
| `catboost_plain` | catboost | 0.783 (0.555–0.924) | 18/23 | 0.217 | 1.134 | 0.892 | 0.262 | 0.007 |
| `lgbm_plain` | lightgbm | 0.783 (0.529–0.958) | 18/23 | 0.136 | 1.989 | 0.872 | 0.216 | 0.004 |
| `tabm_20260919` | tabm | 0.739 (0.500–0.905) | 17/23 | 0.132 | 1.954 | 0.878 | 0.225 | 0.007 |
| `lgbm_jitter_3x` | lightgbm | 0.739 (0.500–0.893) | 17/23 | 0.167 | 1.483 | 0.869 | 0.215 | 0.004 |
| `lgbm_class_weight` | lightgbm | 0.739 (0.571–0.889) | 17/23 | 0.110 | 2.408 | 0.861 | 0.217 | 0.006 |
| `lgbm_scale_pos_20x` | lightgbm | 0.696 (0.499–0.883) | 16/23 | 0.098 | 2.565 | 0.855 | 0.197 | 0.005 |
| `lgbm_smote_3x` | lightgbm | 0.696 (0.470–0.870) | 16/23 | 0.160 | 1.466 | 0.867 | 0.209 | 0.004 |
| `tabm_20260917` | tabm | 0.696 (0.438–0.875) | 16/23 | 0.112 | 2.216 | 0.865 | 0.216 | 0.007 |
| `tabm_20260918` | tabm | 0.696 (0.438–0.875) | 16/23 | 0.119 | 2.059 | 0.865 | 0.201 | 0.006 |
| `ensemble_diverse_6` | ensemble | 0.652 (0.357–0.905) | 15/23 | 0.214 | 0.960 | 0.883 | 0.234 | 0.005 |

### 10 phút — 24 biến cố eligible trên validation

| Phương pháp | Nhóm | Recall (CI95) | Bắt được | PPV | FA/giờ | AUROC | AP | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `map_logistic` | baseline | 0.917 (0.733–1.000) | 22/24 | 0.284 | 1.260 | 0.804 | 0.222 | 0.028 |
| `lgbm_oversample_3x` | lightgbm | 0.917 (0.733–1.000) | 22/24 | 0.174 | 2.631 | 0.834 | 0.205 | 0.012 |
| `catboost_balanced_jitter` | catboost | 0.917 (0.733–1.000) | 22/24 | 0.188 | 2.483 | 0.822 | 0.198 | 0.011 |
| `lgbm_plain` | lightgbm | 0.875 (0.700–1.000) | 21/24 | 0.173 | 2.390 | 0.823 | 0.188 | 0.009 |
| `catboost_plain` | catboost | 0.875 (0.687–1.000) | 21/24 | 0.165 | 2.538 | 0.832 | 0.194 | 0.012 |
| `lgbm_scale_pos_20x` | lightgbm | 0.875 (0.687–1.000) | 21/24 | 0.144 | 3.094 | 0.806 | 0.218 | 0.008 |
| `ensemble_diverse_6` | ensemble | 0.875 (0.700–1.000) | 21/24 | 0.235 | 1.630 | 0.828 | 0.235 | 0.012 |
| `tabm_20260919` | tabm | 0.875 (0.667–1.000) | 21/24 | 0.185 | 2.205 | 0.826 | 0.245 | 0.009 |
| `catboost_balanced` | catboost | 0.875 (0.688–0.973) | 21/24 | 0.171 | 2.427 | 0.795 | 0.177 | 0.007 |
| `tabm_20260917` | tabm | 0.875 (0.700–1.000) | 21/24 | 0.200 | 1.927 | 0.813 | 0.227 | 0.006 |
| `tabm_20260918` | tabm | 0.833 (0.625–0.963) | 20/24 | 0.210 | 1.742 | 0.816 | 0.220 | 0.006 |
| `lgbm_class_weight` | lightgbm | 0.833 (0.621–0.968) | 20/24 | 0.163 | 2.575 | 0.825 | 0.198 | 0.011 |
| `lgbm_jitter_3x` | lightgbm | 0.792 (0.599–0.926) | 19/24 | 0.181 | 2.094 | 0.821 | 0.186 | 0.007 |
| `lgbm_smote_3x` | lightgbm | 0.750 (0.545–0.905) | 18/24 | 0.200 | 1.779 | 0.817 | 0.180 | 0.007 |

## Cái giá của ràng buộc FA/giờ ≤ 0,5 (chính sách fa_budget, cùng model)

| Phương pháp | Phút | Recall recall_first | Recall fa_budget | Δ recall | PPV fa_budget | FA/giờ fa_budget |
|---|---:|---:|---:|---:|---:|---:|
| `map_logistic` | 5 | 0.826 | 0.217 | -0.609 | 0.294 | 0.209 |
| `catboost_balanced` | 5 | 0.826 | 0.435 | -0.391 | 0.345 | 0.331 |
| `lgbm_oversample_3x` | 5 | 0.783 | 0.435 | -0.348 | 0.370 | 0.297 |
| `catboost_balanced_jitter` | 5 | 0.783 | 0.348 | -0.435 | 0.276 | 0.366 |
| `catboost_plain` | 5 | 0.783 | 0.478 | -0.304 | 0.500 | 0.192 |
| `lgbm_plain` | 5 | 0.783 | 0.435 | -0.348 | 0.385 | 0.279 |
| `tabm_20260919` | 5 | 0.739 | 0.261 | -0.478 | 0.400 | 0.157 |
| `lgbm_jitter_3x` | 5 | 0.739 | 0.435 | -0.304 | 0.345 | 0.331 |
| `lgbm_class_weight` | 5 | 0.739 | 0.304 | -0.435 | 0.318 | 0.262 |
| `lgbm_scale_pos_20x` | 5 | 0.696 | 0.348 | -0.348 | 0.308 | 0.314 |
| `lgbm_smote_3x` | 5 | 0.696 | 0.391 | -0.304 | 0.346 | 0.297 |
| `tabm_20260917` | 5 | 0.696 | 0.261 | -0.435 | 0.400 | 0.157 |
| `tabm_20260918` | 5 | 0.696 | 0.304 | -0.391 | 0.389 | 0.192 |
| `ensemble_diverse_6` | 5 | 0.652 | 0.391 | -0.261 | 0.391 | 0.244 |
| `map_logistic` | 10 | 0.917 | 0.250 | -0.667 | 0.273 | 0.296 |
| `lgbm_oversample_3x` | 10 | 0.917 | 0.417 | -0.500 | 0.379 | 0.333 |
| `catboost_balanced_jitter` | 10 | 0.917 | 0.333 | -0.583 | 0.320 | 0.315 |
| `lgbm_plain` | 10 | 0.875 | 0.375 | -0.500 | 0.370 | 0.315 |
| `catboost_plain` | 10 | 0.875 | 0.292 | -0.583 | 0.269 | 0.352 |
| `lgbm_scale_pos_20x` | 10 | 0.875 | 0.333 | -0.542 | 0.360 | 0.296 |
| `ensemble_diverse_6` | 10 | 0.875 | 0.458 | -0.417 | 0.414 | 0.315 |
| `tabm_20260919` | 10 | 0.875 | 0.292 | -0.583 | 0.421 | 0.204 |
| `catboost_balanced` | 10 | 0.875 | 0.375 | -0.500 | 0.321 | 0.352 |
| `tabm_20260917` | 10 | 0.875 | 0.333 | -0.542 | 0.421 | 0.204 |
| `tabm_20260918` | 10 | 0.833 | 0.333 | -0.500 | 0.364 | 0.259 |
| `lgbm_class_weight` | 10 | 0.833 | 0.333 | -0.500 | 0.474 | 0.185 |
| `lgbm_jitter_3x` | 10 | 0.792 | 0.375 | -0.417 | 0.345 | 0.352 |
| `lgbm_smote_3x` | 10 | 0.750 | 0.375 | -0.375 | 0.357 | 0.333 |

## Kiểm chứng calibration — điều kiện để so sánh công bằng

Mean xác suất sau calibrate phải bám prevalence thật; nếu lệch thì mọi so sánh ngưỡng và mọi phép trung bình ensemble đều sai (đây chính là lỗi F2 của v1).

| Phương pháp | Phút | Mean thô | Mean sau calibrate | Prevalence |
|---|---:|---:|---:|---:|
| `map_logistic` | 5 | 0.0345 | 0.0251 | 0.0251 |
| `lgbm_plain` | 5 | 0.0329 | 0.0252 | 0.0251 |
| `lgbm_class_weight` | 5 | 0.2229 | 0.0252 | 0.0251 |
| `lgbm_scale_pos_20x` | 5 | 0.4186 | 0.0251 | 0.0251 |
| `lgbm_oversample_3x` | 5 | 0.0697 | 0.0251 | 0.0251 |
| `lgbm_smote_3x` | 5 | 0.0339 | 0.0251 | 0.0251 |
| `lgbm_jitter_3x` | 5 | 0.0347 | 0.0251 | 0.0251 |
| `catboost_plain` | 5 | 0.0351 | 0.0252 | 0.0251 |
| `catboost_balanced` | 5 | 0.1834 | 0.0252 | 0.0251 |
| `catboost_balanced_jitter` | 5 | 0.1116 | 0.0252 | 0.0251 |
| `tabm_20260917` | 5 | 0.0348 | 0.0251 | 0.0251 |
| `tabm_20260918` | 5 | 0.0337 | 0.0251 | 0.0251 |
| `tabm_20260919` | 5 | 0.0384 | 0.0251 | 0.0251 |
| `ensemble_diverse_6` | 5 | 0.0251 | 0.0251 | 0.0251 |
| `map_logistic` | 10 | 0.0631 | 0.0553 | 0.0553 |
| `lgbm_plain` | 10 | 0.0646 | 0.0553 | 0.0553 |
| `lgbm_class_weight` | 10 | 0.2922 | 0.0553 | 0.0553 |
| `lgbm_scale_pos_20x` | 10 | 0.5408 | 0.0553 | 0.0553 |
| `lgbm_oversample_3x` | 10 | 0.1311 | 0.0553 | 0.0553 |
| `lgbm_smote_3x` | 10 | 0.0674 | 0.0553 | 0.0553 |
| `lgbm_jitter_3x` | 10 | 0.0676 | 0.0553 | 0.0553 |
| `catboost_plain` | 10 | 0.0665 | 0.0553 | 0.0553 |
| `catboost_balanced` | 10 | 0.2417 | 0.0553 | 0.0553 |
| `catboost_balanced_jitter` | 10 | 0.1436 | 0.0552 | 0.0553 |
| `tabm_20260917` | 10 | 0.0629 | 0.0553 | 0.0553 |
| `tabm_20260918` | 10 | 0.0644 | 0.0552 | 0.0553 |
| `tabm_20260919` | 10 | 0.0697 | 0.0552 | 0.0553 |
| `ensemble_diverse_6` | 10 | 0.0553 | 0.0553 | 0.0553 |

## Cách đọc bảng

- **Recall là chỉ tiêu ưu tiên số một** theo yêu cầu, FA/giờ đi kèm đã được chấp nhận ở vòng này; cột `fa_budget` chỉ để thấy phải trả bao nhiêu recall nếu siết FA về 0,5.
- CI95 bootstrap theo bệnh nhân (200 lần, unit=subjectid). Với 23-24 biến cố, CI rất rộng: chênh lệch recall nhỏ hơn ~1 biến cố (4,2-4,3 điểm) **không** kết luận được phương pháp nào hơn.
- AUROC/AP/ECE không phụ thuộc ngưỡng nên giống nhau ở cả hai chính sách của cùng một model; dùng chúng để so khả năng phân biệt độc lập với điểm vận hành.
- Đây là development validation, **không phải** bằng chứng xác nhận độc lập; chưa mở pilot_test/global test.

[Tạo bởi scripts/e08/run_comparison.py](../../../../scripts/e08/run_comparison.py) · [Lịch sử version và các lỗi của v1](../../../../scripts/e08/README.md)
