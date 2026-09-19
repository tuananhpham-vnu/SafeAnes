# E08 v2 — so sánh phương pháp trên development300

Chạy 2683 giây · TabM: backbone đông lạnh từ E06 (train trên development300), chỉ calibrate lại. [Tóm tắt trong README](../../README.md#e08) · [các version](../../scripts/e08/README.md) · [script](../../scripts/e08/run_comparison.py)

### Phương pháp được chọn

- **5 phút: `lgbm_oversample_3x`** — VALIDATION recall 0.783 (18/23 biến cố), PPV 0.131, FA 2.076/giờ, AUROC 0.866; đạt 1/6 gate, còn thiếu: `auroc`, `event_sensitivity`, `alarm_ppv`, `prediction_coverage`, `false_alarms_per_hour`.
  - Trên CALIBRATION **hòa recall** với `lgbm_plain` (44/65 biến cố), chỉ phân định bằng PPV 0.1239 so với 0.1236 — thực chất **chưa phân biệt được** các phương pháp dẫn đầu ở horizon này.
  - Trên VALIDATION, recall cao nhất lại là `map_logistic` (19/23, PPV 0.216, FA 1.204/giờ) — thứ hạng giữa hai tập độc lập chưa ổn định.
- **10 phút: `map_logistic`** — VALIDATION recall 0.917 (22/24 biến cố), PPV 0.284, FA 1.260/giờ, AUROC 0.804; đạt 2/7 gate, còn thiếu: `auroc`, `alarm_ppv`, `prediction_coverage`, `early_5min_sensitivity`, `false_alarms_per_hour`.
  - Trên CALIBRATION bắt 55/73 biến cố, hơn á quân `lgbm_plain` 2 biến cố.
- Ở 10 phút, baseline MAP đứng đầu trên **cả** CALIBRATION (dùng để chọn) lẫn VALIDATION (dùng để báo cáo): model phức tạp chưa cho thấy lợi ích trên mẫu này — cùng hướng với [Mulder 2024](https://pubmed.ncbi.nlm.nih.gov/38558038/) (HPI ≈ ngưỡng MAP).

### Tham số chung (áp dụng cho mọi phương pháp)

| Tham số | Giá trị |
|---|---|
| Dữ liệu | development300 (300 ca của E05), chia theo `subjectid` |
| Ca / bệnh nhân | FIT 170 ca / 168 BN · CALIBRATION 58 ca / 57 BN · VALIDATION 24 ca / 24 BN |
| Biến cố eligible (5 / 10 phút) | FIT 211/229 · CALIBRATION 65/73 · VALIDATION 23/24 |
| Định nghĩa biến cố | MAP < 65 mmHg liên tục ≥ 60 s |
| Đặc trưng | 140 numeric (bỏ static); `map_logistic` chỉ dùng `map_current`, `map_300_slope`, `map_300_std` |
| Imputation | median từng cột, tính trên FIT, điền y hệt cho FIT / CALIBRATION / VALIDATION |
| Calibration | StandardScaler → LogisticRegression `C=1e+06, max_iter=2000` trên log-odds của điểm thô; fit trên CALIBRATION, riêng từng phương pháp × horizon |
| TabM | backbone đông lạnh từ E06 (train trên development300), chỉ calibrate lại |
| Lưới ngưỡng | 40 điểm cố định 0,01–0,99 + 201 quantile xác suất trên CALIBRATION |
| Chọn ngưỡng | `recall_first` trên CALIBRATION: recall cao nhất → PPV cao hơn → FA/giờ thấp hơn |
| Chọn phương pháp | trên CALIBRATION, cùng thứ tự `recall_first` |
| Báo cáo | VALIDATION — không tham gia chọn ngưỡng hay chọn phương pháp |
| Chính sách cảnh báo | 2 decision liên tiếp vượt ngưỡng · cooldown 300 s · nhịp 30 s |
| Khoảng tin cậy | bootstrap theo `subjectid`, 200 lần |
| Seed | model 20260917 · augmentation 20260919 |

### Tham số từng phương pháp

| Phương pháp | Nhóm | Đặc trưng | Siêu tham số | Cân bằng lớp | Augmentation | Window dương khi train (5 / 10 phút) |
|---|---|---|---|---|---|---|
| `map_logistic` | baseline | 3 (MAP) | StandardScaler → LogisticRegression `C=1, max_iter=2000` | không | không | 1451 / 2566 |
| `lgbm_plain` | LightGBM | 140 | `n_estimators=500, num_leaves=7, learning_rate=0.03, min_child_samples=150, reg_lambda=10`; đơn điệu giảm theo `map_current`, `map_60_mean`, `map_300_mean` | không | không | 1451 / 2566 |
| `lgbm_class_weight` | LightGBM | 140 | `n_estimators=500, num_leaves=7, learning_rate=0.03, min_child_samples=150, reg_lambda=10`; đơn điệu giảm theo `map_current`, `map_60_mean`, `map_300_mean` | `class_weight='balanced'` | không | 1451 / 2566 |
| `lgbm_scale_pos_20x` | LightGBM | 140 | `n_estimators=500, num_leaves=7, learning_rate=0.03, min_child_samples=150, reg_lambda=10`; đơn điệu giảm theo `map_current`, `map_60_mean`, `map_300_mean` | `scale_pos_weight` = 20 × âm/dương = 611 / 320 | không | 1451 / 2566 |
| `lgbm_oversample_3x` | LightGBM | 140 | `n_estimators=500, num_leaves=7, learning_rate=0.03, min_child_samples=150, reg_lambda=10`; đơn điệu giảm theo `map_current`, `map_60_mean`, `map_300_mean` | không | nhân bản nguyên văn, +3 bản / window dương | 5804 / 10264 |
| `lgbm_smote_3x` | LightGBM | 140 | `n_estimators=500, num_leaves=7, learning_rate=0.03, min_child_samples=150, reg_lambda=10`; đơn điệu giảm theo `map_current`, `map_60_mean`, `map_300_mean` | không | SMOTE 5 láng giềng, +3 mẫu / window dương | 5804 / 10264 |
| `lgbm_jitter_3x` | LightGBM | 140 | `n_estimators=500, num_leaves=7, learning_rate=0.03, min_child_samples=150, reg_lambda=10`; đơn điệu giảm theo `map_current`, `map_60_mean`, `map_300_mean` | không | nhiễu Gaussian σ = 0.1 × std, +3 bản / window dương | 5804 / 10264 |
| `catboost_plain` | CatBoost | 140 | `iterations=300, depth=5, learning_rate=0.05, l2_leaf_reg=5` | không | không | 1451 / 2566 |
| `catboost_balanced` | CatBoost | 140 | `iterations=300, depth=5, learning_rate=0.05, l2_leaf_reg=5` | `auto_class_weights='Balanced'` | không | 1451 / 2566 |
| `catboost_balanced_jitter` | CatBoost | 140 | `iterations=300, depth=5, learning_rate=0.05, l2_leaf_reg=5` | `auto_class_weights='Balanced'` | nhiễu Gaussian σ = 0.1 × std, +3 bản / window dương | 5804 / 10264 |
| `tabm_20260917` | TabM | 140 | `k=16, width=64, epochs=30, patience=6, batch_size=512`; `n_blocks=2, dropout=0.1, PLE 8 bins x 4, AdamW lr=0.002 wd=3e-4`; dừng ở epoch 5 (backbone đông lạnh từ E06 (train trên development300), chỉ calibrate lại) | không | không | — |
| `tabm_20260918` | TabM | 140 | `k=16, width=64, epochs=30, patience=6, batch_size=512`; `n_blocks=2, dropout=0.1, PLE 8 bins x 4, AdamW lr=0.002 wd=3e-4`; dừng ở epoch 5 (backbone đông lạnh từ E06 (train trên development300), chỉ calibrate lại) | không | không | — |
| `tabm_20260919` | TabM | 140 | `k=16, width=64, epochs=30, patience=6, batch_size=512`; `n_blocks=2, dropout=0.1, PLE 8 bins x 4, AdamW lr=0.002 wd=3e-4`; dừng ở epoch 5 (backbone đông lạnh từ E06 (train trên development300), chỉ calibrate lại) | không | không | — |
| `ensemble_diverse` | ensemble | — | trung bình xác suất đã calibrate của `map_logistic`, `lgbm_jitter_3x`, `catboost_balanced_jitter`, `tabm_20260917`, `tabm_20260918`, `tabm_20260919` | — | — | — |

### Kết quả 5 phút — `recall_first`, VALIDATION (23 biến cố)

| Phương pháp | Recall (CI95) | Bắt được | PPV (CI95) | FA/giờ (CI95) | AUROC | AP | ECE |
|---|---|---:|---|---|---:|---:|---:|
| `map_logistic` | 0.826 (0.641–0.953) | 19/23 | 0.216 (0.133–0.311) | 1.204 (0.809–1.694) | 0.881 | 0.195 | 0.004 |
| `catboost_balanced` | 0.826 (0.625–1.000) | 19/23 | 0.135 (0.059–0.230) | 2.129 (1.418–3.020) | 0.855 | 0.213 | 0.006 |
| `catboost_plain` | 0.783 (0.555–0.924) | 18/23 | 0.217 (0.118–0.315) | 1.134 (0.643–1.689) | 0.892 | 0.262 | 0.007 |
| `lgbm_plain` | 0.783 (0.529–0.958) | 18/23 | 0.136 (0.059–0.248) | 1.989 (1.322–2.734) | 0.872 | 0.216 | 0.004 |
| `lgbm_oversample_3x` ★ | 0.783 (0.588–0.913) | 18/23 | 0.131 (0.064–0.207) | 2.076 (1.352–2.903) | 0.866 | 0.209 | 0.005 |
| `catboost_balanced_jitter` | 0.783 (0.588–0.913) | 18/23 | 0.129 (0.062–0.212) | 2.129 (1.381–2.933) | 0.874 | 0.205 | 0.005 |
| `lgbm_jitter_3x` | 0.739 (0.500–0.893) | 17/23 | 0.167 (0.073–0.263) | 1.483 (1.000–2.060) | 0.869 | 0.215 | 0.004 |
| `tabm_20260919` | 0.739 (0.500–0.905) | 17/23 | 0.132 (0.058–0.223) | 1.954 (1.201–2.714) | 0.878 | 0.225 | 0.007 |
| `lgbm_class_weight` | 0.739 (0.571–0.889) | 17/23 | 0.110 (0.054–0.189) | 2.408 (1.649–3.255) | 0.861 | 0.217 | 0.006 |
| `lgbm_smote_3x` | 0.696 (0.470–0.870) | 16/23 | 0.160 (0.074–0.257) | 1.466 (0.969–2.010) | 0.867 | 0.209 | 0.004 |
| `tabm_20260918` | 0.696 (0.438–0.875) | 16/23 | 0.119 (0.055–0.198) | 2.059 (1.271–2.820) | 0.865 | 0.201 | 0.006 |
| `tabm_20260917` | 0.696 (0.438–0.875) | 16/23 | 0.112 (0.048–0.191) | 2.216 (1.375–3.167) | 0.865 | 0.216 | 0.007 |
| `lgbm_scale_pos_20x` | 0.696 (0.499–0.883) | 16/23 | 0.098 (0.045–0.164) | 2.565 (1.713–3.512) | 0.855 | 0.197 | 0.005 |
| `ensemble_diverse` | 0.652 (0.357–0.905) | 15/23 | 0.214 (0.103–0.333) | 0.960 (0.491–1.421) | 0.883 | 0.234 | 0.005 |

### Kết quả 10 phút — `recall_first`, VALIDATION (24 biến cố)

| Phương pháp | Recall (CI95) | Bắt được | PPV (CI95) | FA/giờ (CI95) | AUROC | AP | ECE |
|---|---|---:|---|---|---:|---:|---:|
| `map_logistic` ★ | 0.917 (0.733–1.000) | 22/24 | 0.284 (0.162–0.443) | 1.260 (0.813–1.774) | 0.804 | 0.222 | 0.028 |
| `catboost_balanced_jitter` | 0.917 (0.733–1.000) | 22/24 | 0.188 (0.110–0.321) | 2.483 (1.704–3.424) | 0.822 | 0.198 | 0.011 |
| `ensemble_diverse` | 0.875 (0.700–1.000) | 21/24 | 0.235 (0.142–0.365) | 1.630 (1.021–2.324) | 0.828 | 0.235 | 0.012 |
| `tabm_20260917` | 0.875 (0.700–1.000) | 21/24 | 0.200 (0.113–0.325) | 1.927 (1.145–2.787) | 0.813 | 0.227 | 0.006 |
| `tabm_20260919` | 0.875 (0.667–1.000) | 21/24 | 0.185 (0.096–0.331) | 2.205 (1.330–3.110) | 0.826 | 0.245 | 0.009 |
| `lgbm_plain` | 0.875 (0.700–1.000) | 21/24 | 0.173 (0.106–0.283) | 2.390 (1.661–3.362) | 0.823 | 0.188 | 0.009 |
| `catboost_balanced` | 0.875 (0.688–0.973) | 21/24 | 0.171 (0.100–0.286) | 2.427 (1.589–3.350) | 0.795 | 0.177 | 0.007 |
| `lgbm_oversample_3x` | 0.875 (0.687–1.000) | 21/24 | 0.166 (0.093–0.269) | 2.705 (1.918–3.540) | 0.825 | 0.193 | 0.011 |
| `catboost_plain` | 0.875 (0.687–1.000) | 21/24 | 0.165 (0.090–0.261) | 2.538 (1.815–3.381) | 0.832 | 0.194 | 0.012 |
| `lgbm_scale_pos_20x` | 0.875 (0.687–1.000) | 21/24 | 0.144 (0.072–0.235) | 3.094 (2.142–4.106) | 0.806 | 0.218 | 0.008 |
| `tabm_20260918` | 0.833 (0.625–0.963) | 20/24 | 0.210 (0.108–0.352) | 1.742 (1.020–2.497) | 0.816 | 0.220 | 0.006 |
| `lgbm_class_weight` | 0.833 (0.621–0.968) | 20/24 | 0.163 (0.092–0.268) | 2.575 (1.808–3.470) | 0.825 | 0.198 | 0.011 |
| `lgbm_jitter_3x` | 0.792 (0.599–0.926) | 19/24 | 0.181 (0.105–0.288) | 2.094 (1.344–2.966) | 0.821 | 0.186 | 0.007 |
| `lgbm_smote_3x` | 0.750 (0.545–0.905) | 18/24 | 0.200 (0.116–0.303) | 1.779 (1.190–2.571) | 0.817 | 0.180 | 0.007 |

**Đọc bảng:**

- ★ = phương pháp chọn trên CALIBRATION; VALIDATION chỉ để báo cáo.
- Mỗi biến cố trên VALIDATION = 4.2–4.3 điểm recall; chênh lệch nằm gọn trong CI95 **không** đủ kết luận phương pháp nào hơn.
- AUROC / AP / ECE không phụ thuộc ngưỡng — dùng để so khả năng phân biệt tách khỏi điểm vận hành.
- Development validation, **không phải** bằng chứng xác nhận độc lập; chưa mở pilot_test/global test.

## Cái giá của ràng buộc FA/giờ ≤ 0,5 (`fa_budget`, cùng model)

| Phương pháp | Phút | Recall `recall_first` | Recall `fa_budget` | Δ recall | PPV `fa_budget` | FA/giờ `fa_budget` |
|---|---:|---:|---:|---:|---:|---:|
| `map_logistic` | 5 | 0.826 | 0.217 | -0.609 | 0.294 | 0.209 |
| `catboost_balanced` | 5 | 0.826 | 0.435 | -0.391 | 0.345 | 0.331 |
| `lgbm_plain` | 5 | 0.783 | 0.435 | -0.348 | 0.385 | 0.279 |
| `lgbm_oversample_3x` | 5 | 0.783 | 0.435 | -0.348 | 0.400 | 0.262 |
| `catboost_plain` | 5 | 0.783 | 0.478 | -0.304 | 0.500 | 0.192 |
| `catboost_balanced_jitter` | 5 | 0.783 | 0.348 | -0.435 | 0.276 | 0.366 |
| `lgbm_class_weight` | 5 | 0.739 | 0.304 | -0.435 | 0.318 | 0.262 |
| `lgbm_jitter_3x` | 5 | 0.739 | 0.435 | -0.304 | 0.345 | 0.331 |
| `tabm_20260919` | 5 | 0.739 | 0.261 | -0.478 | 0.400 | 0.157 |
| `lgbm_scale_pos_20x` | 5 | 0.696 | 0.348 | -0.348 | 0.308 | 0.314 |
| `lgbm_smote_3x` | 5 | 0.696 | 0.391 | -0.304 | 0.346 | 0.297 |
| `tabm_20260917` | 5 | 0.696 | 0.261 | -0.435 | 0.400 | 0.157 |
| `tabm_20260918` | 5 | 0.696 | 0.304 | -0.391 | 0.389 | 0.192 |
| `ensemble_diverse` | 5 | 0.652 | 0.391 | -0.261 | 0.391 | 0.244 |
| `map_logistic` | 10 | 0.917 | 0.250 | -0.667 | 0.273 | 0.296 |
| `catboost_balanced_jitter` | 10 | 0.917 | 0.333 | -0.583 | 0.320 | 0.315 |
| `lgbm_plain` | 10 | 0.875 | 0.375 | -0.500 | 0.370 | 0.315 |
| `lgbm_scale_pos_20x` | 10 | 0.875 | 0.333 | -0.542 | 0.360 | 0.296 |
| `lgbm_oversample_3x` | 10 | 0.875 | 0.250 | -0.625 | 0.280 | 0.333 |
| `catboost_plain` | 10 | 0.875 | 0.292 | -0.583 | 0.269 | 0.352 |
| `catboost_balanced` | 10 | 0.875 | 0.375 | -0.500 | 0.321 | 0.352 |
| `tabm_20260917` | 10 | 0.875 | 0.333 | -0.542 | 0.421 | 0.204 |
| `tabm_20260919` | 10 | 0.875 | 0.292 | -0.583 | 0.421 | 0.204 |
| `ensemble_diverse` | 10 | 0.875 | 0.458 | -0.417 | 0.414 | 0.315 |
| `lgbm_class_weight` | 10 | 0.833 | 0.333 | -0.500 | 0.474 | 0.185 |
| `tabm_20260918` | 10 | 0.833 | 0.333 | -0.500 | 0.364 | 0.259 |
| `lgbm_jitter_3x` | 10 | 0.792 | 0.375 | -0.417 | 0.345 | 0.352 |
| `lgbm_smote_3x` | 10 | 0.750 | 0.375 | -0.375 | 0.357 | 0.333 |

## Mặt Pareto của phương pháp được chọn (VALIDATION)

Các điểm không bị điểm nào trội hơn đồng thời về recall, PPV và FA/giờ — chọn điểm nào là quyết định lâm sàng, không phải kỹ thuật.

| Phút | Phương pháp | Ngưỡng | Recall | Bắt được | PPV | FA/giờ |
|---:|---|---:|---:|---:|---:|---:|
| 5 | `lgbm_oversample_3x` | 0.049 | 0.826 | 19/23 | 0.181 | 1.500 |
| 5 | `lgbm_oversample_3x` | 0.051 | 0.783 | 18/23 | 0.176 | 1.466 |
| 5 | `lgbm_oversample_3x` | 0.072 | 0.696 | 16/23 | 0.232 | 0.925 |
| 5 | `lgbm_oversample_3x` | 0.085 | 0.652 | 15/23 | 0.283 | 0.663 |
| 5 | `lgbm_oversample_3x` | 0.088 | 0.609 | 14/23 | 0.275 | 0.646 |
| 5 | `lgbm_oversample_3x` | 0.092 | 0.565 | 13/23 | 0.271 | 0.611 |
| 5 | `lgbm_oversample_3x` | 0.148 | 0.478 | 11/23 | 0.393 | 0.297 |
| 5 | `lgbm_oversample_3x` | 0.186 | 0.435 | 10/23 | 0.435 | 0.227 |
| 5 | `lgbm_oversample_3x` | 0.236 | 0.304 | 7/23 | 0.467 | 0.140 |
| 5 | `lgbm_oversample_3x` | 0.284 | 0.174 | 4/23 | 0.571 | 0.052 |
| 5 | `lgbm_oversample_3x` | 0.337 | 0.130 | 3/23 | 0.750 | 0.017 |
| 5 | `lgbm_oversample_3x` | 0.538 | 0.000 | 0/23 | N/A | 0.000 |
| 10 | `map_logistic` | 0.056 | 1.000 | 24/24 | 0.142 | 3.242 |
| 10 | `map_logistic` | 0.093 | 0.958 | 23/24 | 0.277 | 1.352 |
| 10 | `map_logistic` | 0.108 | 0.917 | 22/24 | 0.333 | 0.963 |
| 10 | `map_logistic` | 0.116 | 0.875 | 21/24 | 0.424 | 0.630 |
| 10 | `map_logistic` | 0.120 | 0.792 | 19/24 | 0.407 | 0.593 |
| 10 | `map_logistic` | 0.124 | 0.667 | 16/24 | 0.396 | 0.537 |
| 10 | `map_logistic` | 0.125 | 0.625 | 15/24 | 0.419 | 0.463 |
| 10 | `map_logistic` | 0.128 | 0.583 | 14/24 | 0.400 | 0.445 |
| 10 | `map_logistic` | 0.132 | 0.542 | 13/24 | 0.395 | 0.426 |
| 10 | `map_logistic` | 0.136 | 0.500 | 12/24 | 0.375 | 0.371 |
| 10 | `map_logistic` | 0.142 | 0.333 | 8/24 | 0.320 | 0.315 |
| 10 | `map_logistic` | 0.144 | 0.250 | 6/24 | 0.273 | 0.296 |
| 10 | `map_logistic` | 0.151 | 0.208 | 5/24 | 0.250 | 0.278 |
| 10 | `map_logistic` | 0.211 | 0.083 | 2/24 | 0.250 | 0.111 |
| 10 | `map_logistic` | 0.257 | 0.042 | 1/24 | 0.250 | 0.056 |
| 10 | `map_logistic` | 0.462 | 0.000 | 0/24 | N/A | 0.000 |

## Kiểm chứng calibration

Mean xác suất sau calibrate phải bám prevalence thật, nếu không mọi so sánh ngưỡng và phép trung bình ensemble đều sai (lỗi F2 của v1).

| Phương pháp | Phút | Mean thô | Mean sau calibrate | Prevalence |
|---|---:|---:|---:|---:|
| `map_logistic` | 5 | 0.0345 | 0.0251 | 0.0251 |
| `lgbm_plain` | 5 | 0.0329 | 0.0252 | 0.0251 |
| `lgbm_class_weight` | 5 | 0.2229 | 0.0252 | 0.0251 |
| `lgbm_scale_pos_20x` | 5 | 0.4186 | 0.0251 | 0.0251 |
| `lgbm_oversample_3x` | 5 | 0.0825 | 0.0251 | 0.0251 |
| `lgbm_smote_3x` | 5 | 0.0339 | 0.0251 | 0.0251 |
| `lgbm_jitter_3x` | 5 | 0.0347 | 0.0251 | 0.0251 |
| `catboost_plain` | 5 | 0.0351 | 0.0252 | 0.0251 |
| `catboost_balanced` | 5 | 0.1834 | 0.0252 | 0.0251 |
| `catboost_balanced_jitter` | 5 | 0.1116 | 0.0252 | 0.0251 |
| `tabm_20260917` | 5 | 0.0348 | 0.0251 | 0.0251 |
| `tabm_20260918` | 5 | 0.0337 | 0.0251 | 0.0251 |
| `tabm_20260919` | 5 | 0.0384 | 0.0251 | 0.0251 |
| `ensemble_diverse` | 5 | 0.0251 | 0.0251 | 0.0251 |
| `map_logistic` | 10 | 0.0631 | 0.0553 | 0.0553 |
| `lgbm_plain` | 10 | 0.0646 | 0.0553 | 0.0553 |
| `lgbm_class_weight` | 10 | 0.2922 | 0.0553 | 0.0553 |
| `lgbm_scale_pos_20x` | 10 | 0.5408 | 0.0553 | 0.0553 |
| `lgbm_oversample_3x` | 10 | 0.1546 | 0.0553 | 0.0553 |
| `lgbm_smote_3x` | 10 | 0.0674 | 0.0553 | 0.0553 |
| `lgbm_jitter_3x` | 10 | 0.0676 | 0.0553 | 0.0553 |
| `catboost_plain` | 10 | 0.0665 | 0.0553 | 0.0553 |
| `catboost_balanced` | 10 | 0.2417 | 0.0553 | 0.0553 |
| `catboost_balanced_jitter` | 10 | 0.1436 | 0.0552 | 0.0553 |
| `tabm_20260917` | 10 | 0.0629 | 0.0553 | 0.0553 |
| `tabm_20260918` | 10 | 0.0644 | 0.0552 | 0.0553 |
| `tabm_20260919` | 10 | 0.0697 | 0.0552 | 0.0553 |
| `ensemble_diverse` | 10 | 0.0553 | 0.0553 | 0.0553 |
