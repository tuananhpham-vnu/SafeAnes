# EDA tuần 15: tương quan, tầm quan trọng và chọn đặc trưng ngoài MAP

Sinh tự động từ `scripts/beyond_map_eda.py` (Kaggle). Mọi con số là **đánh giá thật**: CV 5 fold theo bệnh nhân trên train (dùng để chọn), validation (chỉ để báo cáo). Test không được đọc. W = 120 s; mốc dự báo 5 / 10 / 15 phút; sự kiện MAP < 65 mmHg ≥ 60 s.

Quy ước kết luận: thay đổi AUROC trung bình 3 mốc > 0.0005 là **tốt hơn** (giữ), < −0.0005 là **xấu hơn** (ghi chú, không giữ), còn lại **không đổi**.

## 1. Mốc so sánh: mô hình chỉ-MAP

Lõi MAP = mọi thống kê MAP/SBP/DBP trong W, xu hướng MAP 5–30 phút, thời gian ở vùng 65–80, % giảm so với nền, MAP ngoại suy, lịch sử đợt tụt trong ca. AUROC out-of-fold trên train:

| Mốc | Luật MAP hiện tại | LightGBM lõi MAP | Tỷ lệ dương |
|---|---|---|---|
| 5 phút | 0.814 | 0.887 | 5.1% |
| 10 phút | 0.756 | 0.847 | 9.7% |
| 15 phút | 0.734 | 0.830 | 13.8% |

## 2. Tương quan giữa các biến (Spearman, train)

205 cặp có |ρ| ≥ 0,8 trên 244 biến. Các cặp gần trùng lặp nhất (chỉ cần giữ một biến mỗi cụm):

| Biến A | Biến B | |ρ| |
|---|---|---|
| cx_ppv_grey | bt_ppv30_w120 | 1.00 |
| f1a_min_70_80_m15 | cx_map_grey_burden | 0.99 |
| f1c_drug_rftn_vol_m5 | f1c_drug_rftn_rate_mean_m5 | 0.99 |
| f1c_drug_ppf_vol_m5 | f1c_drug_ppf_rate_mean_m5 | 0.99 |
| cx_no_hr_comp | cx_no_hr_comp_age | 0.98 |
| hr_mean_w120 | f1a_hr_mean_m5 | 0.98 |
| spo2_mean_w120 | spo2_min_w120 | 0.98 |
| map_std_w120 | cx_map_cv | 0.98 |
| sbp_missing_w120 | pp_missing_w120 | 0.98 |
| f1c_drug_ppf_cp | f1b_ppf | 0.97 |
| f1b_mac | f1c_drug_exp_des | 0.97 |
| f1a_hr_mean_m5 | f1a_hr_mean_m15 | 0.97 |
| map_drop_pct | cx_drop_age | 0.97 |
| f1a_min_since_opstart | f1a_min_since_anestart | 0.97 |
| rr_mean_w120 | rr_current | 0.97 |
| f1b_mac | f1c_drug_exp_sevo | 0.97 |
| hr_current | hr_mean_w120 | 0.96 |
| etco2_mean_w120 | f1a_etco2_mean_m5 | 0.96 |
| pp_mean_w120 | f1a_pp_mean_m5 | 0.96 |
| shock_index_current | shock_index_mean_w120 | 0.96 |

Mức "MAP trá hình" của các biến đứng đầu: |ρ| lớn nhất với một cột của lõi MAP. Biến có |ρ| cao chủ yếu lặp lại MAP; biến có |ρ| thấp mang thông tin khác.

| Biến | |ρ| max với lõi MAP | Cột MAP gần nhất |
|---|---|---|
| f1b_rftn_d1 | 0.26 | sbp_mean_w120 |
| cx_depth_d5 | 0.25 | f1a_map_slope_m15 |
| f1b_rftn_d10 | 0.32 | f1a_map_slope_m30 |
| cx_depth_d5_grey | 0.20 | f1a_map_mean_m5 |
| f1b_rftn_d5 | 0.27 | f1a_map_mean_m5 |
| f1b_mac | 0.11 | sbp_max_w120 |
| f1a_min_since_opstart | 0.45 | f1a_min_since_last_event |
| f1b_mac_d10 | 0.12 | f1a_map_slope_m30 |
| f1b_mac_d5 | 0.11 | f1a_map_slope_m15 |
| f1c_drug_rftn_rate_d1 | 0.06 | f1a_map_slope_m5 |
| f1c_drug_exp_sevo_d5 | 0.13 | f1a_map_slope_m15 |
| f1c_drug_rftn_rate_d5 | 0.18 | f1a_map_slope_m10 |
| f1c_drug_exp_sevo | 0.10 | sbp_current |
| f1b_ppf_d1 | 0.08 | sbp_mean_w120 |
| f1b_mac_d1 | 0.06 | f1a_map_slope_m15 |
| f1a_etco2_slope_m5 | 0.28 | f1a_map_slope_m10 |
| f1c_drug_ppf_rate_d5 | 0.08 | f1a_n_prev_events |
| f1c_vent_pplat | 0.18 | dbp_min_w120 |
| f1c_drug_rftn_vol_m5 | 0.35 | sbp_mean_w120 |
| etco2_std_w120 | 0.31 | sbp_std_w120 |
| f1b_approach_open | 0.20 | f1a_n_prev_events |
| f1b_etco2_d5 | 0.27 | f1a_map_slope_m10 |
| f1c_drug_rftn_rate_mean_m5 | 0.35 | sbp_mean_w120 |
| f1c_drug_rftn_rate | 0.27 | sbp_mean_w120 |
| f1b_ppf_d5 | 0.10 | f1a_map_slope_m30 |

## 3. Tương quan với nhãn khi đã biết MAP, từng biến (trước → sau khi thêm)

Trước = mô hình chỉ-MAP (offset out-of-fold); sau = offset + một mô hình nhỏ trên biến đó, CV 5 fold theo bệnh nhân. ρ riêng phần = Spearman của biến với phần dư y − p(MAP), CI bootstrap theo ca. Vùng xám = MAP hiện tại 65–80 mmHg, nơi MAP một mình mơ hồ.

Tổng 203 biến: **28 tốt hơn**, **4 xấu hơn**, 171 không đổi.

### 3a. 30 biến thêm được nhiều nhất

| Biến | Nhóm | AUROC đứng riêng 5'/15' | ρ với phần dư MAP (15') [CI 95%] | AUROC vùng xám 15' | ΔAUROC khi thêm vào chỉ-MAP 5'/10'/15' | ΔAUPRC 15' | Kết luận |
|---|---|---|---|---|---|---|---|
| f1b_rftn_d1 | anaes | 0.505 / 0.500 | 0.082 [0.075; 0.090] | 0.533 | +0.0018 / +0.0022 / +0.0016 | +0.0043 | tốt hơn → giữ để xét chọn |
| cx_depth_d5 | composite | 0.532 / 0.516 | 0.062 [0.054; 0.072] | 0.542 | +0.0023 / +0.0025 / +0.0020 | +0.0036 | tốt hơn → giữ để xét chọn |
| f1b_rftn_d10 | anaes | 0.532 / 0.513 | 0.069 [0.060; 0.079] | 0.544 | +0.0016 / +0.0019 / +0.0013 | +0.0025 | tốt hơn → giữ để xét chọn |
| cx_depth_d5_grey | composite | 0.514 / 0.505 | 0.089 [0.081; 0.097] | 0.542 | +0.0014 / +0.0016 / +0.0014 | +0.0032 | tốt hơn → giữ để xét chọn |
| f1b_rftn_d5 | anaes | 0.519 / 0.508 | 0.076 [0.069; 0.084] | 0.543 | +0.0018 / +0.0021 / +0.0015 | +0.0027 | tốt hơn → giữ để xét chọn |
| f1b_mac | anaes | 0.527 / 0.529 | 0.012 [-0.009; 0.033] | 0.514 | +0.0013 / +0.0017 / +0.0018 | +0.0028 | tốt hơn → giữ để xét chọn |
| f1a_min_since_opstart | context | 0.512 / 0.517 | 0.014 [-0.006; 0.034] | 0.504 | +0.0008 / +0.0012 / +0.0012 | +0.0021 | tốt hơn → giữ để xét chọn |
| f1b_mac_d10 | anaes | 0.538 / 0.518 | 0.021 [0.011; 0.029] | 0.520 | +0.0018 / +0.0017 / +0.0016 | +0.0014 | tốt hơn → giữ để xét chọn |
| f1b_mac_d5 | anaes | 0.530 / 0.519 | 0.028 [0.021; 0.033] | 0.519 | +0.0014 / +0.0018 / +0.0015 | +0.0019 | tốt hơn → giữ để xét chọn |
| f1c_drug_rftn_rate_d1 | anaes | 0.517 / 0.508 | -0.007 [-0.010; -0.003] | 0.512 | +0.0004 / +0.0005 / +0.0003 | +0.0021 | không đổi |
| f1c_drug_exp_sevo_d5 | anaes | 0.536 / 0.519 | 0.032 [0.022; 0.043] | 0.522 | +0.0011 / +0.0014 / +0.0011 | +0.0013 | tốt hơn → giữ để xét chọn |
| f1c_drug_rftn_rate_d5 | anaes | 0.508 / 0.503 | 0.003 [-0.002; 0.007] | 0.512 | +0.0003 / +0.0005 / +0.0004 | +0.0012 | không đổi |
| f1c_drug_exp_sevo | anaes | 0.517 / 0.517 | 0.022 [0.000; 0.047] | 0.504 | +0.0009 / +0.0011 / +0.0012 | +0.0015 | tốt hơn → giữ để xét chọn |
| f1b_ppf_d1 | anaes | 0.501 / 0.509 | 0.019 [0.010; 0.027] | 0.502 | +0.0005 / +0.0009 / +0.0013 | +0.0014 | tốt hơn → giữ để xét chọn |
| f1b_mac_d1 | anaes | 0.509 / 0.508 | 0.022 [0.017; 0.026] | 0.506 | +0.0009 / +0.0010 / +0.0009 | +0.0010 | tốt hơn → giữ để xét chọn |
| f1a_etco2_slope_m5 | context | 0.505 / 0.511 | 0.014 [0.008; 0.019] | 0.526 | +0.0002 / +0.0005 / +0.0004 | +0.0017 | không đổi |
| f1c_drug_ppf_rate_d5 | anaes | 0.519 / 0.515 | 0.018 [0.009; 0.024] | 0.508 | +0.0004 / +0.0009 / +0.0013 | +0.0015 | tốt hơn → giữ để xét chọn |
| f1c_vent_pplat | anaes | 0.593 / 0.584 | 0.052 [0.032; 0.068] | 0.568 | +0.0002 / +0.0003 / +0.0001 | +0.0014 | không đổi |
| f1c_drug_rftn_vol_m5 | anaes | 0.543 / 0.573 | 0.147 [0.132; 0.167] | 0.509 | +0.0003 / -0.0001 / -0.0001 | +0.0006 | không đổi |
| etco2_std_w120 | numeric | 0.553 / 0.550 | -0.033 [-0.048; -0.021] | 0.555 | +0.0005 / +0.0008 / +0.0007 | +0.0013 | tốt hơn → giữ để xét chọn |
| f1b_approach_open | clinical | 0.574 / 0.568 | -0.038 [-0.054; -0.020] | 0.556 | +0.0006 / +0.0005 / +0.0005 | +0.0012 | tốt hơn → giữ để xét chọn |
| f1b_etco2_d5 | anaes | 0.509 / 0.508 | 0.011 [0.006; 0.017] | 0.521 | +0.0003 / +0.0004 / +0.0003 | +0.0011 | không đổi |
| f1c_drug_rftn_rate_mean_m5 | anaes | 0.539 / 0.571 | 0.149 [0.132; 0.169] | 0.508 | +0.0004 / -0.0002 / -0.0003 | +0.0006 | không đổi |
| f1c_drug_rftn_rate | anaes | 0.565 / 0.578 | 0.148 [0.132; 0.165] | 0.517 | -0.0001 / -0.0001 / +0.0001 | +0.0013 | không đổi |
| f1b_ppf_d5 | anaes | 0.502 / 0.510 | 0.017 [0.008; 0.028] | 0.504 | +0.0004 / +0.0010 / +0.0013 | +0.0011 | tốt hơn → giữ để xét chọn |
| f1b_ppf_d10 | anaes | 0.502 / 0.508 | 0.017 [0.005; 0.030] | 0.501 | +0.0005 / +0.0010 / +0.0014 | +0.0012 | tốt hơn → giữ để xét chọn |
| f1b_rftn | anaes | 0.552 / 0.581 | 0.136 [0.117; 0.156] | 0.517 | +0.0000 / +0.0000 / +0.0001 | +0.0007 | không đổi |
| f1b_etco2_d1 | anaes | 0.522 / 0.509 | 0.007 [0.004; 0.011] | 0.505 | +0.0003 / +0.0004 / +0.0003 | +0.0012 | không đổi |
| f1c_drug_ppf_rate_d1 | anaes | 0.513 / 0.512 | 0.011 [0.007; 0.015] | 0.509 | +0.0004 / +0.0008 / +0.0012 | +0.0009 | tốt hơn → giữ để xét chọn |
| etco2_min_w120 | numeric | 0.560 / 0.558 | 0.064 [0.052; 0.077] | 0.536 | +0.0004 / +0.0004 / +0.0003 | +0.0005 | không đổi |

### 3b. Biến làm xấu hơn (ghi chú, không giữ)

| Biến | Nhóm | AUROC đứng riêng 5'/15' | ρ với phần dư MAP (15') [CI 95%] | AUROC vùng xám 15' | ΔAUROC khi thêm vào chỉ-MAP 5'/10'/15' | ΔAUPRC 15' | Kết luận |
|---|---|---|---|---|---|---|---|
| st_preop_gluc | clinical | 0.539 / 0.542 | -0.043 [-0.059; -0.023] | 0.530 | -0.0004 / -0.0008 / -0.0010 | -0.0043 | xấu hơn → ghi chú, bỏ |
| cx_depth_age | composite | 0.509 / 0.528 | 0.042 [0.022; 0.064] | 0.502 | -0.0003 / -0.0008 / -0.0007 | -0.0025 | xấu hơn → ghi chú, bỏ |
| f1c_pre_bmi | clinical | 0.523 / 0.519 | 0.015 [-0.004; 0.030] | 0.511 | -0.0005 / -0.0005 / -0.0009 | -0.0020 | xấu hơn → ghi chú, bỏ |
| f1c_pre_na | clinical | 0.534 / 0.530 | 0.020 [0.003; 0.042] | 0.520 | -0.0004 / -0.0005 / -0.0008 | -0.0024 | xấu hơn → ghi chú, bỏ |

### 3c. Ý nghĩa thống kê (từng biến, từng mốc)

Các dòng trong một ca tự tương quan rất mạnh nên mọi p đều tính từ SE bootstrap theo ca hoặc SE cluster-robust theo bệnh nhân, không từ số dòng. q = Benjamini–Hochberg trên mọi biến của một mốc. OR/IQR: hồi quy logistic y ~ offset(logit chỉ-MAP) + x/IQR + cờ thiếu, tức odds ratio khi x tăng một IQR ở cùng mức nguy cơ theo MAP. ΔAUROC: AUROC(chỉ-MAP + x) − AUROC(chỉ-MAP), out-of-fold, ghép cặp. "Có ý nghĩa ngoài MAP" = q(OR) < 0,05 và q(ΔAUROC) < 0,05 và ΔAUROC > 0.

| Mốc | Số biến | Có ý nghĩa khi đứng riêng (q<0,05) | Có ý nghĩa ngoài MAP |
|---|---|---|---|
| 5 phút | 217 | 176 | 17 |
| 10 phút | 217 | 180 | 19 |
| 15 phút | 217 | 179 | 13 |

Mốc 5 phút: 25 biến có ΔAUROC lớn nhất

| Biến | Nhóm | Hướng | AUROC riêng [CI] | ρ với nhãn | ρ với phần dư MAP [CI] | OR/IQR [CI] | q(OR) | ΔAUROC [CI] | q(ΔAUROC) | Ngoài MAP |
|---|---|---|---|---|---|---|---|---|---|---|
| cx_depth_d5 | composite | cao → nguy cơ | 0.532 [0.520; 0.539] | +0.024 | +0.055 [+0.048; +0.062] | 1.01 [1.01; 1.01] | 5.1e-26 | +0.0022 [+0.0014; +0.0030] | 9.6e-07 | có ý nghĩa |
| f1b_rftn_d1 | anaes | thấp → nguy cơ | 0.505 [0.497; 0.512] | -0.003 | +0.093 [+0.085; +0.100] | 1.00 [1.00; 1.00] | 9.7e-27 | +0.0017 [+0.0010; +0.0025] | 1.1e-04 | có ý nghĩa |
| f1b_rftn_d5 | anaes | cao → nguy cơ | 0.519 [0.509; 0.528] | +0.014 | +0.079 [+0.072; +0.086] | 1.01 [1.00; 1.01] | 6.4e-26 | +0.0017 [+0.0010; +0.0025] | 8.3e-05 | có ý nghĩa |
| f1b_mac_d10 | anaes | cao → nguy cơ | 0.538 [0.530; 0.544] | +0.041 | +0.006 [-0.001; +0.014] | 3.07 [2.41; 3.92] | 5.3e-18 | +0.0017 [+0.0011; +0.0023] | 1.2e-06 | có ý nghĩa |
| f1b_rftn_d10 | anaes | cao → nguy cơ | 0.532 [0.523; 0.542] | +0.024 | +0.062 [+0.054; +0.070] | 1.02 [1.02; 1.03] | 5.8e-23 | +0.0016 [+0.0009; +0.0022] | 1.3e-04 | có ý nghĩa |
| f1b_mac_d5 | anaes | cao → nguy cơ | 0.530 [0.524; 0.537] | +0.036 | +0.012 [+0.006; +0.020] | 2.96 [2.32; 3.78] | 5.4e-17 | +0.0013 [+0.0008; +0.0017] | 4.9e-05 | có ý nghĩa |
| cx_depth_d5_grey | composite | cao → nguy cơ | 0.514 [0.504; 0.524] | +0.013 | +0.117 [+0.109; +0.124] | 2.49 [2.07; 3.00] | 1.8e-20 | +0.0013 [+0.0008; +0.0019] | 3.7e-05 | có ý nghĩa |
| f1b_mac | anaes | cao → nguy cơ | 0.527 [0.515; 0.539] | +0.023 | -0.020 [-0.043; +0.007] | 1.32 [1.22; 1.43] | 7.1e-11 | +0.0012 [+0.0004; +0.0019] | 9.0e-03 | có ý nghĩa |
| f1c_drug_exp_sevo_d5 | anaes | cao → nguy cơ | 0.536 [0.528; 0.544] | +0.035 | +0.019 [+0.011; +0.027] | 2.92 [2.16; 3.95] | 8.2e-11 | +0.0010 [+0.0005; +0.0015] | 7.9e-04 | có ý nghĩa |
| f1b_mac_d1 | anaes | cao → nguy cơ | 0.509 [0.505; 0.512] | +0.015 | +0.013 [+0.009; +0.018] | 1.93 [1.65; 2.26] | 6.2e-15 | +0.0008 [+0.0005; +0.0012] | 1.2e-04 | có ý nghĩa |
| f1c_drug_exp_sevo | anaes | cao → nguy cơ | 0.517 [0.499; 0.534] | +0.014 | -0.003 [-0.032; +0.028] | 1.35 [1.22; 1.49] | 1.4e-07 | +0.0008 [+0.0002; +0.0013] | 2.8e-02 | có ý nghĩa |
| f1a_min_since_opstart | context | cao → nguy cơ | 0.512 [0.494; 0.526] | +0.008 | +0.014 [-0.004; +0.032] | 0.96 [0.92; 1.01] | 2.1e-01 | +0.0007 [+0.0001; +0.0011] | 2.7e-02 | không |
| f1b_approach_open | clinical | cao → nguy cơ | 0.574 [0.563; 0.585] | +0.065 | -0.086 [-0.103; -0.066] | 1.25 [1.16; 1.35] | 1.3e-07 | +0.0006 [+0.0002; +0.0009] | 8.4e-03 | có ý nghĩa |
| f1c_vent_compliance_d1 | anaes | cao → nguy cơ | 0.509 [0.505; 0.514] | +0.007 | +0.000 [-0.002; +0.003] | 1.01 [1.00; 1.02] | 6.2e-02 | +0.0005 [+0.0002; +0.0007] | 3.3e-03 | không |
| f1b_ppf_d1 | anaes | cao → nguy cơ | 0.501 [0.492; 0.509] | +0.001 | +0.020 [+0.013; +0.028] | 1.00 [1.00; 1.00] | 2.4e-04 | +0.0004 [+0.0000; +0.0008] | 7.2e-02 | không |
| etco2_std_w120 | numeric | cao → nguy cơ | 0.553 [0.543; 0.563] | +0.040 | -0.077 [-0.091; -0.064] | 1.02 [1.01; 1.03] | 1.8e-12 | +0.0004 [+0.0002; +0.0008] | 2.5e-02 | có ý nghĩa |
| f1b_ppf_d10 | anaes | cao → nguy cơ | 0.502 [0.491; 0.513] | +0.002 | +0.011 [+0.002; +0.022] | 1.00 [1.00; 1.00] | 2.6e-01 | +0.0004 [-0.0000; +0.0008] | 1.5e-01 | không |
| f1c_vent_tv_d1 | anaes | cao → nguy cơ | 0.503 [0.499; 0.507] | +0.002 | +0.001 [-0.001; +0.005] | 1.00 [1.00; 1.01] | 5.4e-01 | +0.0004 [+0.0001; +0.0006] | 2.5e-02 | không |
| f1b_approach_videoscopic | clinical | thấp → nguy cơ | 0.564 [0.554; 0.577] | -0.056 | +0.075 [+0.055; +0.091] | 0.83 [0.77; 0.89] | 1.5e-05 | +0.0004 [+0.0001; +0.0006] | 3.5e-02 | có ý nghĩa |
| f1b_ppf_d5 | anaes | cao → nguy cơ | 0.502 [0.489; 0.513] | +0.001 | +0.017 [+0.009; +0.026] | 1.00 [1.00; 1.00] | 9.1e-02 | +0.0004 [-0.0000; +0.0007] | 1.5e-01 | không |
| f1c_drug_rftn_rate_d1 | anaes | thấp → nguy cơ | 0.517 [0.512; 0.522] | -0.013 | -0.002 [-0.006; +0.001] | 1.00 [1.00; 1.00] | 2.1e-01 | +0.0004 [+0.0001; +0.0007] | 9.4e-02 | không |
| f1c_drug_ppf_rate_d1 | anaes | thấp → nguy cơ | 0.513 [0.508; 0.519] | -0.010 | +0.020 [+0.015; +0.025] | 1.00 [1.00; 1.00] | 8.7e-01 | +0.0003 [+0.0000; +0.0007] | 1.3e-01 | không |
| f1c_vent_mv_d1 | anaes | cao → nguy cơ | 0.502 [0.497; 0.506] | +0.001 | -0.000 [-0.003; +0.003] | 1.03 [0.87; 1.21] | 8.1e-01 | +0.0003 [+0.0001; +0.0005] | 8.4e-03 | không |
| ep_ppg_amp_rel600 | ecgppg | cao → nguy cơ | 0.527 [0.518; 0.533] | +0.019 | -0.015 [-0.021; -0.009] | 1.03 [1.00; 1.05] | 5.7e-02 | +0.0003 [+0.0001; +0.0007] | 1.1e-01 | không |
| f1c_drug_ppf_rate_d5 | anaes | thấp → nguy cơ | 0.519 [0.508; 0.530] | -0.014 | +0.028 [+0.020; +0.034] | 1.00 [1.00; 1.00] | 4.8e-01 | +0.0003 [-0.0000; +0.0007] | 2.0e-01 | không |

Mốc 5 phút: biến có ý nghĩa khi đứng riêng nhưng **không** thêm gì khi đã biết MAP (chủ yếu lặp lại thông tin MAP)

| Biến | Nhóm | Hướng | AUROC riêng [CI] | ρ với nhãn | ρ với phần dư MAP [CI] | OR/IQR [CI] | q(OR) | ΔAUROC [CI] | q(ΔAUROC) | Ngoài MAP |
|---|---|---|---|---|---|---|---|---|---|---|
| cx_map_proj_5 | composite | thấp → nguy cơ | 0.733 [0.725; 0.742] | -0.176 | +0.373 [+0.362; +0.381] | 0.99 [0.96; 1.02] | 6.5e-01 | -0.0003 [-0.0004; -0.0002] | 9.5e-06 | không |
| cx_dsi | composite | cao → nguy cơ | 0.718 [0.706; 0.727] | +0.165 | -0.354 [-0.367; -0.340] | 1.08 [1.03; 1.13] | 3.7e-03 | +0.0001 [-0.0002; +0.0003] | 6.8e-01 | không |
| bt_svr_mean_w120 | wave | thấp → nguy cơ | 0.712 [0.700; 0.724] | -0.161 | +0.350 [+0.337; +0.367] | 0.86 [0.81; 0.91] | 1.1e-05 | +0.0001 [-0.0003; +0.0004] | 6.4e-01 | không |
| cx_msi | composite | cao → nguy cơ | 0.699 [0.686; 0.709] | +0.151 | -0.322 [-0.333; -0.308] | 1.07 [1.02; 1.12] | 9.5e-03 | +0.0000 [-0.0003; +0.0003] | 9.7e-01 | không |
| cx_map_proj_10 | composite | thấp → nguy cơ | 0.687 [0.677; 0.696] | -0.140 | +0.306 [+0.295; +0.314] | 0.98 [0.94; 1.02] | 4.4e-01 | -0.0003 [-0.0004; -0.0001] | 5.9e-04 | không |
| cx_map_proj_15 | composite | thấp → nguy cơ | 0.672 [0.661; 0.683] | -0.128 | +0.295 [+0.284; +0.303] | 1.00 [0.96; 1.04] | 8.8e-01 | -0.0003 [-0.0004; -0.0001] | 5.9e-04 | không |
| cx_map_ttc_m5 | composite | thấp → nguy cơ | 0.671 [0.663; 0.681] | -0.143 | +0.207 [+0.198; +0.213] | 0.98 [0.94; 1.03] | 5.4e-01 | -0.0002 [-0.0003; -0.0001] | 1.0e-03 | không |
| cx_map_event_rate | composite | cao → nguy cơ | 0.655 [0.643; 0.667] | +0.122 | -0.284 [-0.301; -0.267] | 1.05 [1.01; 1.09] | 5.7e-02 | -0.0003 [-0.0006; -0.0001] | 5.9e-02 | không |
| shock_index_current | numeric | cao → nguy cơ | 0.652 [0.639; 0.665] | +0.115 | -0.247 [-0.262; -0.233] | 1.05 [1.01; 1.10] | 4.9e-02 | -0.0001 [-0.0004; +0.0001] | 5.4e-01 | không |
| cx_drop_age | composite | cao → nguy cơ | 0.648 [0.631; 0.666] | +0.112 | -0.234 [-0.256; -0.214] | 1.02 [0.97; 1.07] | 5.3e-01 | -0.0003 [-0.0005; -0.0002] | 5.9e-04 | không |
| shock_index_max_w120 | numeric | cao → nguy cơ | 0.644 [0.633; 0.657] | +0.110 | -0.274 [-0.289; -0.260] | 1.03 [0.99; 1.07] | 2.6e-01 | -0.0003 [-0.0005; -0.0001] | 2.8e-02 | không |
| cx_map_cv | composite | cao → nguy cơ | 0.638 [0.629; 0.646] | +0.105 | -0.286 [-0.298; -0.275] | 1.00 [0.98; 1.02] | 9.3e-01 | -0.0002 [-0.0004; -0.0001] | 6.7e-03 | không |
| shock_index_mean_w120 | numeric | cao → nguy cơ | 0.635 [0.623; 0.648] | +0.103 | -0.250 [-0.265; -0.237] | 1.04 [0.99; 1.08] | 2.1e-01 | -0.0003 [-0.0005; -0.0001] | 3.7e-02 | không |
| cx_map_grey_burden | composite | cao → nguy cơ | 0.631 [0.622; 0.640] | +0.100 | -0.490 [-0.500; -0.479] | 0.99 [0.93; 1.06] | 9.1e-01 | -0.0006 [-0.0007; -0.0004] | 9.4e-11 | không |
| cx_map_ttc_m15 | composite | thấp → nguy cơ | 0.626 [0.619; 0.633] | -0.110 | +0.194 [+0.185; +0.201] | 0.99 [0.95; 1.03] | 6.3e-01 | -0.0003 [-0.0004; -0.0002] | 6.2e-09 | không |

Mốc 10 phút: 25 biến có ΔAUROC lớn nhất

| Biến | Nhóm | Hướng | AUROC riêng [CI] | ρ với nhãn | ρ với phần dư MAP [CI] | OR/IQR [CI] | q(OR) | ΔAUROC [CI] | q(ΔAUROC) | Ngoài MAP |
|---|---|---|---|---|---|---|---|---|---|---|
| cx_depth_d5 | composite | cao → nguy cơ | 0.526 [0.517; 0.532] | +0.026 | +0.062 [+0.053; +0.071] | 1.01 [1.01; 1.01] | 2.6e-34 | +0.0025 [+0.0017; +0.0032] | 1.6e-08 | có ý nghĩa |
| f1b_rftn_d1 | anaes | cao → nguy cơ | 0.504 [0.496; 0.510] | +0.005 | +0.088 [+0.081; +0.096] | 1.00 [1.00; 1.00] | 1.4e-48 | +0.0021 [+0.0013; +0.0028] | 2.4e-07 | có ý nghĩa |
| f1b_rftn_d5 | anaes | cao → nguy cơ | 0.517 [0.507; 0.525] | +0.017 | +0.079 [+0.072; +0.086] | 1.00 [1.00; 1.00] | 4.1e-31 | +0.0020 [+0.0013; +0.0027] | 3.9e-06 | có ý nghĩa |
| f1b_rftn_d10 | anaes | cao → nguy cơ | 0.522 [0.513; 0.530] | +0.023 | +0.068 [+0.059; +0.077] | 1.02 [1.01; 1.02] | 1.6e-20 | +0.0018 [+0.0012; +0.0026] | 1.1e-05 | có ý nghĩa |
| f1b_mac_d5 | anaes | cao → nguy cơ | 0.526 [0.520; 0.531] | +0.042 | +0.027 [+0.019; +0.033] | 2.70 [2.27; 3.21] | 3.0e-27 | +0.0017 [+0.0011; +0.0023] | 8.7e-06 | có ý nghĩa |
| f1b_mac_d10 | anaes | cao → nguy cơ | 0.528 [0.521; 0.533] | +0.040 | +0.019 [+0.011; +0.029] | 2.46 [1.98; 3.04] | 4.6e-15 | +0.0016 [+0.0010; +0.0022] | 2.2e-04 | có ý nghĩa |
| f1b_mac | anaes | cao → nguy cơ | 0.527 [0.515; 0.539] | +0.030 | -0.003 [-0.024; +0.021] | 1.34 [1.24; 1.45] | 4.0e-12 | +0.0016 [+0.0006; +0.0024] | 3.8e-03 | có ý nghĩa |
| cx_depth_d5_grey | composite | cao → nguy cơ | 0.509 [0.502; 0.518] | +0.011 | +0.101 [+0.093; +0.108] | 2.45 [2.08; 2.88] | 8.9e-26 | +0.0014 [+0.0009; +0.0019] | 4.6e-06 | có ý nghĩa |
| f1c_drug_exp_sevo_d5 | anaes | cao → nguy cơ | 0.529 [0.521; 0.535] | +0.038 | +0.033 [+0.024; +0.041] | 2.75 [2.14; 3.53] | 4.0e-14 | +0.0012 [+0.0006; +0.0018] | 3.8e-04 | có ý nghĩa |
| f1a_min_since_opstart | context | cao → nguy cơ | 0.513 [0.496; 0.526] | +0.012 | +0.019 [-0.002; +0.039] | 0.98 [0.93; 1.03] | 4.5e-01 | +0.0011 [+0.0005; +0.0018] | 3.7e-03 | không |
| f1c_drug_exp_sevo | anaes | cao → nguy cơ | 0.515 [0.498; 0.533] | +0.016 | +0.011 [-0.015; +0.031] | 1.35 [1.22; 1.50] | 1.5e-07 | +0.0009 [+0.0001; +0.0016] | 3.8e-02 | có ý nghĩa |
| f1b_ppf_d10 | anaes | cao → nguy cơ | 0.501 [0.492; 0.514] | +0.001 | +0.018 [+0.009; +0.030] | 1.00 [1.00; 1.00] | 2.7e-02 | +0.0009 [+0.0003; +0.0016] | 2.4e-02 | có ý nghĩa |
| f1b_ppf_d5 | anaes | thấp → nguy cơ | 0.504 [0.496; 0.514] | -0.004 | +0.018 [+0.009; +0.029] | 1.00 [1.00; 1.00] | 1.7e-02 | +0.0009 [+0.0003; +0.0015] | 1.6e-02 | có ý nghĩa |
| f1c_drug_ppf_vol | anaes | thấp → nguy cơ | 0.618 [0.600; 0.641] | -0.116 | +0.159 [+0.131; +0.185] | 0.93 [0.87; 0.99] | 5.2e-02 | +0.0009 [+0.0001; +0.0017] | 7.6e-02 | không |
| f1b_mac_d1 | anaes | cao → nguy cơ | 0.509 [0.507; 0.511] | +0.021 | +0.020 [+0.014; +0.025] | 1.91 [1.69; 2.16] | 5.3e-24 | +0.0008 [+0.0005; +0.0012] | 1.8e-05 | có ý nghĩa |
| f1c_drug_ppf_rate_d5 | anaes | thấp → nguy cơ | 0.518 [0.509; 0.527] | -0.017 | +0.021 [+0.014; +0.028] | 1.00 [1.00; 1.00] | 9.4e-02 | +0.0008 [+0.0002; +0.0015] | 2.9e-02 | không |
| f1b_ppf_d1 | anaes | thấp → nguy cơ | 0.507 [0.499; 0.514] | -0.007 | +0.018 [+0.010; +0.028] | 1.00 [1.00; 1.00] | 1.7e-04 | +0.0008 [+0.0002; +0.0013] | 2.3e-02 | có ý nghĩa |
| f1c_drug_ppf_cp | anaes | thấp → nguy cơ | 0.550 [0.533; 0.569] | -0.049 | +0.056 [+0.033; +0.077] | 0.96 [0.90; 1.03] | 4.0e-01 | +0.0008 [+0.0001; +0.0014] | 4.4e-02 | không |
| f1c_drug_ppf_rate_mean_m5 | anaes | thấp → nguy cơ | 0.507 [0.488; 0.526] | -0.007 | +0.025 [+0.002; +0.044] | 1.02 [0.98; 1.07] | 4.6e-01 | +0.0007 [+0.0001; +0.0014] | 7.3e-02 | không |
| f1c_drug_ppf_vol_m5 | anaes | thấp → nguy cơ | 0.507 [0.489; 0.526] | -0.007 | +0.025 [+0.002; +0.044] | 1.02 [0.98; 1.06] | 5.0e-01 | +0.0007 [+0.0001; +0.0014] | 7.4e-02 | không |
| f1b_ppf | anaes | thấp → nguy cơ | 0.551 [0.533; 0.571] | -0.050 | +0.052 [+0.028; +0.074] | 0.95 [0.89; 1.01] | 1.9e-01 | +0.0007 [-0.0000; +0.0013] | 9.4e-02 | không |
| f1c_drug_ppf_rate_d1 | anaes | thấp → nguy cơ | 0.513 [0.509; 0.518] | -0.013 | +0.015 [+0.010; +0.019] | 1.00 [1.00; 1.00] | 8.4e-01 | +0.0007 [+0.0001; +0.0012] | 4.9e-02 | không |
| etco2_std_w120 | numeric | cao → nguy cơ | 0.554 [0.544; 0.564] | +0.055 | -0.052 [-0.066; -0.040] | 1.02 [1.01; 1.03] | 2.0e-12 | +0.0006 [+0.0002; +0.0010] | 1.8e-02 | có ý nghĩa |
| f1c_vent_compliance | anaes | cao → nguy cơ | 0.581 [0.569; 0.596] | +0.083 | -0.077 [-0.091; -0.056] | 1.13 [1.07; 1.18] | 7.8e-06 | +0.0006 [+0.0001; +0.0011] | 6.4e-02 | không |
| ep_ppg_amp_rel600 | ecgppg | cao → nguy cơ | 0.519 [0.512; 0.524] | +0.019 | +0.001 [-0.004; +0.008] | 1.01 [1.00; 1.03] | 2.2e-01 | +0.0006 [+0.0003; +0.0011] | 2.7e-02 | không |

Mốc 10 phút: biến có ý nghĩa khi đứng riêng nhưng **không** thêm gì khi đã biết MAP (chủ yếu lặp lại thông tin MAP)

| Biến | Nhóm | Hướng | AUROC riêng [CI] | ρ với nhãn | ρ với phần dư MAP [CI] | OR/IQR [CI] | q(OR) | ΔAUROC [CI] | q(ΔAUROC) | Ngoài MAP |
|---|---|---|---|---|---|---|---|---|---|---|
| bt_svr_mean_w120 | wave | thấp → nguy cơ | 0.695 [0.683; 0.704] | -0.200 | +0.279 [+0.265; +0.295] | 0.87 [0.82; 0.92] | 1.6e-05 | +0.0003 [-0.0001; +0.0007] | 2.6e-01 | không |
| cx_dsi | composite | cao → nguy cơ | 0.688 [0.676; 0.697] | +0.192 | -0.268 [-0.280; -0.254] | 1.07 [1.02; 1.12] | 1.0e-02 | -0.0000 [-0.0004; +0.0003] | 9.6e-01 | không |
| cx_map_proj_5 | composite | thấp → nguy cơ | 0.679 [0.672; 0.686] | -0.183 | +0.263 [+0.253; +0.277] | 0.99 [0.97; 1.02] | 6.6e-01 | -0.0003 [-0.0004; -0.0001] | 8.0e-04 | không |
| cx_msi | composite | cao → nguy cơ | 0.667 [0.655; 0.676] | +0.171 | -0.238 [-0.251; -0.225] | 1.06 [1.02; 1.11] | 1.6e-02 | -0.0002 [-0.0005; +0.0001] | 3.3e-01 | không |
| cx_map_event_rate | composite | cao → nguy cơ | 0.653 [0.640; 0.665] | +0.162 | -0.246 [-0.264; -0.233] | 1.04 [0.99; 1.08] | 1.7e-01 | -0.0005 [-0.0008; -0.0001] | 3.4e-02 | không |
| cx_map_proj_10 | composite | thấp → nguy cơ | 0.642 [0.633; 0.650] | -0.143 | +0.216 [+0.206; +0.229] | 0.99 [0.96; 1.02] | 6.5e-01 | -0.0005 [-0.0006; -0.0003] | 2.6e-08 | không |
| cx_map_proj_15 | composite | thấp → nguy cơ | 0.632 [0.623; 0.641] | -0.132 | +0.212 [+0.203; +0.225] | 1.00 [0.97; 1.04] | 9.1e-01 | -0.0005 [-0.0006; -0.0003] | 5.9e-08 | không |
| cx_map_grey_burden | composite | cao → nguy cơ | 0.630 [0.619; 0.639] | +0.134 | -0.426 [-0.436; -0.412] | 1.00 [0.93; 1.07] | 9.5e-01 | -0.0007 [-0.0009; -0.0005] | 2.3e-09 | không |
| bt_sv_mean_w120 | wave | cao → nguy cơ | 0.628 [0.613; 0.642] | +0.131 | -0.167 [-0.185; -0.152] | 1.08 [1.03; 1.14] | 1.1e-02 | -0.0001 [-0.0005; +0.0001] | 5.3e-01 | không |
| shock_index_max_w120 | numeric | cao → nguy cơ | 0.623 [0.611; 0.635] | +0.126 | -0.202 [-0.218; -0.186] | 1.03 [0.99; 1.07] | 2.2e-01 | -0.0005 [-0.0007; -0.0003] | 6.1e-04 | không |
| cx_map_cv | composite | cao → nguy cơ | 0.622 [0.615; 0.630] | +0.125 | -0.212 [-0.223; -0.202] | 1.01 [0.99; 1.02] | 6.1e-01 | -0.0003 [-0.0004; -0.0002] | 8.0e-04 | không |
| cx_map_ttc_m5 | composite | thấp → nguy cơ | 0.622 [0.616; 0.628] | -0.137 | +0.126 [+0.119; +0.134] | 0.96 [0.93; 1.00] | 1.2e-01 | -0.0002 [-0.0003; -0.0001] | 1.6e-03 | không |
| bt_co_mean_w120 | wave | cao → nguy cơ | 0.621 [0.609; 0.632] | +0.124 | -0.152 [-0.168; -0.135] | 1.10 [1.05; 1.15] | 8.8e-05 | -0.0001 [-0.0005; +0.0002] | 6.6e-01 | không |
| bt_co_std_w120 | wave | cao → nguy cơ | 0.621 [0.611; 0.632] | +0.124 | -0.190 [-0.204; -0.179] | 1.02 [1.00; 1.03] | 5.2e-02 | -0.0003 [-0.0005; -0.0000] | 4.6e-02 | không |
| cx_drop_age | composite | cao → nguy cơ | 0.620 [0.606; 0.637] | +0.123 | -0.175 [-0.194; -0.155] | 1.01 [0.97; 1.06] | 6.6e-01 | -0.0005 [-0.0007; -0.0003] | 1.2e-04 | không |

Mốc 15 phút: 25 biến có ΔAUROC lớn nhất

| Biến | Nhóm | Hướng | AUROC riêng [CI] | ρ với nhãn | ρ với phần dư MAP [CI] | OR/IQR [CI] | q(OR) | ΔAUROC [CI] | q(ΔAUROC) | Ngoài MAP |
|---|---|---|---|---|---|---|---|---|---|---|
| cx_depth_d5 | composite | cao → nguy cơ | 0.516 [0.509; 0.522] | +0.020 | +0.062 [+0.054; +0.072] | 1.01 [1.01; 1.01] | 1.7e-28 | +0.0018 [+0.0012; +0.0025] | 7.9e-06 | có ý nghĩa |
| f1b_mac | anaes | cao → nguy cơ | 0.529 [0.518; 0.541] | +0.039 | +0.012 [-0.009; +0.033] | 1.38 [1.27; 1.50] | 1.6e-13 | +0.0017 [+0.0005; +0.0026] | 1.2e-02 | có ý nghĩa |
| f1b_rftn_d1 | anaes | thấp → nguy cơ | 0.500 [0.495; 0.507] | -0.001 | +0.082 [+0.075; +0.090] | 1.00 [1.00; 1.00] | 2.5e-42 | +0.0014 [+0.0008; +0.0020] | 2.0e-04 | có ý nghĩa |
| f1b_mac_d10 | anaes | cao → nguy cơ | 0.518 [0.513; 0.523] | +0.031 | +0.021 [+0.011; +0.029] | 2.04 [1.65; 2.52] | 6.5e-10 | +0.0014 [+0.0007; +0.0021] | 2.7e-03 | có ý nghĩa |
| f1b_rftn_d5 | anaes | cao → nguy cơ | 0.508 [0.500; 0.515] | +0.009 | +0.076 [+0.069; +0.084] | 1.00 [1.00; 1.00] | 4.7e-25 | +0.0013 [+0.0007; +0.0020] | 1.1e-03 | có ý nghĩa |
| f1c_drug_ppf_cp | anaes | thấp → nguy cơ | 0.559 [0.540; 0.578] | -0.067 | +0.047 [+0.023; +0.067] | 0.94 [0.87; 1.00] | 1.2e-01 | +0.0013 [+0.0004; +0.0022] | 1.1e-02 | không |
| f1b_mac_d5 | anaes | cao → nguy cơ | 0.519 [0.514; 0.524] | +0.037 | +0.028 [+0.021; +0.033] | 2.28 [1.93; 2.70] | 2.2e-20 | +0.0013 [+0.0008; +0.0019] | 5.1e-04 | có ý nghĩa |
| f1c_drug_ppf_vol | anaes | thấp → nguy cơ | 0.612 [0.594; 0.636] | -0.128 | +0.119 [+0.095; +0.146] | 0.92 [0.86; 0.99] | 5.0e-02 | +0.0013 [+0.0003; +0.0023] | 5.4e-02 | không |
| f1b_ppf_d10 | anaes | thấp → nguy cơ | 0.508 [0.496; 0.515] | -0.009 | +0.017 [+0.005; +0.030] | 1.00 [1.00; 1.00] | 1.1e-01 | +0.0012 [+0.0004; +0.0021] | 1.8e-02 | không |
| f1b_ppf_d1 | anaes | thấp → nguy cơ | 0.509 [0.503; 0.515] | -0.011 | +0.019 [+0.010; +0.027] | 1.00 [1.00; 1.00] | 5.0e-05 | +0.0012 [+0.0004; +0.0020] | 1.5e-02 | có ý nghĩa |
| cx_depth_d5_grey | composite | cao → nguy cơ | 0.505 [0.497; 0.512] | +0.007 | +0.089 [+0.081; +0.097] | 2.29 [1.96; 2.69] | 3.5e-23 | +0.0012 [+0.0008; +0.0016] | 3.3e-05 | có ý nghĩa |
| f1c_drug_ppf_rate_d5 | anaes | thấp → nguy cơ | 0.515 [0.507; 0.523] | -0.017 | +0.018 [+0.009; +0.024] | 1.00 [1.00; 1.00] | 5.7e-02 | +0.0012 [+0.0004; +0.0021] | 1.8e-02 | không |
| f1b_ppf | anaes | thấp → nguy cơ | 0.559 [0.540; 0.579] | -0.067 | +0.044 [+0.019; +0.065] | 0.92 [0.86; 0.99] | 5.2e-02 | +0.0012 [+0.0002; +0.0021] | 2.9e-02 | không |
| f1b_ppf_d5 | anaes | thấp → nguy cơ | 0.510 [0.502; 0.518] | -0.012 | +0.017 [+0.008; +0.028] | 1.00 [1.00; 1.00] | 3.9e-02 | +0.0012 [+0.0004; +0.0021] | 1.8e-02 | có ý nghĩa |
| f1c_drug_ppf_vol_m5 | anaes | thấp → nguy cơ | 0.517 [0.495; 0.535] | -0.019 | +0.027 [+0.006; +0.044] | 1.00 [0.96; 1.05] | 8.6e-01 | +0.0012 [+0.0004; +0.0021] | 2.9e-02 | không |
| f1b_rftn_d10 | anaes | cao → nguy cơ | 0.513 [0.505; 0.519] | +0.015 | +0.069 [+0.060; +0.079] | 1.01 [1.01; 1.02] | 1.7e-16 | +0.0011 [+0.0005; +0.0019] | 1.1e-02 | có ý nghĩa |
| f1c_drug_ppf_rate_mean_m5 | anaes | thấp → nguy cơ | 0.517 [0.495; 0.535] | -0.019 | +0.027 [+0.005; +0.045] | 1.01 [0.96; 1.05] | 8.4e-01 | +0.0011 [+0.0003; +0.0020] | 3.3e-02 | không |
| f1c_drug_ppf_rate_d1 | anaes | thấp → nguy cơ | 0.512 [0.508; 0.516] | -0.013 | +0.011 [+0.007; +0.015] | 1.00 [1.00; 1.00] | 7.0e-01 | +0.0011 [+0.0004; +0.0019] | 1.8e-02 | không |
| f1a_min_since_opstart | context | cao → nguy cơ | 0.517 [0.500; 0.530] | +0.019 | +0.014 [-0.006; +0.034] | 0.98 [0.93; 1.03] | 5.2e-01 | +0.0011 [+0.0002; +0.0020] | 2.7e-02 | không |
| f1c_drug_exp_sevo | anaes | cao → nguy cơ | 0.517 [0.501; 0.537] | +0.022 | +0.022 [+0.000; +0.047] | 1.41 [1.26; 1.56] | 4.1e-09 | +0.0010 [+0.0001; +0.0018] | 6.0e-02 | không |
| f1c_drug_ppf_rate | anaes | thấp → nguy cơ | 0.529 [0.511; 0.548] | -0.033 | +0.035 [+0.014; +0.053] | 1.00 [0.96; 1.03] | 8.4e-01 | +0.0010 [+0.0002; +0.0017] | 5.4e-02 | không |
| f1c_drug_exp_sevo_d5 | anaes | cao → nguy cơ | 0.519 [0.513; 0.526] | +0.029 | +0.032 [+0.022; +0.043] | 2.08 [1.66; 2.61] | 2.7e-09 | +0.0009 [+0.0003; +0.0014] | 1.5e-02 | có ý nghĩa |
| f1c_vent_compliance | anaes | cao → nguy cơ | 0.582 [0.569; 0.598] | +0.098 | -0.051 [-0.064; -0.030] | 1.15 [1.09; 1.20] | 1.3e-06 | +0.0008 [+0.0002; +0.0015] | 7.2e-02 | không |
| f1c_drug_ppf_vol_m15 | anaes | thấp → nguy cơ | 0.522 [0.502; 0.543] | -0.024 | +0.029 [-0.001; +0.048] | 1.00 [0.95; 1.05] | 9.6e-01 | +0.0008 [-0.0000; +0.0016] | 1.5e-01 | không |
| f1b_mac_d1 | anaes | cao → nguy cơ | 0.508 [0.506; 0.510] | +0.022 | +0.022 [+0.017; +0.026] | 1.82 [1.63; 2.03] | 6.0e-25 | +0.0007 [+0.0004; +0.0010] | 6.3e-04 | có ý nghĩa |

Mốc 15 phút: biến có ý nghĩa khi đứng riêng nhưng **không** thêm gì khi đã biết MAP (chủ yếu lặp lại thông tin MAP)

| Biến | Nhóm | Hướng | AUROC riêng [CI] | ρ với nhãn | ρ với phần dư MAP [CI] | OR/IQR [CI] | q(OR) | ΔAUROC [CI] | q(ΔAUROC) | Ngoài MAP |
|---|---|---|---|---|---|---|---|---|---|---|
| bt_svr_mean_w120 | wave | thấp → nguy cơ | 0.687 [0.676; 0.696] | -0.223 | +0.221 [+0.206; +0.238] | 0.88 [0.83; 0.94] | 3.3e-04 | +0.0001 [-0.0003; +0.0005] | 6.5e-01 | không |
| cx_dsi | composite | cao → nguy cơ | 0.675 [0.662; 0.684] | +0.208 | -0.211 [-0.223; -0.197] | 1.06 [1.01; 1.12] | 3.1e-02 | -0.0002 [-0.0005; +0.0001] | 4.3e-01 | không |
| cx_map_proj_5 | composite | thấp → nguy cơ | 0.660 [0.653; 0.666] | -0.190 | +0.203 [+0.191; +0.217] | 0.99 [0.96; 1.01] | 3.5e-01 | -0.0002 [-0.0004; -0.0001] | 9.1e-03 | không |
| cx_msi | composite | cao → nguy cơ | 0.654 [0.642; 0.662] | +0.184 | -0.186 [-0.197; -0.172] | 1.06 [1.01; 1.12] | 3.4e-02 | -0.0002 [-0.0004; +0.0002] | 4.7e-01 | không |
| cx_map_event_rate | composite | cao → nguy cơ | 0.653 [0.639; 0.665] | +0.189 | -0.207 [-0.226; -0.193] | 1.02 [0.98; 1.07] | 4.1e-01 | -0.0005 [-0.0009; -0.0002] | 1.8e-02 | không |
| cx_map_grey_burden | composite | cao → nguy cơ | 0.633 [0.622; 0.642] | +0.159 | -0.366 [-0.376; -0.351] | 1.00 [0.94; 1.07] | 9.2e-01 | -0.0006 [-0.0008; -0.0005] | 1.2e-06 | không |
| bt_sv_mean_w120 | wave | cao → nguy cơ | 0.627 [0.612; 0.641] | +0.152 | -0.134 [-0.153; -0.118] | 1.08 [1.02; 1.14] | 1.8e-02 | -0.0002 [-0.0006; +0.0002] | 5.2e-01 | không |
| cx_map_proj_10 | composite | thấp → nguy cơ | 0.625 [0.618; 0.632] | -0.148 | +0.171 [+0.161; +0.185] | 0.99 [0.96; 1.02] | 5.1e-01 | -0.0004 [-0.0006; -0.0003] | 1.5e-06 | không |
| cx_map_proj_15 | composite | thấp → nguy cơ | 0.618 [0.611; 0.627] | -0.139 | +0.170 [+0.160; +0.184] | 1.00 [0.97; 1.04] | 9.2e-01 | -0.0005 [-0.0007; -0.0003] | 1.2e-06 | không |
| bt_co_mean_w120 | wave | cao → nguy cơ | 0.617 [0.606; 0.629] | +0.140 | -0.120 [-0.137; -0.104] | 1.09 [1.04; 1.14] | 1.0e-03 | -0.0003 [-0.0007; +0.0001] | 2.2e-01 | không |
| shock_index_max_w120 | numeric | cao → nguy cơ | 0.614 [0.603; 0.625] | +0.136 | -0.157 [-0.172; -0.141] | 1.03 [0.99; 1.08] | 2.3e-01 | -0.0004 [-0.0006; -0.0001] | 2.8e-02 | không |
| bt_co_std_w120 | wave | cao → nguy cơ | 0.612 [0.601; 0.624] | +0.134 | -0.145 [-0.158; -0.133] | 1.01 [1.00; 1.02] | 2.0e-01 | -0.0004 [-0.0006; -0.0002] | 6.4e-04 | không |
| cx_drop_age | composite | cao → nguy cơ | 0.612 [0.598; 0.626] | +0.133 | -0.138 [-0.157; -0.119] | 1.02 [0.97; 1.06] | 5.5e-01 | -0.0006 [-0.0009; -0.0004] | 1.6e-04 | không |
| f1c_drug_ppf_vol | anaes | thấp → nguy cơ | 0.612 [0.594; 0.636] | -0.128 | +0.119 [+0.095; +0.146] | 0.92 [0.86; 0.99] | 5.0e-02 | +0.0013 [+0.0003; +0.0023] | 5.4e-02 | không |
| cx_map_cv | composite | cao → nguy cơ | 0.611 [0.603; 0.620] | +0.133 | -0.153 [-0.163; -0.141] | 1.01 [0.99; 1.03] | 3.4e-01 | -0.0003 [-0.0005; -0.0002] | 9.7e-04 | không |

## 4. Cặp biến: yếu khi riêng, mạnh khi cùng nhau

Synergy = Δlog-loss(a, b) − Δlog-loss(a) − Δlog-loss(b), tất cả trên nền chỉ-MAP; > 0 nghĩa là hai biến cùng nhau thêm nhiều hơn tổng từng biến. "Ổn định" = dương ở ≥ 80% số fold × mốc.

| A | B | Synergy 5' (×1e4) | Synergy 15' (×1e4) | ΔAUROC cặp 15' | Ổn định |
|---|---|---|---|---|---|
| f1b_rftn_d10 | f1a_min_since_opstart | 5.03 | 8.62 | +0.0032 | có |
| f1b_mac | f1a_min_since_opstart | 3.09 | 9.95 | +0.0045 | có |
| f1c_drug_rftn_vol_m5 | f1c_drug_ppf_rate_d1 | 5.72 | 7.03 | +0.0018 | có |
| f1c_drug_rftn_vol_m5 | f1b_ppf_d5 | 5.81 | 6.83 | +0.0018 | có |
| f1c_drug_ppf_rate_d5 | f1c_drug_rftn_vol_m5 | 5.65 | 6.87 | +0.0019 | có |
| f1b_ppf_d1 | f1c_drug_rftn_vol_m5 | 5.47 | 6.46 | +0.0018 | có |
| f1b_mac | f1c_drug_rftn_vol_m5 | 5.24 | 6.53 | +0.0024 | có |
| f1c_drug_rftn_vol_m5 | f1b_ppf_d10 | 4.68 | 6.72 | +0.0019 | có |
| f1a_min_since_opstart | f1c_drug_exp_sevo | 2.00 | 9.10 | +0.0038 | có |
| f1b_rftn_d1 | f1a_min_since_opstart | 3.24 | 7.02 | +0.0035 | có |
| f1b_mac | f1c_vent_pplat | 3.63 | 6.25 | +0.0027 | có |
| f1c_drug_exp_sevo | f1c_drug_rftn_vol_m5 | 3.62 | 5.95 | +0.0018 | có |
| cx_depth_d5 | f1a_min_since_opstart | 4.01 | 5.56 | +0.0036 | có |
| f1a_min_since_opstart | f1c_drug_exp_sevo_d5 | 2.43 | 6.97 | +0.0032 | có |
| f1b_rftn_d10 | cx_pre_incision | 2.63 | 6.41 | +0.0020 | có |
| f1a_min_since_opstart | f1c_drug_ppf_rate_d1 | 1.89 | 6.92 | +0.0034 | có |
| f1a_min_since_opstart | f1b_ppf_d5 | 1.64 | 6.59 | +0.0034 | có |
| f1b_rftn_d1 | f1c_drug_ppf_rate_d5 | 3.21 | 4.81 | +0.0033 | có |
| f1a_min_since_opstart | f1b_ppf_d1 | 1.51 | 6.47 | +0.0034 | có |
| f1a_min_since_opstart | f1b_approach_open | 1.74 | 6.21 | +0.0025 | có |

### Biến điều biến tác động của MAP (SHAP interaction, mốc 15 phút)

Interaction lớn = cùng một mức MAP nhưng nguy cơ khác nhau tùy biến này (xem `figures/l2_map_modulation_h900.png`).

| A | B | mean |interaction| |
|---|---|---|
| f1a_min_since_last_event | f1a_min_since_opstart | 0.0487 |
| f1a_n_prev_events | f1a_min_since_opstart | 0.0317 |
| f1a_map_min_m15 | f1a_min_since_opstart | 0.0118 |
| f1a_min_since_last_event | f1b_approach_open | 0.0111 |
| dbp_current | f1b_rftn_d10 | 0.0105 |
| f1a_map_min_m30 | f1c_vent_pplat | 0.0101 |
| f1a_map_min_m10 | f1a_min_since_opstart | 0.0097 |
| f1a_min_since_last_event | etco2_std_w120 | 0.0092 |
| f1a_min_since_last_event | etco2_min_w120 | 0.0091 |
| map_extrap_w120 | f1b_rftn_d1 | 0.0088 |
| f1a_n_prev_events | f1b_mac | 0.0086 |
| f1a_map_min_m30 | f1b_approach_open | 0.0084 |
| f1a_map_rel_max_m15 | f1b_rftn_d1 | 0.0083 |
| dbp_current | f1a_min_since_opstart | 0.0080 |
| f1a_map_min_m30 | f1a_min_since_opstart | 0.0079 |

## 5. Đặc trưng tổ hợp từ suy luận sinh lý

Dấu kỳ vọng: +1 = cao hơn thì nguy cơ cao hơn. Dấu quan sát lấy từ ρ riêng phần mốc 15 phút; "đúng" = cùng dấu và CI không chứa 0.

| Tổ hợp | Giả thuyết | Kỳ vọng | ρ riêng phần [CI] | Khớp sinh lý | ΔAUROC 5'/15' |
|---|---|---|---|---|---|
| cx_depth_d5 | Thay đổi độ sâu mê gộp 5 phút | 1 | 0.062 [0.054; 0.072] | đúng | +0.0023 / +0.0020 |
| cx_depth_d5_grey | Tăng độ sâu mê khi MAP đang ở vùng xám | 1 | 0.089 [0.081; 0.097] | đúng | +0.0014 / +0.0014 |
| cx_pre_incision | Trước rạch da (chưa có kích thích phẫu thuật) [0/1] | 1 | -0.115 [-0.129; -0.098] | ngược | +0.0001 / +0.0001 |
| cx_pre_incision_min | Số phút từ khởi mê khi chưa rạch da | 0 | 0.065 [0.039; 0.088] | không có kỳ vọng | +0.0000 / +0.0002 |
| cx_etco2_drop | EtCO2 TB 5 phút − 15 phút (giảm = cung lượng tim giảm) | -1 | -0.005 [-0.015; 0.002] | không có ý nghĩa | +0.0000 / +0.0004 |
| cx_etco2_map_coslope | Độ dốc EtCO2 × độ dốc MAP khi cả hai giảm | 1 | -0.029 [-0.035; -0.022] | ngược | +0.0000 / -0.0000 |
| cx_dlog_hr | Δlog HR (TB 5 phút so với 15 phút) | 0 | 0.024 [0.016; 0.032] | không có kỳ vọng | +0.0001 / +0.0004 |
| cx_no_hr_comp | MAP giảm mà HR không tăng (mất bù) [0/1] | 1 | -0.016 [-0.023; -0.009] | ngược | -0.0000 / -0.0000 |
| cx_dlog_sv | Δlog SV proxy (Liljestrand-Zander) 5 vs 15 phút | -1 | 0.012 [0.003; 0.020] | ngược | -0.0002 / +0.0001 |
| cx_map_cv | Hệ số biến thiên MAP trong W (bất ổn) | 1 | -0.153 [-0.163; -0.141] | ngược | -0.0001 / -0.0001 |
| cx_dsi | Chỉ số sốc tâm trương HR/DBP | 1 | -0.211 [-0.223; -0.197] | ngược | +0.0002 / +0.0001 |
| cx_msi | Chỉ số sốc hiệu chỉnh HR/MAP | 1 | -0.186 [-0.197; -0.172] | ngược | +0.0001 / +0.0001 |
| cx_bis_low_grey | BIS < 40 khi MAP ở vùng xám [0/1] | 1 | -0.150 [-0.164; -0.138] | ngược | +0.0000 / +0.0000 |
| cx_etco2_per_mv | EtCO2 / thông khí phút (CO2 đến phổi theo thông khí) | -1 | -0.034 [-0.054; -0.017] | đúng | -0.0001 / -0.0002 |
| cx_map_ttc_m5 | Thời gian (phút) để MAP chạm 65 theo độ dốc 5 phút | -1 | 0.085 [0.078; 0.095] | ngược | -0.0001 / -0.0001 |
| cx_dlog_svr | Δlog SVR proxy = Δlog MAP − Δlog HR − Δlog SV | -1 | -0.029 [-0.040; -0.021] | đúng | -0.0002 / +0.0001 |
| cx_baro_gain | ΔHR/ΔMAP khi MAP đang giảm (bù phản xạ áp cảm thụ; âm = có bù) | 1 | -0.025 [-0.035; -0.016] | ngược | -0.0000 / +0.0000 |
| cx_map_ttc_m15 | Thời gian (phút) để MAP chạm 65 theo độ dốc 15 phút | -1 | 0.104 [0.095; 0.114] | ngược | -0.0003 / -0.0001 |
| cx_no_hr_comp_age | Mất bù HR × tuổi/60 | 1 | -0.022 [-0.029; -0.015] | ngược | -0.0001 / -0.0001 |
| cx_ppv_grey | PPV khi MAP ở vùng xám (phụ thuộc tiền tải) | 1 | -0.020 [-0.035; -0.005] | ngược | -0.0002 / -0.0001 |
| cx_ppv_x_dp | PPV × áp lực đẩy (thông khí làm giảm hồi lưu tĩnh mạch) | 1 | -0.011 [-0.030; 0.009] | không có ý nghĩa | -0.0003 / -0.0001 |
| cx_map_event_rate | Số đợt tụt mỗi giờ gây mê | 1 | -0.207 [-0.226; -0.193] | ngược | -0.0002 / -0.0004 |
| cx_map_proj_5 | MAP dự phóng 5 phút (MAP + độ dốc 5 phút × 5) | -1 | 0.203 [0.191; 0.217] | ngược | -0.0002 / -0.0000 |
| cx_map_proj_15 | MAP dự phóng 15 phút (độ dốc 15 phút) | -1 | 0.170 [0.160; 0.184] | ngược | -0.0002 / -0.0002 |
| cx_drop_age | % MAP giảm so với nền × tuổi/60 | 1 | -0.138 [-0.157; -0.119] | ngược | -0.0002 / -0.0004 |
| cx_drop_htn | % MAP giảm so với nền × tăng huyết áp (dịch ngưỡng tự điều hòa) | 1 | -0.047 [-0.066; -0.030] | ngược | -0.0001 / -0.0004 |
| cx_map_proj_10 | MAP dự phóng 10 phút (độ dốc 10 phút) | -1 | 0.171 [0.161; 0.185] | ngược | -0.0002 / -0.0002 |
| cx_depth | Chỉ số độ sâu mê gộp: Ce ppf/3 + Ce rftn/4 + MAC | 1 | 0.108 [0.088; 0.129] | đúng | -0.0002 / +0.0000 |
| cx_map_grey_burden | Phút MAP ở vùng xám 65-80 (15 phút) + 65-75 (W) | 1 | -0.366 [-0.376; -0.351] | ngược | -0.0004 / -0.0004 |
| cx_depth_age | Độ sâu mê gộp × tuổi/60 (người già nhạy thuốc) | 1 | 0.042 [0.022; 0.064] | đúng | -0.0003 / -0.0007 |

## 6. Chọn đặc trưng tham lam (train CV, quyết định giữ / bỏ)

Mỗi bước thêm bundle (một biến hoặc một cặp ổn định) tốt nhất vào lõi MAP. Điểm = (ΔAUROC + ΔAUPRC)/2 so với chỉ-MAP, trung bình mốc 5 và 15 phút, 3 fold. Dừng khi bước tốt nhất thêm < 0,001 hoặc không dương ở mọi fold.

| Bước | Thêm | Điểm trước → sau | Thêm được | Kết luận |
|---|---|---|---|---|
| 1 | f1b_rftn_d1 + f1a_min_since_opstart | 0.0000 → 0.0064 | +0.0064 | giữ |
| 2 | f1b_mac + f1c_drug_rftn_vol_m5 | 0.0064 → 0.0115 | +0.0051 | giữ |
| 3 | f1b_approach_open | 0.0115 → 0.0130 | +0.0015 | bỏ (not positive in every fold) |

Mô hình LightGBM đầy đủ trên lõi MAP + k bundle đầu (AUROC CV train, 2 fold):

| k | Số biến | Thêm | 5' | 10' | 15' |
|---|---|---|---|---|---|
| 0 | 41 | (chỉ lõi MAP) | 0.8910 | 0.8521 | 0.8327 |
| 1 | 43 | f1b_rftn_d1 f1a_min_since_opstart | 0.8948 | 0.8583 | 0.8386 |
| 2 | 45 | f1b_mac f1c_drug_rftn_vol_m5 | 0.8975 | 0.8610 | 0.8415 |

## 7. Trước / sau trên validation, xuất phát từ mô hình hiện tại `lgbm_context`

Thêm lần lượt các bundle theo thứ tự đã chọn trên train. Bước xấu hơn chỉ được ghi chú, không mang sang bước sau. Protocol: mọi thời điểm (của nhóm) và Lee/HypoBridCast.

| Bước | Thêm | Số biến | Train | AUROC 5' (mọi t.đ.) | AUROC 10' (mọi t.đ.) | AUROC 15' (mọi t.đ.) | AUROC 15' (Lee) | AUPRC 15' | Kết luận |
|---|---|---|---|---|---|---|---|---|---|
| 0 | lgbm_context (97) | 97 | giữ | 0.8855 | 0.8437 | 0.8220 | 0.9158 | 0.4378 | mốc |
| 1 | f1b_rftn_d1 + f1a_min_since_opstart | 98 | giữ | 0.8868 (+0.0014) | 0.8456 (+0.0018) | 0.8236 (+0.0016) | 0.9158 | 0.4455 | tốt hơn |
| 2 | f1b_mac + f1c_drug_rftn_vol_m5 | 100 | giữ | 0.8885 (+0.0017) | 0.8460 (+0.0004) | 0.8252 (+0.0016) | 0.9172 | 0.4444 | tốt hơn |
| 3 | f1b_approach_open | 101 | bỏ (dừng) | 0.8900 (+0.0015) | 0.8483 (+0.0024) | 0.8276 (+0.0024) | 0.9195 | 0.4498 | tốt hơn |

## 8. So sánh cuối trên validation (CI 95% bootstrap theo bệnh nhân)

### Mọi thời điểm (protocol của nhóm)

| Mô hình | 5' | 10' | 15' | Tăng so với chỉ-MAP 15' |
|---|---|---|---|---|
| MAP hiện tại (luật) | 0.807 [0.789; 0.825] | 0.748 [0.726; 0.770] | 0.723 [0.698; 0.748] | -0.092 |
| MAP-only (lõi MAP) | 0.880 [0.865; 0.893] | 0.836 [0.817; 0.855] | 0.815 [0.794; 0.836] | +0.000 |
| lgbm_context (97) | 0.885 [0.870; 0.899] | 0.844 [0.825; 0.861] | 0.822 [0.800; 0.843] | +0.007 |
| Tất cả 214 | 0.890 [0.874; 0.903] | 0.850 [0.831; 0.868] | 0.830 [0.810; 0.852] | +0.015 |
| Tất cả 214 + tổ hợp | 0.890 [0.875; 0.904] | 0.851 [0.832; 0.869] | 0.830 [0.810; 0.853] | +0.016 |
| Chọn lọc (45) | 0.886 [0.871; 0.900] | 0.844 [0.825; 0.863] | 0.822 [0.802; 0.842] | +0.007 |
| lgbm_context + chọn lọc | 0.888 [0.873; 0.902] | 0.846 [0.828; 0.863] | 0.825 [0.805; 0.845] | +0.010 |

### Lee 2021 / HypoBridCast 2026

| Mô hình | 5' | 10' | 15' | Tăng so với chỉ-MAP 15' |
|---|---|---|---|---|
| MAP hiện tại (luật) | 0.848 [0.817; 0.872] | 0.818 [0.790; 0.844] | 0.817 [0.790; 0.847] | -0.081 |
| MAP-only (lõi MAP) | 0.927 [0.913; 0.941] | 0.907 [0.890; 0.925] | 0.898 [0.877; 0.918] | +0.000 |
| lgbm_context (97) | 0.933 [0.919; 0.946] | 0.912 [0.896; 0.929] | 0.904 [0.885; 0.923] | +0.006 |
| Tất cả 214 | 0.938 [0.925; 0.949] | 0.915 [0.899; 0.931] | 0.908 [0.888; 0.927] | +0.009 |
| Tất cả 214 + tổ hợp | 0.937 [0.924; 0.948] | 0.916 [0.899; 0.932] | 0.908 [0.889; 0.927] | +0.009 |
| Chọn lọc (45) | 0.933 [0.920; 0.946] | 0.912 [0.895; 0.928] | 0.903 [0.884; 0.923] | +0.005 |
| lgbm_context + chọn lọc | 0.935 [0.923; 0.947] | 0.913 [0.897; 0.931] | 0.906 [0.887; 0.924] | +0.007 |

### Hatib 2018 (HPI)

| Mô hình | 5' | 10' | 15' | Tăng so với chỉ-MAP 15' |
|---|---|---|---|---|
| MAP hiện tại (luật) | 0.920 [0.900; 0.938] | 0.908 [0.885; 0.928] | 0.906 [0.878; 0.926] | -0.070 |
| MAP-only (lõi MAP) | 0.984 [0.977; 0.990] | 0.980 [0.971; 0.987] | 0.976 [0.965; 0.984] | +0.000 |
| lgbm_context (97) | 0.986 [0.979; 0.991] | 0.983 [0.975; 0.989] | 0.979 [0.969; 0.987] | +0.003 |
| Tất cả 214 | 0.988 [0.982; 0.993] | 0.984 [0.976; 0.991] | 0.981 [0.971; 0.987] | +0.005 |
| Tất cả 214 + tổ hợp | 0.987 [0.981; 0.992] | 0.984 [0.976; 0.991] | 0.981 [0.972; 0.987] | +0.005 |
| Chọn lọc (45) | 0.987 [0.981; 0.992] | 0.983 [0.976; 0.989] | 0.979 [0.969; 0.986] | +0.003 |
| lgbm_context + chọn lọc | 0.987 [0.980; 0.992] | 0.984 [0.976; 0.990] | 0.979 [0.969; 0.987] | +0.003 |

Chênh lệch so với `lgbm_context` (bootstrap ghép cặp):

| Protocol | Mốc | Mô hình | ΔAUROC [CI] |
|---|---|---|---|
| every | 5 | Chọn lọc (45) | +0.0007 [-0.0019; +0.0034] |
| every | 5 | lgbm_context + chọn lọc | +0.0030 [+0.0010; +0.0056] |
| every | 10 | Chọn lọc (45) | -0.0000 [-0.0040; +0.0043] |
| every | 10 | lgbm_context + chọn lọc | +0.0023 [-0.0007; +0.0059] |
| every | 15 | Chọn lọc (45) | +0.0002 [-0.0048; +0.0055] |
| every | 15 | lgbm_context + chọn lọc | +0.0032 [-0.0010; +0.0079] |
| hatib | 5 | Chọn lọc (45) | +0.0016 [-0.0006; +0.0049] |
| hatib | 5 | lgbm_context + chọn lọc | +0.0014 [-0.0006; +0.0043] |
| hatib | 10 | Chọn lọc (45) | +0.0002 [-0.0026; +0.0029] |
| hatib | 10 | lgbm_context + chọn lọc | +0.0008 [-0.0017; +0.0035] |
| hatib | 15 | Chọn lọc (45) | -0.0005 [-0.0047; +0.0040] |
| hatib | 15 | lgbm_context + chọn lọc | +0.0001 [-0.0030; +0.0029] |
| lee | 5 | Chọn lọc (45) | +0.0005 [-0.0022; +0.0044] |
| lee | 5 | lgbm_context + chọn lọc | +0.0027 [+0.0004; +0.0053] |
| lee | 10 | Chọn lọc (45) | -0.0004 [-0.0042; +0.0040] |
| lee | 10 | lgbm_context + chọn lọc | +0.0014 [-0.0021; +0.0052] |
| lee | 15 | Chọn lọc (45) | -0.0007 [-0.0052; +0.0037] |
| lee | 15 | lgbm_context + chọn lọc | +0.0017 [-0.0013; +0.0050] |

## 9. Chi phí suy luận (realtime)

Một dòng mỗi 30 giây ở giường bệnh; đo trên CPU Kaggle, một luồng gọi `predict` cho từng dòng.

| Mô hình | Số biến | ms / dòng (trung vị) | ms / dòng (p95) | dòng / s (lô) | Kích thước (kB) |
|---|---|---|---|---|---|
| MAP-only (lõi MAP) | 41 | 0.092 | 0.114 | 140,098 | 458 |
| lgbm_context (97) | 97 | 0.094 | 0.115 | 119,516 | 461 |
| Tất cả 214 | 214 | 0.108 | 0.131 | 115,327 | 471 |
| Tất cả 214 + tổ hợp | 244 | 0.105 | 0.128 | 116,156 | 473 |
| Chọn lọc (45) | 45 | 0.093 | 0.114 | 139,621 | 459 |
| lgbm_context + chọn lọc | 100 | 0.097 | 0.118 | 126,941 | 462 |
