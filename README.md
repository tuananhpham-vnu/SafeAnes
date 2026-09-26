# SafeAnes — UC04

UC04: dự báo sớm tụt huyết áp trong mổ (IOH: MAP < 65 mmHg ≥ 60 s), gợi ý nguyên nhân và ước lượng độ tin cậy, trên toàn bộ
VitalDB (6.388 ca; 3.626 ca eligible). Global final test (`unseen_test`) được khóa cho tới khi mô hình, calibrator, ngưỡng và
chính sách cảnh báo đã cố định.

**Tài liệu chính**
- [Insight dữ liệu VitalDB: mô tả, nhiễu, quan hệ, feature ảnh hưởng](reports/EDA/INSIGHTS.md)
- [EDA toàn cohort](reports/EDA/REPORT.md) · [notebook](notebooks/05_eda_vitaldb.ipynb)
- [Đề xuất phương pháp HemoDecomp](docs/UC04_METHOD_PROPOSAL.md) (dự báo + phân tách nguyên nhân + độ tin cậy)
- [Kế hoạch nghiên cứu và yêu cầu độ chính xác](UC04_RESEARCH_PLAN.md)
- [Protocol](docs/PROTOCOL.md) · [Data card](docs/DATA_CARD.md) · [Model card](docs/MODEL_CARD.md) · [Nguồn](docs/SOURCES.md)
- E07 đánh giá toàn VitalDB: [plan](docs/experiments/E07_PLAN.md), [hiệu chỉnh](docs/experiments/E07_CORRECTION.md)
- [E08: so sánh phương pháp trên cùng quy trình](#e08) · [plan](docs/experiments/E08_PLAN.md) · [lịch sử version](scripts/e08/README.md)

**Bố cục repo (dọn ngày 24/09/2026).** Đã xóa pipeline pilot/development300 (E01–E06, v0.3/v0.4): tập con dữ liệu, script,
notebook 01–04, configs, báo cáo và artifacts tương ứng. Code/tài liệu đã commit vẫn còn trong lịch sử git (mốc v0.1 và các
commit trước 24/09/2026); báo cáo/artifacts chưa từng commit của E01–E06 đã bị xóa hẳn. Giữ lại: E06 backbone TabM
(`artifacts/E06`, cho E08 `TABM=frozen`), E07, E08, EDA.

| Thư mục | Nội dung |
|---|---|
| `data/vitaldb_full/meta` | `cases`, `trks`, cohort manifest, `uc04_tracks.csv`, labs |
| `data/vitaldb_full/raw` | Track số VitalDB (`<tid>.csv.gz` + `.source.json` có SHA-256) |
| `data/vitaldb_full/cases` | Decision row 30 s, nhãn y5/y10, biến cố theo ca (E07) |
| `data/sequences_full` | Chuỗi 2 s của 7 track lõi |
| `data/beats_full` | Beat features 2 s từ waveform ART |
| `reports/E07`, `reports/E08`, `reports/EDA` | Kết quả đánh giá toàn VitalDB, so sánh phương pháp, EDA/insight |

<!-- e08:begin -->
<a id="e08"></a>

## E08 v2 — so sánh phương pháp trên development300

Mọi phương pháp đi qua đúng một quy trình, chỉ khác nhau ở bản thân phương pháp; FA/giờ đi kèm chính sách `recall_first` đã được chấp nhận. Sinh tự động bởi [`scripts/e08/run_comparison.py`](scripts/e08/run_comparison.py) từ chính các tham số đã chạy. [Báo cáo đầy đủ](reports/E08/METHOD_COMPARISON.md) · [các version và lỗi đã sửa](scripts/e08/README.md)

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
<!-- e08:end -->

## Chạy lại

Python ≥ 3.11: `python -m pip install -e ".[dev,boosting,plots]"`, rồi `python -m pytest -q`. Tải dữ liệu cần Internet và
resume được (chạy lại lệnh); mọi file raw được kiểm SHA-256.

```sh
python scripts/e08/build_sequences_full.py             # chuỗi 2 s cho 7 track lõi
python scripts/data/fetch_uc04_tracks.py --workers 8    # track UC04 + labs
python scripts/data/extract_beats.py --workers 8        # beat features từ SNUADC/ART (không lưu waveform thô)
python scripts/eda/run_eda.py                           # reports/EDA/REPORT.md
python scripts/eda/run_insights.py                      # reports/EDA/INSIGHTS.md
python scripts/e08/run_comparison.py --dataset full     # E08 v3 trên toàn VitalDB (trừ global test)
```

Trên Windows, đặt `PYTHONIOENCODING=utf-8` khi chuyển log ra file (log tiếng Việt). Các script đánh giá E07
(`scripts/evaluate_full_vitaldb.py`, `continue_full_vitaldb.py`, `report_full_vitaldb.py`) dùng model E05 và tập con
development300 đã xóa, nên chỉ còn giá trị tham khảo; kết quả E07 vẫn ở `reports/E07`.

<a id="release"></a>

## Version và đợt thí nghiệm

Git có mốc `v0.1`; v0.2 đang phát triển, chưa push. Mỗi lần push mới tạo một version; thí nghiệm dùng mã E01, E02…

| ID | Nội dung | Trạng thái |
|---|---|---|
| E01–E06 | Pilot 60 ca, TCN/Transformer, Inception/TimesNet, CatBoost, development 300 ca, TabM | Đã xóa khỏi repo (xem lịch sử git) |
| E07 | Đánh giá toàn VitalDB, audit cohort | [plan](docs/experiments/E07_PLAN.md), `reports/E07` |
| E08 | So sánh phương pháp trên một quy trình | [plan](docs/experiments/E08_PLAN.md), [kết quả](#e08) |
| EDA | EDA + insight toàn cohort cho UC04 | [REPORT](reports/EDA/REPORT.md), [INSIGHTS](reports/EDA/INSIGHTS.md) |
