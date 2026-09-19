# SafeAnes

**E07 đang đánh giá toàn bộ VitalDB theo yêu cầu ngày 19/09/2026:** audit 6.388 ca,
3.626 ca đủ điều kiện; model/ngưỡng được khóa trước chạy. Global final test được mở
trong đợt này; các ghi chú “chưa mở final test” ở báo cáo E01–E06 là trạng thái lịch sử.
[Plan E07](docs/experiments/E07_PLAN.md) · [Tiến độ](reports/E07/progress.json).

UC04: dự báo sớm tụt huyết áp trong mổ. **v0.2 đang phát triển, chưa push**, gộp toàn bộ mô hình mới, sửa lỗi và thí nghiệm hiện tại: TCN/Transformer, Inception/TimesNet, ensemble, CatBoost và mở rộng development. Mỗi lần push mới tăng release; mỗi thí nghiệm dùng mã E01, E02… Global final test vẫn chưa sử dụng.

- [Kế hoạch và yêu cầu độ chính xác](UC04_RESEARCH_PLAN.md)
- [Báo cáo tổng hợp v0.2: 300 ca, ba seed và ablation](reports/v0_2/REPORT.md)
- [Các bước tìm hiểu/triển khai](docs/IMPLEMENTATION_STEPS.md)
- [Tiến độ và các mốc còn thiếu của v0.2](docs/IMPLEMENTATION_STEPS.md#tien-do)
- [Nguồn và ánh xạ vào code](docs/SOURCES.md)
- [Protocol và giới hạn](docs/PROTOCOL.md)
- [Hướng dẫn TCN/Transformer, resume và ablation](docs/SEQUENCE_RUNBOOK.md)
- [Model card](docs/MODEL_CARD.md) · [Data card](docs/DATA_CARD.md)
- [Release v0.2 và các đợt thí nghiệm](README.md#release)
- [Rà soát phương pháp mạnh/SOTA](docs/SOURCES.md#review-e03) · [E03: CNN/ensemble](reports/v0_3/REPORT.md)
- [E04: CatBoost và ngưỡng](reports/v0_4/REPORT.md#nhan-xet): có cải thiện trên pilot, chưa đạt mọi mục tiêu; [E05: mở rộng development](docs/experiments/E05_PLAN.md).
- [E06: TabM và mô hình thay thế](reports/E06/REPORT.md) · [rà soát TabICLv2/TabPFN-3.5](docs/SOTA_E06.md). Đánh giá thăm dò trên cùng development300.
- [Kiểm chứng riêng Pilot / VitalDB](reports/benchmarks/ASSESSMENT.md): replay 156 kết quả, 5.000 bootstrap ghép cặp; chưa xác nhận TabM tốt hơn toàn diện.
- [E08: so sánh 14 phương pháp trên cùng quy trình](#e08--so-sánh-phương-pháp-v2-bản-đang-dùng) · [lịch sử version](scripts/e08/README.md).

<!-- e08:begin -->
## E08 — so sánh phương pháp (v2, bản đang dùng)

Mọi phương pháp đi qua đúng một quy trình, chỉ khác nhau ở bản thân phương pháp; FA/giờ đi kèm chính sách `recall_first` đã được chấp nhận. Sinh tự động bởi [`scripts/e08/run_comparison.py`](scripts/e08/run_comparison.py) từ chính các tham số đã chạy. [Báo cáo đầy đủ](reports/E08/V2_METHOD_COMPARISON.md) · [lịch sử version và 6 lỗi của v1](scripts/e08/README.md)

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
| Dữ liệu | development300, chia theo `subjectid`: FIT 170 ca · CALIBRATION 58 ca · VALIDATION 24 ca |
| Biến cố eligible (5 / 10 phút) | FIT 211/229 · CALIBRATION 65/73 · VALIDATION 23/24 |
| Định nghĩa biến cố | MAP < 65 mmHg liên tục ≥ 60 s |
| Đặc trưng | 140 numeric (bỏ static); `map_logistic` chỉ dùng `map_current`, `map_300_slope`, `map_300_std` |
| Imputation | median, fit trên FIT, áp dụng y hệt cho FIT / CALIBRATION / VALIDATION |
| Calibration | StandardScaler → LogisticRegression `C=1e+06, max_iter=2000` trên log-odds của điểm thô; fit trên CALIBRATION, riêng từng phương pháp × horizon |
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
| `tabm_20260917` | TabM | 140 | `k=16, width=64, epochs=30, patience=6, batch_size=512`; `n_blocks=2, dropout=0.1, PLE 8 bins x 4, AdamW lr=0.002 wd=3e-4`; dừng ở epoch 5 (backbone đông lạnh từ E06) | không | không | — (train ở E06) |
| `tabm_20260918` | TabM | 140 | `k=16, width=64, epochs=30, patience=6, batch_size=512`; `n_blocks=2, dropout=0.1, PLE 8 bins x 4, AdamW lr=0.002 wd=3e-4`; dừng ở epoch 5 (backbone đông lạnh từ E06) | không | không | — (train ở E06) |
| `tabm_20260919` | TabM | 140 | `k=16, width=64, epochs=30, patience=6, batch_size=512`; `n_blocks=2, dropout=0.1, PLE 8 bins x 4, AdamW lr=0.002 wd=3e-4`; dừng ở epoch 5 (backbone đông lạnh từ E06) | không | không | — (train ở E06) |
| `ensemble_diverse_6` | ensemble | — | trung bình xác suất đã calibrate của `map_logistic`, `lgbm_jitter_3x`, `catboost_balanced_jitter`, `tabm_20260917`, `tabm_20260918`, `tabm_20260919` | — | — | — |

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
| `ensemble_diverse_6` | 0.652 (0.357–0.905) | 15/23 | 0.214 (0.103–0.333) | 0.960 (0.491–1.421) | 0.883 | 0.234 | 0.005 |

### Kết quả 10 phút — `recall_first`, VALIDATION (24 biến cố)

| Phương pháp | Recall (CI95) | Bắt được | PPV (CI95) | FA/giờ (CI95) | AUROC | AP | ECE |
|---|---|---:|---|---|---:|---:|---:|
| `map_logistic` ★ | 0.917 (0.733–1.000) | 22/24 | 0.284 (0.162–0.443) | 1.260 (0.813–1.774) | 0.804 | 0.222 | 0.028 |
| `catboost_balanced_jitter` | 0.917 (0.733–1.000) | 22/24 | 0.188 (0.110–0.321) | 2.483 (1.704–3.424) | 0.822 | 0.198 | 0.011 |
| `ensemble_diverse_6` | 0.875 (0.700–1.000) | 21/24 | 0.235 (0.142–0.365) | 1.630 (1.021–2.324) | 0.828 | 0.235 | 0.012 |
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
- Mỗi biến cố trên VALIDATION = 4,2–4,3 điểm recall, CI95 rất rộng: chênh lệch dưới ~1 biến cố **không** đủ kết luận phương pháp nào hơn.
- AUROC / AP / ECE không phụ thuộc ngưỡng — dùng để so khả năng phân biệt tách khỏi điểm vận hành.
- Development validation, **không phải** bằng chứng xác nhận độc lập; chưa mở pilot_test/global test.
<!-- e08:end -->

## Chạy local

Python ≥3.11. Tạo môi trường bằng `python -m venv .venv`, kích hoạt trước khi cài: Windows PowerShell dùng `.venv\Scripts\Activate.ps1`; Linux dùng `source .venv/bin/activate`.

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m safeanes.cli fetch-pilot --cases 60 --workers 4
python -m safeanes.cli build-pilot --out data/pilot_v1
python -m safeanes.cli run-pilot --dataset data/pilot_v1 --out artifacts/pilot_v1
```

Tải cần Internet; các bước sau dùng cache. Lỗi tải có thể chạy lại để resume. Dataset và experiment output phải mới/rỗng để không ghi đè kết quả. Có thể chạy trực tiếp với `PYTHONPATH=src` nếu môi trường đã có dependencies.

Baseline mặc định: score MAP, logistic MAP/slope/variability, HistGradientBoosting. LightGBM tùy chọn: cài `python -m pip install -e ".[boosting]"` rồi thêm `--models map logistic hist_gradient lightgbm`. Bootstrap pilot mặc định 200 theo bệnh nhân.

## Kaggle

Dùng [notebook baseline](notebooks/01_uc04_pilot.ipynb), [chuẩn bị sequence](notebooks/02_numeric_sequences.ipynb), [train TCN/Transformer](notebooks/03_tcn_transformer.ipynb) và [phát lại/báo cáo](notebooks/04_case_replay.ipynb). Sửa đường dẫn Input theo tài khoản; output ở `/kaggle/working`. Chạy bước chuẩn bị dữ liệu trên CPU và chỉ bật T4 cho notebook train. Chưa có benchmark thực thi trên T4/Kaggle.

## TCN và Transformer

```sh
python -m pip install -e ".[deep,plots]"
python -m safeanes.cli build-sequences --dataset data/pilot_v1 --root data/vitaldb --out data/sequences_v1
python -m safeanes.cli train-sequence --config configs/tcn.json --out artifacts/tcn_v1
python -m safeanes.cli train-sequence --config configs/transformer.json --out artifacts/transformer_v1
python -m safeanes.cli report --run artifacts/tcn_v1 --out reports/tcn_v1
python -m safeanes.cli report --run artifacts/transformer_v1 --out reports/transformer_v1
```

Mỗi output phải mới/rỗng. Với lần train bị gián đoạn, thêm `--resume` và giữ nguyên code/data/config; xem [runbook](docs/SEQUENCE_RUNBOOK.md). CUDA dùng FP16, CPU dùng FP32. Không coi việc hoàn tất train là đạt mục tiêu hiệu năng.

## Đầu ra

| File | Nội dung |
|---|---|
| `data/vitaldb/cohort_manifest.csv` | Toàn bộ ca, subject split, lý do loại |
| `data/vitaldb/raw/*.source.json` | URL, thời điểm tải và SHA-256 |
| `data/pilot_v1/quality.csv` | Độ phủ nhãn, nhịp MAP, event/decision từng ca |
| `data/pilot_v1/windows.csv.gz` | Feature causal, eligibility và nhãn; -1 là censored |
| `artifacts/pilot_v1/report.json` | Metric, CI, threshold selection và quality gates |
| `artifacts/pilot_v1/*_predictions.csv.gz` | Dự báo để kiểm tra/phát lại |
| `artifacts/pilot_v1/*_alarms.json` | Episode thật, giả hoặc censored |

`pilot_test` là tập thăm dò bên trong global train, **không phải final test**. Các con số trong kế hoạch là mục tiêu, không phải hiệu năng được hứa trước. Dữ liệu lớn và artifacts không đưa vào Git.

<!-- consolidated:release -->
<a id="release"></a>

## Version phát hành và đợt thí nghiệm

**Bản đang làm: v0.2 (chưa push).** Theo yêu cầu của chủ dự án, mỗi lần push mới tạo một version phát hành. Sửa lỗi, thêm mô hình, chạy seed hay mở rộng dữ liệu trong cùng đợt chưa push đều gộp vào v0.2; không tự tăng version sau mỗi thí nghiệm.

[Báo cáo tổng hợp v0.2](reports/v0_2/REPORT.md) · [Tiến độ plan](docs/IMPLEMENTATION_STEPS.md#tien-do)

Git hiện có mốc `3cdadde — v0.1`. Không tạo commit/tag/push chỉ để đổi cách đặt tên.

| Release | Trạng thái | Phạm vi |
|---|---|---|
| v0.1 | Mốc Git đã có | Baseline/pipeline ban đầu |
| v0.2 | Đang phát triển, chưa push | Sequence models, Inception/TimesNet/ensemble, CatBoost, sửa ngưỡng, mở rộng development và toàn bộ sửa lỗi hiện tại |

### Các đợt thí nghiệm trong v0.2

| ID | Nội dung | Kết quả / kế hoạch |
|---|---|---|
| E01 | Baseline 60 ca | [Kết quả](reports/PILOT_BASELINE.md) |
| E02 | TCN/Transformer | [Kiểm chứng triển khai](reports/IMPLEMENTATION_VALIDATION.md) |
| E03 | Inception, TimesNet, ensemble | [Báo cáo](reports/v0_3/REPORT.md), [nhận xét](reports/v0_3/REPORT.md#nhan-xet) |
| E04 | CatBoost ba seed, ablation ngưỡng | [Báo cáo](reports/v0_4/REPORT.md), [nhận xét](reports/v0_4/REPORT.md#nhan-xet), [hướng dẫn](docs/versions/V0_4_RUNBOOK.md) |
| E05 | Development 300 ca, split ổn định và ablation | [Plan](docs/experiments/E05_PLAN.md), [kết quả](reports/E05/REPORT.md), [subgroup/lead time](reports/E05/SUBGROUPS.md) |
| E06 | TabM+PLE ba seed/ensemble, monotone LightGBM | [Plan](docs/experiments/E06_PLAN.md), [nguồn SOTA](docs/SOTA_E06.md), [kết quả](reports/E06/REPORT.md) |
| E08 | So sánh 14 phương pháp (cân bằng lớp, augmentation, CatBoost, TabM, ensemble) trên một quy trình; v1 đã rút lại do 6 lỗi | [Plan](docs/experiments/E08_PLAN.md), [kết quả v2](#e08--so-sánh-phương-pháp-v2-bản-đang-dùng), [lịch sử version](scripts/e08/README.md) |

Tên đường dẫn cũ `v0_3`, `v0_4`, `run_version03.py`, `run_version04.py` là **mã thí nghiệm lịch sử**, không phải release v0.3/v0.4. Giữ chúng để không làm hỏng đường dẫn checkpoint, source snapshot và hashes đã đăng ký. Không sửa lại số liệu, timestamp hay hash trong các artifact cũ.

Các báo cáo/plan đã khóa có thể giữ nhãn cũ trong nội dung lưu trữ. Quy ước phát hành tại trang này thay thế cách gọi version ở tài liệu lịch sử. Mọi công việc mới dùng E05, E06… hoặc tên đợt thí nghiệm; package version chỉ đổi ở đợt push tiếp theo.

Source thay đổi sau thí nghiệm khiến runner cũ từ chối ghi đè là hành vi đúng. Tái lập lịch sử dùng source snapshot và đúng input hashes; tiếp tục nghiên cứu dùng đợt mới. Global final test vẫn chưa sử dụng.

