# Kết quả

Tất cả số liệu ở đây là của tập **validation**, là tập dùng để chọn ngưỡng và cấu hình. Kết quả báo cáo cuối là của tập test, chạy một lần sau khi khóa mô hình (chưa chạy).

**Đọc gì trước:** [ANALYSIS.md](ANALYSIS.md) (phân tích theo từng bối cảnh, bảng rút gọn và nhận xét) và [FULL_RESULTS.md](FULL_RESULTS.md) (bảng đầy đủ theo từng mô hình). Hai file này do `scripts/make_analysis.py` sinh từ các CSV dưới đây.

Thư mục này do `scripts/export_results.py` sinh ra từ `reports/` và `artifacts/` (thư mục làm việc, không commit). Chỉ chứa bảng tổng hợp và cấu hình; **không** có mô hình đã train, dự đoán theo từng ca hay số đếm theo từng ca. Những thứ đó nằm trên repo Hugging Face private `yungdyo3112/uc04-v2-runs`.

## Cấu trúc

```
results/
├── data/                                  dữ liệu và nhãn (chỉ số tổng hợp)
│   ├── data_check.json                    19 kiểm tra prep_v1 (tất cả PASS)
│   ├── cohort_flow.csv                    luồng cohort A → D (train, calibration, validation)
│   ├── cohort_excluded_train_cal_val.csv  ca bị loại và lý do
│   ├── label_report.csv                   nhãn theo tập × mốc × chính sách nhãn × luật loại mẫu
│   ├── abstain_report.csv                 lý do dòng không eligible
│   ├── samples_summary.json               samples.json của samples_v2 (sha256 9b17751a…)
│   ├── samples_provenance.json            commit và thư viện khi dựng samples
│   └── dl_cpu_speed.json                  tốc độ suy luận DL trên CPU của máy (ước tính NB04)
├── tabular/
│   ├── summary/                           bảng tổng
│   │   ├── tabular_validation.csv         65 tổ hợp (map_threshold lặp cho mỗi W) — schema mục 8.6
│   │   ├── tabular_validation_by_policy.csv  65 tổ hợp × 2 chính sách cảnh báo (drop_below, cooldown)
│   │   ├── model_comparison.csv           W* = 120, 5 mốc, 4 mô hình, 2 chính sách, CI 95%
│   │   ├── model_comparison_pairs.csv     bootstrap ghép cặp: wave−numeric, wave−logistic, numeric−threshold
│   │   ├── calibration_check_tabular.csv  trung bình xác suất so với tỷ lệ dương
│   │   └── w_star.json                    cửa sổ chính W* và căn cứ
│   ├── map_threshold/h<h>/                baseline: −MAP hiện tại (không phụ thuộc W)
│   ├── map_logistic/W<W>/h<h>/            baseline: logistic trên MAP
│   ├── lgbm_numeric/W<W>/h<h>/            LightGBM, 66 đặc trưng chỉ số
│   ├── lgbm_wave/W<W>/h<h>/               LightGBM, 66 chỉ số + 27 đặc trưng sóng
│   │     mỗi thư mục: metrics.csv, calibration_check.csv, threshold.json (kèm đường cong 40 điểm), provenance.json
│   ├── analysis/
│   │   ├── shap/                          |SHAP| theo cột (shap_W<W>_h<h>.csv) và theo nhóm (shap_groups.csv)
│   │   ├── ablation.csv                   bỏ từng nhóm cột ở W*, mốc 5 và 10 phút, bootstrap ghép cặp
│   │   └── confidence/                    mức tin cậy từ 5 seed và kiểm chứng theo thập phân vị (KHÔNG QUA → bỏ)
│   └── exploratory/                       THĂM DÒ: W = 300, 600 (quyết định sau khi xem validation)
└── dl/                                    Conv1D + Transformer: điền sau khi train trên Kaggle
```

`W` = độ dài cửa sổ nhìn lại (30, 60, 90, 120 giây). `h` = mốc dự báo (300, 600, 900, 1200, 1800 giây).

## Tóm tắt (W = 120, chính sách `drop_below`, ngưỡng cho tối đa 1 cảnh báo sai/giờ)

Độ nhạy theo đợt tụt:

| Mô hình | 5 phút | 10 phút | 15 phút | 20 phút | 30 phút |
|---|--:|--:|--:|--:|--:|
| `map_threshold` | 0,560 | 0,622 | 0,611 | 0,615 | 0,647 |
| `map_logistic` | 0,622 | 0,669 | 0,650 | 0,675 | 0,684 |
| `lgbm_numeric` | 0,628 | 0,686 | 0,675 | 0,686 | 0,686 |
| `lgbm_wave` | 0,635 | 0,686 | 0,661 | 0,688 | 0,678 |

AUROC:

| Mô hình | 5 phút | 10 phút | 15 phút | 20 phút | 30 phút |
|---|--:|--:|--:|--:|--:|
| `map_threshold` | 0,828 | 0,767 | 0,739 | 0,733 | 0,720 |
| `map_logistic` | 0,839 | 0,780 | 0,755 | 0,749 | 0,738 |
| `lgbm_numeric` | 0,874 | 0,817 | 0,790 | 0,779 | 0,764 |
| `lgbm_wave` | 0,875 | 0,820 | 0,791 | 0,781 | 0,768 |

Cách đọc (chi tiết ở `docs/INTERPRETATION_NOTES.md`):
- **Sóng động mạch chưa cho thấy lợi ích:** `lgbm_wave − lgbm_numeric` có CI chứa 0 ở độ nhạy, AUROC và AUPRC ở mọi mốc. Ở mốc 5 phút, FA/h và thời gian cảnh báo của `lgbm_wave` cao hơn (CI không chứa 0).
- LightGBM xếp hạng rủi ro tốt hơn `map_logistic` (AUROC, AUPRC có ý nghĩa danh nghĩa). Độ nhạy không khác có ý nghĩa; hai mô hình chỉ cùng thỏa giới hạn ≤ 1 cảnh báo sai/giờ, FA/h thực tế không bằng nhau.
- `lgbm_numeric` có độ nhạy, AUROC và AUPRC cao hơn `map_threshold` ở cả 5 mốc, nhưng FA/h cũng cao hơn ở 3/5 mốc.
- Chỉ bỏ nhóm MAP mới làm giảm độ nhạy có ý nghĩa (ablation). Cửa sổ dài (thăm dò, chỉ `lgbm_numeric`): W = 300 và 600 không tốt hơn W = 120, và W = 600 thấp hơn ở mốc 10 và 20 phút.
- Mọi CI là bootstrap theo bệnh nhân với ngưỡng cố định, 200 lần, không hiệu chỉnh đa so sánh. Hạn chế đầy đủ ở ANALYSIS.md mục 13.
