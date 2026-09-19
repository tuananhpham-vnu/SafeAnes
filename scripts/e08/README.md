# E08 — code và lịch sử version

**Bản đang dùng: v2** — [`run_comparison.py`](run_comparison.py) + logic dùng chung
[`src/safeanes/e08_methods.py`](../../src/safeanes/e08_methods.py). Bảng tham số và kết quả:
[README gốc](../../README.md#e08--so-sánh-phương-pháp-v2-bản-đang-dùng) ·
[báo cáo đầy đủ](../../reports/E08/V2_METHOD_COMPARISON.md).

Các bản kém hơn nằm trong `version/` (code) và `reports/E08/version/`, `artifacts/E08/version/`
(kết quả), giữ nguyên để truy vết — **không dùng số liệu của chúng để ra quyết định**.

| Version | Code | Kết quả | Trạng thái |
|---|---|---|---|
| **v2** | [`run_comparison.py`](run_comparison.py) | [V2_METHOD_COMPARISON.md](../../reports/E08/V2_METHOD_COMPARISON.md) | Đang dùng — sửa đủ 6 lỗi |
| v2_initial | đã gộp vào `run_comparison.py` | [version/v2_initial/](../../reports/E08/version/v2_initial/) | Thay thế — còn lỗi F6, ensemble chưa có TabM |
| v1 | [`version/v1_sequential_stages/`](version/v1_sequential_stages/) | [version/v1/REPORT.md](../../reports/E08/version/v1/REPORT.md) | Rút lại — 6 lỗi |

## Sáu lỗi của v1 và cách v2 sửa

Tìm được khi audit code ngày 19/09/2026; mỗi lỗi đều kiểm chứng bằng số đo.

| Mã | Lỗi | Bằng chứng | Cách sửa trong v2 |
|---|---|---|---|
| F1 | Train/serve skew: model augment train trên ma trận đã impute nhưng chấm điểm VALIDATION bằng feature thô còn NaN | 948 ô NaN trên 271/7.401 dòng VALIDATION (3,7%) | Một imputer fit trên FIT, áp dụng y hệt cho FIT/CALIBRATION/VALIDATION |
| F2 | Ensemble trộn xác suất **đã calibrate** (E05/E06) với `predict_proba` **thô** | Đã calibrate: mean 0,018–0,020 ≈ prevalence 0,0244; CatBoost balanced thô: mean 0,0871 (gấp ~3,5 lần) | Mọi phương pháp calibrate bằng cùng sigmoid-trên-log-odds, fit trên CALIBRATION; báo cáo có bảng kiểm chứng mean ≈ prevalence |
| F3 | Gate tính trên vote fraction `votes/N` như thể là xác suất | ECE = 0,210 ở **mọi** mức vote, gate yêu cầu ≤ 0,05 — không bao giờ đạt được | Gate tính trên điểm liên tục đã calibrate (ECE đo được 0,004–0,03) |
| F4 | So sánh augmentation bị nhiễu: baseline train dữ liệu thô-NaN, jitter/SMOTE train dữ liệu đã impute | Hai thay đổi lẫn vào một so sánh | Mọi phương pháp dùng chung một preprocessing |
| F5 | Chọn ngưỡng trên VALIDATION rồi báo cáo cũng trên VALIDATION | VALIDATION chỉ có 23–24 biến cố → mỗi biến cố đổi recall 4,3 điểm | Chọn ngưỡng **và** chọn phương pháp trên CALIBRATION (65–73 biến cố), báo cáo trên VALIDATION |
| F6 | `oversample "3x"` thêm 2 bản sao mỗi window dương, jitter/SMOTE `"3x"` thêm 3 | Đếm trực tiếp: 10 dương → 30 (oversample) so với → 40 (jitter/SMOTE) | Mọi kiểu augmentation cùng thêm `AUGMENT_COPIES = 3` mẫu mỗi window dương |

Thêm một lỗi hiển thị ở v1: horizon 10 phút có 7 gate (thêm `early_5min_sensitivity`) nhưng báo cáo ghi "/6".

## Thiết kế v2

- **Một nguồn sự thật cho tham số.** Siêu tham số khai báo ở đầu `e08_methods.py` (`LIGHTGBM_PARAMS`,
  `CATBOOST_PARAMS`, `METHODS`, …); model dùng đúng các dict đó và bảng trong README được sinh từ
  chính giá trị đã chạy, nên bảng không thể lệch với code.
- **Backbone TabM lấy từ E06 và được kiểm tra khi nạp**: bộ 140 feature phải trùng thứ tự và
  `k/width/epochs/patience/batch_size` phải khớp `TABM_EXPECTED`, nếu không thì dừng.
- **Hai chính sách ngưỡng**: `recall_first` (bản đang dùng — recall cao nhất, chấp nhận FA/giờ đi kèm)
  và `fa_budget` (recall cao nhất trong FA/giờ ≤ 0,5) để thấy cái giá của ràng buộc FA.

## Chạy

```powershell
$env:PYTHONPATH = "$PWD/src;$PWD/.local_deps"
python scripts/e08/run_comparison.py                 # ~45 phút trên CPU: train, ghi CSV, rồi render
python scripts/e08/run_comparison.py --render-only   # vài giây: chỉ dựng lại báo cáo và README từ CSV
```

Không script E08 nào mở `pilot_test` hoặc global test, và không script nào ghi đè artifact E05/E06.
