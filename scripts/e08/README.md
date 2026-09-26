# E08 — code và các version

Một runner duy nhất, [`run_comparison.py`](run_comparison.py), cùng logic dùng chung
[`src/safeanes/e08_methods.py`](../../src/safeanes/e08_methods.py). Bảng tham số và kết quả của bản
mới nhất luôn nằm trong [README gốc](../../README.md#e08) và
[`reports/E08/METHOD_COMPARISON.md`](../../reports/E08/METHOD_COMPARISON.md).

Bản mới nhất nằm ở `reports/E08/`; mọi bản cũ hơn nằm trong `reports/E08/version/` (kết quả) và
`scripts/e08/version/` (code), giữ lại để truy vết — **không dùng số liệu của chúng để ra quyết định**.

| Version | Dữ liệu | Cách chạy | Trạng thái |
|---|---|---|---|
| **v3** | VitalDB trừ global test. Đủ dữ liệu: FIT 2.559 ca · CALIBRATION 280 · VALIDATION 271 (307/333 biến cố). Với dữ liệu hiện có trên máy (thiếu 1.148 ca): FIT 1.542 · CALIBRATION 181 · VALIDATION 175 (221/239 biến cố) | [`run_v3_full_vitaldb.sh`](run_v3_full_vitaldb.sh) | Đã viết, **chờ chạy** |
| v2 | development300: FIT 170 ca · CALIBRATION 58 ca · VALIDATION 24 ca (23/24 biến cố) | `run_comparison.py --dataset development300` | Đang hiển thị cho tới khi v3 chạy xong |
| v2_initial | development300, bản v2 đầu tiên | đã gộp vào `run_comparison.py` | Thay thế — còn lỗi F6, ensemble chưa có TabM |
| v1 | development300, tối ưu tuần tự theo giai đoạn | `version/v1_sequential_stages/` (đã xóa) | Rút lại — 6 lỗi |

Khi v3 chạy xong, runner tự chuyển kết quả v2 vào `reports/E08/version/v2_development300_tabm-frozen/`
và README gốc chuyển sang hiển thị v3.

## Chạy v3

```bash
bash scripts/e08/run_v3_full_vitaldb.sh                  # bỏ TabM — nhanh nhất
TABM=frozen  bash scripts/e08/run_v3_full_vitaldb.sh     # dùng lại backbone TabM của E06, chỉ calibrate lại
TABM=retrain bash scripts/e08/run_v3_full_vitaldb.sh     # train lại TabM trên FIT — thêm nhiều giờ CPU
```

- Chạy từ Git Bash trên Windows (hoặc bash bất kỳ). Script tự đặt `PYTHONPATH`, kiểm tra dữ liệu E07 và
  thư viện trước khi chạy, và ghi log vào `reports/E08/run_v3_tabm-*.log`.
- **Ước tính 4–8 giờ trên CPU, cần ~3–4 GB RAM trống** (chưa đo trên dữ liệu đầy đủ). Máy hiện có 7,7 GB
  nên nên đóng bớt ứng dụng khác. Runner chạy tuần tự vì RAM không đủ cho song song.
- **Bị ngắt hoặc lỗi giữa chừng thì chạy lại đúng lệnh cũ**: mỗi phương pháp × horizon đã xong được lưu ở
  `artifacts/E08/cache/` và được nạp lại, chỉ khi code phương pháp/đánh giá không đổi.
- Global test `unseen_test` (516 ca) không bao giờ được đọc: bộ nạp chỉ mở các nhóm FIT/CALIBRATION/
  VALIDATION và dừng lại nếu phát hiện ca của global test hoặc bệnh nhân dùng chung giữa các tập.
- Bộ nạp nhận `data/vitaldb_full/csv_cases` (bản E07b) hoặc `data/vitaldb_full/cases` (bản E07 chạy
  trước). Ca nào thiếu file tiền xử lý thì bỏ qua kèm cảnh báo, và số ca thiếu được ghi thành một dòng
  riêng trong bảng tham số của báo cáo — đây là **tập con do dữ liệu cục bộ**, không phải tiêu chí
  cohort của protocol. Muốn có đủ cohort thì phải chạy lại tiền xử lý E07 cho các ca còn thiếu
  (tải lại track thô từ VitalDB).

Các lệnh khác:

```bash
python scripts/e08/run_comparison.py --dataset development300   # chạy lại v2 (~45 phút)
python scripts/e08/run_comparison.py --render-only              # vài giây: dựng lại báo cáo và README từ kết quả đã lưu
```

## Sáu lỗi của v1 và cách sửa

Tìm được khi audit code ngày 19/09/2026; mỗi lỗi đều kiểm chứng bằng số đo.

| Mã | Lỗi | Bằng chứng | Cách sửa |
|---|---|---|---|
| F1 | Train/serve skew: model augment train trên ma trận đã impute nhưng chấm điểm VALIDATION bằng feature thô còn NaN | 948 ô NaN trên 271/7.401 dòng VALIDATION (3,7%) | Median từng cột tính trên FIT, điền y hệt cho FIT/CALIBRATION/VALIDATION |
| F2 | Ensemble trộn xác suất **đã calibrate** (E05/E06) với `predict_proba` **thô** | Đã calibrate: mean 0,018–0,020 ≈ prevalence 0,0244; CatBoost balanced thô: mean 0,0871 (gấp ~3,5 lần) | Mọi phương pháp calibrate bằng cùng sigmoid-trên-log-odds, fit trên CALIBRATION; báo cáo có bảng kiểm chứng mean ≈ prevalence |
| F3 | Gate tính trên vote fraction `votes/N` như thể là xác suất | ECE = 0,210 ở **mọi** mức vote, gate yêu cầu ≤ 0,05 — không bao giờ đạt được | Gate tính trên điểm liên tục đã calibrate |
| F4 | So sánh augmentation bị nhiễu: baseline train dữ liệu thô-NaN, jitter/SMOTE train dữ liệu đã impute | Hai thay đổi lẫn vào một so sánh | Mọi phương pháp dùng chung một preprocessing |
| F5 | Chọn ngưỡng trên VALIDATION rồi báo cáo cũng trên VALIDATION | VALIDATION development300 chỉ có 23–24 biến cố → mỗi biến cố đổi recall 4,3 điểm | Chọn ngưỡng **và** chọn phương pháp trên CALIBRATION, báo cáo trên VALIDATION |
| F6 | `oversample "3x"` thêm 2 bản sao mỗi window dương, jitter/SMOTE `"3x"` thêm 3 | Đếm trực tiếp: 10 dương → 30 (oversample) so với → 40 (jitter/SMOTE) | Mọi kiểu augmentation cùng thêm `AUGMENT_COPIES = 3` mẫu mỗi window dương |

Thêm một lỗi hiển thị ở v1: horizon 10 phút có 7 gate (thêm `early_5min_sensitivity`) nhưng báo cáo ghi "/6".

## Thiết kế

- **Một nguồn sự thật cho tham số.** Siêu tham số khai báo ở đầu `e08_methods.py` (`LIGHTGBM_PARAMS`,
  `CATBOOST_PARAMS`, `METHODS`, …); model dùng đúng các dict đó và bảng trong README được sinh từ chính
  giá trị đã chạy, nên bảng không thể lệch với code.
- **Backbone TabM của E06 được kiểm tra khi nạp**: bộ 140 feature phải trùng thứ tự và
  `k/width/epochs/patience/batch_size` phải khớp `TABM_EXPECTED`, nếu không thì dừng.
- **Hai chính sách ngưỡng**: `recall_first` (recall cao nhất, chấp nhận FA/giờ đi kèm — chính sách đang
  dùng) và `fa_budget` (recall cao nhất trong FA/giờ ≤ 0,5) để thấy cái giá của ràng buộc FA.
- **Bộ nạp VitalDB đầy đủ đọc từng ca vào numpy** và tính median theo từng cột, nên bộ nhớ đỉnh khoảng
  gấp đôi ma trận FIT (~1,3 GB) thay vì gấp ~5 lần như cách dùng `SimpleImputer` trên DataFrame, vốn đã
  hết RAM khi thử trên máy 7,7 GB. Trên development300, cách mới cho ma trận giống hệt từng bit.
