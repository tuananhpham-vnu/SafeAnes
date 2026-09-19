# Các bước tìm hiểu và triển khai UC04

Yêu cầu xuyên suốt: độ chính xác cao và ít cảnh báo giả theo mục 8 của [kế hoạch nghiên cứu](../UC04_RESEARCH_PLAN.md). Mọi công việc chưa push thuộc v0.2; dùng mã E03/E04/E05 cho các đợt thí nghiệm, xem [quy ước release](../README.md#release).

| Bước | Cần tìm hiểu | Nguồn | Code / sản phẩm | Điều kiện chuyển tiếp |
|---|---|---|---|---|
| 1 | Track, timestamp, mổ lại, missing | S01/S02 | `data.py`, manifest, raw hashes | Xác minh schema thật; tải có provenance |
| 2 | Endpoint, horizon, rò rỉ thời gian | S03/S08 + protocol dự án | `signals.py`, `dataset.py` | Test ranh giới/missing/future perturbation qua; QC dữ liệu thật |
| 3 | Baseline và calibration độc lập | S04/S05/S06/S07 | `experiment.py` | Đúng split; báo cáo cả mục tiêu không đạt |
| 4 | Alarm và khoảng tin cậy | S03/S05 + protocol dự án | `evaluation.py`, report/CI | Không đếm lặp event; censored xử lý đúng; audit exposure |
| 5 | TCN causal, sequence loader tiết kiệm RAM | S09/S10 | `sequences.py`, `models.py`, `training.py`; FP16/checkpoint/resume | Code và kiểm thử đã có; benchmark T4 chưa chạy |
| 6 | Transformer nhỏ, waveform fusion | S11; Zhu/HypoBridCast trong kế hoạch | Transformer numeric và cấu hình ablation đã có; waveform chưa triển khai | Thêm độ phức tạp phải cải thiện metric chính |
| 6b | Backbones mạnh và ensemble cố định | S12/S13; [review](SOURCES.md#review-e03) | E03: Inception-style, TimesNet, ensemble bốn model và báo cáo | Giữ cohort/protocol, báo cả kết quả không cải thiện |
| 6c | Mất cân bằng, ngưỡng, ba seed | [review E04](SOURCES.md#review-e04) | CatBoost None/SqrtBalanced, 12 model theo horizon, 32 operating point; [kết quả](../reports/v0_4/REPORT.md#nhan-xet) | Đã chạy; chưa đạt mọi gate |
| 6d | Cohort lớn và ablation | [plan E05](experiments/E05_PLAN.md) | 300 ca; bảo toàn role bệnh nhân cũ; baseline/CatBoost/MAP-only/numeric/+static | Khóa matrix trước chạy; báo riêng 37 bệnh nhân đánh giá mới |
| 6e | Thay thế backbone và regularization | [review E06](SOTA_E06.md) | TabM+PLE chính thức ba seed/ensemble; monotone LightGBM; [kết quả](../reports/E06/REPORT.md) | Đánh giá exploratory cùng holdout E05; model/ngưỡng chọn bằng validation |
| 7 | Final test và kiểm định ngoài | Protocol khóa, VitalDB/MOVER | Báo cáo riêng từng cohort | Đạt mục tiêu hoặc ghi chưa đạt; không tune test |

S01–S14 có citation trong [SOURCES.md](SOURCES.md). Release v0.2 gộp các bước 5–6d và sửa lỗi hiện tại. Benchmark T4 cần môi trường T4 thực; waveform có điều kiện; final test chưa mở; nhánh nguyên nhân cần bộ nhãn chuyên gia. Không đánh dấu các mốc đó hoàn thành chỉ vì đã tạo code/plan.

<!-- consolidated:tien-do -->
<a id="tien-do"></a>

## Tiến độ v0.2 — release chưa push

Quy ước: [mỗi lần push mới tăng version](../README.md#release). Các đợt thí nghiệm E01–E05 đều được gộp vào v0.2, không tạo release riêng khi chạy model hoặc sửa lỗi.

| Hạng mục plan | Trạng thái thực tế | Bằng chứng / phụ thuộc |
|---|---|---|
| Pipeline numeric, QC, nhãn causal | Đã triển khai và chạy dữ liệu thật | 60 ca cũ; E05 đã dựng 300 ca/297 bệnh nhân, 99.132 windows, 497 episode |
| Split bệnh nhân và provenance | Đã mở rộng | Roles E05 khóa trước tải track mới; bệnh nhân cũ không chuyển vai trò; raw/source hashes |
| TCN/Transformer và checkpoint/resume | Đã chạy CPU | E02; chưa phải log T4 |
| Inception/TimesNet/ensemble | Đã chạy | E03: không cải thiện alarm operating point trên pilot nhỏ |
| CatBoost, ngưỡng, ba seed | Đã chạy pilot 60 ca | E04: 12 model theo horizon, 32 operating point; [kết quả](../reports/v0_4/REPORT.md#nhan-xet) |
| Cohort lớn và feature ablation | Đã chạy đủ E05 | 16 model theo horizon × hai policy; MAP/logistic/LightGBM; CatBoost ba seed; MAP-only/numeric/+static; [kết quả](../reports/E05/REPORT.md) |
| TabM và phương án SOTA mới | Pipeline E06 đã tích hợp, có kiểm thử | [Plan](experiments/E06_PLAN.md), [benchmark](../reports/E06/REPORT.md); TabICLv2/TabPFN-3.5 mới rà soát, chưa đo SafeAnes |
| Ablation mask/time-since của sequence | Chưa chạy đủ ma trận | Config/code đã có; ablation tabular E05 không thay cho kiểm chứng riêng của sequence |
| Audit cohort không tim | Đã đối chiếu nguồn và metadata E05 | VitalDB gốc mô tả non-cardiac; kiểm kê chuyên khoa và cờ tên thủ thuật tại reports/E05/cohort_audit.csv; không coi regex là adjudication |
| Calibration/alarm/CI | Đã chạy theo protocol v1 | Bệnh nhân calibration tách biệt; CI bootstrap bệnh nhân; fast selection đối chiếu evaluator tham chiếu |
| CI lead time/subgroup | Đã chạy phân tích mô tả E05 | [28 bản ghi nhóm, kể cả nhóm trống](../reports/E05/SUBGROUPS.md); bootstrap bệnh nhân; CI lead có điều kiện đã phát hiện, chưa phải xác nhận ngoài |
| Exposure từng giây | Chưa hoàn tất | Giữ denominator 30 giây của protocol v1 trong E05; cần đợt evaluator riêng có plan khóa |
| Mục tiêu chất lượng MVP | Chưa nghiệm thu | E04 và E05 chưa đạt toàn bộ mục tiêu; [báo cáo tổng hợp](../reports/v0_2/REPORT.md), không suy từ số test phần mềm |
| Notebook/replay | Đã có code và artifact local | Chưa có log notebook chạy thực trên Kaggle/T4 |
| Benchmark T4 | Cần môi trường T4 thực | [Kế hoạch kiểm định](../UC04_RESEARCH_PLAN.md#kiem-dinh); CPU hoặc RTX3050 không thay bằng chứng T4 |
| Waveform | Có điều kiện, chưa triển khai | Chỉ bổ sung khi validation ủng hộ giá trị tăng thêm; cần cohort giao và QC riêng |
| Final test / external / tiến cứu | Chưa mở / chưa thực hiện | Model và policy phải khóa; external cần dữ liệu và quyền truy cập tương ứng |
| Nhánh cơ chế nguyên nhân | Đã chuẩn bị draft quy trình, chưa có nhãn | [Rubric và schema đề xuất](../UC04_RESEARCH_PLAN.md#gan-nhan); cần hai reviewer/chuyên gia và bằng chứng thực |

Không đánh dấu toàn bộ plan hoàn thành khi còn thiếu bằng chứng thực nghiệm, môi trường hoặc dữ liệu. Trạng thái “chuẩn bị quy trình” không thay trạng thái “đã thực hiện”.

