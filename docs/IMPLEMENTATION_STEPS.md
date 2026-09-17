# Các bước tìm hiểu và triển khai UC04

Yêu cầu xuyên suốt: độ chính xác cao và ít cảnh báo giả theo mục 8 của [kế hoạch nghiên cứu](../UC04_RESEARCH_PLAN.md).

| Bước | Cần tìm hiểu | Nguồn | Code / sản phẩm | Điều kiện chuyển tiếp |
|---|---|---|---|---|
| 1 | Track, timestamp, mổ lại, missing | S01/S02 | `data.py`, manifest, raw hashes | Xác minh schema thật; tải có provenance |
| 2 | Endpoint, horizon, rò rỉ thời gian | S03/S08 + protocol dự án | `signals.py`, `dataset.py` | Test ranh giới/missing/future perturbation qua; QC dữ liệu thật |
| 3 | Baseline và calibration độc lập | S04/S05/S06/S07 | `experiment.py` | Đúng split; báo cáo cả mục tiêu không đạt |
| 4 | Alarm và khoảng tin cậy | S03/S05 + protocol dự án | `evaluation.py`, report/CI | Không đếm lặp event; censored xử lý đúng; audit exposure |
| 5 | TCN causal, sequence loader tiết kiệm RAM | S09 và PyTorch chính thức | Mốc tiếp: trainer/checkpoint FP16 | Cùng label/split/evaluator; đo VRAM T4 thật |
| 6 | Transformer nhỏ, waveform fusion | Zhu/HypoBridCast trong kế hoạch | Ablation | Thêm độ phức tạp phải cải thiện metric chính |
| 7 | Final test và kiểm định ngoài | Protocol khóa, VitalDB/MOVER | Báo cáo riêng từng cohort | Đạt mục tiêu hoặc ghi chưa đạt; không tune test |

S01–S09 có đầy đủ citation trong [SOURCES.md](SOURCES.md). Bước 1–4 là phạm vi code đầu; bước 5–7 dựa trên kết quả thực nghiệm. Nhánh nguyên nhân cần bộ nhãn chuyên gia riêng như kế hoạch gốc.

