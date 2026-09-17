# Kiểm chứng triển khai UC04 v0.2 — 17/09/2026

## Kiểm thử phần mềm

- `python -m pytest -q`: **38 passed**, gồm baseline và PyTorch; không có test bị bỏ qua trong môi trường đã cài dependencies.
- Kiểm tra causal feature/sequence/convolution, nhãn thiếu, ranh giới horizon, split theo bệnh nhân, normalizer fit-only, ordered horizons và calibration.
- Kiểm thử resume từ checkpoint cho weights khớp tuyệt đối với chạy liên tục trên cùng môi trường CPU.
- Trên 120 decision lấy mẫu cố định từ dữ liệu thật, giá trị cuối chuỗi khớp feature hiện tại của baseline cho cả 7 track (tolerance float32).
- Nạp lại `model.pt` của từng mô hình và suy luận 64 decision: xác suất khớp predictions đã lưu (`rtol=1e-5`, `atol=1e-7`).
- Bốn notebook có JSON hợp lệ và toàn bộ code cell biên dịch được. Notebook GPU chưa được thực thi trên Kaggle; kiểm tra cú pháp không được coi là benchmark.
- Đã kiểm tra ảnh replay và các liên kết Markdown nội bộ; không có liên kết file bị thiếu.

## Thực nghiệm đã chạy

Dataset giữ nguyên 60 ca/60 bệnh nhân trong global train. Tập fit có 11.169 decision đủ điều kiện với ít nhất một nhãn biết. Không giới hạn số window mỗi epoch. PyTorch `2.14.0+cpu`; log thiết bị và source hashes nằm trong `artifacts/*/environment.json`.

| Mô hình | Tham số | Epoch đã chạy / chọn | Thời gian các epoch | AUROC 5 / 10 phút | Event recall 5 / 10 phút |
|---|---:|---:|---:|---:|---:|
| TCN | 44.418 | 8 / 3 | 476,8 giây | 0,985 / 0,897 | 0/3 / 0/4 |
| Transformer | 120.066 | 6 / 1 | 174,9 giây | 0,941 / 0,846 | 0/3 / 0/4 |

Các thời gian là log CPU của phiên này; có giai đoạn hai job chạy đồng thời nên không dùng bảng để benchmark tốc độ tương đối giữa kiến trúc. Không có phép đo VRAM hoặc T4.

TCN ở cả hai horizon và Transformer 5 phút chọn fallback không phát cảnh báo (threshold >1) trên validation. Transformer 10 phút chọn threshold khoảng 0,437 nhưng không phát hiện được event đủ điều kiện trên pilot_test. **Cả hai chưa đạt yêu cầu hiệu năng; AUROC cao của TCN không bù được độ nhạy cảnh báo bằng 0.** Không thay threshold dựa trên pilot_test.

Xem báo cáo đầy đủ: [TCN](tcn_v1/REPORT.md), [Transformer](transformer_v1/REPORT.md), [baseline](PILOT_BASELINE.md). CI bootstrap 200 lần theo bệnh nhân nằm trong JSON của mỗi run và `diagnostics.json` của báo cáo.

## Phần cần tiếp tục

Mở rộng development và audit eligibility/censoring/cohort không tim; hoàn thiện exposure từng giây; đánh giá ablation/ba seed theo validation; benchmark T4. Final test, kiểm định ngoài, nhãn cơ chế và nghiên cứu tiến cứu chưa được thực hiện. Danh sách đầy đủ tại mục 13 của [kế hoạch](../UC04_RESEARCH_PLAN.md); lệnh tái lập tại [runbook](../docs/SEQUENCE_RUNBOOK.md).
