# Kết quả pilot UC04 trên dữ liệu thật

Báo cáo sinh từ artifacts bằng `scripts/summarize_pilot.py`. Đây là **pilot bên trong global train**, không phải final test hoặc kiểm định lâm sàng.

- Cohort pilot: 60 ca / 60 bệnh nhân.
- Dataset: 19,869 decision windows, 85 đợt IOH theo protocol hiện tại.
- Độ phủ nhãn 1 giây trung bình theo ca: 99.90%.
- Protocol hash: `7b50386febe54262130beed10b12e7dded06b332ca59a06dca3e8de58bb2572a`.
- Dataset hash: `4404bb59c4e926f7530b0731398ce1883647ed9a6e70e8d103f8ded0e8b512c7`.

## Chia tập pilot

| Vai trò | Ca | Bệnh nhân |
|---|---:|---:|
| calibration | 9 | 9 |
| fit | 36 | 36 |
| pilot_test | 9 | 9 |
| validation | 6 | 6 |

## Hiệu năng tại ngưỡng chọn trên validation

Các threshold thuộc chế độ nghiên cứu; giữ nguyên khi chạy pilot_test. Mọi số ở đây là kết quả đo, không phải số lấy từ paper.

| Model / horizon (giây) | AUROC | AP | Event sensitivity | Alarm PPV | False alarms/giờ | Event phát hiện/đủ điều kiện | Đạt mọi mục tiêu điểm? |
|---|---:|---:|---:|---:|---:|---:|---|
| map_300 | 0.965 | 0.241 | 0.000 | 0.000 | 0.077 | 0/3 | Chưa đạt |
| logistic_300 | 0.965 | 0.257 | 0.000 | N/A | 0.000 | 0/3 | Chưa đạt |
| hist_gradient_300 | 0.982 | 0.355 | 0.000 | 0.000 | 0.155 | 0/3 | Chưa đạt |
| map_600 | 0.764 | 0.190 | 0.000 | N/A | 0.000 | 0/4 | Chưa đạt |
| logistic_600 | 0.757 | 0.140 | 0.000 | N/A | 0.000 | 0/4 | Chưa đạt |
| hist_gradient_600 | 0.927 | 0.212 | 0.000 | 0.000 | 0.247 | 0/4 | Chưa đạt |
| lightgbm_300 | 0.970 | 0.304 | 0.333 | 0.500 | 0.077 | 1/3 | Chưa đạt |
| lightgbm_600 | 0.928 | 0.253 | 0.250 | 0.333 | 0.165 | 1/4 | Chưa đạt |

## Giới hạn phải đọc cùng kết quả

- Cỡ mẫu pilot_test nhỏ; xem số event đủ điều kiện, không chỉ số window. Khoảng tin cậy 95% và số bootstrap replicate hợp lệ nằm trong `report.json` của từng run.
- N/A PPV nghĩa không có alarm đánh giá được, không có nghĩa PPV bằng 100%.
- AUROC cao chỉ phản ánh thứ hạng nguy cơ; còn cần độ nhạy và PPV ở ngưỡng cảnh báo đã khóa. Chưa thể tuyên bố đạt yêu cầu chỉ nhờ AUROC.
- Eligibility, history, censoring và recovery làm một số event không có cơ hội dự báo. Metrics lưu cả `events_all` và `event_sensitivity_all` để công bố phần này.
- FA/hour hiện xấp xỉ trên lưới 30 giây; xem [protocol](../docs/PROTOCOL.md). Chưa dùng metric này để nghiệm thu lâm sàng.
- Báo cáo này giữ kết quả baseline CPU. Kết quả DL bổ sung: [TCN](tcn_v1/REPORT.md), [Transformer](transformer_v1/REPORT.md); chưa có benchmark T4.

## Bước tiếp theo dựa trên pilot

1. Audit các event không đủ cơ hội dự báo và alarm bị censored; rà soát nhãn bằng timeline MAP gốc.
2. Mở rộng pilot trong global train để calibration/validation có nhiều bệnh nhân và biến cố hơn; giữ final test chưa mở.
3. Cải thiện evaluator exposure từng giây và kiểm tra độ nhạy với recovery/persistence/cooldown trong development; không hạ mục tiêu chất lượng.
4. TCN và sequence loader đã được bổ sung với cùng split/nhãn/evaluator; cần mở rộng dữ liệu, đo VRAM và thời gian trên T4 trước kết luận giá trị tăng thêm.

## Nguồn của phương pháp

Dữ liệu từ [VitalDB](https://doi.org/10.1038/s41597-022-01411-5); API theo [tài liệu chính thức](https://vitaldb.net/docs/?documentId=API%2FWeb_API_OpenDataset.md). Đánh giá liên tục được thúc đẩy bởi nghiên cứu [selection bias của Yang et al.](https://pubmed.ncbi.nlm.nih.gov/40404499/). Calibration dựa trên [scikit-learn](https://scikit-learn.org/stable/modules/calibration.html). Các quy tắc cụ thể là thiết kế dự án, được phân biệt trong [sổ nguồn](../docs/SOURCES.md).

Artifacts dùng để sinh báo cáo:

- `artifacts/pilot_v1`
- `artifacts/pilot_lightgbm_v1`
