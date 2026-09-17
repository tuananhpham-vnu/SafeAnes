# Model card — SafeAnes UC04 numeric v0.2

**Trạng thái:** phần mềm nghiên cứu hồi cứu. Việc code chạy được và test phần mềm qua không chứng minh đạt mục tiêu dự báo của UC04.

| Nội dung | Mô tả |
|---|---|
| Mục đích | Dự báo onset IOH mới trong 300/600 giây từ 600 giây sinh hiệu quá khứ; phát lại và đánh giá cảnh báo. |
| Người sử dụng | Nhóm nghiên cứu ML/lâm sàng để audit và tái lập thí nghiệm. |
| Input | 7 numeric VitalDB, mask/time-since tùy cấu hình; age/BMI/ASA tùy cấu hình. |
| Output | Hai xác suất đã calibration có p600≥p300, trạng thái đủ dữ liệu, episode alarm theo policy. |
| Kiến trúc | TCN causal hoặc Transformer nhỏ theo patch; hai head logits có thứ tự. |
| Nhãn | MAP <65 liên tục ít nhất 60 giây theo protocol v1; censoring đối xứng nếu thiếu follow-up. |
| Dữ liệu phát triển | Pilot VitalDB trong global train, split theo bệnh nhân. Dataset cụ thể được xác định bằng hash. |
| Chọn mô hình | Checkpoint theo validation BCE; calibration trên bệnh nhân riêng; threshold trên validation. |
| Kết quả | Xem `reports/tcn_v1/REPORT.md`, `reports/transformer_v1/REPORT.md` và JSON đi kèm; không lấy AUROC paper làm kết quả code. |
| Giới hạn | Pilot nhỏ; chỉ dữ liệu có arterial MAP; FA/giờ còn xấp xỉ lưới 30 giây; chưa có kiểm định ngoài/tiến cứu. |
| Không nằm trong phạm vi | Chẩn đoán cơ chế tụt huyết áp, gợi ý thuốc/liều, suy luận nhân quả, sử dụng lâm sàng tự động. |

Mọi triển khai mới phải xác minh track/unit/timestamp, preprocessing, protocol hash và calibrator tương ứng. Không dùng cùng checkpoint để diễn giải NIBP thưa như MAP động mạch. Một nguy cơ thấp không loại trừ biến cố; trạng thái thiếu dữ liệu phải được giữ trong đầu ra.

Tái lập: [runbook](SEQUENCE_RUNBOOK.md). Đánh giá thiên lệch: tuổi/ASA và coverage chỉ là bước đầu; cần loại mổ, thiết bị, cơ sở, can thiệp và missingness khi mở rộng. SHAP/attention không cung cấp nhãn nguyên nhân.
