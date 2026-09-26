# Model card — SafeAnes UC04 v0.2

**Cập nhật E07, 19/09/2026:** đang đánh giá toàn bộ 3.626 ca đủ điều kiện trong 6.388 ca
VitalDB, bao gồm global final test theo yêu cầu người dùng. Không train/tune/recalibrate.
Primary TabM ensemble so CatBoost numeric trên global test; [plan khóa trước chạy](experiments/E07_PLAN.md).
Các ghi chú final test chưa mở bên dưới mô tả trạng thái trước E07.

**Trạng thái:** bản nghiên cứu đang phát triển, chưa push. Mọi sửa lỗi/thí nghiệm hiện tại thuộc v0.2; [quy ước release](../README.md#release). Code chạy được không đồng nghĩa đạt mục tiêu dự báo.

| Nội dung | Mô tả |
|---|---|
| Mục đích | Dự báo onset IOH mới trong 300/600 giây từ 600 giây sinh hiệu quá khứ; phát lại cảnh báo. |
| Input | Numeric VitalDB, mask/time-since tùy cấu hình; age/BMI/ASA tùy ablation. |
| Output | Xác suất theo horizon, trạng thái đủ dữ liệu và episode alarm. Sequence/ensemble bảo toàn p600≥p300; classifier tabular E05 độc lập chưa bảo đảm thứ tự này. E06 chiếu xác suất sau calibration để giữ p600≥p300. |
| Kiến trúc | MAP/logistic/HistGradient/LightGBM/CatBoost; TCN, patch Transformer, Inception-style, TimesNet adaptation; E06 bổ sung TabM+PLE chính thức, ordered head, ensemble xác suất và monotone LightGBM. |
| Nhãn | MAP <65 liên tục ít nhất 60 giây theo protocol v1; label -1 nếu không xác định follow-up. |
| Dữ liệu | Pilot 60 ca; E05 đã chạy 300 ca/297 bệnh nhân trong global train, giữ role cũ; đánh giá riêng 37 bệnh nhân mới. Dataset và split có hash riêng. |
| Fitting/calibration/selection | Tách bệnh nhân fit/calibration/validation/pilot_test. Sequence chọn checkpoint theo validation BCE; tabular theo matrix khóa trước. Ngưỡng chọn trên validation. |
| Kết quả | Tổng hợp v0.2 (đã xóa), E05: 300 ca/ba seed/ablation (đã xóa), CI subgroup/lead time (đã xóa). Kết quả cohort lớn chưa đạt đầy đủ gate; không suy chất lượng từ pilot 60 ca. |
| Tính nhất quán horizon | Có p600<p300 ở một phần window tabular do train/calibrate độc lập; số cụ thể trong E05 verification (đã xóa) và E04 (đã xóa). Không diễn giải hai đầu ra như risk có thứ tự. |
| Giới hạn | Pilot nhỏ; test đã được xem nên exploratory; chỉ cohort có arterial MAP; FA/giờ xấp xỉ lưới 30 giây; chưa kiểm định ngoài/tiến cứu. |
| Ngoài phạm vi đã hoàn thành | Chẩn đoán cơ chế, thuốc/liều, suy luận nhân quả, lâm sàng tự động. Chưa có nhãn chuyên gia cho nguyên nhân. |

Mỗi checkpoint phải đi kèm preprocessing, feature order, protocol/data hashes, calibrator và threshold đúng policy. NIBP thưa không được thay thế trực tiếp MAP động mạch. Tái lập: [sequence runbook](SEQUENCE_RUNBOOK.md), E04 runbook (đã xóa).

Đánh giá phân nhóm tuổi/ASA và coverage mới là bước đầu; cần loại mổ, thiết bị, cơ sở, can thiệp và missingness. SHAP/attention không cung cấp nhãn nguyên nhân. Global final test vẫn được giữ riêng.

E06 dùng lại holdout E05, nên toàn bộ kết quả thay thế (đã xóa) là exploratory.
Checkpoint TabM chọn bằng BCE trên bệnh nhân inner stopping thuộc FIT; preprocessing chỉ fit
trên inner optimization. Đề xuất model/ngưỡng khóa bằng validation, không chọn seed từ holdout.
[TabICLv2 và TabPFN-3.5](SOTA_E06.md) mới được rà soát, chưa có kết quả SafeAnes.
