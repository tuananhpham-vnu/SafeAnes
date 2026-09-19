# UC04 v0.3 — kế hoạch thí nghiệm khóa trước chạy

Ngày: 17/09/2026. Mục tiêu: kiểm tra hai họ mô hình mạnh từ nghiên cứu phân loại chuỗi và một ensemble cố định trên pilot hiện có. Đây là adaptation cho UC04, không phải tái lập SOTA lâm sàng.

## Thiết kế

- Giữ nguyên `data/pilot_v1`, `data/sequences_v1`, subject roles, nhãn, eligibility, calibration và grid chọn ngưỡng của v0.2. Không thay các mục tiêu AUROC/PPV/event recall/FAH.
- Hai ứng viên mới: một mạng Inception-style nhỏ và TimesNet nhỏ; cấu hình tại `configs/v0_3/`. Seed 20260917, tối đa 30 epoch, patience 5, toàn bộ fit windows; checkpoint chọn bằng validation BCE.
- Inception-style: 6 module, 3 nhánh convolution kernel 39/19/9, bottleneck, nhánh max-pool, residual mỗi 3 module, global average pooling. Width 16 mỗi nhánh, khác cấu hình gốc 32 và ensemble 5 mạng.
- TimesNet adaptation: 2 block, width 16, top-2 chu kỳ FFT **theo từng mẫu**, reshape 2D và convolution kernel 1/3/5; global average pooling thay flatten head. Không chọn tần số trung bình cả batch vì một dự báo không được phụ thuộc những bệnh nhân khác đi cùng batch.
- Cả hai dùng static age/BMI/ASA, mask/time-since, đầu ra logits có thứ tự và masked BCE như v0.2.
- Ensemble: trung bình logits của **bốn** checkpoint TCN, Transformer, Inception-style, TimesNet, trọng số 0,25 cố định. Fit một temperature dương/bias chung trên calibration patients, chọn threshold trên validation. Không chọn thành viên/trọng số theo pilot_test.
- Baseline và TCN/Transformer v0.2 được tái sử dụng để đối chiếu cùng cohort; không train lại chỉ để tìm kết quả đẹp hơn.
- Mọi cấu hình đều báo cáo pilot_test, kể cả kết quả kém. So sánh chính gồm AUROC/AP, event recall, PPV, FAH, ECE, coverage, lead time và CI theo bệnh nhân. Chọn ứng viên cho vòng sau bằng validation, không bằng bảng test.

## Giới hạn biết trước

Pilot_test đã được xem ở các version trước, nên **mọi kết quả v0.3 tiếp tục là exploratory**, không phải đánh giá xác nhận trên test mới. Validation chỉ 6 bệnh nhân, calibration 9, pilot_test 9; số biến cố hạn chế. Không suy luận ưu thế thống kê hoặc độ vững từ một seed. Exposure còn là lưới 30 giây. Không tuyên bố hoàn thành chất lượng nếu mô hình vẫn không phát hiện được biến cố ở ngưỡng đã chọn.

## Tiêu chí hoàn tất version

Hai model mới và ensemble chạy xong, checkpoint/config/source/data hashes được lưu, kiểm thử tính độc lập giữa bệnh nhân trong batch qua, báo cáo đối chiếu sinh từ artifacts thật. Không yêu cầu phải cải thiện điểm số mới được công bố version.

v0.4 dự kiến: audit/mở rộng cohort và validation, lưới ngưỡng phù hợp risk distribution, ablation/ba seed theo development. Các thay đổi dữ liệu/evaluator phải ghi thành version riêng; không ghi đè v0.3. Waveform/HMF hoặc phương pháp IOH chuyên biệt là vòng sau theo kết quả và nguồn lực.
