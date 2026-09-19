# E07 — đồng nhất đầu vào CSV trước tổng hợp kết quả

Audit trên ca holdout development cũ số 55 phát hiện feature trực tiếp từ build_case
khác feature đã ghi/đọc CSV của E05 tối đa 2,84e-14. QuantileTransformer có thể khuếch đại
sai số này ở feature gần hằng: xác suất TabM khác tối đa 0,00022668. Sau `to_csv` rồi
`pd.read_csv` mặc định, feature khớp tuyệt đối và xác suất khớp đúng E06 trên ca này.

Sửa đường input để mô phỏng đúng bước dataset CSV dùng khi train/evaluate E05/E06:
giữ nguyên build_case, labels, model, calibration, threshold và primary comparison.
Không sửa để cải thiện điểm holdout. Chưa đọc/tổng hợp metric cohort mới hoặc final test.

Kết quả chạy trực tiếp từ feature ở artifacts/E07 là bản nháp không dùng kết luận.
Giữ nguyên các file đó và source snapshot. Kết quả sửa được lưu ở artifacts/E07b;
raw/preprocessed cache cũ tái sử dụng. Source bổ sung có hash riêng liên kết registration E07.
Xử lý inference ba process, download/preprocessing tám process để giảm thời gian chờ;
mỗi inference process dùng lại đúng predict_case sau khi chuẩn hóa vòng CSV.

E07b là lần sửa thực thi của E07, không phải version phát hành mới. Final test vẫn
là một lần đánh giá của các mô hình/ngưỡng khóa trước, không có tuning theo kết quả mới.
