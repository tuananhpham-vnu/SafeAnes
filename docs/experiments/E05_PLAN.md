# E05 — mở rộng development trong release v0.2

Đây là đợt thí nghiệm, không phải version phát hành. Thiết kế khóa trước khi đánh giá cohort mở rộng.

- Chọn 300 ca đầu từ cùng thứ tự lấy mẫu global train, seed 20260917; giữ 60 ca cũ. Cache mới, không sửa pilot cũ.
- Bệnh nhân cũ giữ nguyên fit/calibration/validation/pilot_test. Bệnh nhân mới dùng SHA-256 với namespace E05 và tỉ lệ 60/15/10/15, không dùng outcome để phân nhóm.
- Báo riêng pilot_test cũ và bệnh nhân pilot_test mới. Cả hai vẫn thuộc global train; không mở global final test.
- Giữ protocol v1, labels, eligibility, exposure 30 giây và alarm budget để so cùng định nghĩa. Audit chuyên khoa/loại mổ theo metadata; không gọi một bộ lọc tên đơn giản là xác nhận lâm sàng tiêu chí không tim.
- Đối chứng MAP/logistic/LightGBM; CatBoost không weighting, ba seed 20260917/18/19. CatBoost 300 cây/depth5/lr.05/l2=5 giống E04. Calibration riêng bệnh nhân. Lưới ngưỡng cố định cũ là primary; lưới validation quantile là ablation secondary.
- Ablation CatBoost seed 20260917: MAP-only, numeric không static, numeric+static. Mỗi cấu hình báo cả 5/10 phút; không chọn cấu hình từ test.
- Lưu role manifest trước tải track mới; source/config/data hashes trước train; mọi model/predictions/validation curve/CI và nhóm bệnh nhân đều có báo cáo. Không điều chỉnh matrix theo điểm test mới.

Các mốc cần môi trường ngoài (T4/Kaggle, MOVER, nhãn chuyên gia, tiến cứu) được theo dõi riêng; không đánh dấu hoàn thành bằng dữ liệu mô phỏng hoặc đổi tên file.
