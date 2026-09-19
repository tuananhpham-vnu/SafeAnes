# v0.4 — ngưỡng cảnh báo và CatBoost trên pilot cố định

Khóa thiết kế trước chạy: giữ 60 ca, patient roles, nhãn, persistence, cooldown và mọi quality gate của v0.3. Không mở global test. Mở rộng cohort là bước riêng sau khi xác định tác động của policy.

1. Replay các predictions TCN và ensemble đã lưu với lưới cũ và lưới mới: hợp 40 mức cũ, 201 quantile của xác suất validation eligible và sentinel 1.000001. Không lấy quantile từ test. Quy tắc chọn giữ nguyên: ưu tiên đủ gate, rồi FA/giờ <=0.5, tối đa recall/PPV/ngưỡng. Đây là ablation policy, không phải model mới.
2. CatBoost CPU 300 cây, depth=5, learning_rate=.05, l2_leaf_reg=5; hai cấu hình không weighting và SqrtBalanced; ba seed 20260917/18/19. Imputation median chỉ fit trên fit patients; calibration sigmoid trên calibration patients. Không early stopping/tuning trên test. Mỗi model chạy cả hai policy để tách tác động.
3. Báo cáo mọi run, CI bootstrap 200 theo bệnh nhân. Chọn ứng viên bằng validation; không chọn seed tốt nhất bằng test. Lưu registration, source, predictions, model, validation curves và kết quả riêng.

Nguồn: [benchmark NeurIPS 2022](https://proceedings.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html) hỗ trợ thử cây trên tabular, không chứng minh SOTA IOH. [CatBoost chính thức](https://catboost.ai/docs/en/references/training-parameters/common) mô tả SqrtBalanced; không kết hợp với class_weights/scale_pos_weight. Các thông số và lưới ngưỡng là lựa chọn dự án, không phải tái lập nguyên bản paper.

Pilot_test đã được xem ở version trước; kết quả tiếp tục exploratory. Chỉ 3/4 event đủ điều kiện ở 5/10 phút; không suy ra cải thiện chắc chắn hoặc đạt triển khai từ vài cảnh báo.
