# E06 — thay thế mô hình, thăm dò trong v0.2

Khóa trước chạy E06. Dùng lại development300 và roles E05; holdout đã được xem ở E05,
nên **mọi kết quả E06 là exploratory**, không phải kiểm định độc lập. Không mở final test.

- TabM chính thức `tabm==0.0.3`, `rtdl_num_embeddings==0.0.12`: ba seed 20260917/18/19;
  16 thành viên, hai block width 64, PLE 8 bins/4 chiều, dropout .1, AdamW .002/.0003,
  batch 512, tối đa 30 epoch, patience 6. Đây là cấu hình CPU của dự án.
- Chỉ numeric features E05 (không static). Median imputation, bỏ feature hằng,
  quantile-normal và bins đều fit trên bệnh nhân optimization. Inner stopping là
  SHA256 `E06-stop:{subjectid}` modulo 5 == 0, chỉ trong FIT; chọn bằng BCE, không refit.
- Hai logits có thứ tự, loss BCE trung bình từng thành viên/horizon, bỏ nhãn censored.
  Sigmoid calibration riêng mỗi horizon trên role calibration; chiếu cặp xác suất
  vi phạm về trung bình để bảo đảm p10 >= p5 sau calibration.
- Đối chứng bổ sung LightGBM numeric: 500 cây, 7 leaves, lr .03, min_child_samples 150,
  reg_lambda 10; ràng buộc giảm theo map_current, map_60_mean, map_300_mean.
  Đây là giả thuyết regularization của dự án, không gọi LightGBM là SOTA mới.
- Thêm ensemble trung bình xác suất ba TabM; không chọn seed theo test.
- Giữ labels, eligibility, alarm persistence/cooldown, exposure và gates E05.
  Ngưỡng primary lưới 40 điểm cũ, secondary quantiles 201. Không đổi budget .5 FA/giờ.
- Khóa đề xuất model theo validation primary: ưu tiên đạt toàn bộ gates, rồi nằm
  trong budget, recall, PPV, AUROC; tie-break theo tên. Chỉ chọn giữa ensemble TabM
  và monotone LightGBM. Sau đó mới đánh giá holdout, báo mọi model và baseline E05.
- CI 200 bootstrap theo bệnh nhân cho primary nhóm 37 bệnh nhân của E05;
  báo đủ nhóm 9 cũ, 37 mở rộng, và gộp. Lưu snapshot, hashes, model, calibration,
  predictions, curve, training history và kiểm tra nạp lại model.

Nguồn và lý do chọn: [review SOTA E06](../SOTA_E06.md).
