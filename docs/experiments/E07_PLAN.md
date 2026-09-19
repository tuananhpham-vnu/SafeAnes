# E07 — đánh giá toàn bộ VitalDB hiện có, khóa trước khi mở final test

Người dùng yêu cầu đánh giá toàn bộ VitalDB ngày 19/09/2026; yêu cầu này cho phép
mở global final test để đánh giá các model đã khóa. Không chạy `tests/`.

- Audit toàn bộ 6.388 ca từ metadata VitalDB đã lưu cùng provenance. Có 3.626 ca
  đủ điều kiện protocol v1; mọi ca không đủ điều kiện vẫn có lý do loại trong manifest.
- Xử lý tất cả ca đủ điều kiện ở mọi split. Không gọi đánh giá trên ca fit là khả năng
  tổng quát hóa. Phân nhóm theo subjectid: development_seen và các nhóm chưa dùng
  global train/calibration/validation/test. Những lần mổ mới của bệnh nhân đã biết
  vẫn thuộc development_seen.
- Primary: TabM ensemble E06 (ba seed đều) so với CatBoost numeric E05 seed 20260917,
  300/600 giây, ngưỡng fixed đã khóa từ validation E05/E06. Không train, recalibrate,
  đổi threshold, lựa seed hoặc chọn model bằng kết quả toàn bộ VitalDB.
- Secondary: CatBoost all-features seed 20260917, MAP E05, LightGBM monotone E06.
- Kết luận xác nhận chính dựa trên global test: 516 ca/502 bệnh nhân. Báo riêng các
  nhóm chưa dùng khác, development_seen và toàn bộ cohort (mô tả, có phần in-sample).
- Dùng nguyên build_case/labels/eligibility/evaluator của protocol v1, raw numeric 7
  track; không nội suy waveform hoặc thay arterial MAP bằng NIBP. Tái sử dụng cache
  đã kiểm hash, download GET vào cache mới, tiếp tục được sau gián đoạn.
- Đăng ký source/model/threshold/metadata hashes trước tải track chưa dùng. Lưu
  kết quả theo ca để resume, mọi lỗi phải được báo; không lặng lẽ loại ca khó.
- Point metrics AUROC/AP/Brier/ECE và alarm recall/PPV/FAH; 5.000 bootstrap ghép cặp
  theo bệnh nhân cho chênh lệch alarm của primary trên global test. Không tune sau xem
  kết quả. Báo cả mục tiêu không đạt và coverage/exposure v1 còn xấp xỉ lưới 30 giây.
- Một benchmark retrospective trên một nguồn không thay kiểm định ngoài/tiến cứu.
