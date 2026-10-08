## Dataset

Dữ liệu lấy từ **VitalDB**: 6.388 ca mổ không phải tim tại Bệnh viện Đại học Quốc gia Seoul. Chỉ dùng dữ liệu monitor:

| Nguồn | Tín hiệu | Tần số |
|---|---|---|
| Solar8000 | MAP, SBP, DBP, nhịp tim, SpO₂, EtCO₂, nhịp thở | khoảng 1–2 giây/lần |
| SNUADC/ART | Sóng huyết áp động mạch | 500 Hz |

Không dùng thuốc, BIS, máy gây mê hay thông tin trước mổ.

## Data Preprocessing

**1. Chọn ca**

| Bước | Điều kiện | Số ca |
|---|---|--:|
| 0 | Toàn bộ VitalDB | 6.388 |
| 1 | Người lớn, gây mê toàn thân, đủ thời gian mổ, có MAP động mạch | 3.626 |
| 2 | Tiền xử lý thành công (bỏ 4 ca có sóng sai tần số) | 3.622 |
| 3 | Có sóng động mạch | 3.544 |
| 4 | Arterial line thật sự hoạt động (MAP dùng được ≥ 10% thời gian mổ) | **3.363** |

**2. Xử lý tín hiệu**

| Tín hiệu | Xử lý |
|---|---|
| Chỉ số monitor | Đưa về lưới 2 giây, lấy giá trị đo gần nhất (cũ tối đa 30 giây), đánh dấu nhiễu |
| Sóng ART 500 Hz | Tách từng nhịp tim và loại nhịp lỗi; hạ xuống 100 Hz cho mô hình DL; đánh dấu đoạn mất sóng, flush, sóng phẳng |
| MAP để gán nhãn | Lưới 1 giây, đã làm sạch nhiễu; chỉ dùng để gán nhãn, không làm input |

Mọi đặc trưng tại thời điểm t chỉ dùng dữ liệu **trước t**, không nhìn tương lai.

**3. Định nghĩa nhãn**

- **Đợt tụt huyết áp:** MAP < 65 mmHg kéo dài ≥ 60 giây. Hai đợt cách nhau < 2 phút được gộp làm một.
- **Thời điểm dự đoán:** cứ 30 giây một lần.
  - Bỏ các thời điểm đang tụt, trong 2 phút sau đợt tụt, hoặc thiếu MAP.
- **Nhãn:**
  - y = 1 nếu có đợt tụt mới bắt đầu trong h phút tới (h = 5, 10, 15, 20, 30);
  - y = 0 nếu không có;
  - không xác định nếu thiếu dữ liệu theo dõi phía sau.

**4. Chia tập theo bệnh nhân.** Mọi ca của cùng một người nằm trong cùng một tập, để mô hình không được đánh giá trên người nó đã thấy khi train. Số liệu từng tập ở bảng dưới.

## Processed Data

**Dữ liệu của từng tập.** Ba tập gồm các bệnh nhân khác nhau, không trùng người.

| Tập | Mục đích | Số ca | Bệnh nhân | Số mẫu dự đoán | Số đợt tụt |
|---|---|--:|--:|--:|--:|
| Train | Huấn luyện | 2.393 | 2.302 | 747.762 | 4.922 |
| Calibration | Hiệu chỉnh xác suất | 258 | 244 | 78.263 | 476 |
| Validation | Chọn ngưỡng, báo cáo | 248 | 245 | 78.909 | 524 |
| Test | Đánh giá cuối (đang khóa, chưa dùng) | 464 | — | — | — |

- **Số ca và bệnh nhân:** một bệnh nhân có thể mổ nhiều lần, nên số bệnh nhân ít hơn số ca (ví dụ calibration có 258 ca từ 244 người).
- **Số mẫu dự đoán:** số thời điểm mô hình đưa ra dự đoán (30 giây một lần). AUROC và AUPRC được tính trên các mẫu này.
- **Số đợt tụt:** số lần bệnh nhân thực sự tụt huyết áp. Độ nhạy và PPV được tính trên các đợt tụt này.

**Tỷ lệ mẫu dương trên train theo horizon:** dữ liệu mất cân bằng mạnh, nhất là ở mốc ngắn.

| Horizon | 5 phút | 10 phút | 15 phút | 20 phút | 30 phút |
|---|--:|--:|--:|--:|--:|
| Tỷ lệ dương | 4,1% | 7,9% | 11,5% | 14,9% | 21,0% |

- Tỷ lệ dương tính từ **nhãn thật** của dữ liệu (dựa vào MAP đo được sau đó), không phải do mô hình đoán.
- Mô hình luôn dự đoán 30 giây một lần. Horizon là khoảng thời gian dự báo trước: "trong h phút tới có tụt huyết áp không?".
- Ví dụ ở 5 phút: cứ 100 thời điểm thì chỉ khoảng 4 thời điểm có tụt huyết áp trong 5 phút tới.
- Nhìn càng xa (horizon dài) thì càng dễ gặp một đợt tụt, nên tỷ lệ này tăng dần.
- Vì vậy ở horizon dài, cảnh báo dễ đúng hơn. PPV cao hơn ở 30 phút không có nghĩa là mô hình giỏi hơn.

**Đầu vào cho mô hình**

| Loại | Số lượng | Nội dung |
|---|--:|---|
| Đặc trưng chỉ số | 66 | Mean, std, min, max, độ dốc, tỷ lệ thiếu và giá trị hiện tại của 9 tín hiệu; % MAP giảm so với MAP nền; thời gian MAP ở vùng 65–75; MAP dự đoán sau 5 phút |
| Đặc trưng sóng | 27 | Đặc trưng hình dạng từng nhịp tim (dP/dt, diện tích tâm thu, thời gian tống máu…), PPV, SV/CO/SVR ước lượng |
| Sóng thô (chỉ DL) | W × 100 điểm | Sóng 100 Hz đã chuẩn hóa, kèm kênh đánh dấu nhiễu |

Cửa sổ nhìn lại W = 30, 60, 90 hoặc 120 giây.

---

## Mô hình học sâu Conv1D + Transformer trên sóng động mạch: dl_conv_tf

Khác với các mô hình trước dùng đặc trưng tính sẵn, mô hình này **đọc trực tiếp sóng huyết áp động mạch**. Cứ 30 giây, mô hình xử lý hai nguồn:

- **Sóng 100 Hz trong W giây gần nhất:** Conv1D trích các mẫu hình ngắn của sóng, rồi Transformer học quan hệ giữa các đoạn sóng theo thời gian.
- **66 đặc trưng chỉ số monitor:** giống `lgbm_numeric`.

Hai nhánh được ghép lại để ra xác suất tụt huyết áp cho cả 5 horizon cùng lúc. Mô hình được train 5 lần với 5 seed khác nhau, rồi lấy trung bình 5 lần đó để kết quả ổn định.

| Thiết lập | Giá trị |
|---|---|
| Kiến trúc | 3 lớp Conv1D + Transformer 2 lớp (sóng); MLP (chỉ số) |
| Huấn luyện | AdamW, batch 256, tối đa 20 epoch, dừng sớm sau 4 epoch không cải thiện |
| Số lần train | 4 cửa sổ W × 5 seed = 20 lần, trên Kaggle GPU T4 x2 |

**Ảnh hưởng của cửa sổ W:** cửa sổ dài hơn cho kết quả tốt hơn, nhưng chênh lệch nhỏ.

| W | AUROC 5 phút | AUROC 10 phút | Độ nhạy 5 phút | Độ nhạy 10 phút |
|---|--:|--:|--:|--:|
| 30 giây | 0,857 | 0,801 | 62,0% | 65,0% |
| 60 giây | 0,867 | 0,811 | 62,0% | 67,1% |
| 90 giây | 0,872 | 0,818 | 62,3% | 67,1% |
| **120 giây** | **0,873** | **0,820** | **62,5%** | **67,4%** |

**Kết quả với W = 120 theo horizon**

| Horizon | AUROC | AUPRC | Độ nhạy theo đợt tụt | Cảnh báo sai/giờ | PPV | Báo trước (trung vị) |
|---|--:|--:|--:|--:|--:|--:|
| 5 phút | 0,873 | 0,272 | 62,5% | 0,96 | 29,7% | 1,7 phút |
| 10 phút | 0,820 | 0,308 | 67,4% | 0,99 | 36,9% | 3,0 phút |
| 15 phút | 0,791 | 0,345 | 67,5% | 0,98 | 41,5% | 3,6 phút |
| 20 phút | 0,780 | 0,392 | 69,5% | 0,98 | 47,0% | 4,3 phút |
| 30 phút | 0,767 | 0,472 | 70,4% | 0,98 | 55,1% | 7,1 phút |

**So sánh với các mô hình khác (W = 120, horizon 5 phút)**

| Mô hình | AUROC | Độ nhạy theo đợt tụt | Cảnh báo sai/giờ | PPV |
|---|--:|--:|--:|--:|
| map_threshold | 0,828 | 56,0% | 0,83 | 30,6% |
| map_logistic | 0,839 | 62,3% | 0,98 | 29,2% |
| lgbm_numeric | 0,874 | 62,7% | 0,94 | 30,4% |
| lgbm_wave | **0,875** | **63,5%** | 0,99 | 29,4% |
| **dl_conv_tf** | 0,873 | 62,5% | 0,96 | 29,7% |

**Nhận xét**

Việc dùng mô hình học sâu để học trực tiếp từ sóng động mạch chưa cải thiện được khả năng dự báo. Ở horizon 5 phút, Conv1D + Transformer đạt AUROC 0,873 và độ nhạy 62,5%, gần như bằng lgbm_numeric (0,874 và 62,7%) dù mô hình này chỉ dùng chỉ số monitor. Ba cách tiếp cận khác nhau là lgbm_numeric, lgbm_wave và Conv1D + Transformer đều dừng ở cùng một mức, nên giới hạn có lẽ nằm ở dữ liệu đầu vào hơn là ở mô hình. Mô hình học sâu chỉ nhỉnh hơn một chút ở horizon dài, với độ nhạy 70,4% ở 30 phút so với khoảng 68–69% của LightGBM, nhưng chênh lệch này còn nhỏ để kết luận. Ngoài ra, thời gian báo trước ở horizon 5 phút chỉ khoảng 1,7 phút, nghĩa là mô hình thường cảnh báo khi huyết áp đã bắt đầu giảm. Với hiệu năng tương đương nhưng chi phí huấn luyện thấp hơn nhiều, LightGBM vẫn là lựa chọn hợp lý hơn ở giai đoạn này.
