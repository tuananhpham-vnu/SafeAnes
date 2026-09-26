Tổng hợp từ lần chạy toàn cohort ngày 24/09/2026. Các số liệu tương ứng nằm ở các mục bên dưới và ở các CSV.

1. **Dữ liệu đủ cho nhánh nguyên nhân, nhưng không đều giữa các ca.**
   - Waveform ART trích được beat features cho 3.548/3.626 ca eligible. Tỉ lệ beat hợp lệ median là 97,6%. Có 180 ca dưới 50% beat hợp lệ; nhiều ca trong số này có kênh SNUADC/ART chỉ là nhiễu quanh 0 dù Solar8000 vẫn ghi ART_MBP.
   - BIS, MAC/khí mê và thở máy có ở trên 90% ca. Ce remifentanil có ở 83%, Ce propofol ở 54% (TIVA), CVP ở 43%.
   - CO/SV đo bằng thiết bị ít hơn nhiều: EV1000 có ở 590 ca (16%), EV1000 SVR chỉ 239 ca, Vigileo 321 ca.
   - Thuốc vận mạch truyền qua bơm Orchestra có ở dưới 4% ca. Bolus, dịch, máu và EBL chỉ có tổng cuối ca.
   - Hb trong mổ có mốc thời gian chỉ ở 97 ca thuộc phạm vi outcome. Vì vậy nhãn "mất máu" gần như không có tín hiệu theo thời gian trong VitalDB.

2. **IOH phổ biến và phần lớn ngắn.**
   - 50,9% ca có ít nhất một biến cố; tổng cộng 5.306 biến cố, tức 0,60 biến cố mỗi giờ theo dõi.
   - Thời lượng median 186 s. MAP thấp nhất median 56 mmHg.
   - Prevalence theo decision row là 3,1% ở y5 và 6,2% ở y10.
   - Khoảng 25–30% biến cố thiếu lịch sử cho y5/y10, do rơi sát đầu ca hoặc sát biến cố trước.

3. **Protocol hiện tại gần như bỏ qua tụt huyết áp sau khởi mê.**
   - Nhãn chỉ tính trong khoảng opstart→opend. Chỉ 0,6% biến cố khởi phát trong 30 phút đầu sau bắt đầu gây mê.
   - Median từ bắt đầu gây mê đến rạch da khoảng 1 giờ, trong khi giai đoạn khởi mê là nơi giãn mạch do thuốc mê thường gặp nhất.
   - Nếu UC04 muốn gợi ý "giãn mạch do thuốc mê", cần mở rộng cửa sổ nhãn về anestart. Việc này phải làm dưới dạng protocol mới có đăng ký, không sửa protocol v1.

4. **Chất lượng nhãn: 4,9% biến cố có MAP thấp nhất ≤ 30 mmHg (2,2% ≤ 20).**
   - Nhiều khả năng đây là artefact đường động mạch: flush, lấy máu, zeroing, damping. Hiện chúng được giữ vì `BOUNDS` giữ mọi giá trị thấp.
   - Cần audit bằng beat SQI trước khi dùng các biến cố này để huấn luyện nhánh nguyên nhân.

5. **Trước IOH, tín hiệu nổi bật nhất là giảm SVR proxy và giảm dP/dt; SV proxy trung vị gần như không đổi.**
   - Median thay đổi trong 15 phút trước khởi phát: ΔlogMAP −0,060; ΔlogSVR −0,067; Δlog dP/dt −0,070; ΔlogSV +0,004; ΔPPV +0,8 điểm. Nhóm đối chứng có các giá trị này ≈ 0.
   - Sự sụt diễn ra từ từ từ khoảng −10 phút rồi tăng tốc trong 2–3 phút cuối. Vì vậy lead time khả thi cho nhánh cơ chế ngắn hơn nhánh nguy cơ.
   - Theo mức giảm ≥10%:

     | Tiêu chí | Trước IOH | Đối chứng |
     |---|---|---|
     | SVR giảm | 41,7% | 23,6% |
     | dP/dt giảm | 44,4% | 27,2% |
     | PPV cuối ≥ 13% | 56,5% | 39,4% |

   - PPV **mức nền** cao hơn ở nhóm trước IOH (khoảng 13,5% so với 11%), tương tự SV/SVR EV1000 nền thấp hơn. Tức là tình trạng tiền tải/hậu tải nền là yếu tố nguy cơ, không chỉ là thay đổi cấp.

6. **Phân rã SV × HR × SVR** trên các mốc có beat, trước IOH so với đối chứng:
   - SVR chi phối: 49% so với 32%. Trong số đó có kèm tăng thuốc mê: 15% so với 9%.
   - SV giảm kèm PPV cao: 13% so với 11%.
   - SV giảm kèm dP/dt giảm: 2% so với 2%.
   - HR giảm: 12% so với 14%.
   - Thay đổi nhỏ (dưới 5%): 23% so với 37%.
   - 15% biến cố thiếu dữ liệu beat.

   **Cảnh báo:** chỉ 27% biến cố có tăng thuốc mê trong 15 phút trước, so với 24% ở đối chứng. Do đó "giãn mạch do thuốc mê" khó tách khỏi giãn mạch nói chung nếu chỉ dựa vào Ce/MAC. Mẫu hình ở đây là heuristic, không phải nhãn.

7. **Cụm k-means** (k=5, 4.599 biến cố) gần với các endotype trong y văn:
   - Cụm 0 (1.787): hầu như không đổi, khó giải thích từ proxy.
   - Cụm 1 (1.646): SVR giảm cùng dP/dt giảm.
   - Cụm 2 (402): SV giảm, PPV tăng khoảng 8 điểm, dP/dt giảm; gợi ý tiền tải.
   - Cụm 3 (586): SVR giảm mạnh, SV/HR tăng; gợi ý giãn mạch với CO cao, giống endotype "severe vasodilation with high CI" của Kouz 2023.
   - Cụm 4 (178): PPV cực lớn, là **cụm artefact**; cho thấy bắt buộc có SQI.

8. **Proxy ART bám khá tốt theo thiết bị ở mức xu hướng** (ca có EV1000, ngoài global test; median theo ca):

   | Cặp so sánh | Spearman r | Đồng hướng xu hướng 5 phút |
   |---|---|---|
   | SV proxy vs EV1000 SV (508 ca) | 0,61 | 81% |
   | SVR proxy vs EV1000 SVR (201 ca) | 0,62 | 84% |
   | PPV vs SVV | 0,58 | 73% |

   Mức này đủ để dùng Δ tương đối và dùng EV1000 làm chuẩn bạc. Chưa đủ để thay giá trị tuyệt đối.

9. **NIBP không thay thế được ART cho nhãn:**
   - bias −3,4 mmHg, LoA −49 đến +43 mmHg;
   - 55,6% thời điểm ART < 65 mmHg có NIBP ≥ 65 mmHg.

10. **Gánh nặng IOH tăng theo nguy cơ bệnh nhân:**
    - Nhóm ASA / tỉ lệ ca có IOH / biến cố mỗi giờ: ASA 1 là 39% và 0,38; ASA 3 là 63% và 0,88; ASA 4 là 86% và 1,66 (chỉ 29 ca).
    - Tuổi trên 75: 67% và 0,79.
    - Mổ cấp cứu: 0,88 so với 0,56 biến cố mỗi giờ.
    - Mất máu: Spearman ρ giữa biến cố/giờ và EBL là 0,21, RBC 0,23, giảm Hb 0,32 (97 ca). Đây là liên hệ ở mức ca, không có mốc thời gian.

11. **Tải dữ liệu:** đủ 116.123/116.124 file track UC04. Một file (Orchestra/RFTN20_CP, ca 4613; đã có RFTN20_CE) bị API trả 403, đã ghi trong `data/vitaldb_full/fetch_log.csv`.
