# Chú thích hình — reports/EDA

Mỗi mục `## <tên file hình>` là chú thích của một hình. `run_eda.py` và `run_insights.py` tự chèn chú thích ngay dưới hình
và đánh số theo thứ tự xuất hiện trong từng báo cáo. Sửa chú thích ở file này rồi chạy lại script, không sửa trực tiếp trong
REPORT.md hoặc INSIGHTS.md. Dòng đầu mỗi mục là tiêu đề hình. Các số liệu lấy từ lần chạy toàn cohort ngày 24/09/2026.

## track_availability.png
Mức sẵn có của các track liên quan UC04.
**Cách đọc:** mỗi thanh là một track VitalDB. Độ dài thanh là % trong 3.626 ca eligible có track đó. Màu là nhóm sinh lý: lõi, huyết động, cung lượng tim, độ mê, vận mạch, thở máy, waveform.
**Nhận xét:**
- Tín hiệu monitor, máy thở, BIS và waveform ART có ở trên 90% ca.
- Ce remifentanil có ở 83% ca, Ce propofol ở 54% (chỉ ca TIVA), CVP ở 43%.
- Cung lượng tim đo bằng thiết bị hiếm: EV1000 16%, EV1000 SVR 7%, Vigileo 9%.
- Thuốc vận mạch truyền qua bơm có ở dưới 4% ca, nên không làm input chung được; chỉ dùng làm thí nghiệm tự nhiên.

## beat_snippets.png
Kiểm tra trực quan thuật toán phát hiện nhịp trên waveform ART.
**Cách đọc:** mỗi hàng là 20 giây waveform ART thô (đường xám) giữa ca của một ca mẫu. Mỗi vạch đứng nối DBP với SBP của một nhịp được phát hiện; xanh là nhịp hợp lệ, đỏ là nhịp bị loại.
**Nhận xét:**
- Ở các ca có đường động mạch tốt, thuật toán bắt đúng chân sóng và đỉnh của từng nhịp, kể cả khi có sóng dội.
- Ca 3 và ca 14: kênh SNUADC chỉ là nhiễu khoảng ±30 mmHg quanh 0, trong khi monitor vẫn ghi MAP. Thuật toán loại toàn bộ các nhịp này, đúng như mong muốn.
- Đây là lý do nhánh giải thích cần cổng chất lượng tín hiệu (SQI).

## nibp_vs_art.png
Đồng thuận giữa huyết áp không xâm lấn (NIBP) và huyết áp động mạch xâm lấn (ART).
**Cách đọc:** biểu đồ Bland–Altman. Mỗi điểm là một lần đo NIBP ghép với MAP động mạch trong 10 giây trước đó (365.209 cặp, 282 ca). Trục ngang là trung bình hai phép đo, trục dọc là NIBP − ART. Đường liền là độ lệch trung bình; hai đường đứt là giới hạn đồng thuận 95%.
**Nhận xét:**
- Độ lệch trung bình nhỏ (−3,4 mmHg), nhưng giới hạn đồng thuận rất rộng (−49 đến +43 mmHg).
- Các vệt chéo xuất hiện vì NIBP giữ nguyên giá trị giữa hai lần đo trong khi ART thay đổi.
- Ở 55,6% thời điểm ART < 65 mmHg, NIBP vẫn ≥ 65 mmHg. Vì vậy NIBP không thay ART để gán nhãn IOH được.

## events_overview.png
Tổng quan các biến cố IOH trong phạm vi phát triển (3.110 ca, 5.306 biến cố).
**Cách đọc:** sáu panel mô tả phân bố:
- số biến cố mỗi ca (≥ 15 gộp);
- thời lượng biến cố (thang log);
- thời điểm khởi phát tính từ lúc bắt đầu gây mê (đường đứt là median thời điểm rạch da; cột cuối gộp ≥ 6 giờ);
- MAP thấp nhất trong biến cố (≤ 30 gộp);
- diện tích dưới ngưỡng 65 mmHg;
- % thời gian MAP < 65 của từng ca.

**Nhận xét:**
- Khoảng một nửa số ca không có biến cố nào; phân bố số biến cố lệch phải.
- Đa số biến cố kéo dài 1–5 phút (median 186 s).
- Khởi phát tập trung ngay sau rạch da; gần như không có biến cố trong 30 phút đầu vì protocol chỉ tính từ opstart.
- MAP thấp nhất có median 56 mmHg. Cột ≤ 30 mmHg (4,9% biến cố) nhiều khả năng là artefact.
- Median theo ca của % thời gian MAP < 65 chỉ khoảng 2%, nhưng có đuôi dài.

## trajectories_core.png
Quỹ đạo huyết áp và các đặc trưng theo nhịp trong 15 phút trước và 5 phút sau khởi phát IOH.
**Cách đọc:**
- Xanh: median các biến cố IOH, căn theo thời điểm khởi phát (phút 0).
- Cam: mốc đối chứng có MAP ≥ 65 mmHg và không có IOH trong ±15 phút.
- Dải màu: khoảng tứ phân vị (IQR).
- Các tín hiệu có nhãn "tỉ lệ so với nền" được chia cho giá trị nền (−15 đến −10 phút) của chính ca đó, nên 1,0 nghĩa là không đổi.

**Nhận xét:**
- MAP tách khỏi nhóm đối chứng từ khoảng −8 phút, giảm dần chừng 5%, rồi rơi nhanh trong 2–3 phút cuối (đáy khoảng 0,85 lần nền).
- SVR proxy, dP/dt và áp lực mạch (PP) có cùng hình dạng (giảm 13–19% tại đáy). SV/CO proxy và HR gần như phẳng.
- PPV của nhóm trước IOH cao hơn đối chứng suốt 15 phút (khoảng 13,5% so với 11%).
- Chữ ký chính là giảm hậu tải. dP/dt động mạch giảm cùng SVR, nên nó phản ánh cả trương lực mạch, không chỉ sức co bóp.

## trajectories_context.png
Quỹ đạo các tín hiệu ngữ cảnh (độ mê, thuốc, thở máy, thiết bị đo cung lượng tim) quanh khởi phát IOH.
**Cách đọc:** cùng cách căn thời gian và màu như hình trước, nhưng hiển thị **giá trị tuyệt đối**. Tín hiệu chỉ có ở một phần ca, nên số ca (n) trong chú giải khác nhau giữa các panel.
**Nhận xét:**
- **BIS:** không khác giữa hai nhóm.
- **Ce propofol:** thấp hơn ở nhóm IOH và giảm tiếp sau khởi phát.
- **Ce remifentanil:** tăng từ khoảng 2,5 lên 2,8 ng/mL trong 5 phút trước khởi phát, rồi giảm sau đó (bác sĩ hạ liều). Điều này phù hợp với giãn mạch do opioid.
- **EV1000:** SV nền thấp hơn (khoảng 64 so với 68 mL), SVR nền thấp hơn (khoảng 960 so với 1.140) và giảm tại khởi phát; SVV tăng sau khởi phát.
- **NIBP:** median thấp hơn (74 so với 82 mmHg).
- **MAC, PEEP, CVP:** median không khác. Median MAC bằng 0 vì nhiều ca là TIVA.
- **Phenylephrine:** chỉ có ở rất ít ca, không đủ để kết luận.

## decomposition.png
Phân rã thay đổi MAP trước IOH thành SV × HR × SVR (proxy từ waveform ART).
**Cách đọc:**
- **Trái:** tỉ lệ các mẫu hình heuristic, xác định theo thành phần giảm mạnh nhất trong 15 phút trước mốc; chỉ tính các mốc có dữ liệu beat. Xanh là trước IOH, cam là đối chứng.
- **Phải:** mỗi điểm là một mốc. Trục ngang là Δlog SVR proxy, trục dọc là Δlog SV proxy; điểm ở nửa trái là SVR giảm.

**Nhận xét:**
- "Giảm SVR" (tính cả trường hợp kèm tăng thuốc mê) chiếm 49% biến cố so với 32% đối chứng. Đây là khác biệt lớn nhất.
- "Thay đổi nhỏ" ít hơn ở nhóm IOH (23% so với 37%).
- "Giảm SV + PPV cao" chỉ nhỉnh hơn (13% so với 11%). Mẫu hình co bóp và nhịp chậm không khác đối chứng.
- Ở panel phải, điểm trước IOH dồn sang trái và rải theo một đường chéo âm, tức SV và SVR proxy bù trừ nhau.
- Mẫu hình ở đây là heuristic, không phải nhãn nguyên nhân.

## clusters.png
Năm cụm k-means của thay đổi huyết động trước IOH (4.599 biến cố có đủ dữ liệu beat).
**Cách đọc:** mỗi nhóm cột là một đặc trưng thay đổi (15 phút trước khởi phát). Màu là cụm; chiều cao cột là median trong cụm. ΔPPV đã chia 10 để cùng thang với các đặc trưng khác.
**Nhận xét:**

| Cụm | Số biến cố | Mẫu hình | Diễn giải |
|---|---|---|---|
| 0 | 1.787 | hầu như không đổi | proxy không giải thích được |
| 1 | 1.646 | SVR giảm, dP/dt giảm | giãn mạch |
| 2 | 402 | SV giảm, PPV tăng khoảng 8 điểm, dP/dt giảm | gợi ý thiếu dịch / tiền tải |
| 3 | 586 | SVR giảm mạnh, SV và HR tăng | giãn mạch với CO cao |
| 4 | 178 | PPV tăng rất lớn | artefact |

- Cụm 3 giống endotype "giãn mạch nặng, CI cao" trong y văn (Kouz 2023).
- Cụm 4 cho thấy cần SQI cho PPV.

## proxy_vs_ev1000.png
Kiểm chứng proxy từ waveform ART bằng thiết bị đo cung lượng tim EV1000 (ca ngoài global test).
**Cách đọc:**
- **Trái:** phân bố hệ số Spearman r theo từng ca, trên lưới 30 s, giữa proxy và giá trị EV1000.
- **Phải:** tỉ lệ đồng hướng của thay đổi 5 phút. Chỉ xét các thay đổi mà EV1000 dao động > 10% (vùng loại). Giá trị 1 nghĩa là proxy luôn đi cùng chiều với thiết bị.

**Nhận xét:**

| Cặp so sánh | Số ca | Median r | Median đồng hướng |
|---|---|---|---|
| SV proxy vs EV1000 SV | 508 | 0,61 | 81% |
| SVR proxy vs EV1000 SVR | 201 | 0,62 | 84% |
| PPV vs SVV | 508 | 0,58 | 73% |

- Có một đuôi nhỏ các ca với r ≈ 0 hoặc âm, là ca proxy hỏng (damping, vị trí đầu dò).
- Proxy đủ dùng cho **thay đổi tương đối**, chưa đủ để thay giá trị tuyệt đối.

## bleeding.png
Mất máu và gánh nặng IOH ở mức ca.
**Cách đọc:**
- **Trái:** 89 ca có cả Hb trước mổ và Hb trong mổ. Trục ngang là mức giảm Hb, trục dọc là số biến cố IOH mỗi giờ (giá trị ≥ 6 gộp).
- **Phải:** 2.278 ca có EBL. Trục ngang là EBL theo thang log.

**Nhận xét:**
- Liên quan dương nhưng yếu: Spearman ρ = 0,32 với giảm Hb, 0,21 với EBL.
- Đám điểm rất rộng. EBL tập trung ở các giá trị làm tròn (50, 100, 200… mL).
- EBL chỉ là tổng cuối ca, nên không dùng được cho dự báo hay gán nguyên nhân "mất máu" theo thời gian.

## insight_missing.png
Mức thiếu dữ liệu của các feature chính tại thời điểm dự báo.
**Cách đọc:** mỗi thanh là % thời điểm dự báo (407.490 điểm, 2.961 ca) mà feature đó không có giá trị. Màu là nhóm tín hiệu. `_cur` là giá trị hiện tại; `_d300` là thay đổi so với 5 phút trước.
**Nhận xét:**
- MAP và các tín hiệu monitor gần như đầy đủ.
- Beat features thiếu khoảng 3% (ca có kênh ART hỏng). BIS thiếu 7,5%, Ce remifentanil 14,7%.
- Ce propofol thiếu 47% và CVP thiếu 55%, theo chủ ý lâm sàng: chỉ có khi gây mê TIVA hoặc có catheter trung tâm.
- Việc thiếu **không ngẫu nhiên** và mang thông tin. Mô hình nên nhận mask thiếu thay vì điền giá trị.

## insight_noise.png
Mức nhiễu của các track thô trong thời gian mổ (3.626 ca eligible).
**Cách đọc:**
- **Trái:** median theo ca của % thời gian mổ không có giá trị hợp lệ mới. Ngưỡng là 30 s; 60 s với BIS, thuốc, CVP, nhiệt độ; 600 s với NIBP.
- **Giữa:** % ca bị thiếu hơn 20% thời gian mổ.
- **Phải:** % mẫu nằm ngoài ngưỡng sinh lý (tiêu chí từng tín hiệu ở bảng bên dưới hình).

**Nhận xét:**
- Tín hiệu lõi thiếu dưới 0,5% thời gian. Tuy vậy 5,9% ca mất hơn 20% thời gian ART.
- NIBP thiếu khoảng 61% thời gian, vì đo gián đoạn và thường ngừng khi đã có ART.
- Nhiễu giá trị cao nhất ở CVP (5,4% mẫu bất thường; p99 = 252 mmHg do zeroing/flush) và BIS (3,9% mẫu có SQI < 50).
- 0,7% mẫu MAP bất thường. Con số nhỏ, nhưng nằm ở chính tín hiệu tạo nhãn.

## insight_corr.png
Tương quan Spearman giữa 37 feature chính.
**Cách đọc:** ma trận đối xứng, sắp theo phân cụm phân cấp (khoảng cách 1 − |ρ|), nên các feature liên quan nằm cạnh nhau. Đỏ là tương quan dương, xanh là âm. Số chỉ hiện khi |ρ| ≥ 0,5.
**Nhận xét:**
- **Khối mức huyết áp:** MAP, SBP, DBP, PP, dP/dt, SVR proxy tương quan 0,5–0,9, tức là rất dư thừa.
- **Khối thay đổi 5 phút:** độ dốc MAP, Δ SVR, Δ dP/dt, Δ SV/CO, ρ khoảng 0,5–0,6. MAP giảm đi cùng SVR và dP/dt giảm.
- **SV proxy và SVR proxy:** ρ = −0,70.
- **Ce remifentanil và MAC:** ρ = −0,6, vì TIVA và khí mê thay thế nhau.
- **Thông tin trước mổ** gần như độc lập với tín hiệu động: bổ sung thông tin nhưng yếu.

## insight_future.png
Các feature liên quan nhất với thay đổi MAP trong 5 phút tới.
**Cách đọc:** 25 feature có |ρ| Spearman lớn nhất giữa giá trị tại thời điểm t và ΔMAP(t → t + 5 phút). ρ âm nghĩa là giá trị hiện tại càng cao thì MAP sắp tới càng giảm. Đây là quan hệ dự báo, không phải nhân quả.
**Nhận xét:**
- Hầu hết là mức huyết áp gần đây (ρ từ −0,18 đến −0,27), phần lớn phản ánh hồi quy về trung bình.
- Ngoại lệ duy nhất ngoài nhóm huyết áp là **Ce remifentanil vừa tăng trong 5 phút** (ρ = −0,25, hạng 3). Khi remifentanil vừa được tăng, MAP thường giảm trong 5 phút tới.
- Độ dốc DBP 1 phút có ρ dương (+0,19): DBP đang tăng thì thường tiếp tục tăng.

## insight_risk_curves.png
Nguy cơ IOH trong 5 phút tới theo giá trị của 12 feature.
**Cách đọc:** mỗi điểm là một decile của feature. Trục ngang là median của decile; trục dọc là % thời điểm có IOH trong 5 phút tới. Đường đứt là tỉ lệ nền 3,1%.
**Nhận xét:**
- **MAP hiện tại:** nguy cơ tăng rất nhanh dưới 75 mmHg (15% ở decile thấp nhất) và gần 0 trên 85 mmHg.
- **Dạng chữ U:** xu hướng MAP, Δ SVR, Δ SV, Δ dP/dt. Biến động mạnh theo cả hai chiều đều làm tăng nguy cơ.
- **Tăng đơn điệu:** độ dao động MAP, HR (4,9% khi HR khoảng 100), PPV, tuổi (4,1% ở khoảng 80 tuổi), Δ Ce remifentanil (5,3% khi tăng khoảng 1 ng/mL).
- **Ce propofol:** giảm từ 4,5% (2 µg/mL) xuống 1,5% (3,5 µg/mL), nhưng bị nhiễu bởi loại gây mê.
- **BIS:** gần như phẳng (2,8–3,5%).

## insight_level_trend.png
Nguy cơ IOH trong 5 phút tới theo mức MAP × xu hướng MAP.
**Cách đọc:** cột là MAP hiện tại theo khoảng 5 mmHg; hàng là độ dốc MAP 5 phút (mmHg/phút, âm là đang giảm). Màu và số trong ô là % thời điểm có IOH trong 5 phút tới. Ô trống có dưới 200 thời điểm.
**Nhận xét:**
- **Mức MAP quyết định chính:** MAP 65–70 có nguy cơ 12–33%; 60–65 (chưa thành biến cố) có 29–46%; từ 90 trở lên ≤ 3% (đa số ô ≤ 1%).
- **Xu hướng chỉ thêm thông tin ở vùng giữa (70–85):** ở MAP 75–80, đang giảm 5–6 mmHg/phút có 6%, còn ổn định có 1–2%.
- **MAP thấp đang tăng vẫn nguy cơ cao** (65–70 và tăng 3–4 mmHg/phút: 33%). Đây thường là giai đoạn vừa hồi phục sau một biến cố và dễ tái phát.

## insight_univariate.png
25 feature đơn lẻ phân biệt IOH 5 phút tốt nhất.
**Cách đọc:** độ dài thanh là AUROC khi dùng riêng một feature để xếp hạng nguy cơ. Trục bắt đầu từ 0,5 (mức ngẫu nhiên). AUROC lấy max(AUC, 1 − AUC), nên chỉ đo độ mạnh, không đo chiều; chiều ở bảng bên dưới hình. Màu là nhóm tín hiệu.
**Nhận xét:**
- 24/25 feature thuộc nhóm MAP/SBP/DBP (AUROC 0,65–0,81).
- Ngoại lệ duy nhất là SVR proxy hiện tại (0,72, hạng 15).
- DBP gần ngang MAP (0,798), phù hợp với việc DBP phản ánh trương lực mạch. SBP yếu hơn.

## insight_shap.png
Đóng góp của feature trong mô hình đa biến (LightGBM + TreeSHAP).
**Cách đọc:**
- **Trái:** 20 feature có mean|SHAP| lớn nhất, tính theo % tổng. Mô hình LightGBM 173 feature, huấn luyện trên tập FIT, đo trên 40.000 thời điểm VALIDATION.
- **Phải:** tổng theo nhóm tín hiệu.
- SHAP đo mức mô hình **dùng** feature, không phải quan hệ nhân quả. Các feature tương quan cao chia nhau phần đóng góp.

**Nhận xét:**
- MAP chiếm 37%, SBP/DBP 24%.
- Nhóm độ mê / thuốc mê đứng thứ ba với 14%: MAC hạng 4, Δ Ce remifentanil hạng 6, Ce propofol hạng 12. Đây là thông tin mô hình khai thác được ngoài huyết áp.
- Beat features 8% (SVR proxy hạng 9).
- SpO2, thở máy và CVP dưới 1%.

## insight_ablation.png
Ablation nhóm tín hiệu cho dự báo IOH 5 phút (VALIDATION).
**Cách đọc:** mỗi thanh là một mô hình LightGBM được huấn luyện lại.
- Xanh: chỉ MAP cộng thêm một nhóm.
- Cam: mô hình đầy đủ bỏ đi một nhóm.
- Xám: hai mô hình tham chiếu (tất cả, chỉ MAP).
- Trục ngang đã phóng to (khoảng 0,868–0,883), nên chênh lệch trông lớn hơn thực tế.

**Nhận xét:**
- Mọi cấu hình nằm trong khoảng AUROC 0,873–0,881.
- Thêm nhóm độ mê / thuốc mê tăng nhiều nhất (+0,0045). Bỏ HR làm giảm nhiều nhất (−0,0024).
- AP dao động 0,24–0,26, không theo thứ tự rõ ràng.
- Mô hình đầy đủ so với chỉ MAP: ΔAUROC +0,008 (95% CI +0,001 đến +0,015), ΔAP +0,001 (−0,022 đến +0,029).
- Kết luận: thêm tín hiệu chỉ cải thiện rất ít.
