Phạm vi: 2.961 ca phát triển có ít nhất một thời điểm dự báo hợp lệ (149 ca còn lại không có, chủ yếu vì lịch sử MAP không đủ);
407.490 thời điểm (mỗi 60 s); 173 feature. Nhiễu đo trên 3.626 ca eligible.

### 1. Dữ liệu là gì — nói ngắn gọn

- Mỗi thời điểm dự báo có **173 feature**, chia thành 9 nhóm:
  - 7 tín hiệu monitor: MAP, SBP, DBP, HR, SpO2, EtCO2, RR; mỗi tín hiệu có thống kê 1/5/10 phút;
  - 14 feature theo nhịp từ **waveform ART**;
  - độ mê và thuốc mê: BIS, MAC, Ce propofol/remifentanil;
  - thở máy và CVP;
  - thông tin trước mổ.
- Nhãn rất **mất cân bằng**: chỉ khoảng **3,1%** thời điểm có IOH trong 5 phút tới (6,2% với 10 phút). Accuracy vì vậy vô nghĩa; phải dùng AUROC/AP, PPV và cảnh báo giả/giờ.
- **Thiếu dữ liệu theo cấu trúc, không ngẫu nhiên:**
  - Tín hiệu lõi gần như đầy đủ.
  - Beat features thiếu 2,9% (ca có kênh ART hỏng).
  - BIS thiếu 7,5%; Ce remifentanil thiếu 14,7%.
  - Ce propofol chỉ có khi gây mê TIVA (thiếu 47%).
  - CVP chỉ có khi đặt catheter trung tâm (thiếu 55%).
  - Việc thiếu tự nó mang thông tin (loại gây mê, loại mổ), nên mô hình cần mask thay vì điền giá trị.

### 2. Nhiễu — dữ liệu sạch hơn dự kiến, nhưng nhiễu nằm đúng chỗ nguy hiểm

- **Tín hiệu lõi rất sạch.** MAP, HR, SpO2, EtCO2 thiếu dưới 0,5% thời gian mổ (median theo ca). Nhưng **5,9% ca** mất hơn 20% thời gian ART.
- **Nhiễu tập trung ở huyết áp động mạch, chính là tín hiệu tạo nhãn:**
  - 0,70% mẫu MAP nằm ngoài 20–200 mmHg;
  - trung bình 5,8 lần MAP nhảy trên 30 mmHg mỗi giờ (flush, lấy máu, zeroing);
  - **4,9% biến cố IOH có MAP thấp nhất ≤ 30 mmHg**, nhiều khả năng là artefact được tính thành biến cố.
- **Waveform ART:** median 97,6% nhịp hợp lệ, nhưng 180 ca có dưới 50% nhịp hợp lệ. Nhiều ca có kênh ART ghi nhiễu quanh 0 trong khi monitor vẫn báo MAP.
- **Tín hiệu phụ nhiễu hơn:**
  - CVP: 5,4% mẫu ngoài −5–40 mmHg;
  - BIS: 3,9% mẫu có SQI < 50;
  - HR đứng yên ≥ 60 s ở khoảng 6% thời gian (monitor giữ giá trị).
- **NIBP không dùng được như tín hiệu liên tục.** Ở 91% ca, NIBP thiếu hơn 20% thời gian mổ, vì máy thường ngừng đo khi đã có ART.

### 3. Quan hệ giữa các feature

- **Dư thừa lớn trong nhóm huyết áp.** MAP–DBP ρ = 0,89, MAP–SBP 0,82. SVR proxy đi cùng DBP (0,63). PP, dP/dt và SBP gần như cùng một trục (0,7–0,8). Mô hình chỉ cần một đại diện cho mỗi trục.
- **Khi MAP đang giảm, SVR proxy và dP/dt giảm theo** (ρ ≈ 0,6 giữa độ dốc MAP 5 phút và Δlog SVR / Δlog dP/dt). SV proxy thì không giảm cùng. Đây là dấu hiệu nhất quán với **giãn mạch là cơ chế chính** (khớp EDA chính).
- **SV proxy và SVR proxy ngược chiều** (ρ = −0,70): cùng một MAP có thể là "SV cao + SVR thấp" (giãn mạch) hoặc "SV thấp + SVR cao" (thiếu dịch). Phân rã SV × HR × SVR nhằm tách đúng hai trạng thái này.
- **Thay đổi MAP trong 5 phút tới:**
  - Giá trị MAP/SBP cao hiện tại liên quan với giảm sau đó (ρ ≈ −0,25), một phần là hồi quy về trung bình.
  - Ngoài huyết áp, feature liên quan mạnh nhất là **Ce remifentanil tăng trong 5 phút qua (ρ = −0,25)**.
  - Kế tiếp là PP và dP/dt hiện tại (−0,16 / −0,14), EtCO2 giảm dần (−0,13) và MAC tăng (−0,11).
- **Mức × xu hướng MAP** (ma trận nguy cơ):
  - MAP 65–70 có nguy cơ 12–33% bất kể xu hướng; MAP 60–65 mà chưa thành biến cố có 29–46%.
  - MAP 75–80 đang giảm 5–6 mmHg/phút có 6%, so với 1–2% khi ổn định.
  - Trên 90 mmHg, nguy cơ gần 0 dù đang giảm.
  - Xu hướng chỉ có giá trị ở vùng giữa, đúng với lập luận của mô hình ngoại suy MAP (LepMAP).
- **Đường nguy cơ có dạng chữ U** với Δ SVR, Δ SV và Δ dP/dt: cả giảm mạnh lẫn tăng mạnh đều tăng nguy cơ. Tăng mạnh thường là giai đoạn hồi phục sau một biến cố hoặc sau thuốc vận mạch, tức là "huyết động không ổn định". Dạng quan hệ này là phi tuyến, nên cần mô hình cây hoặc deep learning, không dùng hồi quy tuyến tính.

### 4. Feature ảnh hưởng nhất

- **MAP hiện tại và MAP thấp nhất gần đây là số 1, ở mọi cách đo:**
  - AUROC đơn biến 0,807 (y5);
  - khoảng 37% tổng SHAP (thêm 24% từ SBP/DBP);
  - riêng 20 feature MAP đạt AUROC 0,873 trên VALIDATION.
  - Điều này khớp với y văn: HPI tương quan mạnh với MAP (Frassanito 2024) và không hơn ngưỡng MAP (Mulder 2024).
- **Thêm mọi tín hiệu khác chỉ cải thiện rất ít:**
  - 173 feature đạt AUROC 0,881 so với 0,873 của riêng MAP; ΔAUROC +0,008 (95% CI +0,001 đến +0,015, 251 ca).
  - AP không khác (+0,001, CI −0,022 đến +0,029).
  - → **Giới hạn hiệu năng của UC04 không nằm ở số lượng tín hiệu.** Nó nằm ở chất lượng nhãn (artefact), mất cân bằng và chính sách cảnh báo.
- **Nhóm có giá trị cộng thêm lớn nhất là độ mê / thuốc mê:**
  - 14% SHAP ở y5, 19% ở y10, tăng theo horizon;
  - MAC hiện tại (cao → nguy cơ);
  - **Ce remifentanil tăng trong 5 phút** (tăng > 1 ng/mL: nguy cơ y5 7,6% so với 2,9% khi ổn định);
  - Ce propofol (thấp → nguy cơ, nhưng bị nhiễu bởi loại gây mê: ca khí mê có y5 3,8% so với 2,6% ở ca TIVA).
  - Đây là bằng chứng dữ liệu cho nhánh "giãn mạch do thuốc mê" của UC04, dù chưa phải quan hệ nhân quả.
- **Beat features từ waveform ART:**
  - 8% SHAP; SVR proxy hiện tại là feature đơn lẻ tốt nhất ngoài huyết áp (AUROC 0,72, SVR thấp → nguy cơ).
  - Giá trị chính của chúng **không nằm ở tăng AUROC** (+0,003). Nó nằm ở chỗ chúng là tín hiệu duy nhất cho phép giải thích cơ chế: SV, SVR, dP/dt, PPV.
- **HR** (độ dao động 10 phút, HR cao → nguy cơ) và **ASA** có đóng góp nhỏ.
- **SpO2, thở máy và CVP** gần như không đóng góp cho dự báo.
- **Ý nghĩa cho thiết kế:**
  - Nhánh dự báo nên giữ MAP làm lõi và ưu tiên sửa nhãn và chính sách cảnh báo.
  - Beat features và thuốc mê là đầu vào cho nhánh giải thích nguyên nhân.
  - Mọi cải thiện phải được so với baseline chỉ MAP bằng CI theo ca.
