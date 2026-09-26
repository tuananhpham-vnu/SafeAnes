# Đề xuất phương pháp UC04 — HemoDecomp

**Dự báo sớm tụt huyết áp, phân tách nguyên nhân theo sinh lý và ước lượng độ tin cậy**

Phiên bản nháp 23/09/2026. Tài liệu này mở rộng [UC04_RESEARCH_PLAN.md](../UC04_RESEARCH_PLAN.md) (mục 8–9) và dựa trên
[EDA toàn cohort VitalDB](../reports/EDA/REPORT.md). Mọi con số ở mục 2 và mục 4 được trích từ EDA hoặc đo trên dữ liệu đã tải. Các chỉ tiêu ở mục 6 kế thừa
mục tiêu đã khóa trong kế hoạch; tài liệu này không nới hay đổi các mục tiêu đó.

> Phạm vi: nghiên cứu hồi cứu trên VitalDB. Đầu ra "nguyên nhân" là **gợi ý kèm bằng chứng quan sát được**, không phải
> chẩn đoán hay khuyến nghị điều trị. Chỉ được trình bày như chẩn đoán sau khi đã kiểm chứng với chuẩn tham chiếu (mục 6).

---

## 0. Tóm tắt

UC04 yêu cầu ba đầu ra: (1) nguy cơ tụt huyết áp **trước khi xảy ra** đủ sớm để can thiệp; (2) **nguyên nhân chiếm ưu thế**
(thiếu dịch, giãn mạch do thuốc mê, giảm co bóp, mất máu); (3) **mức tin cậy** của dự báo. Pipeline E01–E08 hiện có mới
giải phần (1) bằng 7 tín hiệu số Solar8000, và chưa đạt đồng thời các mục tiêu chất lượng.

HemoDecomp đề xuất bốn nhánh dùng chung một bộ mã hóa chuỗi thời gian:

```
 ART waveform ─► beat features (SV/CO/SVR proxy, PPV, dP/dt, SQI) ─┐
 Solar8000 / BIS / MAC / Ce propofol-remi / vận mạch / thở máy ─────┼─► lưới 2 s, mask + age ─► Encoder (LightGBM | TCN/PatchTST)
 static (tuổi, ASA, loại mổ) ──────────────────────────────────────┘                                  │
      ┌────────────────────────────────────────┬───────────────────────────────────┬────────────────────┤
      ▼                                        ▼                                   ▼                    ▼
 A. Hazard rời rạc 1–15 phút          B. Dự báo quỹ đạo Δlog SV,        C. Gợi ý nguyên nhân     D. Độ tin cậy
    → P(IOH trong h), lead time          Δlog HR, Δlog SVR                  đa nhãn (weak           calibration + ensemble
                                        → phân rã ΔMAP dự kiến              supervision)            + conformal + cổng SQI
                                                                            + "không đủ bằng chứng"   → Cao / Trung bình / Thấp / Từ chối
```

Có ba điểm mới so với pipeline hiện tại:
1. **Phân rã sinh lý thay cho SHAP.** Ta dùng đồng nhất thức MAP ≈ SV × HR × SVR và dự báo sự thay đổi của **từng thành phần**,
   rồi gán phần sụt MAP dự kiến cho thành phần tương ứng. Cách giải thích này dựa trên sinh lý, không phải attribution thống kê.
   Hằng số chưa biết của proxy pulse-contour triệt tiêu khi lấy Δlog so với nền của chính ca.
2. **Nhãn nguyên nhân có kiểm chứng độc lập.** Ta tạo nhãn yếu từ proxy ART cộng ngữ cảnh thuốc, dịch và máu (Snorkel-style).
   Nhãn này được **đánh giá trên các ca có EV1000/Vigileo**, nơi SV/SVR/SVV được đo bằng thiết bị và **không nằm trong input**.
   Sau đó mới qua hội chẩn chuyên gia.
3. **Độ tin cậy nhiều tầng có trạng thái từ chối.** Tầng 1 là calibration xác suất (ECE). Tầng 2 là bất định mô hình (ensemble).
   Tầng 3 là conformal prediction có bảo đảm độ phủ. Tầng 4 là cổng chất lượng tín hiệu: khi dữ liệu không đủ thì từ chối
   thay vì đưa ra con số sai.

---

## 1. Bài toán và đầu ra

Tại mỗi thời điểm quyết định *t* (mỗi 30 s, như protocol hiện tại), chỉ dùng dữ liệu ≤ *t*:

| Đầu ra | Định nghĩa | Dạng |
|---|---|---|
| **A. Nguy cơ** | P(khởi phát IOH trong (t, t+h]), h = 1…15 phút; IOH = MAP < 65 mmHg ≥ 60 s (Protocol v1) | Đường cong nguy cơ đơn điệu theo h, kèm cảnh báo y5/y10 theo chính sách E08 |
| **B. Phân rã** | Dự báo Δlog SV, Δlog HR, Δlog SVR trong 5–10 phút tới, và tỉ phần đóng góp vào ΔlogMAP dự kiến | 3 số + tỉ phần |
| **C. Nguyên nhân** | Xác suất đa nhãn cho: *thiếu dịch/giảm tiền tải*, *giãn mạch* (có cờ "liên quan thuốc mê"), *giảm co bóp*, *mất máu*, *nhịp chậm*; thêm trạng thái *hỗn hợp* và *không đủ bằng chứng* | Danh sách xếp hạng + bằng chứng (tín hiệu, mốc thời gian) |
| **D. Tin cậy** | Xác suất đã calibration; khoảng/tập conformal; mức Cao/Trung bình/Thấp; hoặc *Từ chối* kèm lý do (mất ART, SQI thấp, thiếu lịch sử) | Nhãn + lý do |

*Nhịp chậm* được thêm vì cả hai nghiên cứu endotype lớn đều tìm thấy nó như một endotype riêng (mục 3), và EDA cũng thấy nó.
Nhịp chậm không thay thế bốn nguyên nhân mà UC04 yêu cầu.

Các nguyên nhân **không loại trừ lẫn nhau**, đúng như §9 của kế hoạch. Mất máu là một dạng giảm tiền tải, nhưng có thêm bằng
chứng mất máu (Hb, giai đoạn phẫu thuật).

---

## 2. Căn cứ từ EDA

<!-- eda-evidence:begin -->
Nguồn: [reports/EDA/REPORT.md](../reports/EDA/REPORT.md). Phạm vi outcome là 3.110 ca / 2.991 bệnh nhân ngoài global test.

| Phát hiện EDA | Số liệu | Quyết định thiết kế |
|---|---|---|
| IOH phổ biến, ngắn | 5.306 biến cố; 50,9% ca; 0,60/giờ; thời lượng median 186 s; prevalence y5 3,1%, y10 6,2% | Nhánh A giữ đánh giá luồng liên tục và alarm policy E08; dùng hazard để khai thác thêm horizon 1–15 phút |
| Beat features dùng được | 3.548/3.626 ca; median 97,6% beat hợp lệ; 180 ca < 50% | Beat features là kênh chính cho B/C; cổng D4 dùng tỉ lệ beat hợp lệ |
| Proxy ART bám thiết bị ở mức xu hướng | SV proxy vs EV1000 SV: r 0,61, đồng hướng 81% (508 ca); SVR: r 0,62, 84% (201 ca); PPV vs SVV: r 0,58 | Chỉ dùng Δlog tương đối; EV1000/Vigileo làm chuẩn bạc; học ánh xạ g(·) (§5.3) |
| Trước IOH: SVR↓ và dP/dt↓, SV trung vị không đổi | ΔlogSVR −0,067, Δlog dP/dt −0,070, ΔlogSV +0,004; SVR giảm ≥10% ở 41,7% biến cố so với 23,6% đối chứng | Giãn mạch là nhóm lớn nhất; dP/dt phải đi cùng SV khi gán "giảm co bóp" để khỏi nhầm với giãn mạch |
| Mức nền quan trọng, không chỉ thay đổi | PPV nền khoảng 13,5% trước IOH so với 11% ở đối chứng; SV/SVR EV1000 nền thấp hơn | Input gồm cả **mức** và **Δ**; LF tiền tải dùng PPV mức cuối ≥ 13% |
| Phân rã | SVR chi phối 49% (đối chứng 32%); SV + PPV cao 13%; SV + dP/dt↓ 2%; HR↓ 12%; thay đổi nhỏ 23% | "Không đủ bằng chứng" là đầu ra hợp lệ, dự kiến chiếm khoảng 1/4 biến cố; giảm co bóp hiếm, cần precision cao |
| Cụm k-means gần endotype y văn | cụm "giãn mạch + CO cao" (586), "tiền tải" (402), "SVR↓ + dP/dt↓" (1.646), "không đổi" (1.787), **artefact PPV** (178) | Xác nhận cấu trúc nhãn; bắt buộc có SQI cho PPV |
| Ngữ cảnh thuốc mê ít phân biệt | Tăng Ce/MAC trong 15 phút trước: 27% trước IOH so với 24% đối chứng | Cờ "do thuốc mê" chỉ là bằng chứng phụ; không có thì báo "giãn mạch" chung |
| Protocol v1 bỏ qua giai đoạn khởi mê | chỉ 0,6% biến cố trong 30 phút đầu sau anestart; nhãn chỉ trong opstart→opend | Đề xuất **protocol v2** mở rộng cửa sổ nhãn về anestart (đăng ký riêng, E14) |
| Nhãn nghi artefact | 4,9% biến cố có MAP thấp nhất ≤ 30 mmHg | Audit nhãn bằng beat SQI trước khi huấn luyện B/C; báo cáo độ nhạy kết quả khi loại các biến cố này |
| Mất máu gần như không có timeline | Hb trong mổ có mốc thời gian ở 97 ca; EBL/RBC chỉ tổng cuối ca (ρ với biến cố/giờ 0,21/0,23) | Nhãn "mất máu" phần lớn sẽ là "không đủ bằng chứng" trên VitalDB; cần dữ liệu bệnh viện có timeline máu/dịch |
| Thiết bị CO hiếm và lệch | EV1000 590 ca (16%), EV1000 SVR 239, Vigileo 321; vận mạch bơm < 4% ca | Chuẩn bạc nhỏ và thiên về mổ lớn; báo cáo theo nhóm; thí nghiệm đáp ứng vận mạch chỉ mang tính mô tả |
| NIBP không thay ART | bias −3,4 mmHg, LoA −49 đến +43; 55,6% thời điểm ART < 65 có NIBP ≥ 65 | UC04 giới hạn ở ca có đường động mạch |
| Nguy cơ theo bệnh nhân | ASA 1 → 4: 0,38 → 1,66 biến cố/giờ; > 75 tuổi 0,79; cấp cứu 0,88 | Static features và Mondrian conformal theo nhóm ASA/tuổi |
<!-- eda-evidence:end -->

---

## 3. Tài liệu liên quan và khoảng trống

| Nguồn | Nội dung liên quan | Hệ quả cho HemoDecomp |
|---|---|---|
| Hatib et al., *Anesthesiology* 2018 — HPI ([PubMed](https://pubmed.ncbi.nlm.nih.gov/29894315/)) | Hàng nghìn đặc trưng waveform ART; tài liệu HPI mô tả các nhóm đặc trưng tiền tải (PPV/SVV), co bóp (dP/dt), hậu tải (SVR, Eadyn) | Dùng cùng các họ đặc trưng beat, nhưng giữ **tách theo thành phần** để giải thích được |
| Davies et al., *Anesth Analg* 2020 — kiểm định HPI | Độ nhạy/đặc hiệu ~86/82/81% ở 5/10/15 phút trong thiết lập của họ | Cần đường cong nguy cơ đến 15 phút; không lấy số này làm mục tiêu vì giao thức khác |
| Mulder et al., *Anesthesiology* 2024; Yang et al., *BJA* 2025 (đã có ở kế hoạch §2) | HPI ≈ ngưỡng MAP; selection bias làm PPV giảm rất mạnh | Giữ baseline MAP, đánh giá luồng liên tục bằng evaluator E08 |
| Kouz et al., *BJA* 2023; 130: 253–261 — endotype IOH ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0007091222005578)) | 5 endotype từ SVI, HR, SVRI, CI trong mổ bụng lớn: giãn mạch nặng (CI cao), thiếu dịch, suy giảm cơ tim, nhịp chậm, giãn mạch nhẹ | Nhóm nguyên nhân của UC04 **có cơ sở sinh lý đo được**; bổ sung nhãn nhịp chậm |
| Jian et al., *BJA* 2025; 134: 308–316 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/39788817/)) | Autoencoder + GMM trên SVI/HR/SVRI/SVV; 4 endotype (giãn mạch, thiếu dịch, suy giảm cơ tim, nhịp chậm), lặp lại trên 2 bộ kiểm định ngoài | Endotype ổn định qua cohort, hỗ trợ việc dùng chúng làm **chuẩn tham chiếu bạc** |
| Zhu et al., *Perioper Med* 2025; 14: 109 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12523014/)) | **VitalDB**, EV1000; LSTM-autoencoder + k-means; 5 endotype; endotype có thể chuyển trong cùng một episode; thiếu dịch trội đầu ca, nhịp chậm tăng cuối ca | Cùng dữ liệu nguồn nên dùng để đối chiếu. Hạn chế của họ: chỉ phân cụm **sau** khi IOH xảy ra, cần thiết bị CO, 184 bệnh nhân, không dự báo |
| Sun et al., *Computers in Cardiology* 2005 ([PhysioNet](https://physionet.org/physiotools/cardiac-output)) | So sánh 11 bộ ước lượng CO từ ART; Liljestrand–Zander thuộc nhóm tốt khi có hiệu chỉnh | Dùng proxy LZ ở dạng **tương đối** (Δlog), không dùng giá trị tuyệt đối |
| Ratner et al., *PVLDB* 2017 — Snorkel ([PDF](https://www.vldb.org/pvldb/vol11/p269-ratner.pdf)) | Labelling functions + label model sinh nhãn xác suất mà không cần nhãn tay | Khung cho nhánh C khi chưa có nhãn chuyên gia |
| Angelopoulos & Bates 2021 — *A Gentle Introduction to Conformal Prediction* (arXiv:2107.07511) | Split/Mondrian conformal: bảo đảm độ phủ biên theo nhóm | Khung cho tầng 3 của nhánh D |

**Khoảng trống mà HemoDecomp nhắm tới.** Các nghiên cứu endotype mô tả IOH **sau khi đã xảy ra** và cần máy đo CO.
Các mô hình dự báo (HPI, TCN/Transformer) không nói rõ **vì sao**. Chưa có nghiên cứu nào trong số đã kiểm tra kết hợp được bốn điều:
- dự báo trước,
- gợi ý cơ chế chỉ từ ART,
- kiểm chứng cơ chế với CO đo bằng thiết bị,
- đánh giá cảnh báo trên luồng liên tục.

---

## 4. Dữ liệu và lựa chọn tín hiệu

### 4.1 Bộ dữ liệu

#### 4.1.1 VitalDB

**Nguồn.** VitalDB (Lee et al., *Sci Data* 2022; tài liệu tham khảo [1]) gồm 6.388 ca phẫu thuật không tim, thu thập tại
Bệnh viện Đại học Quốc gia Seoul từ 08/2016 đến 06/2017. Các nhóm phẫu thuật là tổng quát, lồng ngực, tiết niệu và phụ khoa,
chiếm 91% trong 7.051 ca đủ điều kiện của nghiên cứu gốc.

**Quy mô.** Bộ dữ liệu có 486.451 track (waveform và số) của 196 tham số theo dõi trong mổ, 73 tham số lâm sàng quanh mổ và
34 loại xét nghiệm có mốc thời gian. Waveform lấy mẫu 62,5–500 Hz; track số có chu kỳ 1–7 s.

**Truy cập.** Dữ liệu mở (open access) trên [PhysioNet](https://physionet.org/content/vitaldb/1.0.0/) và qua API
`api.vitaldb.net`, giấy phép CC BY 4.0. Dự án tải qua API bằng `safeanes.data.fetch_csv`. Mỗi file raw lưu kèm URL, thời điểm
tải và SHA-256.

**Thiết bị** (liệt kê trên trang PhysioNet; thiết bị dự án dùng in đậm):
- **Monitor Solar 8000M** (GE): track số ART/NIBP/HR/SpO2/EtCO2/CVP/BT.
- **SNUADC**: bộ thu analog 500 Hz cho ART, ECG, PPG, CVP.
- **Máy mê Primus** (Dräger): khí mê, MAC, thông số thở máy, capnogram.
- **BIS Vista**.
- **Bơm TCI Orchestra**: propofol, remifentanil, thuốc vận mạch.
- **Monitor cung lượng tim EV1000 và Vigileo/FloTrac.**
- Ngoài ra: Vigilance II, CardioQ-ODM+, FMS2000, INVOS.

**Cohort UC04** (`safeanes.data.cohort_manifest`, [EDA §1](../reports/EDA/REPORT.md)):

| Bước | Số ca |
|---|---|
| Tất cả ca VitalDB | 6.388 |
| Loại: không xác định là người lớn / không gây mê toàn thể / thời gian mổ không đủ / không có ART MAP | −57 / −342 / −133 / −2.230 |
| **Đủ điều kiện (eligible)** | **3.626 ca / 3.493 bệnh nhân** |
| trong đó global test (khóa, không dùng cho EDA outcome) | 516 ca / 502 bệnh nhân |
| phạm vi phát triển (fit / calibration / validation) | 3.110 ca / 2.991 bệnh nhân |

**Đặc điểm cohort eligible:**
- tuổi median 60 [51–69], nữ 44,2%, BMI 22,9 [20,7–25,2];
- ASA ≥ 3: 15,5%; mổ cấp cứu: 13,0%; tăng huyết áp trước mổ: 33,2%;
- thời gian mổ 2,5 giờ [1,6–3,8]; EBL 200 mL [100–400];
- khoa: ngoại tổng quát 1.971 ca, lồng ngực 923, phụ khoa 148, tiết niệu 68 (số ca phạm vi phát triển).

**Chia dữ liệu.** Chia theo `subjectid` bằng băm ổn định (`subject_bucket`), nên mọi lần mổ của một bệnh nhân nằm cùng một phần.
Nhóm đánh giá lấy theo E07 (`reports/E07/cohort_manifest.csv`). Global test chỉ mở một lần sau khi đã khóa mô hình (mục 6.1).

**Nhãn.** IOH theo Protocol v1 là MAP ART < 65 mmHg liên tục ≥ 60 s, chỉ xét trong khoảng opstart → opend
([PROTOCOL.md](PROTOCOL.md)). Trong phạm vi phát triển có 5.306 biến cố. Hai điểm cần xử lý:
- Protocol v1 bỏ qua giai đoạn khởi mê; protocol v2 sẽ mở rộng về anestart (E14).
- 4,9% biến cố có MAP ≤ 30 mmHg, nghi là artefact.

**Dữ liệu dự án đã lấy về** (bố cục trong [DATA_CARD.md](DATA_CARD.md)):

| Thành phần | Nội dung | Vị trí |
|---|---|---|
| Metadata | `cases`, `trks`, cohort manifest, labs (928.448 kết quả có mốc thời gian) | `data/vitaldb_full/meta` |
| Track số | 7 track lõi và 116.123/116.124 file track UC04 của 3.626 ca eligible. Một file RFTN20_CP bị API trả 403 | `data/vitaldb_full/raw` |
| Beat features | 14 kênh trên lưới 2 s từ SNUADC/ART của 3.548 ca; raw waveform không lưu | `data/beats_full` |
| Dẫn xuất | Chuỗi 2 s của 7 track lõi; window 30 s, nhãn y5/y10 | `data/sequences_full`, `data/vitaldb_full/cases` |

**Giới hạn của VitalDB cho UC04:**
- Chỉ một trung tâm, dữ liệu 2016–2017.
- Bolus vận mạch, dịch, máu và EBL chỉ có tổng cuối ca.
- Hb trong mổ có mốc thời gian chỉ ở 97 ca phát triển.
- CO đo bằng thiết bị chỉ có ở 16% ca và thiên về mổ lớn.
- Không có nhãn cơ chế.

Kiểm định ngoài (MOVER hoặc bệnh viện) nằm ngoài phạm vi mục này; xem kế hoạch §3.

### 4.2 Lựa chọn tín hiệu

Có bốn tiêu chí chọn tín hiệu:
1. **Liên quan sinh lý** tới MAP ≈ SV × HR × SVR, hoặc tới bốn nguyên nhân của UC04.
2. **Độ phủ** trong cohort eligible. Tín hiệu có dưới khoảng 50% ca chỉ dùng làm kênh tùy chọn có mask, hoặc làm chuẩn tham chiếu.
3. **Dùng được theo thời gian thực**: chỉ dữ liệu ≤ t, có mốc thời gian.
4. **Kiểm soát được chất lượng**: có tiêu chí SQI.

Độ phủ trong bảng dưới là số ca eligible (trên 3.626) có track. Tần số lấy mẫu được đo trên dữ liệu đã tải.

#### 4.2.1 Huyết áp động mạch xâm lấn (ART)

| Track | Tần số | Độ phủ | Vai trò |
|---|---|---|---|
| Solar8000/ART_MBP, ART_SBP, ART_DBP | 2 s | 3.626 (100%, theo tiêu chí chọn) | **Nguồn nhãn IOH** và input lõi |
| SNUADC/ART (waveform) | 500 Hz | 3.548 (97,8%) | **Beat features**: SBP/DBP/MAP/PP theo nhịp, HR, dP/dt max, diện tích tâm thu, proxy SV/CO/SVR (Liljestrand–Zander), PPV, SPV, tỉ lệ beat hợp lệ |
| Solar8000/FEM_*, SNUADC/FEM | 2 s / 500 Hz | 111 / 96 | Không dùng (quá ít) |

ART là tín hiệu trung tâm vì cùng lúc cho nhãn, cho dự báo và cho cả ba thành phần của phân rã sinh lý. Chất lượng:
- Median 97,6% beat hợp lệ theo ca; 180 ca có dưới 50%. Nhiều ca trong số này có kênh SNUADC là nhiễu quanh 0 dù Solar8000 vẫn ghi MAP.
- Artefact flush và lấy máu được loại ở mức beat, theo các giới hạn DBP/PP/chu kỳ/biên độ trong `safeanes.waveform`.
- Proxy bám EV1000 ở mức xu hướng: SV đồng hướng 81%, SVR 84%.
- NIBP không thay được ART: LoA từ −49 đến +43 mmHg, và 55,6% thời điểm ART < 65 có NIBP ≥ 65. Vì vậy UC04 **giới hạn ở ca có đường động mạch**.

#### 4.2.2 ECG

| Track | Tần số | Độ phủ | Vai trò |
|---|---|---|---|
| Solar8000/HR | 2 s | 3.626 | Input lõi; thành phần HR của phân rã; LF "nhịp chậm" |
| SNUADC/ECG_II | 500 Hz | 3.605 (99,4%) | **Chưa dùng**. Ứng viên giai đoạn sau: R-peak để tính HRV và rối loạn nhịp, và pulse arrival time (R → chân sóng ART/PPG) như chỉ dấu trương lực mạch |
| SNUADC/ECG_V5 | 500 Hz | 1.308 | Không dùng (độ phủ thấp) |
| Solar8000/ST_II | 2 s | 3.490 | Ứng viên phụ cho LF "giảm co bóp" (thiếu máu cơ tim); chưa dùng |

HR từ monitor đã đủ cho phân rã SV × HR × SVR. Beat features cũng tính HR độc lập từ ART. ECG waveform chỉ được thêm nếu
ablation cho thấy lợi ích (mục 6.3), vì xử lý 500 Hz cho 3.600 ca tốn chi phí tương đương nhánh ART.

#### 4.2.3 PPG (photoplethysmography)

| Track | Tần số | Độ phủ | Vai trò |
|---|---|---|---|
| Solar8000/PLETH_SPO2 | 2 s | 3.626 | Input lõi (SpO2) |
| Solar8000/PLETH_HR | 2 s | 3.626 | Kiểm tra chéo HR; phát hiện mất tín hiệu |
| SNUADC/PLETH | 500 Hz | 3.418 (94,3%) | **Chưa dùng**. Ứng viên: biến thiên biên độ theo hô hấp (tương tự PVI) làm chỉ dấu tiền tải thay thế khi ART nhiễu; perfusion index; pulse arrival time cùng ECG |

PPG không dùng làm nguồn nhãn. Biên độ PPG phụ thuộc trương lực mạch ngoại vi và vị trí đầu dò, nên chỉ đưa vào ở dạng
Δ tương đối, cùng SQI riêng.

#### 4.2.4 Capnography

| Track | Tần số | Độ phủ | Vai trò |
|---|---|---|---|
| Solar8000/ETCO2, RR_CO2 | 2 s | 3.618 / 3.612 | Input lõi |
| Primus/ETCO2, INCO2 | ~7 s | 3.608 | Dự phòng khi thiếu Solar8000 |
| Primus/CO2 (capnogram) | 62,5 Hz | 3.610 | **Chưa dùng** |

Khi thông khí không đổi, EtCO2 giảm theo lưu lượng máu phổi. Vì vậy ΔEtCO2 kèm TV/RR ổn định là bằng chứng phụ cho CO
giảm, dùng trong LF thiếu dịch và giảm co bóp. Capnogram 62,5 Hz chưa cần cho mục tiêu này.

#### 4.2.5 Các sinh hiệu và dữ liệu khác

| Nhóm | Track (tần số) | Độ phủ | Vai trò |
|---|---|---|---|
| Độ mê | BIS/BIS, SQI, SR (1 s); Primus/MAC, EXP_SEVO, EXP_DES (~7 s) | BIS 3.357; MAC 3.608; sevo 2.270; des 1.161 | Ngữ cảnh "giãn mạch do thuốc mê" |
| Thuốc TCI | Orchestra PPF20_CE/CP/RATE (~1 s); RFTN20_CE/CP/RATE | propofol 1.944 (TIVA); remifentanil 3.008 | Ngữ cảnh thuốc mê; Ce là nồng độ ước lượng theo mô hình dược động học, không phải đo |
| Vận mạch / giãn mạch qua bơm | Orchestra PHEN, NEPI, EPI, DOPA, NTG, PGE1… RATE/VOL | PHEN 115; NEPI 79; các thuốc khác < 90 | Thí nghiệm tự nhiên (đáp ứng SVR); ngữ cảnh; **không** làm nhãn |
| Thở máy | Primus PEEP, PIP, Pplat, TV, MV, compliance (~7 s) | ≈ 3.590–3.600 | Tiền tải (PEEP, áp lực đường thở ảnh hưởng hồi lưu tĩnh mạch); điều kiện diễn giải PPV |
| Áp lực tĩnh mạch trung tâm | Solar8000/CVP (2 s); SNUADC/CVP (500 Hz) | 1.563 / 1.542 | Kênh tùy chọn có mask; LF giảm co bóp (CVP ↑) |
| NIBP | Solar8000/NIBP_* (theo chu kỳ đo) | 3.028 | Chỉ để mô tả; không làm nhãn |
| Nhiệt độ | Solar8000/BT (2 s) | 3.523 | Ngữ cảnh (hạ thân nhiệt, giãn mạch khi ủ ấm) |
| Cung lượng tim đo bằng thiết bị | EV1000 SV/SVI/SVV/CO/CI (2 s), SVR/SVRI; Vigileo SV/SVV/CO | EV1000 590 (SVR 239); Vigileo 321 | **Chỉ làm chuẩn tham chiếu bạc** cho nhánh B/C; không vào input |
| Xét nghiệm | labs API: Hb, Hct, lactate, khí máu… có mốc thời gian | Hb trong mổ ở 97 ca phát triển | LF mất máu (Hb giảm); rất thưa |
| Lâm sàng tĩnh | tuổi, BMI, ASA, cấp cứu, THA, khoa, loại mổ, đường mổ, Hb trước mổ | toàn bộ | Input static; nhóm Mondrian conformal |
| Tổng cuối ca | EBL, dịch tinh thể/keo, RBC, ephedrine, phenylephrine | toàn bộ | **Không dùng làm input theo thời gian** (rò rỉ tương lai); chỉ để mô tả |

Mức thiếu, nhiễu và tầm quan trọng đo được của từng nhóm tín hiệu (AUROC đơn biến, SHAP, ablation): [INSIGHTS.md](../reports/EDA/INSIGHTS.md).

Tóm tắt lựa chọn cho HemoDecomp:
- **ART** (số + beat) là bắt buộc.
- **HR, SpO2, EtCO2/RR, độ mê, TCI, thở máy, static** là input thường quy.
- **CVP** là tùy chọn có mask.
- **EV1000/Vigileo** chỉ dùng làm chuẩn tham chiếu.
- **ECG, PPG waveform và capnogram** hoãn đến khi ablation chứng minh lợi ích.

---

## 5. Phương pháp

### 5.1 Dữ liệu vào

Tín hiệu và lý do chọn: mục 4.2. Lưới 2 s, causal, cùng lattice với `sequences_full`. Mỗi kênh có giá trị, mask thiếu và *age* (thời gian từ quan sát cuối).
Cấu trúc này mở rộng `FastWindowDataset` ([e08_sequence.py](../src/safeanes/e08_sequence.py)).

| Nhóm | Kênh | Nguồn / module |
|---|---|---|
| Lõi (7) | ART MAP/SBP/DBP, HR, SpO2, EtCO2, RR | `config.TRACKS`, `sequences_full` |
| Beat (14) | SBP/DBP/MAP/PP theo beat, HR, dP/dt max, diện tích tâm thu, SV_LZ, CO_LZ, SVR_LZ, PPV, SPV, tỉ lệ beat hợp lệ, age | `safeanes.waveform`, `data/beats_full` |
| Độ mê / thuốc | BIS, SQI, MAC, EtSevo/Des, Ce/Cp/rate propofol và remifentanil; rate phenylephrine, norepinephrine, epinephrine, NTG… | `config.UC04_TRACKS` |
| Tiền tải / thở máy | PEEP, PIP, Pplat, TV, compliance, CVP (khi có) | idem |
| Static | tuổi, BMI, ASA, cấp cứu, THA, khoa, loại mổ, đường mổ, Hb trước mổ | cohort manifest |
| **Không vào input** | EV1000/Vigileo SV/CO/SVV/SVR | Chỉ dùng làm **chuẩn tham chiếu** (mục 6.2) |

Các đặc trưng Δ dùng thang log so với nền trượt của chính ca: Δlog x = log(median x trong 2 phút gần nhất) − log(median x
trong 10–15 phút trước). Nhờ vậy hệ số chưa biết của proxy LZ triệt tiêu. Beat chỉ được dùng khi *avail = kết thúc beat + 1 s ≤ t*,
đã có kiểm thử chống đọc tương lai ([tests/test_waveform.py](../tests/test_waveform.py)).

### 5.2 Nhánh A — dự báo biến cố bằng hazard rời rạc

Chia tương lai thành K = 15 khoảng 1 phút. Mô hình xuất hazard h_k(t) = P(khởi phát trong phút k | chưa khởi phát trước đó, dữ liệu ≤ t).
Từ đó:

- F(h) = 1 − ∏_{k≤h} (1 − h_k): đơn điệu theo h **theo cấu trúc**. Cách này tổng quát hóa ordered head 2 logit hiện có
  ([models.py](../src/safeanes/models.py)).
- Loss là negative log-likelihood có censoring. Các phút sau khi hết theo dõi hoặc mất nhãn bị mask, giống `masked_bce` hiện tại.
- Lead time kỳ vọng = Σ_k k·P(khởi phát ở phút k). Đại lượng này giúp phân biệt nguy cơ "sắp xảy ra" với "trong 10–15 phút".
- Backbone được so sánh công bằng trong khung E08: (i) LightGBM với K đầu ra (một model/horizon, sau đó đơn điệu hóa bằng
  cumulative max), vốn đang là họ mạnh nhất ở E08; (ii) TCN/PatchTST nhỏ chạy trên T4, dùng code E08-DL.
- Cảnh báo dùng lại evaluator và chính sách của E08 (2 lần vượt ngưỡng, cooldown 5 phút, chọn ngưỡng trên CALIBRATION).

Giả thuyết H-A: thêm beat features và ngữ cảnh thuốc giúp tăng **PPV và độ nhạy biến cố ở cùng số cảnh báo giả/giờ** so với
E08 (7 track). Nếu không đạt, báo cáo không đạt.

### 5.3 Nhánh B — phân rã sinh lý ΔMAP

Với proxy Liljestrand–Zander, SV ∝ PP/(SBP+DBP) và SVR ∝ MAP/(SV·HR), nên theo định nghĩa:

  **Δlog MAP = Δlog SV + Δlog HR + Δlog SVR** (với CVP ≈ 0; khi có CVP thì dùng MAP − CVP).

1. **Multi-task head.** Cùng encoder, dự báo ŷ_c(t, h) = Δlog c trong h ∈ {5, 10} phút cho c ∈ {SV, HR, SVR}. Loss là Huber, có mask
   khi thiếu beat hợp lệ.
2. **Phân rã.** Tổng Σ_c ŷ_c = ΔlogMAP dự kiến; phần đóng góp s_c = ŷ_c / Σ ŷ_c chỉ tính khi tổng âm (MAP dự kiến giảm).
   "Thành phần chi phối" là c có ŷ_c âm nhất.
3. **Hiệu chỉnh proxy.** Trên các ca có EV1000/Vigileo **thuộc tập fit**, học một ánh xạ nhỏ g(beat features) → Δlog SV_thiết bị
   (ridge hoặc GBM). Dùng ánh xạ này thay Δlog SV_LZ nếu nó cải thiện concordance trên tập calibration. Ánh xạ SVR/CO suy ra tương tự.
   Mô hình không bao giờ dùng giá trị thiết bị làm input lúc chạy.
4. **Nhất quán với nhánh A.** Thêm regularizer mềm: P(IOH) từ nhánh A và ΔlogMAP dự kiến từ nhánh B phải đồng biến. Việc này
   được kiểm tra bằng ablation; nếu làm giảm chất lượng nhánh A thì bỏ.

### 5.4 Nhánh C — gợi ý nguyên nhân bằng weak supervision

Chưa có nhãn chuyên gia (kế hoạch §9), nên nhãn được tạo từ các **labelling function (LF)**. Mỗi LF chỉ dùng thông tin
**trước khởi phát** và trả về {ủng hộ, phản đối, bỏ phiếu trắng} cho từng nhãn:

| Nhãn | LF (ví dụ — ngưỡng khóa trên tập fit, ghi phiên bản) |
|---|---|
| Thiếu dịch / giảm tiền tải | Δlog SV ≤ −0,1 **và** (PPV ≥ 13% hoặc ΔPPV ≥ +3) · HR tăng bù · giảm PEEP/đổi tư thế (nếu có) · Δlog SVR ≥ 0 |
| Giãn mạch | Δlog SVR ≤ −0,1 là thành phần chi phối · SV giữ (≥ −5%) · **cờ "do thuốc mê"** khi ΔCe propofol > 0,1 µg/mL, ΔCe remifentanil > 0,3 ng/mL, ΔMAC > 0,1 hoặc BIS giảm > 5 trong 10 phút · giai đoạn sau khởi mê |
| Giảm co bóp | Δlog SV ≤ −0,1 **và** Δlog dP/dt ≤ −0,1 · PPV không tăng · CVP tăng (khi có) · HR không tăng bù |
| Mất máu | LF thiếu dịch ủng hộ **và** có bằng chứng mất máu trước t: Hb giảm ≥ 1,5 g/dL so với trước mổ (xét nghiệm có mốc thời gian) · SV giảm tiến triển > 15 phút · loại mổ có nguy cơ mất máu cao. EBL/RBC chỉ có ở mức ca, **không** dùng làm LF theo thời gian |
| Nhịp chậm | Δlog HR là thành phần chi phối · HR < 50 hoặc giảm > 15% |
| Không đủ bằng chứng | beat hợp lệ < 50% trong 10 phút · thiếu nền · mọi LF bỏ phiếu trắng |

- **Label model.** Mỗi nhãn là một bài toán nhị phân. Dùng Dawid–Skene / Snorkel để ước lượng độ chính xác và tương quan của
  các LF, rồi sinh nhãn mềm. LF được viết thành code có test (như `safeanes.eda._pattern` hiện tại, nhưng tách từng LF và đa nhãn).
- **Cause head.** Cùng encoder, mỗi nhãn một sigmoid. Huấn luyện trên nhãn mềm với noise-aware loss. Head chạy trên **mọi thời điểm
  có nguy cơ cao**, không chỉ sau khi IOH xảy ra. Nhờ vậy đầu ra đến kèm cảnh báo, trước biến cố.
- **Đầu ra cho người dùng.** Tối đa 2 nguyên nhân có p ≥ τ, mỗi nguyên nhân kèm 2–3 bằng chứng cụ thể (ví dụ "PPV 18% ↑ từ 9%
  trong 12 phút"; "Ce propofol 3,2 → 4,0 µg/mL"). Nếu không nhãn nào đạt τ, hoặc LF bất đồng mạnh (entropy cao), thì báo
  "hỗn hợp" hoặc "không đủ bằng chứng".

**Rủi ro vòng tròn.** Cause head học từ LF dùng chính các đặc trưng đầu vào, nên khớp tốt với LF không chứng minh được gì.
Vì vậy nhánh C **chỉ được đánh giá** bằng chuẩn độc lập với LF (mục 6.2). Giá trị thực của cause head so với việc dùng thẳng
LF nằm ở ba chỗ: nó dự báo được **trước** khi Δ đầy đủ hình thành, nó xử lý được dữ liệu thiếu, và nó được kiểm chứng.

### 5.5 Nhánh D — độ tin cậy

| Tầng | Kỹ thuật | Đầu ra / kiểm định |
|---|---|---|
| D1 Calibration | Platt/isotonic cho từng horizon trên CALIBRATION (đã có `fit_calibration`, `calibrate_member`); giữ đơn điệu F(h) | ECE ≤ 0,05 (mục tiêu kế hoạch), reliability plot, Brier |
| D2 Bất định mô hình | Ensemble 5 seed (DL) hoặc bagging LightGBM; độ lệch chuẩn của F(h) giữa các thành viên | Tương quan giữa độ lệch chuẩn và sai số; báo cáo riêng, không gọi là xác suất đúng |
| D3 Conformal | Split conformal Mondrian theo nhóm (tuổi, ASA, loại mổ, có/không beat) trên CALIBRATION → tập dự đoán {IOH}, {không}, {IOH, không} với 1−α = 0,9; tương tự tập nguyên nhân | Độ phủ thực tế theo nhóm trên VALIDATION; tỉ lệ tập mơ hồ (kích thước 2) |
| D4 Cổng chất lượng | Từ chối khi: MAP age > 30 s, beat hợp lệ < 50%, SQI flush/damping, lịch sử < 10 phút, điểm OOD (Mahalanobis trên embedding) > p99 của tập fit | Coverage dự báo ≥ 90% thời gian đủ điều kiện (mục tiêu kế hoạch); báo tỉ lệ từ chối theo lý do |

**Mức tin cậy hiển thị:**
- *Cao*: qua D4, tập conformal có 1 phần tử, và độ lệch chuẩn D2 < p50.
- *Trung bình*: qua D4 nhưng có một điều kiện còn lại không đạt.
- *Thấp*: tập conformal mơ hồ.
- *Từ chối*: không qua D4, kèm lý do.

Với nguyên nhân, dùng thêm entropy của label model và **độ phủ bằng chứng**, tức tỉ lệ LF không bỏ phiếu trắng.

---

## 6. Kiểm định

### 6.1 Nhánh A và D
- Chia dữ liệu theo vai trò E07/E08: fit / calibration / validation ngoài `unseen_test`. Global test **chỉ mở một lần** sau khi
  đã khóa model, calibrator, ngưỡng và chính sách.
- Metric và mục tiêu theo **kế hoạch §8**: AUROC; AP; ECE; độ nhạy biến cố; PPV theo episode; cảnh báo giả/giờ; lead time;
  coverage. Bootstrap CI theo `subjectid`. So cặp với baseline `map_logistic` và mô hình chọn ở E08.
- Metric riêng cho hazard: time-dependent AUROC theo h, integrated Brier score, calibration theo từng horizon.

### 6.2 Nhánh B và C — chuẩn tham chiếu
1. **Chuẩn bạc từ thiết bị.** Dùng các ca có EV1000/Vigileo, chia ca theo vai trò như trên. Tham chiếu là endotype định nghĩa
   **chỉ** từ SV/SVI, SVR/SVRI, SVV và HR của thiết bị, theo tiêu chí kiểu Kouz 2023 / Jian 2025. Tiêu chí được khóa trước khi
   xem kết quả mô hình. Metric: macro-F1, precision/recall từng nhãn, calibration đa nhãn, concordance xu hướng của Δlog SV/SVR
   dự báo so với thiết bị (vùng loại 10%).
2. **Đáp ứng điều trị (thí nghiệm tự nhiên).** Ở ca bắt đầu truyền phenylephrine/norepinephrine qua Orchestra, SVR proxy phải
   tăng. Đây là phép kiểm tra tính hợp lệ của proxy. Cỡ mẫu nhỏ nên kết quả chỉ mang tính mô tả.
3. **Hội chẩn chuyên gia.** Pilot 100–200 episode theo quy trình gán nhãn §9 đã có: hai người gán độc lập, không thấy dự báo.
   Đo agreement giữa người với người trước, sau đó mới so với mô hình.
4. **Không dùng** việc bác sĩ đã truyền dịch hay dùng vận mạch làm nhãn nguyên nhân (kế hoạch §9).

### 6.3 Ablation (cùng tập bệnh nhân, cùng evaluator)

Các cấu hình được so sánh:
- 7 track (E08);
- + beat features;
- + ngữ cảnh thuốc/độ mê;
- + thở máy;
- hazard so với 2 head;
- có/không multi-task B;
- có/không regularizer nhất quán;
- LF trực tiếp so với cause head;
- proxy LZ so với proxy đã hiệu chỉnh g(·).

---

## 7. Lộ trình đề xuất

| Đợt | Nội dung | Điều kiện qua mốc |
|---|---|---|
| E09 (1–2 tuần) | Đưa beat + UC04 tracks vào window features (LightGBM) và sequence (DL); chạy lại so sánh E08 trên full | Bảng so sánh có CI; quyết định giữ nhóm kênh nào |
| E10 (2 tuần) | Hazard 15 phút + calibration + conformal + cổng D4 | ECE, coverage conformal, tỉ lệ từ chối trên VALIDATION |
| E11 (2–3 tuần) | Nhánh B: multi-task Δlog; ánh xạ g(·) trên ca EV1000 | Concordance Δlog SV/SVR dự báo so với thiết bị |
| E12 (3 tuần) | Nhánh C: thư viện LF + label model + cause head; chuẩn bạc EV1000; soạn rubric chuyên gia | Macro-F1 trên chuẩn bạc; báo cáo mọi nhãn không đạt |
| E14 (song song) | Protocol v2: nhãn từ anestart (khởi mê); audit nhãn MAP ≤ 30 bằng SQI; đăng ký trước khi chạy | Tài liệu protocol v2; bảng so sánh số biến cố v1 và v2 |
| E13 | Khóa toàn bộ; mở global test một lần; báo cáo; kế hoạch kiểm định ngoài (MOVER) và pilot hội chẩn | Kế hoạch §8/§10 |

Sản phẩm bàn giao:
- `safeanes.hazard`, `safeanes.decomposition`, `safeanes.causes` (LF + label model), `safeanes.confidence`;
- notebook phát lại ca với 4 bảng (nguy cơ, phân rã, nguyên nhân, tin cậy);
- model card cập nhật.

---

## 8. Rủi ro và giới hạn

- **Thiếu mốc thời gian can thiệp.** Bolus phenylephrine/ephedrine, dịch, máu và EBL chỉ có tổng cuối ca. Vì vậy mô hình không
  biết điều trị đã được thực hiện, gây confounding by indication. Nhãn "mất máu" dựa chủ yếu vào Hb có mốc thời gian (thưa) và
  mẫu hình SV, nên độ nhạy dự kiến thấp và phải báo cáo rõ.
- **Proxy pulse-contour** nhạy với damping, flush, đổi vị trí đầu dò và hiện tượng khuếch đại sóng ngoại vi. Cần SQI chặt; chỉ
  dùng thay đổi tương đối.
- **EV1000/Vigileo chỉ có ở một phần nhỏ, thiên về mổ lớn** (số ca: xem mục 2), gây selection bias cho chuẩn bạc. Cần báo cáo
  theo nhóm và không ngoại suy.
- **Vòng tròn giữa LF và đặc trưng** (mục 5.4). Chỉ chấp nhận kết quả đánh giá trên chuẩn độc lập.
- **Chỉ tiêu nhánh A** chưa đạt ở E08. Thêm kênh không bảo đảm đạt mục tiêu; kết quả không đạt phải được công bố.
- **Phạm vi.** Đây là nghiên cứu hồi cứu một trung tâm (SNUH). Kiểm định ngoài và tiến cứu chế độ im lặng là điều kiện trước
  khi dùng lâm sàng (kế hoạch §10).

---

## 9. Tài liệu tham khảo

1. Lee HC, Park Y, Yoon SB, et al. VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. *Sci Data* 2022;9:279. https://www.nature.com/articles/s41597-022-01411-5
2. Hatib F, Jian Z, et al. Machine-learning algorithm to predict hypotension based on high-fidelity arterial pressure waveform analysis. *Anesthesiology* 2018. https://pubmed.ncbi.nlm.nih.gov/29894315/
3. Davies SJ, Vistisen ST, Jian Z, Hatib F, Scheeren TWL. Ability of an arterial waveform analysis–derived hypotension prediction index to predict future hypotensive events in surgical patients. *Anesth Analg* 2020;130:352–359.
4. Kouz K, et al. Endotypes of intraoperative hypotension during major abdominal surgery: a retrospective machine learning analysis of an observational cohort study. *Br J Anaesth* 2023;130:253–261. https://www.sciencedirect.com/science/article/pii/S0007091222005578
5. Jian Z, Liu X, Kouz K, et al. Deep learning model to identify and validate hypotension endotypes in surgical and critically ill patients. *Br J Anaesth* 2025;134:308–316. https://pubmed.ncbi.nlm.nih.gov/39788817/
6. Zhu Y, Hao X, Li P, et al. Identification of intraoperative hypotension endotypes and revolution with a temporal deep learning algorithm. *Perioper Med* 2025;14:109. https://pmc.ncbi.nlm.nih.gov/articles/PMC12523014/
7. Sun JX, Reisner AT, Saeed M, Mark RG. Estimating cardiac output from arterial blood pressure waveforms: a critical evaluation using the MIMIC II database. *Computers in Cardiology* 2005;32:295–298.
8. Ratner A, Bach SH, Ehrenberg H, et al. Snorkel: rapid training data creation with weak supervision. *PVLDB* 2017;11(3):269–282. https://www.vldb.org/pvldb/vol11/p269-ratner.pdf
9. Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511, 2021.
10. Mulder MP, et al. *Anesthesiology* 2024; Yang et al. *BJA* 2025 — xem [kế hoạch §2](../UC04_RESEARCH_PLAN.md#2-paper-và-mã-nguồn-nên-đọc).

Ghi chú xác minh: các tài liệu 2, 4, 5, 6 và 8 đã được đối chiếu metadata (tác giả/năm/tạp chí/phương pháp chính) qua trang
nhà xuất bản, PubMed hoặc PMC ngày 23/09/2026. Tài liệu 3 và 7 được đối chiếu qua trang tóm tắt, chưa đọc toàn văn.
Tài liệu 9 trích theo metadata arXiv, chưa đọc toàn văn trong đợt này. Các con số hiệu năng của paper chỉ là số tác giả
công bố, không phải kết quả tái lập.
