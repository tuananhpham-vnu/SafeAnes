# Phân tích kết quả UC04 (validation)

> Mọi con số là của tập **validation** (248 ca, 245 bệnh nhân). Đây cũng là tập dùng để chọn W* và ngưỡng, nên các con số có phần lạc quan. Kết quả báo cáo cuối là của tập test, chạy một lần sau khi khóa mô hình (**chưa chạy**). Tập **test** chưa được dùng để huấn luyện, chọn cấu hình, chọn ngưỡng hay đánh giá. Tuy vậy, trước khi khóa mô hình, tổng số đợt tụt của test đã hai lần vô tình bị xem (DEVIATIONS mục 8 và 26); không quyết định nào dựa trên các con số đó, nhưng test không còn hoàn toàn "chưa được nhìn thấy". "Có ý nghĩa" trong file này là **danh nghĩa** (CI 95% không chứa 0), chưa hiệu chỉnh cho việc so sánh nhiều lần. Bảng đầy đủ theo từng mô hình: [FULL_RESULTS.md](FULL_RESULTS.md). File này được sinh bởi `scripts/make_analysis.py` từ các CSV trong `results/`.

**Mục lục:** [0. Tóm tắt](#0-tóm-tắt) · [1. Bối cảnh](#1-bối-cảnh-và-điều-kiện) · [2. Dữ liệu và nhãn](#2-dữ-liệu-và-nhãn) · [3. Kết quả chính](#3-kết-quả-chính-theo-mốc-dự-báo) · [4. Điểm vận hành](#4-điểm-vận-hành-cảnh-báo-trông-ra-sao) · [5. Cửa sổ W](#5-ảnh-hưởng-của-cửa-sổ-nhìn-lại-w) · [6. So sánh ghép cặp](#6-so-sánh-ghép-cặp-chênh-lệch-nào-là-thật) · [7. Calibration](#7-calibration-xác-suất-có-đáng-tin-không) · [8. Chính sách cảnh báo](#8-chính-sách-cảnh-báo-drop_below-hay-cooldown) · [9. Ablation](#9-ablation-nhóm-tín-hiệu-nào-không-thể-thiếu) · [10. SHAP](#10-shap-mô-hình-dựa-vào-đâu) · [11. Mức tin cậy](#11-mức-tin-cậy-có-nói-được-dự-đoán-này-ít-chắc-chắn-không) · [12. Cửa sổ dài (thăm dò)](#12-thăm-dò-cửa-sổ-dài-300-và-600-giây) · [13. Hạn chế](#13-hạn-chế-và-lưu-ý-khi-diễn-giải) · [14. Việc tiếp theo](#14-việc-tiếp-theo)

## 0. Tóm tắt

1. **LightGBM đạt độ nhạy theo đợt tụt khoảng 64% ở mốc 5 phút** dưới cùng giới hạn trên ≤ 1 cảnh báo sai/giờ, so với 56% của ngưỡng MAP đơn thuần. Mỗi mô hình tự chọn ngưỡng riêng, và LightGBM dùng nhiều ngân sách cảnh báo hơn (0,99 so với 0,83 cảnh báo sai/giờ).
2. **Đặc trưng sóng thủ công không cải thiện độ nhạy, AUROC hay AUPRC:** `lgbm_wave − lgbm_numeric` có CI chứa 0 ở cả 5 mốc cho ba chỉ số này; còn làm **tăng nhẹ gánh nặng cảnh báo** (FA/giờ 5 phút +0,060; thời gian cảnh báo bật 5 phút +0,0021).
3. **LightGBM xếp hạng nguy cơ tốt hơn ngưỡng MAP:** `lgbm_numeric − map_threshold` có AUROC cao hơn ở 5/5 mốc, và độ nhạy cao hơn ở 5/5 mốc (kèm FA/giờ cao hơn ở 3/5 mốc).
4. **MAP là nguồn thông tin chính:** chiếm khoảng 40% đóng góp SHAP; chỉ bỏ nhóm MAP mới làm giảm độ nhạy (-0,051 ở mốc 5 phút).
5. **Mức tin cậy từ 5 seed LightGBM không dùng được** (không qua kiểm chứng). Phân tích thăm dò không thấy W = 300 hoặc 600 cải thiện độ nhạy của `lgbm_numeric` so với W = 120.

> **Kết luận hiện tại (validation):** LightGBM cải thiện khả năng phân biệt nguy cơ và đạt độ nhạy cao hơn dưới giới hạn ≤ 1 cảnh báo sai/giờ so với ngưỡng MAP, nhưng đặc trưng sóng thủ công không tạo lợi ích đo được và có thể làm tăng nhẹ gánh nặng cảnh báo ở mốc 5 phút. Kết luận cuối phải chờ tập test (sau khi khóa mô hình) và mô hình DL học trực tiếp từ sóng.

## 1. Bối cảnh và điều kiện

*Kết quả này được tạo ra như thế nào?*

| Mục | Giá trị |
| --- | --- |
| Bài toán | Tại mỗi thời điểm (cứ 30 giây), dự báo có đợt tụt huyết áp **mới** bắt đầu trong h phút tới |
| Đợt tụt | MAP < 65 mmHg kéo dài ≥ 60 giây; hai đợt cách nhau < 120 giây được gộp |
| Dữ liệu | VitalDB, cohort D (có sóng và arterial line hoạt động); `samples.json` sha256 `9b17751a…` |
| Chia tập (theo bệnh nhân) | train → fit; calibration → Platt; validation → chọn ngưỡng và báo cáo; test → chưa dùng để đánh giá (xem mục 13) |
| Nhãn | `label_policy` = lenient, luật nhãn âm = possible_event, `exclusion_reset` = event |
| Cửa sổ nhìn lại W | 30, 60, 90, 120 giây; W* = 120 (chọn theo `lgbm_wave`, trung bình độ nhạy mốc 5 và 10 phút) |
| Mốc dự báo h | 5, 10, 15, 20, 30 phút |
| Chọn ngưỡng | Độ nhạy theo đợt tụt cao nhất với ≤ 1 cảnh báo sai/giờ (quét ~300 ứng viên) |
| Chính sách cảnh báo | `drop_below`: 2 mốc liên tiếp vượt ngưỡng, nghỉ 300 giây, phải xuống dưới ngưỡng mới cảnh báo lại |
| Khoảng tin cậy | Bootstrap theo bệnh nhân, 200 lần rút; so sánh mô hình dùng bootstrap ghép cặp |
| Không dùng | Augmentation, oversample, trọng số lớp |
| Code | commit `30c7021`, dirty = False; scikit-learn 1.9.1, LightGBM 4.7.0 |

**Bốn mô hình:**

| Mô hình | Mô tả | Số tổ hợp |
| --- | --- | --: |
| `map_threshold` | Điểm nguy cơ = −MAP hiện tại (không fit, không phụ thuộc W) | 5 |
| `map_logistic` | Logistic trên MAP hiện tại, độ dốc MAP, độ lệch chuẩn MAP | 20 (4 W × 5 h) |
| `lgbm_numeric` | LightGBM, 66 đặc trưng chỉ số monitor | 20 (4 W × 5 h) |
| `lgbm_wave` | LightGBM, 66 đặc trưng chỉ số + 27 đặc trưng sóng động mạch | 20 (4 W × 5 h) |

## 2. Dữ liệu và nhãn

*Mô hình học trên bao nhiêu ca, bao nhiêu đợt tụt? Nhãn có cân bằng không?*

| Tập | Ca | Bệnh nhân | Dòng eligible | Đợt tụt | Đợt cảnh báo được (5 phút) |
| --- | --: | --: | --: | --: | --: |
| train | 2393 | 2302 | 747.762 | 4922 | 3970 |
| calibration | 258 | 244 | 78.263 | 476 | 364 |
| validation | 248 | 245 | 78.909 | 524 | 400 |

Nhãn trên train (dòng eligible):

| Mốc | Tỷ lệ dương | Nhãn −1 | −1 do hết ca | −1 do có thể che đợt tụt |
| --- | --: | --: | --: | --: |
| 5 phút | 4,1% | 6,5% | 3,5% | 3,0% |
| 10 phút | 7,9% | 11,3% | 6,3% | 5,0% |
| 15 phút | 11,5% | 15,6% | 9,0% | 6,6% |
| 20 phút | 14,9% | 19,5% | 11,6% | 7,8% |
| 30 phút | 21,0% | 26,3% | 16,6% | 9,7% |

- Dữ liệu mất cân bằng mạnh ở mốc ngắn (khoảng 4% dương ở 5 phút), nên AUPRC phải đọc cùng tỷ lệ dương.
- Ở mốc dài, phần lớn nhãn −1 là do không đủ thời gian theo dõi trước khi hết ca, không phải do nhiễu.

## 3. Kết quả chính theo mốc dự báo

*Mô hình nào cảnh báo được nhiều đợt tụt nhất?* (W = 120, ngưỡng cho ≤ 1 cảnh báo sai/giờ)

**Độ nhạy theo đợt tụt [CI 95%] ↑**

| Mô hình | 5 phút | 10 phút | 15 phút | 20 phút | 30 phút |
| --- | --: | --: | --: | --: | --: |
| `map_threshold` | 0,560 [0,511–0,604] | 0,622 [0,575–0,668] | 0,611 [0,563–0,662] | 0,615 [0,572–0,665] | 0,647 [0,601–0,693] |
| `map_logistic` | 0,623 [0,574–0,665] | 0,669 [0,625–0,715] | 0,650 [0,608–0,693] | 0,675 [0,630–0,721] | 0,684 [0,638–0,732] |
| `lgbm_numeric` | 0,627 [0,572–0,680] | **0,686 [0,642–0,730]** | **0,675 [0,637–0,712]** | 0,686 [0,645–0,726] | **0,686 [0,648–0,728]** |
| `lgbm_wave` | **0,635 [0,591–0,678]** | **0,686 [0,639–0,727]** | 0,661 [0,623–0,704] | **0,688 [0,649–0,733]** | 0,678 [0,635–0,725] |

**AUROC [CI 95%] ↑**

| Mô hình | 5 phút | 10 phút | 15 phút | 20 phút | 30 phút |
| --- | --: | --: | --: | --: | --: |
| `map_threshold` | 0,828 [0,803–0,851] | 0,767 [0,738–0,793] | 0,739 [0,708–0,771] | 0,733 [0,700–0,767] | 0,720 [0,687–0,753] |
| `map_logistic` | 0,839 [0,818–0,861] | 0,780 [0,753–0,806] | 0,755 [0,725–0,786] | 0,749 [0,718–0,782] | 0,738 [0,707–0,772] |
| `lgbm_numeric` | 0,874 [0,856–0,891] | 0,817 [0,794–0,841] | 0,790 [0,759–0,816] | 0,779 [0,750–0,808] | 0,764 [0,732–0,798] |
| `lgbm_wave` | **0,875 [0,857–0,892]** | **0,820 [0,795–0,843]** | **0,791 [0,764–0,819]** | **0,781 [0,751–0,811]** | **0,768 [0,734–0,802]** |

**AUPRC ↑** (so với tỷ lệ dương)

| Mô hình | 5 phút | 10 phút | 15 phút | 20 phút | 30 phút |
| --- | --: | --: | --: | --: | --: |
| `map_threshold` | 0,217 | 0,257 | 0,296 | 0,344 | 0,423 |
| `map_logistic` | 0,228 | 0,260 | 0,299 | 0,348 | 0,428 |
| `lgbm_numeric` | 0,278 | 0,316 | 0,355 | 0,398 | 0,473 |
| `lgbm_wave` | 0,282 | 0,318 | 0,354 | 0,403 | 0,478 |
| *Tỷ lệ dương* | 0,039 | 0,077 | 0,114 | 0,147 | 0,210 |

- LightGBM có độ nhạy cao nhất ở hầu hết các mốc, nhưng khoảng cách với `map_logistic` nhỏ và các CI chồng lên nhau. Mỗi mô hình có ngưỡng riêng (cùng giới hạn trên ≤ 1 FA/giờ), nên độ nhạy phải đọc cùng FA/giờ ở phần 4 và 6.
- AUROC và AUPRC của LightGBM cao hơn hai baseline: mô hình xếp hạng nguy cơ tốt hơn (phần 6).
- Theo mốc dài hơn, độ nhạy tăng nhẹ còn AUROC giảm dần.

## 4. Điểm vận hành: cảnh báo trông ra sao?

*Ở ngưỡng đã chọn, người dùng nhận được gì?* (W = 120)

| Mô hình | Mốc | Cảnh báo sai/giờ ↓ | Cảnh báo đúng / sai / censored | PPV ↑ | Báo trước ≥ 5 ph ↑ | Thời gian báo trước (trung vị [Q1–Q3]) | Thời gian cảnh báo bật ↓ | Coverage |
| --- | --- | --: | --: | --: | --: | --: | --: | --: |
| `map_threshold` | 5 phút | 0,83 | 224 / 509 / 41 | 31% | 0% | 1,6 ph [0,7 ph–2,8 ph] | 3,0% | 88% |
| `map_threshold` | 10 phút | 0,94 | 309 / 548 / 89 | 36% | 16% | 2,8 ph [1,1 ph–5,1 ph] | 4,4% | 88% |
| `map_logistic` | 5 phút | 0,98 | 249 / 605 / 44 | 29% | 0% | 1,6 ph [0,6 ph–3,1 ph] | 3,0% | 88% |
| `map_logistic` | 10 phút | 0,99 | 343 / 576 / 102 | 37% | 18% | 2,6 ph [1,1 ph–5,2 ph] | 3,5% | 88% |
| `lgbm_numeric` | 5 phút | 0,93 | 251 / 575 / 47 | 30% | 0% | 1,7 ph [0,8 ph–3,0 ph] | 2,7% | 88% |
| `lgbm_numeric` | 10 phút | 0,98 | 346 / 571 / 96 | 38% | 20% | 2,9 ph [1,2 ph–5,6 ph] | 3,6% | 88% |
| `lgbm_wave` | 5 phút | 0,99 | 254 / 611 / 40 | 29% | 0% | 1,6 ph [0,7 ph–3,1 ph] | 2,9% | 88% |
| `lgbm_wave` | 10 phút | 0,99 | 342 / 577 / 97 | 37% | 21% | 3,0 ph [1,2 ph–5,5 ph] | 3,7% | 88% |

*Cảnh báo censored là cảnh báo tại thời điểm có nhãn −1 (không đủ dữ liệu tương lai để biết đúng hay sai); chúng không nằm trong mẫu số của PPV và của FA/giờ.*

- Với `lgbm_wave` ở mốc 5 phút: PPV 29,4% là tỷ lệ đúng **trong số cảnh báo đánh giá được** (254/865). Tính trên toàn bộ 905 cảnh báo: 28,1% đúng, 67,5% sai, 4,4% censored. Trạng thái cảnh báo bật khoảng 2,9% thời gian theo dõi eligible.
- **Thời gian báo trước ngắn:** trung vị 1,6 ph ở mốc 5 phút và 3,0 ph ở mốc 10 phút (`lgbm_wave`): thời gian để can thiệp trước khi MAP xuống dưới 65 thường chỉ vài phút.
- Ở mốc 5 phút, tỷ lệ báo trước ≥ 5 phút **quan sát được** là 0%. Về định nghĩa, một cảnh báo đúng đúng 300 giây trước khi đợt tụt bắt đầu vẫn có thể được tính, nên đây là kết quả quan sát, không phải điều bắt buộc. Ở mốc 10 phút, `lgbm_wave` báo trước ≥ 5 phút cho 21% số đợt có thể cảnh báo.
- Số dòng kết quả không đạt ngân sách ≤ 1 cảnh báo sai/giờ: 0/80.
- Coverage khoảng 88%: phần còn lại là lúc MAP đã < 65, đang trong hoặc ngay sau đợt tụt, hoặc thiếu MAP.

## 5. Ảnh hưởng của cửa sổ nhìn lại W

*Nhìn lại bao lâu là đủ?* (độ nhạy theo đợt tụt)

| Mô hình | W (giây) | 5 phút | 10 phút | 15 phút | 20 phút | 30 phút |
| --- | --: | --: | --: | --: | --: | --: |
| `lgbm_wave` | 30 | 0,627 | 0,652 | 0,648 | 0,659 | 0,656 |
| `lgbm_wave` | 60 | 0,640 | 0,664 | 0,655 | 0,653 | 0,662 |
| `lgbm_wave` | 90 | 0,623 | 0,674 | 0,645 | 0,679 | 0,675 |
| `lgbm_wave` | 120 | 0,635 | 0,686 | 0,661 | 0,688 | 0,678 |
| `lgbm_numeric` | 30 | 0,625 | 0,648 | 0,641 | 0,659 | 0,660 |
| `lgbm_numeric` | 60 | 0,650 | 0,667 | 0,666 | 0,668 | 0,658 |
| `lgbm_numeric` | 90 | 0,618 | 0,669 | 0,664 | 0,684 | 0,691 |
| `lgbm_numeric` | 120 | 0,627 | 0,686 | 0,675 | 0,686 | 0,686 |
| `map_logistic` | 30 | 0,593 | 0,638 | 0,641 | 0,661 | 0,686 |
| `map_logistic` | 60 | 0,605 | 0,648 | 0,652 | 0,673 | 0,673 |
| `map_logistic` | 90 | 0,635 | 0,674 | 0,659 | 0,670 | 0,686 |
| `map_logistic` | 120 | 0,623 | 0,669 | 0,650 | 0,675 | 0,684 |

- Căn cứ chọn W*: trung bình độ nhạy mốc 5 và 10 phút của `lgbm_wave` là W=30: 0,640, W=60: 0,652, W=90: 0,648, W=120: 0,660 → W* = 120.
- Chênh lệch giữa các W nhỏ (2,0% theo tiêu chí chọn W*) so với độ rộng CI của độ nhạy (khoảng ±4–5 điểm phần trăm): chọn W không quyết định kết quả.

## 6. So sánh ghép cặp: chênh lệch nào là thật?

*Hai mô hình được so trên cùng các lần rút bootstrap theo bệnh nhân. **Chữ đậm** = CI 95% không chứa 0 (danh nghĩa, chưa hiệu chỉnh so sánh nhiều lần). Mỗi mô hình có ngưỡng riêng, chỉ cùng giới hạn trên ≤ 1 cảnh báo sai/giờ, nên độ nhạy phải đọc cùng Δ FA/giờ và Δ thời gian cảnh báo bật.*

| Cặp | Mốc | Δ Độ nhạy | Δ FA/giờ | Δ Thời gian cảnh báo bật | Δ AUROC | Δ AUPRC |
| --- | --- | --: | --: | --: | --: | --: |
| `lgbm_wave − lgbm_numeric` | 5 phút | +0,007 [-0,019; +0,033] | **+0,060 [+0,022; +0,093]** | **+0,0021 [+0,0010; +0,0033]** | +0,001 [-0,001; +0,004] | +0,004 [-0,004; +0,012] |
| `lgbm_wave − lgbm_numeric` | 10 phút | +0,000 [-0,019; +0,018] | +0,012 [-0,038; +0,056] | +0,0005 [-0,0011; +0,0020] | +0,003 [-0,001; +0,006] | +0,002 [-0,007; +0,011] |
| `lgbm_wave − lgbm_numeric` | 15 phút | -0,014 [-0,035; +0,003] | +0,001 [-0,044; +0,041] | +0,0002 [-0,0018; +0,0024] | +0,001 [-0,003; +0,005] | -0,001 [-0,009; +0,006] |
| `lgbm_wave − lgbm_numeric` | 20 phút | +0,003 [-0,016; +0,020] | -0,017 [-0,067; +0,024] | +0,0008 [-0,0021; +0,0042] | +0,002 [-0,002; +0,006] | +0,005 [-0,004; +0,014] |
| `lgbm_wave − lgbm_numeric` | 30 phút | -0,009 [-0,028; +0,008] | -0,019 [-0,072; +0,037] | +0,0008 [-0,0035; +0,0053] | +0,003 [-0,002; +0,008] | +0,005 [-0,004; +0,013] |
| `lgbm_wave − map_logistic` | 5 phút | +0,012 [-0,026; +0,053] | +0,013 [-0,058; +0,093] | -0,0009 [-0,0036; +0,0021] | **+0,036 [+0,025; +0,047]** | **+0,053 [+0,029; +0,079]** |
| `lgbm_wave − map_logistic` | 10 phút | +0,016 [-0,014; +0,051] | +0,004 [-0,096; +0,098] | +0,0022 [-0,0014; +0,0063] | **+0,040 [+0,026; +0,054]** | **+0,058 [+0,038; +0,080]** |
| `lgbm_wave − map_logistic` | 15 phút | +0,010 [-0,024; +0,046] | +0,042 [-0,081; +0,164] | **+0,0116 [+0,0069; +0,0162]** | **+0,036 [+0,021; +0,050]** | **+0,054 [+0,031; +0,076]** |
| `lgbm_wave − map_logistic` | 20 phút | +0,013 [-0,018; +0,047] | +0,029 [-0,090; +0,126] | **+0,0184 [+0,0116; +0,0263]** | **+0,032 [+0,016; +0,048]** | **+0,054 [+0,028; +0,079]** |
| `lgbm_wave − map_logistic` | 30 phút | -0,006 [-0,040; +0,026] | -0,008 [-0,101; +0,093] | **+0,0223 [+0,0133; +0,0316]** | **+0,030 [+0,012; +0,046]** | **+0,049 [+0,026; +0,074]** |
| `lgbm_numeric − map_threshold` | 5 phút | **+0,068 [+0,021; +0,117]** | **+0,109 [+0,035; +0,207]** | **-0,0034 [-0,0070; -0,0002]** | **+0,046 [+0,034; +0,059]** | **+0,061 [+0,040; +0,090]** |
| `lgbm_numeric − map_threshold` | 10 phút | **+0,064 [+0,029; +0,113]** | +0,039 [-0,052; +0,152] | **-0,0079 [-0,0130; -0,0027]** | **+0,051 [+0,035; +0,067]** | **+0,060 [+0,040; +0,083]** |
| `lgbm_numeric − map_threshold` | 15 phút | **+0,063 [+0,024; +0,103]** | **+0,113 [+0,019; +0,208]** | +0,0012 [-0,0047; +0,0075] | **+0,051 [+0,034; +0,069]** | **+0,059 [+0,036; +0,085]** |
| `lgbm_numeric − map_threshold` | 20 phút | **+0,070 [+0,036; +0,112]** | **+0,231 [+0,113; +0,336]** | **+0,0133 [+0,0075; +0,0207]** | **+0,046 [+0,029; +0,065]** | **+0,054 [+0,031; +0,081]** |
| `lgbm_numeric − map_threshold` | 30 phút | **+0,040 [+0,012; +0,075]** | +0,102 [-0,007; +0,211] | **+0,0121 [+0,0038; +0,0224]** | **+0,045 [+0,024; +0,065]** | **+0,050 [+0,025; +0,079]** |

- **Đặc trưng sóng thủ công (`lgbm_wave − lgbm_numeric`):** không cải thiện độ nhạy, AUROC hay AUPRC (CI chứa 0 ở mọi mốc); ngược lại, làm **tăng nhẹ gánh nặng cảnh báo**: FA/giờ 5 phút +0,060; thời gian cảnh báo bật 5 phút +0,0021.
- **`lgbm_wave` so với `map_logistic`:** AUROC/AUPRC cao hơn ở 5/5 mốc (xếp hạng nguy cơ tốt hơn); độ nhạy dưới giới hạn ≤ 1 FA/giờ khác biệt ở 0/5 mốc.
- **`lgbm_numeric` so với ngưỡng MAP:** độ nhạy cao hơn ở 5/5 mốc dưới cùng giới hạn trên ≤ 1 FA/giờ, nhưng **dùng nhiều ngân sách cảnh báo hơn** (FA/giờ cao hơn ở 3/5 mốc: FA/giờ 5 phút +0,11; FA/giờ 15 phút +0,11; FA/giờ 20 phút +0,23). Đây không phải so sánh ở cùng một mức FA/giờ; bằng chứng xếp hạng tốt hơn là AUROC cao hơn ở 5/5 mốc.

## 7. Calibration: xác suất có đáng tin không?

*Trung bình xác suất dự đoán so với tỷ lệ dương thực tế trên validation* (W = 120)

| Mô hình | 5 phút (dự đoán / thực) | 10 phút (dự đoán / thực) | 15 phút (dự đoán / thực) | 20 phút (dự đoán / thực) | 30 phút (dự đoán / thực) | ECE 5 ph ↓ |
| --- | --: | --: | --: | --: | --: | --: |
| `map_threshold` | 3,9% / 3,9% | 7,7% / 7,7% | 11,2% / 11,4% | 14,5% / 14,7% | 20,8% / 21,0% | 0,0128 |
| `map_logistic` | 3,9% / 3,9% | 7,7% / 7,7% | 11,2% / 11,4% | 14,4% / 14,7% | 20,7% / 21,0% | 0,0139 |
| `lgbm_numeric` | 3,8% / 3,9% | 7,7% / 7,7% | 11,1% / 11,4% | 14,4% / 14,7% | 20,8% / 21,0% | 0,0016 |
| `lgbm_wave` | 3,9% / 3,9% | 7,7% / 7,7% | 11,2% / 11,4% | 14,5% / 14,7% | 20,8% / 21,0% | 0,0011 |

- **Calibration trung bình tốt:** trung bình xác suất lệch tỷ lệ dương tối đa 0,34% trên cả 65 tổ hợp. Calibration được fit trên tập calibration riêng, nên đây là bằng chứng độc lập.
- **Nhưng trung bình khớp chưa đủ để khẳng định calibration tốt trên toàn miền nguy cơ.** ECE của LightGBM từ 0,0011 đến 0,0150; ECE lớn nhất là 0,0428 (`map_threshold`, mốc 30 phút). Chưa có đồ thị reliability theo từng khoảng xác suất.

## 8. Chính sách cảnh báo: `drop_below` hay `cooldown`?

*`drop_below`: phải xuống dưới ngưỡng mới cảnh báo lại. `cooldown`: vẫn trên ngưỡng thì cảnh báo lại mỗi 5 phút.* (W = 120, mốc 5 và 10 phút)

| Mô hình | Mốc | Độ nhạy drop_below | Độ nhạy cooldown | FA/giờ drop_below | FA/giờ cooldown | Cảnh báo bật drop_below | Cảnh báo bật cooldown |
| --- | --- | --: | --: | --: | --: | --: | --: |
| `map_threshold` | 5 phút | 0,560 | 0,565 | 0,83 | 0,85 | 3,0% | 3,0% |
| `map_threshold` | 10 phút | 0,622 | 0,629 | 0,94 | 0,99 | 4,4% | 4,4% |
| `map_logistic` | 5 phút | 0,623 | 0,625 | 0,98 | 0,99 | 3,0% | 3,0% |
| `map_logistic` | 10 phút | 0,669 | 0,669 | 0,99 | 0,99 | 3,5% | 3,5% |
| `lgbm_numeric` | 5 phút | 0,627 | 0,632 | 0,93 | 0,94 | 2,7% | 2,7% |
| `lgbm_numeric` | 10 phút | 0,686 | 0,683 | 0,98 | 0,99 | 3,6% | 3,6% |
| `lgbm_wave` | 5 phút | 0,635 | 0,630 | 0,99 | 0,96 | 2,9% | 2,7% |
| `lgbm_wave` | 10 phút | 0,686 | 0,683 | 0,99 | 1,00 | 3,7% | 3,6% |

- Hai chính sách cho độ nhạy chênh tối đa 2,4% trên cả 65 tổ hợp: lựa chọn chính sách không đổi kết luận.
- Chính sách chính vẫn **chờ nhóm chốt** (mặc định `drop_below`).

## 9. Ablation: nhóm tín hiệu nào không thể thiếu?

*Bỏ lần lượt từng nhóm cột khỏi `lgbm_wave` (W = 120); chênh lệch so với mô hình đầy đủ, bootstrap ghép cặp.*

| Nhóm bị bỏ | Δ Độ nhạy 5 ph | Δ AUROC 5 ph | Δ Độ nhạy 10 ph | Δ AUROC 10 ph |
| --- | --: | --: | --: | --: |
| MAP | **-0,051 [-0,082; -0,021]** | **-0,005 [-0,008; -0,003]** | **-0,047 [-0,072; -0,026]** | **-0,006 [-0,009; -0,003]** |
| SBP/DBP/PP | +0,007 [-0,011; +0,029] | +0,000 [-0,001; +0,001] | +0,007 [-0,003; +0,018] | -0,001 [-0,002; +0,001] |
| HR, shock index | +0,005 [-0,005; +0,016] | -0,001 [-0,002; +0,000] | -0,005 [-0,017; +0,005] | **-0,002 [-0,004; -0,000]** |
| SpO2 | +0,008 [-0,007; +0,024] | +0,000 [-0,000; +0,001] | +0,000 [-0,006; +0,007] | -0,000 [-0,001; +0,001] |
| EtCO2, RR | +0,005 [-0,012; +0,024] | -0,001 [-0,003; +0,001] | +0,005 [-0,007; +0,016] | -0,001 [-0,005; +0,002] |
| Hình dạng sóng | -0,000 [-0,022; +0,019] | +0,000 [-0,001; +0,001] | -0,010 [-0,026; +0,005] | -0,001 [-0,003; +0,001] |
| PPV | +0,002 [-0,004; +0,011] | +0,000 [-0,000; +0,000] | -0,002 [-0,008; +0,000] | -0,000 [-0,000; +0,000] |
| SV/CO/SVR (proxy) | -0,007 [-0,033; +0,018] | -0,001 [-0,003; +0,001] | +0,000 [-0,014; +0,017] | -0,001 [-0,004; +0,001] |

- Chỉ bỏ **MAP** làm giảm **độ nhạy** có ý nghĩa, ở cả hai mốc; AUROC cũng giảm.
- Các thay đổi khác có CI không chứa 0 nhưng rất nhỏ, không ảnh hưởng độ nhạy: HR, shock index (AUROC 10 phút: -0,002).
- Bỏ các nhóm khác gần như không đổi kết quả. Điều này **không** có nghĩa các nhóm đó vô ích: nhiều nhóm trùng thông tin với nhau (SBP/DBP, PP và proxy SV/CO/SVR đều tính từ huyết áp), nên nhóm còn lại bù vào.

## 10. SHAP: mô hình dựa vào đâu?

*Tỷ lệ |SHAP| theo nhóm cột, `lgbm_wave`, W = 120, 50.000 dòng validation.*

| Nhóm | 5 phút | 10 phút | 15 phút | 20 phút | 30 phút |
| --- | --: | --: | --: | --: | --: |
| MAP | 40% | 36% | 33% | 32% | 31% |
| SBP/DBP/PP | 18% | 17% | 18% | 18% | 18% |
| SV/CO/SVR (proxy) | 16% | 18% | 19% | 20% | 20% |
| Hình dạng sóng | 10% | 10% | 9% | 9% | 10% |
| HR, shock index | 7% | 8% | 7% | 6% | 4% |
| EtCO2, RR | 7% | 10% | 11% | 12% | 14% |
| SpO2 | 1% | 1% | 1% | 1% | 1% |
| Chất lượng nhịp | 1% | 1% | 1% | 2% | 2% |
| PPV | 0% | 0% | 1% | 0% | 0% |

- MAP chiếm tỷ lệ lớn nhất ở mọi mốc; cùng với SBP/DBP/PP, thông tin huyết áp chiếm phần lớn.
- SV/CO/SVR là **proxy tính từ huyết áp** (SV = PP/(SBP + DBP)), không phải số đo cung lượng tim: không nên kết luận mô hình dùng cung lượng tim.
- PPV đóng góp khoảng 1% hoặc ít hơn ở mọi mốc.

## 11. Mức tin cậy: có nói được "dự đoán này ít chắc chắn" không?

*Độ lệch chuẩn logit giữa 5 seed LightGBM → 3 mức theo tam phân vị. Tiêu chí: trong từng thập phân vị nguy cơ, nhóm "tin cậy thấp" phải có sai số calibration lớn hơn nhóm "tin cậy cao" ở ≥ 8/10 thập phân vị, sai số gộp ≥ 1,2 lần, và Brier tệ hơn.*

| Mốc | Thập phân vị "thấp" kém hơn | Sai số calibration (thấp / cao) | Brier (thấp / cao) | Kết luận |
| --- | --: | --: | --: | --: |
| 5 phút | 4/10 | 0,0031 / 0,0027 | 0,0402 / 0,0245 | **Không qua** |
| 10 phút | 7/10 | 0,0061 / 0,0053 | 0,0726 / 0,0512 | **Không qua** |

- Không qua ở cả hai mốc → **bỏ mức tin cậy của LightGBM** (DEVIATIONS 28).
- Độ lệch giữa các seed của DL sẽ được kiểm bằng cùng tiêu chí.

## 12. Thăm dò: cửa sổ dài 300 và 600 giây

*W* = 120 là cửa sổ dài nhất đã thử. Phân tích này quyết định **sau khi xem validation**, nên chỉ là thăm dò: báo cáo riêng, không thay W*. Đặc trưng của cả 3 cửa sổ được tính lại bằng cùng một hàm (khớp `prep_v1` ở 99,9% dòng).*

| Mốc | W=120 | W=300 | W=600 | Δ W300−W120 | Δ W600−W120 |
| --- | --: | --: | --: | --: | --: |
| 5 phút | 0,630 | 0,632 | 0,608 | +0,003 [-0,027; +0,033] | -0,022 [-0,060; +0,010] |
| 10 phút | 0,681 | 0,667 | 0,638 | -0,015 [-0,042; +0,012] | **-0,044 [-0,074; -0,017]** |
| 15 phút | 0,666 | 0,673 | 0,652 | +0,006 [-0,017; +0,029] | -0,014 [-0,043; +0,017] |
| 20 phút | 0,682 | 0,677 | 0,659 | -0,005 [-0,025; +0,014] | **-0,023 [-0,047; -0,002]** |
| 30 phút | 0,684 | 0,684 | 0,684 | -0,000 [-0,016; +0,019] | -0,000 [-0,024; +0,021] |

- **Không thấy W = 300 hoặc 600 cải thiện độ nhạy của `lgbm_numeric` so với W = 120** trên validation; W = 600 còn thấp hơn (danh nghĩa) ở mốc 10 phút, 20 phút.
- Kết luận này **có phạm vi hẹp:** chỉ thử hai cửa sổ dài, chỉ cho `lgbm_numeric`, trên cùng tập validation và sau khi đã xem kết quả. Nó không loại trừ các W trung gian (ví dụ 180–240 giây) hay mô hình học trực tiếp từ sóng thô.

## 13. Hạn chế và lưu ý khi diễn giải

**Về phương pháp thống kê**

- **Validation vừa để chọn vừa để báo cáo:** W* và ngưỡng được chọn trên chính tập này, nên con số có phần lạc quan.
- **CI chưa gồm bất định do chọn ngưỡng:** bootstrap giữ cố định ngưỡng đã chọn, không lặp lại bước chọn ngưỡng trong từng lần rút, nên CI của độ nhạy và FA/giờ có thể hẹp hơn thực tế.
- **200 lần rút bootstrap** tương đối ít cho CI 95%; giới hạn CI có thể dao động khi chạy lại với seed khác.
- **Chưa hiệu chỉnh so sánh nhiều lần:** nhiều mô hình, mốc, cặp và nhóm ablation được kiểm; "có ý nghĩa" chỉ là danh nghĩa (CI 95% không chứa 0).
- **Chỉ số theo dòng (AUROC, AUPRC, Brier, ECE)** tính trên các mốc 30 giây có tương quan mạnh trong cùng ca, và chịu ảnh hưởng nhiều hơn từ các ca mổ dài; CI vẫn bootstrap theo bệnh nhân.
- **So sánh độ nhạy không ở cùng một mức FA/giờ:** mỗi mô hình có ngưỡng riêng, chỉ cùng giới hạn trên ≤ 1 FA/giờ.

**Về dữ liệu, đặc trưng và phạm vi**

- **Test:** Tập **test** chưa được dùng để huấn luyện, chọn cấu hình, chọn ngưỡng hay đánh giá. Tuy vậy, trước khi khóa mô hình, tổng số đợt tụt của test đã hai lần vô tình bị xem (DEVIATIONS mục 8 và 26); không quyết định nào dựa trên các con số đó, nhưng test không còn hoàn toàn "chưa được nhìn thấy".
- **Phạm vi:** kết luận chỉ áp dụng cho cohort D (ca có sóng động mạch và arterial line hoạt động) của VitalDB.
- **MAP nền:** chỉ khoảng 4% ca có MAP đo trước khởi mê; `map_drop_pct` thực chất là mức giảm so với MAP sau khởi mê.
- **Proxy SV/CO/SVR** tính từ huyết áp, chưa hiệu chỉnh; không diễn giải như số đo cung lượng tim.
- **Các nhóm tín hiệu trùng thông tin:** ablation "không đổi" không có nghĩa nhóm đó vô ích.
- **`in_event`/`post_event`** dùng đợt tụt phát hiện từ `label_map`, có thể nhìn trước tối đa khoảng 60 giây (DEVIATIONS 5).
- **Nhãn −1** ở mốc dài chiếm tới khoảng 26% dòng eligible (phần lớn do hết ca); chỉ số theo dòng tính trên dòng có nhãn.
- **Chưa có DL:** câu hỏi "sóng thô có thêm thông tin không" mới được trả lời cho đặc trưng sóng (LightGBM), chưa cho mô hình học trực tiếp từ sóng.

## 14. Việc tiếp theo

| Việc | Ai | Trạng thái |
| --- | --- | --- |
| Chốt `label_policy` và chính sách cảnh báo | Nhóm | Chờ |
| Train DL (Conv1D + Transformer) trên Kaggle | Kaggle + HF | Notebook sẵn sàng; cần dataset `uc04-prep-v1` |
| Khóa mô hình (`lock_models.py`) | Máy | Kết quả tabular đủ điều kiện (195 file) |
| Chạy test một lần (NB04) | Máy | Sau khi khóa |
