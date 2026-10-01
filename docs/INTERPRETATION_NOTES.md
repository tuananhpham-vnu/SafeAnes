# Ghi chú diễn giải kết quả (validation)

Mọi con số trong file này là của tập **validation**, là tập dùng để chọn ngưỡng và cấu hình. Kết quả báo cáo cuối là của tập test, chạy một lần sau khi khóa mô hình.

## 1. Quyết định

| Câu hỏi | Trạng thái | Căn cứ |
|---|---|---|
| `exclusion_reset` | **Đề xuất chốt `event`** (đang dùng trong config) | Luật `any_low` làm mất 16% số đợt tụt có thể cảnh báo ở mốc 5 phút (3.970 → 3.345 trên train). Đó là các đợt tụt nối tiếp nhau, tình huống lâm sàng quan trọng. `event` đã loại dòng có MAP hiện tại < 65 (`in_hypotension`), và là luật proposal đã viết |
| Luật nhãn âm | `possible_event` (DEVIATIONS 23) | Tỷ lệ −1 ở mốc 5 phút: 18,1% → 6,5% |
| `label_policy` | Chờ nhóm chốt (`lenient` đang dùng) | `label_report.csv` |
| Chính sách cảnh báo (`evaluation.alarm_rearm`) | **Chưa chốt.** Mặc định `drop_below` (mục 8.2 bản 2.5) | Đã tính cả `cooldown` (cảnh báo lặp lại mỗi 5 phút khi vẫn trên ngưỡng) từ dự đoán có sẵn: `reports/model_comparison.csv`. Hai chính sách cho độ nhạy chênh nhau tối đa khoảng 1,5 điểm phần trăm và không đổi kết luận. Nếu chốt `cooldown`, đặt config rồi chạy lại tabular để mọi file của từng tổ hợp dùng cùng chính sách |

## 2. Câu hỏi nghiên cứu chính: sóng động mạch có thêm thông tin không?

Bảng đầy đủ: `reports/model_comparison.csv` (5 mốc, 4 mô hình, 2 chính sách cảnh báo) và `reports/model_comparison_pairs.csv` (bootstrap ghép cặp theo bệnh nhân, 200 lần rút). W = 120, chính sách `drop_below`:

| Cặp | Độ nhạy theo đợt tụt (FA ≤ 1/giờ) | AUROC | AUPRC |
|---|---|---|---|
| `lgbm_wave − lgbm_numeric` | −0,014 đến +0,007, **mọi CI chứa 0** | +0,001 đến +0,003, CI chứa 0 | −0,001 đến +0,005, CI chứa 0 |
| `lgbm_wave − map_logistic` | −0,006 đến +0,016, CI chứa 0 | **+0,030 đến +0,040**, CI > 0 | **+0,049 đến +0,058**, CI > 0 |
| `lgbm_numeric − map_threshold` | **+0,040 đến +0,070**, CI > 0 ở mọi mốc | **+0,045 đến +0,051** | **+0,050 đến +0,061** |

Cách đọc:
- Trên validation, **đặc trưng sóng chưa cho thấy lợi ích** so với 66 đặc trưng chỉ số: độ nhạy, AUROC và AUPRC có CI chứa 0. Ở mốc 5 phút, sóng làm FA/giờ (+0,060) và thời gian cảnh báo bật (+0,0021) tăng nhẹ, với CI không chứa 0.
- LightGBM xếp hạng rủi ro tốt hơn `map_logistic` (AUROC, AUPRC), nhưng số đợt tụt được cảnh báo không khác có ý nghĩa. Hai mô hình chỉ cùng thỏa giới hạn FA ≤ 1/giờ; FA/giờ thực tế của chúng không bằng nhau.
- `lgbm_numeric` có độ nhạy, AUROC và AUPRC cao hơn ngưỡng MAP ở mọi mốc, nhưng FA/giờ cũng cao hơn ở 3/5 mốc. Mọi "có ý nghĩa" ở đây là danh nghĩa (không hiệu chỉnh đa so sánh).

## 3. Lưu ý khi viết về ablation và SHAP

- **"Bỏ nhóm nào cũng không đổi" không có nghĩa là nhóm đó không chứa thông tin.** Các nhóm trùng thông tin với nhau (MAP, SBP/DBP, PP, proxy SV/CO/SVR đều tính từ huyết áp), nên khi bỏ một nhóm, các nhóm còn lại bù vào. Chỉ riêng việc bỏ MAP làm giảm độ nhạy có ý nghĩa (−0,05 ở mốc 5 phút, CI [−0,082; −0,021]).
- **16% SHAP của SV/CO/SVR không có nghĩa là cung lượng tim quan trọng.** Các proxy Liljestrand–Zander được tính trực tiếp từ huyết áp (SV = PP/(SBP + DBP), SVR = MAP/CO), nên phần lớn đóng góp của chúng vẫn là thông tin huyết áp. Không viết rằng "mô hình dùng thông tin cung lượng tim".
- SHAP ở mốc 5 phút: MAP 40%, SBP/DBP 18%, SV/CO/SVR 16%, hình dạng sóng 10%, PPV dưới 1%. Kết quả này khớp với EDA: MAP là nguồn thông tin chính.

## 4. MAP nền (cần sửa trong proposal)

Trong cohort D (không tính test, 2.899 ca), `baseline_source` phân bố như sau:

| Nguồn | Số ca | Tỷ lệ |
|---|--:|--:|
| NIBP trước khởi mê (1) | 129 | **4%** |
| NIBP đầu tiên trước rạch da (2) | 2.231 | 77% |
| MAP arterial line 5 phút đầu (3) | 469 | 16% |
| Không có (0) | 70 | 2% |

Nguồn 2 và 3 thường được đo **sau khởi mê**, vì VitalDB hiếm khi ghi dữ liệu trước khởi mê. Do đó `map_drop_pct` thực chất là mức giảm so với MAP **sau khởi mê**, và sẽ nhỏ hơn mức giảm so với MAP nền thật. Proposal đang viết "MAP nền là NIBP trước khởi mê". Cần sửa lại cho đúng, và nêu điều này khi diễn giải đặc trưng.

## 5. Cửa sổ W\* nằm ở biên

W\* = 120 là cửa sổ dài nhất đã thử. Độ nhạy trung bình của `lgbm_wave` ở mốc 5 và 10 phút lần lượt theo W = 30/60/90/120 là 0,640 / 0,652 / 0,648 / 0,660: chênh nhau nhỏ và nằm trong CI. Vì W\* rơi vào biên, và EDA cho thấy MAP bắt đầu giảm khoảng 8 phút trước đợt tụt, đã chạy thêm `lgbm_numeric` với W = 300 và 600 giây.

**Đây là phân tích thăm dò**, quyết định sau khi xem validation, báo cáo riêng (`reports/exploratory/`), và **không thay W\*** của phân tích chính.

Đặc trưng của cả W = 120, 300 và 600 được tính lại bằng cùng một hàm từ lưới 2 giây (`uc04.long_windows`). Hàm này khớp các cột W = 30–120 của `prep_v1` ở 99,9% số dòng, nên ba cửa sổ được so sánh công bằng. W = 120 tính lại cho độ nhạy 0,630 ở mốc 5 phút, so với 0,628 khi dùng cột gốc.

Kết quả (`lgbm_numeric`, bootstrap ghép cặp so với W = 120 tính lại):

| Mốc | W300 − W120: độ nhạy | W600 − W120: độ nhạy | AUROC |
|---|---|---|---|
| 5 phút | +0,003 [−0,027; +0,033] | −0,022 [−0,060; +0,010] | không đổi |
| 10 phút | −0,015 [−0,042; +0,012] | **−0,044 [−0,074; −0,017]** | không đổi |
| 15 phút | +0,006 [−0,017; +0,029] | −0,014 [−0,043; +0,017] | W300 +0,007 (CI > 0) |
| 20 phút | −0,005 [−0,025; +0,014] | **−0,023 [−0,047; −0,002]** | +0,007 / +0,008 (CI > 0) |
| 30 phút | 0,000 [−0,016; +0,019] | 0,000 [−0,024; +0,021] | +0,009 / +0,012 (CI > 0) |

**Cửa sổ dài hơn không làm tăng số đợt tụt được cảnh báo.** W = 600 còn kém hơn có ý nghĩa ở mốc 10 và 20 phút. AUROC tăng nhẹ ở các mốc dài (+0,007 đến +0,012), nhưng không chuyển thành độ nhạy cao hơn ở điểm vận hành FA ≤ 1/giờ. Vậy việc W\* = 120 nằm ở biên không làm bỏ sót một cửa sổ tốt hơn. W\* của phân tích chính giữ nguyên.

## 6. Mức tin cậy

Độ lệch chuẩn giữa các seed giờ được tính trên thang logit. Tiêu chí kiểm chứng: trong từng thập phân vị nguy cơ, nhóm "tin cậy thấp" phải có sai số calibration lớn hơn nhóm "tin cậy cao" ở ít nhất 8/10 thập phân vị, sai số gộp lớn hơn ít nhất 20%, và Brier gộp cũng tệ hơn. Với mức tin cậy ngẫu nhiên, tiêu chí này chỉ qua nhầm 1/100 lần mô phỏng. **Nếu không qua, bỏ mức tin cậy của LightGBM** và chỉ giữ độ lệch giữa các seed của DL (kiểm bằng cùng tiêu chí). Kết quả: `reports/tabular_confidence/*_cuts.json` (trường `check`).

**Kết quả: KHÔNG QUA ở cả hai mốc → bỏ mức tin cậy của LightGBM.**

| Mốc | Thập phân vị mà "thấp" kém hơn "cao" | Sai số calibration gộp (thấp / cao) | Brier gộp (thấp / cao) |
|---|--:|--:|--:|
| 5 phút | 4/10 | 0,0031 / 0,0027 | 0,040 / 0,025 |
| 10 phút | 7/10 | 0,0061 / 0,0053 | 0,073 / 0,051 |

Brier của nhóm "thấp" cao hơn chủ yếu vì nhóm này có tỷ lệ dương cao hơn trong cùng thập phân vị, không phải vì dự đoán kém chính xác hơn. Độ lệch giữa các seed của DL sẽ được kiểm bằng cùng tiêu chí trong `03_dl_finalize`.
