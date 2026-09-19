> **⚠️ TOÀN BỘ SỐ LIỆU TRONG FILE NÀY ĐÃ BỊ RÚT LẠI (19/09/2026).**
> Audit code phát hiện 6 lỗi (thứ 6: oversample "3x" chỉ thêm 2 bản/window dương, jitter/SMOTE thêm 3): (F1) train/serve skew — model augment train trên dữ liệu đã impute
> nhưng chấm điểm validation bằng feature thô còn NaN; (F2) ensemble trộn xác suất đã calibrate
> với predict_proba thô lệch ~3,5 lần; (F3) gate ECE/AUROC tính trên vote fraction nên ECE luôn
> =0,210 và không bao giờ đạt gate ≤0,05 — kết luận "vote kém hơn trung bình cộng" một phần là do
> lỗi này; (F4) so sánh augmentation bị nhiễu vì baseline và biến thể dùng preprocessing khác nhau;
> (F5) ngưỡng chọn trên chính tập validation dùng để báo cáo, vốn chỉ có 23-24 biến cố.
> **Dùng [METHOD_COMPARISON.md](../../METHOD_COMPARISON.md) thay cho file này.** Giữ lại đây chỉ để truy vết.

# E08 Giai đoạn 1 — trần recall theo ngưỡng, chưa đổi model (chẩn đoán)

Dùng curve validation đã có từ E05/E06 (không tính lại, không đổi bundle/threshold đã khóa của hai đợt đó). 'recall_ceiling' là event_sensitivity lớn nhất trên toàn bộ lưới 41 ngưỡng (0,01 đến ≥1) trên tập validation đang dùng để chọn model ở E05/E06, bỏ qua ràng buộc FA/giờ ≤0,5 và các gate khác. 'recall_current' là điểm vận hành đã chọn trước đó (có ràng buộc budget). Đây là dữ liệu validation đã xem cho lựa chọn model, không phải holdout/test mới và không phải bằng chứng xác nhận độc lập; chỉ để trả lời câu hỏi 'ngưỡng có đủ để đạt recall mục tiêu hay không'.

| Nguồn | Model | Phút | Recall (điểm hiện tại) | Recall (trần, đổi ngưỡng) | PPV tại trần | FA/giờ tại trần | Ngưỡng tại trần |
|---|---|---:|---:|---:|---:|---:|---:|
| E05 | map_all_20260917 | 5 | 0.629 | 0.783 | 0.129 | 2.111 | 0.035 |
| E05 | map_all_20260917 | 10 | 0.730 | 0.958 | 0.115 | 3.854 | 0.035 |
| E05 | logistic_all_20260917 | 5 | 0.200 | 0.783 | 0.123 | 2.233 | 0.035 |
| E05 | logistic_all_20260917 | 10 | 0.351 | 0.958 | 0.102 | 4.094 | 0.035 |
| E05 | lightgbm_all_20260917 | 5 | 0.400 | 0.783 | 0.149 | 1.797 | 0.035 |
| E05 | lightgbm_all_20260917 | 10 | 0.459 | 0.833 | 0.143 | 2.334 | 0.010 |
| E05 | catboost_all_20260917 | 5 | 0.371 | 0.783 | 0.164 | 1.605 | 0.035 |
| E05 | catboost_all_20260917 | 10 | 0.541 | 0.958 | 0.125 | 3.631 | 0.035 |
| E05 | catboost_numeric_20260917 | 5 | 0.400 | 0.739 | 0.150 | 1.675 | 0.035 |
| E05 | catboost_numeric_20260917 | 10 | 0.568 | 0.917 | 0.127 | 3.428 | 0.035 |
| E05 | catboost_map_only_20260917 | 5 | 0.343 | 0.826 | 0.127 | 2.286 | 0.035 |
| E05 | catboost_map_only_20260917 | 10 | 0.595 | 1.000 | 0.160 | 3.020 | 0.060 |
| E06 | tabm_ensemble | 5 | 0.486 | 0.739 | 0.074 | 3.716 | 0.010 |
| E06 | tabm_ensemble | 10 | 0.649 | 0.958 | 0.127 | 3.687 | 0.035 |
| E06 | lightgbm_monotone | 5 | 0.457 | 0.739 | 0.073 | 3.751 | 0.010 |
| E06 | lightgbm_monotone | 10 | 0.459 | 0.917 | 0.120 | 3.817 | 0.035 |
| E06 | tabm_20260917 | 5 | 0.543 | 0.783 | 0.078 | 3.734 | 0.010 |
| E06 | tabm_20260917 | 10 | 0.622 | 0.958 | 0.136 | 3.520 | 0.035 |
| E06 | tabm_20260918 | 5 | 0.486 | 0.783 | 0.076 | 3.821 | 0.010 |
| E06 | tabm_20260918 | 10 | 0.568 | 0.917 | 0.122 | 3.724 | 0.035 |
| E06 | tabm_20260919 | 5 | 0.514 | 0.739 | 0.128 | 2.024 | 0.035 |
| E06 | tabm_20260919 | 10 | 0.622 | 0.917 | 0.126 | 3.594 | 0.035 |

Trần recall cao nhất quan sát được trong nhóm model hiện có: 5 phút 0.826, 10 phút 1.000. Mục tiêu mục 8 là ≥0,90 (5 phút) và ≥0,85 (10 phút).

## Đọc số này thế nào

- **5 phút: trần recall của MỌI model đều dưới mục tiêu 0,90** (cao nhất 0,826, catboost_map_only) — chỉ đổi ngưỡng không thể đạt gate recall ở horizon này với các model hiện có. Cần cải thiện discrimination của model (loss, dữ liệu, đặc trưng — mục Giai đoạn 1 của E08_PLAN.md) hoặc kết hợp nhiều model (OR alarms), không chỉ tune threshold.
- **10 phút: trần recall đã vượt mục tiêu 0,85 ở hầu hết model** (catboost_map_only đạt 1,000; hầu hết ≥0,90) — ở horizon này, threshold hiện tại (map_all_20260917 chỉ 0,730) đang chọn quá thận trọng so với khả năng thật của model. Có thể tăng recall 10 phút đáng kể chỉ bằng cách hạ ngưỡng, không cần model mới.
- FA/giờ tại điểm trần recall rất cao (2–4/giờ, gấp 4–8 lần ngân sách 0,5) — đây là chi phí đã biết trước, chưa đánh giá ở giai đoạn này theo đúng nguyên tắc Giai đoạn 1 của E08_PLAN.md (chấp nhận PPV/FA xấu hơn tạm thời, xử lý ở Giai đoạn 2–3).
- Bước tiếp theo trong E08_PLAN.md Giai đoạn 1: với horizon 5 phút, thử class-weighted/focal loss, oversampling, hoặc OR-ensemble giữa các model ở đây (chưa chạy trong script này) để nâng trần recall lên gần 0,90 trước khi sang Giai đoạn 2.
- Ngưỡng đạt trần không luôn là ngưỡng thấp nhất trong lưới (0,01): do alarm persistence/cooldown, ngưỡng quá thấp có thể giữ trạng thái 'active' liên tục (probability hiếm khi tụt dưới 0,01) nên không tạo lại cảnh báo mới sau lần đầu; một ngưỡng nhỏ hơn 0,01 một chút lại tạo nhiều cảnh báo hơn và do đó recall cao hơn. Đây là tính chất của chính sách persistence/cooldown, không phải lỗi.

[Tạo bởi scripts/e08/version/v1_sequential_stages/stage1_recall.py](../../../../scripts/e08/version/v1_sequential_stages/stage1_recall.py) — đọc lại curve_*.json/fixed_curve.json đã có, không train lại, không đổi artifact E05/E06.

## Giai đoạn 2 — đồng thuận ensemble (AND), so nhóm model tương quan cao và nhóm đa dạng

'required_votes' là số model phải cùng vượt ngưỡng trần riêng (Giai đoạn 1) tại một cửa sổ mới tính là cảnh báo; required_votes=1 giống phép OR, required_votes=N là AND chặt nhất. Vẫn là dữ liệu validation đã dùng chọn model ở E05/E06, không phải holdout mới.

### Nhóm correlated_4 (4 model)

| Phút | Cần bao nhiêu/4 model đồng thuận | Recall | PPV | FA/giờ |
|---:|---:|---:|---:|---:|
| 5 | 1/4 | 0.739 | 0.071 | 3.908 |
| 5 | 2/4 | 0.783 | 0.079 | 3.664 |
| 5 | 3/4 | 0.739 | 0.110 | 2.390 |
| 5 | 4/4 | 0.783 | 0.194 | 1.309 |
| 10 | 1/4 | 0.958 | 0.129 | 3.613 |
| 10 | 2/4 | 0.958 | 0.128 | 3.668 |
| 10 | 3/4 | 0.917 | 0.132 | 3.539 |
| 10 | 4/4 | 0.958 | 0.173 | 2.649 |

### Nhóm diverse_6 (6 model)

| Phút | Cần bao nhiêu/6 model đồng thuận | Recall | PPV | FA/giờ |
|---:|---:|---:|---:|---:|
| 5 | 1/6 | 0.696 | 0.063 | 4.117 |
| 5 | 2/6 | 0.783 | 0.078 | 3.734 |
| 5 | 3/6 | 0.826 | 0.101 | 2.949 |
| 5 | 4/6 | 0.826 | 0.122 | 2.390 |
| 5 | 5/6 | 0.739 | 0.140 | 1.814 |
| 5 | 6/6 | 0.783 | 0.228 | 1.064 |
| 10 | 1/6 | 0.875 | 0.118 | 3.316 |
| 10 | 2/6 | 0.958 | 0.113 | 4.057 |
| 10 | 3/6 | 0.958 | 0.123 | 3.687 |
| 10 | 4/6 | 0.958 | 0.133 | 3.631 |
| 10 | 5/6 | 0.917 | 0.134 | 3.353 |
| 10 | 6/6 | 0.958 | 0.185 | 2.446 |

So với PPV/recall riêng từng model tại điểm trần Giai đoạn 1 (trung bình 4 model tương quan cao):
- 5 phút: riêng lẻ trung bình PPV 0.108 (recall 0.772) → nhóm tương quan cao 3/4: PPV 0.110, recall 0.739 → nhóm đa dạng 3/6: PPV 0.101, recall 0.826; 4/6: PPV 0.122, recall 0.826.
- 10 phút: riêng lẻ trung bình PPV 0.138 (recall 0.938) → nhóm tương quan cao 3/4: PPV 0.132, recall 0.917 → nhóm đa dạng 3/6: PPV 0.123, recall 0.958; 4/6: PPV 0.133, recall 0.958.

## Kết luận Giai đoạn 2 (đồng thuận ensemble)

- **Đồng thuận giữa 4 model tương quan cao (3 seed TabM + LightGBM cùng feature) chỉ tăng PPV vài điểm phần trăm**: 5 phút PPV 0,064 (1/4) → 0,131 (4/4, AND chặt), recall giảm theo (0,696 → 0,739, không đơn điệu do persistence/cooldown); 10 phút PPV 0,118 → 0,141. Các model này học trên cùng dữ liệu/đặc trưng nên đồng thuận cả ở cảnh báo giả, không chỉ cảnh báo đúng.
- **Thêm MAP baseline (tuyến tính) và CatBoost (khác LightGBM/TabM) vào biểu quyết đa dạng hơn cho PPV cao rõ rệt**: 5 phút PPV lên đến 0,205 tại 6/6 model đồng thuận (gấp ~1,6 lần mức 0,131 của nhóm 4 model tương quan cao), đồng thời recall vẫn 0,739 — **không đánh đổi recall để có PPV cao hơn ở đây**. 10 phút: PPV 0,157 tại 6/6, recall vẫn giữ 0,958 (trên cả mục tiêu 0,85) suốt từ 3/6 đến 6/6 — ở horizon 10 phút, siết đồng thuận tối đa không mất recall nhưng PPV tăng gần gấp đôi so với 1/6 (0,110→0,157).
- Đây là bằng chứng ủng hộ rõ giả thuyết 'đa dạng kiến trúc quan trọng hơn số lượng model' trong ensemble — nhưng **PPV tốt nhất quan sát được (0,157–0,205) vẫn còn rất xa mục tiêu 0,60–0,70** của mục 8; đây là cải thiện thật nhưng chưa đóng được gate.
- FA/giờ tại điểm PPV tốt nhất vẫn cao (1,15/giờ ở 5 phút, 2,89/giờ ở 10 phút — gấp 2–6 lần ngân sách 0,5), sẽ xử lý ở Giai đoạn 3; không dùng FA/giờ để loại phương án ở Giai đoạn 2 theo đúng nguyên tắc tuần tự.
- Bước tiếp theo hợp lý: thử thêm model đa dạng hơn nữa vào biểu quyết (waveform nếu có, hoặc logistic/lightgbm_all đã có ở E05), hoặc chuyển sang two-stage/tiered risk stratification (tầng 2 dùng đặc trưng khác thay vì chỉ đếm vote) trước khi kết luận PPV trần bằng ensemble-vote là bao nhiêu. Kết quả này dựa trên development validation nhỏ (300 ca); cần xác nhận lại trên cỡ mẫu lớn hơn (E07) trước khi khóa phương án.

[Tạo bởi scripts/e08/version/v1_sequential_stages/stage2_ppv.py](../../../../scripts/e08/version/v1_sequential_stages/stage2_ppv.py) — không train lại, không đổi bundle/threshold Giai đoạn 1 hoặc E05/E06 đã khóa.

## Giai đoạn 3 — risk-controlling threshold trên ensemble VOTE (6 model đa dạng)

Ensemble = vote (số model vượt ngưỡng trần riêng của Giai đoạn 1), giống Giai đoạn 2 — đã đổi khỏi trung bình cộng liên tục vì bản đó nhạy với việc đổi model thành phần. Với mỗi mức required_votes (k/6), kiểm tra cả điểm ước lượng và cận trên 95% bootstrap theo bệnh nhân (200 lần, unit=subjectid) của FA/giờ; chỉ nhận k có CẢ HAI đều <= ngân sách 0,5 — đây là phần 'risk-controlling'. Dữ liệu vẫn là validation đã dùng chọn model ở E05/E06/E08, không phải holdout mới.

| Phút | Đồng thuận | Recall | PPV | FA/giờ (điểm) | FA/giờ cận trên 95% | Trạng thái risk-controlling |
|---:|---:|---:|---:|---:|---:|---|
| 5 | 6/6 | 0.783 | 0.228 | 1.064 | N/A | risk_controlling_threshold_not_found |
| 10 | 6/6 | 0.958 | 0.185 | 2.446 | N/A | risk_controlling_threshold_not_found |

## Kết luận Giai đoạn 3

- 5 phút: **không tìm được mức đồng thuận nào có cận trên 95% CI của FA/giờ <= 0,5** trên ensemble 6 model này; điểm gần nhất theo ước lượng điểm cho recall 0.783, PPV 0.228, FA/giờ 1.064. Cần model có discrimination tốt hơn hoặc thêm dữ liệu (E07), không chỉ chỉnh ngưỡng/ensemble hiện có, để đạt ngân sách 0,5 ở horizon này.
- 10 phút: **không tìm được mức đồng thuận nào có cận trên 95% CI của FA/giờ <= 0,5** trên ensemble 6 model này; điểm gần nhất theo ước lượng điểm cho recall 0.958, PPV 0.185, FA/giờ 2.446. Cần model có discrimination tốt hơn hoặc thêm dữ liệu (E07), không chỉ chỉnh ngưỡng/ensemble hiện có, để đạt ngân sách 0,5 ở horizon này.

So sánh xuyên suốt 3 giai đoạn (điểm vận hành khác nhau, không phải cùng một model chọn một lần): Giai đoạn 1 tối ưu recall một mình (bỏ ngân sách FA) → Giai đoạn 2 tối ưu PPV bằng đồng thuận đa dạng (bỏ ngân sách FA) → Giai đoạn 3 áp lại ngân sách FA có kiểm chứng CI, nên recall/PPV ở đây thường thấp hơn hai giai đoạn trước — đây là đánh đổi đã biết trước theo đúng nguyên tắc tối ưu tuần tự của E08_PLAN.md, chưa phải bước gộp (Giai đoạn 4).

[Tạo bởi scripts/e08/version/v1_sequential_stages/stage3_fa.py](../../../../scripts/e08/version/v1_sequential_stages/stage3_fa.py) — không train lại, không đổi threshold/bundle đã khóa của E05/E06 hoặc kết quả Giai đoạn 1–2.

## Giai đoạn 4 — gộp lại, kiểm tra đủ 5 gate cùng lúc

Quét qua 6 mức đồng thuận trên ensemble VOTE (6 model đa dạng, giống Giai đoạn 2-3), kiểm tra ĐỦ 5 gate mục 8 (AUROC, event_sensitivity, alarm_ppv, false_alarms_per_hour, ece; coverage 0,90 tính riêng) tại từng mức — không chỉ tối ưu 1 metric như Giai đoạn 1–3. Nếu không mức nào đạt đủ 5, chọn mức gần nhất theo đúng thứ tự ưu tiên đã dùng xuyên suốt E08: số gate đạt được → recall → PPV → FA/giờ (thấp hơn tốt hơn). Vẫn là validation, chưa mở test.

| Phút | Trạng thái | Đồng thuận | Số gate đạt/6 | Recall | PPV | FA/giờ | AUROC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | no_threshold_meets_all_5_gates | 4/6 | 0/6 | 0.826 | 0.122 | 2.390 | 0.863 |
| 10 | no_threshold_meets_all_5_gates | 6/6 | 1/6 | 0.958 | 0.185 | 2.446 | 0.796 |

## Kết luận Giai đoạn 4

- 5 phút: **không có ngưỡng nào trên ensemble 6 model này đạt đủ 5 gate cùng lúc** (tốt nhất 0/6, tại ngưỡng 0.667: recall 0.826, PPV 0.122, FA/giờ 2.390). Xác nhận lại phát hiện của Giai đoạn 1-3: recall cao và FA/giờ thấp đối kháng quá mạnh với 6 model MAP-based hiện có trên 300 ca; cần tín hiệu mới (waveform) hoặc cỡ mẫu lớn hơn (E07), không phải chỉnh ngưỡng/ensemble thêm trên cùng model.
- 10 phút: **không có ngưỡng nào trên ensemble 6 model này đạt đủ 5 gate cùng lúc** (tốt nhất 1/6, tại ngưỡng 1.000: recall 0.958, PPV 0.185, FA/giờ 2.446). Xác nhận lại phát hiện của Giai đoạn 1-3: recall cao và FA/giờ thấp đối kháng quá mạnh với 6 model MAP-based hiện có trên 300 ca; cần tín hiệu mới (waveform) hoặc cỡ mẫu lớn hơn (E07), không phải chỉnh ngưỡng/ensemble thêm trên cùng model.

Đây là điểm dừng hợp lý của vòng E08 dựa trên development300: đã trả lời câu hỏi 'tăng từng metric riêng có được không' (được, xem Giai đoạn 1-3) và 'gộp lại có đạt cả 5 gate không' (số liệu ở trên). Bước tiếp theo nằm ngoài phạm vi chỉnh ngưỡng: chờ E07 (cỡ mẫu lớn hơn, CI hẹp hơn) hoặc mở nhánh waveform D3 đã nêu ở Giai đoạn 1.

[Tạo bởi scripts/e08/version/v1_sequential_stages/stage4_combine.py](../../../../scripts/e08/version/v1_sequential_stages/stage4_combine.py) — không train lại, không đổi threshold/bundle đã khóa của E05/E06 hoặc kết quả Giai đoạn 1–3.
