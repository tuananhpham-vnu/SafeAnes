# E08 — cải thiện tuần tự từng metric (recall → PPV → FA/giờ → gộp)

> **CẢNH BÁO — mọi số liệu Giai đoạn 1-4 bên dưới thuộc E08 v1 và đã bị rút lại.** Audit code ngày
> 19/09/2026 tìm được 6 lỗi làm sai lệch số liệu (train/serve skew, trộn thang xác suất, gate tính
> trên vote fraction, so sánh bị nhiễu preprocessing, chọn ngưỡng trên chính tập báo cáo, augmentation
> thêm lượng mẫu không đều) — xem [lịch sử version](../../scripts/e08/README.md). Kết quả hiện hành:
> [bảng tham số và kết quả trong README](../../README.md#e08--so-sánh-phương-pháp-v2-bản-đang-dùng) ·
> [báo cáo v2](../../reports/E08/V2_METHOD_COMPARISON.md). Nội dung dưới đây và
> [báo cáo v1](../../reports/E08/version/v1/REPORT.md) giữ lại để truy vết, **không dùng để ra quyết định**.

Nháp lập kế hoạch ngày 19/09/2026. Thay cho
cách xếp ưu tiên theo "đòn bẩy kỳ vọng" trộn nhiều metric cùng lúc ở bản trước, tài liệu này theo
đúng yêu cầu: tối ưu **từng metric riêng biệt theo thứ tự recall → PPV → FA/giờ**, đạt tốt từng
cái mới gộp lại. Không đổi model/threshold đã khóa của E05/E06 đang chờ E07
([progress](../../reports/E07/progress.json)); E08 mở sau khi E07 báo cáo, dùng cohort chưa nằm
trong global test của E07 — điều kiện này giữ nguyên từ bản trước.

## Nguyên tắc tối ưu tuần tự

- Mỗi giai đoạn chỉ tối ưu một metric, đo cả ba metric trên cùng slice dữ liệu (cùng labels,
  split, eligibility, evaluator) để so sánh công bằng — không đổi bất cứ phần nào khác của
  pipeline khi đang ở một giai đoạn.
- "Đạt" ở mỗi giai đoạn là **ngưỡng nội bộ để chuyển giai đoạn**, không phải gate cuối ở
  [mục 8 kế hoạch chính](../../UC04_RESEARCH_PLAN.md). Gate cuối chỉ áp dụng ở Giai đoạn 4.
- Cải thiện một metric gần như luôn đánh đổi các metric khác (đã thấy ở E06: TabM tăng recall
  5 phút đi kèm tăng FA/giờ, [REPORT.md](../../reports/E06/REPORT.md)). Vì vậy mỗi giai đoạn phải
  ghi lại giá trị của hai metric còn lại làm baseline cho giai đoạn sau, không bỏ qua.
- Không suy diễn số liệu paper (rare-event, ICU, sepsis, seizure) thành số liệu UC04 — chỉ dùng
  làm bằng chứng phương pháp có tác dụng đúng hướng, phải đo lại trên development VitalDB.

## Giai đoạn 0 — điều kiện mở đầu (giữ từ bản trước)

1. E07 đã báo cáo global test 516 ca/502 bệnh nhân; đọc CI ghép cặp trước khi bắt đầu Giai đoạn 1.
2. Cohort dùng cho E08 không trùng global test đã khóa của E07.
3. Không đưa IOHFuseLM vào (yêu cầu A100/GPT-4o, vi phạm ràng buộc T4 đơn/CPU đã quyết ở mục 2
   kế hoạch chính).

## Giai đoạn 1 — Recall (event sensitivity), ưu tiên cao nhất

Mục tiêu tạm: đưa event recall 5 phút / 10 phút tiến gần 90% / 85% (mục 8). Ở giai đoạn này **tạm
chấp nhận PPV và FA/giờ xấu hơn hiện tại** — vẫn đo và ghi lại, không dùng để loại phương pháp.

**Đã chạy chẩn đoán method #1 (hạ ngưỡng)** trên toàn bộ model E05/E06 bằng
[stage1_recall.py](../../scripts/e08/version/v1_sequential_stages/stage1_recall.py), tái sử dụng curve
validation đã có, không train lại. [Kết quả](../../reports/E08/version/v1/REPORT.md):

- **5 phút: trần recall của mọi model đều dưới 0,90** (cao nhất 0,826, catboost_map_only) — chỉ
  đổi ngưỡng không đủ ở horizon này; cần các phương pháp #2–4 dưới đây (loss/augmentation/
  pretraining/ensemble) hoặc waveform, không chỉ tune threshold.
- **10 phút: trần recall đã vượt 0,85 ở hầu hết model** (catboost_map_only đạt 1,000) — điểm vận
  hành đã chọn trước đó (ví dụ map_all_20260917 chỉ 0,730) đang thận trọng hơn khả năng thật của
  model; có thể nâng recall 10 phút đáng kể chỉ bằng hạ ngưỡng, chưa cần model mới.
- Cả hai horizon: FA/giờ tại trần recall là 2–4/giờ (gấp 4–8 lần ngân sách 0,5) — chi phí đã biết
  trước, xử lý ở Giai đoạn 2–3, không dùng để loại phương pháp ở đây.

Vì vậy trọng tâm còn lại của Giai đoạn 1 là **horizon 5 phút**; horizon 10 phút có thể tạm dùng
ngưỡng thấp hơn để đạt recall sàn và chuyển sang Giai đoạn 2 sớm hơn.

Phương pháp, xếp từ rẻ đến đắt:

1. **Hạ ngưỡng quyết định trực tiếp theo recall sàn** trên validation — không cần train lại,
   thử trước tiên bằng lưới ngưỡng đã có ở E06.
2. **Class-weighted loss / focal loss** cho nhãn hiếm: có bằng chứng cải thiện recall lớp thiểu
   số trong dự báo IOH bằng feature engineering + ML
   ([PMC9100985](https://pmc.ncbi.nlm.nih.gov/articles/PMC9100985/)).
3. **Oversampling/SMOTE** cho window dương hiếm, kèm mask-based/reconstruction-based augmentation
   để tăng độ đa dạng mẫu hiếm mà không nhân bản y nguyên
   ([dataset rare event sepsis onset](https://arxiv.org/pdf/2602.02930)).
4. **Temporal/event-based contrastive pretraining** trên chuỗi sinh hiệu VitalDB chưa cần nhãn,
   fine-tune sau: cải thiện cả AUROC/AP và calibration trong early-warning từ HR/BP/SpO2
   ([Early-Warning Systems from Clinical Time-Series, temporal contrastive learning](https://www.researchsquare.com/article/rs-8531226/v1.pdf); [Event-Based Contrastive Learning for Medical Time Series](https://arxiv.org/pdf/2312.10308)). VitalDB có nhiều ca/track chưa dùng hết ở E05–E07, phù hợp để pretrain không giám sát trước.
5. **Ensemble theo phép hợp (OR)** giữa các model độc lập đã có (TabM 3 seed, CatBoost, LightGBM
   monotone) — cảnh báo khi bất kỳ model nào vượt ngưỡng riêng. Biết trước sẽ tăng FA, đo riêng
   nhưng không loại ở giai đoạn này.
6. **Waveform ABP (Hatib-style, D3 kế hoạch chính)** — ứng viên tăng recall mạnh nhất về mặt lý
   thuyết (AUROC gốc 0,97/0,95) nhưng chi phí triển khai cao nhất; đặt làm nhánh phụ, không bắt
   buộc để qua giai đoạn 1.

Điều kiện qua giai đoạn: chọn một tổ hợp (model + threshold + loss/augmentation) đạt recall cao
nhất trên validation trong ngân sách hợp lý; ghi nhận PPV và FA/giờ tại đúng điểm vận hành đó làm
baseline cho Giai đoạn 2.

## Giai đoạn 2 — PPV, sau khi recall đã chấp nhận được

Mục tiêu tạm: từ điểm vận hành recall cao đã chọn ở Giai đoạn 1, tăng PPV bằng các lớp lọc/hiệu
chỉnh phía sau, **không hạ recall đã đạt xuống dưới biên dung sai** (ví dụ không giảm quá 5 điểm
phần trăm so với Giai đoạn 1 — chốt số cụ thể khi có kết quả validation thật, không chốt trước
khi thấy dữ liệu).

**Đã chạy phương pháp #2 (đồng thuận ensemble)** bằng
[stage2_ppv.py](../../scripts/e08/version/v1_sequential_stages/stage2_ppv.py), so hai nhóm: 4 model tương
quan cao (TabM 3 seed + LightGBM cùng feature) và 6 model đa dạng (thêm MAP + CatBoost).
[Kết quả](../../reports/E08/version/v1/REPORT.md):

- Nhóm 4 model tương quan cao: đồng thuận chỉ nâng PPV vài điểm phần trăm (5 phút 0,064→0,131;
  10 phút 0,118→0,141) vì các model học cùng dữ liệu/đặc trưng, đồng thuận cả ở cảnh báo giả.
- **Nhóm 6 model đa dạng (thêm MAP + CatBoost) tăng PPV rõ rệt hơn**: 5 phút lên đến 0,205 tại
  đồng thuận 6/6 (gấp ~1,6 lần nhóm 4 model), **recall không đổi (0,739)**; 10 phút PPV 0,157 tại
  6/6, **recall vẫn giữ 0,958** — không đánh đổi recall lấy PPV ở đây.
- Vẫn còn rất xa mục tiêu PPV 0,60–0,70: đây là cải thiện thật (đa dạng kiến trúc > số lượng
  model), nhưng một mình chưa đóng được gate. FA/giờ tại điểm này còn 1,15–2,89, xử lý ở Giai
  đoạn 3.

Phương pháp còn lại, chưa chạy:

1. **Tiered/two-stage risk stratification**: tầng 1 giữ nguyên độ nhạy cao của Giai đoạn 1, tầng
   2 thêm điều kiện/nâng ngưỡng chỉ cho các trường hợp tầng 1 đã gắn cờ. AI-TEW báo tăng PPV từ
   9,8–18,8% lên 32,5–40,5% qua tối ưu ngưỡng phân tầng, giữ NPV >98%
   ([AI-TEW, npj Digital Medicine 2026](https://www.nature.com/articles/s41746-026-02522-8)).
   SENTINEL giảm 58% số cảnh báo ở cùng mức sensitivity bằng cảnh báo hai tầng có ràng buộc thời
   gian ([SENTINEL, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2590005626004960)).
2. **Yêu cầu đồng thuận ensemble (AND)** giữa các seed/model đã có trước khi phát cảnh báo — đối
   lập trực tiếp với phép OR ở Giai đoạn 1, dùng ở đây để nâng precision thay vì recall.
3. **Điểm neo lâm sàng để đánh giá mức PPV đạt được có ý nghĩa hay chưa**: HPI-85 popup báo PPV
   ≈55% so với MAP<72 mmHg PPV ≈30% trong một nghiên cứu quan sát tiến cứu
   ([HPI vs MAP, EJA 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12052080/)) — dùng để so sánh
   mức độ, không suy diễn cùng cohort hay cam kết đạt cùng số.
4. Đọc phản biện ["False positives or unrecognized physiology?"](https://pubmed.ncbi.nlm.nih.gov/40598867/):
   nhiều mô hình MAP-based hiện có PPV thấp mang tính hệ thống (khoảng 4/5 dự báo dương là false
   alarm) — dùng để đặt kỳ vọng thực tế cho mục tiêu PPV giai đoạn này, không coi PPV thấp là lỗi
   riêng của implementation UC04.

Điều kiện qua giai đoạn: chọn được lớp lọc PPV giữ recall trong biên dung sai của Giai đoạn 1; ghi
nhận FA/giờ tại điểm này làm baseline cho Giai đoạn 3.

## Giai đoạn 3 — FA/giờ, sau khi PPV đã chấp nhận được

Mục tiêu tạm: giảm FA/giờ về gần ngân sách 0,5, ưu tiên kỹ thuật có **đảm bảo hình thức** hơn là
chỉnh tay ngưỡng/cooldown.

Phương pháp:

1. **Conformal prediction / risk-controlling calibration**: hậu xử lý, không cần train lại model,
   khống chế tỷ lệ false alarm ở mức chọn trước với đảm bảo thống kê marginal. Đã báo giảm FA 92%
   trong dự báo cơn động kinh ([risk-controlling calibration, PMC10543660](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10543660/))
   và 57% trong sepsis ngoại viện ([conformal prediction sepsis, PMC11601686](https://pmc.ncbi.nlm.nih.gov/articles/PMC11601686/)).
   Ưu tiên thử trước vì rẻ và có guarantee hình thức, không phải chỉ heuristic.
2. **ML-based false-alarm suppression từ waveform** (nếu Giai đoạn 1 đã mở nhánh ABP): mô hình
   lọc false alarm chuyên biệt trên dạng sóng đạt loại bỏ 86–100% false alarm ở một số loại cảnh
   báo ICU mà không mất true positive
   ([reduction of false alarms in ICU, npj Digital Medicine 2019](https://www.nature.com/articles/s41746-019-0160-7));
   nguyên lý tương tự cần kiểm chứng lại cho false alarm IOH, không suy diễn thẳng từ arrhythmia.
3. **Contrastive-learning-based false alarm suppression** cho dữ liệu mất cân bằng
   ([No More False Alert, Sensors 2026](https://doi.org/10.3390/s26113561)).
4. Cooldown/k-of-n persistence đã có ở E06 — dùng làm đối chứng so sánh, không phải phương pháp
   mới của giai đoạn này.

Điều kiện qua giai đoạn: FA/giờ giảm về gần ngân sách 0,5 mà recall/PPV vẫn trong biên dung sai đã
ghi nhận ở Giai đoạn 1–2.

## Giai đoạn 4 — gộp lại, chỉ mở sau khi 1–3 mỗi cái đều có phương pháp tốt riêng

- Ghép: phương pháp thắng ở Giai đoạn 1 (recall) → lớp lọc thắng ở Giai đoạn 2 (PPV) →
  conformal/suppression thắng ở Giai đoạn 3 (FA) thành một pipeline duy nhất.
- Đánh giá lại đồng thời cả 5 gate ở mục 8 kế hoạch chính trên cùng validation, sau đó trên
  holdout/cohort E07 chưa dùng.
- Nếu gộp làm mất lợi ích của một giai đoạn (ví dụ lớp lọc PPV kéo recall vượt biên dung sai),
  quay lại điều chỉnh thứ tự áp dụng hoặc nới biên dung sai một cách có ghi nhận — không lặng lẽ
  đổi cấu hình rồi báo cáo như thể đã theo đúng kế hoạch.
- Vẫn tuân thủ nguyên tắc đã khóa ở mục 8: không chọn ngưỡng/model bằng kết quả test.

## Việc không đưa vào E08 (giữ từ bản trước)

- IOHFuseLM (yêu cầu A100/GPT-4o) — không phù hợp ràng buộc T4 đơn/CPU của dự án.
- Không hạ gate mục 8 để "đạt" ở Giai đoạn 4 — nếu hết cả 4 giai đoạn vẫn chưa đạt, báo cáo thiếu
  gì và kế hoạch tiếp theo, không nới ngưỡng.
