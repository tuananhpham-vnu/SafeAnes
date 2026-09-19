# Protocol numeric v1 — triển khai UC04 v0.3

Trạng thái: thử nghiệm kỹ thuật, chưa phải mô hình đạt mục tiêu. Các ID nguồn nằm trong [SOURCES.md](SOURCES.md). Khi thay nhãn/feature, tăng `Protocol.version`, tạo dataset mới và giữ hash cũ.

## Cohort và chia tập

Theo S01/S02: tuổi ≥18, `ane_type=General`, có `Solar8000/ART_MBP`, `opstart/opend` hợp lệ, đủ history + horizon lớn nhất + duration. Chỉ phân tích thời gian phẫu thuật, chưa đánh giá riêng khởi mê. Manifest lưu tất cả ca và lý do loại đầu tiên.

SHA-256(seed, subjectid) chia xấp xỉ 70% train, 7,5% calibration, 7,5% validation, 15% test. Đây là tỷ lệ kỳ vọng, không ép số lượng chính xác. Thêm ca không làm thay đổi split cũ; mọi lần mổ của một người cùng split.

`fetch-pilot` chỉ tải tín hiệu global train. Pilot tách tiếp theo bệnh nhân: 60% fit, 15% calibration, 10% validation, 15% pilot_test. **pilot_test vẫn thuộc global train**, chỉ dùng thăm dò/sửa pipeline; không phải final test. Release 0.2 không tự mở final test.

## Input và nhãn tách riêng

Input lấy bản ghi cuối có timestamp ≤ t, tuổi tối đa 30 giây; giữ NaN nếu thiếu/invalid, không backfill. Bounds trong config là kiểm tra lỗi đo thô, chưa thay artifact detector waveform. MAP thấp vẫn được giữ, riêng BP/HR bằng 0 coi là invalid. Input grid 2 giây, history `(t−600,t]`, thống kê 60/300/600 giây, cadence 30 giây. Static chỉ gồm age/BMI/ASA; không dùng tổng liều, dịch hay mất máu cuối ca.

Nhãn grid 1 giây: ô `[s,s+1)` chỉ hợp lệ khi được bao phủ bởi các khoảng giữa hai phép đo hữu hạn, mỗi khoảng ≤10 giây. Khoảng mất dài bị unknown toàn bộ; không kéo dài bản ghi cuối. Nếu một ô chứa nhiều giá trị, dùng maximum: toàn ô phải thấp mới tính là thấp. Đây là cách bảo thủ, có thể lệch onset/duration khoảng 1 giây; cần sensitivity analysis.

IOH là MAP <65 trong ít nhất 60 ô liên tiếp hợp lệ (S08 cho định nghĩa, discretization là của dự án). Gộp hai đoạn nếu khoảng giữa <120 giây và không có unknown.

Eligibility online: MAP còn mới ≤10 giây; 120 giây vừa qua liên tục ≥65 theo dữ liệu causal; ≥80% history MAP có giá trị. Cả đoạn thấp ngắn hoặc missing cũng khởi động lại recovery, bảo thủ hơn chỉ chờ sau IOH xác nhận. Không dùng nhãn tương lai để quyết định eligibility.

`y_300`, `y_600`: có onset mới trong `(t,t+h]`. Chỉ gán 0/1 khi toàn khoảng tương lai tới `t+h+60` đủ dữ liệu; nếu không gán `-1` censored, kể cả khi đã thấy biến cố trước khoảng mất. Quy tắc đối xứng tránh giữ mẫu dương thiếu follow-up trong khi loại mẫu âm tương tự; complete-follow-up vẫn có thể gây selection bias và phải báo cáo độ phủ.

Giữ cả decision ineligible/censored để replay biết lúc reset. Feature chỉ từ quá khứ; nhãn hồi cứu không được đưa vào estimator hoặc alarm state.

## Fit và calibration

Baseline: -MAP; logistic MAP/slope/std 5 phút; HistGradientBoosting 143 numeric/static feature; LightGBM tùy chọn. Imputer/scaler chỉ fit trên fit patients. Sigmoid calibration (scaler + logistic raw score) chỉ fit calibration patients. Ngưỡng chọn validation patients; không random internal validation theo window. Thiếu một lớp ở fit/calibration thì báo thiếu dữ liệu, không reshuffle để tìm split đẹp.

Hai horizon baseline độc lập; chưa ép p10≥p5 hay hợp nhất cảnh báo. Không trình bày chúng như một phân phối sống còn chung. Mốc multi-horizon DL sẽ kiểm tra tính nhất quán.

## Replay và metric

Vượt ngưỡng 2 lần liên tiếp thì emit tại lần thứ hai. High kéo dài tạo một episode. Phải trở về dưới ngưỡng hoặc reset do ineligible/missing/gap mới mở episode tiếp; cooldown 300 giây từ lần emit trước. Lead time tính từ emit.

Alarm ghép onset gần nhất thỏa `alarm_time < onset ≤ alarm_time+h`. Nhiều alarm có thể ghép một event: PPV đếm episode, event sensitivity đếm event một lần. Nhãn unknown lúc emit thì alarm censored, không ép false. Future label không ngăn emit.

Event có cơ hội dự báo khi có ít nhất một decision eligible, label=1 trong horizon trước onset. Báo cáo số event đủ điều kiện và tổng event; `event_sensitivity_all` dùng tổng event để thấy ảnh hưởng abstention. `early_5min_sensitivity` dùng mọi event đủ điều kiện làm mẫu số và chỉ tính alarm thực sự báo trước ≥300 giây. Horizon không tự chứng minh thời gian báo trước.

**FA/hour v0.1 là xấp xỉ theo lưới quyết định 30 giây.** Mỗi decision eligible, nhãn biết và probability có giá trị đóng góp tối đa 30 giây, cắt tại opend. Eligibility có thể đổi trong ô; đây chưa phải tích phân từng giây. Báo cáo `evaluable_hours`, `prediction_coverage`, `censored_alarms`. Cần nâng cấp exposure trước nghiệm thu cuối.

AUROC/AP thiếu một lớp trả null; ECE có 10 bin bằng nhau trên [0,1]. Brier/prevalence báo cùng. Bootstrap theo subjectid, giữ mọi ca/window của người được lấy; công bố số replicate đủ hai lớp. Pilot mặc định 200 lần. Các ngưỡng nghiệm thu vẫn là mục tiêu đề xuất, không phải kết quả.

Threshold chọn từ grid cố định trên validation, ưu tiên đạt toàn bộ tiêu chí; nếu không có thì tối ưu sensitivity trong giới hạn FA/hour và ghi `exploratory_fallback_targets_unmet`. Fallback không có nghĩa đạt yêu cầu.

## Bổ sung triển khai numeric trong release 0.2

Nhãn, split và evaluator v1 được giữ để so sánh với baseline. `sequences.py` dùng chung `sample_case` với feature pipeline; cửa sổ memmap 300 bước. TCN 7 residual block và Transformer 2 layer theo patch được huấn luyện bằng BCE hai horizon, bỏ riêng các nhãn -1. Scaler chỉ fit ca fit. Checkpoint chọn bằng BCE validation; calibration trên bệnh nhân calibration; ngưỡng trên validation. Hai tác vụ tuning/threshold còn dùng chung validation trong pilot và cần tách/OOF trước vòng xác nhận.

Đầu ra DL có `logit600=logit300+softplus(delta)`. Calibration dùng temperature dương và bias chung, bảo toàn p600≥p300; không dùng sigmoid độc lập như baseline. Đây là adaptation dự án, không phải tái lập chính xác một paper. Chưa đánh giá một luồng alarm hợp nhất hai horizon.

CUDA dùng FP16 autocast và GradScaler; CPU dùng FP32. Checkpoint lưu model, optimizer, scaler, RNG, config, split và source/data hashes; run hoàn tất không ghi đè. Xem [runbook](SEQUENCE_RUNBOOK.md) để thực thi và resume.

`reporting.py` bổ sung reliability bins, slope/intercept mô tả, phân nhóm tuổi/ASA theo toàn ca và hình replay từ output thật. Không dùng diagnostic test để sửa calibrator. Protocol v1 còn xấp xỉ exposure và coverage trên lưới decision; chưa cung cấp exposure từng giây hoặc CI riêng cho lead-time/subgroup.

## Các mốc còn thiếu trước nghiệm thu

V0.3 bổ sung hai backbone Inception-style và TimesNet adaptation cùng ensemble bốn checkpoint. Protocol nhãn/eligibility/alarm không đổi. Ensemble trung bình **logits thô**, không trung bình các điểm đã qua calibrator riêng; trọng số đều được khóa trước chạy, rồi fit một calibrator chung bằng calibration patients. Không dùng nhãn validation để học trọng số ensemble và không dùng pilot_test để chọn thành viên. Chi tiết và nguồn tại [kế hoạch v0.3](versions/V0_3_PLAN.md), [review](SOURCES.md#review-e03) và [hướng dẫn version](../README.md#release).

Train/benchmark T4, waveform artifact detector, kiểm định ngoài, nhãn cơ chế và khuyến nghị điều trị chưa hoàn tất. Trước mở global final test: rà soát nhãn với chuyên gia, xác nhận cohort không tim, audit censored/exclusion, hoàn thiện exposure, chốt model/threshold/protocol/hash. Sửa sau khi mở test cần holdout mới hoặc công bố exploratory.
