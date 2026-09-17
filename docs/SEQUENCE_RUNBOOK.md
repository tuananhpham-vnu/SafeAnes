# Chạy nghiên cứu numeric UC04

Phiên bản 0.2 bổ sung TCN, Transformer theo patch, calibration bảo toàn thứ tự horizon, checkpoint/resume và phát lại ca. Phạm vi thực thi là **pilot trong global train**. Mục tiêu hiệu năng tại mục 8 của kế hoạch vẫn giữ nguyên.

## 1. Môi trường và dữ liệu

Python ≥3.11. Dùng môi trường riêng và cài:

```sh
python -m pip install -e ".[dev,boosting,deep,plots]"
python -m pytest -q
```

Trên Kaggle, sử dụng PyTorch có CUDA đã có trong môi trường; không thay bằng wheel CPU. Xác minh `torch.cuda.is_available()` và `torch.cuda.get_device_name(0)`; chọn một T4 trong cài đặt notebook. Cấu hình `device=auto` có thể chạy CPU nếu không có CUDA, nên phải đọc `environment.json` khi diễn giải benchmark.

Local Windows có thể thêm dependencies riêng của workspace bằng `$env:PYTHONPATH = "$PWD/src;$PWD/.local_deps"`. Thư mục `.local_deps` không phải môi trường phân phối; phiên bản chính xác của lần chạy được lưu trong artifacts. Không đưa thư mục này lên Kaggle.

```sh
python -m safeanes.cli fetch-pilot --cases 60 --workers 4
python -m safeanes.cli build-pilot --out data/pilot_v1
python -m safeanes.cli build-sequences --dataset data/pilot_v1 --root data/vitaldb --out data/sequences_v1
```

Bỏ qua các lệnh đã hoàn tất nếu thư mục output tương ứng tồn tại. Raw cache và metadata nguồn phải đi cùng nhau. `build-sequences` chỉ dùng cache raw đã tải, xác minh hash, không tự tải ca khác. Mỗi ca lưu một mảng float32 gồm 7 giá trị và 7 tuổi phép đo; mask sinh khi đọc. Cửa sổ `(t−600,t]` gồm 300 bước, đọc memmap và giữ tối đa 8 ca trong cache mở.

Khi mở rộng pilot, tạo thư mục dataset/sequence/run mới. Thuật toán chia global subject giữ ổn định; chia **vai trò nội bộ pilot** phụ thuộc danh sách bệnh nhân nên có thể đổi khi mở rộng. Không so sánh kết quả trên hai cohort như một ablation cùng bệnh nhân.

## 2. Train và so sánh

```sh
python -m safeanes.cli run-pilot --dataset data/pilot_v1 --out artifacts/baselines_v2 --models map logistic hist_gradient lightgbm
python -m safeanes.cli train-sequence --config configs/tcn.json --dataset data/pilot_v1 --sequences data/sequences_v1 --out artifacts/tcn_v1 --bootstrap 200
python -m safeanes.cli train-sequence --config configs/transformer.json --dataset data/pilot_v1 --sequences data/sequences_v1 --out artifacts/transformer_v1 --bootstrap 200
```

Hai cấu hình chính dùng toàn bộ cửa sổ fit đủ điều kiện có ít nhất một nhãn biết; không cân bằng lại lớp. Seed mô hình không đổi vai trò bệnh nhân. Thiếu một lớp tại bất kỳ horizon nào trong fit/calibration/validation thì dừng và báo thiếu dữ liệu.

| Thành phần | Quy tắc thực thi |
|---|---|
| Split | Cùng `pilot_roles` với baseline: 60% fit, 15% calibration, 10% validation, 15% pilot_test; tất cả thuộc global train. |
| Chuẩn hóa | Mean/std tính trên grid của ca fit, mỗi điểm đếm một lần; giá trị thiếu được thay bằng 0 sau chuẩn hóa, kèm mask và age nếu bật. |
| Static | Age/BMI/ASA, chuẩn hóa từ fit; có mask riêng. |
| TCN | 7 residual block, dilation 1–64, mỗi block 2 causal Conv1D kernel 3; receptive field 509 bước; width 32. |
| Transformer | Patch 10 bước, width 64, 2 layer, 4 head; positional embedding học được. Attention chỉ trên lịch sử trong cửa sổ. |
| Hai horizon | `logit10 = logit5 + softplus(delta)`; cùng output dùng sigmoid, bảo đảm p10≥p5. Đây là thiết kế dự án. |
| Loss | BCE trên từng nhãn biết, bỏ `-1`; gradient clipping 1.0; AdamW. |
| Chọn checkpoint | BCE trên validation, tối đa 30 epoch, patience 5. Validation cũng chọn ngưỡng như pilot baseline; đây là mức tái sử dụng development được công bố, chưa phải protocol final. |
| Calibration | Một temperature dương và bias chung fit trên calibration patients; bảo toàn thứ tự horizon. Khác sigmoid độc lập từng horizon của baseline. |
| Ngưỡng | Chọn validation, sau persistence/cooldown; mọi tiêu chí đồng thời hoặc ghi fallback chưa đạt. |
| Test | Chỉ pilot_test; không thay threshold/calibration sau khi thấy metric. |
| Tài nguyên | CUDA dùng FP16 autocast/GradScaler; CPU dùng FP32; lưu thời gian epoch, số tham số, peak allocated VRAM (CPU: null). |

## 3. Resume và artifacts

Nếu phiên bị ngắt sau ít nhất một epoch:

```sh
python -m safeanes.cli train-sequence --config configs/tcn.json --dataset data/pilot_v1 --sequences data/sequences_v1 --out artifacts/tcn_v1 --resume
```

Giữ nguyên code, data, sequence cache, split và cấu hình; chỉ được tăng `epochs`. Resume khôi phục optimizer, AMP scaler, RNG, best weights và lịch sử. Run đã có `report.json` là hoàn tất, không resume/ghi đè; dùng thư mục mới. Checkpoint lưu sau mỗi epoch, nên một epoch dở sẽ được chạy lại. Đổi môi trường/phần cứng có thể làm khác kết quả số học; phép thử khớp tuyệt đối hiện kiểm tra trên CPU cùng môi trường.

| Artifact | Ý nghĩa |
|---|---|
| `last.pt` | Trạng thái resume cuối epoch, RNG và best checkpoint. |
| `model.pt` | Best weights, kiến trúc, normalizer, calibration, thresholds, protocol và định danh run. |
| `normalizer.json`, `pilot_roles.csv` | Dữ liệu dùng fit scaler và vai trò bệnh nhân. |
| `environment.json`, `config.json`, `history.json` | Phiên bản môi trường, source hashes, cấu hình và lịch sử epoch. |
| `report.json` | Metric, patient bootstrap CI, quality gates, tài nguyên đã đo. |
| `*_predictions.csv.gz` | Toàn bộ decision của pilot_test, gồm abstention, nhãn censored, raw/calibrated risks. |
| `*_validation_predictions.csv.gz` | Dự báo validation để nghiên cứu ngưỡng/ablation. |
| `*_case_metrics.csv`, `*_alarms.json` | Mẫu số theo ca và episode true/false/censored. |

## 4. Báo cáo và phát lại

```sh
python -m safeanes.cli report --dataset data/pilot_v1 --run artifacts/tcn_v1 --out reports/tcn_v1
python -m safeanes.cli report --dataset data/pilot_v1 --run artifacts/transformer_v1 --out reports/transformer_v1
```

Báo cáo Markdown lấy metric đã lưu, không fit lại mô hình hoặc chọn ngưỡng. Hình thể hiện MAP, nguy cơ, eligibility, biến cố và alarm thật. Chọn ví dụ true/false/censored và ca có biến cố bỏ sót khi có; bảng chính vẫn dùng toàn bộ pilot_test. Calibration slope/intercept fit trên test chỉ là chẩn đoán mô tả, không áp lại để sửa xác suất.

`diagnostics.json` gồm 10 calibration bins cố định, slope/intercept, CI và phân nhóm tuổi/ASA theo toàn ca. Chưa có CI riêng cho subgroup hoặc phân nhóm loại mổ/missingness. Không cắt các hàng rời rạc để chạy lại alarm state cho một subgroup theo MAP tức thời.

## 5. Vòng nghiên cứu tiếp theo

Sau khi xác định ứng viên bằng validation, chạy `seed=20260917,20260918,20260919` vào ba thư mục riêng; không chọn seed có pilot_test đẹp nhất. Với ablation, giữ dataset/split/seed và đổi `feature_set` thành `map`, `numeric`, `numeric_static`; so `masks=false/true`. `max_windows` chỉ dùng thăm dò, để null hoặc bỏ khóa khi train đầy đủ. Mỗi lần chạy phải lưu cấu hình riêng.

Chưa mở final test bằng các lệnh này. Trước vòng xác nhận: mở rộng development, chia riêng tuning/calibration/threshold hoặc OOF; audit nhãn với chuyên gia; hoàn thiện exposure từng giây; khóa model/ngưỡng/calibrator; đo T4 và đánh giá ba seed. Waveform chỉ bổ sung khi validation cho thấy cần thiết và có thiết kế so sánh cùng cohort. Kiểm định ngoài, nhãn cơ chế và nghiên cứu tiến cứu cần dữ liệu/quyền truy cập/nhân lực riêng.
