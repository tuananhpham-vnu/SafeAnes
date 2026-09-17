# SafeAnes

UC04: dự báo sớm tụt huyết áp trong mổ. Release 0.2 chạy **dữ liệu thật → nhãn → baseline/TCN/Transformer → calibration → cảnh báo → báo cáo và phát lại**, dùng riêng global train để giữ final test chưa sử dụng.

- [Kế hoạch và yêu cầu độ chính xác](UC04_RESEARCH_PLAN.md)
- [Các bước tìm hiểu/triển khai](docs/IMPLEMENTATION_STEPS.md)
- [Nguồn và ánh xạ vào code](docs/SOURCES.md)
- [Protocol và giới hạn](docs/PROTOCOL.md)
- [Hướng dẫn TCN/Transformer, resume và ablation](docs/SEQUENCE_RUNBOOK.md)
- [Model card](docs/MODEL_CARD.md) · [Data card](docs/DATA_CARD.md)

## Chạy local

Python ≥3.11. Tạo môi trường bằng `python -m venv .venv`, kích hoạt trước khi cài: Windows PowerShell dùng `.venv\Scripts\Activate.ps1`; Linux dùng `source .venv/bin/activate`.

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m safeanes.cli fetch-pilot --cases 60 --workers 4
python -m safeanes.cli build-pilot --out data/pilot_v1
python -m safeanes.cli run-pilot --dataset data/pilot_v1 --out artifacts/pilot_v1
```

Tải cần Internet; các bước sau dùng cache. Lỗi tải có thể chạy lại để resume. Dataset và experiment output phải mới/rỗng để không ghi đè kết quả. Có thể chạy trực tiếp với `PYTHONPATH=src` nếu môi trường đã có dependencies.

Baseline mặc định: score MAP, logistic MAP/slope/variability, HistGradientBoosting. LightGBM tùy chọn: cài `python -m pip install -e ".[boosting]"` rồi thêm `--models map logistic hist_gradient lightgbm`. Bootstrap pilot mặc định 200 theo bệnh nhân.

## Kaggle

Dùng [notebook baseline](notebooks/01_uc04_pilot.ipynb), [chuẩn bị sequence](notebooks/02_numeric_sequences.ipynb), [train TCN/Transformer](notebooks/03_tcn_transformer.ipynb) và [phát lại/báo cáo](notebooks/04_case_replay.ipynb). Sửa đường dẫn Input theo tài khoản; output ở `/kaggle/working`. Chạy bước chuẩn bị dữ liệu trên CPU và chỉ bật T4 cho notebook train. Chưa có benchmark thực thi trên T4/Kaggle.

## TCN và Transformer

```sh
python -m pip install -e ".[deep,plots]"
python -m safeanes.cli build-sequences --dataset data/pilot_v1 --root data/vitaldb --out data/sequences_v1
python -m safeanes.cli train-sequence --config configs/tcn.json --out artifacts/tcn_v1
python -m safeanes.cli train-sequence --config configs/transformer.json --out artifacts/transformer_v1
python -m safeanes.cli report --run artifacts/tcn_v1 --out reports/tcn_v1
python -m safeanes.cli report --run artifacts/transformer_v1 --out reports/transformer_v1
```

Mỗi output phải mới/rỗng. Với lần train bị gián đoạn, thêm `--resume` và giữ nguyên code/data/config; xem [runbook](docs/SEQUENCE_RUNBOOK.md). CUDA dùng FP16, CPU dùng FP32. Không coi việc hoàn tất train là đạt mục tiêu hiệu năng.

## Đầu ra

| File | Nội dung |
|---|---|
| `data/vitaldb/cohort_manifest.csv` | Toàn bộ ca, subject split, lý do loại |
| `data/vitaldb/raw/*.source.json` | URL, thời điểm tải và SHA-256 |
| `data/pilot_v1/quality.csv` | Độ phủ nhãn, nhịp MAP, event/decision từng ca |
| `data/pilot_v1/windows.csv.gz` | Feature causal, eligibility và nhãn; -1 là censored |
| `artifacts/pilot_v1/report.json` | Metric, CI, threshold selection và quality gates |
| `artifacts/pilot_v1/*_predictions.csv.gz` | Dự báo để kiểm tra/phát lại |
| `artifacts/pilot_v1/*_alarms.json` | Episode thật, giả hoặc censored |

`pilot_test` là tập thăm dò bên trong global train, **không phải final test**. Các con số trong kế hoạch là mục tiêu, không phải hiệu năng được hứa trước. Dữ liệu lớn và artifacts không đưa vào Git.
