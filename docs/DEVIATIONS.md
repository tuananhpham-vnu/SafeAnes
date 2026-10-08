# Những điểm khác proposal và kế hoạch

Kế hoạch: `TRAINING_PLAN_V2.md` bản 2.5. File này được ghi dần theo tiến độ. Mỗi mục ghi: khác gì, lý do, ảnh hưởng dự kiến.

## Theo mục 14 của kế hoạch

| # | Khác gì | Lý do | Ảnh hưởng dự kiến |
|---|---|---|---|
| 1 | Bỏ 4 ca `4458, 6119, 6133, 6375` | Sóng ART 100 Hz, tiền xử lý yêu cầu 500 Hz nên lỗi | Không đáng kể (4/3.626 ca; 2 train, 2 test) |
| 2 | Cohort chính yêu cầu `has_wave` và `map_coverage ≥ 0,10` | 180 ca có track sóng nhưng không có arterial line hoạt động | Cohort D: 3.363 ca, 3.242 bệnh nhân |
| 3 | Nhãn dùng chính sách `lenient` | Tránh mất mẫu dương ở mốc dài chỉ vì nhiễu sau đợt tụt | `label_report.csv` báo cả `strict` |
| 4 | PPV tính lại theo đoạn 30 giây, lấy trung vị trong W | PPV trên cả cửa sổ lẫn xu hướng huyết áp | Ở W = 30 trùng khớp bản cũ; W = 120 trên train: NaN 38,2% → 14,6% |
| 5 | `in_event` và `post_event` dùng đợt tụt từ `label_map` | `label_map` được làm sạch với cờ nhìn về sau tối đa khoảng 60 giây | Nhỏ |
| 6 | DL có thêm cờ `baseline_missing` | 77 ca không có MAP nền (75 ca trong cohort D) | — |
| 7 | NB06 dùng thêm EV1000 ngoài `prep_v1` | `prep_v1` không có track EV1000 | Chưa làm |
| 8 | Ngày 29/09, agent đã đọc `label_map` của test để đếm tổng số đợt tụt (gộp mọi tập) | Ghi lại để minh bạch | Không có mô hình, ngưỡng hay tiêu chí cohort nào dựa trên số này |
| 9 | `eligible` và `abstain_reason` của test cũng được niêm phong | `in_event`, `post_event` suy ra từ đợt tụt | — |
| 10 | Không đánh giá hiệu năng riêng trên 78 ca không có sóng | Validation chỉ 3 ca, test 14 ca | — |
| 11 | `map_threshold` chỉ có 5 tổ hợp, tổng tabular 65 | Mô hình không phụ thuộc W | — |
| 12 | Sóng trên Kaggle lấy từ Kaggle Dataset, không từ `kernel_sources` | Tránh dữ liệu đổi giữa chừng | — |
| 13 | (Chỉ khi xảy ra) `use_deterministic_algorithms(True, warn_only=True)` | Một số phép backward trên CUDA không có bản tất định | Chưa xảy ra |

## Phát sinh khi làm

| # | Khác gì | Lý do | Ảnh hưởng dự kiến |
|---|---|---|---|
| 14 | Dữ liệu test ghi vào `data/sealed_v2/`, không phải `samples_v2/sealed/` | Tải cả thư mục `samples_v2` lên Kaggle thì không thể lẫn test vào | Không |
| 15 | `samples_v2` đưa lên Kaggle bằng Kaggle Dataset `uc04-samples-v2`, không qua HF `samples_repo` | Người dùng chọn: HF sẽ thiết lập sau, chỉ để lưu kết quả | Notebook đọc samples từ `/kaggle/input` |
| 16 | Chọn ngưỡng bằng quét toàn bộ khoảng 300 ứng viên, không chia đôi | FA/giờ **không** đơn điệu theo ngưỡng: ở ngưỡng rất thấp xác suất luôn trên ngưỡng, cảnh báo bắn một lần mỗi ca rồi đứng yên, FA/giờ lại thấp. Chia đôi có thể bỏ sót ngưỡng tốt nhất | Kết quả đúng với định nghĩa ở mục 8.4; chi phí vài giây mỗi tổ hợp |
| 17 | `label_report.csv` chỉ tính trên cohort D | Mọi so sánh mô hình dùng cohort D | — |
| 18 | Code đưa lên Kaggle bằng Kaggle Dataset `uc04-code` (git archive của commit, có file `CODE_COMMIT`), không qua GitHub | Chưa có repo GitHub; notebook kiểm tra commit của dataset khớp commit ghi trong notebook | Provenance trên Kaggle lấy commit từ `CODE_COMMIT` |
| 19 | ~~Notebook gộp đọc output của notebook trước qua `kernel_sources` khi chưa có HF~~ **Đã thay bằng mục 31** | Mâu thuẫn với mục 25 (HF bắt buộc trên Kaggle) | Không còn notebook nào dùng `kernel_sources` |
| 20 | Mức tin cậy của LightGBM tính cho `lgbm_wave` ở W\* với mốc 5 và 10 phút | Kế hoạch chỉ ghi "cấu hình được chọn" | 2 × 5 lần train thêm |
| 21 | Conv1D dùng `padding = kernel // 2` | Để W = 30 ra đúng 150 token (5 token/giây) như kế hoạch | Không |
| 22 | Loss validation để dừng sớm có thể tính trên một mẫu con cố định (`--max-val-rows`) | Tiết kiệm GPU nếu đọc sóng chậm | Mặc định dùng toàn bộ |
| 23 | **Luật nhãn âm mới** (`labels.negative_rule = "possible_event"`): mẫu không dương được gán 0 trừ khi các giây không xác định hoặc có MAP < 65 trong (t, t + h + 60] có thể chứa một đợt tụt, tức có một đoạn liên tiếp như vậy dài ≥ 60 giây bắt đầu trong (t, t + h]. Thay cho hai ngưỡng 5% và 30 giây của mục 3.5 | Luật cũ không đối xứng: mẫu dương không bao giờ bị −1, mẫu âm bị −1 mỗi lần flush khoảng 30 giây. Hệ quả: tỷ lệ dương bị đẩy lên, calibration học theo tỷ lệ đó, cảnh báo sai tại dòng −1 không được đếm | Tỷ lệ −1 ở mốc 5 phút (train): 18,1% → 6,5%. Tỷ lệ dương: 4,63% → 4,06%. Số đợt tụt eligible không đổi. `label_report.csv` in cả `lenient_possible`, `lenient_fraction` (luật cũ) và `strict` |
| 24 | Tabular (02a–02d) chạy trên máy, không trên Kaggle | Một tổ hợp khoảng 20 giây; train và NB04 cùng một máy nên không có rủi ro lệch phiên bản scikit-learn và LightGBM khi nạp `.joblib` | Kaggle chỉ dùng cho `00_env_check` và `03_dl_*` |
| 25 | Trên Kaggle, thiếu token HF hoặc `hub.runs_repo` còn là placeholder thì dừng (`HubRequired`; `00_env_check` báo FAIL; ô setup báo lỗi nếu thiếu Secret) | Phiên GPU bị dừng có thể mất output, và chạy tiếp từ `last.pt` cần HF | Trên máy vẫn cho phép tắt việc đẩy |
| 26 | Ngày 29/09, khi mô tả dữ liệu, agent liệt kê kích thước các file trong `sealed_v2` và in số dòng của `events_test.parquet`, tức tổng số đợt tụt của test | Lỗi của agent, ghi lại để minh bạch | Không có mô hình, ngưỡng, luật nhãn hay tiêu chí nào được chọn dựa trên số này. Từ nay không đọc metadata của file nhãn hay đợt tụt trong `sealed_v2` trước khi khóa |
| 27 | Thêm tùy chọn `evaluation.alarm_rearm`: `drop_below` (mục 8.2, mặc định) hoặc `cooldown` (lặp lại mỗi 300 giây khi vẫn trên ngưỡng) | Nhóm đang cân nhắc chính sách cảnh báo | Hai chính sách đã được tính song song trên validation (`model_comparison.csv`) |
| 28 | Mức tin cậy LightGBM tính trên thang logit và phải qua tiêu chí theo thập phân vị nguy cơ. **Không qua ở cả hai mốc → bỏ** | Trên thang xác suất, độ lệch chuẩn chỉ phản ánh mức nguy cơ | Chỉ giữ độ lệch giữa các seed của DL, kiểm cùng tiêu chí |
| 29 | Phân tích thăm dò `lgbm_numeric` với W = 300, 600 (và W = 120 tính lại) | W\* = 120 nằm ở biên khoảng thử; quyết định sau khi xem validation | Báo cáo riêng trong `reports/exploratory/`, không thay W\* |
| 30 | Định nghĩa đặc trưng của `prep_v1` xác định bằng đối chiếu: `map_time_65_75` dùng 65 ≤ MAP < 75; `*_missing_w{W}` chia cho số ô của cửa sổ nằm trong ca (cửa sổ bị cắt tại lúc bắt đầu ca) | Phụ lục A chỉ ghi "vùng 65–75" và "tỷ lệ ô thiếu" | Hàm tính lại khớp `prep_v1` ở 99,9% dòng (224 cột, 40 ca) |
| 31 | Trên Kaggle, `03_dl_finalize` chỉ đọc logit của từng seed từ HF; không notebook nào dùng `kernel_sources` | Giải quyết mâu thuẫn giữa mục 19 và 25 | Nếu HF chưa sẵn sàng, `03_dl_finalize` dừng ngay |
| 32 | Tên 3 Kaggle Dataset: `<KAGGLE_USER>/uc04-code`, `<KAGGLE_USER>/uc04-samples-v2`, `<KAGGLE_USER>/uc04-prep-v1` (kế hoạch bản 2.6 gọi dataset sóng là `safeanes-prep-v1`) | Thống nhất một tiền tố `uc04-` cho cả 3 dataset; tên lấy từ `configs/uc04_v2.json` (mục `kaggle`) | Người dùng đặt đúng tên này khi tạo dataset sóng trên giao diện Kaggle |
| 33 | Ô setup của notebook tự kiểm tra: đúng một dataset khớp cho mỗi input (nhiều hơn thì dừng), `CODE_COMMIT`, `SAMPLES_SHA256` (sha256 của `samples.json`, điền sẵn khi sinh notebook), sha256 `qc.csv` = `input.qc_sha256`; trên Kaggle, `00_env_check` và mọi `03_dl_*` dừng ngay nếu thiếu Secret `HF_TOKEN` hoặc `hub.runs_repo` là placeholder | Không để notebook chạy trên dữ liệu hoặc code sai phiên bản, hay mất kết quả khi phiên bị dừng | Chạy trên máy vẫn cho phép `--no-push` |
| 34 | Provenance chỉ coi là dirty khi có thay đổi chưa commit (kể cả file mới) trong `src/`, `scripts/`, `configs/`, `templates/`, và ghi `dirty_paths`. `lock_models.py` từ chối khóa kết quả dirty hoặc có provenance kiểu cũ (không có `dirty_paths`) | Thay đổi tài liệu không làm thay đổi kết quả | **Kết quả tabular hiện có dùng provenance kiểu cũ → phải chạy lại tabular (khoảng 15 phút) trước khi khóa** |
| 35 | Thêm chỉ số `alarm_time_fraction` = thời gian trạng thái cảnh báo bật / tổng `exposure_seconds` của dòng eligible (kèm CI). Tính cho cả hai chính sách từ dự đoán đã lưu (`reevaluate_tabular.py`) | Mục 8.2–8.3 bản 2.6 | `tabular_validation_by_policy.csv`, `model_comparison*.csv`. Mặc định vẫn `drop_below`, chờ nhóm chốt |
| 36 | Đóng gói bằng `scripts/package_for_kaggle.py` vào `dist/` (không commit): zip samples dạng ZIP_STORED, zip code bằng `git archive` + `CODE_COMMIT`, thư mục staging có `dataset-metadata.json`. Hai script cũ (`make_kaggle_dataset.py`, `make_kaggle_code.py`) đã xóa vì chúng ghi file vào `data/samples_v2` | Không được sửa `samples_v2` | `samples.json` không ghi commit; commit và cờ dirty của lần dựng samples (`e16dc24`, không dirty, provenance kiểu cũ) được ghi vào `dist/for_review/upload_manifest.json` từ `samples_v2/provenance.json` |
| 37 | `verify_package.py` (script kiểm tra độc lập, không sửa) nằm ở `D:\SafeAneserify_package.py`, không phải `uc04/tools/` | Giữ nguyên vị trí người dùng đặt | Script không nhận ra cấu trúc `norm_tabular.json` (`{"columns": {...}}`) nên báo WARN ở phần trung vị; không sửa `samples_v2` để tránh cảnh báo này |
| 38 | DL: khi máy có nhiều GPU (Kaggle T4 x2), các seed còn lại của một W chạy song song, mỗi GPU một tiến trình con (`CUDA_VISIBLE_DEVICES`), DataLoader worker chia đều. `--gpus 1` để chạy tuần tự như cũ | Code cũ chỉ dùng 1 trong 2 GPU; quota Kaggle tính theo thời gian phiên | Không đổi kết quả: test so logit chạy song song với chạy tuần tự, giống hệt. Thời gian mỗi W giảm khoảng 5 → 3 lượt seed, nếu 4 vCPU đủ nạp dữ liệu cho 2 GPU (đo bằng `03_dl_smoke`) |
| 39 | DL: chạy tiếp từ HF kéo cả thư mục seed (`best.pt`, `last.pt`, `train_log.csv`), không chỉ `last.pt`. `last.pt` lưu thêm trạng thái RNG của CUDA. Checkpoint ghi ra file tạm rồi đổi tên | Trước đây nếu sau khi chạy tiếp không epoch nào tốt hơn thì không có `best.pt` để dự đoán, script dừng với lỗi. Dropout trên GPU dùng RNG của CUDA nên chạy tiếp không lặp lại đúng. Phiên bị dừng giữa lúc ghi có thể để lại checkpoint hỏng (mục 2.12) | Chỉ ảnh hưởng các lần chạy bị ngắt rồi chạy tiếp |
| 40 | Bỏ khóa phiên bản thư viện: `requirements-lock.txt`, `requirements-lock.json`, `requirements-freeze-local.txt` và `scripts/lock_requirements.py` được thay bằng một `requirements.txt` chỉ ghi phiên bản tối thiểu (`pyproject.toml` đọc từ file này). Kaggle dùng phiên bản có sẵn của image (kể cả torch CUDA), chỉ cài thêm hoặc nâng cấp gói thiếu hay cũ hơn mức tối thiểu; `env_check.py` kiểm tra mức tối thiểu thay vì so với lock | Người dùng chọn: không cần cố định phiên bản; khỏi cài lại torch mỗi phiên Kaggle | Toàn bộ test UC04 qua ở mức tối thiểu (numpy 1.26, pandas 2.2, pyarrow 15, scipy 1.11, scikit-learn 1.4, lightgbm 4.3, torch 2.3, huggingface_hub 0.24) và ở bản mới nhất. Phiên bản thực tế ghi trong `provenance.json`. **NB04 vẫn so phiên bản với provenance của mô hình** (`env_check.py --provenance`): trước khi chạy test phải cài trên máy đúng numpy, pandas, scipy, scikit-learn, joblib, lightgbm, torch mà Kaggle đã dùng |

Ghi chú cho mục 23: với luật mới, một dòng tại t mà MAP thấp đã biết tiếp diễn sau t (đợt tụt đang diễn ra) được gán −1 thay vì 0. Những dòng này luôn không eligible (`in_event`, `in_hypotension` hoặc `post_event`), nên không ảnh hưởng đến fit hay đánh giá.

## Chạy lại trên samples v3 (08/10/2026)

Samples v3 (Kaggle `datnguyen31112/safeanes-v3`) được build từ `prep_v2` bằng đúng config v2, chỉ khác `input.qc_sha256`
(digest khớp `samples.json`). Mỗi ca bắt đầu từ đầu bản ghi thay vì từ lúc rạch da. Số thứ tự 46–57 mà README của v3
nhắc tới nằm ở repo build dữ liệu, nên các mục dưới đây đánh số riêng.

| # | Khác gì | Lý do | Ảnh hưởng dự kiến |
|---|---|---|---|
| v3-1 | `configs/uc04_v3.json` = config v2, đổi `input.qc_sha256` (prep_v2), `paths.samples`, `kaggle.samples_dataset`, `kaggle.samples_json_sha256`, `hub.runs_repo` (`uc04-v3-runs`, chưa tạo); thêm `kaggle.wave_qc_sha256` (qc.csv của `uc04-prep-v1`) | Dataset sóng vẫn là prep_v1, khác qc.csv của samples | `kaggle_run.py` kiểm tra dataset sóng bằng `wave_qc_sha256` nếu có |
| v3-2 | DL: gốc thời gian của sóng là `surgery_start` (từ `surgery_start.csv`), không phải `case_index.start`; `wave_mask` của v3 được cắt từ `surgery_start` (`dl_data.load_case_meta`) | `wave100` của prep_v1 bắt đầu lúc rạch da, còn v3 bắt đầu từ đầu bản ghi: dùng thẳng sẽ lệch sóng 2–224 phút (trung vị 40,5 phút). Đã kiểm tra: sau rạch da, cửa sổ sóng trùng với v2; mask trùng v2 ở 3089/3108 ca (19 ca lệch nhẹ do prep_v2 tính lại) | Dòng trước rạch da (≈ 13% số dòng eligible) không có sóng: kênh sóng bị mask toàn bộ, mô hình chỉ dựa vào nhánh chỉ số |
| v3-3 | DL v3 trên Kaggle chạy bằng `kaggle_run.py --no-hf` (truyền `--no-push`), code nhúng trong notebook (tar base64 + `CODE_COMMIT`), mỗi W tách 2 notebook (2 + 3 seed), kết quả tải về bằng `kaggle kernels output` | Chưa có Secret `HF_TOKEN` trên các tài khoản mới; không push code chưa commit lên GitHub | Không chạy tiếp được nếu phiên quá 12 giờ (mỗi notebook ước tính ≤ 9 giờ) |
| v3-4 | Thêm `uc04.phases` và `scripts/compare_versions.py`; `dl_finalize.py` ghi `reports/dl_version_comparison.csv` (theo giai đoạn, `--compare-with` một run cũ) | Nhãn v3 khác v2 nên số tổng không so trực tiếp được; giai đoạn "surgery" của v3 là phép so gần nhất với v2 | Mô tả, không bootstrap. Cảnh báo được phát trên cả ca rồi mới chia theo thời điểm |
| v3-5 | Thêm 2 mô hình ngoài kế hoạch, định nghĩa trong code (`tabular.EXTRA_VERSIONS`) để digest config không đổi: `lgbm_context` (66 chỉ số + 31 cột `f1a_*` của `features_v5`) và `catboost_numeric` (CatBoost của E08 trên 66 chỉ số) | Thử ngữ cảnh của ca (xu hướng 5–30 phút, MAP so với trung vị lũy tiến của ca, thời gian từ khởi mê/rạch da, số đợt tụt trước) và một họ cây khác | Không dùng `f1b_*`, `f1c_*` (thuốc, BIS, PEEP, xét nghiệm: ngoài phạm vi monitor của kế hoạch 3.2). `features_v5` chỉ có dòng cohort chính và khác thứ tự dòng: nối theo `(caseid, time)` |
| v3-6 | Thêm `dl_context` (`train_dl.py --context`, `dl_finalize.py --context`): Conv1D + Transformer như kế hoạch 7, nhánh chỉ số nhận thêm 31 cột `f1a_*` (chuẩn hóa bằng thống kê tính trên dòng train eligible, như `norm_tabular.json`) và 2 cờ thiếu cho `f1a_min_since_opstart`, `f1a_min_since_last_event`; ghi `artifacts/dl_context/`, report có hậu tố `_context` | Mô hình mới cho giai đoạn trước rạch da (không có sóng): ngữ cảnh của ca thay cho phần thông tin sóng bị thiếu. Hai cờ giữ thông tin "đang trước rạch da" / "chưa có đợt tụt" mà điền median sẽ xóa mất | 20 lần train trên Kaggle như DL gốc; mọi siêu tham số giữ nguyên |
| v3-7 | Thêm TabM (`scripts/train_tabm.py`, `uc04.tabm`): `tabm_numeric` (66 chỉ số), `tabm_context` (+ 31 cột `f1a_*`). Một mô hình 5 horizon có thứ tự cho mỗi W (luật đầu ra như DL), 1 seed; sau đó mỗi horizon dùng chung Platt, chọn ngưỡng, bootstrap và file kết quả với `train_tabular.py` | Họ TabM của E06/E08 chưa chạy trên cohort UC04. Siêu tham số giữ như E06 (k=16, 2 khối × 64, PLE 8 bin, AdamW 2e-3, dừng sớm trên 1/5 bệnh nhân train) | Chạy trên GPU Kaggle; khác E06 ở số horizon và cách tách nhóm dừng sớm (băm `subjectid` với salt mới) |

## Ghi chú từ NB01 (không phải sai lệch, để nhóm cân nhắc)

- (Với luật cũ, đã thay bằng mục 23.) Ở mốc 5 phút, khoảng 18% dòng eligible có nhãn −1 (`lenient`). Phân tích 300 ca train: 3,4% do sát cuối ca, 14,9% do hơn 5% số giây tương lai không xác định. Khoảng trống điển hình chỉ khoảng 31 giây (trung vị, 75% ≤ 49 giây), rất có thể là lấy máu hoặc flush. Mỗi khoảng trống như vậy làm mất nhãn của khoảng 12 mốc dự đoán trước nó.
- Đợt tụt nghi nhiễu (`min_map ≤ 30`): 57/6.024 = 0,95% (dưới ngưỡng 2%).

## Ghi chú về `exposure_seconds`

Không có dòng nào ở giữa ca có `exposure_seconds < 30`. Trong cohort D của train, 2.160/2.393 ca có mốc dự đoán cuối đúng bằng thời điểm kết thúc ca (t = end), nên phơi nhiễm bằng 0; 233 ca còn lại có dòng cuối từ 1 đến 29 giây. Vì vậy tổng số giờ (6.964) thấp hơn số dòng × 30 giây (6.983) khoảng 19 giờ. Các dòng t = end có mọi nhãn bằng −1 (`end_of_case`) và phơi nhiễm 0, nên không ảnh hưởng tới số cảnh báo sai mỗi giờ.

## Ghi chú về mức tin cậy của LightGBM

Ba mức tin cậy chia theo tam phân vị của độ lệch chuẩn xác suất giữa 5 seed. Độ lệch chuẩn này tăng theo mức xác suất: nhóm "cao" có xác suất trung bình 0,5%, nhóm "thấp" có 9,8% (mốc 5 phút). Như vậy mức tin cậy hiện gần như phản ánh mức nguy cơ, chưa phản ánh độ bất định riêng. Nên cân nhắc tính độ lệch chuẩn trên thang logit trước khi dùng cho báo cáo.

## Ghi chú về thời gian chạy (đo trên máy, 12 nhân, 16 GB RAM)

- NB01: 73 giây cho 3.622 ca.
- Một tổ hợp LightGBM trên toàn bộ dữ liệu (fit, calibrate, chọn ngưỡng, 200 bootstrap): khoảng 20 giây.
- Suy luận DL trên CPU của máy: 177–614 mẫu/giây tùy W. NB04 với cả 4 W × 5 seed trên 178.365 dòng test: ước khoảng 3 giờ; một W: khoảng 45 phút (`reports/dl_cpu_speed.json`). Nếu cần rút ngắn, NB04 chỉ chạy DL cho W đã chốt.
