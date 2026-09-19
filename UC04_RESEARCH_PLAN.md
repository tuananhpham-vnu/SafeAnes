# SafeAnes UC04 — Kế hoạch nghiên cứu và triển khai trên Kaggle T4 16 GB

**Quy ước phát hành cập nhật:** toàn bộ công việc chưa push hiện tại được gộp vào **v0.2**; mỗi lần push mới tăng version. Các nhãn v0.3/v0.4 cũ tương ứng đợt thí nghiệm E03/E04, không phải release riêng. E04 đã chạy CatBoost ba seed và ablation ngưỡng; E05 tiếp tục mở rộng 300 ca theo [plan](docs/experiments/E05_PLAN.md). Xem [quy ước và kết quả từng đợt](README.md#release).

Ngày tra cứu/cập nhật: 17/09/2026. Phạm vi: dự báo sớm tụt huyết áp trong mổ, hiệu chỉnh xác suất, đánh giá cảnh báo và lộ trình nghiên cứu nguyên nhân. Phiên bản 0.2 đã triển khai pipeline numeric, baseline, TCN/Transformer, calibration, checkpoint/resume và báo cáo/phát lại ca. Thực nghiệm hiện dùng pilot VitalDB trên CPU; chưa có benchmark T4/Kaggle hoặc nghiệm thu hiệu năng. Xem [trạng thái và phần còn thiếu ở mục 13](UC04_RESEARCH_PLAN.md#13-trạng-thái-bàn-giao-và-các-mốc-còn-thiếu).

Tài liệu thực thi: [các bước triển khai](docs/IMPLEMENTATION_STEPS.md), [runbook và lệnh chạy](docs/SEQUENCE_RUNBOOK.md), [nguồn gắn với code](docs/SOURCES.md), [protocol](docs/PROTOCOL.md), [model card](docs/MODEL_CARD.md), [data card](docs/DATA_CARD.md). Kết quả đo: [baseline](reports/PILOT_BASELINE.md), [TCN](reports/tcn_v1/REPORT.md), [Transformer](reports/transformer_v1/REPORT.md); [biên bản kiểm thử và thực nghiệm](reports/IMPLEMENTATION_VALIDATION.md).

## 1. Hướng đề xuất

**Yêu cầu ưu tiên: độ chính xác dự báo cao, phát hiện được phần lớn biến cố và ít cảnh báo giả.** T4 16 GB và chi phí thấp là ràng buộc triển khai. Mô hình phải đạt đồng thời các mục tiêu chất lượng ở mục 8; việc chạy được trên T4 hoặc tốt hơn baseline chưa đủ để nghiệm thu hiệu năng.

Bắt đầu bằng **VitalDB → baseline MAP/logistic regression → LightGBM → TCN nhỏ → Transformer nhỏ**. Chỉ bổ sung waveform nếu chứng minh được giá trị tăng thêm so với sinh hiệu dạng số trên cùng nhóm bệnh nhân và cùng giao thức đánh giá.

Sản phẩm đầu tiên là pipeline tái lập được và bản phát lại ca mổ: mỗi 30 giây xuất xác suất có biến cố mới trong 5 và 10 phút tới, chất lượng dữ liệu và trạng thái cảnh báo. Khả năng báo trước thực tế phải đo bằng thời gian từ cảnh báo đến khởi phát biến cố.

Ba câu hỏi nghiên cứu phù hợp nguồn lực:

1. Mô hình nhỏ có cải thiện phát hiện biến cố ở cùng số cảnh báo giả/giờ so với MAP hiện tại và xu hướng MAP không?
2. Thêm sinh hiệu đa biến hoặc waveform có mang lại lợi ích đủ lớn so với chi phí xử lý và yêu cầu thiết bị không?
3. Hiệu chỉnh xác suất và chính sách cảnh báo giúp thay đổi PPV, độ nhạy biến cố và thời gian báo trước như thế nào?

Không có căn cứ từ các nguồn đã kiểm tra để tuyên bố một mô hình là “SOTA tuyệt đối”: cohort, nhãn, khoảng dự báo, cách lấy mẫu âm và thước đo khác nhau. Cần phân biệt mô hình mới, mô hình có kết quả tốt trong nghiên cứu riêng và mô hình tốt nhất trên benchmark của UC04.

## 2. Paper và mã nguồn nên đọc

Các kết quả dưới đây là số tác giả công bố, không phải kết quả tái lập. Không xếp hạng các paper bằng cách so trực tiếp AUROC khác giao thức.

| Ưu tiên | Paper / trạng thái | Phương pháp và điểm liên quan | Hành động cho UC04 |
|---|---|---|---|
| P0 | Lee HC et al., **VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients**, Scientific Data, 2022; phản biện | Bộ dữ liệu 6.388 ca, có waveform và sinh hiệu dạng số; có định danh bệnh nhân để nhận biết mổ lại. | Đọc cấu trúc dữ liệu, kiểm kê track và nhóm theo subjectid. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9178032/), [dữ liệu](https://physionet.org/content/vitaldb/1.0.0/). |
| P0 | Yang et al., **The effect of selection bias on the performance of a deep learning-based intraoperative hypotension prediction model…**, BJA, 2025; phản biện | Khi thay tập kiểm tra có chọn lọc bằng mẫu ít thiên lệch, PPV mô hình DL ở mốc 5 phút giảm từ 0,937 xuống 0,068 trong thiết lập của nghiên cứu. | Thiết kế tập test liên tục, gồm cả các đoạn gần ngưỡng; đọc trước khi cắt window. [Paper](https://pubmed.ncbi.nlm.nih.gov/40404499/). |
| P0 | Mulder et al., **Hypotension Prediction Index Is Equally Effective… Compared to a Mean Arterial Pressure Threshold**, Anesthesiology, 2024; nghiên cứu quan sát tiến cứu | Trong cohort nghiên cứu, HPI và ngưỡng MAP có khả năng dự báo tương đương. | Bắt buộc có baseline MAP hiện tại và MAP + slope. [Paper](https://pubmed.ncbi.nlm.nih.gov/38558038/), [code phân tích](https://github.com/crph-utwente/HPIvalidation). |
| P1 | Hatib et al., **Machine-learning Algorithm to Predict Hypotension Based on High-fidelity Arterial Pressure Waveform Analysis**, Anesthesiology, 2018; phản biện | Đặc trưng waveform động mạch; báo cáo AUROC 0,97 ở 5 phút và 0,95 ở 10 phút. Đây là mốc tham chiếu lịch sử. | Hiểu bài toán và lựa chọn đặc trưng; không dùng các con số này làm cam kết nghiệm thu. [Paper](https://pubmed.ncbi.nlm.nih.gov/29894315/). |
| P1 | Lee S et al., **Deep learning models for the prediction of intraoperative hypotension**, BJA, 2021; phản biện | Hướng deep learning dự báo IOH, nền tảng cho dòng nghiên cứu sau này. | Tham khảo thiết kế nghiên cứu waveform. Lần tra cứu này xác minh được metadata, chưa kiểm tra trọn phần Methods; không chép thông số huấn luyện hay điểm số chưa xác minh. [Paper](https://pubmed.ncbi.nlm.nih.gov/33558051/). |
| P1 | Zhu et al., **Transformer-based deep learning model for real-time prediction of intraoperative hypotension using dynamic time-series vital signs**, PLOS Medicine, 25/03/2026; phản biện | Dùng sinh hiệu dạng số; báo cáo AUC 0,904 / 0,892 / 0,882 ở 5 / 10 / 15 phút, có kiểm định ngoài trên VitalDB. | Hướng hiện đại phù hợp để đối chiếu với TCN và LightGBM. Dữ liệu phát triển gốc không công khai đầy đủ; huấn luyện trên VitalDB là thích nghi phương pháp, không tái lập toàn bộ nghiên cứu. [Paper](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1005024), [code](https://github.com/ShouqiangZhu/IOH_Transformer). |
| P1 | Wang et al., **Early prediction of intraoperative hypotension: development and validation of the HypoBridCast hybrid deep learning model**, BMC Anesthesiology, 13/05/2026; phản biện | Conv1D + Transformer; 4 waveform và biến trước mổ; 3.369 ca VitalDB và 437 ca kiểm định ngoài. AUROC nội bộ 5 phút 0,9442. Mẫu âm lấy từ giai đoạn ổn định và tránh 20 phút trước biến cố. | Tham khảo nhánh waveform, nhưng đánh giá lại trên luồng liên tục. Code được cung cấp theo yêu cầu tác giả, chưa có đường dẫn repository công khai được xác minh. [Paper](https://link.springer.com/article/10.1186/s12871-026-03888-8). |
| P2 | Cheng et al., **HMF / A Hybrid Multi-Factor Network with Dynamic Sequence Modeling for Early Warning of Intraoperative Hypotension**, arXiv từ 2024, bản v3; nguồn kiểm tra là preprint | Dự báo MAP/SBP bằng decomposition, normalization và Transformer theo patch. Phần mô tả split chia timeline trong từng bệnh nhân; khác mục tiêu tổng quát hóa sang bệnh nhân mới. | Mượn ý tưởng patch/decomposition; chạy lại với split theo subjectid. [Paper](https://arxiv.org/html/2409.11064v3), [code](https://github.com/ustc-time-series/HMF). |
| P2 | **A Self-Adaptive Frequency Domain Network for Continuous Intraoperative Hypotension Prediction**, 2026; bài proceedings trên trang nhà xuất bản | Hướng mô hình miền tần số; có repository tác giả. | Ứng viên bổ sung nếu còn ngân sách; chưa benchmark T4 và chưa xác minh đầy đủ giao thức/số liệu, nên không chọn làm mô hình chính. [Paper](https://journals.sagepub.com/doi/10.3233/FAIA251101), [code](https://github.com/sleepwithwind/safdnet). |
| P3 | Zhang et al., **Multimodal Forecasting of Sparse Intraoperative Hypotension Events Powered by Language Model / IOHFuseLM**, 2025; nguồn kiểm tra là arXiv v1 | Kết hợp ngôn ngữ và chuỗi thời gian. Paper mô tả RTX 4090, một số thí nghiệm dùng A100, và GPT-4o tạo mô tả lâm sàng. | Đọc để biết xu hướng; không ưu tiên tái lập với T4 đơn và yêu cầu tiết kiệm. Không suy diễn rằng toàn bộ IOHFuseLM cần 8 A100. [Paper](https://arxiv.org/html/2505.22116v1), [code](https://github.com/zjt-gpu/IOHFuseLM). |

Hai paper kiến trúc để triển khai mô hình nhỏ: **Bai, Kolter & Koltun, TCN (2018)** dùng convolution cho chuỗi thời gian ([paper](https://arxiv.org/abs/1803.01271)); **Nie et al., PatchTST, ICLR 2023** dùng patch để biểu diễn chuỗi ([paper](https://arxiv.org/abs/2211.14730)). Đây là nguồn phương pháp tổng quát, không phải bằng chứng TCN/PatchTST đạt SOTA lâm sàng cho UC04.

Lưu ý khi tái lập: kiểm tra mã nguồn và phụ lục trước khi chép preprocessing. Ví dụ phần văn bản của Zhu et al. có công thức thay thế MAP đáng nghi về hệ số SBP/DBP; UC04 ưu tiên MAP đo được, không dùng công thức ấy để tạo nhãn. Các báo cáo tài nguyên của paper cũng không thay thế phép đo trên T4.

## 3. Dữ liệu và phạm vi cohort

### Giai đoạn công khai

Chọn người lớn, phẫu thuật không tim, gây mê toàn thân và có MAP động mạch đủ chất lượng để xác định biến cố. Không mặc định tất cả 6.388 ca đều đủ điều kiện. Báo cáo số ca và số bệnh nhân sau từng bước lọc.

VitalDB có thông số dạng số cách nhau khoảng 1–7 giây và waveform khoảng 62,5–500 Hz tùy track. Dữ liệu còn artefact thực tế nên cần xử lý chất lượng. [Mô tả chính thức](https://physionet.org/content/vitaldb/1.0.0/).

Các nhóm đầu vào dự kiến, tên track cụ thể xác nhận bằng manifest:

| Nhóm | Biến | Cách sử dụng |
|---|---|---|
| Bắt buộc | MAP động mạch; subjectid, caseid, timestamp | MAP vừa cung cấp lịch sử đầu vào vừa tạo nhãn từ phần tương lai; tách tuyệt đối hai miền thời gian. |
| Numeric mở rộng | SBP, DBP, HR, SpO₂, EtCO₂; RR nếu đủ phủ | Không loại toàn bộ ca chỉ vì thiếu một biến tùy chọn; thêm mask và thời gian từ lần đo gần nhất. |
| Trước mổ | Tuổi, giới, BMI, ASA và thông tin đã biết trước dự báo | Chỉ dùng trường có nguồn và thời điểm khả dụng rõ ràng. |
| Waveform tùy chọn | ABP, sau đó ECG hoặc PPG | Đánh giá giá trị tăng thêm theo từng modality. |
| Thuốc/dịch tùy chọn | Sự kiện, tốc độ truyền, bolus đã thực sự xảy ra trước t | Chỉ thêm khi xác minh timestamp và độ đầy đủ; không lấy tổng cuối ca. |

Phân biệt bài toán numeric từ đường động mạch với triển khai chỉ có băng quấn NIBP: giảm tần số đầu vào không biến một cohort có đường động mạch thành bằng chứng cho mọi phòng mổ. NIBP thưa không đủ xác nhận chắc chắn một biến cố kéo dài 60 giây.

### Kiểm định ngoài

MOVER là ứng viên ở giai đoạn tiếp: dữ liệu phẫu thuật tại UCI, gồm EHR và waveform; công bố 83.468 cuộc mổ của 58.799 bệnh nhân, yêu cầu ký DUA để truy cập. Không cần tải toàn bộ mới bắt đầu UC04. Chỉ lấy cohort/track phù hợp và kiểm tra điều kiện xử lý trên Kaggle trước khi đưa dữ liệu lên. [Paper và truy cập](https://escholarship.org/uc/item/82r1k55m).

VitalDB có subjectid cho các ca mổ lại; split theo subjectid, không chỉ caseid. Thời gian VitalDB được đưa về mốc tương đối của từng ca nên không giả định có thể chia train/test theo năm lịch. Kiểm định theo thời gian sẽ cần dữ liệu có mốc thời gian phù hợp. [Mô tả định danh](https://pmc.ncbi.nlm.nih.gov/articles/PMC9178032/).

## 4. Đóng băng định nghĩa bài toán trước khi train

Đây là giao thức đề xuất của dự án, cần chốt với nhóm lâm sàng trước nghiên cứu tiến cứu.

| Thành phần | Cấu hình ban đầu |
|---|---|
| Biến cố IOH | MAP < 65 mmHg liên tục ít nhất 60 giây trên tín hiệu hợp lệ; khởi phát là đầu đoạn đủ điều kiện. |
| Bối cảnh đầu vào | 10 phút trước t, tuyệt đối không có dữ liệu sau t. |
| Lưới numeric | 2 giây/bước, tương đương 300 bước/10 phút. Tăng mật độ lưới không tạo thêm phép đo độc lập. |
| Chu kỳ dự báo | 30 giây. |
| Nhãn y5 | Có khởi phát biến cố mới trong (t, t + 5 phút]. |
| Nhãn y10 | Có khởi phát biến cố mới trong (t, t + 10 phút]. |
| Hiện đang tụt huyết áp | Tách trạng thái “đang có biến cố”; không tính như dự báo sớm. |
| Phục hồi | Mặc định yêu cầu MAP hợp lệ ≥ 65 trong 2 phút để tái kích hoạt dự báo biến cố mới. |
| Theo dõi tương lai thiếu | Gắn nhãn không xác định và loại khỏi loss/metric của horizon tương ứng; không coi là mẫu âm. |
| Biến cố sát cuối ca | Chỉ xác nhận khi có đủ 60 giây dữ liệu cho nhãn; báo cáo phần thời gian không đánh giá được. |

**“Có biến cố trong 5 phút tới” khác “cảnh báo trước ít nhất 5 phút”.** y5 có thể đúng dù chỉ báo trước 30 giây. Vì vậy:

- Đo lead time thực tế theo từng biến cố, không dùng tên horizon để khẳng định thời gian báo trước.
- Với y10, báo cáo tỷ lệ biến cố có cảnh báo trong khoảng từ 10 đến 5 phút trước khởi phát; biến cố chỉ có cảnh báo muộn hơn được tính riêng.
- Nếu cần cam kết nghiên cứu báo trước ít nhất 10 phút, bổ sung horizon 15 phút và đo cảnh báo trong khoảng 15–10 phút trước khởi phát.
- Có thể tái lập phép đánh giá tại mốc cố định 5/10 phút của paper ở bảng phụ; bảng chính phải là phát lại liên tục.

Chốt quy tắc gộp các đợt sát nhau, tái kích hoạt và matching trước test. Các thay đổi ngưỡng MAP 60/65 hoặc độ dài 30/60/120 giây là sensitivity analysis đã đăng ký, không chọn lại sau khi thấy kết quả đẹp.

## 5. Tiền xử lý và chống rò rỉ

1. Kiểm kê track và timestamp trên 50–100 ca trước; lập cohort manifest, missingness, số biến cố và thời gian đủ điều kiện.
2. Chia khoảng 70/15/15 theo subjectid thành train/validation/test rồi mới sinh window; giữ mọi ca của một người trong cùng split. Cố định seed và lưu manifest.
3. Dùng 1 phút/5 phút/10 phút lịch sử để tính current, mean, standard deviation, minimum, maximum, slope và biến thiên; tất cả rolling phải kết thúc tại t.
4. Xử lý artefact bằng chất lượng tín hiệu, sự liên tục và đối chiếu track khi có. Không loại mọi giá trị thấp chỉ vì chúng hiếm; chúng có thể là biến cố cần dự báo.
5. Numeric đầu vào chỉ forward-fill có giới hạn, ban đầu tối đa 30 giây, kèm mask và time-since-last-observed. Không backfill hoặc nội suy xuyên qua t.
6. Tạo nhãn theo MAP gốc và khoảng thời gian có dữ liệu hợp lệ. Không dùng chuỗi đã fill 30 giây để chứng minh tụt liên tục 60 giây. Xác định ngưỡng gián đoạn theo nhịp cập nhật thực tế; đoạn không đủ bằng chứng là unknown.
7. Fit scaler, imputer, giới hạn thống kê và lựa chọn đặc trưng chỉ trên train. Nếu chuẩn hóa trong cửa sổ, giữ MAP tuyệt đối hoặc các thống kê mức nền để không làm mất ý nghĩa ngưỡng 65.
8. Loại biến chứa tương lai: tổng mất máu/dịch cuối ca, tổng liều cuối ca, độ dài ca thực tế, biến cố sau mổ. Với xét nghiệm, dùng thời điểm kết quả sẵn có.
9. Không cân bằng 50/50 validation/test. Nếu lấy mẫu hoặc weighting khi train, hiệu chỉnh xác suất trên dữ liệu validation có tỷ lệ tự nhiên.
10. Giữ tất cả thời điểm đủ điều kiện khi phát lại test, gồm cả các đoạn MAP 65–75. Không chọn riêng đoạn “rất ổn định” làm mẫu âm.

Thuốc xử trí trong tương lai có thể ngăn biến cố xuất hiện. Giai đoạn đầu mô hình dự báo kết cục quan sát được dưới thực hành điều trị hiện tại. Nếu có timeline thuốc, báo cáo phân tầng và phân tích cạnh tranh với can thiệp; không đổi mọi cảnh báo trước dùng vận mạch thành true positive, cũng không dùng thuốc tương lai làm feature.

## 6. Ma trận mô hình phù hợp T4

Đây là cấu hình khởi đầu do dự án đề xuất, không phải cấu hình nguyên bản hoặc số VRAM đã đo của paper.

| ID | Mô hình | Cấu hình / mục đích | Ưu tiên |
|---|---|---|---|
| B0 | MAP hiện tại | Dùng -MAP làm risk score; ngưỡng chọn trên validation. | Bắt buộc |
| B1 | Logistic regression | MAP hiện tại + slope + variability; L2, chuẩn hóa từ train. | Bắt buộc |
| B2 | LightGBM | Khoảng 50–150 đặc trưng lịch sử; 15–31 leaves, early stopping; CPU. | Baseline mạnh, rẻ |
| D1 | TCN | 7 residual block, mỗi block 2 causal Conv1D kernel 3; dilation 1,2,4,8,16,32,64; 32–64 channel; dropout 0,1–0,2. Receptive field 509 bước, đủ phủ 300 bước. | Mô hình DL chính |
| D2 | Transformer nhỏ theo patch | Patch 10–15 bước; d_model 64/128, 2–3 layer, 4 head; fusion static qua MLP. | Đối chứng hiện đại |
| D3 | 1D ResNet/CNN waveform + numeric | 30–60 giây ABP, thử 100/125 Hz; encoder nén trước khi fusion. | Chỉ sau khi D1/D2 và evaluator ổn định |
| D4 | HMF hoặc SAFDNet | Dùng code tác giả sau audit; bắt buộc cùng split và nhãn của UC04. | Tùy ngân sách |

TCN và Transformer dùng BCE cho hai horizon. Có thể bảo đảm p10 ≥ p5 bằng tham số hóa p10 = p5 + (1 − p5) × q, với q trong [0,1]; vẫn phải kiểm tra lại tính đơn điệu sau calibration. Nếu chọn hai head độc lập, báo cáo vi phạm và xử lý nhất quán trên validation.

Bắt đầu bằng BCE chuẩn và batch theo prevalence tự nhiên. Chỉ thử class weighting/focal loss khi có thất bại rõ ràng; luôn đánh giá lại calibration. Forecast MAP hoặc auxiliary loss là ablation về sau, vì MAE thấp không tự chứng minh dự báo đúng biến cố hiếm.

Đóng góp khả thi cho đề tài: **mô hình nhỏ với đánh giá liên tục không chọn lọc, mức cảnh báo giả được kiểm soát, ablation chất lượng dữ liệu và kiểm định ngoài**. Tính mới phải kiểm tra tiếp trên kết quả thực nghiệm; chưa tuyên bố novelty hoặc SOTA trước benchmark.

## 7. Kế hoạch sử dụng Kaggle

- Dùng một T4 16 GB, PyTorch FP16 autocast + GradScaler; không mặc định cần hai GPU.
- TCN/Transformer numeric thử batch 64; tăng/giảm sau đo. Waveform thử batch 16, giảm còn 8 nếu cần. Mục tiêu vận hành là giữ peak allocated VRAM dưới khoảng 12 GB để có dư địa.
- Epoch tối đa 30, early stopping patience 5; giới hạn số window/epoch lúc thăm dò. Chạy một seed để loại phương án yếu, sau đó 3 seed cho hai mô hình tốt nhất.
- Lưu mảng theo ca bằng float32 và index window; đọc lazy/memmap, không lưu lặp toàn bộ overlapping windows.
- Tách notebook download/preprocess trên CPU khỏi notebook GPU. GPU chỉ bật khi thực sự train/inference.
- Pin phiên bản môi trường sau smoke test, lưu checkpoint, optimizer, seed, split hash, config và lịch sử metric sau mỗi epoch để có thể resume.
- Ghi thời gian/epoch, peak VRAM, RAM, dung lượng cache và tốc độ inference bằng phép đo. Chưa có căn cứ để hứa thời gian train hoặc giá tiền cụ thể.
- Hạn mức Kaggle có thể thay đổi theo tài khoản; dùng mục quota thực tế trong tài khoản làm giới hạn, không lập kế hoạch dựa trên một mức giờ miễn phí cố định.

Ước lượng lưu trữ minh họa: 3.000 ca × 3 giờ × 0,5 Hz × 12 numeric × 4 byte ≈ 0,78 GB, chưa gồm mask, timestamp và overhead. Cùng giả định, một ABP 100 Hz ≈ 12,96 GB; bốn waveform ≈ 51,84 GB. Đây là phép tính dung lượng mảng theo giả định, không phải kích thước tải thật của VitalDB. Mở rộng grid không đồng nghĩa có thêm thông tin đo.

Ngân sách thực nghiệm do mình đề xuất: khoảng 20–30 GPU-giờ cho vòng numeric đầu; đây là **trần cho một vòng thăm dò**, không phải giới hạn tổng ngân sách do người dùng đặt ra. Dùng subset train để sàng lọc; ứng viên cuối được huấn luyện trên toàn bộ train đủ điều kiện trong giới hạn tài nguyên thực tế. Nếu chưa đạt chất lượng, lập vòng cải tiến dữ liệu, waveform hoặc mô hình tiếp theo; không hạ tiêu chí để tuyên bố hoàn thành. Ngân sách vòng tiếp theo được dự toán từ thời gian và VRAM thực đo.

## 8. Đánh giá và tiêu chí chọn mô hình

### Yêu cầu định lượng về độ chính xác cao

Các ngưỡng sau là **mục tiêu nghiên cứu đề xuất để cụ thể hóa yêu cầu của người dùng**, chưa được chứng minh khả thi trên cohort này và không phải chuẩn nghiệm thu lâm sàng đã được công nhận. Yêu cầu độ chính xác cao là bắt buộc; các trị số cụ thể được khóa trong protocol trước khi huấn luyện/chọn mô hình. Không đổi ngưỡng sau khi xem test để biến kết quả không đạt thành đạt.

| Chỉ tiêu | Biến cố khởi phát trong 5 phút tới (y5) | Biến cố khởi phát trong 10 phút tới (y10) |
|---|---|---|
| AUROC theo cửa sổ | ≥ 0,95 | ≥ 0,90 |
| Độ nhạy theo biến cố, sau chính sách cảnh báo | ≥ 90% | ≥ 85% |
| PPV theo episode cảnh báo | ≥ 70% | ≥ 60% |
| Số episode cảnh báo giả/giờ đủ điều kiện | ≤ 0,5 | ≤ 0,5 |
| Sai số hiệu chỉnh ECE, cách chia bin khóa trước | ≤ 0,05 | ≤ 0,05 |

Mỗi cột phải đạt đồng thời tại một cấu hình mô hình, calibrator, ngưỡng và chính sách cảnh báo đã khóa cho horizon đó. Không ghép độ nhạy ở một ngưỡng với PPV ở ngưỡng khác. Hai luồng được đánh giá riêng; nếu giao diện hợp nhất hai luồng, phải đánh giá lại PPV và số cảnh báo giả của luồng hợp nhất.

Điều kiện bổ sung: độ phủ dự báo ≥ 90% thời gian đủ điều kiện theo protocol; với luồng y10, mục tiêu ≥ 80% biến cố đủ điều kiện có cảnh báo trong khoảng 10–5 phút trước khởi phát. Báo cáo cả số biến cố thiếu lịch sử và kết quả toàn bộ ca để làm rõ phạm vi đánh giá. Các chỉ tiêu trong bảng đo dự báo **trong** một horizon, không tự chứng minh cảnh báo **trước** đúng 5 hoặc 10 phút. Mục tiêu báo trước ít nhất 10 phút cần nhánh horizon 15 phút như mục 4.

Accuracy tổng thể vẫn được báo cáo nhưng không dùng làm tiêu chí duy nhất: nếu 95% cửa sổ không biến cố, luôn dự đoán “không biến cố” cũng đạt accuracy 95% dù bỏ sót mọi biến cố. AUROC 0,95 không có nghĩa 95% cảnh báo là đúng; tỷ lệ cảnh báo đúng được đo bằng PPV.

Các ngưỡng áp dụng cho ước lượng điểm trên test nội bộ tách bệnh nhân; mọi chỉ tiêu kèm khoảng tin cậy 95% và mẫu số. Nếu khoảng tin cậy rộng hoặc cắt ngưỡng mục tiêu, ghi rõ bằng chứng còn chưa chắc chắn. Kiểm định ngoài báo cáo riêng cùng bộ chỉ tiêu; chỉ tuyên bố đạt ở cơ sở ngoài khi có kết quả đo tại đó.

### Giao thức đánh giá

Chia validation theo bệnh nhân thành phần tuning và calibration/threshold nếu số biến cố cho phép; nếu không, dùng dự đoán out-of-fold trong development. Không dùng test để chọn mô hình, calibrator, ngưỡng hay cooldown.

| Nhóm metric | Cần báo cáo |
|---|---|
| Theo cửa sổ | AUROC; average precision/AUPRC với định nghĩa tính rõ ràng; prevalence; Brier score; reliability plot và calibration slope/intercept. |
| Theo biến cố | Độ nhạy biến cố; độ nhạy khi báo trước ≥ 5 phút; lead time median/IQR; số biến cố bỏ sót. |
| Theo cảnh báo | PPV theo episode cảnh báo; false alert episodes/eligible monitored hour; tổng cảnh báo/giờ; tỷ lệ thời gian ở trạng thái cảnh báo. |
| Độ phủ | Tỷ lệ thời gian mô hình đủ dữ liệu để dự báo; số ca/biến cố bị loại và nguyên nhân. |
| Độ vững | Kết quả theo nhóm tuổi, ASA, loại mổ, missingness và MAP hiện tại; số ca/số biến cố của mỗi nhóm. |
| Chi phí | Peak VRAM, thời gian train, độ trễ inference và dung lượng input. |

Chính sách cảnh báo ban đầu: vượt ngưỡng hai lần liên tiếp; gộp thành một episode; cooldown 5 phút; tắt hoặc chuyển trạng thái khi có IOH/thiếu tín hiệu. Đây là tham số nghiên cứu phải tune trên validation. Báo cáo cả risk score thô và hiệu quả sau chính sách để thấy độ trễ do persistence/cooldown.

Matching: một cảnh báo ghép với khởi phát đầu tiên trong horizon đã khai báo; đếm mỗi biến cố một lần trong event sensitivity. Một episode cảnh báo không ghép được là false alert; báo cáo rõ xử lý episode kéo dài và các lần nhắc lại. Chuẩn hóa false alerts theo thời gian đủ điều kiện dự báo, đồng thời công bố tổng thời gian theo dõi và coverage để tránh cải thiện giả bằng cách từ chối quá nhiều.

Chọn ngưỡng trên validation bằng đường cong event sensitivity theo false alerts/hour, đồng thời kiểm tra PPV và các điều kiện ở bảng mục tiêu. Mốc 1 cảnh báo giả/giờ chỉ là điểm so sánh phụ, không thay thế yêu cầu ≤ 0,5 cảnh báo giả/giờ. Nếu không có ngưỡng đạt đồng thời các mục tiêu, ghi nhận chưa đạt và cải tiến trong development; không nới tiêu chí sau khi xem test.

Bootstrap khoảng tin cậy 95% theo subjectid, ví dụ 1.000 lần; bootstrap cặp khi so mô hình trên cùng bệnh nhân. Không coi hàng triệu window chồng lấp là hàng triệu quan sát độc lập.

Ablation tối thiểu: MAP-only; numeric đa biến; +static; +mask/time-since; +calibration; +chính sách cảnh báo. Nếu có waveform, so numeric và numeric+waveform trên cùng tập giao bệnh nhân, đồng thời báo cáo coverage của toàn cohort.

Ưu tiên đạt đồng thời các mục tiêu chất lượng, sau đó mới tối ưu tài nguyên. Trong các mô hình đạt yêu cầu, chọn mô hình đơn giản hơn khi hiệu năng tương đương theo biên so sánh khóa trước. Nếu LightGBM đạt mục tiêu và tương đương DL, có thể chọn LightGBM. Nếu chưa mô hình nào đạt, chỉ có bản thử nghiệm kỹ thuật; thử waveform, fusion hoặc ensemble nhỏ phù hợp T4 và kiểm tra lại trên holdout chưa sử dụng.

## 9. Phân tách nguyên nhân và độ tin cậy

UC04 trong tài liệu gốc gồm hai bài toán khác nhau: dự báo biến cố và gợi ý cơ chế. Nhánh nguyên nhân tiếp tục sau pipeline dự báo, với điều kiện dữ liệu/gán nhãn riêng.

| Mốc | Đầu ra | Dữ liệu / cách kiểm chứng |
|---|---|---|
| A — MVP nghiên cứu | Xác suất đã calibration, diễn biến MAP/HR và chất lượng tín hiệu; đặc trưng đóng góp chỉ giải thích dự báo. | Reliability plot, Brier, subgroup performance; nếu dùng ensemble nhỏ, đánh giá disagreement riêng, không gọi đó là xác suất đúng của từng ca. |
| B — Tập nhãn cơ chế | Các nhãn có thể đồng thời: giảm tiền tải/thiếu dịch, giãn mạch, giảm co bóp, mất máu; thêm hỗn hợp/không đủ bằng chứng. | Timeline thuốc, dịch, mất máu, Hb, phẫu thuật; CO/SV hoặc siêu âm khi có. Bác sĩ xây rubric, hai người gán độc lập và hội chẩn bất đồng. |
| C — Gợi ý có kiểm chứng | Danh sách cơ chế nghi ngờ và bằng chứng quan sát được, có trạng thái không kết luận. | Đánh giá agreement, macro-F1, precision từng nhãn, calibration và coverage; kiểm định trên ca tách biệt. |

Không ép bốn nguyên nhân thành nhãn loại trừ lẫn nhau: mất máu có thể gây giảm tiền tải và nhiều cơ chế có thể cùng tồn tại. Không dùng việc bác sĩ đã truyền dịch hoặc dùng vận mạch làm nhãn nguyên nhân duy nhất. Dữ liệu bổ sung là yêu cầu thiết kế cần thống nhất với chuyên gia, không phải lời khẳng định VitalDB đã có nhãn cơ chế đủ tin cậy.

Đề xuất pilot gán 100–200 biến cố để kiểm tra tính khả thi của rubric và độ đồng thuận; đây không phải cỡ mẫu đảm bảo huấn luyện/kiểm định đủ mạnh. Chỉ dự toán số nhãn tiếp theo sau khi biết prevalence, mức bất đồng và độ rộng CI mong muốn.

SHAP, attention và gradient attribution không chứng minh quan hệ nhân quả. Các điểm giải thích của MVP không được trình bày như chẩn đoán cơ chế hoặc khuyến nghị liều. Khuyến nghị điều trị cá thể hóa/PK-PD không nằm trong kết quả nghiệm thu của nhánh dự báo này.

## 10. Lộ trình 10 tuần và điều kiện hoàn thành

Ước lượng cho một người đã có nền tảng Python/ML, dùng dữ liệu công khai; thời gian phê duyệt và tiếp cận dữ liệu ngoài không nằm trong 10 tuần này.

| Tuần | Công việc | Deliverable / điều kiện qua mốc |
|---|---|---|
| 1 | Đọc P0, audit 50–100 ca, chốt endpoint và split. | Protocol v1; data dictionary; cohort manifest; hình timeline nhãn. |
| 2 | Pipeline numeric, QC, window index, kiểm tra leakage và nhãn. | Cache tái sử dụng; subjectid không trùng; nhãn được kiểm tra thủ công trên ca mẫu. |
| 3 | B0/B1/B2 và evaluator phát lại liên tục. | Bảng baseline, PPV/FAH/event recall; xác minh matching trước DL. |
| 4–5 | TCN và Transformer nhỏ; giới hạn trial và đo T4. | Checkpoint, config, resource log; chọn hai ứng viên bằng validation. |
| 6 | Calibration, cảnh báo, ablation và 3 seed cho ứng viên cuối. | Chốt mô hình và chính sách, đóng băng trước test. |
| 7 | Chạy test một lần theo protocol; CI và phân tích lỗi. | Báo cáo chính + phân nhóm + ví dụ false/missed alerts. |
| 8 | Nhánh waveform nhỏ nếu được mốc validation ủng hộ; nếu không, tăng robustness/reproducibility. | So sánh cùng cohort hoặc báo cáo lý do dừng waveform. Mọi mô hình phát sinh dùng holdout mới hoặc được đánh dấu exploratory. |
| 9 | Notebook phát lại ca, biểu đồ nguy cơ/chất lượng/cảnh báo; đóng gói chạy lại. | Demo từ dữ liệu thực đã ẩn danh; không giả lập số điểm mô hình. |
| 10 | Viết báo cáo, model card, data card và kế hoạch kiểm định ngoài/nhãn cơ chế. | Bộ kết quả đủ truy vết từ raw manifest đến bảng báo cáo. |

Nếu muốn đưa waveform vào so sánh xác nhận ngay trong bản đầu, chuyển thử nghiệm waveform lên tuần 5–6 và dời việc mở test đến khi mọi ứng viên đã khóa. Không dùng kết quả test tuần 7 để phát minh rồi chọn mô hình tuần 8 trên cùng test.

Định nghĩa hoàn thành bản thử nghiệm kỹ thuật: pipeline chạy lại trên Kaggle; tối thiểu ba baseline và một mô hình DL được đánh giá công bằng; có calibration, metric theo biến cố/cảnh báo và demo phát lại; công bố cả kết quả không đạt. **MVP đáp ứng yêu cầu hiệu năng phải đạt đồng thời các mục tiêu đã khóa ở mục 8**; hoàn thành pipeline hoặc hết 10 tuần chưa đồng nghĩa đạt độ chính xác yêu cầu. Nếu chưa đạt, báo cáo chỉ tiêu thiếu và kế hoạch thực nghiệm tiếp theo. Đạt MVP hồi cứu chưa đồng nghĩa hoàn thiện UC04 lâm sàng.

Để hoàn thiện UC04: kiểm định ngoài với mô hình/ngưỡng đã khóa → thu dữ liệu bệnh viện và đánh giá tiến cứu ở chế độ im lặng → kiểm chứng nhánh cơ chế → nghiên cứu tác động của cảnh báo cùng nhóm lâm sàng. Cỡ mẫu mỗi giai đoạn dựa trên số biến cố và độ chính xác ước lượng, không chỉ một mốc tổng số ca.

## 11. Cấu trúc triển khai hiện tại

```text
notebooks/
  01_uc04_pilot.ipynb
  02_numeric_sequences.ipynb
  03_tcn_transformer.ipynb
  04_case_replay.ipynb
src/safeanes/
  config.py          # protocol, tracks, validity bounds
  data.py            # cohort, subject split, raw provenance
  signals.py         # causal sampling và nhãn
  dataset.py         # windows, feature baseline và QC
  sequences.py       # mảng theo ca, memmap, scaler fit-only
  experiment.py      # baseline và calibration riêng bệnh nhân
  models.py          # TCN/Transformer, ordered heads, masked BCE
  training.py        # AMP, early stopping, calibration, resume
  evaluation.py      # alarm replay, matching, metric, bootstrap
  reporting.py       # báo cáo, calibration/subgroup, hình replay
  cli.py
configs/
  tcn.json
  transformer.json
tests/
docs/
reports/
artifacts/  # manifest, metrics, configs, checkpoints; dữ liệu lớn lưu riêng
```

Các kiểm thử đã có: không trùng subjectid; feature/sequence/causal convolution không đọc tương lai; sự kiện sát ranh giới 5/10 phút; biến cố dài đúng 60 giây; missingness không bị gán âm; nhãn hai horizon nhất quán; matching/cooldown không đếm lặp; normalizer chỉ fit ca fit; ordered probability sau calibration; khôi phục checkpoint và resume cho kết quả CPU khớp lần chạy liên tục. Chạy bằng `python -m pytest -q` với dependencies phù hợp. Kiểm thử phần mềm không thay thế audit nhãn bằng chuyên gia hoặc xác nhận chất lượng dự báo.

## 12. Giới hạn tra cứu và đầu vào để cập nhật

Tra cứu ưu tiên bài gốc, trang nhà xuất bản, PubMed/arXiv và repository tác giả. Firecrawl Research CLI bị lỗi kết nối EACCES trong môi trường này nên đã dùng tìm kiếm web và đọc nguồn trực tiếp. Một số toàn văn PubMed/ScienceDirect/PMC không mở ổn định; các hạn chế xác minh đã nêu ở từng mục. Mã nguồn liên kết chưa được thực thi; không bảo đảm tái lập chỉ từ việc repository tồn tại.

```yaml
workflow: firecrawl-research-papers
topic: intraoperative hypotension prediction; selection bias; continuous alarm evaluation
scope: SafeAnes UC04
date: 2026-09-17
hardware: single NVIDIA T4 16 GB
data_default: VitalDB
output: Vietnamese research and implementation plan
```

## 13. Trạng thái bàn giao và các mốc còn thiếu

Cập nhật đợt **E03 trong v0.2**: Inception-style CNN, TimesNet và ensemble theo [review](docs/SOURCES.md#review-e03), [plan lưu trữ](docs/versions/V0_3_PLAN.md), [báo cáo](reports/v0_3/REPORT.md). **E04** đã chạy CatBoost ba seed và ablation ngưỡng, có cải thiện từng chỉ tiêu nhưng chưa đạt mọi gate: [kết quả](reports/v0_4/REPORT.md#nhan-xet). **E05** mở rộng development và ablation; các experiment không tự tạo release mới. Xem [quy ước version](README.md#release).

Không đánh dấu toàn bộ nghiên cứu hoàn thành chỉ vì đã có code. Trạng thái dưới đây phân biệt sản phẩm đã triển khai với bằng chứng thực nghiệm còn cần bổ sung.

| Hạng mục | Trạng thái | Bằng chứng / điều kiện còn thiếu |
|---|---|---|
| Pilot dữ liệu thật và provenance | Pilot 60 ca; E05 đã chạy 300 ca/297 bệnh nhân, 99.132 decision, 497 episode IOH | Raw hashes, manifest, quality; [E05](reports/E05/REPORT.md). |
| Cohort đúng phạm vi kế hoạch | Đã đối chiếu nguồn và metadata E05 | VitalDB gốc mô tả non-cardiac; kiểm kê chuyên khoa và cờ tên thủ thuật; chưa adjudication lại mã thủ thuật độc lập. |
| Baseline MAP/logistic/boosting | Đã chạy CPU | Baseline hiện chưa đạt mọi mục tiêu mục 8. |
| Dataset chuỗi và TCN/Transformer | Đã triển khai và chạy pilot CPU | Cùng nhãn, vai trò bệnh nhân và evaluator; chưa phải phép đo T4. |
| Calibration, alarm và CI | Đã có trong pipeline | Calibration DL bảo toàn p10≥p5; CI bootstrap bệnh nhân; exposure còn xấp xỉ 30 giây. |
| Notebook, checkpoint/resume và replay | Đã có code, kiểm thử và output local | Notebook Kaggle cần chạy thực trên tài khoản có T4 và lưu log môi trường. |
| Độ chính xác yêu cầu | Chưa nghiệm thu | Đọc bảng metric/gates từng run; không dùng AUROC đơn lẻ để đánh dấu đạt. |
| Ba seed, ablation và cohort lớn | E04/E05 đã chạy CatBoost ba seed và E05 MAP-only/numeric/+static | 16 model theo horizon trên 300 ca; mask/time-since ablation sequence chưa chạy đủ. |
| Evaluator cho nghiên cứu xác nhận | Đã bổ sung CI lead time/subgroup mô tả E05; còn thiếu phần xác nhận | Exposure từng giây chưa triển khai; cần audit nhãn/eligibility và luồng cảnh báo hợp nhất nếu có. |
| Waveform | Chưa triển khai, có điều kiện | Cần bằng chứng validation và cohort giao để so lợi ích với chi phí. |
| Final test, kiểm định ngoài, tiến cứu | Chưa chạy | Final test giữ chưa sử dụng; cần protocol khóa và dữ liệu/quyền truy cập phù hợp. |
| Nhãn/cơ chế nguyên nhân | Chưa có bộ nhãn chuyên gia | Rubric và pilot gán nhãn như mục 9; không lấy attribution thay nhãn cơ chế. |

Lệnh thực thi và cách nối các notebook được ghi tại [SEQUENCE_RUNBOOK.md](docs/SEQUENCE_RUNBOOK.md). Kết quả không đạt phải dẫn tới vòng development mới có ghi nhận cohort/config/seed; mục tiêu nghiên cứu và giới hạn lâm sàng của kế hoạch được giữ nguyên.

<!-- consolidated:kiem-dinh -->
<a id="kiem-dinh"></a>

## Kế hoạch kiểm định còn phụ thuộc dữ liệu/môi trường ngoài — v0.2

### Kaggle/T4

Chạy các notebook hiện có trên T4 16 GB thực. Lưu GPU/driver/CUDA/PyTorch, config, source/data hashes, peak VRAM, epoch/inference time và artifact đầu ra. Đối chiếu CPU/AMP bằng sai số số học phù hợp; không thay benchmark T4 bằng thời gian CPU hay RTX3050. Hiện chưa có log T4 để nghiệm thu mốc này.

### Final test VitalDB

Chỉ mở sau khi quyết định pipeline/calibrator/threshold bằng development và khóa manifest, protocol, code/config. Nếu development chưa đạt mục tiêu, báo không đạt hoặc đăng ký vòng development mới; không mở final test để tìm cấu hình. Báo CI theo bệnh nhân, event denominator, coverage và FA/giờ; không chọn lại threshold từ kết quả test.

### Kiểm định ngoài

MOVER hoặc cohort bệnh viện cần quyền truy cập, ánh xạ track/unit/timestamp, tiêu chí cohort và kiểm tra nhãn riêng. Khóa rõ đánh giá zero-shot hay recalibration; nếu recalibration thì cần tập adaptation tách khỏi external test. Kiểm tra thiếu track/thiết bị/loại mổ và gộp lần mổ theo bệnh nhân. Không gọi chia ngẫu nhiên thêm VitalDB là external validation.

### Tiến cứu chế độ im lặng

Cần protocol được phê duyệt tại cơ sở, luồng dữ liệu thực, log thời gian trễ/mất dữ liệu và endpoint được kiểm tra độc lập. Ghi cảnh báo để đánh giá, không coi báo cáo hồi cứu hiện tại là bằng chứng hiệu quả can thiệp. Cỡ mẫu theo số biến cố và độ rộng CI mong muốn.

### Nhánh cơ chế

Dùng [bản nháp gán nhãn](UC04_RESEARCH_PLAN.md#gan-nhan); cần chuyên gia hoàn thiện rubric và dữ liệu thực. Không suy cơ chế hoặc thuốc/liều từ attention/SHAP.

Các mốc trên được chuẩn bị về quy trình, chưa được đánh dấu đã thực nghiệm. Không có dữ liệu/quyền truy cập/log phần cứng tương ứng trong workspace để xác nhận hoàn thành.


<!-- consolidated:gan-nhan -->
<a id="gan-nhan"></a>

## Bản nháp quy trình gán nhãn cơ chế — v0.2

Tài liệu chuẩn bị cho nhánh B của kế hoạch UC04; **chưa có nhãn chuyên gia và chưa phải rubric được nhóm lâm sàng phê duyệt**. Không biến dự báo IOH, SHAP hoặc phản ứng với thuốc thành nhãn nguyên nhân.

### Đơn vị và nội dung gán nhãn

Một episode IOH theo protocol, có mã ca/mã episode và mốc onset. Hai bác sĩ xem cùng cửa sổ dữ liệu đã quy định trước, gán độc lập trước khi hội chẩn. Lưu rõ thông tin nào xảy ra trước onset, trong episode và sau can thiệp; nếu xây mô hình dự báo cơ chế, không để thông tin hậu nghiệm lọt vào input.

Mỗi nhãn nhận một trong `supported`, `not_supported`, `insufficient_evidence`: giảm tiền tải/thiếu dịch; giãn mạch; giảm co bóp; mất máu. Cho phép nhiều nhãn đồng thời, hỗn hợp và không kết luận. Danh sách nhãn xuất phát từ mục 9 của kế hoạch, không phải chẩn đoán tự động dựa trên ngưỡng sinh hiệu.

Mỗi quyết định phải dẫn tới mốc thời gian và loại bằng chứng: sinh hiệu, thuốc/dịch, mất máu/Hb, phẫu thuật, CO/SV/siêu âm khi có; ghi nguồn bị thiếu. Không coi “được truyền dịch” hay “được dùng vận mạch” là bằng chứng duy nhất xác nhận cơ chế.

### Quy trình pilot 100–200 episode

1. Khóa rubric cùng chuyên gia và thống nhất dữ liệu được xem; lưu revision, người phê duyệt, ngày hiệu lực.
2. Chọn episode theo bệnh nhân, mô tả loại mổ/missingness; không chỉ chọn những ca mô hình dự báo đúng.
3. Hai reviewer độc lập, không thấy dự báo hay attribution của mô hình và không thấy nhãn của nhau.
4. Ghi agreement từng nhãn, tỉ lệ insufficient evidence và bất đồng; hội chẩn có lý do, giữ nguyên cả hai nhãn ban đầu.
5. Báo prevalence, độ rộng CI theo bệnh nhân và quyết định tiếp tục/sửa rubric. 100–200 chỉ là pilot tính khả thi, không bảo đảm cỡ mẫu mô hình.
6. Chỉ sau khi dữ liệu và agreement phù hợp mới định nghĩa task ML, chia bệnh nhân và khóa bộ kiểm định cơ chế riêng.

### Schema đề xuất

`caseid, subjectid, episode_id, onset_seconds, reviewer_id, rubric_revision, review_time, preload_label, vasodilation_label, contractility_label, bleeding_label, mixed_label, evidence_timestamps, evidence_sources, missing_sources, confidence_category, disagreement_reason, adjudicator_id, adjudicated_labels`

Không điền dữ liệu mẫu vào kết quả thật. Nhãn confidence là tự đánh giá của reviewer, không phải xác suất đúng đã calibration. Mọi liên kết định danh và quyền truy cập phải theo quy trình quản trị dữ liệu của nghiên cứu.

### Đầu vào còn cần

Nhóm chuyên gia, timeline bằng chứng đủ tin cậy, quyền sử dụng dữ liệu và tập nhãn đã được kiểm tra. Các đầu vào này chưa có trong repository; không thể hoàn thành nhánh nguyên nhân chỉ bằng việc chạy model trên VitalDB numeric.

