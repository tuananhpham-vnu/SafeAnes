# Nguồn và cách sử dụng trong code UC04

Rà soát bổ sung ngày 18/09/2026: [E06 — TabM, TabICLv2 và TabPFN-3.5](SOTA_E06.md).

Ngày đối chiếu: 17/09/2026. Đây là sổ nguồn cho phiên bản 0.3; danh mục rộng hơn nằm trong [kế hoạch](../UC04_RESEARCH_PLAN.md) và [review v0.3](SOURCES.md#review-e03). Nguồn là bài gốc hoặc tài liệu chính thức. Không gán hiệu năng của paper cho code này.

| ID | Nguồn | Thông tin đã đối chiếu | Áp dụng / giới hạn |
|---|---|---|---|
| S01 | Lee HC, Park Y, Yoon SB, Yang SM, Park D, Jung CW. **VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients.** Scientific Data 9, 279 (2022). [DOI](https://doi.org/10.1038/s41597-022-01411-5). [Bản dữ liệu](https://physionet.org/content/vitaldb/1.0.0/), DOI 10.13026/czw8-9p62. | Dữ liệu gốc còn artefact; subjectid nhận diện các lần mổ của một người. | `data.py`: chia theo bệnh nhân; `dataset.py`: báo cáo chất lượng. |
| S02 | VitalDB, **Web API OpenDataset**. [API](https://vitaldb.net/docs/?documentId=API%2FWeb_API_OpenDataset.md), [parameter list](https://vitaldb.net/dataset/). | GET `/cases`, `/trks`, `/{tid}`; CSV gzip; thời gian giây từ casestart; dòng numeric thiếu bị bỏ; waveform có cách mã hóa thời gian khác. | `data.py`, `config.py`: giữ timestamp gốc và provenance; 7 numeric track; dùng ART_MBP, không thay bằng NIBP_MBP. |
| S03 | Yang HL et al. **The effect of selection bias on the performance of a deep learning-based intraoperative hypotension prediction model using real-world samples from a publicly available database.** BJA (2025). [PubMed](https://pubmed.ncbi.nlm.nih.gov/40404499/), DOI 10.1016/j.bja.2025.03.024. | Abstract báo cáo hiệu năng giảm khi kiểm tra trên mẫu ít thiên lệch. | Giữ mọi cửa sổ đủ điều kiện và đánh giá cảnh báo liên tục. Đã đọc abstract; không tuyên bố tái lập toàn bộ Methods. |
| S04 | Mulder MP et al. **Hypotension Prediction Index Is Equally Effective in Predicting Intraoperative Hypotension during Noncardiac Surgery Compared to a Mean Arterial Pressure Threshold: A Prospective Observational Study.** Anesthesiology (2024). [DOI](https://doi.org/10.1097/ALN.0000000000004990), [code tác giả](https://github.com/crph-utwente/HPIvalidation). | Cần so với baseline MAP đơn giản. | `experiment.py`: -MAP và logistic MAP/slope/variability; implementation riêng, không phải bản sao HPI. |
| S05 | scikit-learn, **Probability calibration**. [User guide](https://scikit-learn.org/stable/modules/calibration.html). | Calibration cần dữ liệu độc lập với dữ liệu fit; reliability curve và Brier đo các khía cạnh khác nhau. | Sigmoid calibration trên bệnh nhân riêng; ECE/Brier; không random CV theo window. |
| S06 | scikit-learn, **HistGradientBoostingClassifier**. [API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html). | Histogram gradient boosting hỗ trợ dữ liệu thiếu. | Baseline có sẵn local; tắt early stopping tự chia mẫu nội bộ. Không gọi mô hình này là LightGBM. |
| S07 | LightGBM, **LGBMClassifier**. [API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html). | Tham số estimator, số cây, num_leaves, regularization. | Lựa chọn bổ sung khi cài optional dependency; báo rõ khi chưa cài. |
| S08 | Zhu et al. **Transformer-based deep learning model for real-time prediction of intraoperative hypotension using dynamic time-series vital signs.** PLOS Medicine (2026). [Paper](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1005024). | Methods dùng MAP <65 kéo dài ≥1 phút và sinh hiệu dạng số. | Tham khảo endpoint; cách dựng nhãn/split code này là protocol riêng. |
| S09 | Bai S, Kolter JZ, Koltun V. **An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.** 2018. [arXiv](https://arxiv.org/abs/1803.01271). | Kiến trúc TCN cho chuỗi. | `models.py`: causal residual convolution, cấu hình riêng UC04; chưa có benchmark GPU. |
| S10 | PyTorch, [AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html), [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html). | Autocast/GradScaler; lưu state_dict mô hình và optimizer khi resume. | `training.py`: FP16 CUDA, checkpoint mỗi epoch, RNG và source/data hashes; CPU được kiểm thử riêng. |
| S11 | Nie Y et al. **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.** ICLR 2023. [arXiv](https://arxiv.org/abs/2211.14730). | Ý tưởng biểu diễn chuỗi theo patch trong kế hoạch gốc. | Transformer của UC04 dùng patch đa kênh và phân loại IOH; không phải bản tái lập đầy đủ PatchTST hoặc mô hình đã chứng minh hiệu năng lâm sàng. |

## Lựa chọn của dự án

Nguồn mới cho v0.3:

- **S12 — InceptionTime**, Fawaz et al., DMKD 2020: [paper](https://arxiv.org/abs/1909.04939), [repo tác giả](https://github.com/hfawaz/InceptionTime). `advanced_models.py` triển khai Inception-style nhỏ, một mạng, không phải ensemble/cấu hình gốc.
- **S13 — TimesNet**, Wu et al., ICLR 2023: [paper](https://arxiv.org/abs/2210.02186), [code chính thức](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py). UC04 dùng FFT/2D convolution nhưng chọn period theo từng sample và mean pooling; chi tiết khác biệt trong review.
- **S14 — iTransformer**, Liu et al., ICLR 2024: [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html), [repo](https://github.com/thuml/iTransformer). Đã rà soát, chưa tích hợp trong v0.3 vì bằng chứng gốc thuộc forecasting.

Ensemble bốn kiến trúc/trọng số logits đều là thiết kế dự án, không phải thuật toán được quy cho S12/S13. Firecrawl Research đã chạy được sau khi cho phép truy cập mạng trong phiên v0.3; những hạn chế full-text/citation graph được ghi tại review.

Các lựa chọn sau không được trình bày như chuẩn y khoa hay kết luận paper: gap label 10 giây; forward-fill input 30 giây; grid nhãn 1 giây; input 2 giây; history 600 giây; cadence 30 giây; recovery 120 giây; persistence 2 lần; cooldown 300 giây; 10 bin ECE; tỷ lệ chia tập và mục tiêu nghiệm thu. Xem [PROTOCOL.md](PROTOCOL.md).

GET API lưu URL, thời điểm UTC, SHA-256, kích thước cho từng response tại `data/vitaldb/raw/*.source.json`. `fetch_report.json`, `dataset.json`, `environment.json` nối nguồn dữ liệu với kết quả. Các file dữ liệu bệnh nhân không được đưa vào Git.

Firecrawl Research đã lỗi kết nối trong phiên tra cứu trước; phiên triển khai dùng công cụ web để đọc nguồn gốc và GET API công khai để xác minh schema thật. Không dùng snippet tìm kiếm để thay dữ liệu. Tài liệu nguồn có thể thay đổi; môi trường chạy được ghi riêng trong artifacts.

<!-- consolidated:review-e03 -->
<a id="review-e03"></a>

## Rà soát phương pháp mạnh cho UC04 — version 0.3

Ngày đối chiếu: 17/09/2026. Tìm bằng Firecrawl Research, sau đó đọc paper và repository tác giả. Đây là danh sách ứng viên có lý do lựa chọn, không phải bảng xếp hạng SOTA tuyệt đối: bài toán, cohort, split, prevalence và cách đếm cảnh báo khác nhau.

### Kết luận phục vụ triển khai

Version 0.3 thêm Inception-style CNN và TimesNet adaptation, rồi kết hợp với TCN/Transformer hiện có bằng trung bình logits cố định. Hai họ kiến trúc đã có bằng chứng mạnh trên phân loại chuỗi thời gian, dùng được với numeric hiện tại. Điều này **không chứng minh chúng đạt SOTA dự báo IOH**, và implementation nhỏ của UC04 khác cấu hình paper.

### Paper và code đã đối chiếu

| Phương pháp | Nguồn gốc và phạm vi bằng chứng | Quyết định UC04 |
|---|---|---|
| InceptionTime — Fawaz et al., DMKD 2020 | CNN nhiều độ dài kernel; ensemble các mạng; thử nghiệm phân loại UCR/UEA. Tác giả báo cáo độ chính xác tương đương HIVE-COTE trong thiết lập paper với khả năng mở rộng tốt hơn. [Paper](https://arxiv.org/abs/1909.04939), [repo tác giả](https://github.com/hfawaz/InceptionTime). | Chọn một mạng Inception-style nhỏ trong E03 (v0.2). Không gọi là tái lập đầy đủ ensemble InceptionTime. Repo tham khảo ghi GPL-3.0; không chép/vendoring source Keras vào dự án. |
| TimesNet — Wu et al., ICLR 2023 | Tìm đa chu kỳ, biến chuỗi 1D thành tensor 2D, mô hình hóa biến thiên trong/giữa chu kỳ. Có đánh giá classification bên cạnh forecasting, imputation và anomaly detection; tuyên bố SOTA của tác giả gắn với benchmark thời điểm công bố. [Paper](https://arxiv.org/abs/2210.02186), [repo](https://github.com/thuml/TimesNet), [implementation chính thức](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py). | Chọn adaptation nhỏ. FFT chọn chu kỳ riêng từng sample thay vì trung bình batch trong code tham khảo; dùng mean pooling để giảm số tham số. Repo Time-Series-Library ghi MIT. |
| iTransformer — Liu et al., ICLR 2024 Spotlight | Chuỗi lịch sử mỗi biến được nhúng thành một token; attention học quan hệ giữa biến. Kết quả chính thuộc multivariate forecasting, không phải onset classification IOH. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html), [repo MIT](https://github.com/thuml/iTransformer). | Ứng viên version sau; ít kênh và classification head của UC04 cần adaptation và kiểm chứng riêng. |
| Zhu et al. — PLOS Medicine, 25/03/2026 | Trực tiếp dự báo IOH từ numeric; Conv1D + positional encoding + Transformer + pooling. Paper báo cáo nội bộ AUC 0,904/0,892/0,882 và ngoài 0,897/0,862/0,822 ở 5/10/15 phút. Dữ liệu phát triển bệnh viện không công khai đầy đủ. [Paper](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1005024), [code](https://github.com/ShouqiangZhu/IOH_Transformer). | Chưa lấy nguyên repo vào benchmark. Cần giải quyết mâu thuẫn cấu hình giữa Methods và code. Kết quả paper không so trực tiếp với event recall/FAH của pilot. |
| HMF — Cheng et al., arXiv 2409.11064, bản v4 | Forecast MAP/SBP bằng decomposition, normalization và Transformer; nội dung đọc được mô tả chia timeline trong từng bệnh nhân. IOH suy ra từ chuỗi MAP dự báo, khác onset classifier hiện tại. [Bản đã đọc](https://arxiv.org/html/2409.11064v4), [repo tác giả](https://github.com/ustc-time-series/HMF). | Để vòng forecasting riêng, sửa split theo subjectid trước khi benchmark UC04. Không dùng điểm số HMF để tuyên bố tổng quát hóa sang bệnh nhân mới. |
| HypoBridCast — Wang et al., BMC Anesthesiology 2026 | Conv1D + Transformer, waveform và biến trước mổ; công bố cohort VitalDB và kiểm định ngoài. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13340010/). | Chưa phù hợp input numeric hiện có. Đưa vào nhánh waveform khi có cache/QC và so sánh cùng cohort; không giả dữ liệu waveform từ numeric. |

### Audit code trước khi tích hợp

Trong [code TimesNet](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py), `FFT_for_Period` lấy trung bình amplitude theo batch để chọn frequency. Với mục tiêu một dự báo cho một bệnh nhân, UC04 chọn frequency riêng từng mẫu. Đây là khác biệt implementation có chủ ý, được kiểm thử bằng so output khi chạy một mình và khi ghép batch.

Repo Zhu được rà soát có mặc định depth/width/head không khớp mô tả bài. `transformer_model.py` truyền tensor dạng batch/time/feature nhưng `TransformerEncoderLayer` không chỉ định `batch_first=True`. **Suy luận từ code:** có nguy cơ attention chạy theo trục batch nếu không có bước chuyển trục tương ứng. Cần xác nhận tác giả hoặc tái triển khai kiểm chứng; không gọi code đó là mô hình đã tái lập. Danh sách file gốc không thấy LICENSE tại thời điểm kiểm tra; chưa xác minh quyền tái phân phối toàn repo. [Nguồn code](https://github.com/ShouqiangZhu/IOH_Transformer).

Implementation `advanced_models.py` là bản nhỏ viết cho input và loss của dự án; không nhập nguyên các repository bên ngoài. Thông số khác paper được khóa trong kế hoạch E03 (v0.2) (đã xóa). Ensemble bốn kiến trúc là thiết kế thí nghiệm UC04, không gán cho tác giả InceptionTime hoặc TimesNet.

### Giới hạn và lần tra cứu sau

Firecrawl ban đầu bị EACCES trong sandbox; truy cập mạng được cho phép và search-papers chạy được. Related-paper expansion từ PMC13046278 không phân giải được seed trong citation graph; read-paper không có full-text Zhu, nên dùng PLOS/PMC và GitHub làm đối chiếu. HMF có full-text passages. OpenReview hiển thị bước xác minh trình duyệt; dùng arXiv và repo tác giả thay thế. Chưa tái lập kết quả gốc của bất kỳ paper nào.

```yaml
workflow: firecrawl-research-papers
topic: intraoperative hypotension; numeric time series classification; strong backbones
queries:
  - intraoperative hypotension prediction numeric vital signs transformer
  - InceptionTime time series classification official paper
  - TimesNet classification
  - iTransformer multivariate forecasting
version: UC04-E03 (v0.2)
output: Vietnamese source review plus versioned measured benchmark
```


<!-- consolidated:review-e04 -->
<a id="review-e04"></a>

## Nguồn và lựa chọn phương pháp E04 (v0.2)

Tra cứu ngày 2026-09-17 qua Firecrawl Research và trang gốc. Đây là rà soát bổ sung cho [E03 (v0.2)](SOURCES.md#review-e03), không phải tuyên bố xác định mô hình SOTA IOH mới nhất.

| Nguồn | Bằng chứng dùng được | Quyết định |
|---|---|---|
| [CatBoost, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html) | Họ gradient boosting với ordered boosting và xử lý categorical. | Thêm CatBoost như baseline mạnh khác LightGBM. Thí nghiệm hiện dùng numeric features, không tuyên bố tái lập các thí nghiệm categorical của paper. |
| [CatBoost parameters](https://catboost.ai/docs/en/references/training-parameters/common) | SqrtBalanced tính trọng số lớp từ căn bậc hai tỉ lệ tổng trọng số lớp; hỗ trợ CPU. | Ablation None/SqrtBalanced, calibration trên phân bố tự nhiên. Cấu hình runner là lựa chọn dự án, không gọi toàn bộ pipeline là thuật toán mới. |
| [Grinsztajn et al., NeurIPS 2022](https://proceedings.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html) | Boosted trees là đối chứng cần thiết trên các benchmark tabular vừa và nhỏ. | Không tiếp tục chỉ tăng độ phức tạp DL khi số bệnh nhân rất ít. Không suy rộng thứ hạng benchmark sang IOH. |
| [TabPFN, Nature 2025](https://www.nature.com/articles/s41586-024-08328-6) | Foundation model cho bảng, khác mô hình sequence đã chạy. | Ứng viên vòng sau; chưa tải weights hoặc benchmark nên không ghi kết quả dự kiến vào bảng thực nghiệm. |
| [TabICL, 2025](https://arxiv.org/abs/2502.05564) | In-context learning cho bảng lớn hơn, hướng tiếp cận khác boosted trees. | Cần thí nghiệm riêng với fit/calibration/validation tách bệnh nhân và chi phí inference thực tế. Chưa triển khai E04 (v0.2). |
| [Imbalanced clinical tabular benchmark, arXiv:2512.21602](https://arxiv.org/abs/2512.21602) | Bản nội dung truy cập so sánh DT/RF/XGBoost/TabNet/TabICL/TabPFN trên MIMIC-IV-ED và eICU; kết luận thứ hạng phụ thuộc tập dữ liệu. Foundation models ở chế độ inference, khác việc tối ưu mô hình trainable. | Preprint bổ trợ lựa chọn phương pháp; không phải bài dự báo IOH, không dùng Macro-F1 hoặc thứ hạng trong bài như kết quả SafeAnes. |

### Thay đổi ngưỡng là một giả thuyết riêng

Risk đã calibration có thể tập trung trong khoảng nhỏ. Lưới validation quantile bổ sung ứng viên vào lưới cũ, giữ ngân sách FA/giờ và alarm policy. Lưới hữu hạn 201 quantile vẫn không đảm bảo tối ưu toàn bộ ngưỡng; validation nhỏ cũng có nguy cơ overfit. Mọi thay đổi recall ở cùng predictions là tác động operating point, không phải AUROC tốt lên.

Thiết kế chạy thật: V0_4_PLAN (đã xóa). Kết quả: báo cáo (đã xóa).

