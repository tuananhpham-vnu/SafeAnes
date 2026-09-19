# Rà soát phương án thay thế — 18/09/2026

AUROC E05 đã cao nhưng event recall/PPV thấp. Bảng SOTA tổng quát không chứng minh
một phương pháp sẽ cải thiện cảnh báo liên tục IOH; cần cùng split và alarm budget.

| Phương pháp | Bằng chứng gốc | Quyết định |
|---|---|---|
| TabPFN-3.5 (15/09/2026) | [Báo cáo tác giả](https://priorlabs.ai/technical-reports/tabpfn-3-5) công bố dẫn đầu TabArena/BeyondArena; base/Fast có trong package, các chế độ khác có API. | Ứng viên SOTA mới nhất tìm thấy trong lần rà soát này; chưa chạy, không gán điểm benchmark thành điểm IOH. |
| TabICLv2, ICML 2026; arXiv v2 16/09/2026 | [Paper](https://arxiv.org/abs/2602.11139), [repo chính thức](https://github.com/soda-inria/tabicl). So sánh TabArena/TALENT với RealTabPFN-2.5; đó không phải so sánh với TabPFN-3.5 vừa ra. | Ưu tiên thử foundation model trên GPU với đúng toàn bộ fit context và patient split; chưa có kết quả SafeAnes. |
| TabM, ICLR 2025 | [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html), [repo chính thức Apache-2.0](https://github.com/yandex-research/tabm). MLP ensemble chia sẻ tham số, có numerical embeddings. | Thực thi ngay E06 bằng package chính thức. Cấu hình CPU nhỏ và ordered IOH head là adaptation; không tuyên bố tái lập điểm paper hoặc SOTA 2026. |
| Transformer IOH, PLOS Medicine 2026 | [Nghiên cứu gốc](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1005024) dùng chuỗi vital signs, có kiểm định ngoài. | Gần bài toán hơn benchmark bảng, nhưng protocol/metrics khác; review E03 đã chỉ ra vấn đề cần audit trong code tác giả. Không chép nguyên rồi gọi là mô hình đã kiểm chứng. |

Lựa chọn thực thi: TabM+PLE và regularized monotone LightGBM để kiểm tra cả kiến trúc
lẫn overfitting. Máy hiện có PyTorch CPU, không có CUDA khả dụng. Các tên TabICLv2 và
TabPFN-3.5 là hướng thay thế cần benchmark tiếp, không phải thành tích đã đo.

Firecrawl CLI đã kiểm tra nhưng scrape bị EACCES mạng; nguồn được đối chiếu qua web
trực tiếp. Thư viện TabM được tải từ PyPI vào `.local_deps`, không gửi dữ liệu nghiên cứu
lên API dự báo bên ngoài.
