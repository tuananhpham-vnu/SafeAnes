# Data card — UC04 pilot VitalDB

| Thuộc tính | Nội dung |
|---|---|
| Nguồn | VitalDB OpenDataset; nguồn bài báo/API tại S01/S02 trong [SOURCES.md](SOURCES.md). |
| Cohort pilot đang có | 60 ca / 60 bệnh nhân, 19.869 decision windows, 85 đợt IOH theo protocol v1. Số liệu nguồn: `data/pilot_v1/dataset.json`. |
| Định danh | subjectid giữ toàn bộ lần mổ của một người cùng split; caseid xác định ca. |
| Lọc thực thi | Người lớn, General anesthesia, arterial MAP, interval đủ dài và timestamp hợp lệ. |
| Khoảng thời gian | Thời gian tương đối từ case start; chỉ phân tích opstart–opend. Không suy ra năm lịch hoặc đánh giá riêng giai đoạn khởi mê. |
| Numeric | ART_MBP/ART_SBP/ART_DBP, HR, PLETH_SPO2, ETCO2, RR_CO2; thiếu track tùy chọn không tự loại ca. |
| Static | Age/BMI/ASA; không dùng tổng thuốc, dịch hoặc mất máu cuối ca. |
| Raw provenance | Mỗi response có URL, UTC retrieval time, SHA-256 và số byte trong `.source.json`. |
| Dữ liệu dẫn xuất | Windows thống kê trong `pilot_v1`; mỗi ca một `.npy` trong `sequences_v1`; sequence manifest gắn hash dữ liệu và raw. |
| Thiếu dữ liệu | Input causal giữ NaN khi quá hạn; label `-1` nếu follow-up không xác định; giữ decision ineligible để replay reset đúng. |
| Mục đích dùng | Nghiên cứu hồi cứu, không phải nguồn nhãn nguyên nhân đã adjudication. |

Phạm vi đề xuất trong kế hoạch là phẫu thuật không tim. Bộ lọc hiện tại chưa có tiêu chí loại phẫu thuật tim độc lập theo chuyên khoa/mã thủ thuật; cần kiểm kê metadata và khóa ánh xạ cohort trước final benchmark. Không coi bộ lọc code đã thực hiện tất cả điều kiện mô tả trong kế hoạch.

Quyền sử dụng/điều kiện phân phối phải đối chiếu bản dữ liệu gốc trước chia sẻ. Repository không chứa raw dữ liệu bệnh nhân hoặc checkpoint trong Git. Các hình replay dẫn xuất chỉ dùng mã ca của dữ liệu công khai; không thêm định danh bệnh viện khác. MOVER và dữ liệu bệnh viện chưa có trong pipeline này.

Thông số nhãn, coverage và giới hạn lấy mẫu nằm trong [protocol](PROTOCOL.md). Không đánh giá độ mạnh thống kê bằng số window chồng lấp; dùng bệnh nhân làm đơn vị bootstrap và báo số biến cố.
