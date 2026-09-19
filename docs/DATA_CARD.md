# Data card — UC04 pilot VitalDB

Release hiện tại **v0.2** gộp các đợt E03/E04/E05. E04 giữ pilot 60 ca để ablation CatBoost/ngưỡng; E05 đã dựng development **300 ca/297 bệnh nhân, 99.132 windows, 497 episode IOH**. Roles cũ được giữ nguyên; nhóm đánh giá mới có 37 bệnh nhân và 35/37 event đủ điều kiện ở 5/10 phút. Xem [plan E05](experiments/E05_PLAN.md), [data support](../reports/E05/data_support.json) và [quy ước release](../README.md#release).

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

Phạm vi là phẫu thuật không tim. [Bài gốc VitalDB](https://www.nature.com/articles/s41597-022-01411-5) mô tả nguồn non-cardiac thuộc general/thoracic/urological/gynecological surgery. E05 đã kiểm kê đúng bốn nhóm này (190/87/7/16 ca), không có cờ tên tim theo regex sơ bộ; [metadata audit](../reports/E05/cohort_audit.csv). Bộ lọc code vẫn dựa trên tuổi/gây mê/MAP/interval, chưa có phân loại mã thủ thuật độc lập. Đây là đối chiếu nguồn và metadata, không phải adjudication lâm sàng lại toàn bộ ca.

Quyền sử dụng/điều kiện phân phối phải đối chiếu bản dữ liệu gốc trước chia sẻ. Repository không chứa raw dữ liệu bệnh nhân hoặc checkpoint trong Git. Các hình replay dẫn xuất chỉ dùng mã ca của dữ liệu công khai; không thêm định danh bệnh viện khác. MOVER và dữ liệu bệnh viện chưa có trong pipeline này.

Thông số nhãn, coverage và giới hạn lấy mẫu nằm trong [protocol](PROTOCOL.md). Không đánh giá độ mạnh thống kê bằng số window chồng lấp; dùng bệnh nhân làm đơn vị bootstrap và báo số biến cố.
