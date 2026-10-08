# SafeAnes Monitor — demo UI + API

Giao diện dự báo nguy cơ tụt huyết áp trong mổ cho mô hình UC04 (samples v3). **Bản nghiên cứu, không dùng cho lâm sàng.**

| Tab | Cho ai | Làm gì |
|---|---|---|
| **Dự báo** (mặc định) | người dùng | tải CSV sinh hiệu của một ca → nguy cơ mỗi 30 giây cho 5 mốc, cảnh báo, lý do, chất lượng dữ liệu, tải kết quả CSV |
| Phát lại ca mẫu | trình bày | phát lại 14 ca validation VitalDB như đang theo dõi, hoặc hồi cứu cả ca |
| Mô hình & kết quả | nhóm nghiên cứu | chỉ số validation, v2 vs v3 theo giai đoạn, so với các hệ thống cùng loại |

```
deploy/
  run_local.ps1   một lệnh: export dữ liệu + build UI nếu thiếu, rồi chạy http://127.0.0.1:8000
  backend/        FastAPI: dự báo 5 mốc, giải thích (TreeSHAP), cảnh báo theo đúng chính sách đã đánh giá,
                  và phục vụ luôn bản build của UI (cùng origin)
    export_bundle.py   lấy mô hình + ca demo từ repo  ->  backend/data/   (git-ignored)
    check_extract.py   kiểm tra extract.py với samples v3, tạo file mẫu (data/examples/)
    app/               engine.py (mô hình, cảnh báo), extract.py (sinh hiệu thô -> đặc trưng),
                       features.py (tên/nhóm đặc trưng), main.py (API + UI)
  frontend/       React + Vite + TypeScript, biểu đồ SVG tự vẽ, sáng/tối
```

## Chạy (local)

```powershell
powershell -ExecutionPolicy Bypass -File deploy
un_local.ps1            # lần đầu: tự export dữ liệu và build UI
powershell -ExecutionPolicy Bypass -File deploy
un_local.ps1 -Rebuild   # sau khi train lại mô hình hoặc sửa UI
```

Mở http://127.0.0.1:8000 (tài liệu API: `/docs`). Server chỉ nghe trên `127.0.0.1`, không mở ra mạng ngoài.
Yêu cầu: Python 3.11 (fastapi, uvicorn, lightgbm, pandas, pyarrow; script tự dùng `.local_deps` của repo hoặc cài từ
`backend/requirements.txt`), Node 20+ để build UI.

## Chi tiết từng bước (script ở trên đã làm sẵn)

1. **Xuất mô hình và ca demo** (cần `artifacts/v3` có `lgbm_context`, `lgbm_numeric` W = 120 và `data/samples_v3`):
   `PYTHONPATH="src;.local_deps" python deploy/backend/export_bundle.py --config configs/uc04_v3.json --work artifacts/v3 --cases 14`.
   Ghi ra `deploy/backend/data/`: booster LightGBM dạng text, Platt, ngưỡng, chỉ số validation, 14 ca validation
   (có đợt tụt trước rạch da, trong mổ, cả hai, không có), và các bảng so sánh v2/v3.
2. **Build UI:** `cd deploy/frontend && npm install && npm run build` → `dist/`, được backend phục vụ tại `/`.
3. **Sửa UI có hot reload:** chạy backend (`cd deploy/backend && uvicorn app.main:app --port 8000`) và
   `cd deploy/frontend && npm run dev` → http://localhost:5173 (Vite chuyển `/api` sang :8000).

> `backend/data/` chứa dữ liệu bệnh nhân VitalDB (theo thỏa thuận sử dụng dữ liệu của VitalDB): không commit,
> không chia sẻ ra ngoài máy.

## Trang "Dự báo": file đầu vào

- Một cột **thời gian**: số giây, `HH:MM:SS` hoặc ngày giờ. Cột **MAP động mạch xâm lấn** là bắt buộc. Nên có thêm SBP, DBP,
  HR, SpO₂, EtCO₂, nhịp thở; nếu có NIBP thì MAP nền được tính tự động.
- Nhận tên cột kiểu VitalDB (`Solar8000/ART_MBP`…) hoặc `map`, `sbp`, `dbp`, `hr`, `spo2`, `etco2`, `rr`. Ô trống được phép.
- Nhập thời điểm khởi mê và rạch da theo cùng đồng hồ với cột thời gian. Thiếu thì các cột ngữ cảnh liên quan bị bỏ trống,
  và trang sẽ báo.
- Thử nhanh: http://127.0.0.1:8000/?example=2905 (tự nạp một ca mẫu và chạy).

Đặc trưng được tính bằng chính code của pipeline: `safeanes.preprocess.clean_numeric` / `grid_numeric`,
`uc04.long_windows`, `uc04.events`. Các cột ngữ cảnh `f1a_*` được dựng lại theo định nghĩa đã đối chiếu với samples v3.
`check_extract.py` chạy lại toàn bộ trên track Solar8000 thô của 14 ca validation (4.888 thời điểm) và so với
samples v3:
- **Xác suất cuối:** tương quan 0,999 với `val_predictions`; lệch trung vị 0, phân vị 95 là 0,008.
- **Theo cột:** 48/97 cột trùng ≥ 99% số dòng. Phần lệch còn lại nằm ở thống kê cửa sổ 10–30 phút, vì pipeline loại thêm
  nhiễu nhờ dạng sóng ART (flush, zero), mà file upload không có.

## API

| Endpoint | Trả về |
|---|---|
| `GET /api/health` | trạng thái, mô hình, số ca |
| `GET /api/models` | mô hình, ngưỡng và chỉ số validation theo mốc |
| `GET /api/report` | bảng v2 vs v3 theo giai đoạn, bảng độ ổn định ngưỡng |
| `GET /api/cases` | danh sách ca demo |
| `GET /api/cases/{id}/timeline?model=` | thời gian, sinh hiệu, xác suất 5 mốc, cảnh báo (đúng/sai), đợt tụt, mốc rạch da |
| `GET /api/cases/{id}/explain?t=&horizon=&model=` | xác suất và đóng góp theo nhóm sinh lý / đặc trưng tại thời điểm t |
| `POST /api/infer` | `{ "csv": "...", "anestart": "08:05", "opstart": "08:40", "baseline_map": null, "model": "lgbm_context" }` → giống timeline + `session`, `quality` (độ phủ, cảnh báo dữ liệu) |
| `GET /api/infer/{session}/explain?t=&horizon=` | giải thích tại thời điểm t của dữ liệu đã tải (phiên lưu trong RAM, tối đa 20) |
| `GET /api/examples`, `/api/examples/{id}.csv` | file mẫu (track Solar8000 thô của ca demo) |
| `POST /api/predict` | `{ "features": {cột: giá trị}, "model": "lgbm_context" }` → nguy cơ 5 mốc + giải thích; dùng cho luồng dữ liệu thật |

## Đúng với kết quả đã báo cáo

Backend không có logic riêng. Xác suất bằng `val_predictions.parquet` của pipeline (lệch ≤ 5·10⁻⁵, do bước kẹp logit
ở 1e-6). Cảnh báo dùng đúng luật của `uc04.alarms`: 2 dự báo liên tiếp ≥ ngưỡng, nghỉ 300 giây, phát lại chỉ sau khi
nguy cơ xuống dưới ngưỡng. Đã kiểm tra trên 14 ca demo ở mốc 5 và 30 phút: 125/125 cảnh báo trùng thời điểm và loại
với evaluator.

## Thiết kế

- **Peer:** Acumen HPI của Edwards (một chỉ số, cảnh báo ≥ 85, màn hình phụ nêu nguyên nhân) và Prescience
  (Lundberg et al., *Nat Biomed Eng* 2018: giải thích từng thời điểm bằng SHAP).
- **Khác biệt của SafeAnes:**
  - Hiển thị xác suất đã hiệu chỉnh cho 5 mốc, không phải một chỉ số.
  - Có giai đoạn trước rạch da / trong mổ, và giải thích theo nhóm sinh lý cùng ngữ cảnh của ca.
  - Cảnh báo được đánh giá theo đợt tụt; chế độ phát lại không lộ tương lai.
- **Biểu đồ** theo quy tắc dataviz:
  - Không dùng 2 trục y: nguy cơ và MAP ở 2 panel chung trục thời gian.
  - Đường 2 px, lưới mảnh.
  - Màu trạng thái luôn đi kèm biểu tượng và chữ.
  - Bảng màu đã qua bộ kiểm tra mù màu ở cả nền sáng lẫn tối.
  - Có crosshair, tooltip, và bấm để tua.
