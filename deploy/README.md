# SafeAnes Monitor: trạm theo dõi mô phỏng

Màn hình kiểu trạm theo dõi trung tâm cho mô hình dự báo tụt huyết áp UC04. **Bản nghiên cứu, không dùng cho lâm sàng.**

- 50 ca của **tập test** `safeanes-v3` được phát lại như đang theo dõi trực tiếp. Tập test này đã được dùng để xác nhận
  các mô hình trước, nên số liệu trên đó chỉ mang tính mô tả.
- 4 mô hình (LightGBM, CatBoost, hồi quy logistic, TabM) dùng cùng 71 đặc trưng của `features_v8`
  (`configs/features_v8_71.json`), W = 120 giây, 5 mốc 5–30 phút.
- **Mỗi 2 giây:** sinh hiệu mới từ track Solar 8000 gốc.
- **Mỗi 30 giây:** máy chủ chạy cả 4 mô hình trên hàng đặc trưng của thời điểm đó (tính chỉ từ dữ liệu tới thời điểm
  đó), rồi cập nhật cảnh báo theo luật đã đánh giá: 2 lần liên tiếp vượt ngưỡng, nghỉ 5 phút, chỉ báo lại khi nguy cơ
  đã xuống dưới ngưỡng.
- Không gửi dữ liệu tương lai. Đúng/sai của một cảnh báo chỉ hiện khi thời gian đã trôi qua.

| Trang | Nội dung |
|---|---|
| Home (`#/`) | Giới thiệu, cách mô phỏng, AUROC của 4 mô hình (validation, test mô tả), nút **Start** |
| Trạm theo dõi (`#/ward`) | Lưới 50 giường: % nguy cơ, MAP 10 phút, HR, SpO₂, 5 mốc; chọn mô hình, mốc, tốc độ 1× / 5× / 20× |
| Giường (`#/bed/<n>`) | 4 mô hình × 5 mốc, diễn tiến từ đầu ca, giải thích "vì sao", nhật ký cảnh báo, ngữ cảnh ca |

```
deploy/
  Dockerfile          một image: UI (build bằng node) + API (python, torch CPU); không chứa dữ liệu
  AZURE.md            hướng dẫn deploy lên Azure Container Apps (Azure for Students)
  run_local.ps1       xem thử trên máy, chế độ replay (không chạy mô hình)
  backend/
    export_ward.py    repo -> data_ward/: 50 ca test, sinh hiệu 2 giây, 4 mô hình, dự báo offline để đối chiếu
    check_engine.py   (Kaggle) engine live = dự báo offline? thời gian mỗi tick
    fetch_bundle.py   khi container khởi động: tải bundle từ HF dataset private nếu chưa mount
    app/  models.py   4 mô hình sau một giao diện (score, explain); Platt như uc04.tabular
          ward.py     đồng hồ mô phỏng, chạy mô hình theo lô khi qua mốc 30 giây, cảnh báo, lịch sử, giải thích
          feed.py     luồng Server-Sent Events, 1 tin/giây (tách riêng để đổi sang WebSocket / SignalR)
          main.py     API + phục vụ UI
  frontend/           React + Vite + TypeScript, biểu đồ SVG tự vẽ, sáng/tối
```

## 1. Bundle (dữ liệu + mô hình)

Cần các output train của các mô hình `*_v8` (Kaggle: `train_tabular.py --models logistic_v8 lgbm_v8 catboost_v8 --test-dir ...`,
`train_tabm.py --fv8 --test-dir ...`), `data/samples_v3_test/` (tập test của `safeanes-v3`) và `data/vitaldb_full`
(track gốc, manifest).

```powershell
$env:PYTHONPATH = "src;.local_deps"
python deploy/backend/export_ward.py --work <output tabular> --work-tabm <output tabm> --cases 50
```

Script ghi vào `deploy/backend/data_ward/`. Thư mục này nằm trong `.gitignore` vì là dữ liệu bệnh nhân VitalDB (theo thoả
thuận sử dụng dữ liệu): không commit, không đưa vào image, không công khai.

## 2. Kiểm tra engine (Kaggle)

Chạy `deploy/backend/check_engine.py --data <bundle>` trên Kaggle. Script đối chiếu xác suất của engine live với
`reference/` cho mọi giường, mô hình và mốc (sai khác ≤ 1e-6), đồng thời đo thời gian một tick (50 giường × 4 mô hình).

## 3. Xem trên máy (replay)

```powershell
powershell -ExecutionPolicy Bypass -File deploy\run_local.ps1
```

Mở http://127.0.0.1:8000 (tài liệu API: `/docs`). Ở chế độ replay, phần giải thích "vì sao" không có vì không chạy
mô hình. Khi sửa UI có hot reload: chạy `cd deploy/frontend && npm run dev`, mở http://localhost:5173 (Vite chuyển
`/api` sang :8000).

## 4. Docker và Azure

```powershell
docker build -f deploy/Dockerfile -t safeanes-ward .
```

Chạy image (chế độ live, có chạy mô hình) trên cloud: xem [AZURE.md](AZURE.md).

## API

| Endpoint | Trả về |
|---|---|
| `GET /api/health` | engine (live/replay), số giường, mô hình, số hàng đã chạy mô hình |
| `GET /api/models` | 4 mô hình: ngưỡng, chỉ số validation và test (mô tả) theo mốc |
| `GET /api/ward` | thông tin giường (tuổi, giới, ASA, khoa, loại mổ, rạch da, đợt tụt), tốc độ, thời gian ảo hiện tại |
| `GET /api/ward/stream?speed=&v=` | SSE, mỗi giây: mọi giường (nguy cơ 4 mô hình × 5 mốc, cảnh báo, sinh hiệu, giai đoạn) + mẫu 2 giây mới |
| `GET /api/beds/{bed}/history?v=&wave_s=` | một giường từ đầu ca tới hiện tại: nguy cơ, cảnh báo (đúng/sai/chờ), đợt tụt đã biết, sinh hiệu 2 giây |
| `GET /api/beds/{bed}/explain?v=&model=&horizon=` | đóng góp theo đặc trưng / nhóm ở lần dự báo gần nhất (TabM: dùng giải thích của LightGBM để tham khảo) |
