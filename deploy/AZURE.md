# Deploy SafeAnes Monitor lên Azure (Azure for Students, Docker)

Kiến trúc: **một container** (UI + API, `deploy/Dockerfile`) chạy trên **Azure Container Apps**. Image lấy từ
**Azure Container Registry (ACR)**. Image **không chứa dữ liệu bệnh nhân hay file mô hình**: hai thứ này nằm trong
bundle (`deploy/backend/data_ward/`, tạo bằng `export_ward.py`). Khi khởi động, container tải bundle từ một HF dataset
private.

```
máy local: docker build ──push──► ACR ──pull──► Container Apps (1 replica, 1 vCPU / 2 GiB)
                                                       │ khởi động: tải bundle (HF_TOKEN)
                                                       ▼
                                            https://<app>.<region>.azurecontainerapps.io
```

## Lưu ý riêng cho Azure for Students

- **1 môi trường Container Apps / subscription**, tối đa 10 core ([quotas](https://learn.microsoft.com/azure/container-apps/quotas)).
  Nếu đã có môi trường cũ thì dùng lại nó (bỏ bước 5a).
- **`az acr build` (build trên cloud) có thể bị chặn** với credit miễn phí ("ACR task runs are temporarily paused from
  Azure free credits"). Vì vậy hướng dẫn này **build bằng Docker trên máy rồi push**.
- **Region bị giới hạn bởi policy** "Allowed resource deployment regions" của subscription. Với subscription này
  (kiểm tra ngày 22/10/2026) chỉ được dùng `koreacentral`, `malaysiawest`, `eastasia`, `indiasouthcentral` và
  `japaneast`, **không có `southeastasia`**. Resource group thì tạo được ở bất kỳ đâu, nhưng tài nguyên bên trong bị chặn
  với lỗi `RequestDisallowedByAzure`. Xem danh sách:
  `az policy assignment list --query "[].parameters.listOfAllowedLocations.value" -o json`.
- **Chi phí** (trừ vào $100 credit/năm):
  - Container Apps miễn phí 180.000 vCPU-giây và 360.000 GiB-giây mỗi tháng. Ở mức 1 vCPU / 2 GiB, phần miễn phí đủ
    khoảng 50 giờ chạy/tháng. Khi scale về 0 (không có ai xem) thì không tính tiền
    ([giá](https://azure.microsoft.com/pricing/details/container-apps/)).
  - ACR Basic khoảng 5 USD/tháng.
  - Log Analytics: vài cent.
  - **Xoá resource group khi không dùng** (bước 9) và đặt budget alert trong Cost Management.

## 0. Chuẩn bị (một lần)

- Kích hoạt Azure for Students: https://azure.microsoft.com/free/students
- Cài Azure CLI (máy đã có) và Docker Desktop (máy đã có, bật engine Linux).
- Bundle đã export: thư mục `deploy/backend/data_ward/` có `ward.json`, `cases/`, `wave/`, `models/`, `reference/`.
- Tài khoản Hugging Face, kèm hai token: một token **write** để upload, một token **read** cho container.

## 1. Đăng nhập và đăng ký dịch vụ

```powershell
az login
az account show --query "{name:name, id:id}" -o table      # đúng subscription "Azure for Students"
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.ContainerRegistry
```

## 2. Biến dùng chung

```powershell
$RG   = "safeanes-rg"
$LOC  = "eastasia"                                 # một region được policy cho phép
$ACR  = "safeanes$((Get-Random -Maximum 99999))"   # tên ACR: chữ thường + số, duy nhất toàn cầu
$ENV  = "safeanes-env"
$APP  = "safeanes-monitor"
$TAG  = (git rev-parse --short HEAD)
$BUNDLE_REPO = "tuananhpham-vnu/safeanes-ward"     # HF dataset private chứa bundle (đã upload 22/10)
```

## 3. Đưa bundle lên HF dataset private (chạy lại mỗi khi export lại bundle)

Đã làm ngày 22/10/2026: `tuananhpham-vnu/safeanes-ward` (private, 126 file). Chỉ cần làm lại khi export lại bundle.

```powershell
pip install -U huggingface_hub
huggingface-cli login                               # token write
huggingface-cli repo create safeanes-ward --type dataset --private
huggingface-cli upload $BUNDLE_REPO deploy/backend/data_ward . --repo-type dataset
```

Kiểm tra trên trang HF rằng repo ở chế độ **Private**: bundle là dữ liệu bệnh nhân VitalDB, theo thoả thuận sử dụng dữ
liệu không được công khai.

## 4. Build image và push lên ACR

```powershell
az group create -n $RG -l $LOC
az acr create -n $ACR -g $RG -l $LOC --sku Basic --admin-enabled true   # -l: mặc định là region của RG
az acr login -n $ACR

# từ thư mục gốc của repo (context = repo, .dockerignore chỉ cho code + UI vào image)
docker build -f deploy/Dockerfile -t "$ACR.azurecr.io/safeanes-ward:$TAG" .
docker push "$ACR.azurecr.io/safeanes-ward:$TAG"
```

Build lần đầu mất khoảng 5–10 phút vì phải tải torch bản CPU. Image khoảng 2,7 GB, nên lần push đầu cũng mất vài
phút; các lần sau chỉ đẩy lớp thay đổi (code, UI).

## 5. Tạo Container App

```powershell
# 5a. môi trường (bỏ qua nếu subscription đã có một môi trường: dùng tên của nó cho $ENV)
az containerapp env create -n $ENV -g $RG -l $LOC

# 5b. app: 1 replica duy nhất (bộ đệm dự báo dùng chung trong một tiến trình), scale về 0 khi không ai xem
$ACR_PWD = az acr credential show -n $ACR --query "passwords[0].value" -o tsv
az containerapp create -n $APP -g $RG --environment $ENV `
  --image "$ACR.azurecr.io/safeanes-ward:$TAG" `
  --registry-server "$ACR.azurecr.io" --registry-username $ACR --registry-password $ACR_PWD `
  --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1 `
  --ingress external --target-port 8000 `
  --secrets "hf-token=<HF token read>" `
  --env-vars "HF_TOKEN=secretref:hf-token" "SAFEANES_BUNDLE_REPO=$BUNDLE_REPO" "SAFEANES_ENGINE=live"

az containerapp show -n $APP -g $RG --query properties.configuration.ingress.fqdn -o tsv
```

Mở `https://<fqdn>`, kiểm tra `https://<fqdn>/api/health`. Kết quả phải là `"engine": "live"`, `"beds": 50`, và có đủ 4 model.

- **Lần mở đầu sau khi app ngủ:** chờ khoảng 1–2 phút (tải bundle, nạp torch). Muốn luôn sẵn sàng thì đặt
  `--min-replicas 1`, nhưng khi đó app tính tiền liên tục và phần miễn phí hết sau khoảng 50 giờ.
- **Giữ `--max-replicas 1`:** nếu có nhiều replica thì mỗi replica giữ một bộ đệm riêng. Đồng hồ mô phỏng ở 1× vẫn
  đúng, nhưng cùng một dự báo bị tính lại nhiều lần.

## 6. Cập nhật phiên bản

```powershell
$TAG = (git rev-parse --short HEAD)
docker build -f deploy/Dockerfile -t "$ACR.azurecr.io/safeanes-ward:$TAG" .
docker push "$ACR.azurecr.io/safeanes-ward:$TAG"
az containerapp update -n $APP -g $RG --image "$ACR.azurecr.io/safeanes-ward:$TAG"
```

Chỉ đổi bundle (export lại model hoặc ca) thì upload lại ở bước 3, rồi khởi động lại app:
`az containerapp revision restart -n $APP -g $RG --revision (az containerapp revision list -n $APP -g $RG --query "[0].name" -o tsv)`.

## 7. Xem log

```powershell
az containerapp logs show -n $APP -g $RG --follow            # log của server
az containerapp logs show -n $APP -g $RG --type system       # pull image, khởi động, OOM
```

## 8. Lỗi hay gặp

| Triệu chứng | Nguyên nhân / cách sửa |
|---|---|
| `RequestDisallowedByAzure` khi tạo tài nguyên | Region không nằm trong danh sách policy → `$LOC = "eastasia"` và truyền `-l $LOC` cho mọi lệnh tạo |
| `MaxNumberOfEnvironmentsInSubExceeded` | Đã có 1 môi trường → dùng lại môi trường đó |
| `az acr build` báo task runs paused | Credit miễn phí không chạy ACR Tasks → build local (bước 4) |
| Container khởi động lại liên tục, log hệ thống có OOM | Tăng `--memory` (cặp hợp lệ: 1.0/2.0Gi, 1.5/3.0Gi, 2.0/4.0Gi) |
| Log `no bundle ... SAFEANES_BUNDLE_REPO is not set` / 401 | Sai tên dataset hoặc token HF không có quyền read |
| Trang mở nhưng lưới không chạy (`mất kết nối`) | Xem log; stream SSE gửi dữ liệu mỗi giây nên không bị ngắt vì idle (ingress ngắt sau 4 phút idle) |

## 9. Dừng / xoá

```powershell
az containerapp update -n $APP -g $RG --min-replicas 0     # không tốn tiền khi không ai xem
az group delete -n $RG --yes                                # xoá hết (ACR, app, log)
```

## Phương án tiết kiệm: không dùng ACR

Image chỉ chứa code (repo đã public) và UI, không có dữ liệu. Vì vậy có thể push image lên **GitHub Container
Registry** công khai thay cho ACR, tiết kiệm khoảng 5 USD/tháng:

```powershell
docker login ghcr.io -u <github-user>                         # mật khẩu = PAT có quyền write:packages
docker build -f deploy/Dockerfile -t ghcr.io/<github-user>/safeanes-ward:$TAG .
docker push ghcr.io/<github-user>/safeanes-ward:$TAG
# trên GitHub: Packages → safeanes-ward → Package settings → Change visibility → Public
az containerapp create ... --image ghcr.io/<github-user>/safeanes-ward:$TAG   # bỏ --registry-*
```
