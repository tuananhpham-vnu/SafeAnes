"""Package and optionally upload the four 71-feature models, without loading models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "SafeAnesUC04_models"
HORIZONS = (300, 600, 900, 1200, 1800)
RUNS = {
    "lgbm_v8": "uc04-w16-v8-tabular/tabular",
    "catboost_v8": "uc04-w16-v8-tabular/tabular",
    "logistic_v8": "uc04-w16-v8-tabular/tabular",
    "tabm_v8": "uc04-w16-v8-tabm/tabm",
}


def package() -> tuple[Path, list[dict]]:
    output = ROOT / "artifacts" / "huggingface" / NAME
    output.mkdir(parents=True, exist_ok=True)
    selected = json.loads((ROOT / "configs/features_v8_71.json").read_text(encoding="utf-8"))
    assert len(selected["columns"]) == 71
    paths: set[str] = set()

    def copy(source: Path, relative: str) -> None:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        paths.add(relative)

    metrics = []
    for model, run in RUNS.items():
        base = ROOT / "artifacts/v3/kaggle_dl" / run
        for horizon in HORIZONS:
            checkpoint = base / f"artifacts/tabular/{model}/W120/h{horizon}"
            done = json.loads((checkpoint / "done.json").read_text(encoding="utf-8"))
            assert done["status"] == "ok", checkpoint
            for filename in ("model.joblib", "threshold.json", "provenance.json", "done.json"):
                copy(checkpoint / filename, f"models/{model}/W120/h{horizon}/{filename}")
            source = base / f"reports/tabular_validation/{model}_W120_h{horizon}.csv"
            copy(source, f"metrics/validation/{source.name}")
            with source.open(encoding="utf-8-sig", newline="") as stream:
                metrics.extend(csv.DictReader(stream))
    for relative in ("configs/features_v8_71.json", "configs/uc04_v3.json",
                     "requirements.txt", "pyproject.toml"):
        copy(ROOT / relative, relative)
    # Pickles refer to project classes (including uc04.tabm.OneHorizon).
    for source in sorted((ROOT / "src").rglob("*.py")):
        copy(source, source.relative_to(ROOT).as_posix())

    table = []
    for model in RUNS:
        rows = {int(row["horizon"]): row for row in metrics if row["model"] == model}
        table.append("| " + model + " | " + " | ".join(
            f"{float(rows[h]['auroc']):.3f}" for h in HORIZONS) + " |")
    readme = """---
language:
- en
tags:
- tabular-classification
- lightgbm
- catboost
- tabm
- logistic-regression
- safeanes
---
# SafeAnes UC04: 71-feature model checkpoints

Four model families trained on Kaggle on 2,393 training cases (810,559 labeled
rows), with 71 features from `features_v8` and a 120-second input window.
There are 20 per-horizon checkpoint files: four families at 5, 10, 15, 20,
and 30 minutes. TabM uses a shared multi-horizon model exposed by each
per-horizon wrapper; the five TabM files are not five independent training runs.

## Files and loading

- `models/<model>/W120/h<seconds>/model.joblib`: original checkpoint dictionary,
  including estimator, Platt calibration parameters, ordered input columns,
  window, horizon and configuration digest.
- `threshold.json`: validation-selected alarm threshold and operating curve.
- `provenance.json`, `done.json`: original run metadata and completion status.
- `configs/features_v8_71.json`: selected feature names and groups.
- `metrics/validation/`: aggregate validation metrics (no patient-level rows).
- `src/`, `requirements.txt`, `pyproject.toml`: project code and dependencies
  required to resolve custom classes in the serialized checkpoints.
- `manifest.json`: SHA-256 and size for every packaged file except the manifest.

Use a Kaggle environment for loading and inference. Install the project with
`pip install -r requirements.txt` and `pip install --no-deps -e .` in a downloaded
snapshot. Check `provenance.json` for the package versions used during training.
The checkpoints require the custom `uc04` classes; they are not Transformers
`AutoModel` checkpoints. Use the stored Platt parameters and matching threshold
when reproducing calibrated probabilities and alarms. Input features must follow
the checkpoint's ordered column list; raw monitor CSVs are not direct inputs.
Code for constructing all `f2_*` features from raw signals is not yet included.

## Validation AUROC

| Model | 5 min | 10 min | 15 min | 20 min | 30 min |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(table) + """

LightGBM is the selected model based on validation point estimates. These results
do not establish superiority over the earlier 349-feature models: a comparison
under matched training and evaluation conditions is still needed.

## Evaluation context

Hypotension: MAP below 65 mmHg for at least 60 seconds. Validation comprises
248 cases. Platt calibration uses the calibration split; thresholds are selected
on validation for approximately one false alarm per hour. Feature count 71 was
also selected after viewing validation, so validation is not an untouched final
evaluation set. The existing test split had previously been used; its results
are descriptive only and are not included in this checkpoint package.

No patient-level inputs, predictions, case metrics, or credentials are included.
These are research checkpoints, not a clinically validated decision system.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    paths.add("README.md")
    entries = []
    for relative in sorted(paths):
        content = (output / relative).read_bytes()
        entries.append({"path": relative, "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"name": NAME, "features": 71, "window_seconds": 120,
                "horizons_seconds": list(HORIZONS), "models": list(RUNS),
                "checkpoint_count": 20, "files": entries}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output, entries


def prepare_support(output: Path) -> list[str]:
    """Prepare the source, feature list and training notes requested by the user."""
    allowed = ["configs/features_v8_71.json", "configs/uc04_v3.json",
               "requirements.txt", "pyproject.toml"]
    allowed += [path.relative_to(output).as_posix() for path in sorted((output / "src").rglob("*.py"))]
    for relative in ("scripts/train_tabm.py", "scripts/train_tabular.py", "scripts/ablation_214.py"):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        allowed.append(relative)
    versions = {}
    for model in RUNS:
        for horizon in HORIZONS:
            relative = f"models/{model}/W120/h{horizon}/provenance.json"
            allowed.append(relative)
            if horizon == 300:
                provenance = json.loads((output / relative).read_text(encoding="utf-8"))
                versions[model] = {"recorded_packages": provenance["packages"],
                                   "catboost": None, "tabm": None, "rtdl_num_embeddings": None,
                                   "git": provenance["git"]}
    source_entries = [{"path": relative, "sha256": hashlib.sha256((output / relative).read_bytes()).hexdigest()}
                      for relative in allowed if relative.startswith(("src/", "scripts/"))]
    (output / "source_manifest.json").write_text(json.dumps({
        "status": "current_workspace_support_snapshot_not_verified_exact_training_snapshot",
        "checkpoint_static_reference": "uc04.tabm.OneHorizon",
        "training_log_code_label": "a887f7f269c1+week15-ablation(uncommitted)",
        "files": source_entries}, indent=2) + "\n", encoding="utf-8")
    (output / "training_versions.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    notes = """# Thông tin từ phía huấn luyện

## 1. Code và khả năng nạp checkpoint

Repo bổ sung snapshot code hiện có ở workspace: `src/`, hai script huấn luyện,
script ablation, cấu hình và danh sách dependencies. SHA-256 của từng file code
nằm trong `source_manifest.json`. Đây là snapshot hỗ trợ, **chưa xác nhận là toàn
bộ code chính xác tại thời điểm huấn luyện**.

Đọc tĩnh file TabM (không unpickle, không chạy model) xác nhận checkpoint chứa
tham chiếu `uc04.tabm`, `OneHorizon` và `TabMMulti`. Class hiện nằm trong
`src/uc04/tabm.py`; `scripts/train_tabm.py` import từ module này.

Provenance ghi `git.commit: null`; log Kaggle ghi nhãn code
`a887f7f269c1+week15-ablation(uncommitted)`. Đây không phải commit tái lập đầy đủ.
Chưa thể khẳng định code hiện tại trùng từng byte với source lúc train.
Không cung cấp một commit giả định hoặc khẳng định đã kiểm tra nạp pickle.

## 2. Dữ liệu dùng để fit mô hình

Theo code và log hiện có, bốn mô hình dùng bộ 71 cột từ features_v8.
Train có 2.393 ca; log TabM ghi 810.559 dòng eligible với ít nhất một nhãn đã biết.

- LightGBM, CatBoost và logistic: `fit_and_calibrate` fit estimator trên train;
  `fit_rows` giữ eligible và y_h thuộc {0, 1} cho từng mốc.
- TabM: chỉ nhận features/labels train vào `TabMMulti.fit`; hàng phải eligible
  và có ít nhất một nhãn đã biết; loss bỏ qua nhãn -1. Early stopping dùng một
  phần bệnh nhân tách từ train theo hash subjectid. Imputer/scaler chỉ fit trên
  các bệnh nhân train dùng tối ưu, không dùng nhóm early stopping.
- Calibration dùng để fit Platt, không dùng để fit trọng số estimator cuối.
- Validation dùng chọn ngưỡng và báo cáo; test chỉ cho kết quả mô tả.

Đây là mô tả từ source/log hiện có, chưa phải audit độc lập toàn bộ dataset.
Giai đoạn ablation khác với huấn luyện cuối: ablation fit trên 258 ca calibration
vì khi đó features_v8 chưa có train; không nhầm số liệu ablation với fit cuối.

## 3. Danh sách 71 cột và cách chọn

Danh sách có thứ tự nằm trong `configs/features_v8_71.json` (trường `columns`).
RFE bắt đầu từ 349 đặc trưng; thứ hạng |SHAP| được học qua CV 5 fold theo bệnh
nhân trên calibration. Điểm 71 được chốt sau khi xem đường validation: ở ba mốc
5/10/15 phút và ba cách đánh giá, AUROC giảm không quá 0,004 so với bộ đầy đủ
và CI của chênh lệch chứa 0. Điểm dừng tự động theo CV là 11, không phải 71.

**Lựa chọn 71 không chỉ dựa trên train. Validation đã tham gia chọn số cột**,
nên kết quả validation không phải ước lượng độc lập sau khi khóa lựa chọn.
Test hiện có cũng đã dùng trước; cần dữ liệu độc lập để xác nhận sau cùng.

## 4. Phiên bản thư viện

`training_versions.json` và các provenance gốc ghi Python 3.13.15,
LightGBM 4.6.0, scikit-learn 1.6.1, Torch 2.11.0+cpu, NumPy 2.1.3,
pandas 2.3.3 và joblib 1.6.0. Provenance không ghi phiên bản CatBoost, tabm
và rtdl_num_embeddings; các trường này để null, không lấy phiên bản máy local
hoặc mức tối thiểu trong requirements làm phiên bản lúc train.

`requirements.txt` là các mức tối thiểu, không phải lockfile của môi trường train.
Chỉ nạp/chạy model và các phép tính nặng trên Kaggle. Lần chuẩn bị gói này chỉ
đọc/copy/hash file, không chạy model tại local.

## Nội dung bổ sung

Gói bổ sung gồm source hỗ trợ, cấu hình, danh sách features, provenance và ghi
chú này. Không gửi thêm ngưỡng/hiệu chỉnh (checkpoint đã chứa Platt), không có
dữ liệu bệnh nhân, dự báo từng ca hoặc credentials. Code tạo đầy đủ các cột
f2_* từ tín hiệu thô hiện chưa có trong repo.
"""
    (output / "TRAINING_NOTES.md").write_text(notes, encoding="utf-8")
    readme = """---
language:
- vi
- en
tags:
- lightgbm
- catboost
- tabm
- logistic-regression
- safeanes
---
# SafeAnesUC04_models

Checkpoint của LightGBM, CatBoost, TabM và hồi quy logistic dùng 71 đặc trưng,
W = 120 giây, tại các mốc 5 / 10 / 15 / 20 / 30 phút.

- `models/<model>/W120/h<seconds>/model.joblib`: 20 checkpoint đã tải trước đó.
- `configs/features_v8_71.json`: danh sách 71 cột theo thứ tự.
- `src/`, `scripts/`: snapshot code hỗ trợ và các script huấn luyện/ablation.
- `models/.../provenance.json`: metadata huấn luyện gốc.
- `training_versions.json`: phiên bản đã ghi nhận; phiên bản thiếu để null.
- `source_manifest.json`: SHA-256 của snapshot code hiện tại.
- [TRAINING_NOTES.md](TRAINING_NOTES.md): chi tiết và các giới hạn tái lập.

**Code hiện tại chưa được xác minh là snapshot chính xác lúc train.** TabM
checkpoint tham chiếu `uc04.tabm.OneHorizon`, có trong source kèm theo.
Provenance không có commit đầy đủ. CatBoost chưa có phiên bản thư viện xác nhận.

Trọng số mô hình cuối được fit trên train; Platt dùng calibration. Việc chốt
71 cột đã xem validation, nên không tuyên bố lựa chọn hoàn toàn dựa trên train.

Dùng Kaggle để nạp/chạy checkpoint. Trong môi trường đó, cài dependencies với
`pip install -r requirements.txt` và `pip install --no-deps -e .` từ snapshot.
Đây là pickle có custom classes, không phải Transformers AutoModel; cần có
package `uc04` khi nạp. Chưa kiểm tra nạp model trong lần bổ sung tài liệu này.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return sorted(allowed + ["README.md", "TRAINING_NOTES.md", "training_versions.json", "source_manifest.json"])


def publish(output: Path, entries: list[dict], *, support: bool = False) -> None:
    import os

    sys.path.insert(0, str(ROOT / ".local_deps"))
    from dotenv import load_dotenv
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()

    load_dotenv(ROOT / ".env")
    token = os.getenv("SAFEANES_HF_TOKEN_WRITE") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("No Hugging Face write token configured")
    api = HfApi(token=token)
    owner = api.whoami()["name"]
    repo_id = f"{owner}/{NAME}"
    print(f"Uploading {repo_id}: {'source and training metadata' if support else '20 checkpoints'}", flush=True)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    # Remote payload is strictly the 20 user-requested checkpoint files.
    # All supporting source, configuration, metrics and manifests stay local.
    allowed = prepare_support(output) if support else [
        f"models/{model}/W120/h{horizon}/model.joblib" for model in RUNS for horizon in HORIZONS]
    assert all((output / path).is_file() for path in allowed)
    operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(output / path))
                  for path in allowed]
    commit = api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                               commit_message=("Add requested source, feature list and training metadata" if support
                                               else "Upload only 20 checkpoints for four 71-feature models"))
    info = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
    remote = {file.rfilename: file for file in info.siblings}
    for relative in allowed:
        content = (output / relative).read_bytes()
        file = remote[relative]
        assert file.size == len(content), f"Size mismatch: {relative}"
        if file.lfs:
            assert file.lfs.sha256 == hashlib.sha256(content).hexdigest(), relative
        else:
            blob = b"blob " + str(len(content)).encode() + b"\0" + content
            assert file.blob_id == hashlib.sha1(blob).hexdigest(), relative
    result = {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}",
              "private": info.private, "commit": commit.oid,
              "verified_files": len(allowed), "checkpoint_count": 20}
    suffix = "support_upload" if support else "upload"
    (output.parent / f"{NAME}_{suffix}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--support", action="store_true", help="upload requested source and training metadata")
    args = parser.parse_args()
    folder, files = package()
    print(json.dumps({"package": str(folder), "files": len(files) + 1,
                      "bytes": sum(file["size"] for file in files), "checkpoints": 20}), flush=True)
    if args.upload:
        publish(folder, files, support=args.support)
    elif args.support:
        print(json.dumps({"support_files": len(prepare_support(folder))}), flush=True)
