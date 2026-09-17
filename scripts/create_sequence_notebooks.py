"""Generate plain-Python notebooks for local or Kaggle research runs."""

import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


SETUP = '''from pathlib import Path
import sys
import os

# Sửa SOURCE_ROOT nếu dùng source được gắn qua Kaggle Input.
SOURCE_ROOT = Path.cwd()
if not (SOURCE_ROOT / "src").is_dir() and (SOURCE_ROOT.parent / "src").is_dir():
    SOURCE_ROOT = SOURCE_ROOT.parent
# SOURCE_ROOT = Path("/kaggle/input/safeanes-source")
assert (SOURCE_ROOT / "src" / "safeanes").is_dir(), "Đặt SOURCE_ROOT tới repository"
WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else SOURCE_ROOT
sys.path.insert(0, str(SOURCE_ROOT / "src"))
# Không đưa .local_deps Windows sang Kaggle.
if os.name == "nt" and (SOURCE_ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(SOURCE_ROOT / ".local_deps"))
print("Source:", SOURCE_ROOT, "Output:", WORK_ROOT)
'''


def write(name, cells):
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}}, "cells": cells}
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i:02}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{name}:cell-{i}", "exec")
    target = Path(__file__).resolve().parents[1] / "notebooks" / name
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    write("02_numeric_sequences.ipynb", [
        md("# UC04 — chuẩn bị dữ liệu chuỗi trên CPU\n\nChạy sau notebook pilot. Dữ liệu numeric và nhãn dùng cùng protocol; không mở final test. Cài dependencies theo `docs/SEQUENCE_RUNBOOK.md`."),
        code(SETUP),
        code('''from safeanes.sequences import build_sequences, SequenceStore, load_dataset
DATASET = WORK_ROOT / "data/pilot_v1"
RAW_ROOT = WORK_ROOT / "data/vitaldb"
SEQUENCES = WORK_ROOT / "data/sequences_v1"
# Có thể đặt DATASET và RAW_ROOT vào Kaggle Input; output SEQUENCES phải writable.
if (SEQUENCES / "sequences.json").is_file():
    store = SequenceStore(SEQUENCES, DATASET)  # xác minh toàn bộ hashes
    print("Reusing verified cache:", len(store.meta["cases"]), "cases")
else:
    print(build_sequences(DATASET, RAW_ROOT, SEQUENCES))
'''),
        code('''import pandas as pd
meta, protocol, manifest, windows = load_dataset(DATASET)
display(pd.read_csv(DATASET / "quality.csv"))
display(windows.groupby("eligible").size().rename("decisions"))
print("Cases:", len(manifest), "Subjects:", manifest.subjectid.nunique())
print("Dataset SHA-256:", meta["windows_sha256"])
'''),
        md("Lưu `pilot_v1` và `sequences_v1` thành Input riêng cho notebook GPU. Không cần tải raw vào phiên GPU. Khi mở rộng cohort, tạo dataset/cache mới và không so hai split khác nhau như ablation.")])
    write("03_tcn_transformer.ipynb", [
        md("# UC04 — TCN và Transformer\n\nChạy trên một T4 nếu có; notebook hiển thị thiết bị thực tế. Hai mô hình dùng cùng split/nhãn và evaluator. Kết quả là pilot thăm dò, không phải final test."),
        code(SETUP),
        code('''os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
print("PyTorch:", torch.__version__)
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU — không phải benchmark T4")
# Trên Kaggle dùng PyTorch có CUDA sẵn trong môi trường, không cài wheel CPU.
DATASET = WORK_ROOT / "data/pilot_v1"
SEQUENCES = WORK_ROOT / "data/sequences_v1"
# Nếu dùng Input từ notebook CPU, sửa hai đường dẫn trên.
RUN_ROOT = WORK_ROOT / "artifacts"
'''),
        code('''import json
from dataclasses import asdict
from safeanes.training import TrainConfig, train_sequence
from safeanes.sequences import load_dataset

reports = {}
dataset_meta, _, _, _ = load_dataset(DATASET)
for architecture in ("tcn", "transformer"):
    config = TrainConfig(**json.loads((SOURCE_ROOT / "configs" / f"{architecture}.json").read_text()))
    output = RUN_ROOT / f"{architecture}_v1"
    if (output / "report.json").is_file():
        saved_config = json.loads((output / "config.json").read_text())
        saved_env = json.loads((output / "environment.json").read_text())
        assert saved_config == asdict(config), "Config đã đổi; chọn tên run mới"
        assert saved_env["dataset_hash"] == dataset_meta["windows_sha256"], "Dataset đã đổi; chọn tên run mới"
        reports[architecture] = json.loads((output / "report.json").read_text())
        print("Đọc lại run hoàn tất:", output)
        continue
    resume = (output / "last.pt").is_file()
    reports[architecture] = train_sequence(DATASET, SEQUENCES, output, config, repeats=200, resume=resume)
'''),
        code('''import pandas as pd
rows = []
for architecture, report in reports.items():
    for model, result in report["models"].items():
        rows.append({"model": model, **result["pilot_test"],
                     "all_targets_met": result["gates"]["all_point_targets_met"]})
display(pd.DataFrame(rows)[["model", "auroc", "average_precision", "event_sensitivity",
                          "alarm_ppv", "false_alarms_per_hour", "ece", "all_targets_met"]])
'''),
        md("Đọc `history.json`, `environment.json`, `report.json` trước diễn giải. Chọn ứng viên bằng validation; ba seed/ablation tạo run mới theo runbook. Không chọn seed theo pilot_test. Benchmark T4 cần lưu đúng tên GPU và peak VRAM thực đo.")])
    write("04_case_replay.ipynb", [
        md("# UC04 — báo cáo và phát lại ca thật\n\nNotebook chỉ đọc predictions/checkpoint đã hoàn tất, không huấn luyện lại hoặc đổi ngưỡng."),
        code(SETUP),
        code('''from safeanes.reporting import build_report
DATASET = WORK_ROOT / "data/pilot_v1"
RUN = WORK_ROOT / "artifacts/tcn_v1"
OUT = WORK_ROOT / "reports/tcn_v1"
print(build_report(DATASET, RUN, OUT))
'''),
        code('''from IPython.display import display, Markdown, Image
display(Markdown((OUT / "REPORT.md").read_text(encoding="utf-8")))
for figure in sorted(OUT.glob("*.png")):
    display(Image(filename=str(figure)))
'''),
        code('''import json
diagnostics = json.loads((OUT / "diagnostics.json").read_text(encoding="utf-8"))
for model, result in diagnostics.items():
    print(model, "calibration slope:", result["calibration"]["slope"],
          "intercept:", result["calibration"]["intercept"])
    print("Quality gates:", result["targets"])
'''),
        md("Xem mẫu số biến cố, CI, abstention và alarm censored. Hình MAP dùng giá trị tại lưới quyết định 30 giây; audit onset chính xác cần trở lại raw 1 giây. Diagnostic slope/intercept trên pilot_test không được áp lại để cải thiện score.")])


if __name__ == "__main__":
    main()
