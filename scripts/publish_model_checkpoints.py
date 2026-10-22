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


def publish(output: Path, entries: list[dict]) -> None:
    import os

    sys.path.insert(0, str(ROOT / ".local_deps"))
    from dotenv import load_dotenv
    from huggingface_hub import CommitOperationAdd, HfApi

    load_dotenv(ROOT / ".env")
    token = os.getenv("SAFEANES_HF_TOKEN_WRITE") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("No Hugging Face write token configured")
    api = HfApi(token=token)
    owner = api.whoami()["name"]
    repo_id = f"{owner}/{NAME}"
    print(f"Uploading {repo_id}: 4 families, 20 checkpoints", flush=True)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    # Remote payload is strictly the 20 user-requested checkpoint files.
    # All supporting source, configuration, metrics and manifests stay local.
    allowed = [f"models/{model}/W120/h{horizon}/model.joblib"
               for model in RUNS for horizon in HORIZONS]
    assert len(allowed) == 20 and all((output / path).is_file() for path in allowed)
    operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(output / path))
                  for path in allowed]
    commit = api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                               commit_message="Upload only 20 checkpoints for four 71-feature models")
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
    (output.parent / f"{NAME}_upload.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    folder, files = package()
    print(json.dumps({"package": str(folder), "files": len(files) + 1,
                      "bytes": sum(file["size"] for file in files), "checkpoints": 20}), flush=True)
    if args.upload:
        publish(folder, files)
