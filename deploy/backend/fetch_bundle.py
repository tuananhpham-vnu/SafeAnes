"""Make sure the ward bundle (export_ward.py output, patient data: never in the image or in git) is in SAFEANES_DATA.

Run at container start, before the server. If SAFEANES_DATA already holds ward.json (a mounted volume), nothing is
done. Otherwise the bundle is downloaded from a private Hugging Face dataset: SAFEANES_BUNDLE_REPO (repo id) with
HF_TOKEN (read access).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    data = Path(os.environ.get("SAFEANES_DATA", Path(__file__).resolve().parent / "data_ward"))
    if (data / "ward.json").exists():
        print(f"bundle present in {data}")
        return 0
    repo = os.environ.get("SAFEANES_BUNDLE_REPO")
    if not repo:
        print(f"no bundle in {data} and SAFEANES_BUNDLE_REPO is not set", file=sys.stderr)
        return 1
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(data), token=os.environ.get("HF_TOKEN"))
    if not (data / "ward.json").exists():
        print(f"{repo} has no ward.json", file=sys.stderr)
        return 1
    print(f"bundle downloaded from {repo} to {data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
