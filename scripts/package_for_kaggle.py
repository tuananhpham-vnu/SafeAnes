"""Package samples_v2 and the code for upload as private Kaggle Datasets (nothing is uploaded).

    python scripts/package_for_kaggle.py [--kaggle-user <user>] [--out dist]
    python scripts/package_for_kaggle.py --code-only     # new uc04-code version only (no samples_v2 needed)

Reads data/samples_v2 (never writes to it) and writes, under dist/ (not committed):
  uc04-samples-v2.zip            every file of samples_v2, ZIP_STORED (parquet is already compressed)
  uc04-code.zip                  git archive of HEAD + CODE_COMMIT (full HEAD sha)
  kaggle/uc04-samples-v2/        staging copy of samples_v2 + dataset-metadata.json
  kaggle/uc04-code/              unpacked uc04-code.zip + dataset-metadata.json
  for_review/                    upload_manifest.json, norm_tabular.json, provenance.json, samples.json
Refuses to run unless the working tree is clean, and checks that the code package holds
no data, artifacts, sealed files, credentials or model files.
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from uc04.config import load_config
from uc04.io import read_json, sha256_file, write_json
from uc04.provenance import git_commit
from uc04.runtime import REPO_CONFIG

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("data/*", "artifacts*", "*sealed*", "*.env", ".env", "*kaggle.json", "*token*", "*.pt", "*.joblib",
             "*.npz", "*.npy", "*.parquet")


def git(*args: str, binary: bool = False):
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode().strip()


def forbidden_names(names) -> list[str]:
    bad = []
    for n in names:
        low = n.lower()
        parts = low.split("/")
        if any(fnmatch.fnmatch(low, pat) or any(fnmatch.fnmatch(p, pat) for p in parts) for pat in FORBIDDEN):
            bad.append(n)
    return bad


def listing(folder: Path) -> list[dict]:
    return [{"path": p.relative_to(folder).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(folder.rglob("*")) if p.is_file()]


def metadata(user: str, slug: str, subtitle: str) -> dict:
    return {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "other"}], "subtitle": subtitle}


def package_code(out: Path, user: str, head: str):
    """dist/uc04-code.zip (git archive of HEAD + CODE_COMMIT) and its staging folder.
    Returns (zip, staging dir, file names, failed) where failed is truthy if the package check failed."""
    c_zip = out / "uc04-code.zip"
    archive = git("archive", "--format=zip", "HEAD", binary=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as src, zipfile.ZipFile(c_zip, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            if not info.is_dir():
                dst.writestr(info.filename, src.read(info.filename))
        dst.writestr("CODE_COMMIT", head + "\n")
    with zipfile.ZipFile(c_zip) as zf:
        code_names = [n for n in zf.namelist() if not n.endswith("/")]
        bad = forbidden_names(code_names)
        committed = zf.read("CODE_COMMIT").decode().strip()
        c_stage = out / "kaggle" / "uc04-code"
        zf.extractall(c_stage)
    if bad or committed != head:
        print(f"code package check failed: forbidden {bad[:10]}, CODE_COMMIT {committed[:12]} vs HEAD {head[:12]}",
              file=sys.stderr)
        return c_zip, c_stage, code_names, bad or ["CODE_COMMIT mismatch"]
    write_json(c_stage / "dataset-metadata.json", metadata(user, "uc04-code", f"UC04 v2 code at commit {head[:12]}"))
    return c_zip, c_stage, code_names, []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_CONFIG))
    ap.add_argument("--kaggle-user", help="default: kaggle.user in the config (<KAGGLE_USER> placeholder)")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--code-only", action="store_true",
                    help="package only uc04-code (after a code change; samples_v2 on Kaggle is unchanged)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    user = args.kaggle_user or cfg.kaggle.user
    if git("status", "--porcelain", "--untracked-files=no"):
        print("working tree has uncommitted changes; package only from a clean commit", file=sys.stderr)
        return 1
    head = git("rev-parse", "HEAD")
    samples = cfg.path("samples")
    if args.code_only:
        out = Path(args.out)
        shutil.rmtree(out / "kaggle" / "uc04-code", ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        c_zip, c_stage, _, bad = package_code(out, user, head)
        if bad:
            return 1
        print(f"code zip {c_zip} {c_zip.stat().st_size / 1e3:.0f} kB sha256 {sha256_file(c_zip)}; commit {head}")
        print(f"staging: {c_stage}  ->  kaggle datasets version -p {c_stage} --dir-mode zip -m 'code {head[:12]}'")
        return 0
    if (samples / "dataset-metadata.json").exists():
        print(f"{samples}/dataset-metadata.json exists: samples_v2 must contain only build_samples output",
              file=sys.stderr)
        return 1
    out = Path(args.out)
    for d in ("kaggle", "for_review"):
        shutil.rmtree(out / d, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    # ---- samples: zip (stored) and staging copy
    s_zip = out / "uc04-samples-v2.zip"
    s_files = sorted(p for p in samples.rglob("*") if p.is_file())
    with zipfile.ZipFile(s_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in s_files:
            zf.write(p, p.relative_to(samples).as_posix())
    s_stage = out / "kaggle" / "uc04-samples-v2"
    shutil.copytree(samples, s_stage)
    write_json(s_stage / "dataset-metadata.json", metadata(
        user, "uc04-samples-v2", "UC04 v2 samples (train/calibration/validation only, no test data)"))

    # ---- code: git archive of HEAD + CODE_COMMIT
    c_zip, c_stage, code_names, bad = package_code(out, user, head)
    if bad:
        return 1

    # ---- manifest and review copies
    review = out / "for_review"
    review.mkdir(parents=True)
    for f in ("norm_tabular.json", "provenance.json", "samples.json"):
        shutil.copy2(samples / f, review / f)
    sprov = read_json(samples / "provenance.json")
    manifest = {
        "kaggle_user": user,
        "datasets": {"samples": f"{user}/uc04-samples-v2", "code": f"{user}/uc04-code",
                     "wave": cfg.kaggle.wave_dataset.replace("<KAGGLE_USER>", user)},
        "code": {"commit": head, "git": git_commit(ROOT), "zip": {"path": str(c_zip), "bytes": c_zip.stat().st_size,
                                                                  "sha256": sha256_file(c_zip)},
                 "files": [n for n in code_names], "forbidden_found": bad},
        "samples": {"samples_json_sha256": sha256_file(samples / "samples.json"),
                    "built_by": {"step": sprov.get("step"), "commit": sprov.get("git", {}).get("commit"),
                                 "dirty": sprov.get("git", {}).get("dirty"),
                                 "dirty_paths": sprov.get("git", {}).get("dirty_paths", "not recorded (older format)"),
                                 "created_utc": sprov.get("created_utc"), "config_digest": sprov.get("config_digest")},
                    "zip": {"path": str(s_zip), "bytes": s_zip.stat().st_size, "sha256": sha256_file(s_zip),
                            "compression": "stored"},
                    "files": listing(samples)},
        "staging": {"samples": str(s_stage), "code": str(c_stage)},
    }
    write_json(review / "upload_manifest.json", manifest)
    print(f"samples zip {s_zip} {s_zip.stat().st_size / 1e6:.0f} MB sha256 {manifest['samples']['zip']['sha256']}")
    print(f"code zip    {c_zip} {c_zip.stat().st_size / 1e3:.0f} kB sha256 {manifest['code']['zip']['sha256']}")
    print(f"samples.json sha256 {manifest['samples']['samples_json_sha256']}; code commit {head}")
    print(f"staging: {s_stage} and {c_stage}; manifest: {review / 'upload_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
