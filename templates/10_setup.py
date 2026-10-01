# Runtime, inputs, integrity checks and paths. The same notebook runs on Kaggle and locally.
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_INPUT = Path(os.environ.get("UC04_KAGGLE_INPUT", "/kaggle/input"))       # overridable for tests only
KAGGLE_WORKING = Path(os.environ.get("UC04_KAGGLE_WORKING", "/kaggle/working"))
RUNTIME = os.environ.get("UC04_RUNTIME_OVERRIDE") or ("kaggle" if KAGGLE_INPUT.exists() else "local")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["UC04_RUNTIME"] = RUNTIME  # on Kaggle the scripts refuse to run without HF (results would be lost)


def find_dirs(root, marker, want_dir=None):
    """Directories under `root` containing `marker` (and sub-directory `want_dir` if given)."""
    out = sorted({p.parent for p in Path(root).rglob(marker)})
    return [d for d in out if want_dir is None or (d / want_dir).is_dir()]


def one(dirs, what):
    """Exactly one match, otherwise stop (never guess between several attached datasets)."""
    if len(dirs) != 1:
        raise RuntimeError(f"expected exactly one {what} in the notebook inputs, found {len(dirs)}: {dirs}. "
                           "Check Add Input and !ls " + str(KAGGLE_INPUT))
    return dirs[0]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if RUNTIME == "kaggle":
    code_src = one(find_dirs(KAGGLE_INPUT, "CODE_COMMIT"), "uc04-code dataset")
    REPO = KAGGLE_WORKING / "uc04"
    shutil.copytree(code_src, REPO, dirs_exist_ok=True)
    got = (REPO / "CODE_COMMIT").read_text().strip()
    if got != CODE_COMMIT:
        raise RuntimeError(f"uc04-code dataset is commit {got[:12]}, this notebook was generated for {CODE_COMMIT[:12]}")
    cfg = json.loads((REPO / "configs" / "uc04_v2.json").read_text(encoding="utf-8"))
    SAMPLES = one(find_dirs(KAGGLE_INPUT, "samples.json"), "uc04-samples-v2 dataset")
    PREP = one(find_dirs(KAGGLE_INPUT, "qc.csv", want_dir="wave100"), "wave dataset (uc04-prep-v1)") if NEEDS_WAVE else None
    INPUTS = []  # results of earlier notebooks come from Hugging Face, never from kernel outputs
    WORK = KAGGLE_WORKING / "work"
    EXTRA = []
else:
    REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs" / "uc04_v2.json").exists())
    cfg = json.loads((REPO / "configs" / "uc04_v2.json").read_text(encoding="utf-8"))
    SAMPLES = REPO / cfg["paths"]["samples"]
    PREP = Path(cfg["paths"]["prep"]) if NEEDS_WAVE else None
    INPUTS = [REPO / "artifacts_dryrun" / f"nb_{n}" for n in LOCAL_INPUT_NOTEBOOKS]
    WORK = REPO / "artifacts_dryrun" / f"nb_{NOTEBOOK}"
    EXTRA = LOCAL_ARGS.split()

# integrity: the exact samples and wave data that were checked locally
got = sha256_of(Path(SAMPLES) / "samples.json")
if got != SAMPLES_SHA256:
    raise RuntimeError(f"samples.json sha256 {got[:12]} != expected {SAMPLES_SHA256[:12]}: wrong uc04-samples-v2 version")
if NEEDS_WAVE:
    got = sha256_of(Path(PREP) / "qc.csv")
    if got != cfg["input"]["qc_sha256"]:
        raise RuntimeError(f"qc.csv sha256 {got[:12]} != input.qc_sha256 {cfg['input']['qc_sha256'][:12]}: "
                           "wrong uc04-prep-v1 dataset")

# Hugging Face is required on Kaggle for notebooks that write results (a stopped session may lose its output)
if RUNTIME == "kaggle" and NEEDS_HF:
    if "<" in cfg["hub"]["runs_repo"]:
        raise RuntimeError("hub.runs_repo in configs/uc04_v2.json is still a placeholder: create a private HF repo, "
                           "set it in the config, commit, and rebuild uc04-code")
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret(cfg["hub"]["token_env"])
    except Exception:
        raise RuntimeError("Kaggle Secret HF_TOKEN is missing or not enabled for this notebook "
                           "(Add-ons -> Secrets). Training on Kaggle requires HF.") from None
    if not token:
        raise RuntimeError("Kaggle Secret HF_TOKEN is empty")
    os.environ[cfg["hub"]["token_env"]] = token  # never printed
    del token
    print("HF_TOKEN secret found; runs_repo =", cfg["hub"]["runs_repo"])

WORK.mkdir(parents=True, exist_ok=True)
print(f"RUNTIME={RUNTIME}\nREPO={REPO}\nSAMPLES={SAMPLES}\nPREP={PREP}\nWORK={WORK}\nINPUTS={INPUTS}\nEXTRA={EXTRA}")
print("checks passed: " + ", ".join((["code commit"] if RUNTIME == "kaggle" else []) + ["samples.json sha256"]
                                  + (["qc.csv sha256"] if NEEDS_WAVE else [])))
