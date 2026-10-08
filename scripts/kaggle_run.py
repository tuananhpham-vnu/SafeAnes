"""Run one UC04 Kaggle step from a git clone instead of the uc04-code dataset.

In a Kaggle notebook (GPU T4 x2 for 03_dl_*, Internet on, Secret HF_TOKEN enabled,
inputs uc04-samples-v2 and, except 03_dl_finalize, uc04-prep-v1):

    !git clone --depth 1 --filter=blob:none --sparse https://github.com/tuananhpham-vnu/SafeAnes.git
    !cd SafeAnes && git sparse-checkout set src/uc04 scripts templates configs
    !python SafeAnes/scripts/kaggle_run.py 03_dl_W60

Options of this runner go before the step; anything after the step is passed to the
step's script (e.g. `03_dl_W60 --gpus 1`).

Steps are the notebooks of make_kaggle_notebooks.py (00_env_check, 03_dl_smoke,
03_dl_W60, 03_dl_W30, 03_dl_W120, 03_dl_W90, 03_dl_finalize) and run the same
commands with the same checks as templates/: exactly one matching input per dataset,
sha256 of samples.json (kaggle.samples_json_sha256) and of qc.csv (input.qc_sha256),
HF_TOKEN and a real hub.runs_repo, library versions at or above requirements.txt.
The code version is the cloned git commit and the library versions are those of the
Kaggle image (upgraded where older than requirements.txt); provenance.json records both.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE_INPUT = Path(os.environ.get("UC04_KAGGLE_INPUT", "/kaggle/input"))
KAGGLE_WORKING = Path(os.environ.get("UC04_KAGGLE_WORKING", "/kaggle/working"))


def find_dirs(root: Path, marker: str, want_dir: str | None = None) -> list[Path]:
    out = sorted({p.parent for p in root.rglob(marker)})
    return [d for d in out if want_dir is None or (d / want_dir).is_dir()]


def one(dirs: list[Path], what: str) -> Path:
    if len(dirs) != 1:
        raise SystemExit(f"expected exactly one {what} in the notebook inputs, found {len(dirs)}: {dirs}. "
                         f"Check Add Input and !ls {KAGGLE_INPUT}")
    return dirs[0]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sh(*cmd) -> None:
    print("$", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode:
        raise SystemExit(f"command failed with exit code {r.returncode}")


def hf_token(env: str) -> str:
    if os.environ.get(env):
        return os.environ[env]
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret(env)
    except Exception:
        token = None
    if not token:
        raise SystemExit(f"Kaggle Secret {env} is missing or not enabled for this notebook (Add-ons -> Secrets). "
                         "Training on Kaggle requires HF.")
    return token


def main(argv=None) -> int:
    sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]  # uc04 is pip-installed only later
    from make_kaggle_notebooks import NOTEBOOKS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=[n for n in NOTEBOOKS if not n.startswith("02")])
    ap.add_argument("--work", default=str(KAGGLE_WORKING / "work"))
    ap.add_argument("--config", help="config file (default: configs/uc04_v2.json of this clone)")
    ap.add_argument("--skip-install", action="store_true", help="libraries already installed in this session")
    ap.add_argument("--no-hf", action="store_true",
                    help="no HF_TOKEN needed: pass --no-push to the step, outputs stay in --work (no resume)")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="extra arguments for the step's script")
    args = ap.parse_args(argv)
    spec = NOTEBOOKS[args.step]

    if not KAGGLE_INPUT.exists():
        raise SystemExit(f"{KAGGLE_INPUT} not found: this runner is for Kaggle (locally use the scripts directly)")
    os.environ["UC04_RUNTIME"] = "kaggle"  # the scripts refuse to run without HF (results would be lost)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    cfg_path = Path(args.config) if args.config else REPO / "configs" / "uc04_v2.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()

    # inputs and integrity: the exact samples and wave data that were checked locally
    samples = one(find_dirs(KAGGLE_INPUT, "samples.json"), "uc04-samples-v2 dataset")
    got = sha256_of(samples / "samples.json")
    want = cfg["kaggle"]["samples_json_sha256"]
    if got != want:
        raise SystemExit(f"samples.json sha256 {got[:12]} != kaggle.samples_json_sha256 {want[:12]}: "
                         "wrong uc04-samples-v2 version")
    prep = None
    if spec["wave"]:
        prep = one(find_dirs(KAGGLE_INPUT, "qc.csv", want_dir="wave100"), "wave dataset (uc04-prep-v1)")
        got = sha256_of(prep / "qc.csv")
        want = cfg["kaggle"].get("wave_qc_sha256") or cfg["input"]["qc_sha256"]
        if got != want:
            raise SystemExit(f"qc.csv sha256 {got[:12]} != {want[:12]} (kaggle.wave_qc_sha256 or input.qc_sha256): "
                             "wrong uc04-prep-v1 dataset")

    extra = list(args.extra)
    if args.no_hf:  # results stay in /kaggle/working (download the notebook output); no resume across sessions
        print("--no-hf: not using Hugging Face; outputs stay in", args.work)
        if spec.get("hf", True):
            extra.append("--no-push")
    elif spec.get("hf", True):
        if "<" in cfg["hub"]["runs_repo"]:
            raise SystemExit("hub.runs_repo in configs/uc04_v2.json is still a placeholder")
        os.environ[cfg["hub"]["token_env"]] = hf_token(cfg["hub"]["token_env"])  # never printed
        print("HF_TOKEN found; runs_repo =", cfg["hub"]["runs_repo"])

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    print(f"STEP={args.step}\nREPO={REPO} (commit {commit[:12]})\nSAMPLES={samples}\nPREP={prep}\nWORK={work}",
          flush=True)

    # libraries (as templates/20_install.py and 30_check.py): keep the Kaggle image's versions, including
    # its CUDA build of torch, and install or upgrade only what is missing or older than requirements.txt
    if not args.skip_install:
        sh(sys.executable, "-m", "pip", "install", "-q", "-r", REPO / "requirements.txt")
        sh(sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", REPO)
    sh(sys.executable, REPO / "scripts" / "env_check.py", *([] if spec["torch"] else ["--ignore", "torch"]))

    # the step itself (as templates/40_run.py); results of earlier steps come from HF only
    fill = {"{SAMPLES}": samples, "{WORK}": work, "{PREP}": prep}
    for step in spec["run"]:
        cmd = [fill.get(a, a) for a in step]
        if "{INPUTS}" in cmd:  # no earlier-notebook inputs on Kaggle: drop the placeholder and its flag
            i = cmd.index("{INPUTS}")
            j = i - 1 if i > 0 and cmd[i - 1] == "--inputs" else i
            cmd = cmd[:j] + cmd[i + 1:]
        sh(sys.executable, REPO / cmd[0], *cmd[1:], *(["--config", cfg_path] if args.config else []), *extra)
    print(f"{args.step} done (commit {commit[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
