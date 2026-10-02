"""Generate the Kaggle notebooks (plan 15.4) from templates/.

    python scripts/make_kaggle_notebooks.py --kaggle-user <user> [--only 02a_baselines ...]
    python scripts/make_kaggle_notebooks.py --kaggle-user <user> --run-local [--only ...]

Each notebook gets notebooks/kaggle/<name>/<name>.ipynb and kernel-metadata.json,
with the current commit filled in. The same .ipynb runs on Kaggle and locally
(RUNTIME is detected); locally it uses the dry-run subset and never pushes.
--run-local executes the notebooks here with nbconvert, in dependency order.
Needs a clean working tree so the notebook matches the code dataset.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import nbformat

from uc04.config import load_config
from uc04.io import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ("10_setup.py", "20_install.py", "30_check.py", "40_run.py", "50_summary.py")
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
TAB_LOCAL = "--subset dryrun --bootstrap 5"
DL_LOCAL = "--subset dryrun --max-samples 2000 --epochs 1 --seeds 1 --num-workers 2"
DL_LOCAL_SMOKE = "--subset dryrun --max-samples 2000 --num-workers 2"

# name: gpu, needs wave dataset, needs torch, kernel inputs, steps, local args, local input notebooks, title
NOTEBOOKS: dict[str, dict] = {
    "00_env_check": dict(
        gpu=False, wave=True, torch=False, after=[], local="--no-push",
        run=[["scripts/kaggle_env_check.py", "--samples", "{SAMPLES}", "--prep", "{PREP}", "--work", "{WORK}"]],
        title="Check inputs (samples, wave dataset), library versions and HF before training."),
    "02a_baselines": dict(
        gpu=False, wave=False, torch=False, hf=False, after=[], local=TAB_LOCAL,
        run=[["scripts/train_tabular.py", "--models", "map_threshold", "map_logistic",
              "--samples", "{SAMPLES}", "--work", "{WORK}"]],
        title="Baselines: map_threshold (5 combinations) and map_logistic (20)."),
    "02b_lgbm_numeric": dict(
        gpu=False, wave=False, torch=False, hf=False, after=[], local=TAB_LOCAL,
        run=[["scripts/train_tabular.py", "--models", "lgbm_numeric", "--samples", "{SAMPLES}", "--work", "{WORK}"]],
        title="LightGBM on 66 numeric features: 4 windows x 5 horizons."),
    "02c_lgbm_wave": dict(
        gpu=False, wave=False, torch=False, hf=False, after=[], local=TAB_LOCAL,
        run=[["scripts/train_tabular.py", "--models", "lgbm_wave", "--samples", "{SAMPLES}", "--work", "{WORK}"]],
        title="LightGBM on 66 numeric + 27 waveform features: 4 windows x 5 horizons."),
    "02d_tabular_analysis": dict(
        gpu=False, wave=False, torch=False, hf=False, after=["02a_baselines", "02b_lgbm_numeric", "02c_lgbm_wave"],
        local=TAB_LOCAL,
        run=[["scripts/tabular_analysis.py", "--samples", "{SAMPLES}", "--work", "{WORK}", "--inputs", "{INPUTS}"]],
        title="Combine the 65 results, choose W*, SHAP, ablation, confidence levels."),
    "03_dl_smoke": dict(
        gpu=True, wave=True, torch=True, after=[], local=DL_LOCAL_SMOKE,
        run=[["scripts/train_dl.py", "--window", "60", "--smoke", "--samples", "{SAMPLES}", "--prep", "{PREP}",
              "--work", "{WORK}"]],
        title="GPU check and 1 epoch at W = 60: time per epoch, RAM, VRAM, estimated GPU hours (timing only)."),
    **{f"03_dl_W{W}": dict(
        gpu=True, wave=True, torch=True, after=[], local=DL_LOCAL,
        run=[["scripts/train_dl.py", "--window", str(W), "--samples", "{SAMPLES}", "--prep", "{PREP}",
              "--work", "{WORK}"],
             ["scripts/train_dl.py", "--window", str(W), "--final-push-only", "--samples", "{SAMPLES}",
              "--prep", "{PREP}", "--work", "{WORK}"]],
        title=f"Conv1D + Transformer, W = {W} s, all seeds (resumes and skips finished seeds).")
       for W in (60, 30, 120, 90)},
    "03_dl_finalize": dict(
        gpu=False, wave=False, torch=False, after=["03_dl_W60", "03_dl_W30", "03_dl_W120", "03_dl_W90"],
        local="--subset dryrun --bootstrap 5",
        run=[["scripts/dl_finalize.py", "--samples", "{SAMPLES}", "--work", "{WORK}", "--inputs", "{INPUTS}"]],
        title="Ensemble seeds, shared temperature and bias, thresholds, confidence (CPU, per-seed logits only)."),
}


def dataset_ids(cfg, user: str) -> dict[str, str]:
    """<user>/uc04-code, <user>/uc04-samples-v2, <user>/uc04-prep-v1 (names from the config)."""
    k = cfg.kaggle
    return {key: getattr(k, f"{key}_dataset").replace("<KAGGLE_USER>", user) for key in ("code", "samples", "wave")}


# everything a Kaggle notebook executes: if none changed, notebooks may target an older code commit
KAGGLE_RUNTIME_PATHS = ("src", "configs", "templates", "requirements.txt",
                        "pyproject.toml", "scripts/train_dl.py", "scripts/dl_finalize.py",
                        "scripts/kaggle_env_check.py", "scripts/env_check.py")


def slug(name: str) -> str:
    return "uc04-v2-" + name.replace("_", "-").lower()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def build(name: str, spec: dict, user: str, cfg, commit: str, torch_index: str = TORCH_INDEX,
          samples_sha: str = "", out_root: Path | None = None) -> Path:
    params = "\n".join([
        f"# {name}: {spec['title']}",
        "# Generated by scripts/make_kaggle_notebooks.py; edit templates/, not this file.",
        f"NOTEBOOK = {name!r}",
        f"CODE_COMMIT = {commit!r}",
        f"RUN = {spec['run']!r}",
        f"SAMPLES_SHA256 = {samples_sha!r}  # sha256 of samples_v2/samples.json that was checked locally",
        f"NEEDS_WAVE = {spec['wave']!r}",
        f"NEEDS_HF = {spec.get('hf', True)!r}  # on Kaggle: stop at once without HF_TOKEN or a real hub.runs_repo",
        f"NEEDS_TORCH = {spec['torch']!r}",
        f"TORCH_INDEX = {torch_index!r}  # CUDA build of the locked torch version",
        f"LOCAL_ARGS = {spec['local']!r}  # local run only: dry-run subset, never pushes",
        f"LOCAL_INPUT_NOTEBOOKS = {spec['after']!r}",
    ])
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(f"# {name}\n\n{spec['title']}\n\nCode commit `{commit[:12]}`. "
                                              "Runs on Kaggle and locally (dry-run subset).")]
    nb.cells.append(nbformat.v4.new_code_cell(params))
    for t in TEMPLATES:
        nb.cells.append(nbformat.v4.new_code_cell((ROOT / "templates" / t).read_text(encoding="utf-8").rstrip()))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = Path(out_root or ROOT / "notebooks" / "kaggle") / name
    out.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out / f"{name}.ipynb")
    ds = dataset_ids(cfg, user)
    meta = {
        "id": f"{user}/{slug(name)}", "title": slug(name), "code_file": f"{name}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": spec["gpu"], "enable_internet": True,
        "dataset_sources": [ds["code"], ds["samples"], *([ds["wave"]] if spec["wave"] else [])],
        "kernel_sources": [],  # results of earlier notebooks are read from HF only (DEVIATIONS 31)
        "competition_sources": [],
    }
    if spec["gpu"]:
        meta["machine_shape"] = cfg.kaggle.dl_machine.get("machine_shape", "NvidiaTeslaT4")
    write_json(out / "kernel-metadata.json", meta)
    return out / f"{name}.ipynb"


def run_local(path: Path) -> bool:
    out_dir = ROOT / "artifacts_dryrun" / "executed"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--execute", "--to", "notebook",
                        "--ExecutePreprocessor.timeout=3600", "--output-dir", str(out_dir), str(path)],
                       cwd=path.parent, capture_output=True, text=True)
    print(f"  {'OK  ' if r.returncode == 0 else 'FAIL'} {path.name} ({time.time() - t0:.0f} s)")
    if r.returncode:
        print(r.stderr[-3000:])
    return r.returncode == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "uc04_v2.json"))
    ap.add_argument("--kaggle-user")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--run-local", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="for local runs while developing")
    ap.add_argument("--torch-index", default=TORCH_INDEX, help="pip index with the CUDA build of torch")
    ap.add_argument("--include-tabular", action="store_true",
                    help="also generate the 02* notebooks (tabular normally runs locally)")
    ap.add_argument("--samples-sha256", help="sha256 of samples.json on Kaggle (default: computed from the local "
                                             "samples_v2; pass it when samples_v2 is not on this machine)")
    ap.add_argument("--code-commit", help="commit of the uc04-code dataset already on Kaggle (default HEAD); "
                                          "allowed only if everything the notebooks run is unchanged since then")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    user = args.kaggle_user or cfg.kaggle.user  # "<KAGGLE_USER>" until the account is known
    if git("status", "--porcelain", "--untracked-files=no") and not args.allow_dirty:
        print("working tree has uncommitted changes; commit first (the notebook records the commit)", file=sys.stderr)
        return 1
    commit = git("rev-parse", "HEAD")
    if args.code_commit:
        target = git("rev-parse", args.code_commit)
        changed = git("diff", "--name-only", target, "HEAD", "--", *KAGGLE_RUNTIME_PATHS)
        if changed:
            print(f"cannot generate for {target[:12]}: files the Kaggle notebooks run changed since then: "
                  f"{changed.split()}; re-package and upload uc04-code at HEAD instead", file=sys.stderr)
            return 1
        commit = target
    # tabular (02*) runs on the local machine (same library versions for NB04, DEVIATIONS 24)
    names = args.only or [n for n in NOTEBOOKS if args.include_tabular or not n.startswith("02")]
    samples_sha = args.samples_sha256 or sha256_file(cfg.path("samples") / "samples.json")
    paths = [build(n, NOTEBOOKS[n], user, cfg, commit, args.torch_index, samples_sha) for n in names]
    for p in paths:
        print(f"wrote {p.relative_to(ROOT)}")
    if args.run_local:
        print("running locally (dry-run subset, no push):")
        if not all(run_local(p) for p in paths):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
