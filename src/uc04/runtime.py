"""Shared command-line plumbing for the training scripts (NB02, NB03)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config
from .io import read_json, sha256_file
from .loaders import load_split
from .samples import SAMPLES_SECTIONS


REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "uc04_v2.json"  # works from any cwd


def common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--config", default=str(REPO_CONFIG))
    ap.add_argument("--samples", help="samples_v2 directory (on Kaggle: /kaggle/input/uc04-samples-v2)")
    ap.add_argument("--work", help="output root with artifacts/ and reports/ (default: repo root; "
                                   "artifacts_dryrun/ with --subset)")
    ap.add_argument("--subset", help="name of a subset in <samples>/subsets/ (e.g. dryrun); implies --no-push")
    ap.add_argument("--no-push", action="store_true", help="never push to Hugging Face")
    return ap


def setup(args):
    """Load config, check the samples match it, resolve the work directory and push flag."""
    cfg = load_config(args.config).with_paths(samples=args.samples)
    samples = cfg.path("samples")
    rec = read_json(samples / "samples.json")
    if rec.get("samples_digest") != cfg.digest(*SAMPLES_SECTIONS) or rec.get("qc_sha256") != cfg.input.qc_sha256:
        raise SystemExit(f"{samples}/samples.json does not match the config (samples_digest / qc_sha256)")
    if rec.get("limit"):
        raise SystemExit("samples were built with --limit; use the full build")
    root = Path(cfg.root)
    work = Path(args.work) if args.work else (root / "artifacts_dryrun" if args.subset else root)
    push = not (args.no_push or args.subset)
    subset = None
    if args.subset:
        subset = read_json(samples / "subsets" / f"{args.subset}.json")["cases"]
    return cfg, samples, work, push, subset


def load_data(samples: Path, columns: list[str], subset: dict | None, splits=("train", "calibration", "validation")):
    labels, feats = {}, {}
    for s in splits:
        lab, f = load_split(samples, s, columns)
        if subset is not None:
            keep = lab["caseid"].isin(subset.get(s, [])).to_numpy()
            lab, f = lab[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
        labels[s], feats[s] = lab, f
    events = pd.read_parquet(samples / "events.parquet")
    cases = pd.concat([labels[s]["caseid"] for s in splits]).unique()
    return labels, feats, events[events["caseid"].isin(cases)].reset_index(drop=True)


def samples_inputs(samples: Path) -> dict:
    rec = read_json(samples / "samples.json")
    return {"samples_digest": rec.get("samples_digest"), "qc.csv": rec.get("qc_sha256"),
            "samples.json": sha256_file(samples / "samples.json")}
