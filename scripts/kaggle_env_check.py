"""00_env_check: check inputs, versions and HF before training on Kaggle (CPU).

    python scripts/kaggle_env_check.py --samples <dir> [--prep <wave dataset prep_v1 dir>] --work <dir>

- samples.json matches the config (samples_digest, qc.csv sha256);
- if --prep is given (the Kaggle Dataset with wave100): its qc.csv sha256 equals
  input.qc_sha256, and wave100/ has one file per case with a waveform;
- library versions satisfy requirements.txt (torch ignored here: CPU notebook);
- HF: if a token and runs_repo are set, writes runs/healthcheck_<utc>.json and
  pushes it; otherwise says that results will stay in the notebook output.
Exit code 1 if any required check fails.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import pandas as pd

from uc04.config import load_config
from uc04.hub import HubRequired, hub_from_config
from uc04.io import read_json, sha256_file, write_json
from uc04.samples import SAMPLES_SECTIONS

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "uc04_v2.json"))
    ap.add_argument("--samples", required=True)
    ap.add_argument("--prep")
    ap.add_argument("--work", required=True)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ok = True

    def report(name: str, passed: bool, detail="") -> None:
        nonlocal ok
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")

    samples = Path(args.samples)
    rec = read_json(samples / "samples.json")
    report("samples.json samples_digest", rec.get("samples_digest") == cfg.digest(*SAMPLES_SECTIONS))
    report("samples.json qc_sha256", rec.get("qc_sha256") == cfg.input.qc_sha256)
    report("samples has no sealed/test files", not any("test" in p.name or "sealed" in p.parts
                                                       for p in samples.rglob("*") if p.is_file()))
    if args.prep:
        prep = Path(args.prep)
        sha = sha256_file(prep / "qc.csv")
        report("wave dataset qc.csv sha256 == input.qc_sha256", sha == cfg.input.qc_sha256, sha[:16])
        qc = pd.read_csv(prep / "qc.csv")
        need = set(qc.loc[qc.has_wave.astype(bool), "caseid"].astype(int))
        have = {int(p.stem) for p in (prep / "wave100").glob("*.npy")}
        report("wave100 files", need <= have, f"{len(have)} files, missing {len(need - have)}")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "env_check.py"), "--ignore", "torch"])
    report("library versions >= requirements.txt (torch ignored)", r.returncode == 0)

    work = Path(args.work)
    try:
        hub = hub_from_config(cfg, work, push=not args.no_push)
    except HubRequired as exc:
        report("HF set up (required on Kaggle)", False, str(exc))
        print("SOME CHECKS FAILED")
        return 1
    if hub.enabled:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rel = f"runs/healthcheck_{stamp}.json"
        write_json(work / rel, {"utc": stamp, "samples_digest": rec.get("samples_digest")})
        hub.register(rel)
        report("HF push (healthcheck)", hub.push([rel], message="healthcheck"))
    else:
        print("[INFO] HF not set up: results will only be in the notebook output (/kaggle/working)")
    print("ALL REQUIRED CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
