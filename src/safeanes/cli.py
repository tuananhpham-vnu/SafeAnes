"""Commands intentionally stop at a TRAIN-pool pilot in release 0.1."""

import argparse
import json

from .data import fetch_pilot
from .dataset import build_pilot
from .experiment import run_pilot


def main():
    parser = argparse.ArgumentParser(description="SafeAnes UC04 research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-pilot", help="Download only global training-pool cases")
    fetch.add_argument("--root", default="data/vitaldb")
    fetch.add_argument("--cases", type=int, default=60)
    fetch.add_argument("--workers", type=int, default=4)
    build = sub.add_parser("build-pilot")
    build.add_argument("--root", default="data/vitaldb")
    build.add_argument("--out", default="data/pilot_v1")
    train = sub.add_parser("run-pilot")
    train.add_argument("--dataset", default="data/pilot_v1")
    train.add_argument("--out", default="artifacts/pilot_v1")
    train.add_argument("--models", nargs="+", default=["map", "logistic", "hist_gradient"],
                       choices=["map", "logistic", "hist_gradient", "lightgbm"])
    train.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    if args.command == "fetch-pilot":
        result = {"downloaded_cases": len(fetch_pilot(args.root, args.cases, args.workers))}
    elif args.command == "build-pilot":
        result = build_pilot(args.root, args.out)
    else:
        report = run_pilot(args.dataset, args.out, args.models, args.bootstrap)
        result = {"scope": report["scope"], "models_completed": list(report["models"]), "errors": report["errors"]}
    print(json.dumps(result, indent=2))
    if result.get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
