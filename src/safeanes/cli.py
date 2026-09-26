"""Research commands keep the global final test sealed."""

import argparse
import json

from .data import fetch_pilot
from .dataset import build_pilot
from .experiment import run_pilot


def main():
    parser = argparse.ArgumentParser(description="SafeAnes UC04 research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-pilot", help="Download only global training-pool cases")
    fetch.add_argument("--root", default="data/vitaldb_full")
    fetch.add_argument("--cases", type=int, default=60)
    fetch.add_argument("--workers", type=int, default=4)
    build = sub.add_parser("build-pilot")
    build.add_argument("--root", default="data/vitaldb_full")
    build.add_argument("--out", default="data/pilot_v1")
    train = sub.add_parser("run-pilot")
    train.add_argument("--dataset", default="data/pilot_v1")
    train.add_argument("--out", default="artifacts/pilot_v1")
    train.add_argument("--models", nargs="+", default=["map", "logistic", "hist_gradient"],
                       choices=["map", "logistic", "hist_gradient", "lightgbm"])
    train.add_argument("--bootstrap", type=int, default=200)
    sequence = sub.add_parser("build-sequences", help="Cache causal numeric arrays from existing pilot raw data")
    sequence.add_argument("--dataset", default="data/pilot_v1")
    sequence.add_argument("--root", default="data/vitaldb_full")
    sequence.add_argument("--out", default="data/sequences_v1")
    deep = sub.add_parser("train-sequence", help="Train TCN/Transformer in the global training pool")
    deep.add_argument("--dataset", default="data/pilot_v1")
    deep.add_argument("--sequences", default="data/sequences_v1")
    deep.add_argument("--out", required=True)
    deep.add_argument("--config", required=True, help="Training config JSON")
    deep.add_argument("--bootstrap", type=int, default=200)
    deep.add_argument("--resume", action="store_true")
    render = sub.add_parser("report", help="Generate measured report and case replay figures")
    render.add_argument("--dataset", default="data/pilot_v1")
    render.add_argument("--run", required=True)
    render.add_argument("--out", required=True)
    ensemble = sub.add_parser("ensemble", help="Fixed-weight ensemble of completed sequence checkpoints")
    ensemble.add_argument("--dataset", default="data/pilot_v1")
    ensemble.add_argument("--sequences", default="data/sequences_v1")
    ensemble.add_argument("--runs", nargs="+", required=True)
    ensemble.add_argument("--out", required=True)
    ensemble.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    if args.command == "fetch-pilot":
        result = {"downloaded_cases": len(fetch_pilot(args.root, args.cases, args.workers))}
    elif args.command == "build-pilot":
        result = build_pilot(args.root, args.out)
    elif args.command == "build-sequences":
        from .sequences import build_sequences
        result = build_sequences(args.dataset, args.root, args.out)
    elif args.command == "ensemble":
        from .ensemble import run_ensemble
        report = run_ensemble(args.dataset, args.sequences, args.runs, args.out, repeats=args.bootstrap)
        result = {"scope": report["scope"], "models_completed": list(report["models"]), "errors": report["errors"]}
    elif args.command == "report":
        from .reporting import build_report
        result = build_report(args.dataset, args.run, args.out)
    elif args.command == "train-sequence":
        import os
        from pathlib import Path
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from .training import TrainConfig, train_sequence
        config = TrainConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
        report = train_sequence(args.dataset, args.sequences, args.out, config, args.bootstrap, args.resume)
        result = {"scope": report["scope"], "models_completed": list(report["models"]), "errors": report["errors"]}
    else:
        report = run_pilot(args.dataset, args.out, args.models, args.bootstrap)
        result = {"scope": report["scope"], "models_completed": list(report["models"]), "errors": report["errors"]}
    print(json.dumps(result, indent=2))
    if result.get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
