"""Command-line entry point: ``python -m bramastra_lab.research.cli``.

``--help`` and argument parsing never import torch, load a model, or touch
the accelerator. Each subcommand imports its implementing module lazily so
operators can inspect availability without paying for the backend.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from bramastra_lab.research.errors import CommandError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bramastra_lab.research.cli",
        description="BRAMASTRA integrated build: one from-scratch learner with "
                    "controlled plasticity. Subcommands inspect availability, "
                    "prepare data, train, resume, infer, evaluate and package.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    inspect = subparsers.add_parser(
        "inspect", help="validate config, corpus supply and predicted device fit "
                        "without launching anything")
    inspect.add_argument("--config", required=True, help="path to a build-config JSON file")
    inspect.add_argument("--data-manifest", default=None,
                         help="path to a dataset manifest produced by prepare-data")
    inspect.add_argument("--device", action="store_true",
                         help="report the installed torch backend and predicted memory fit")
    inspect.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    prepare = subparsers.add_parser(
        "prepare-data", help="validate a local dataset manifest and build packed "
                             "training artifacts with content identities")
    prepare.add_argument("--manifest", required=True, help="operator-supplied dataset manifest JSON")
    prepare.add_argument("--out", required=True, help="output directory for prepared data")
    prepare.add_argument("--config", required=True, help="build-config JSON used for packing limits")

    train = subparsers.add_parser(
        "train", help="train from random initialization through the real integrated path")
    train.add_argument("--config", required=True, help="build-config JSON")
    train.add_argument("--data", required=True, help="directory produced by prepare-data")
    train.add_argument("--run-dir", required=True, help="new run directory (never overwrite)")
    train.add_argument("--max-updates", type=int, default=None,
                       help="stop after this many optimizer updates")
    train.add_argument("--smoke", action="store_true",
                       help="declare a bounded learned-smoke run; refused when the "
                            "cumulative session smoke ledger is exhausted")

    resume = subparsers.add_parser(
        "resume", help="restore the latest valid checkpoint in a fresh process and continue")
    resume.add_argument("--run-dir", required=True, help="existing run directory")
    resume.add_argument("--max-updates", type=int, default=None,
                        help="stop after this many additional optimizer updates")
    resume.add_argument("--expect-parent", default=None,
                        help="restore only the checkpoint descending from this identity")

    infer = subparsers.add_parser(
        "infer", help="produce complete answers or action scores with a checkpointed model")
    infer.add_argument("--config", required=True, help="build-config JSON")
    infer.add_argument("--checkpoint", required=True, help="checkpoint directory from train/resume")
    infer.add_argument("--input", required=True, help="JSON file with the public inference request")
    infer.add_argument("--out", default=None, help="write the inference report JSON here")
    infer.add_argument("--max-new-tokens", type=int, default=None,
                       help="generation cap; defaults to the configured sequence limit")

    evaluate = subparsers.add_parser(
        "evaluate", help="recompute complete-answer and goal metrics from raw outcomes")
    evaluate.add_argument("--checkpoint", required=True, help="checkpoint directory to evaluate")
    evaluate.add_argument("--config", required=True, help="build-config JSON")
    evaluate.add_argument("--data", required=True, help="prepared data directory with the eval split")
    evaluate.add_argument("--split", default="development",
                          help="which prepared split to score (never the sealed pool)")
    evaluate.add_argument("--out", required=True, help="output report JSON path")

    package = subparsers.add_parser(
        "package", help="write the operator handoff manifest for a finished run")
    package.add_argument("--run-dir", required=True, help="existing run directory")
    package.add_argument("--out", required=True, help="output package manifest JSON path")

    return parser


def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise CommandError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CommandError(f"{path} is not valid JSON: {exc}") from exc


def cmd_inspect(args: argparse.Namespace) -> int:
    from bramastra_lab.research.config import BuildConfig
    from bramastra_lab.research.runtime import readiness

    raw_config = _load_json(args.config)
    report = readiness.inspect(
        raw_config,
        data_manifest_path=args.data_manifest,
        check_device=args.device,
    )
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        for key in ("config_status", "data_status", "device_status", "parameter_count",
                    "base_decoder_parameter_count"):
            if key in report:
                print(f"{key}: {report[key]}")
        if "detail" in report:
            print(f"detail: {report['detail']}")
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    from bramastra_lab.research.commands import prepare_data

    return prepare_data(manifest_path=args.manifest, out_dir=args.out, config_path=args.config)


def cmd_train(args: argparse.Namespace) -> int:
    from bramastra_lab.research.commands import train

    return train(config_path=args.config, data_dir=args.data, run_dir=args.run_dir,
                 max_updates=args.max_updates, smoke=args.smoke)


def cmd_resume(args: argparse.Namespace) -> int:
    from bramastra_lab.research.commands import resume

    return resume(run_dir=args.run_dir, max_updates=args.max_updates,
                  expect_parent=args.expect_parent)


def cmd_infer(args: argparse.Namespace) -> int:
    from bramastra_lab.research.commands import infer

    return infer(config_path=args.config, checkpoint=args.checkpoint, input_path=args.input,
                 out_path=args.out, max_new_tokens=args.max_new_tokens)


def cmd_evaluate(args: argparse.Namespace) -> int:
    from bramastra_lab.research.commands import evaluate

    return evaluate(checkpoint=args.checkpoint, config_path=args.config, data_dir=args.data,
                    split=args.split, out_path=args.out)


def cmd_package(args: argparse.Namespace) -> int:
    from bramastra_lab.research.commands import package

    return package(run_dir=args.run_dir, out_path=args.out)


HANDLERS = {
    "inspect": cmd_inspect,
    "prepare-data": cmd_prepare_data,
    "train": cmd_train,
    "resume": cmd_resume,
    "infer": cmd_infer,
    "evaluate": cmd_evaluate,
    "package": cmd_package,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    handler = HANDLERS[args.command]
    try:
        return handler(args)
    except CommandError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
