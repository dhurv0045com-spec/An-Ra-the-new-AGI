#!/usr/bin/env python3
"""Canonical An-Ra V4 command-line entry point.

The retired Phase-2 MasterSystem CLI used to start placeholder autonomy,
duplicate memory, and self-improvement systems.  This entry point exposes only
the supported API service and read-only readiness/status commands.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anra", description="An-Ra V4 runtime")
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve = subcommands.add_parser("serve", help="Start the canonical API and Developer UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    subcommands.add_parser("status", help="Print truthful component and artifact status")

    preflight = subcommands.add_parser("preflight", help="Run the V4 training preflight")
    preflight.add_argument("--runtime-class", default="t4_v4_session")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        import uvicorn

        uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
        return 0
    if args.command == "status":
        from runtime.system_registry import build_system_manifest

        manifest = build_system_manifest()
        print(
            json.dumps(
                {
                    "schema_version": manifest["schema_version"],
                    "capabilities": manifest["capabilities"],
                    "artifacts": manifest["artifacts"],
                    "training_readiness": manifest["training_readiness"],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "preflight":
        from training.preflight import run_preflight
        from training.v2_config import CANONICAL_MODEL_PROFILE

        print(
            json.dumps(
                run_preflight(
                    CANONICAL_MODEL_PROFILE,
                    runtime_class=args.runtime_class,
                ).to_dict(),
                indent=2,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
