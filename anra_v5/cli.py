"""Single dispatcher for every executable operator command.

Each subcommand is a thin caller of its owning module's ``main``: business
logic never lives here. Submodules resolve lazily by dotted path so the
dispatcher carries no static dependency on any research or control plane.
"""

from __future__ import annotations

import importlib
import sys


SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "contracts": ("v5_contracts.certify", "verify implementation contracts"),
    "training-spec": ("v5_contracts.training_spec", "regenerate the frozen training-spec receipt"),
    "readiness": ("v5_contracts.launch_readiness", "evaluate launch-gate evidence inventory"),
    "boundaries": ("v5_contracts.import_boundaries", "enforce dependency-boundary scan"),
    "collect": ("v5_remote.collect", "collect and bind a remote job result"),
    "e0-certify": ("e0_cognition.certify", "run E0 development certification"),
    "e0-scoring": ("e0_cognition.scoring_certification", "certify the model-scoring adapter"),
    "e0-seal": ("e0_cognition.sealed", "commit to an external sealed fixture"),
    "e1-audit": ("e1_tokenizer.audit", "audit one tokenizer candidate artifact"),
    "e1-plan": ("e1_tokenizer.tournament", "emit the matched-budget tournament plan"),
    "e1-local": ("e1_tokenizer.local_tournament", "run the local development tournament"),
    "e1-perturbation": ("e1_tokenizer.perturbation_sweep", "run the tokenizer perturbation sweep"),
    "e2-plan": ("e2_architecture.plan", "emit the P35 architecture screen plan"),
    "e2-device": ("e2_architecture.device_benchmark", "run the attention-kernel device probe"),
    "e2-device-aggregate": ("e2_architecture.aggregate", "aggregate device receipts"),
    "e2-block": ("e2_architecture.block_benchmark", "run the full-stack shape canary"),
    "e2-block-aggregate": ("e2_architecture.block_aggregate", "aggregate block receipts"),
    "e2-signal": ("e2_architecture.signal_benchmark", "run the residual-scaling probe"),
    "e2-qk-norm": ("e2_architecture.qk_norm_benchmark", "run the QK-norm scale-control probe"),
    "e2-precision": ("e2_architecture.precision_benchmark", "run the BF16 parity probe"),
    "e2-rope": ("e2_architecture.rope_benchmark", "run the RoPE conformance probe"),
    "e2-update": ("e2_architecture.update_benchmark", "run the real-update canary"),
    "e2-cursor": ("e2_architecture.cursor_benchmark", "run the sampler-cursor canary"),
    "e2-scoring": ("e2_architecture.scoring_benchmark", "run the scoring parity probe"),
    "e2-scoring-policy-plan": ("e2_architecture.scoring_policy", "preregister the scoring tournament"),
    "e2-scoring-policy-fixture": (
        "e2_architecture.scoring_policy_fixture",
        "compile the scoring policy fixture",
    ),
    "e2-scoring-policy-tournament": (
        "e2_architecture.scoring_policy_tournament",
        "run the scoring policy tournament",
    ),
    "e3-plan": ("e3_data_objective.plan", "emit the cognition-mixture experiment plan"),
    "transaction": ("v5_training.transaction_canary", "run the checkpoint-transaction canary"),
    "p35-canary": ("v5_training.torch_canary", "run the bounded P35 update/resume canary"),
    "durability": ("v5_training.durability_canary", "run the local durability canary"),
    "target-preflight": ("v5_training.target_preflight", "probe the target TPU/XLA stack"),
}


def _usage() -> str:
    lines = ["usage: anra-v5 <command> [args...]", "", "commands:"]
    for name in sorted(SUBCOMMANDS):
        lines.append(f"  {name:<28} {SUBCOMMANDS[name][1]}")
    lines.append("")
    lines.append("Each command emits one JSON receipt path to stdout; diagnostics go to stderr.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0 if args else 2
    name, rest = args[0], args[1:]
    if name not in SUBCOMMANDS:
        print(f"unknown command: {name}\n\n{_usage()}", file=sys.stderr)
        return 2
    module_name = SUBCOMMANDS[name][0]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        print(f"command module unavailable: {module_name}: {exc}", file=sys.stderr)
        return 1
    entry = getattr(module, "main", None)
    if not callable(entry):
        print(f"command module has no main: {module_name}", file=sys.stderr)
        return 1
    sys.argv = [f"anra-v5 {name}", *rest]
    return int(entry())


if __name__ == "__main__":
    raise SystemExit(main())
