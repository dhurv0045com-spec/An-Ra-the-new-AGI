"""Bounded end-to-end canary for the V5 checkpoint transaction contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from .checkpoint import CheckpointStore, InjectedCrash, _canonical_json
from .state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState


def _hash(character: str) -> str:
    return character * 64


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in ("state.py", "checkpoint.py", "transaction_canary.py"):
        normalized = (root / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        digest.update(name.encode("utf-8") + b"\0" + normalized.encode("utf-8") + b"\0")
    return digest.hexdigest()


def _cursor(pack_hash: str, update: int) -> CursorState:
    return CursorState(CURSOR_SCHEMA, pack_hash, update // 2, update, update % 3)


def _payloads(state: TrainingState) -> dict[str, bytes]:
    generation = state.generation
    return {
        "model.bin": f"model-generation-{generation}".encode(),
        "optimizer.bin": f"adam-generation-{generation}-step-{state.optimizer_step_max}".encode(),
        "scheduler.json": _canonical_json({"schedule_tokens": state.schedule_tokens}),
        "rng.bin": state.rng_state_sha256.encode(),
        "cursor.json": _canonical_json(asdict(state.cursor)),
        "ledger.json": _canonical_json(dict(state.tokens_by_source)),
        "training_state.json": _canonical_json(state.canonical()),
    }


def _initial() -> TrainingState:
    identities = IdentityBindings(
        IDENTITY_SCHEMA,
        "a" * 40,
        _hash("1"), _hash("2"), _hash("3"), _hash("4"),
        _hash("5"), _hash("6"), _hash("7"), _hash("8"),
    )
    return TrainingState.initial(
        lineage_id="canary",
        token_budget=10,
        tokens_per_update=4,
        cursor=_cursor(identities.pack_manifest_sha256, 0),
        rng_state_sha256=_hash("9"),
        curriculum_phase="uniform",
        identities=identities,
    )


def _advance(state: TrainingState, parent: str | None) -> TrainingState:
    remaining = state.token_budget - state.cumulative_tokens
    count = min(state.tokens_per_update, remaining)
    natural = count - (1 if count > 1 else 0)
    cognition = count - natural
    return state.advance(
        tokens_by_source={"natural": natural, "verified_cognition": cognition},
        cursor=_cursor(state.identities.pack_manifest_sha256, state.global_update + 1),
        rng_state_sha256=f"{state.global_update + 1:064x}",
        parent_checkpoint_sha256=parent,
    )


def run_canary() -> dict[str, object]:
    checks: dict[str, bool] = {}
    initial = _initial()
    uninterrupted = initial
    for _ in range(3):
        uninterrupted = _advance(uninterrupted, uninterrupted.parent_checkpoint_sha256)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = CheckpointStore(root / "primary", "canary")
        first = _advance(initial, None)
        first_sha = store.publish(state=first, payloads=_payloads(first), expected_parent_sha256=None)

        clean_root = root / "clean-copy"
        shutil.copytree(store.lineage_root, clean_root / "canary")
        clean_store = CheckpointStore(clean_root, "canary")
        restored, _ = clean_store.restore(first_sha)
        checks["clean_copy_restore"] = restored == first

        resumed = restored
        parent = first_sha
        for _ in range(2):
            resumed = _advance(resumed, parent)
            parent = clean_store.publish(
                state=resumed, payloads=_payloads(resumed), expected_parent_sha256=parent
            )
        checks["uninterrupted_equals_resume"] = (
            resumed.global_update == uninterrupted.global_update
            and resumed.cumulative_tokens == uninterrupted.cumulative_tokens
            and dict(resumed.tokens_by_source) == dict(uninterrupted.tokens_by_source)
            and resumed.cursor == uninterrupted.cursor
        )
        checks["nondivisible_budget_exact"] = (
            resumed.cumulative_tokens == 10 and resumed.complete and resumed.global_update == 3
        )
        checks["schedule_and_optimizer_exact"] = (
            resumed.schedule_tokens == 10 and resumed.optimizer_step_max == 3
        )

        stale_rejected = False
        try:
            clean_store.publish(
                state=resumed, payloads=_payloads(resumed),
                expected_parent_sha256=first_sha,
            )
        except ValueError:
            stale_rejected = True
        checks["stale_writer_rejected"] = stale_rejected

        corruption_root = root / "corrupt-copy"
        shutil.copytree(clean_store.lineage_root, corruption_root / "canary")
        corrupt_store = CheckpointStore(corruption_root, "canary")
        corrupt_path = corrupt_store.objects / parent / "model.bin"
        corrupt_path.write_bytes(b"corrupt")
        try:
            corrupt_store.restore(parent)
        except ValueError:
            checks["corruption_rejected"] = True
        else:
            checks["corruption_rejected"] = False

        missing_root = root / "missing-copy"
        shutil.copytree(clean_store.lineage_root, missing_root / "canary")
        missing_store = CheckpointStore(missing_root, "canary")
        (missing_store.objects / parent / "rng.bin").unlink()
        try:
            missing_store.restore(parent)
        except ValueError:
            checks["missing_component_rejected"] = True
        else:
            checks["missing_component_rejected"] = False

        for crash_stage, expected_pointer in (
            ("after_stage", None),
            ("after_publish_before_pointer", None),
            ("after_pointer", "committed"),
        ):
            crash_store = CheckpointStore(root / f"crash-{crash_stage}", "canary")
            crash_state = _advance(initial, None)
            try:
                crash_store.publish(
                    state=crash_state,
                    payloads=_payloads(crash_state),
                    expected_parent_sha256=None,
                    inject_crash_at=crash_stage,
                )
            except InjectedCrash:
                pass
            pointer = crash_store.latest_sha256()
            checks[f"crash_{crash_stage}_safe"] = (
                pointer is None if expected_pointer is None else crash_store.restore(pointer)[0] == crash_state
            )

    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema": "anra-v5-training-transaction-canary/v1",
        "implementation_sha256": implementation_sha256(),
        "status": status,
        "scope": "local framework-neutral transaction semantics; no real model, distributed, TPU, or remote durability claim",
        "run": {"token_budget": 10, "tokens_per_update": 4, "update_sizes": [4, 4, 2]},
        "final_state": resumed.canonical(),
        "checks": checks,
        "limitations": [
            "Payload bytes stand in for already-separately-tested model and optimizer tensors.",
            "Remote upload, redownload, object-store CAS, distributed rank RNG, and TPU restore remain open.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/v5/training_transaction_canary.json"))
    args = parser.parse_args()
    receipt = run_canary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
