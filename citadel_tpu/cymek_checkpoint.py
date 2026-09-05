"""Thin adapter over Cymek production checkpoint contracts (no duplication).

Uses `v5_training.state` (TrainingState/CursorState/IdentityBindings) and
`v5_training.checkpoint.CheckpointStore` DIRECTLY from the pinned read-only
runtime — Citadel never reimplements the transaction, fencing, or verification
logic. Citadel only assembles inputs: model/optimizer tensors (opaque bytes),
cursor position, ledger counts, RNG bytes, schedule value, identity SHAs.
All Cymek imports are lazy (inside functions); this module imports cleanly
without torch. Deterministic PRECHECK_* errors on any contract violation.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping


def _cymek():
    try:
        from v5_training import state as v5_state
        from v5_training import checkpoint as v5_ckpt
    except Exception as exc:
        raise RuntimeError(
            "PRECHECK_IMPORT_FAILURE: pinned Cymek runtime does not expose "
            f"v5_training.state/checkpoint: {exc!r}")
    return v5_state, v5_ckpt


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def tokenizer_identity_sha256() -> str:
    """Honest canary-codec identity (NOT production BPE — labeled as such)."""
    from citadel_tpu import calculator_eval as cev

    return _sha256_hex(f"citadel-char-codec/1.0:{sorted(cev.DECODABLE_IDS.items())!r}".encode())


def spec_json_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_hex(_canonical(dict(value)))


def build_identities(*, model_spec_sha256: str, data_manifest_sha256: str,
                     pack_manifest_sha256: str, run_spec: Mapping[str, Any],
                     optimizer_spec: Mapping[str, Any],
                     schedule_spec: Mapping[str, Any],
                     curriculum_spec: Mapping[str, Any],
                     source_commit: str | None):
    """Build real Cymek IdentityBindings (assert_valid enforced by Cymek)."""
    v5_state, _ = _cymek()
    return v5_state.IdentityBindings(
        schema=v5_state.IDENTITY_SCHEMA,
        source_commit=source_commit or "0" * 40,
        model_spec_sha256=model_spec_sha256,
        tokenizer_sha256=tokenizer_identity_sha256(),
        data_manifest_sha256=data_manifest_sha256,
        pack_manifest_sha256=pack_manifest_sha256,
        run_spec_sha256=spec_json_sha256(run_spec),
        optimizer_spec_sha256=spec_json_sha256(optimizer_spec),
        schedule_spec_sha256=spec_json_sha256(schedule_spec),
        curriculum_spec_sha256=spec_json_sha256(curriculum_spec),
    )


def initial_state(*, lineage_id: str, token_budget: int, tokens_per_update: int,
                  pack_manifest_sha256: str, identities,
                  rng_state_sha256: str, curriculum_phase: str = "smoke"):
    """Genesis TrainingState (generation 0; Cymek enforces zero-counters)."""
    v5_state, _ = _cymek()
    cursor = v5_state.CursorState(
        schema=v5_state.CURSOR_SCHEMA,
        pack_manifest_sha256=pack_manifest_sha256,
        shard_ordinal=0, sequence_ordinal=0, token_offset=0)
    return v5_state.TrainingState(
        schema=v5_state.STATE_SCHEMA,
        lineage_id=lineage_id,
        generation=0, global_update=0,
        cumulative_tokens=0, token_budget=token_budget,
        tokens_per_update=tokens_per_update,
        tokens_by_source={},
        optimizer_step_max=0, schedule_tokens=0,
        cursor=cursor, rng_state_sha256=rng_state_sha256,
        curriculum_phase=curriculum_phase,
        identities=identities, parent_checkpoint_sha256=None)


def cursor_for_update(pack_manifest_sha256: str, *, sequence_ordinal: int,
                      token_offset: int):
    """Cursor advanced for exactly one completed update."""
    v5_state, _ = _cymek()
    return v5_state.CursorState(
        schema=v5_state.CURSOR_SCHEMA,
        pack_manifest_sha256=pack_manifest_sha256,
        shard_ordinal=0, sequence_ordinal=sequence_ordinal,
        token_offset=token_offset)


def _torch_bytes(state_dict) -> bytes:
    import torch

    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def open_store(root: str | Path, lineage_id: str):
    """Open (creating) a production CheckpointStore under root/lineage."""
    _, v5_ckpt = _cymek()
    return v5_ckpt.CheckpointStore(Path(root), lineage_id)


def advance_state(prev_state, *, cursor, ledger_delta: Mapping[str, int],
                  rng_bytes: bytes, store=None):
    """Pure state advance (no torch): parent = store LATEST (or None if given).

    When `store` is provided, the new state's parent is read from it (the
    checkpoint being superseded); otherwise the caller sets parentage via
    publish_state's expected_parent. Returns the new state.
    """
    current = store.latest_sha256() if store is not None else None
    return prev_state.advance(
        tokens_by_source=dict(ledger_delta),
        cursor=cursor,
        rng_state_sha256=_sha256_hex(bytes(rng_bytes)),
        parent_checkpoint_sha256=current,
    )


def live_payloads(*, model, optimizer, learning_rate: float, rng_bytes: bytes,
                  cursor, ledger: Mapping[str, int]) -> dict[str, bytes]:
    """Assemble the 7 production components from live objects (needs torch)."""
    import torch

    return {
        "model.bin": _torch_bytes(
            {k: v.detach().to("cpu") for k, v in model.state_dict().items()}),
        "optimizer.bin": _torch_bytes(dict(optimizer.state_dict())),
        "scheduler.json": _canonical({"learning_rate": float(learning_rate),
                                      "schedule": "constant-canary"}),
        "rng.bin": bytes(rng_bytes),
        "cursor.json": _canonical({"schema": cursor.schema,
                                   "pack_manifest_sha256": cursor.pack_manifest_sha256,
                                   "shard_ordinal": cursor.shard_ordinal,
                                   "sequence_ordinal": cursor.sequence_ordinal,
                                   "token_offset": cursor.token_offset}),
        "ledger.json": _canonical(dict(ledger)),
    }


def publish_state(store, *, state, payloads: Mapping[str, bytes],
                  expected_parent_sha256: str | None):
    """Publish caller-built payloads (no torch): inventory/fence by Cymek."""
    full = dict(payloads)
    full["training_state.json"] = _canonical(state.canonical())
    return store.publish(state=state, payloads=full,
                         expected_parent_sha256=expected_parent_sha256)


def publish_update(store, *, prev_state, model, optimizer, learning_rate: float,
                   rng_bytes: bytes, cursor, ledger_delta: Mapping[str, int]):
    """Advance exactly once and publish (genesis uses publish_genesis).

    The new state's parent is the store's current LATEST (the checkpoint being
    superseded); the writer fence rejects anything else. Every other invariant
    (exact token count, cursor advance, same pack, inventory) is enforced by
    Cymek itself; any violation raises before anything is written.
    Returns (new_state, checkpoint_sha256).
    """
    current = store.latest_sha256()
    new_state = advance_state(prev_state, cursor=cursor, ledger_delta=ledger_delta,
                              rng_bytes=rng_bytes, store=store)
    payloads = live_payloads(
        model=model, optimizer=optimizer, learning_rate=learning_rate,
        rng_bytes=rng_bytes, cursor=cursor,
        ledger=new_state.tokens_by_source)
    sha = publish_state(store, state=new_state, payloads=payloads,
                        expected_parent_sha256=current)
    return new_state, sha


def publish_genesis(store, *, state, model, optimizer, learning_rate: float,
                    rng_bytes: bytes):
    """Publish the generation-0 state (no parent)."""
    payloads = live_payloads(
        model=model, optimizer=optimizer, learning_rate=learning_rate,
        rng_bytes=rng_bytes, cursor=state.cursor,
        ledger=dict(state.tokens_by_source))
    return publish_state(store, state=state, payloads=payloads,
                         expected_parent_sha256=None)


def restore_latest(store):
    """Restore + fully verify latest (Cymek does the verification)."""
    return store.restore()


def load_model_bytes_into(model, payloads: Mapping[str, bytes]) -> None:
    """Strict model load from production payload bytes."""
    import io as _io

    import torch

    state = torch.load(_io.BytesIO(bytes(payloads["model.bin"])),
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)


def load_optimizer_bytes_into(optimizer, payloads: Mapping[str, bytes]) -> None:
    """Optimizer load from production payload bytes."""
    import io as _io

    import torch

    state = torch.load(_io.BytesIO(bytes(payloads["optimizer.bin"])),
                       map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state)


__all__ = [
    "advance_state",
    "build_identities",
    "cursor_for_update",
    "initial_state",
    "live_payloads",
    "load_model_bytes_into",
    "load_optimizer_bytes_into",
    "open_store",
    "publish_genesis",
    "publish_state",
    "publish_update",
    "restore_latest",
    "spec_json_sha256",
    "tokenizer_identity_sha256",
]
