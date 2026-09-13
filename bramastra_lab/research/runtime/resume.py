"""Run lifecycle and fresh-process resume (B05).

A run directory carries the run manifest, append-only events and checkpoints.
``checkpoint_run`` publishes one atomic checkpoint holding model, optimizer,
schedule, RNG streams, cursors and controller state. ``restore_run`` rebuilds
the trainer in whatever process calls it — same-process serialization alone
is not a resume receipt.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from bramastra_lab.research.config import BuildConfig, tokenizer_identity
from bramastra_lab.research.experience.codec import codec_identity
from bramastra_lab.research.learning.trainer import Trainer
from bramastra_lab.research.models import IntegratedModel
from bramastra_lab.research.runtime import checkpoint as ckpt

RUN_SCHEMA = "bramastra-run/v1"


class RunError(RuntimeError):
    """A run directory violates the run contract."""


@dataclass
class RunContext:
    run_dir: str
    run_id: str
    config: BuildConfig
    data_identity: str
    writer_token: str
    trainer: Trainer
    sampler_state: Mapping[str, Any] | None
    controller_state: Mapping[str, Any] | None
    replay_cursor: Mapping[str, Any] | None
    expected_parent: str | None = None
    source_identity: Mapping[str, str] | None = None


def create_run(run_dir: str, config: BuildConfig, *, data_identity: str,
               run_id: str | None = None) -> str:
    """Initialize a fresh run directory. Refuses to touch an existing run."""
    if os.path.exists(run_dir):
        raise RunError(f"run directory {run_dir} already exists; "
                       "use a new run id instead of overwriting")
    run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
    os.makedirs(run_dir)
    manifest = {
        "schema": RUN_SCHEMA, "run_id": run_id,
        "config_identity": config.identity(),
        "tokenizer_identity": tokenizer_identity(),
        "codec_identity": codec_identity(),
        "data_identity": data_identity,
        "created_unix": time.time(),
    }
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    events_path = os.path.join(run_dir, "events.jsonl")
    with open(events_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "run_created", "run_id": run_id,
                                 "at_unix": time.time()}) + "\n")
    return run_id


def append_event(run_dir: str, event: Mapping[str, Any]) -> None:
    with open(os.path.join(run_dir, "events.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at_unix": time.time(), **event}, sort_keys=True) + "\n")


def read_run_manifest(run_dir: str) -> dict[str, Any]:
    path = os.path.join(run_dir, "run.json")
    if not os.path.exists(path):
        raise RunError(f"{run_dir} is not a run directory (no run.json)")
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != RUN_SCHEMA:
        raise RunError(f"unsupported run schema {manifest.get('schema')!r}")
    return manifest


def checkpoint_run(trainer: Trainer, run_dir: str, ctx: RunContext, *,
                   milestone: str | None = None) -> ckpt.CheckpointManifest:
    """Publish one checkpoint at an update boundary with every live stream
    captured in the same snapshot (B2.2 R3): sampler, controller and replay
    state are written into the payload by the caller BEFORE publication, and
    the publication verifies the writer lease and the expected parent under
    the serialized boundary (B2.2 R4).
    """
    payload = dict(trainer.state_payload())
    payload["rng"] = ckpt.capture_rng_state()
    payload["sampler_state"] = ctx.sampler_state
    payload["controller_state"] = ctx.controller_state
    payload["replay_cursor"] = ctx.replay_cursor
    run_manifest = read_run_manifest(run_dir)
    return ckpt.save_checkpoint(
        run_dir, payload,
        run_id=run_manifest["run_id"],
        update_index=trainer.counters.optimizer_updates,
        config_identity=run_manifest["config_identity"],
        tokenizer_identity=run_manifest["tokenizer_identity"],
        data_identity=run_manifest["data_identity"],
        parent_checkpoint_id=None,
        code_identity=ctx.source_identity.get("source_closure_sha256", "unavailable")
        if ctx.source_identity else "unavailable",
        milestone=milestone,
        writer_token=ctx.writer_token,
        expected_parent=ctx.expected_parent,
    )


def restore_run(run_dir: str, config: BuildConfig, *,
                allow_source_migration: bool = False,
                acquire_lease: bool = True) -> RunContext:
    """Restore the latest valid checkpoint in this (fresh) process.

    Validates run/config/data/source compatibility before anything loads.
    A changed source closure is an explicit, recorded migration — never a
    silent resume (B2.2 R1/R4).
    """
    from bramastra_lab.research.runtime.provenance import source_identity

    run_manifest = read_run_manifest(run_dir)
    if run_manifest["config_identity"] != config.identity():
        raise RunError("run config identity does not match the supplied config; "
                       "resume requires the identical configuration")
    payload, manifest = ckpt.load_checkpoint(
        run_dir,
        expect_config_identity=run_manifest["config_identity"],
        expect_tokenizer_identity=run_manifest["tokenizer_identity"],
        expect_data_identity=run_manifest["data_identity"],
    )
    current_source = source_identity()
    recorded_closure = manifest.code_identity
    if recorded_closure not in ("unavailable", current_source["source_closure_sha256"]):
        if not allow_source_migration:
            raise RunError(
                f"checkpoint source closure {str(recorded_closure)[:12]}... differs from "
                f"this process {current_source['source_closure_sha256'][:12]}...; a source "
                "migration must be explicit (--allow-source-migration) and is recorded")
        append_event(run_dir, {"event": "source_migration",
                               "from_closure": recorded_closure,
                               "to_closure": current_source["source_closure_sha256"]})
    model = IntegratedModel(config)
    trainer = Trainer(config, model)
    trainer.load_state_payload(payload)
    ckpt.restore_rng_state(payload["rng"])
    # The lease is acquired LATE (F4): identity, parent and source validation
    # happen first, so a failed setup cannot leak an active writer lease to
    # the next attempt. Callers that acquire later pass acquire_lease=False
    # and must fence-recheck at their publication boundary.
    token = ckpt.acquire_writer_fence(run_dir) if acquire_lease else ""
    return RunContext(
        run_dir=run_dir, run_id=run_manifest["run_id"], config=config,
        data_identity=run_manifest["data_identity"], writer_token=token,
        trainer=trainer,
        sampler_state=payload.get("sampler_state"),
        controller_state=payload.get("controller_state"),
        replay_cursor=payload.get("replay_cursor"),
        expected_parent=manifest.checkpoint_id,
        source_identity=current_source,
    )
