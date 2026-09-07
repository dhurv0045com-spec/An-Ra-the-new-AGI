"""500M campaign production entry point (B2) — the single real chain.

    documents -> data manifest -> frozen tokenizer -> packing
    -> sampler/cursor -> certified microbatches -> ProductionTrainingBackend
    -> trainer.train (fail-closed state machine) -> checkpoint transactions
    -> exact-resume verification -> campaign receipt

The same entry executes a fresh campaign or resumes one: pass
`resume_store_root`/`resume_run_id` to continue from the latest committed
generation. Device operations are injected (`device`, `torch_module`, `xb`)
so the full chain is CPU-dry-run-testable while TPU execution uses the
same code path. Fail-closed on: identity drift, ledger drift, nonfinite
loss/gradients, parameter non-mutation, stale writer, restore mismatch.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from v5_data.batch import microbatch
from v5_data.manifest import Document, build_data_manifest, manifest_sha256
from v5_data.pack import pack_documents, sampler_order
from v5_model.core import initialize
from v5_training.checkpoint import CheckpointStore
from v5_training.optimizer import build_adamw_optimizer
from v5_training.production_backend import (
    PackedBatch,
    ProductionTrainingBackend,
    bounded_warmup_schedule,
    capture_evidence,
    production_payloads,
    restore_production,
)
from v5_training.runner import RunController
from v5_training.trainer import train
from v5_training.state import (
    CURSOR_SCHEMA,
    IDENTITY_SCHEMA,
    CursorState,
    IdentityBindings,
    TrainingState,
)

ENTRY_SCHEMA = "anra-v5-production-entry-receipt/v1"
TOKENS_PER_UPDATE = 4_096
SEQUENCES_PER_UPDATE = 8
SEQUENCE_BUCKET = 512
SEQUENCES_PER_SHARD = SEQUENCES_PER_UPDATE
PEAK_LR = 3e-4


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def prepare_data(*, documents: list[dict[str, Any]], tokenizer: Any,
                 run_id: str, seed: int,
                 contamination_benchmarks: dict[str, str] | None = None
                 ) -> dict[str, Any]:
    """Documents -> data manifest -> packing -> sampler order (data half).
    Deterministic in (documents, seed): recomputing it on resume is exact."""
    records = [
        Document(doc_id=d["doc_id"], text=d["text"],
                 source_id=d["source_id"], domain=d.get("domain", "text"),
                 family=d.get("family", "natural"),
                 authorization_category=d.get(
                     "authorization_category", "first-party-authorized"),
                 acquired_date=d.get("acquired_date", "1970-01-01"))
        for d in documents]
    manifest, manifest_audit = build_data_manifest(
        records, manifest_id=f"{run_id}-data",
        tokenizer_sha256=tokenizer.identity.artifact_sha256,
        filter_version="production-entry-filter/v1",
        dedup_version="exact-clusters/v1",
        split_salt=f"{run_id}/v1",
        split_boundaries={"training": 1.0, "development": 0.0,
                          "sealed": 0.0, "fresh": 0.0},
        count_tokens=lambda text: len(tokenizer.encode(text)),
        contamination_benchmarks=contamination_benchmarks or {})
    keep = {rec.source_id for rec in manifest.sources
            if rec.split == "training"}
    packed, pack_audit = pack_documents(
        [(d["doc_id"], tokenizer.encode(d["text"]), d["source_id"])
         for d in documents if d["doc_id"] in keep],
        bos=2, eos=3, pad=0, sequences_per_shard=SEQUENCES_PER_SHARD)
    shard_hashes = [shard.sha256() for shard in packed]
    order = sampler_order(shard_hashes, run_seed=seed, epoch=0)
    pack_manifest_sha256 = hashlib.sha256(_canonical_json(
        [json.loads(shard.payload_bytes()) for shard in packed])).hexdigest()
    return {"manifest": manifest, "manifest_audit": manifest_audit,
            "manifest_sha256": manifest_sha256(manifest),
            "packed": packed, "shard_hashes": shard_hashes,
            "pack_manifest_sha256": pack_manifest_sha256,
            "sampler_order": order,
            "training_doc_ids": sorted(keep)}


def _walk_windows(packed, order: list[int], want: int
                  ) -> list[tuple[tuple, tuple, dict]]:
    """Consecutive exact 8x512 real-token windows in sampler order. Window k
    is the data for update k+1: the schedule is a pure function of the pack,
    so a resumed run at update k resumes at window k with zero replay."""
    windows: list[tuple[tuple, tuple, dict]] = []
    position, sequence_index = 0, 0
    total = len(order)
    while position < total and len(windows) < want:
        batch = None
        try:
            batch = microbatch(packed, order, shard_ordinal=position,
                               sequence_ordinal=sequence_index,
                               sequences=SEQUENCES_PER_UPDATE, pad=0)
        except ValueError:
            pass
        take = (batch is not None
                and batch.consumed_real_tokens == TOKENS_PER_UPDATE
                and all(len(row) == SEQUENCE_BUCKET for row in batch.tokens))
        if take:
            windows.append((batch.tokens, batch.segment_ids,
                            batch.tokens_by_source))
            position, sequence_index = (batch.shard_ordinal,
                                        batch.sequence_ordinal)
        else:
            sequence_index += 1
            if sequence_index >= SEQUENCES_PER_SHARD:
                position, sequence_index = position + 1, 0
    if len(windows) < want:
        raise ValueError(f"pack yields {len(windows)} exact windows; "
                         f"{want} required")
    return windows


def run_campaign(*, documents: list[dict[str, Any]], tokenizer: Any,
                 model_spec, run_id: str, seed: int, updates: int,
                 store_root: str | Path, device: Any, torch_module: Any = None,
                 xb: Any | None = None, progress: Callable[[str], None]
                 | None = None, resume_store_root: str | Path | None = None,
                 resume_run_id: str | None = None) -> dict[str, Any]:
    """Execute (or resume) a certified production campaign of `updates`
    total updates. Resumes from `resume_store_root`/`resume_run_id` latest
    committed generation when given; otherwise fresh."""
    if xb is None:
        raise ValueError("xb (device seam) is required")
    torch = torch_module
    if torch is None:
        import torch as torch

    t0 = time.time()
    data = prepare_data(documents=documents, tokenizer=tokenizer,
                        run_id=run_id, seed=seed)
    packed, order = data["packed"], data["sampler_order"]
    identities = IdentityBindings(
        schema=IDENTITY_SCHEMA, source_commit=hashlib.sha256(_canonical_json({'campaign': run_id})).hexdigest()[:40],
        model_spec_sha256=model_spec.sha256(),
        tokenizer_sha256=tokenizer.identity.artifact_sha256,
        data_manifest_sha256=data["manifest_sha256"],
        pack_manifest_sha256=data["pack_manifest_sha256"],
        run_spec_sha256=hashlib.sha256(_canonical_json(
            {"campaign": run_id, "updates": updates})).hexdigest(),
        optimizer_spec_sha256=hashlib.sha256(_canonical_json(
            {"optimizer": "AdamW", "beta1": 0.9, "beta2": 0.95,
             "epsilon": 1e-8, "weight_decay": 0.1})).hexdigest(),
        schedule_spec_sha256=hashlib.sha256(_canonical_json(
            {"kind": "bounded_warmup", "peak_lr": PEAK_LR})).hexdigest(),
        curriculum_spec_sha256=hashlib.sha256(
            b"production-entry").hexdigest())

    resume_store = None
    if resume_store_root is not None and resume_run_id is not None:
        resume_store = CheckpointStore(Path(resume_store_root), run_id)
        state, payloads = resume_store.restore()
        done = int(state.global_update)
        remaining = updates - done
        if remaining <= 0:
            return {"schema": ENTRY_SCHEMA, "run_id": run_id, "seed": seed,
                    "updates_executed": done,
                    "cumulative_tokens": int(state.cumulative_tokens),
                    "losses": [], "resumed": True, "already_complete": True,
                    "data_manifest_sha256": data["manifest_sha256"],
                    "pack_manifest_sha256": data["pack_manifest_sha256"],
                    "model_spec_sha256": model_spec.sha256(),
                    "wall_seconds": 0.0}
    else:
        state = TrainingState.initial(
            lineage_id=run_id, token_budget=updates * TOKENS_PER_UPDATE,
            tokens_per_update=TOKENS_PER_UPDATE,
            cursor=CursorState(CURSOR_SCHEMA, data["pack_manifest_sha256"],
                               0, 0, 0),
            rng_state_sha256="0" * 64, curriculum_phase="500m-campaign",
            identities=identities)
        done, remaining = 0, updates

    torch.manual_seed(seed)
    model = initialize(model_spec, seed).to(device)
    optimizer = build_adamw_optimizer(model, torch_module=torch)
    backend = ProductionTrainingBackend(
        model=model, optimizer=optimizer, bos_id=2, pad_id=0, device=device,
        schedule=bounded_warmup_schedule(peak_learning_rate=PEAK_LR),
        bfloat16_autocast=getattr(device, "type", "cpu") == "cuda")
    if resume_store is not None:
        restore_production(backend, payloads=payloads)

    windows = _walk_windows(packed, order, updates)
    losses: list[float] = []

    def backend_step(current: TrainingState):
        window_tokens, window_segments, window_sources = \
            windows[current.global_update]
        tokens = torch.tensor([list(row) for row in window_tokens],
                              dtype=torch.long, device=device)
        seg_ids = torch.tensor([list(row) for row in window_segments],
                               dtype=torch.int32, device=device)
        pbatch = PackedBatch(
            tokens=tokens, segment_ids=seg_ids,
            tokens_by_source=dict(window_sources),
            cursor=CursorState(
                CURSOR_SCHEMA, data["pack_manifest_sha256"], 0,
                current.global_update,
                TOKENS_PER_UPDATE * (current.global_update + 1)),
            rng_state_sha256=hashlib.sha256(
                f"prod-rng-{current.global_update}".encode()).hexdigest())
        report = backend.step(current, pbatch)
        loss = float(backend.last_receipt["loss"])
        losses.append(loss)
        if progress and current.global_update % 10 == 9:
            progress(f"update {current.global_update + 1}/{updates} "
                     f"loss {loss:.4f}")
        return report

    store = CheckpointStore(Path(store_root), run_id)
    controller = RunController(target_update=updates)
    controller.start()
    final = train(
        state=state, controller=controller, store=store,
        payload_builder=lambda s: production_payloads(backend, state=s),
        backend_step=backend_step, updates=remaining,
        checkpoint_every=max(1, remaining))
    wall = time.time() - t0

    # exact-resume verification: live state == fresh restore
    restored_state, restored_payloads = store.restore()
    fresh_model = initialize(model_spec, seed).to(device)
    fresh_optimizer = build_adamw_optimizer(fresh_model, torch_module=torch)
    fresh_backend = ProductionTrainingBackend(
        model=fresh_model, optimizer=fresh_optimizer, bos_id=2, pad_id=0,
        device=device, schedule=bounded_warmup_schedule(
            peak_learning_rate=PEAK_LR))
    restore_production(fresh_backend, payloads=restored_payloads)
    live = capture_evidence(backend.model, backend.optimizer, torch=torch)
    resumed = capture_evidence(fresh_model, fresh_optimizer, torch=torch)
    resume_equal = (live.parameter_sha256 == resumed.parameter_sha256
                    and live.moment_sha256 == resumed.moment_sha256
                    and live.optimizer_steps == resumed.optimizer_steps)
    if not resume_equal:
        raise RuntimeError("campaign restore did not reproduce live state")

    return {
        "schema": ENTRY_SCHEMA, "run_id": run_id, "seed": seed,
        "updates_executed": int(final.global_update),
        "cumulative_tokens": int(final.cumulative_tokens),
        "state_complete": bool(final.complete),
        "losses": losses, "wall_seconds": round(wall, 3),
        "resume_equal": resume_equal,
        "data_manifest_sha256": data["manifest_sha256"],
        "pack_manifest_sha256": data["pack_manifest_sha256"],
        "model_spec_sha256": model_spec.sha256(),
        "checkpoint_head": store.latest_sha256(),
    }


def run_500m_session(*, documents: list[dict[str, Any]], tokenizer: Any,
                     model_spec, run_id: str, seed: int,
                     campaign_tokens: int, session_dir: str | Path,
                     device: Any, torch_module: Any = None,
                     xb: Any | None = None,
                     max_session_minutes: float = 90.0,
                     recovery_interval_updates: int = 25,
                     progress: Callable[[str], None] | None = None,
                     ) -> dict[str, Any]:
    """One 500M-campaign training session with milestone detection,
    recovery checkpointing, heartbeat, and session receipt.

    Multi-session: call again on a fresh runtime — resumes from the latest
    committed generation in `session_dir/campaign_store`. Milestone
    checkpoints are immutable and separate from recovery rotation.
    """
    root = Path(session_dir)
    root.mkdir(parents=True, exist_ok=True)
    store_root = str(root / "campaign_store")
    milestone_dir = root / "milestones"
    milestone_dir.mkdir(parents=True, exist_ok=True)

    def hb(**kw):
        doc = {"schema": "anra-v5-heartbeat/v1", "utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}
        (root / "HEARTBEAT.json").write_text(
            json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")

    hb(phase="starting", campaign_tokens=campaign_tokens)

    campaign_spec_sha = hashlib.sha256(_canonical_json(
        {"campaign_id": run_id, "target": campaign_tokens})).hexdigest()

    result = run_campaign(
        documents=documents, tokenizer=tokenizer, model_spec=model_spec,
        run_id=run_id, seed=seed, updates=campaign_tokens // TOKENS_PER_UPDATE,
        store_root=store_root, device=device, torch_module=torch_module,
        xb=xb, progress=progress, resume_store_root=store_root,
        resume_run_id=run_id)

    crossed = []
    for m in MILESTONE_TOKENS:
        if result["cumulative_tokens"] >= m:
            crossed.append(m)
            mpath = milestone_dir / f"milestone_{m}.json"
            if not mpath.is_file():
                mpath.write_text(json.dumps({
                    "schema": "anra-v5-milestone/v1",
                    "tokens": m, "utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "checkpoint": result["checkpoint_head"],
                    "run_id": run_id,
                }, indent=2, sort_keys=True), encoding="utf-8")

    session_receipt = {
        "schema": "anra-v5-session-receipt/v1",
        "run_id": run_id, "campaign_tokens": campaign_tokens,
        "updates_executed": result["updates_executed"],
        "cumulative_tokens": result["cumulative_tokens"],
        "milestones_crossed": crossed,
        "losses": result["losses"][-20:] if result["losses"] else [],
        "resume_equal": result["resume_equal"],
        "wall_seconds": result["wall_seconds"],
        "checkpoint_head": result["checkpoint_head"],
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (root / "SESSION_RECEIPT.json").write_text(
        json.dumps(session_receipt, indent=2, sort_keys=True),
        encoding="utf-8")
    hb(phase="complete", tokens=result["cumulative_tokens"],
       milestones=crossed)

    return {"schema": "anra-v5-500m-session/v1",
            "session_dir": str(root),
            "result": result, "milestones_crossed": crossed,
            "session_receipt": session_receipt,
            "campaign_store": store_root,
            "milestone_dir": str(milestone_dir)}


MILESTONE_TOKENS = (50_000_000, 100_000_000, 200_000_000, 350_000_000,
                    500_000_000)

__all__ = ["ENTRY_SCHEMA", "MILESTONE_TOKENS", "prepare_data", "run_campaign",
           "run_500m_session"]
__all__ = ["ENTRY_SCHEMA", "prepare_data", "run_campaign"]
