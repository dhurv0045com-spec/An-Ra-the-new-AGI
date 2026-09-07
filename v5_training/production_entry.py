"""500M campaign production entry point — the single real chain.

    documents -> data manifest -> frozen tokenizer -> packing
    -> sampler order -> campaign layout -> streaming exact-token microsteps
    -> gradient accumulation to one logical optimizer update
    -> trainer state machine -> checkpoint transactions
    -> exact-resume verification -> campaign receipt

A logical optimizer update is N accumulation microsteps (4 x 32,768 global
real tokens = 131,072 for the frozen topology), never one giant batch.
TrainingState advances exactly once per logical update. The same entry
executes a fresh campaign or resumes one: resume is inferred from a valid
committed LATEST in the campaign store, never from merely supplied paths.
Device operations are injected (device, torch_module, xb) so the full chain
is CPU-dry-run-testable while TPU execution uses the same code path.
Fail-closed on: identity drift, ledger drift, nonfinite loss/gradients,
parameter non-mutation, stale writer, restore mismatch, missing production
contamination commitment.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from v5_contracts.training_spec import build_training_spec
from v5_data.batch import build_flat_index, exact_window
from v5_data.layout import campaign_layout
from v5_data.manifest import Document, build_data_manifest, manifest_sha256
from v5_data.pack import pack_documents, sampler_order
from v5_model.core import initialize
from v5_training.checkpoint import CheckpointStore
from v5_training.optimizer import build_adamw_optimizer
from v5_training.production_backend import (
    ProductionTrainingBackend,
    capture_evidence,
    precision_receipt,
    production_payloads,
    restore_production,
)
from v5_training.runner import RunController
from v5_training.schedule import lr_at, schedule_receipt
from v5_training.state import (
    CURSOR_SCHEMA,
    IDENTITY_SCHEMA,
    CursorState,
    IdentityBindings,
    TrainingState,
    next_update_tokens,
)
from v5_training.trainer import train

ENTRY_SCHEMA = "anra-v5-production-entry-receipt/v1"
# 500M-campaign milestone thresholds (prompt section 16). Distinct from the
# 5B-run thresholds in training_spec; these gate the 500M campaign only.
MILESTONE_TOKENS = (50_000_000, 100_000_000, 200_000_000, 350_000_000, 500_000_000)


def frozen_topology() -> dict[str, Any]:
    """Read the frozen campaign topology mechanically; no duplicate constants."""

    spec = build_training_spec()
    topo = spec["target_topology"]
    opt = spec["optimization"]
    pack = spec["packing"]
    check = spec["checkpointing"]
    topology = {
        "replicas": int(topo["replicas"]),
        "gradient_accumulation_microsteps": int(topo["gradient_accumulation_microsteps"]),
        "tokens_per_replica_microstep": int(topo["real_tokens_per_replica_microstep"]),
        "global_tokens_per_microstep": int(topo["global_real_tokens_per_microstep"]),
        "global_tokens_per_update": int(opt["global_tokens_per_update"]),
        "gradient_clip": float(opt["gradient_clip_global_l2"]),
        "beta1": float(opt["beta1"]),
        "beta2": float(opt["beta2"]),
        "epsilon": float(opt["epsilon"]),
        "weight_decay": float(opt["weight_decay"]),
        "sequences_per_replica_by_bucket": {
            int(bucket): int(count)
            for bucket, count in topo["sequences_per_replica_by_bucket"].items()
        },
        "supercycle": [int(bucket) for bucket in pack["twenty_microstep_supercycle"]],
        "recovery_threshold_tokens": int(check["recovery_threshold_tokens"]),
        "recovery_generations_retained": int(check["recovery_generations_retained"]),
    }
    if (topology["replicas"] * topology["tokens_per_replica_microstep"]
            != topology["global_tokens_per_microstep"]):
        raise ValueError("frozen topology replicas do not multiply to the global microstep")
    if (topology["gradient_accumulation_microsteps"] * topology["global_tokens_per_microstep"]
            != topology["global_tokens_per_update"]):
        raise ValueError("frozen topology microsteps do not multiply to the global update")
    return topology


def topology_sha256(topology: Mapping[str, Any]) -> str:
    """Content identity of the frozen topology actually executed."""

    return hashlib.sha256(
        json.dumps(topology, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def microstep_buckets(
    *, cumulative_tokens: int, microstep_counts: list[int], topo: Mapping[str, Any]
) -> list[int]:
    """Bucket per microstep from the frozen supercycle position.

    Position derives from cumulative real tokens only, so resume reproduces
    it without extra state: microstep ordinal = cumulative // global
    microstep size.
    """

    cycle = topo["supercycle"]
    base = cumulative_tokens // topo["global_tokens_per_microstep"]
    return [int(cycle[(base + index) % len(cycle)]) for index in range(len(microstep_counts))]


def partial_microstep_plan(
    *, remaining_tokens: int, topo: Mapping[str, Any], microstep_ordinal: int
) -> list[int]:
    """Split an update budget into full microsteps plus one exact tail.

    Derived mechanically from the frozen global microstep size: fill
    full-size chunks, then one partial microstep with the remainder, so the
    plan sums to exactly ``remaining_tokens``. ``microstep_ordinal`` binds
    the plan to its supercycle position for audit; the split itself is
    position-independent. No overshoot, no undershoot.
    """

    if microstep_ordinal < 0:
        raise ValueError("microstep ordinal cannot be negative")
    microstep = topo["global_tokens_per_microstep"]
    if remaining_tokens <= 0:
        raise ValueError("remaining tokens must be positive")
    full, tail = divmod(remaining_tokens, microstep)
    return [microstep] * full + ([tail] if tail else [])


def crossed_milestones(previous_tokens: int, new_tokens: int,
                       milestones: tuple = MILESTONE_TOKENS) -> list[int]:
    """Milestones strictly crossed when tokens move prev -> new. Pure."""

    if new_tokens < previous_tokens:
        raise ValueError(f"backward: {new_tokens} < {previous_tokens}")
    return sorted(m for m in milestones if previous_tokens < m <= new_tokens)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def resolve_cymek_sha(explicit: str | None) -> str:
    """Require the real executable SHA: explicit param or mechanical git resolve.

    Never a stale default: an unresolvable tree fails closed here instead of
    stamping a wrong identity on a campaign.
    """

    if explicit is not None:
        if len(explicit) != 40 or any(c not in "0123456789abcdef" for c in explicit):
            raise ValueError("cymek_sha must be a full lowercase git SHA-1")
        return explicit
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        raise ValueError("cannot resolve cymek source SHA outside a git worktree")
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    sha = completed.stdout.strip()
    if completed.returncode != 0 or len(sha) != 40:
        raise ValueError("cannot resolve cymek source SHA from git")
    return sha


def prepare_data(*, documents: list[dict[str, Any]], tokenizer: Any,
                 run_id: str, seed: int,
                 contamination_benchmarks: dict[str, str] | None = None
                 ) -> dict[str, Any]:
    """Documents -> data manifest -> packing -> sampler order (data half).

    Deterministic in (documents, seed): resume recomputes it identically and
    the checkpoint restore compares every content hash, so any drift in
    documents, tokenizer, or seed fails closed at resume.
    """

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
    packed_docs = [(d["doc_id"], tokenizer.encode(d["text"]), d["source_id"])
                   for d in documents if d["source_id"] in keep]
    packed, pack_audit = pack_documents(
        packed_docs, bos=2, eos=3, pad=0, sequences_per_shard=8)
    shard_hashes = [shard.sha256() for shard in packed]
    order = sampler_order(shard_hashes, run_seed=seed, epoch=0)
    pack_manifest_sha256 = hashlib.sha256(_canonical_json(
        [json.loads(shard.payload_bytes()) for shard in packed])).hexdigest()
    return {"manifest": manifest, "manifest_audit": manifest_audit,
            "manifest_sha256": manifest_sha256(manifest),
            "packed": packed, "pack_audit": pack_audit,
            "shard_hashes": shard_hashes,
            "pack_manifest_sha256": pack_manifest_sha256,
            "sampler_order": order,
            "packed_doc_ids": sorted(d["doc_id"] for d in documents
                                     if d["source_id"] in keep),
            "packed_sources": sorted(keep)}


def _predict_supervised(window: Any, *, bos_id: int = 2, pad_id: int = 0) -> int:
    """Predict the loss's supervised-target count for a window, exactly.

    Mirrors the ``causal_lm_loss`` keep rule over shifted positions: target
    j (j >= 1) is supervised iff the window marks it eligible, the segment
    continues across the shift, and the target is neither BOS nor PAD.
    The backend re-derives this count from the live loss during
    accumulation and ``finish_update`` fails closed on any disagreement,
    so a future loss-rule change breaks loudly here instead of drifting
    the global denominator.
    """

    total = 0
    for tokens, segments, eligible in zip(window.tokens, window.segment_ids,
                                          window.eligible):
        for position in range(1, len(tokens)):
            if (eligible[position]
                    and segments[position] == segments[position - 1]
                    and segments[position] >= 0
                    and tokens[position] != bos_id
                    and tokens[position] != pad_id):
                total += 1
    return total


def _runtime_name(device: Any) -> str:
    if device is None:
        return "cpu"
    return str(getattr(device, "type", device))


def run_campaign(*, documents: list[dict[str, Any]], tokenizer: Any,
                 model_spec, run_id: str, seed: int,
                 campaign_tokens: int,
                 max_updates: int | None = None,
                 store_root: str | Path, device: Any, torch_module: Any = None,
                 xb: Any | None = None, progress: Callable[[str], None]
                 | None = None, resume_store_root: str | Path | None = None,
                 resume_run_id: str | None = None,
                 development_mode: bool = False,
                 contamination_benchmarks: dict[str, str] | None = None,
                 cymek_sha: str | None = None,
                 stop_gate: Callable[[TrainingState], str | None] | None = None,
                 clock: Callable[[], float] | None = None,
                 milestones: tuple = MILESTONE_TOKENS,
                 recovery_tokens: int | None = None,
                 ) -> dict[str, Any]:
    """Execute (or resume) a token-targeted certified production campaign.

    Fresh vs resume comes from a valid committed LATEST in the campaign
    store, never from merely supplied paths: resume arguments must name
    this same store/lineage or fail closed. Production mode requires a
    frozen contamination commitment; development mode opts out explicitly
    and the receipt is labeled DEVELOPMENT. ``stop_gate`` must be a
    deterministic predicate of state (monotonic clocks only): it is
    consulted both to commit the stop boundary and to halt, so a
    nondeterministic gate could strand an uncommitted update.
    ``recovery_tokens`` overrides the frozen recovery cadence for
    short-horizon tests; production passes None.
    """

    if xb is None:
        raise ValueError("xb (device seam) is required")
    if not development_mode and not contamination_benchmarks:
        raise ValueError(
            "production campaigns require a frozen contamination commitment; "
            "pass development_mode=True to label a dry run explicitly")
    if campaign_tokens <= 0:
        raise ValueError("campaign token budget must be positive")
    if max_updates is not None and max_updates < 0:
        raise ValueError("max_updates cannot be negative")
    if resume_store_root is not None and Path(resume_store_root).resolve() != Path(store_root).resolve():
        raise ValueError(
            "cross-store resume is unsupported: resume store must equal the campaign store")
    if resume_run_id is not None and resume_run_id != run_id:
        raise ValueError(
            "cross-lineage resume is unsupported: resume run must equal the campaign run")
    torch = torch_module
    if torch is None:
        import torch as torch
    if clock is None:
        clock = time.time

    t0 = clock()
    topo = frozen_topology()
    per_update = topo["global_tokens_per_update"]
    microstep_tokens = topo["global_tokens_per_microstep"]
    recovery_every = recovery_tokens if recovery_tokens is not None else topo["recovery_threshold_tokens"]
    if recovery_every <= 0:
        raise ValueError("recovery cadence must be positive")
    cymek_sha = resolve_cymek_sha(cymek_sha)
    data = prepare_data(documents=documents, tokenizer=tokenizer,
                        run_id=run_id, seed=seed,
                        contamination_benchmarks=contamination_benchmarks)
    packed, sampler = data["packed"], data["sampler_order"]
    layout_order, layout_receipt = campaign_layout(
        packed, run_seed=seed, pattern=topo["supercycle"],
        base_order=list(sampler))
    flat_cache = build_flat_index(packed, layout_order)
    mode_label = "DEVELOPMENT" if development_mode else "PRODUCTION"
    schedule_doc = schedule_receipt()
    topology_digest = topology_sha256(topo)
    identities = IdentityBindings(
        schema=IDENTITY_SCHEMA, source_commit=cymek_sha,
        model_spec_sha256=model_spec.sha256(),
        tokenizer_sha256=tokenizer.identity.artifact_sha256,
        data_manifest_sha256=data["manifest_sha256"],
        pack_manifest_sha256=data["pack_manifest_sha256"],
        run_spec_sha256=hashlib.sha256(_canonical_json(
            {"campaign_tokens": campaign_tokens,
             "global_tokens_per_update": per_update,
             "topology_sha256": topology_digest,
             "seed": seed})).hexdigest(),
        optimizer_spec_sha256=hashlib.sha256(_canonical_json(
            {"optimizer": "AdamW", "beta1": topo["beta1"], "beta2": topo["beta2"],
             "epsilon": topo["epsilon"],
             "weight_decay": topo["weight_decay"]})).hexdigest(),
        schedule_spec_sha256=schedule_doc["sha256"],
        curriculum_spec_sha256=hashlib.sha256(_canonical_json(
            {"curriculum_phase": "500m-campaign",
             "policy": "uniform-supercycle-interleave"})).hexdigest())

    store = CheckpointStore(Path(store_root), run_id)
    runtime = _runtime_name(device)
    precision = precision_receipt(runtime=runtime, torch_module=torch)
    if precision.get("status") != "CERTIFIED_LOCAL":
        raise ValueError(
            f"runtime {runtime!r} is not certified for local execution: XLA/TPU "
            "collectives, memory fit, and bf16 behavior need PRE500M "
            "certification (TPU_EVIDENCE_REQUIRED)")
    latest = store.latest_sha256()
    if latest is None:
        torch.manual_seed(seed)
        model = initialize(model_spec, seed).to(device)
        optimizer = build_adamw_optimizer(model, torch_module=torch)
        backend = ProductionTrainingBackend(
            model=model, optimizer=optimizer, bos_id=2, pad_id=0, device=device,
            schedule=lr_at,
            bfloat16_autocast=runtime == "cuda",
            torch_module=torch)
        state = TrainingState.initial(
            lineage_id=run_id, token_budget=campaign_tokens,
            tokens_per_update=per_update,
            cursor=CursorState(CURSOR_SCHEMA, data["pack_manifest_sha256"],
                               0, 0, 0),
            rng_state_sha256=hashlib.sha256(
                torch.get_rng_state().numpy().tobytes()).hexdigest(),
            curriculum_phase="500m-campaign",
            identities=identities)
        resumed, done_updates = False, 0
    else:
        state, payloads = store.restore()
        if state.lineage_id != run_id:
            raise ValueError("restored state belongs to another lineage")
        for name, expected, actual in (
                ("model spec", identities.model_spec_sha256, state.identities.model_spec_sha256),
                ("tokenizer", identities.tokenizer_sha256, state.identities.tokenizer_sha256),
                ("data manifest", identities.data_manifest_sha256, state.identities.data_manifest_sha256),
                ("pack manifest", identities.pack_manifest_sha256, state.identities.pack_manifest_sha256),
                ("run spec", identities.run_spec_sha256, state.identities.run_spec_sha256),
                ("source commit", identities.source_commit, state.identities.source_commit),
                ("token budget", campaign_tokens, state.token_budget)):
            if expected != actual:
                raise ValueError(
                    f"resume identity drift: {name} changed since the committed checkpoint")
        torch.manual_seed(seed)
        model = initialize(model_spec, seed).to(device)
        optimizer = build_adamw_optimizer(model, torch_module=torch)
        backend = ProductionTrainingBackend(
            model=model, optimizer=optimizer, bos_id=2, pad_id=0, device=device,
            schedule=lr_at,
            bfloat16_autocast=runtime == "cuda",
            torch_module=torch)
        restore_production(backend, payloads=payloads)
        resumed, done_updates = True, state.global_update
        if state.cumulative_tokens >= campaign_tokens:
            return {"schema": ENTRY_SCHEMA, "run_id": run_id, "seed": seed,
                    "mode": mode_label,
                    "updates_executed": int(state.global_update),
                    "cumulative_tokens": int(state.cumulative_tokens),
                    "losses": [], "resumed": True, "already_complete": True,
                    "termination": "COMPLETE",
                    "cymek_sha": cymek_sha,
                    "data_manifest_sha256": data["manifest_sha256"],
                    "pack_manifest_sha256": data["pack_manifest_sha256"],
                    "model_spec_sha256": model_spec.sha256(),
                    "wall_seconds": 0.0}

    remaining_tokens = campaign_tokens - state.cumulative_tokens
    full_updates, tail = divmod(remaining_tokens, per_update)
    total_remaining = full_updates + (1 if tail else 0)
    capped = False
    if max_updates is not None and total_remaining > max_updates:
        total_remaining = max_updates
        capped = True
    if total_remaining <= 0:
        raise ValueError("campaign has no remaining updates")
    controller = RunController(target_update=state.global_update + total_remaining)
    controller.start()
    for _ in range(done_updates):
        controller.complete_update()
    losses: list[float] = []
    bucket_mix: dict[int, int] = {}
    intended_buckets: list[int] = []
    milestones_crossed: list[dict[str, object]] = []
    recovery_shas: list[str] = []
    milestone_shas: dict[int, str] = {}
    last_boundary_tokens = [state.cumulative_tokens]
    updates_this_session = [0]
    stop_reasons: list[str] = []

    def backend_step(current: TrainingState):
        expected = next_update_tokens(
            token_budget=current.token_budget,
            cumulative_tokens=current.cumulative_tokens,
            tokens_per_update=current.tokens_per_update)
        plans = partial_microstep_plan(
            remaining_tokens=expected, topo=topo,
            microstep_ordinal=current.cumulative_tokens // microstep_tokens)
        buckets = microstep_buckets(
            cumulative_tokens=current.cumulative_tokens,
            microstep_counts=list(plans), topo=topo)
        ctx = backend.begin_update(current)
        cursor = (current.cursor.shard_ordinal, current.cursor.sequence_ordinal,
                  current.cursor.token_offset)
        windows = []
        for micro_count, _bucket in zip(plans, buckets):
            intended_buckets.append(int(_bucket))
            window = exact_window(
                packed, layout_order,
                shard_ordinal=cursor[0], sequence_ordinal=cursor[1],
                token_offset=cursor[2], real_tokens=micro_count, pad=0,
                flat_cache=flat_cache)
            windows.append(window)
            cursor = (window.end_shard_ordinal, window.end_sequence_ordinal,
                      window.end_token_offset)
        supervised_total = sum(window.real_tokens for window in windows)
        if supervised_total != expected:
            raise ValueError("microstep accumulation disagrees with the update budget")
        eligible_total = sum(_predict_supervised(window) for window in windows)
        if eligible_total <= 0:
            raise ValueError("abort NO_SUPERVISED_TOKENS: update carried no eligible targets")
        for window in windows:
            for row in window.eligible:
                bucket_mix[len(row)] = bucket_mix.get(len(row), 0) + 1
            width = max(len(row) for row in window.tokens)
            tokens = torch.tensor(
                [list(row) + [0] * (width - len(row)) for row in window.tokens],
                dtype=torch.long, device=device)
            segment_ids = torch.tensor(
                [list(row) + [-1] * (width - len(row)) for row in window.segment_ids],
                dtype=torch.long, device=device)
            eligible = torch.tensor(
                [list(row) + [False] * (width - len(row)) for row in window.eligible],
                dtype=torch.bool, device=device)
            ctx = backend.accumulate_microstep(
                ctx, tokens=tokens, segment_ids=segment_ids, eligible=eligible,
                tokens_by_source=dict(window.tokens_by_source),
                planned_total=eligible_total)
        end_cursor = CursorState(
            CURSOR_SCHEMA, data["pack_manifest_sha256"],
            cursor[0], cursor[1], cursor[2])
        report = backend.finish_update(
            current, ctx, planned_total=eligible_total, cursor=end_cursor)
        updates_this_session[0] += 1
        receipt = backend.last_receipt
        assert receipt is not None
        losses.append(float(receipt["loss"]))
        return report

    def payload_builder(live: TrainingState) -> dict[str, bytes]:
        return production_payloads(backend, state=live)

    def at_stop_boundary(live: TrainingState) -> bool:
        if max_updates is not None and updates_this_session[0] >= max_updates:
            return True
        return stop_gate is not None and stop_gate(live) is not None

    def should_checkpoint(live: TrainingState) -> bool:
        if crossed_milestones(last_boundary_tokens[0], live.cumulative_tokens,
                              milestones):
            return True
        if live.cumulative_tokens // recovery_every > last_boundary_tokens[0] // recovery_every:
            return True
        # Every stop boundary must be committed, or the halted session's
        # work would stay volatile and unresumable.
        return live.complete or at_stop_boundary(live)

    def on_committed(live: TrainingState, checkpoint_sha: str) -> None:
        for threshold in crossed_milestones(last_boundary_tokens[0], live.cumulative_tokens,
                                            milestones):
            if threshold not in milestone_shas:
                milestone_shas[threshold] = checkpoint_sha
                milestones_crossed.append({
                    "threshold_tokens": threshold,
                    "actual_cumulative_tokens": live.cumulative_tokens,
                    "checkpoint_sha256": checkpoint_sha,
                    "global_update": live.global_update,
                })
                if progress is not None:
                    progress(f"milestone {threshold} at update {live.global_update}")
        if live.cumulative_tokens // recovery_every > last_boundary_tokens[0] // recovery_every:
            recovery_shas.append(checkpoint_sha)
            if progress is not None:
                progress(f"recovery at update {live.global_update}")
        last_boundary_tokens[0] = live.cumulative_tokens
        keep = (set(milestone_shas.values())
                | set(recovery_shas[-topo["recovery_generations_retained"]:])
                | {checkpoint_sha})
        store.prune(keep=keep)

    def should_stop(live: TrainingState) -> bool:
        if max_updates is not None and updates_this_session[0] >= max_updates:
            stop_reasons.append("MANUAL_BOUNDARY")
            return True
        if stop_gate is not None:
            reason = stop_gate(live)
            if reason is not None:
                stop_reasons.append(reason)
                return True
        return False

    final = train(
        state=state, controller=controller, store=store,
        payload_builder=payload_builder, backend_step=backend_step,
        updates=total_remaining, checkpoint_every=None,
        should_checkpoint=should_checkpoint, on_committed=on_committed,
        should_stop=should_stop,
        resume_parent_sha256=latest if resumed else None)
    if final.complete:
        termination = "COMPLETE"
    elif stop_reasons:
        termination = stop_reasons[-1]
    elif capped:
        termination = "MANUAL_BOUNDARY"
    else:
        raise RuntimeError("campaign ended without completion, boundary, or stop")
    wall = clock() - t0

    fresh_model = initialize(model_spec, seed).to(device)
    fresh_optimizer = build_adamw_optimizer(fresh_model, torch_module=torch)
    fresh_backend = ProductionTrainingBackend(
        model=fresh_model, optimizer=fresh_optimizer, bos_id=2, pad_id=0,
        device=device, schedule=lr_at,
        bfloat16_autocast=runtime == "cuda",
        torch_module=torch)
    _, restored_payloads = store.restore()
    restore_production(fresh_backend, payloads=restored_payloads)
    live = capture_evidence(backend.model, backend.optimizer, torch=torch)
    resumed_evidence = capture_evidence(fresh_model, fresh_optimizer, torch=torch)
    resume_equal = (live.parameter_sha256 == resumed_evidence.parameter_sha256
                    and live.moment_sha256 == resumed_evidence.moment_sha256
                    and live.optimizer_steps == resumed_evidence.optimizer_steps)
    if not resume_equal:
        raise RuntimeError("campaign restore did not reproduce live state")
    last_update = backend.last_receipt
    assert last_update is not None

    return {
        "schema": ENTRY_SCHEMA, "run_id": run_id, "seed": seed,
        "mode": mode_label, "campaign_tokens": campaign_tokens,
        "updates_executed": int(final.global_update),
        "cumulative_tokens": int(final.cumulative_tokens),
        "state_complete": bool(final.complete),
        "termination": termination,
        "resumed": resumed,
        "losses": losses, "wall_seconds": round(wall, 3),
        "resume_equal": resume_equal,
        "cymek_sha": cymek_sha,
        "topology_sha256": topology_digest,
        "precision": precision,
        "layout_sha256": layout_receipt["layout_sha256"],
        "sampler_order_sha256": layout_receipt["sampler_order_sha256"],
        "microstep_bucket_mix": dict(sorted(bucket_mix.items())),
        "microstep_buckets": list(intended_buckets),
        "milestones_crossed": milestones_crossed,
        "recovery_checkpoint_count": len(recovery_shas),
        "recovery_tokens": recovery_every,
        "data_manifest_sha256": data["manifest_sha256"],
        "pack_manifest_sha256": data["pack_manifest_sha256"],
        "model_spec_sha256": model_spec.sha256(),
        "checkpoint_head": store.latest_sha256(),
        "tokens_by_source": dict(final.tokens_by_source),
        "last_update_receipt": {
            "loss": float(last_update["loss"]),
            "supervised_tokens": int(last_update["supervised_tokens"]),
            "microsteps": int(last_update["microsteps"]),
            "learning_rate": float(last_update["learning_rate"]),
            "grad_norm_post_clip": float(last_update["grad_norm_post_clip"]),
        },
    }


def run_500m_session(*, documents: list[dict[str, Any]], tokenizer: Any,
                     model_spec, run_id: str, seed: int,
                     campaign_tokens: int, session_dir: str | Path,
                     device: Any, torch_module: Any = None,
                     xb: Any | None = None,
                     max_session_minutes: float = 90.0,
                     margin_seconds: float = 60.0,
                     recovery_interval_updates: int = 25,
                     development_mode: bool = False,
                     contamination_benchmarks: dict[str, str] | None = None,
                     cymek_sha: str | None = None,
                     progress: Callable[[str], None] | None = None,
                     clock: Callable[[], float] | None = None,
                     milestones: tuple = MILESTONE_TOKENS,
                     recovery_tokens: int | None = None,
                     ) -> dict[str, Any]:
    """One 500M-campaign training session with milestone detection,
    recovery checkpointing, heartbeat, and session receipt.

    ``recovery_interval_updates`` is accepted for operator continuity but the
    commit cadence is token-indexed (the frozen recovery threshold), never
    update-count-indexed: update counts change meaning between full and
    partial updates. Multi-session: call again on a fresh runtime — resumes
    from the latest committed generation in ``session_dir/campaign_store``.
    Milestone files point at the checkpoint that crossed each threshold,
    written once and never duplicated. Stops cleanly before deadline with a
    RESUMABLE receipt; the next runtime continues automatically.
    """

    if max_session_minutes <= 0:
        raise ValueError("session timebox must be positive")
    if margin_seconds < 0:
        raise ValueError("session margin cannot be negative")
    if recovery_interval_updates <= 0:
        raise ValueError("recovery interval must be positive")
    root = Path(session_dir)
    root.mkdir(parents=True, exist_ok=True)
    store_root = str(root / "campaign_store")
    milestone_dir = root / "milestones"
    milestone_dir.mkdir(parents=True, exist_ok=True)
    if clock is None:
        clock = time.time
    deadline = clock() + max_session_minutes * 60.0

    def hb(**kw):
        doc = {"schema": "anra-v5-heartbeat/v1", "utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}
        (root / "HEARTBEAT.json").write_text(
            json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")

    hb(phase="starting", campaign_tokens=campaign_tokens)

    def stop_gate(live: TrainingState) -> str | None:
        if clock() >= deadline - margin_seconds:
            return "TIMEBOX"
        return None

    result = run_campaign(
        documents=documents, tokenizer=tokenizer, model_spec=model_spec,
        run_id=run_id, seed=seed, campaign_tokens=campaign_tokens,
        store_root=store_root, device=device, torch_module=torch_module,
        xb=xb, progress=progress, resume_store_root=store_root,
        resume_run_id=run_id, development_mode=development_mode,
        contamination_benchmarks=contamination_benchmarks,
        cymek_sha=cymek_sha, stop_gate=stop_gate, clock=clock,
        milestones=milestones, recovery_tokens=recovery_tokens)

    for milestone in result["milestones_crossed"]:
        mpath = milestone_dir / f"milestone_{milestone['threshold_tokens']}.json"
        if not mpath.is_file():
            mpath.write_text(json.dumps({
                "schema": "anra-v5-milestone/v1",
                "tokens": milestone["threshold_tokens"],
                "actual_cumulative_tokens": milestone["actual_cumulative_tokens"],
                "utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "checkpoint": milestone["checkpoint_sha256"],
                "global_update": milestone["global_update"],
                "run_id": run_id,
            }, indent=2, sort_keys=True), encoding="utf-8")

    session_status = "COMPLETE" if result["termination"] == "COMPLETE" else "RESUMABLE"
    session_receipt = {
        "schema": "anra-v5-session-receipt/v1",
        "run_id": run_id, "campaign_tokens": campaign_tokens,
        "updates_executed": result["updates_executed"],
        "cumulative_tokens": result["cumulative_tokens"],
        "milestones_crossed": [m["threshold_tokens"] for m in result["milestones_crossed"]],
        "losses": result["losses"][-20:] if result["losses"] else [],
        "resume_equal": result["resume_equal"],
        "wall_seconds": result["wall_seconds"],
        "checkpoint_head": result["checkpoint_head"],
        "termination": result["termination"],
        "status": session_status,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (root / "SESSION_RECEIPT.json").write_text(
        json.dumps(session_receipt, indent=2, sort_keys=True),
        encoding="utf-8")
    hb(phase="complete" if session_status == "COMPLETE" else "timeboxed",
       tokens=result["cumulative_tokens"],
       milestones=[m["threshold_tokens"] for m in result["milestones_crossed"]])

    return {"schema": "anra-v5-500m-session/v1",
            "session_dir": str(root),
            "result": result,
            "milestones_crossed": result["milestones_crossed"],
            "session_receipt": session_receipt,
            "campaign_store": store_root,
            "milestone_dir": str(milestone_dir)}


__all__ = ["ENTRY_SCHEMA", "MILESTONE_TOKENS", "campaign_layout",
           "crossed_milestones", "frozen_topology", "microstep_buckets",
           "partial_microstep_plan", "prepare_data", "resolve_cymek_sha",
           "run_500m_session", "run_campaign", "topology_sha256"]
