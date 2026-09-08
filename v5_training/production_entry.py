"""500M campaign production entry point — the single real chain.

    documents -> data manifest -> frozen tokenizer -> packing
    -> bucket lanes -> deficit mixture schedule -> streaming exact-token
    microsteps (one frozen bucket shape each) -> gradient accumulation to
    one logical optimizer update -> trainer state machine
    -> checkpoint transactions (+ durable milestones) -> exact-resume
    verification -> campaign receipt

A logical optimizer update is N accumulation microsteps (4 x 32,768 global
real tokens = 131,072 for the frozen topology), never one giant batch.
Each microstep executes ONE frozen bucket shape: every row has native width
<= B and the assembled tensor is exactly width B with the frozen per-replica
sequence count. TrainingState advances exactly once per logical update.
Local CPU/CUDA execution runs the GLOBAL logical microstep on one device (a
mathematical emulation, never TPU evidence); topology_map is the single
authority that shards rows for real data-parallel execution.
Resume is inferred from a valid committed LATEST, never from supplied paths.
Fail-closed on: identity drift, ledger drift, bucket violation, mixture
shortfall, nonfinite loss/gradients, parameter non-mutation, stale writer,
restore mismatch, missing production contamination commitment.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from v5_contracts.training_spec import build_training_spec
from v5_data.bucket_cursor import (
    BUCKET_CURSOR_SCHEMA,
    BucketCursorState,
    build_bucket_lanes,
    cell_key,
    take_cell_window,
)
from v5_data.lifecycle import require_runnable
from v5_data.manifest import Document, build_data_manifest, manifest_sha256
from v5_data.mixture import (
    DeficitScheduler,
    allocate,
    mixture_schedule_sha256,
)
from v5_data.near_dedup import FROZEN_NEAR_DUP_POLICY
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
    IDENTITY_SCHEMA,
    IdentityBindings,
    TrainingState,
    next_update_tokens,
)
from v5_training.topology_map import certify_microstep_shape
from v5_training.trainer import train

ENTRY_SCHEMA = "anra-v5-production-entry-receipt/v1"
MILESTONE_SCHEMA = "anra-v5-milestone-receipt/v1"
# 500M-campaign milestone thresholds. Distinct from the 5B-run schedule in
# training_spec; these gate the 500M campaign only.
MILESTONE_TOKENS = (50_000_000, 100_000_000, 200_000_000, 350_000_000, 500_000_000)
VERIFIED_COGNITION = "verified_cognition"


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


def frozen_mixture_fractions() -> dict[str, float]:
    """Mechanically read the frozen 65/20/15 family fractions (no duplication)."""

    fractions = build_training_spec()["data"]["mixture_fractions"]
    result = {str(name): float(share) for name, share in fractions.items()}
    if abs(sum(result.values()) - 1.0) > 1e-9:
        raise ValueError("frozen mixture fractions do not sum to one")
    return result


def frozen_cognition_fractions() -> dict[str, float]:
    """Mechanically read the frozen within-cognition family fractions."""

    fractions = build_training_spec()["cognition"]["family_fractions_within_cognition"]
    result = {str(name): float(share) for name, share in fractions.items()}
    if abs(sum(result.values()) - 1.0) > 1e-9:
        raise ValueError("frozen cognition fractions do not sum to one")
    return result


def microstep_buckets(
    *, cumulative_tokens: int, microstep_counts: list[int], topo: Mapping[str, Any]
) -> list[int]:
    """Bucket per microstep from the frozen supercycle position.

    Position derives from cumulative real tokens only, so resume reproduces
    it without extra state (stronger than checkpointing the position: there
    is no stored value that could drift from the token count).
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
    position-independent. No overshoot, no undershoot. This is generic over
    campaign targets: it never imports the 5B-run final-partial arithmetic.
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
    """Require the real executable SHA: explicit param or mechanical git resolve."""

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


def _predict_supervised(window: Any, *, bos_id: int = 2, pad_id: int = 0) -> int:
    """Predict the loss's supervised-target count for a window, exactly.

    Mirrors the ``causal_lm_loss`` keep rule over shifted positions. The
    backend re-derives this count from the live loss during accumulation and
    ``finish_update`` fails closed on any disagreement.
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


def prepare_data(*, documents: list[dict[str, Any]], tokenizer: Any,
                 run_id: str, seed: int,
                 contamination_benchmarks: dict[str, str] | None = None,
                 mixture_families: tuple[str, ...] | None = None,
                 cognition_map: Mapping[str, str] | None = None,
                 development_mode: bool = True,
                 ) -> dict[str, Any]:
    """Documents -> data manifest -> packing -> sampler order (data half).

    Deterministic in (documents, seed, mixture config). ``mixture_families``
    enables family-segregated packing with per-source (family, subfamily)
    cells; without it every sequence pools into one cell per bucket.
    Production (development_mode=False) requires strict raw-source
    provenance, the frozen near-duplicate policy, and an artifact-backed
    tokenizer identity — no fallback or provisional tokenizer.
    """

    tokenizer_identity = getattr(tokenizer, "identity", None)
    artifact_sha = getattr(tokenizer_identity, "artifact_sha256", "") or ""
    if not artifact_sha:
        raise ValueError("tokenizer lacks an artifact-backed identity")
    records = [
        Document(doc_id=d["doc_id"], text=d["text"],
                 source_id=d["source_id"], domain=d.get("domain", "text"),
                 family=d.get("family", "natural"),
                 authorization_category=d.get(
                     "authorization_category", "first-party-authorized"),
                 acquired_date=d.get("acquired_date", "1970-01-01"),
                 raw_source_sha256=d.get("raw_source_sha256", ""))
        for d in documents]
    if mixture_families is not None:
        unknown = sorted({rec.family for rec in records} - set(mixture_families))
        if unknown:
            raise ValueError(f"documents carry families outside the mixture: {unknown}")
    cell_of_source: dict[str, tuple[str, str]] | None = None
    if mixture_families is not None:
        cell_of_source = {}
        for d in documents:
            family = d.get("family", "natural")
            sub = ""
            if family == VERIFIED_COGNITION and cognition_map is not None:
                sub = str(cognition_map.get(d["source_id"], ""))
            cell_of_source[d["source_id"]] = (family, sub)
    manifest, manifest_audit = build_data_manifest(
        records, manifest_id=f"{run_id}-data",
        tokenizer_sha256=artifact_sha,
        filter_version="production-entry-filter/v1",
        dedup_version=("exact+near-minhash-lsh/v1" if not development_mode
                       else "exact-clusters/v1"),
        split_salt=f"{run_id}/v1",
        split_boundaries={"training": 1.0, "development": 0.0,
                          "sealed": 0.0, "fresh": 0.0},
        count_tokens=lambda text: len(tokenizer.encode(text)),
        contamination_benchmarks=contamination_benchmarks or {},
        strict_provenance=not development_mode,
        near_duplicate_policy=None if development_mode else dict(FROZEN_NEAR_DUP_POLICY))
    keep = {rec.source_id for rec in manifest.sources
            if rec.split == "training"}
    packed_docs = [(d["doc_id"], tokenizer.encode(d["text"]), d["source_id"])
                   for d in documents if d["source_id"] in keep]
    packed, pack_audit = pack_documents(
        packed_docs, bos=2, eos=3, pad=0, sequences_per_shard=8,
        cell_of_source=cell_of_source)
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
            "cell_of_source": cell_of_source,
            "packed_doc_ids": sorted(d["doc_id"] for d in documents
                                     if d["source_id"] in keep),
            "packed_sources": sorted(keep)}


def campaign_microstep_plan(*, start_tokens: int, campaign_tokens: int,
                            topo: Mapping[str, Any]) -> list[tuple[int, int]]:
    """Exact (bucket, real-token count) sequence for a campaign remainder.

    Pure function of token counts: offline demand planning and live execution
    consume the same plan, so supply shortfalls are proven before training.
    """

    per_update = topo["global_tokens_per_update"]
    microstep = topo["global_tokens_per_microstep"]
    cycle = topo["supercycle"]
    if not 0 <= start_tokens <= campaign_tokens:
        raise ValueError("campaign remainder is outside the budget")
    plan: list[tuple[int, int]] = []
    cumulative, ordinal = start_tokens, start_tokens // microstep
    remaining = campaign_tokens - start_tokens
    while remaining > 0:
        expected = min(per_update, remaining)
        full, tail = divmod(expected, microstep)
        counts = [microstep] * full + ([tail] if tail else [])
        for count in counts:
            plan.append((int(cycle[ordinal % len(cycle)]), count))
            ordinal += 1
        cumulative += expected
        remaining -= expected
    return plan


def assign_mixture_cell(*, fam_consumed: dict[str, int],
                        sub_consumed: dict[str, int],
                        total_consumed: int,
                        fam_scheduler: DeficitScheduler | None,
                        sub_scheduler: DeficitScheduler | None,
                        cognition_mapped: bool) -> tuple[str, str]:
    """Pure per-microstep (family, subfamily) assignment. Single authority.

    Both the offline demand planner and live execution call this with the
    same counters, so planning and execution cannot diverge. Without a family
    scheduler every microstep pools into ("", "").
    """

    if fam_scheduler is None:
        return "", ""
    family = fam_scheduler.next(consumed_total=total_consumed,
                                consumed=fam_consumed)
    sub = ""
    if family == VERIFIED_COGNITION and cognition_mapped and sub_scheduler is not None:
        sub = sub_scheduler.next(
            consumed_total=fam_consumed.get(family, 0), consumed=sub_consumed)
    return family, sub


def plan_campaign_demand(*, microstep_plan: list[tuple[int, int]],
                         fam_scheduler: DeficitScheduler | None,
                         sub_scheduler: DeficitScheduler | None,
                         cognition_mapped: bool,
                         initial_fam_consumed: dict[str, int] | None = None,
                         initial_sub_consumed: dict[str, int] | None = None,
                         initial_total: int = 0) -> dict[str, int]:
    """Exact per-cell real-token demand for a microstep plan. Pure.

    Counters start from the (possibly restored) schedule state, so offline
    planning and live execution from the same state cannot diverge — including
    across resume, where the plan covers only the remainder but the counters
    continue from the checkpoint.
    """

    fam_consumed: dict[str, int] = dict(initial_fam_consumed or {})
    sub_consumed: dict[str, int] = dict(initial_sub_consumed or {})
    total = initial_total
    if total < 0:
        raise ValueError("planning total cannot be negative")
    demand: dict[str, int] = {}
    for bucket, count in microstep_plan:
        family, sub = assign_mixture_cell(
            fam_consumed=fam_consumed, sub_consumed=sub_consumed,
            total_consumed=total, fam_scheduler=fam_scheduler,
            sub_scheduler=sub_scheduler, cognition_mapped=cognition_mapped)
        key = cell_key(bucket, family, sub)
        demand[key] = demand.get(key, 0) + count
        fam_consumed[family] = fam_consumed.get(family, 0) + count
        if sub:
            sub_consumed[sub] = sub_consumed.get(sub, 0) + count
        total += count
    return dict(sorted(demand.items()))


def build_milestone_receipt(*, run_id: str, threshold_tokens: int,
                            actual_cumulative_tokens: int,
                            global_update: int, checkpoint_sha256: str,
                            identity_bundle: Mapping[str, str]) -> dict[str, object]:
    """Self-describing milestone receipt binding every campaign identity."""

    receipt: dict[str, object] = {
        "schema": MILESTONE_SCHEMA,
        "run_id": run_id,
        "threshold_tokens": threshold_tokens,
        "actual_cumulative_tokens": actual_cumulative_tokens,
        "global_update": global_update,
        "checkpoint_sha256": checkpoint_sha256,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt.update(dict(identity_bundle))
    return receipt


def verify_milestone_receipt(receipt: Mapping[str, Any],
                             store: CheckpointStore) -> bool:
    """Prove a milestone receipt matches its referenced checkpoint.

    Restores the referenced generation through the store's own verifying
    path and requires the receipt's tokens, update, lineage, and every bound
    identity to equal the checkpoint TrainingState. A dangling receipt
    (deleted bytes) fails here, loudly. Returns True on success.
    """

    if receipt.get("schema") != MILESTONE_SCHEMA:
        raise ValueError("not a milestone receipt")
    sha = receipt.get("checkpoint_sha256")
    if not isinstance(sha, str):
        raise ValueError("milestone receipt lacks a checkpoint SHA")
    state, _ = store.restore(sha)
    for key, actual in (
            ("actual_cumulative_tokens", state.cumulative_tokens),
            ("global_update", state.global_update),
            ("run_id", state.lineage_id)):
        if receipt.get(key) != actual:
            raise ValueError(
                f"milestone receipt field {key} disagrees with checkpoint {sha[:8]}")
    identities = state.identities
    for key, actual in (
            ("cymek_sha", identities.source_commit),
            ("model_spec_sha256", identities.model_spec_sha256),
            ("tokenizer_sha256", identities.tokenizer_sha256),
            ("data_manifest_sha256", identities.data_manifest_sha256),
            ("pack_manifest_sha256", identities.pack_manifest_sha256),
            ("schedule_spec_sha256", identities.schedule_spec_sha256)):
        if receipt.get(key) != actual:
            raise ValueError(
                f"milestone receipt identity {key} disagrees with checkpoint {sha[:8]}")
    return True


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
                 mixture_fractions: dict[str, float] | None = None,
                 cognition_map: Mapping[str, str] | None = None,
                 allow_replay: bool = False,
                 dataset_lifecycle=None,
                 tokenizer_freeze_sha256: str | None = None,
                 ) -> dict[str, Any]:
    """Execute (or resume) a token-targeted certified production campaign.

    Fresh vs resume comes from a valid committed LATEST, never from supplied
    paths. Production mode requires a frozen contamination commitment, the
    mixture fractions, and the tokenizer freeze SHA; development mode opts
    out explicitly and the receipt is labeled DEVELOPMENT. ``stop_gate``
    must be deterministic in state. ``allow_replay`` permits bounded epoch
    replay on lane exhaustion (default fails closed DATA_NOT_READY).
    ``dataset_lifecycle``, when provided, must already be RUNNABLE.
    """

    if xb is None:
        raise ValueError("xb (device seam) is required")
    if not development_mode and not contamination_benchmarks:
        raise ValueError(
            "production campaigns require a frozen contamination commitment; "
            "pass development_mode=True to label a dry run explicitly")
    if not development_mode and mixture_fractions is None:
        raise ValueError(
            "production campaigns require explicit mixture fractions; "
            "silent pooling is not a mixture")
    if not development_mode and not tokenizer_freeze_sha256:
        raise ValueError(
            "production campaigns require the frozen tokenizer identity; "
            "no fallback or provisional tokenizer")
    if dataset_lifecycle is not None:
        require_runnable(dataset_lifecycle)
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
    replicas = topo["replicas"]
    per_replica_counts = topo["sequences_per_replica_by_bucket"]
    recovery_every = recovery_tokens if recovery_tokens is not None else topo["recovery_threshold_tokens"]
    if recovery_every <= 0:
        raise ValueError("recovery cadence must be positive")
    cymek_sha = resolve_cymek_sha(cymek_sha)
    mixture_families = tuple(sorted(mixture_fractions)) if mixture_fractions else None
    data = prepare_data(documents=documents, tokenizer=tokenizer,
                        run_id=run_id, seed=seed,
                        contamination_benchmarks=contamination_benchmarks,
                        mixture_families=mixture_families,
                        cognition_map=cognition_map,
                        development_mode=development_mode)
    packed, cell_of_source = data["packed"], data["cell_of_source"]

    fam_scheduler = (DeficitScheduler(fractions=dict(mixture_fractions),
                                      order=tuple(sorted(mixture_fractions)))
                     if mixture_fractions else None)
    cognition_mapped = bool(cognition_map) and VERIFIED_COGNITION in (mixture_families or ())
    sub_scheduler = (DeficitScheduler(fractions=frozen_cognition_fractions())
                     if cognition_mapped else None)
    if mixture_fractions is not None:
        allocation = allocate(campaign_tokens, dict(mixture_fractions))
    else:
        allocation = {}
    mixture_plan_sha = (mixture_schedule_sha256(
        fractions=dict(mixture_fractions), allocation=dict(allocation),
        cell_map_sha256=hashlib.sha256(_canonical_json(
            {key: list(value) for key, value in sorted((cell_of_source or {}).items())}
        )).hexdigest() if cell_of_source else None)
        if mixture_fractions else None)

    def build_lanes(epoch: int, required: set[int]):
        return build_bucket_lanes(
            packed, run_seed=seed, pattern=topo["supercycle"], epoch=epoch,
            cell_of_source=cell_of_source, required_buckets=required)

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
             "mixture_plan_sha256": mixture_plan_sha,
             "seed": seed})).hexdigest(),
        optimizer_spec_sha256=hashlib.sha256(_canonical_json(
            {"optimizer": "AdamW", "beta1": topo["beta1"], "beta2": topo["beta2"],
             "epsilon": topo["epsilon"],
             "weight_decay": topo["weight_decay"]})).hexdigest(),
        schedule_spec_sha256=schedule_doc["sha256"],
        curriculum_spec_sha256=hashlib.sha256(_canonical_json(
            {"curriculum_phase": "500m-campaign",
             "policy": "uniform-supercycle-interleave"})).hexdigest())
    identity_bundle = {
        "cymek_sha": cymek_sha,
        "model_spec_sha256": model_spec.sha256(),
        "tokenizer_sha256": tokenizer.identity.artifact_sha256,
        "tokenizer_freeze_sha256": tokenizer_freeze_sha256 or "",
        "data_manifest_sha256": data["manifest_sha256"],
        "pack_manifest_sha256": data["pack_manifest_sha256"],
        "topology_sha256": topology_digest,
        "schedule_spec_sha256": schedule_doc["sha256"],
        "mixture_plan_sha256": mixture_plan_sha or "",
        "run_spec_sha256": identities.run_spec_sha256,
    }

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
        fresh_state = True
        resumed, done_updates = False, 0
        start_tokens = 0
        epoch, replay_events = 0, []
        fam_consumed = {}
        sub_consumed = {}
    else:
        state, payloads = store.restore()
        if state.lineage_id != run_id:
            raise ValueError("restored state belongs to another lineage")
        if not isinstance(state.cursor, BucketCursorState):
            raise ValueError(
                "restored cursor is not a bucket cursor: this engine revision "
                "cannot continue flat-cursor lineages")
        for name, expected, actual in (
                ("model spec", identities.model_spec_sha256, state.identities.model_spec_sha256),
                ("tokenizer", identities.tokenizer_sha256, state.identities.tokenizer_sha256),
                ("data manifest", identities.data_manifest_sha256, state.identities.data_manifest_sha256),
                ("pack manifest", identities.pack_manifest_sha256, state.identities.pack_manifest_sha256),
                ("run spec", identities.run_spec_sha256, state.identities.run_spec_sha256),
                ("optimizer spec", identities.optimizer_spec_sha256, state.identities.optimizer_spec_sha256),
                ("schedule spec", identities.schedule_spec_sha256, state.identities.schedule_spec_sha256),
                ("curriculum spec", identities.curriculum_spec_sha256, state.identities.curriculum_spec_sha256),
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
        start_tokens = state.cumulative_tokens
        fresh_state = False
        epoch = state.cursor.epoch
        replay_events = []
        positions = {key: [int(value[0]), int(value[1])]
                     for key, value in state.cursor.positions.items()}
        fam_consumed = dict(state.cursor.mixture_consumed)
        sub_consumed = dict(state.cursor.sub_consumed)
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

    microstep_plan = campaign_microstep_plan(
        start_tokens=start_tokens, campaign_tokens=campaign_tokens,
        topo=topo)
    required_buckets = {bucket for bucket, _ in microstep_plan}
    lanes, lanes_receipt = build_lanes(epoch, required_buckets)
    if fresh_state:
        positions = {key: [0, 0] for key in lanes}
        state = TrainingState.initial(
            lineage_id=run_id, token_budget=campaign_tokens,
            tokens_per_update=per_update,
            cursor=BucketCursorState(
                BUCKET_CURSOR_SCHEMA, data["pack_manifest_sha256"],
                lanes_receipt["lanes_sha256"], positions, {}, {}, 0, 0),
            rng_state_sha256=hashlib.sha256(
                torch.get_rng_state().numpy().tobytes()).hexdigest(),
            curriculum_phase="500m-campaign",
            identities=identities)
    elif lanes_receipt["lanes_sha256"] != state.cursor.lanes_sha256:
        raise ValueError("resume lane drift: rebuilt lanes disagree with the checkpoint")
    demand = plan_campaign_demand(
        microstep_plan=microstep_plan, fam_scheduler=fam_scheduler,
        sub_scheduler=sub_scheduler, cognition_mapped=cognition_mapped,
        initial_fam_consumed=dict(fam_consumed),
        initial_sub_consumed=dict(sub_consumed),
        initial_total=state.cumulative_tokens)
    supply = {key: int(cell["real_tokens"])
              for key, cell in lanes_receipt["cells"].items()}
    shortfall = {key: demand[key] - supply.get(key, 0) for key in demand
                 if demand[key] > supply.get(key, 0)}
    if shortfall and not allow_replay:
        raise ValueError(
            "abort DATA_NOT_READY: lane supply shortfall "
            f"{dict(sorted(shortfall.items()))}; bounded replay not permitted")

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
    microstep_shapes: list[dict[str, object]] = []
    milestones_crossed: list[dict[str, object]] = []
    recovery_shas: list[str] = []
    milestone_shas: dict[int, str] = {}
    last_boundary_tokens = [state.cumulative_tokens]
    updates_this_session = [0]
    stop_reasons: list[str] = []
    total_consumed = [state.cumulative_tokens]
    base_replays = (state.cursor.replay_count
                    if isinstance(state.cursor, BucketCursorState) else 0)

    def take_window(bucket: int, family: str, sub: str, count: int):
        nonlocal lanes, lanes_receipt, epoch
        from v5_data.bucket_cursor import LaneWindow, lane_remainder
        key = cell_key(bucket, family, sub)
        if key not in lanes:
            raise ValueError(
                f"abort DATA_NOT_READY: no packed supply for cell {key}; "
                "no silent substitution across buckets or families")
        if not lanes[key]:
            raise ValueError(f"abort DATA_NOT_READY: lane {key} holds no sequences")
        merged_tokens: list = []
        merged_segments: list = []
        merged_eligible: list = []
        merged_widths: list = []
        merged_source: dict[str, int] = {}
        merged_family: dict[str, int] = {}
        merged_real = 0
        end_index, end_offset = positions[key]
        remaining = count
        for _attempt in range(1024):
            lane = lanes[key]
            available = lane_remainder(packed, lane, end_index, end_offset, pad=0)
            if available <= 0:
                if not allow_replay:
                    raise ValueError(
                        f"abort DATA_NOT_READY: lane exhausted for cell {key}; "
                        "bounded replay not permitted") from None
                epoch += 1
                lanes, lanes_receipt = build_lanes(epoch, required_buckets)
                for reset_key in positions:
                    positions[reset_key] = [0, 0]
                replay_events.append({"epoch": epoch,
                                      "at_cumulative_tokens": total_consumed[0]})
                if progress is not None:
                    progress(f"epoch replay {epoch}")
                lane = lanes[key]
                if not lane:
                    raise ValueError(
                        f"abort DATA_NOT_READY: cell {key} cannot supply "
                        f"{count} real tokens even fresh") from None
                end_index, end_offset = 0, 0
                continue
            window = take_cell_window(
                packed, lane, end_index, end_offset,
                real_tokens=min(remaining, available), pad=0, bucket=bucket,
                cell_of_source=cell_of_source)
            merged_tokens.extend(window.tokens)
            merged_segments.extend(window.segment_ids)
            merged_eligible.extend(window.eligible)
            merged_widths.extend(window.row_widths)
            for source, amount in window.tokens_by_source.items():
                merged_source[source] = merged_source.get(source, 0) + amount
            for name, amount in window.tokens_by_family.items():
                merged_family[name] = merged_family.get(name, 0) + amount
            merged_real += window.real_tokens
            remaining -= window.real_tokens
            end_index, end_offset = window.end_lane_index, window.end_token_offset
            if remaining <= 0:
                break
        if remaining > 0:
            raise ValueError(
                f"abort DATA_NOT_READY: cell {key} cannot supply {count} real tokens")
        return LaneWindow(
            tokens=tuple(merged_tokens), segment_ids=tuple(merged_segments),
            eligible=tuple(merged_eligible),
            tokens_by_source=dict(sorted(merged_source.items())),
            tokens_by_family=dict(sorted(merged_family.items())),
            real_tokens=merged_real, row_widths=tuple(merged_widths),
            end_lane_index=end_index, end_token_offset=end_offset)

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
        windows = []
        for micro_count, bucket in zip(plans, buckets):
            family, sub = assign_mixture_cell(
                fam_consumed=fam_consumed, sub_consumed=sub_consumed,
                total_consumed=total_consumed[0], fam_scheduler=fam_scheduler,
                sub_scheduler=sub_scheduler, cognition_mapped=cognition_mapped)
            window = take_window(bucket, family, sub, micro_count)
            windows.append((bucket, family, sub, window))
            key = cell_key(bucket, family, sub)
            positions[key] = [window.end_lane_index, window.end_token_offset]
            fam_consumed[family] = fam_consumed.get(family, 0) + window.real_tokens
            if sub:
                sub_consumed[sub] = sub_consumed.get(sub, 0) + window.real_tokens
            total_consumed[0] += window.real_tokens
        supervised_total = sum(window.real_tokens for _, _, _, window in windows)
        if supervised_total != expected:
            raise ValueError("microstep accumulation disagrees with the update budget")
        eligible_total = sum(_predict_supervised(window) for _, _, _, window in windows)
        if eligible_total <= 0:
            raise ValueError("abort NO_SUPERVISED_TOKENS: update carried no eligible targets")
        for bucket, _family, _sub, window in windows:
            if any(width > bucket for width in window.row_widths):
                raise ValueError("microstep row exceeds its requested bucket")
            width = bucket
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
            rows = len(window.tokens)
            shape_receipt = certify_microstep_shape(
                bucket=bucket, sequences_global=rows, replicas=replicas,
                sequences_per_replica=per_replica_counts[bucket])
            microstep_shapes.append({
                "requested_bucket": bucket, "actual_row_widths": sorted(set(window.row_widths)),
                "sequences_global": rows, "real_tokens_global": window.real_tokens,
                "eligible_tokens_global": _predict_supervised(window),
                "physical": shape_receipt})
        end_cursor = BucketCursorState(
            BUCKET_CURSOR_SCHEMA, data["pack_manifest_sha256"],
            lanes_receipt["lanes_sha256"],
            {key: [int(value[0]), int(value[1])] for key, value in positions.items()},
            dict(fam_consumed), dict(sub_consumed), epoch,
            base_replays + len(replay_events))
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
        return live.complete or at_stop_boundary(live)

    def on_committed(live: TrainingState, checkpoint_sha: str) -> None:
        for threshold in crossed_milestones(last_boundary_tokens[0], live.cumulative_tokens,
                                            milestones):
            if threshold not in milestone_shas:
                milestone_shas[threshold] = checkpoint_sha
                store.record_milestone(threshold_tokens=threshold,
                                       checkpoint_sha256=checkpoint_sha)
                milestones_crossed.append(dict(build_milestone_receipt(
                    run_id=run_id, threshold_tokens=threshold,
                    actual_cumulative_tokens=live.cumulative_tokens,
                    global_update=live.global_update,
                    checkpoint_sha256=checkpoint_sha,
                    identity_bundle=identity_bundle)))
                if progress is not None:
                    progress(f"milestone {threshold} at update {live.global_update}")
        if live.cumulative_tokens // recovery_every > last_boundary_tokens[0] // recovery_every:
            recovery_shas.append(checkpoint_sha)
            if progress is not None:
                progress(f"recovery at update {live.global_update}")
        last_boundary_tokens[0] = live.cumulative_tokens
        keep = (set(recovery_shas[-topo["recovery_generations_retained"]:])
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
    final_cursor = final.cursor
    assert isinstance(final_cursor, BucketCursorState)

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
        "lanes_sha256": lanes_receipt["lanes_sha256"],
        "sampler_order_sha256": hashlib.sha256(_canonical_json(
            sampler_order(data["shard_hashes"], run_seed=seed, epoch=0))).hexdigest(),
        "microstep_shapes": microstep_shapes,
        "microstep_buckets": [int(entry["requested_bucket"]) for entry in microstep_shapes],
        "milestones_crossed": milestones_crossed,
        "recovery_checkpoint_count": len(recovery_shas),
        "recovery_tokens": recovery_every,
        "mixture_allocation": dict(allocation),
        "mixture_plan_sha256": mixture_plan_sha,
        "mixture_consumed": dict(final_cursor.mixture_consumed),
        "sub_consumed": dict(final_cursor.sub_consumed),
        "replay_events": list(replay_events),
        "replay_count": int(final_cursor.replay_count),
        "epoch": int(final_cursor.epoch),
        "data_manifest_sha256": data["manifest_sha256"],
        "pack_manifest_sha256": data["pack_manifest_sha256"],
        "model_spec_sha256": model_spec.sha256(),
        "identity_bundle": dict(identity_bundle),
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
                     mixture_fractions: dict[str, float] | None = None,
                     cognition_map: Mapping[str, str] | None = None,
                     allow_replay: bool = False,
                     dataset_lifecycle=None,
                     tokenizer_freeze_sha256: str | None = None,
                     ) -> dict[str, Any]:
    """One 500M-campaign training session with milestone detection,
    recovery checkpointing, heartbeat, and session receipt.

    ``recovery_interval_updates`` is accepted for operator continuity but the
    commit cadence is token-indexed (the frozen recovery threshold), never
    update-count-indexed. Multi-session: call again on a fresh runtime —
    resumes from the latest committed generation. Milestone files carry full
    self-verifying receipts, written once and never duplicated. Stops cleanly
    before deadline with a RESUMABLE receipt.
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
        milestones=milestones, recovery_tokens=recovery_tokens,
        mixture_fractions=mixture_fractions, cognition_map=cognition_map,
        allow_replay=allow_replay, dataset_lifecycle=dataset_lifecycle,
        tokenizer_freeze_sha256=tokenizer_freeze_sha256)

    for milestone in result["milestones_crossed"]:
        mpath = milestone_dir / f"milestone_{milestone['threshold_tokens']}.json"
        if not mpath.is_file():
            mpath.write_text(json.dumps(milestone, indent=2, sort_keys=True),
                             encoding="utf-8")

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


__all__ = ["ENTRY_SCHEMA", "MILESTONE_SCHEMA", "MILESTONE_TOKENS",
           "VERIFIED_COGNITION", "assign_mixture_cell",
           "build_milestone_receipt", "campaign_microstep_plan",
           "crossed_milestones", "frozen_cognition_fractions",
           "frozen_mixture_fractions", "frozen_topology", "microstep_buckets",
           "partial_microstep_plan", "plan_campaign_demand", "prepare_data",
           "resolve_cymek_sha", "run_500m_session", "run_campaign",
           "topology_sha256", "verify_milestone_receipt"]
