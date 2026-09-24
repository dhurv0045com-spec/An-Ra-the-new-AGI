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
By default, local CPU/CUDA execution runs the GLOBAL logical microstep on one
device (a mathematical emulation, never TPU evidence). An opt-in host-only
distributed campaign uses the production sampler to shard the rows and a
replica collective to combine gradients. The production XLA campaign remains
blocked until the backend and exact-resume path pass TPU qualification.
An explicitly separate ``execution="xla-development"`` path is limited to
one or two updates on a discovered TPU and is marked unqualified; it is a
development integration check, not a production campaign or Kaggle evidence.
Resume is inferred from a valid committed LATEST, never from supplied paths.
Fail-closed on: identity drift, ledger drift, bucket violation, mixture
shortfall, nonfinite loss/gradients, parameter non-mutation, stale writer,
restore mismatch, missing production contamination commitment.
"""

from __future__ import annotations

from dataclasses import replace
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
from v5_training.persistent_store import (
    DURABLE_STATUS,
    materialize_local,
    mirror_checkpoint,
    read_mirror_pointer,
)
from v5_training.production_backend import (
    ProductionTrainingBackend,
    capture_evidence,
    precision_receipt,
    production_payloads,
    production_shared_payloads,
    restore_production,
)
from v5_training.production_resume import (
    ProductionRankCapture,
    build_production_distributed_payloads,
    capture_cpu_rng_state,
    next_microstep_fingerprint_sha256,
    restore_production_rank_into_backend,
    terminal_microstep_fingerprint_sha256,
)
from v5_training.runner import RunController
from v5_training.schedule import lr_at, schedule_receipt
from v5_training.state import (
    IDENTITY_SCHEMA,
    IdentityBindings,
    TrainingState,
)
from v5_training.production_microsteps import (
    count_eligible_targets,
    count_rank_real_tokens,
)
from v5_training.production_sampler import (
    VERIFIED_COGNITION,
    ProductionSampler,
    assign_mixture_cell,
    campaign_microstep_plan,
    microstep_buckets,
    partial_microstep_plan,
    plan_campaign_demand,
)
from v5_training.topology_map import certify_microstep_shape
from v5_training.trainer import train
from v5_training.xla_adapter import (
    EVIDENCE_REQUIRED,
    PENDING_STATUS,
    XLAReplicatedBackend,
    require_frozen_topology,
    xla_status,
)

ENTRY_SCHEMA = "anra-v5-production-entry-receipt/v1"
MILESTONE_SCHEMA = "anra-v5-milestone-receipt/v1"
# 500M-campaign milestone thresholds. Distinct from the 5B-run schedule in
# training_spec; these gate the 500M campaign only.
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

    return count_eligible_targets(window, bos_id=bos_id, pad_id=pad_id)


def _runtime_name(device: Any) -> str:
    if device is None:
        return "cpu"
    return str(getattr(device, "type", device))


def prepare_data(*, documents: list[dict[str, Any]], tokenizer: Any,
                 run_id: str, seed: int,
                 data_identity: str | None = None,
                 contamination_benchmarks: dict[str, str] | None = None,
                 mixture_families: tuple[str, ...] | None = None,
                 cognition_map: Mapping[str, str] | None = None,
                 development_mode: bool = True,
                 ) -> dict[str, Any]:
    """Documents -> data manifest -> packing -> sampler order (data half).

    Deterministic in (documents, seed, mixture config). ``data_identity``
    optionally stabilizes the data manifest and split salt across matched
    training runs; ``run_id`` remains a lineage/output label. ``mixture_families``
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
    identity = data_identity or run_id
    if not identity or any(character.isspace() for character in identity):
        raise ValueError("data_identity must be a compact nonempty identity")
    manifest, manifest_audit = build_data_manifest(
        records, manifest_id=f"{identity}-data",
        tokenizer_sha256=artifact_sha,
        filter_version="production-entry-filter/v1",
        dedup_version=("exact+near-minhash-lsh/v1" if not development_mode
                       else "exact-clusters/v1"),
        split_salt=f"{identity}/v1",
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
                 source_tree_sha256: str | None = None,
                 execution: str = "local",
                 mirror_root: str | Path | None = None,
                 checkpoint_coordinator: Any | None = None,
                 replica_collective: Callable[[Any], str] | None = None,
                 ) -> dict[str, Any]:
    """Execute (or resume) a token-targeted certified production campaign.

    Fresh vs resume comes from a valid committed LATEST, never from supplied
    paths. Production mode requires a frozen contamination commitment, the
    mixture fractions, and the tokenizer freeze SHA; development mode opts
    out explicitly and the receipt is labeled DEVELOPMENT. ``stop_gate``
    must be deterministic in state. ``allow_replay`` permits bounded epoch
    replay on lane exhaustion (default fails closed DATA_NOT_READY).
    ``dataset_lifecycle``, when provided, must already be RUNNABLE.
    ``execution="xla"`` currently fails before preprocessing. The
    coordinator implements per-update action agreement and rank-zero
    publication as a tested primitive, but the runtime adapter has not passed
    target qualification. ``ProductionTrainingBackend`` accepts XLA only with
    its explicit unqualified-development opt-in; that bounded path has not
    passed an end-to-end Kaggle test. Exact per-rank XLA RNG/cursor restore
    remains unqualified. XLA canaries do not satisfy those production gates.
    ``execution="xla-development"`` is a separate, explicitly unqualified
    one/two-update TPU integration path. It is not invoked by the Kaggle
    qualification notebook by default and must not be used as research-run
    evidence. Its final checkpoint-equivalence check is deferred to a fresh
    worker group so it does not allocate a second full model and Adam state on
    every TPU rank; the development receipt reports that deferral explicitly.
    ``mirror_root``, when provided, mirrors every committed generation to
    durable storage and recovers a missing local store from the mirror.
    ``checkpoint_coordinator`` opts into v2 replicated checkpoints for a
    separately qualified CPU replica test/runtime. It does not bypass the XLA
    production gate; multi-rank local execution must supply a replica-wide
    gradient collective that returns an auditable SHA-256 receipt. That
    collective requires the checkpoint coordinator so local-update failures,
    loss contributions, and update decisions can be agreed rank-symmetrically.
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
    if execution not in ("local", "xla", "xla-development"):
        raise ValueError("execution must be 'local', 'xla', or 'xla-development'")
    xla_adapter = None
    xla_status_doc: dict[str, object] | None = None
    xla_development = execution == "xla-development"
    if xla_development:
        if not development_mode:
            raise ValueError("xla-development requires development_mode=True")
        if (type(max_updates) is not int or not 1 <= max_updates <= 2):
            raise ValueError("xla-development requires max_updates bounded to one or two")
        if checkpoint_coordinator is not None or replica_collective is not None:
            raise ValueError(
                "xla-development owns its XLA coordinator and gradient collective"
            )
        if mirror_root is not None:
            raise ValueError("xla-development does not qualify durable mirror behavior")
        if _runtime_name(device) != "xla":
            raise ValueError("xla-development requires the live XLA device")
        xla_status_doc = xla_status()
        if xla_status_doc.get("status") != PENDING_STATUS:
            raise ValueError(
                "xla-development requires an initialized XLA runtime: "
                f"{xla_status_doc.get('reason', EVIDENCE_REQUIRED)}"
            )
        if str(xla_status_doc.get("device_type", "")).upper() != "TPU":
            raise ValueError("xla-development is restricted to a live TPU runtime")
    if execution == "xla":
        xla_status_doc = xla_status()
        raise ValueError(
            "XLA production execution is blocked before data preparation: "
            "the production backend and rank-coordinated checkpoint/resume "
            "path are not target-qualified; "
            f"adapter status={xla_status_doc.get('status')}, "
            f"reason={xla_status_doc.get('reason', 'target evidence is still pending')} "
            f"({EVIDENCE_REQUIRED})")
    if replica_collective is not None and checkpoint_coordinator is None:
        raise ValueError(
            "replica gradient collective requires a checkpoint coordinator for "
            "rank-sharded data and shared update decisions"
        )
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
    if xla_development:
        require_frozen_topology(int(xla_status_doc["world_size"]), replicas=replicas)
    per_replica_counts = topo["sequences_per_replica_by_bucket"]
    recovery_every = recovery_tokens if recovery_tokens is not None else topo["recovery_threshold_tokens"]
    if recovery_every <= 0:
        raise ValueError("recovery cadence must be positive")
    cymek_sha = resolve_cymek_sha(cymek_sha)
    from signac_100m.source_identity import build_source_identity
    live_source_tree_sha256 = build_source_identity(
        Path(__file__).resolve().parents[1]
    )["source_tree_sha256"]
    if source_tree_sha256 is not None and source_tree_sha256 != live_source_tree_sha256:
        raise ValueError(
            "supplied source tree identity does not match the live Signac source bundle"
        )
    source_tree_sha256 = live_source_tree_sha256
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
    cognition_fractions = frozen_cognition_fractions() if cognition_mapped else None
    sub_scheduler = (DeficitScheduler(fractions=dict(cognition_fractions))
                     if cognition_mapped else None)
    sampler = ProductionSampler(
        packed=packed,
        run_seed=seed,
        topology=topo,
        pack_manifest_sha256=data["pack_manifest_sha256"],
        cell_of_source=cell_of_source,
        mixture_fractions=mixture_fractions,
        cognition_fractions=cognition_fractions,
        cognition_mapped=cognition_mapped,
        allow_replay=allow_replay,
    )
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
    replicated_topology = f"signac-v2-world{replicas}-{topology_digest}"
    if checkpoint_coordinator is not None:
        if execution != "local":
            raise ValueError("v2 campaign checkpoints are unavailable for unqualified XLA runtime")
        if (getattr(checkpoint_coordinator, "world_size", None) != replicas
                or type(getattr(checkpoint_coordinator, "rank", None)) is not int
                or not 0 <= checkpoint_coordinator.rank < replicas):
            raise ValueError("checkpoint coordinator world differs from frozen campaign topology")
        if _runtime_name(device) != "cpu" or torch.cuda.is_available():
            raise ValueError("v2 production resume currently supports CPU RNG only")
        if replicas > 1 and not callable(replica_collective):
            raise ValueError("multi-rank local training requires a replica gradient collective")
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
             "sampler_spec_sha256": sampler.sha256,
             "source_tree_sha256": source_tree_sha256,
             "seed": seed})).hexdigest(),
        optimizer_spec_sha256=hashlib.sha256(_canonical_json(
            {"optimizer": "AdamW", "beta1": topo["beta1"], "beta2": topo["beta2"],
             "epsilon": topo["epsilon"],
             "weight_decay": topo["weight_decay"]})).hexdigest(),
        schedule_spec_sha256=schedule_doc["sha256"],
        curriculum_spec_sha256=hashlib.sha256(_canonical_json(
            {"curriculum_phase": "500m-campaign",
             "policy": "uniform-supercycle-interleave"})).hexdigest(),
        sampler_spec_sha256=sampler.sha256,
        source_tree_sha256=source_tree_sha256)
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
        "sampler_spec_sha256": sampler.sha256,
        "source_tree_sha256": source_tree_sha256,
        "run_spec_sha256": identities.run_spec_sha256,
    }

    store = CheckpointStore(Path(store_root), run_id)
    runtime = _runtime_name(device)
    precision = precision_receipt(runtime=runtime, torch_module=torch)
    if xla_development:
        precision = {
            "schema": "anra-v5-precision-contract/v1",
            "runtime": "xla",
            "persistent_parameters": "float32",
            "compute": "bfloat16-autocast",
            "logits_loss_reductions": "float32",
            "global_gradient_norm": "float32 replica-global",
            "optimizer_moments": "float32",
            "persistent_bfloat16_shadow": False,
            "loss_scaler": None,
            "status": "UNQUALIFIED_DEVELOPMENT_ONLY",
            "evidence_required": EVIDENCE_REQUIRED,
        }
    elif precision.get("status") != "CERTIFIED_LOCAL":
        raise ValueError(
            f"runtime {runtime!r} is not certified for local execution: XLA/TPU "
            "collectives, memory fit, and bf16 behavior need PRE500M "
            "certification (TPU_EVIDENCE_REQUIRED)")
    campaign_coordinator = checkpoint_coordinator
    model = None
    optimizer = None
    backend = None
    if xla_development:
        # Create one replicated backend and coordinator before inspecting
        # LATEST so fresh and resumed ranks use the same v2 control path.
        torch.manual_seed(seed)
        model = initialize(model_spec, seed).to(device)
        optimizer = build_adamw_optimizer(model, torch_module=torch)
        backend = ProductionTrainingBackend(
            model=model, optimizer=optimizer, bos_id=2, pad_id=0, device=device,
            schedule=lr_at, bfloat16_autocast=True,
            torch_module=torch, allow_unqualified_xla=True)
        xla_adapter = XLAReplicatedBackend(
            replica_backend=backend, replicas=replicas,
            world_size=int(xla_status_doc["world_size"]),
            torch_module=torch)
        campaign_coordinator = xla_adapter.checkpoint_coordinator()
    latest = store.latest_sha256()
    mirrored_recovery = False
    if latest is None and mirror_root is not None:
        mirror_head = read_mirror_pointer(mirror_root, lineage_id=run_id)
        if mirror_head is not None:
            latest = materialize_local(mirror_root, store)
            mirrored_recovery = True
    if latest is None:
        if backend is None:
            torch.manual_seed(seed)
            model = initialize(model_spec, seed).to(device)
            optimizer = build_adamw_optimizer(model, torch_module=torch)
            backend = ProductionTrainingBackend(
                model=model, optimizer=optimizer, bos_id=2, pad_id=0, device=device,
                schedule=lr_at,
                bfloat16_autocast=runtime == "cuda",
                torch_module=torch)
        if execution == "xla":
            require_frozen_topology(int(xla_status_doc["world_size"]), replicas=replicas)
            xla_adapter = XLAReplicatedBackend(
                replica_backend=backend, replicas=replicas,
                world_size=int(xla_status_doc["world_size"]),
                torch_module=torch)
        fresh_state = True
        resumed, done_updates = False, 0
        start_tokens = 0
        epoch, replay_events = 0, []
        fam_consumed = {}
        sub_consumed = {}
    else:
        if campaign_coordinator is None:
            state, payloads = store.restore()
        else:
            state, restored_metadata, _rank_payloads, _shared_payloads = (
                store.restore_distributed_artifacts(
                    rank=campaign_coordinator.rank,
                    expected_world_size=replicas,
                    expected_topology=replicated_topology,
                    checkpoint_sha256=latest,
                )
            )
            campaign_coordinator.restore_rank_progress(
                cumulative_tokens=(
                    restored_metadata.ranks[campaign_coordinator.rank].cumulative_tokens
                ),
            )
            payloads = None
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
                ("sampler spec", identities.sampler_spec_sha256,
                 state.identities.sampler_spec_sha256),
                ("optimizer spec", identities.optimizer_spec_sha256, state.identities.optimizer_spec_sha256),
                ("schedule spec", identities.schedule_spec_sha256, state.identities.schedule_spec_sha256),
                ("curriculum spec", identities.curriculum_spec_sha256, state.identities.curriculum_spec_sha256),
                ("source commit", identities.source_commit, state.identities.source_commit),
                ("source tree", identities.source_tree_sha256,
                 state.identities.source_tree_sha256),
                ("token budget", campaign_tokens, state.token_budget)):
            if expected != actual:
                raise ValueError(
                    f"resume identity drift: {name} changed since the committed checkpoint")
        if backend is None:
            torch.manual_seed(seed)
            model = initialize(model_spec, seed).to(device)
            optimizer = build_adamw_optimizer(model, torch_module=torch)
            backend = ProductionTrainingBackend(
                model=model, optimizer=optimizer, bos_id=2, pad_id=0, device=device,
                schedule=lr_at,
                bfloat16_autocast=runtime == "cuda",
                torch_module=torch)
        if campaign_coordinator is None:
            restore_production(backend, payloads=payloads)
        else:
            rank_resume = restore_production_rank_into_backend(
                backend=backend,
                store=store,
                rank=campaign_coordinator.rank,
                expected_world_size=replicas,
                expected_topology=replicated_topology,
                sampler=sampler,
                checkpoint_sha256=latest,
                runtime="xla" if xla_development else "cpu",
                rng_state_adapter=xla_adapter if xla_development else None,
            )
            state = rank_resume.state
        if execution == "xla":
            require_frozen_topology(int(xla_status_doc["world_size"]), replicas=replicas)
            xla_adapter = XLAReplicatedBackend(
                replica_backend=backend, replicas=replicas,
                world_size=int(xla_status_doc["world_size"]),
                torch_module=torch)
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
                    "execution_mode": execution,
                    "xla_status": dict(xla_status_doc) if xla_status_doc is not None else {
                        "status": "LOCAL_EMULATION",
                        "note": "single-device global-tensor execution; not TPU evidence"},
                    "mirrored_recovery": mirrored_recovery,
                    "mirrored_generations": 0,
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
    mirrored_shas: list[str] = []
    last_boundary_tokens = [state.cumulative_tokens]
    updates_this_session = [0]
    stop_reasons: list[str] = []
    local_collective_receipts: list[str] = []
    latest_xla_rng_state: list[bytes] = []
    def backend_step(current: TrainingState):
        nonlocal lanes_receipt
        def prepare_local_update():
            nonlocal lanes_receipt
            target_rank = (
                xla_adapter.ordinal() if xla_adapter is not None
                else (campaign_coordinator.rank
                      if campaign_coordinator is not None else None)
            )
            plan = sampler.materialize_update(current, rank=target_rank)
            lanes_receipt = dict(plan.lanes_receipt)
            for event in plan.replay_events:
                replay_events.append(dict(event))
                if progress is not None:
                    progress(f"epoch replay {event['epoch']}")
            ctx = backend.begin_update(current)
            for (bucket, _family, _subfamily, window), microstep in zip(
                plan.windows, plan.rank_microsteps,
            ):
                if any(width > bucket for width in window.row_widths):
                    raise ValueError("microstep row exceeds its requested bucket")
                rank_rows: list[int] = []
                # SPMD-correct sharding: every rank runs this same program over
                # the same windows, but accumulates ONLY its own rank shard with
                # the GLOBAL denominator. The ONE SUM collective happens once at
                # the accumulation boundary (after the final microstep below),
                # reuniting the exact global mean before the single clip/step;
                # reducing inside the microstep loop would all-reduce the
                # ACCUMULATED buffer repeatedly and multiply early microstep
                # gradients by powers of the replica count. Ledgers are identical
                # on all ranks, so training states cannot diverge. Correct under
                # the frozen dropout-free contract (no rank RNG is consumed in
                # the compute path); any stochastic op would need rank-aware RNG
                # (PRE500M detail). Rank 0 alone writes checkpoints.
                tokens = torch.tensor(microstep.tokens,
                                      dtype=torch.long, device=device)
                segment_ids = torch.tensor(microstep.segment_ids,
                                           dtype=torch.long, device=device)
                eligible = torch.tensor(microstep.eligible,
                                        dtype=torch.bool, device=device)
                ctx = backend.accumulate_microstep(
                    ctx, tokens=tokens, segment_ids=segment_ids, eligible=eligible,
                    tokens_by_source=dict(microstep.tokens_by_source),
                    planned_total=microstep.planned_total,
                    segment_layout_prevalidated=True)
                rank_rows.append(len(microstep.tokens))
                rows = len(window.tokens)
                shape_receipt = certify_microstep_shape(
                    bucket=bucket, sequences_global=rows, replicas=replicas,
                    sequences_per_replica=per_replica_counts[bucket])
                local_eligible_tokens = count_eligible_targets(microstep)
                shard_rows_global = (
                    len(microstep.tokens) * replicas
                    if target_rank is not None else len(microstep.tokens)
                )
                microstep_shapes.append({
                    "requested_bucket": bucket,
                    "actual_row_widths": sorted(set(window.row_widths)),
                    "sequences_global": rows, "real_tokens_global": window.real_tokens,
                    "eligible_tokens_global": _predict_supervised(window),
                    "execution": (
                        "xla-sharded" if xla_adapter is not None
                        else "host-sharded" if target_rank is not None
                        else "local-global"
                    ),
                    "executing_rank_rows": list(rank_rows),
                    "executing_rank_eligible_tokens": local_eligible_tokens,
                    "replica_padding_rows_added": max(0, shard_rows_global - rows),
                    "physical": shape_receipt})
            expected_local_tokens = sum(
                count_eligible_targets(microstep) for microstep in plan.rank_microsteps
            )
            backend.validate_local_supervision(
                ctx,
                expected_local_tokens,
                allow_zero=(xla_adapter is not None or campaign_coordinator is not None),
            )
            return plan, ctx, expected_local_tokens

        if xla_adapter is None:
            if campaign_coordinator is None:
                plan, ctx, expected_local_tokens = prepare_local_update()
            else:
                prepared = None
                local_error = None
                try:
                    prepared = prepare_local_update()
                except Exception as exc:
                    local_error = f"{type(exc).__name__}: {str(exc)[:500]}"
                # In the CPU replica harness, every rank must vote on local
                # preparation before a healthy rank can enter the gradient
                # collective. The trainer performs a second update-result vote
                # after optimizer/evidence work before it advances state.
                campaign_coordinator.agree_update_result(
                    state=current,
                    checkpoint_requested=False,
                    stop_requested=False,
                    local_error=local_error,
                )
                if prepared is None:
                    raise RuntimeError(
                        "distributed local-update vote passed without a prepared update"
                    )
                plan, ctx, expected_local_tokens = prepared
        else:
            # A rank-local preparation/forward/backward failure must join this
            # status vote before healthy ranks can enter gradient reduction.
            plan, ctx, expected_local_tokens = xla_adapter.run_local_stage(
                stage="local_update", callback=prepare_local_update,
            )
        loss_aggregate = None
        if campaign_coordinator is not None:
            loss_aggregate = campaign_coordinator.aggregate_update_loss(
                global_update=current.global_update + 1,
                global_tokens=plan.eligible_tokens,
                local_eligible_tokens=sum(ctx["eligible_counts"]),
                local_loss_numerator=sum(ctx["loss_numerators"]),
            )
        if xla_adapter is not None:
            # Accumulation boundary: ONE gradient SUM collective for the
            # whole logical update, after every microstep has contributed
            # locally. finish_update then applies the ONE global clip and
            # the ONE optimizer step. Reducing earlier (per microstep)
            # would re-reduce already-accumulated gradients and scale
            # microstep i's contribution by replicas**(microsteps - i).
            xla_adapter.all_reduce_sum_gradients(backend.model)
            collective_sha = xla_adapter.last_collective_receipt_sha256
            if (not isinstance(collective_sha, str) or len(collective_sha) != 64
                    or any(char not in "0123456789abcdef" for char in collective_sha)):
                raise ValueError("XLA gradient collective did not produce a SHA-256 receipt")
            local_collective_receipts.append(collective_sha)
        elif replica_collective is not None:
            collective_sha = replica_collective(backend.model)
            if (not isinstance(collective_sha, str) or len(collective_sha) != 64
                    or any(char not in "0123456789abcdef" for char in collective_sha)):
                raise ValueError("replica gradient collective must return a lowercase SHA-256")
            local_collective_receipts.append(collective_sha)
        elif campaign_coordinator is not None:
            local_collective_receipts.append(hashlib.sha256(_canonical_json({
                "schema": "signac-single-replica-update/v1",
                "rank": campaign_coordinator.rank,
                "world_size": replicas,
                "global_update": current.global_update + 1,
            })).hexdigest())
        report = backend.finish_update(
            current,
            ctx,
            planned_total=plan.eligible_tokens,
            expected_local_tokens=expected_local_tokens,
            allow_zero_local_tokens=(
                xla_adapter is not None or campaign_coordinator is not None
            ),
            loss_aggregate=loss_aggregate,
            loss_rank=(campaign_coordinator.rank
                       if campaign_coordinator is not None else None),
            cursor=plan.end_cursor,
        )
        if xla_adapter is not None:
            # Hash and retain the exact rank-local RNG payload that a checkpoint
            # at this update boundary will serialize and restore.
            rng_payload = xla_adapter.capture_rng_state(torch_module=torch)
            rng_sha256 = hashlib.sha256(rng_payload).hexdigest()
            latest_xla_rng_state[:] = [rng_payload]
            if backend.last_receipt is not None:
                backend.last_receipt["rng_state_sha256"] = rng_sha256
                backend.last_receipt["rng_state_payload_sha256"] = rng_sha256
            report = replace(report, rng_state_sha256=rng_sha256)
        report = replace(
            report,
            local_real_tokens=count_rank_real_tokens(plan.rank_microsteps),
        )
        updates_this_session[0] += 1
        receipt = backend.last_receipt
        assert receipt is not None
        losses.append(float(receipt["loss"]))
        return report

    def payload_builder(live: TrainingState) -> dict[str, bytes]:
        return production_payloads(backend, state=live)

    def rank_capture_builder(live: TrainingState) -> ProductionRankCapture:
        assert campaign_coordinator is not None
        shared = production_shared_payloads(backend)
        rank = campaign_coordinator.rank
        if live.complete:
            next_sha = terminal_microstep_fingerprint_sha256(
                rank=rank,
                world_size=replicas,
                topology=replicated_topology,
                state=live,
            )
        else:
            next_plan = sampler.materialize_update(live, rank=rank)
            next_sha = next_microstep_fingerprint_sha256(
                rank=rank,
                world_size=replicas,
                topology=replicated_topology,
                checkpoint_global_update=live.global_update,
                cursor=live.cursor,
                microsteps=next_plan.rank_microsteps,
            )
        collective_receipt = (
            local_collective_receipts[-1]
            if local_collective_receipts else hashlib.sha256(_canonical_json({
                "schema": "signac-single-replica-update/v1",
                "rank": rank,
                "world_size": replicas,
                "global_update": live.global_update,
            })).hexdigest()
        )
        return ProductionRankCapture(
            rank=rank,
            cumulative_tokens=campaign_coordinator.rank_cumulative_tokens,
            rng_state=(latest_xla_rng_state[-1]
                       if xla_adapter is not None and latest_xla_rng_state
                       else capture_cpu_rng_state(torch)),
            cursor=live.cursor,
            next_microsteps=(),
            collective_receipt_sha256=collective_receipt,
            model_state_sha256=hashlib.sha256(shared["model.bin"]).hexdigest(),
            optimizer_state_sha256=hashlib.sha256(shared["optimizer.bin"]).hexdigest(),
            next_microstep_sha256=next_sha,
        )

    def distributed_payload_builder(
        live: TrainingState, rank_captures,
    ) -> dict[str, bytes]:
        shared = production_shared_payloads(backend)
        return build_production_distributed_payloads(
            state=live,
            topology=replicated_topology,
            model_payload=shared["model.bin"],
            optimizer_payload=shared["optimizer.bin"],
            scheduler_payload=shared["scheduler.json"],
            rank_captures=rank_captures,
            sampler=sampler,
        )

    def should_checkpoint(live: TrainingState) -> bool:
        if xla_development:
            # Keep this explicitly bounded lane restartable after every update.
            return True
        if crossed_milestones(last_boundary_tokens[0], live.cumulative_tokens,
                              milestones):
            return True
        if live.cumulative_tokens // recovery_every > last_boundary_tokens[0] // recovery_every:
            return True
        return live.complete

    def on_committed(live: TrainingState, checkpoint_sha: str) -> None:
        for threshold in crossed_milestones(last_boundary_tokens[0], live.cumulative_tokens,
                                            milestones):
            store.record_milestone(threshold_tokens=threshold,
                                   checkpoint_sha256=checkpoint_sha)
            if progress is not None:
                progress(f"milestone {threshold} at update {live.global_update}")
        is_recovery = (live.cumulative_tokens // recovery_every
                       > last_boundary_tokens[0] // recovery_every)
        pending_recoveries = recovery_shas + ([checkpoint_sha] if is_recovery else [])
        if is_recovery and progress is not None:
            progress(f"recovery at update {live.global_update}")
        keep = (set(pending_recoveries[-topo["recovery_generations_retained"]:])
                | {checkpoint_sha})
        store.prune(keep=keep)
        if mirror_root is not None:
            mirror_receipt = mirror_checkpoint(
                store, mirror_root, checkpoint_sha256=checkpoint_sha)
            if mirror_receipt.get("status") != DURABLE_STATUS:
                raise ValueError("durable mirror did not confirm persistence")
            if progress is not None:
                progress(f"mirrored {checkpoint_sha[:8]}")

    def on_checkpoint_observed(live: TrainingState, checkpoint_sha: str) -> None:
        # The commit callback's shared-storage mutations execute on rank zero.
        # Build the same volatile receipt state on every rank from the shared
        # committed state/SHA so process-local campaign reports agree.
        for threshold in crossed_milestones(last_boundary_tokens[0], live.cumulative_tokens,
                                            milestones):
            milestones_crossed.append(dict(build_milestone_receipt(
                run_id=run_id, threshold_tokens=threshold,
                actual_cumulative_tokens=live.cumulative_tokens,
                global_update=live.global_update,
                checkpoint_sha256=checkpoint_sha,
                identity_bundle=identity_bundle)))
        if live.cumulative_tokens // recovery_every > last_boundary_tokens[0] // recovery_every:
            recovery_shas.append(checkpoint_sha)
        if mirror_root is not None:
            mirrored_shas.append(checkpoint_sha)
        last_boundary_tokens[0] = live.cumulative_tokens

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
        on_checkpoint_observed=on_checkpoint_observed,
        should_stop=should_stop,
        resume_parent_sha256=latest if resumed else None,
        checkpoint_coordinator=campaign_coordinator,
        rank_capture_builder=(rank_capture_builder
                              if campaign_coordinator is not None else None),
        distributed_payload_builder=(distributed_payload_builder
                                     if campaign_coordinator is not None else None))
    if final.complete:
        termination = "COMPLETE"
    elif stop_reasons:
        termination = stop_reasons[-1]
    elif capped:
        termination = "MANUAL_BOUNDARY"
    else:
        raise RuntimeError("campaign ended without completion, boundary, or stop")
    wall = clock() - t0

    if xla_development:
        # A second M102+Adam replica on every TPU rank can exceed device
        # headroom during the final receipt check. The Kaggle development
        # wrapper verifies continuation by launching a fresh worker group
        # against this committed checkpoint, so defer equality to that path.
        resume_equal: bool | None = None
        resume_verification = "DEFERRED_TO_FRESH_WORKER_GROUP"
    else:
        fresh_model = initialize(model_spec, seed).to(device)
        fresh_optimizer = build_adamw_optimizer(fresh_model, torch_module=torch)
        fresh_backend = ProductionTrainingBackend(
            model=fresh_model, optimizer=fresh_optimizer, bos_id=2, pad_id=0,
            device=device, schedule=lr_at,
            bfloat16_autocast=(runtime in {"cuda", "xla"}),
            allow_unqualified_xla=False,
            torch_module=torch)
        if campaign_coordinator is None:
            _, restored_payloads = store.restore()
            restore_production(fresh_backend, payloads=restored_payloads)
        else:
            restore_production_rank_into_backend(
                backend=fresh_backend,
                store=store,
                rank=campaign_coordinator.rank,
                expected_world_size=replicas,
                expected_topology=replicated_topology,
                sampler=sampler,
                runtime="cpu",
            )
        live = capture_evidence(backend.model, backend.optimizer, torch=torch)
        resumed_evidence = capture_evidence(fresh_model, fresh_optimizer, torch=torch)
        resume_equal = (live.parameter_sha256 == resumed_evidence.parameter_sha256
                        and live.moment_sha256 == resumed_evidence.moment_sha256
                        and live.optimizer_steps == resumed_evidence.optimizer_steps)
        if not resume_equal:
            raise RuntimeError("campaign restore did not reproduce live state")
        resume_verification = "IN_PROCESS_FRESH_BACKEND_EQUAL"
    last_update = backend.last_receipt
    assert last_update is not None
    final_cursor = final.cursor
    assert isinstance(final_cursor, BucketCursorState)

    return {
        "schema": ENTRY_SCHEMA, "run_id": run_id, "seed": seed,
        "mode": mode_label, "campaign_tokens": campaign_tokens,
        "execution_mode": execution,
        "xla_status": dict(xla_status_doc) if xla_status_doc is not None else {
            "status": "LOCAL_EMULATION",
            "note": "single-device global-tensor execution; not TPU evidence"},
        "mirrored_recovery": mirrored_recovery,
        "mirrored_generations": len(mirrored_shas),
        "updates_executed": int(final.global_update),
        "cumulative_tokens": int(final.cumulative_tokens),
        "state_complete": bool(final.complete),
        "termination": termination,
        "resumed": resumed,
        "losses": losses, "wall_seconds": round(wall, 3),
        "resume_equal": resume_equal,
        "resume_verification": resume_verification,
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
            "loss_scope": str(last_update["loss_scope"]),
            "loss_numerator": float(last_update["loss_numerator"]),
            "loss_denominator": int(last_update["loss_denominator"]),
            "loss_aggregation_sha256": (
                last_update["loss_aggregation"].get("receipt_sha256")
                if isinstance(last_update.get("loss_aggregation"), Mapping) else None
            ),
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
                     mirror_root: str | Path | None = None,
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
        tokenizer_freeze_sha256=tokenizer_freeze_sha256,
        mirror_root=mirror_root)

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
