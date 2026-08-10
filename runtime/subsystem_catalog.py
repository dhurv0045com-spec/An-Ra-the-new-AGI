"""Authoritative lifecycle and evidence catalog for An-Ra subsystems.

The catalog deliberately separates four different facts which older status
surfaces tended to collapse into one:

* source exists;
* the subsystem is on the canonical execution path;
* a bounded execution has succeeded; and
* a matched experiment has shown that the subsystem improves the model.

Only the final item is promotion evidence.  Importability or file existence is
never treated as proof of capability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from anra.anra_paths import ROOT

CATALOG_SCHEMA = "anra-subsystem-catalog/v1"
LIFECYCLE_STATES = frozenset({"active", "pilot", "disabled", "retired"})
EVIDENCE_STATES = frozenset({"verified", "partial", "missing", "not_applicable"})


@dataclass(frozen=True)
class SubsystemRecord:
    subsystem_id: str
    name: str
    lifecycle: str
    owner: str
    purpose: str
    repository: str
    paths: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    parameter_delta: int | None = None
    runtime_cost: str = "not measured"
    claimed_benefit: str = ""
    evidence_state: str = "missing"
    evidence: str = ""
    evidence_required: str = "matched held-out capability comparison"
    rollback: str = "disable the subsystem and restore the signed parent checkpoint"
    promotion_history: tuple[str, ...] = ()

    @property
    def promotion_eligible(self) -> bool:
        return self.lifecycle == "active" and self.evidence_state == "verified"


def subsystem_records() -> tuple[SubsystemRecord, ...]:
    """Return the single current catalog, ordered from foundation to research."""

    return (
        SubsystemRecord(
            "tokenizer_v4",
            "V4 tokenizer",
            "active",
            "data",
            "The only tokenizer accepted by new training and inference lineages.",
            "anra",
            ("tokenizer/tokenizer_v4_32k.json", "tokenizer/tokenizer_adapter.py"),
            runtime_cost="32,768-token vocabulary",
            claimed_benefit="Stable train/inference token identity.",
            evidence_state="verified",
            evidence="Pinned artifact, metadata hash, fertility and round-trip gates.",
            evidence_required="Artifact hash and round-trip contract remain valid.",
            rollback="Restore the same signed V4 artifact; never fall back to V3.",
            promotion_history=("Promoted as the sole canonical tokenizer.",),
        ),
        SubsystemRecord(
            "licensed_corpus",
            "Licensed corpus pipeline",
            "active",
            "data",
            "Builds immutable licensed, deduplicated and split training data.",
            "anra",
            ("training/corpus_manifest.py", "scripts/download_training_data.py"),
            dependencies=("tokenizer_v4",),
            runtime_cost="Offline acquisition and token packing.",
            claimed_benefit="Reproducible source-pure pretraining data.",
            evidence_state="verified",
            evidence="Signed manifests exist for train, validation and test shards.",
            evidence_required="Every consumed shard must match its signed hash and license record.",
            rollback="Reject the changed manifest and reuse the prior immutable corpus release.",
        ),
        SubsystemRecord(
            "dense_v4",
            "Dense V4 transformer",
            "active",
            "model",
            "The canonical 181,132,071-parameter language-model foundation.",
            "anra",
            ("anra_brain.py", "anra/architecture.py", "training/v2_config.py"),
            dependencies=("tokenizer_v4",),
            parameter_delta=181_132_071,
            runtime_cost="One dense 181M forward/backward path.",
            claimed_benefit="Stable general-purpose language-model substrate.",
            evidence_state="partial",
            evidence=(
                "A protected 181M V4 SFT child exact-loads all model tensors, but an "
                "August 2026 local behavior probe failed basic math, factual, and code tasks."
            ),
            evidence_required="Coherent milestone checkpoint plus held-out behavioral evaluation.",
            rollback=(
                "Restore the last protected dense checkpoint and its exact architecture contract."
            ),
        ),
        SubsystemRecord(
            "exact_resume",
            "Schema-9 exact resume",
            "active",
            "training",
            "Restores model, optimizer, scheduler, scaler, RNG and sampler cursor.",
            "anra",
            ("training/checkpoint.py", "training/v2_runtime.py"),
            dependencies=("dense_v4", "licensed_corpus"),
            runtime_cost="Checkpoint serialization and restore validation.",
            claimed_benefit="Continuation without replaying or skipping accepted token windows.",
            evidence_state="partial",
            evidence="Local optimizer-boundary termination and resume drill passed.",
            evidence_required="Cross-worker remote handoff must also pass before a cloud campaign.",
        ),
        SubsystemRecord(
            "mtp",
            "Multi-token prediction",
            "pilot",
            "model-training",
            "Adds +2/+3 future-token objectives without changing the tokenizer.",
            "anra",
            ("anra_brain.py", "scripts/build_brain.py"),
            dependencies=("dense_v4",),
            parameter_delta=1_607_424,
            runtime_cost="Two additional tied future-token projections and losses.",
            claimed_benefit="Richer predictive representations and data efficiency.",
            evidence_state="partial",
            evidence="One bounded full-model GPU step succeeded; no matched capability win exists.",
            evidence_required=(
                "Paired dense-versus-MTP continuation from the same checkpoint and windows."
            ),
        ),
        SubsystemRecord(
            "moe",
            "Sparse upcycled MoE",
            "disabled",
            "model",
            "Routes tokens through replicated feed-forward experts.",
            "anra",
            ("anra_brain.py",),
            dependencies=("dense_v4",),
            parameter_delta=941_488_128,
            runtime_cost=(
                "Current eight-expert form raises the 181M system to 1,122,620,199 parameters."
            ),
            claimed_benefit="Higher parameter capacity at sparse token compute.",
            evidence_state="partial",
            evidence=(
                "Routing code executes, but the current geometry is unsuitable for the T4 baseline."
            ),
            evidence_required=(
                "A redesigned sparse-upcycling pilot with stable load balance "
                "and useful-compute gain."
            ),
        ),
        SubsystemRecord(
            "mod",
            "Mixture of Depths routing",
            "disabled",
            "model",
            "Selects bounded token computation across transformer layers.",
            "anra",
            ("anra_brain.py",),
            dependencies=("dense_v4",),
            runtime_cost="Routing overhead with potential skipped token-layer work.",
            claimed_benefit="Adaptive compute allocation.",
            evidence_state="partial",
            evidence=(
                "Gradient and routing unit paths execute; no trained capability "
                "gain is established."
            ),
        ),
        SubsystemRecord(
            "rim_esv",
            "RIM and ESV identity modulation",
            "disabled",
            "identity",
            "Provides neutral-initialized residual identity modulation and state estimates.",
            "anra",
            ("anra_brain.py", "identity/esv.py"),
            dependencies=("dense_v4",),
            parameter_delta=1_032_405,
            runtime_cost="Small per-layer projections and state prediction.",
            claimed_benefit="Persistent, inspectable identity-conditioned computation.",
            evidence_state="partial",
            evidence=(
                "Neutral initialization and execution are covered; intelligence value is unproven."
            ),
        ),
        SubsystemRecord(
            "dstp",
            "Dynamic temperature path",
            "disabled",
            "model",
            "Applies bounded learned attention-temperature controls.",
            "anra",
            ("anra_brain.py",),
            dependencies=("dense_v4",),
            runtime_cost="Scalar controls per transformer layer.",
            claimed_benefit="Calibrated adaptive attention sharpness.",
            evidence_state="partial",
            evidence=(
                "The mechanism is wired and neutralized; no matched capability evidence exists."
            ),
        ),
        SubsystemRecord(
            "hal_external",
            "External HAL policy",
            "pilot",
            "runtime-governance",
            "Applies bounded external state and reasoning-budget policy.",
            "anra",
            ("identity/hal.py", "runtime/hal_telemetry.py"),
            dependencies=("dense_v4",),
            runtime_cost="Small policy and telemetry overhead outside the transformer.",
            claimed_benefit="Inspectable control without entangling base weights.",
            evidence_state="partial",
            evidence="External bounded policy executes; task-level benefit is not established.",
        ),
        SubsystemRecord(
            "hal_transformer",
            "Transformer-integrated HAL",
            "disabled",
            "model",
            "Modulates transformer attention from HAL state.",
            "anra",
            ("anra_brain.py",),
            dependencies=("dense_v4", "hal_external"),
            runtime_cost="Per-layer attention modulation.",
            claimed_benefit="State-aware internal computation.",
            evidence_state="missing",
            evidence="Forced off in the canonical dense profile.",
        ),
        SubsystemRecord(
            "csii_growth",
            "Cross-scale identity inheritance",
            "pilot",
            "model-growth",
            "Transfers a trained smaller model into a larger compatible child.",
            "anra",
            ("training/csii.py",),
            dependencies=("dense_v4", "exact_resume"),
            runtime_cost="One-time tensor migration followed by alignment training.",
            claimed_benefit="Reuse learned behavior when growing 181M to 500M.",
            evidence_state="partial",
            evidence=(
                "Tiny-model cosine parity exists; full hybrid-attention parity remains required."
            ),
        ),
        SubsystemRecord(
            "turboquant_kv",
            "TurboQuant KV-cache pilot",
            "pilot",
            "inference-efficiency",
            (
                "Compresses persistent inference keys and values with a "
                "device-resident randomized rotation and bit-packed scalar quantizer."
            ),
            "anra",
            (
                "inference/turboquant.py",
                "anra_brain.py",
                "generate.py",
            ),
            dependencies=("dense_v4",),
            parameter_delta=0,
            runtime_cost=(
                "Dequantizes the retained history before SDPA; no fused QJL "
                "attention kernel exists yet, so latency may regress."
            ),
            claimed_benefit=(
                "Approximately 3.76x persistent KV storage reduction versus "
                "BF16 at 4 bits for 64-dimensional heads."
            ),
            evidence_state="partial",
            evidence=(
                "Real nibble packing, physical-byte accounting, distortion "
                "telemetry, and a fail-closed generation gate are implemented."
            ),
            evidence_required=(
                "A trained V4 checkpoint must pass long-context capability, "
                "distribution-drift, peak-VRAM, and tokens/second gates."
            ),
            rollback="Select the exact float KV backend; model weights are unchanged.",
        ),
        SubsystemRecord(
            "retrieval_memory",
            "Retrieval and long-term memory",
            "active",
            "runtime-intelligence",
            "Adds provenance-grounded knowledge without modifying base weights.",
            "anra",
            ("retrieval/hybrid.py", "memory/memory_router.py"),
            dependencies=("dense_v4",),
            runtime_cost="Index lookup, reranking and context tokens.",
            claimed_benefit="Current, attributable and persistent external knowledge.",
            evidence_state="partial",
            evidence=(
                "Retrieval and isolation tests exist; trained-model end-to-end "
                "quality remains blocked."
            ),
        ),
        SubsystemRecord(
            "self_correction",
            "Verifier-guided self-correction",
            "pilot",
            "cognition",
            "Generates, verifies, revises or abstains under an explicit budget.",
            "anra",
            (
                "cognition/self_correction.py",
                "training/verified_process.py",
                "training/verifier.py",
                "inference/reasoning_budget.py",
            ),
            dependencies=("dense_v4", "retrieval_memory"),
            runtime_cost="Additional candidates, verifier calls and trace storage.",
            claimed_benefit="Detect and repair verifiable mistakes.",
            evidence_state="partial",
            evidence=(
                "Contracts and local verifier paths exist; a trained model has "
                "not passed the loop gate."
            ),
        ),
        SubsystemRecord(
            "capability_adapters",
            "Reversible LoRA/DoRA capabilities",
            "pilot",
            "continual-learning",
            "Adds hash-bound domain capabilities without overwriting base weights.",
            "anra",
            ("anra/extensions.py", "training/sparse_lora.py"),
            dependencies=("dense_v4",),
            runtime_cost="Adapter parameters and optional routing overhead.",
            claimed_benefit="Rollback-safe capability acquisition.",
            evidence_state="partial",
            evidence=(
                "Binding and reversal paths exist; useful trained adapters remain to be produced."
            ),
        ),
        SubsystemRecord(
            "post_training",
            "SFT, STaR, RLVR and DPO",
            "pilot",
            "post-training",
            "Turns a pretrained language model into a useful instruction and reasoning model.",
            "anra",
            (
                "training/posttraining_contract.py",
                "training/star.py",
                "training/rlvr.py",
                "training/dpo.py",
            ),
            dependencies=("dense_v4",),
            runtime_cost="Separate signed fine-tuning stages.",
            claimed_benefit="Instruction following and verifiable reasoning behavior.",
            evidence_state="partial",
            evidence=(
                "A signed assistant-only V4 SFT child improved validation loss from "
                "1.839 to 1.392, but it has not passed the fixed behavior gate and is "
                "therefore a research checkpoint rather than an accepted capability release."
            ),
        ),
        SubsystemRecord(
            "agents_tools",
            "Agents and tools",
            "disabled",
            "agency",
            "Plans and executes typed actions through bounded tools.",
            "anra",
            ("agents/orchestrator.py", "execution/sandbox.py"),
            dependencies=("self_correction", "retrieval_memory"),
            runtime_cost="Planning, tool execution and verification latency.",
            claimed_benefit="Reliable action beyond token generation.",
            evidence_state="partial",
            evidence="Local orchestration exists but canonical activation is operator-gated.",
        ),
        SubsystemRecord(
            "moonshots",
            "Moonshot architecture laboratory",
            "pilot",
            "research",
            "Runs isolated SSM, latent reasoning, world-model and formal-system pilots.",
            "anra",
            ("training/moonshot_pilots.py", "training/moonshot_execution.py"),
            dependencies=("dense_v4",),
            runtime_cost="Pilot-specific; never charged to canonical training by default.",
            claimed_benefit="Discover architectures that outperform current subsystems.",
            evidence_state="partial",
            evidence=(
                "Smoke paths exist; only bounded pilot evidence may promote an individual moonshot."
            ),
        ),
        SubsystemRecord(
            "multimodal_world",
            "Multimodal, robotics and world models",
            "disabled",
            "research",
            "Extends the language foundation to perception and environment interaction.",
            "anra",
            ("multimodal/projector.py", "robotics/world_model.py"),
            dependencies=("dense_v4", "self_correction"),
            runtime_cost="New encoders, data and environment interaction.",
            claimed_benefit="Grounded cross-modal prediction and action.",
            evidence_state="partial",
            evidence="Research modules execute locally; no trained multimodal lineage is accepted.",
        ),
        SubsystemRecord(
            "unified_evidence",
            "ThirdEye and Matrix evidence stream",
            "active",
            "observability",
            (
                "Feeds run-level explanations and aggregate operational status "
                "from one evidence lineage."
            ),
            "anra",
            ("evaluation/thirdeye_adapter.py", "app.py", "engine/telemetry.py"),
            dependencies=("exact_resume",),
            runtime_cost="Structured event and summary storage.",
            claimed_benefit="One auditable truth surface rather than competing monitoring systems.",
            evidence_state="verified",
            evidence=(
                "ThirdEye, Matrix, and the cluster export consume the same "
                "hash-chained anra-evidence-event/v1 contract."
            ),
        ),
        SubsystemRecord(
            "tokenizer_v3",
            "V3 tokenizer lineage",
            "retired",
            "data",
            "Historical tokenizer lineage; forbidden for new runs.",
            "anra",
            (),
            runtime_cost="None.",
            claimed_benefit="None for the V4 mainline.",
            evidence_state="not_applicable",
            evidence="Superseded by the sole V4 tokenizer.",
            evidence_required="No operational manifest may reference it.",
            rollback="No runtime rollback; inspect Git history only.",
            promotion_history=("Retired after V4 promotion.",),
        ),
        SubsystemRecord(
            "cross_colab_sparse_averaging",
            "Cross-Colab sparse weight averaging",
            "retired",
            "cluster",
            "Historical approximation that combined unrelated remote updates.",
            "gpu-cluster",
            (),
            runtime_cost="None; execution must remain unavailable.",
            claimed_benefit="Superseded by exact checkpoint-baton continuation.",
            evidence_state="not_applicable",
            evidence="Retired because it was not exact distributed training.",
            evidence_required="No operational route or manifest may enable it.",
            rollback="Use Git history for audit; do not reactivate.",
            promotion_history=("Retired in favor of one canonical checkpoint writer.",),
        ),
    )


def validate_subsystem_catalog(
    records: tuple[SubsystemRecord, ...] | None = None,
    *,
    root: Path = ROOT,
) -> list[str]:
    """Return validation errors; callers decide whether to fail or report."""

    rows = records or subsystem_records()
    errors: list[str] = []
    identifiers = [row.subsystem_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        errors.append("subsystem ids must be unique")
    known = set(identifiers)
    for row in rows:
        if row.lifecycle not in LIFECYCLE_STATES:
            errors.append(f"{row.subsystem_id}: invalid lifecycle {row.lifecycle!r}")
        if row.evidence_state not in EVIDENCE_STATES:
            errors.append(f"{row.subsystem_id}: invalid evidence state {row.evidence_state!r}")
        unknown = sorted(set(row.dependencies) - known)
        if unknown:
            errors.append(f"{row.subsystem_id}: unknown dependencies {unknown}")
        if row.repository == "anra" and row.lifecycle != "retired":
            missing = [path for path in row.paths if not (root / path).exists()]
            if missing:
                errors.append(f"{row.subsystem_id}: missing paths {missing}")
        if row.lifecycle == "active" and not row.rollback:
            errors.append(f"{row.subsystem_id}: active subsystem has no rollback")
    return errors


def build_subsystem_catalog(*, root: Path = ROOT) -> dict[str, object]:
    records = subsystem_records()
    errors = validate_subsystem_catalog(records, root=root)
    return {
        "schema": CATALOG_SCHEMA,
        "valid": not errors,
        "errors": errors,
        "records": [
            {**asdict(row), "promotion_eligible": row.promotion_eligible} for row in records
        ],
    }
