"""HORM-002: matched A/B mini training run through the real production path.

Control:      hormonal projection inert (raw_alpha=0)  -> attention scale 1.0.
Treatment:    identical seed/data/spec/backend; a hormonal projection scales
              attention queries by 1 + B*tanh(alpha * w . h) after RoPE+QK-norm.

Preregistered hypothesis (HORM-002 in PLAN.md):
  a bounded attention-scale modulation initialized near-zero produces a
  measurable but bounded loss-trajectory difference between matched arms on a
  tiny CPU run, with control arm byte-identical to the historical no-op path.

This is evidence at miniature scale ONLY. It makes no capability or
production-scale claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize, packed_layout
from v5_training.optimizer import build_adamw_optimizer
from v5_training.production_backend import (
    ProductionTrainingBackend,
    bounded_warmup_schedule,
)
from v5_training.runner import RunController
from v5_training.trainer import train
from v5_training.checkpoint import CheckpointStore
from v5_training.production_backend import production_payloads
from v5_training.streaming import batch_from_window
from v5_training.state import (
    CURSOR_SCHEMA,
    IDENTITY_SCHEMA,
    CursorState,
    IdentityBindings,
    TrainingState,
)
from v5_identity import HORMONES, HormonalProjection, HormonalState, appraise_committed
from v5_identity.attention_patch import HormonalAttentionPatch
from v5_tokenizer.artifact import sha256_file

SCHEMA = "anra-horm-002-ab/v1"
SEED = 707_001
PEAK_LEARNING_RATE = 3e-4
UPDATES = 8
TOKENS_PER_UPDATE = 256
SEQUENCES_PER_UPDATE = 8
SEQ_LEN = 32
VOCAB = 512

HORM_SPEC = ModelSpec(
    schema="anra-v5-model-spec/v1",
    family="dense-decoder-transformer",
    vocabulary_size=VOCAB,
    width=32,
    layers=2,
    query_heads=2,
    kv_heads=1,
    head_dimension=16,
    ffn_width=64,
    context_length=SEQ_LEN,
    rope_base=10_000.0,
    norm_epsilon=1e-5,
    tied_embeddings=True,
    qk_norm=True,
    qk_norm_affine=True,
    linear_bias=False,
    dropout=0.0,
)


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pack_manifest(documents: list[list[int]]) -> str:
    return _hash_bytes(json.dumps(documents, sort_keys=True).encode("utf-8"))


def _build_batches(seed: int, updates: int = UPDATES):
    """Deterministic packed batches (tokens, segment_ids) at tiny scale."""

    generator = torch.Generator().manual_seed(seed)
    batches = []
    for update in range(updates):
        tokens = torch.randint(4, VOCAB, (SEQUENCES_PER_UPDATE, SEQ_LEN), generator=generator)
        # reserve 0 (PAD) never used; BOS/EOS/PAD excluded from loss by causal_lm_loss
        segment_ids = torch.zeros((SEQUENCES_PER_UPDATE, SEQ_LEN), dtype=torch.int32)
        # two segments per sequence for pack realism
        segment_ids[:, SEQ_LEN // 2 :] = 1
        tokens_by_source = {
            "natural": int((segment_ids == 0).sum()),
            "verified_cognition": int((segment_ids == 1).sum()),
        }
        batches.append((tokens, segment_ids, tokens_by_source))
    return batches

HORM004_SEEDS = [707021, 707022, 707023, 707024, 707025]
HORM004_UPDATES = 64

# Fixed synthetic probes in the miniature vocab-512 world: constant prompt
# token ids and gold next-tokens, identical for every seed and arm, so the
# appraisal *inputs* are fully matched and only the evolving model differs.
HORM004_PROBES = [
    {"task_id": "horm004-probe-0", "prompt": [4, 5, 6, 7, 8, 9, 10, 11], "gold": 12},
    {"task_id": "horm004-probe-1", "prompt": [20, 21, 22, 23, 24, 25, 26, 27], "gold": 28},
    {"task_id": "horm004-probe-2", "prompt": [100, 101, 102, 103, 104, 105, 106, 107], "gold": 108},
    {"task_id": "horm004-probe-3", "prompt": [200, 201, 202, 203, 204, 205, 206, 207], "gold": 208},
]


def _score_live_probes(model, probes) -> list[object]:
    """Run fixed probes through the live model and the real gold firewall.

    Single-step prediction (argmax of last-position logits) per probe;
    outputs committed, joined to evaluator truth, and scored through
    ``score_committed``. Returns scored results for appraisal. The model is
    briefly switched to eval mode (dropout is 0.0 regardless) and restored.
    """

    import torch as torch_module
    from v5_evaluation.firewall import (
        CommittedOutput,
        build_evaluator_truth,
        build_visible_tasks,
        score_committed,
    )

    records = [
        {"task_id": probe["task_id"],
         "prompt": f"synthetic miniature probe {index}",
         "gold": str(probe["gold"])}
        for index, probe in enumerate(probes)
    ]
    visible = {task.task_id: task for task in build_visible_tasks(records)}
    truth = {item.task_id: item for item in build_evaluator_truth(records)}
    was_training = model.training
    model.eval()
    try:
        scored = []
        with torch_module.no_grad():
            for probe in probes:
                tokens = torch_module.tensor([probe["prompt"]], dtype=torch_module.long)
                segment_ids = torch_module.zeros((1, len(probe["prompt"])), dtype=torch_module.int32)
                positions, mask = packed_layout(segment_ids, torch_module=torch_module)
                logits = model(tokens, positions, mask)
                predicted = int(logits[0, -1].argmax().item())
                committed = CommittedOutput(
                    task_id=probe["task_id"], output=str(predicted),
                    candidate_scores=None)
                scored.append(score_committed(
                    committed, visible[probe["task_id"]], truth[probe["task_id"]]))
    finally:
        if was_training:
            model.train()
    return scored


def _run_arm(*, arm: str, seed: int, batches, appraisal_mode: str = "synthetic",
             probes: list | None = None, updates: int = UPDATES) -> dict[str, object]:
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = initialize(HORM_SPEC, seed=seed)
    optimizer = build_adamw_optimizer(model, lr=PEAK_LEARNING_RATE)
    schedule = bounded_warmup_schedule(peak_learning_rate=PEAK_LEARNING_RATE, warmup_tokens=256)

    hormonal = None
    if arm == "treatment":
        hormonal = HormonalProjection(
            weights=tuple(0.1 if name in ("cortisol", "adrenaline") else 0.05 for name in HORMONES),
            bound=0.2,
            raw_alpha=0.35,
        )
    patch = HormonalAttentionPatch(model, projection=hormonal)

    pack_manifest_sha256 = _pack_manifest([list(b[0].flatten().tolist()) for b in batches])
    identities = IdentityBindings(
        schema=IDENTITY_SCHEMA,
        source_commit="0" * 40,
        model_spec_sha256=HORM_SPEC.sha256(),
        tokenizer_sha256=_hash_bytes(b"horm002-tokenizer"),
        data_manifest_sha256=_hash_bytes(b"horm002-data"),
        pack_manifest_sha256=_hash_bytes(b"horm002-pack"),
        run_spec_sha256=_hash_bytes(f"horm002-{arm}".encode()),
        optimizer_spec_sha256=_hash_bytes(b"adamw-fp32-master/v1"),
        schedule_spec_sha256=_hash_bytes(b"bounded-warmup/v1"),
        curriculum_spec_sha256=_hash_bytes(b"horm002"),
    )
    state = TrainingState.initial(
        lineage_id=f"horm002-{arm}",
        token_budget=updates * TOKENS_PER_UPDATE,
        tokens_per_update=TOKENS_PER_UPDATE,
        cursor=CursorState(CURSOR_SCHEMA, identities.pack_manifest_sha256, 0, 0, 0),
        rng_state_sha256="0" * 64,
        curriculum_phase="horm002",
        identities=identities,
    )
    backend = ProductionTrainingBackend(
        model=model,
        optimizer=optimizer,
        bos_id=2,
        pad_id=0,
        device=None,
        schedule=schedule,
        activation_checkpointing=False,
    )

    losses: list[float] = []
    scales: list[float] = []
    grad_norms: list[float] = []
    outcomes: list[list[str]] = []

    def backend_step(current: TrainingState):
        tokens, segment_ids, tokens_by_source = batches[current.global_update]
        from v5_training.production_backend import PackedBatch
        from v5_training.state import CursorState as CS

        positions = None  # production backend computes packed layout itself
        batch = PackedBatch(
            tokens=tokens,
            segment_ids=segment_ids,
            tokens_by_source=tokens_by_source,
            cursor=CS(
                CURSOR_SCHEMA,
                identities.pack_manifest_sha256,
                current.global_update,
                current.global_update,
                int(tokens.numel()),
            ),
            rng_state_sha256=_hash_bytes(f"{arm}-{current.global_update}".encode()),
        )
        if hormonal is not None:
            if appraisal_mode == "live":
                if probes is None:
                    raise ValueError("live appraisal requires probes")
                step_outcomes = [
                    appraise_committed(patch.state, scored)
                    for scored in _score_live_probes(model, probes)
                ]
                patch.state.decay()
            else:
                patch.state.values.update(HormonalState.baseline().values)
                outcome = "success" if current.global_update % 2 == 0 else "failure"
                patch.state.appraise(outcome)
                patch.state.decay()
                step_outcomes = [outcome]
            applied_scale = hormonal.scale(patch.state.vector())
        else:
            applied_scale = 1.0
            step_outcomes = []
        report = backend.step(current, batch)
        losses.append(float(backend.last_receipt["loss"]))
        grad_norms.append(float(report.grad_norm_post_clip))
        scales.append(applied_scale)
        outcomes.append(step_outcomes)
        return report

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(Path(tmp), state.lineage_id)
        controller = RunController(target_update=updates)
        controller.start()
        started = time.perf_counter()
        final_state = train(
            state=state,
            controller=controller,
            store=store,
            payload_builder=lambda s: production_payloads(backend, state=s),
            backend_step=backend_step,
            updates=updates,
            checkpoint_every=updates,
        )
        elapsed = time.perf_counter() - started

    return {
        "arm": arm,
        "seed": seed,
        "appraisal_mode": appraisal_mode,
        "losses": losses,
        "scales": scales,
        "grad_norms": grad_norms,
        "outcomes": outcomes,
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "cumulative_tokens": final_state.cumulative_tokens,
        "global_update": final_state.global_update,
        "elapsed_seconds": elapsed,
    }

def _run_horm003(output_dir: Path, *, force: bool = False) -> None:
    """Preregistered HORM-003: 5 fresh seeds, provenance-bound, refuse overwrite."""

    import platform
    import sys

    torch.set_num_threads(2)
    if torch.cuda.is_available():
        raise SystemExit(
            "HORM-003 is CPU-only by preregistration; CUDA is available so refusing "
            "to run. Set CUDA_VISIBLE_DEVICES='' to force CPU.")
    seeds = [707011, 707012, 707013, 707014, 707015]
    out = output_dir / "RESULT_horm003_prospective.json"
    if out.exists() and not force:
        raise SystemExit(
            f"refusing to overwrite existing result: {out}; pass --force to replace")
    if out.exists() and force:
        backup = out.with_suffix(".json.previous")
        backup.write_bytes(out.read_bytes())

    pairs = []
    invalid_reasons = []
    for seed in seeds:
        batches = _build_batches(seed=seed)
        control = _run_arm(arm="control", seed=seed, batches=batches)
        treatment = _run_arm(arm="treatment", seed=seed, batches=batches)
        all_losses = [*control["losses"], *treatment["losses"]]
        finite = all(value == value and abs(value) != float("inf") for value in all_losses)
        scales_ok = all(
            0.8 <= scale <= 1.2
            for scale in [*control["scales"], *treatment["scales"]]
        )
        if not finite:
            invalid_reasons.append(f"seed {seed}: non-finite loss")
        if not scales_ok:
            invalid_reasons.append(f"seed {seed}: scale out of [0.8, 1.2]")
        difference = [abs(a - b) for a, b in zip(control["losses"], treatment["losses"])]
        pairs.append({
            "seed": seed,
            "control_losses": control["losses"],
            "treatment_losses": treatment["losses"],
            "control_final_loss": control["final_loss"],
            "treatment_final_loss": treatment["final_loss"],
            "mean_loss_difference": treatment["mean_loss"] - control["mean_loss"],
            "max_abs_loss_difference": max(difference),
            "all_finite": finite,
            "scales_in_bounds": scales_ok,
            "control_scales": control["scales"],
            "treatment_scales": treatment["scales"],
        })

    same_sign = 0
    nonzero = [pair for pair in pairs if pair["mean_loss_difference"] != 0.0]
    if len(nonzero) >= 2:
        majority_sign = 1 if sum(
            1 for pair in nonzero if pair["mean_loss_difference"] > 0) >= len(nonzero) / 2 else -1
        consistent = sum(
            1 for pair in nonzero
            if (1 if pair["mean_loss_difference"] > 0 else -1) == majority_sign)
    else:
        majority_sign = 0
        consistent = 0
    median_difference = sorted(pair["mean_loss_difference"] for pair in pairs)[len(pairs) // 2]
    verdict = (
        "INVALID" if invalid_reasons
        else "SUPPORTED" if consistent >= 4 and majority_sign != 0
        else "NOT_SUPPORTED")
    result = {
        "schema": "anra-horm-003-prospective/v1",
        "preregistration": "experiments/HORM-001/PLAN.md HORM-003 section "
                           "(preregistered 2026-09-18T00:05:10+05:30)",
        "seeds": seeds,
        "pairs": pairs,
        "sign_consistent_seeds": consistent,
        "majority_sign": majority_sign,
        "median_mean_loss_difference": median_difference,
        "verdict": verdict,
        "invalid_reasons": invalid_reasons,
        "updates": UPDATES,
        "tokens_per_update": TOKENS_PER_UPDATE,
        "spec_sha256": HORM_SPEC.sha256(),
        "provenance": {
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "source_file_sha256": {
                path: sha256_file(Path(path))
                for path in (
                    "v5_identity/attention_patch.py",
                    "v5_identity/hormonal_projection.py",
                    "v5_identity/hormonal_state.py",
                )
            },
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "threads": 2,
            "runtime": "cpu",
        },
        "claim_level": "miniature-scale-prospective",
        "honesty_note": (
            "5-seed prospective A/B with corrected state binding. Synthetic "
            "appraisal fixtures, not live verifier outcomes. SUPPORTED means "
            "only sign-consistent loss-trajectory difference; it is not a "
            "capability or quality claim."),
    }
    payload = json.dumps(result, sort_keys=True, indent=2).encode("utf-8")
    result["sha256"] = _hash_bytes(payload)
    out.write_text(json.dumps(result, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "sign_consistent_seeds": consistent,
        "majority_sign": majority_sign,
        "median_mean_loss_difference": median_difference,
        "seeds": seeds,
    }, indent=2))
    print("wrote", out)


def _run_horm004(output_dir: Path, *, force: bool = False) -> None:
    """Preregistered HORM-004: 5 fresh seeds x 64 updates, live appraisal."""

    torch.set_num_threads(2)
    if torch.cuda.is_available():
        raise SystemExit(
            "HORM-004 is CPU-only by preregistration; CUDA is available so refusing "
            "to run. Set CUDA_VISIBLE_DEVICES='' to force CPU.")
    out = output_dir / "RESULT_horm004_prospective.json"
    if out.exists() and not force:
        raise SystemExit(
            f"refusing to overwrite existing result: {out}; pass --force to replace")
    if out.exists() and force:
        backup = out.with_suffix(".json.previous")
        backup.write_bytes(out.read_bytes())

    pairs = []
    invalid_reasons = []
    for seed in HORM004_SEEDS:
        batches = _build_batches(seed, HORM004_UPDATES)
        control = _run_arm(arm="control", seed=seed, batches=batches,
                           updates=HORM004_UPDATES)
        treatment = _run_arm(arm="treatment", seed=seed, batches=batches,
                             appraisal_mode="live", probes=HORM004_PROBES,
                             updates=HORM004_UPDATES)
        all_losses = [*control["losses"], *treatment["losses"]]
        finite = all(value == value and abs(value) != float("inf") for value in all_losses)
        scales_ok = all(
            0.8 <= scale <= 1.2
            for scale in [*control["scales"], *treatment["scales"]]
        )
        if not finite:
            invalid_reasons.append(f"seed {seed}: non-finite loss")
        if not scales_ok:
            invalid_reasons.append(f"seed {seed}: scale out of [0.8, 1.2]")
        flat_outcomes = [label for step in treatment["outcomes"] for label in step]
        success_fraction = (
            sum(1 for label in flat_outcomes if label == "success") / len(flat_outcomes)
            if flat_outcomes else 0.0)
        difference = [abs(a - b) for a, b in zip(control["losses"], treatment["losses"])]
        pairs.append({
            "seed": seed,
            "control_final_loss": control["final_loss"],
            "treatment_final_loss": treatment["final_loss"],
            "mean_loss_difference": treatment["mean_loss"] - control["mean_loss"],
            "max_abs_loss_difference": max(difference),
            "all_finite": finite,
            "scales_in_bounds": scales_ok,
            "treatment_success_fraction": success_fraction,
            "treatment_distinct_scales": len(set(treatment["scales"])),
        })

    nonzero = [pair for pair in pairs if pair["mean_loss_difference"] != 0.0]
    if len(nonzero) >= 2:
        majority_sign = 1 if sum(
            1 for pair in nonzero if pair["mean_loss_difference"] > 0) >= len(nonzero) / 2 else -1
        consistent = sum(
            1 for pair in nonzero
            if (1 if pair["mean_loss_difference"] > 0 else -1) == majority_sign)
    else:
        majority_sign = 0
        consistent = 0
    median_difference = sorted(pair["mean_loss_difference"] for pair in pairs)[len(pairs) // 2]
    verdict = (
        "INVALID" if invalid_reasons
        else "SUPPORTED" if consistent >= 4 and majority_sign != 0
        else "NOT_SUPPORTED")
    result = {
        "schema": "anra-horm-004-prospective/v1",
        "preregistration": "experiments/HORM-001/PLAN.md HORM-004 section",
        "seeds": HORM004_SEEDS,
        "updates": HORM004_UPDATES,
        "tokens_per_update": TOKENS_PER_UPDATE,
        "probes": HORM004_PROBES,
        "appraisal": "live firewall-scored single-step probes; per-probe "
                     "appraisal, one decay per update; persistent state",
        "pairs": pairs,
        "sign_consistent_seeds": consistent,
        "majority_sign": majority_sign,
        "median_mean_loss_difference": median_difference,
        "verdict": verdict,
        "invalid_reasons": invalid_reasons,
        "spec_sha256": HORM_SPEC.sha256(),
        "provenance": {
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "source_file_sha256": {
                path: sha256_file(Path(path))
                for path in (
                    "v5_identity/attention_patch.py",
                    "v5_identity/appraisal.py",
                    "v5_identity/hormonal_projection.py",
                    "v5_identity/hormonal_state.py",
                )
            },
            "torch_version": torch.__version__,
            "runtime": "cpu",
            "threads": 2,
        },
        "claim_level": "miniature-scale-prospective",
        "honesty_note": (
            "5-seed prospective A/B with corrected state binding and live "
            "firewall appraisal. Probes are fixed synthetic token sequences "
            "in the miniature vocab-512 world, not capability probes. "
            "SUPPORTED means only sign-consistent loss-trajectory difference; "
            "it is not a capability or quality claim."),
    }
    payload = json.dumps(result, sort_keys=True, indent=2).encode("utf-8")
    result["sha256"] = _hash_bytes(payload)
    out.write_text(json.dumps(result, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "sign_consistent_seeds": consistent,
        "majority_sign": majority_sign,
        "median_mean_loss_difference": median_difference,
        "seeds": HORM004_SEEDS,
    }, indent=2))
    print("wrote", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--replication", action="store_true",
        help="run seeds 707001-707003 and write RESULT_horm002_replication.json")
    parser.add_argument(
        "--horm003", action="store_true",
        help="preregistered HORM-003: seeds 707011-707015, provenance-bound, no overwrite")
    parser.add_argument(
        "--horm004", action="store_true",
        help="preregistered HORM-004: seeds 707021-707025 x 64 updates, live firewall appraisal")
    parser.add_argument(
        "--force", action="store_true",
        help="prospective runs only: overwrite an existing result (rejected by default)")
    args = parser.parse_args()

    modes = [args.horm003, args.horm004, args.replication]
    if sum(1 for mode in modes if mode) > 1:
        raise SystemExit("choose at most one of --horm003, --horm004, --replication")
    if args.horm003:
        _run_horm003(args.output, force=args.force)
        return
    if args.horm004:
        _run_horm004(args.output, force=args.force)
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    if not args.replication:
        batches = _build_batches(seed=args.seed)
        control = _run_arm(arm="control", seed=args.seed, batches=batches)
        treatment = _run_arm(arm="treatment", seed=args.seed, batches=batches)

        identical = control["losses"] == treatment["losses"]
        difference = [abs(a - b) for a, b in zip(control["losses"], treatment["losses"])]
        result = {
            "schema": SCHEMA,
            "seed": args.seed,
            "updates": UPDATES,
            "tokens_per_update": TOKENS_PER_UPDATE,
            "spec_sha256": HORM_SPEC.sha256(),
            "control": control,
            "treatment": treatment,
            "control_losses_identical_to_treatment": identical,
            "max_abs_loss_difference": max(difference),
            "claim_level": "miniature-scale-only",
            "honesty_note": (
                "tiny CPU A/B; measures loss-trajectory difference of a bounded "
                "attention-scale hormonal modulation at miniature scale. No "
                "capability, G90, or production-scale claim."
            ),
        }
        payload = json.dumps(result, sort_keys=True, indent=2).encode("utf-8")
        result["sha256"] = _hash_bytes(payload)
        out = args.output / "RESULT_horm002_ab.json"
        out.write_text(json.dumps(result, sort_keys=True, indent=2), encoding="utf-8")
        print(json.dumps({k: result[k] for k in (
            "schema", "seed", "max_abs_loss_difference",
            "control_losses_identical_to_treatment")}, indent=2))
        print("control.final_loss  :", control["final_loss"])
        print("treatment.final_loss:", treatment["final_loss"])
        print("wrote", out)
        return

    seeds = [707001, 707002, 707003]
    pairs = []
    for seed in seeds:
        batches = _build_batches(seed=seed)
        control = _run_arm(arm="control", seed=seed, batches=batches)
        treatment = _run_arm(arm="treatment", seed=seed, batches=batches)
        difference = [abs(a - b) for a, b in zip(control["losses"], treatment["losses"])]
        pairs.append({
            "seed": seed,
            "control_final_loss": control["final_loss"],
            "treatment_final_loss": treatment["final_loss"],
            "mean_loss_difference": treatment["mean_loss"] - control["mean_loss"],
            "max_abs_loss_difference": max(difference),
            "all_finite": all(
                value == value and abs(value) != float("inf")
                for value in [*control["losses"], *treatment["losses"]]
            ),
        })
    signs = {1 if pair["mean_loss_difference"] > 0 else
             (-1 if pair["mean_loss_difference"] < 0 else 0) for pair in pairs}
    replication = {
        "schema": "anra-horm-002-replication/v1",
        "seeds": seeds,
        "pairs": pairs,
        "consistent_sign_across_seeds": len(signs) == 1 and 0 not in signs,
        "consistent_sign_verdict": (
            "CONSISTENT" if len(signs) == 1 and 0 not in signs
            else "INCONSISTENT_OR_NOISE_DOMINATED"),
        "claim_level": "miniature-scale-replication",
        "honesty_note": (
            "3-seed replication at miniature scale; sign consistency across "
            "seeds is the primary readout. Still no capability or "
            "production-scale claim."),
    }
    payload = json.dumps(replication, sort_keys=True, indent=2).encode("utf-8")
    replication["sha256"] = _hash_bytes(payload)
    out = args.output / "RESULT_horm002_replication.json"
    out.write_text(json.dumps(replication, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({
        "seeds": seeds,
        "mean_loss_differences": [pair["mean_loss_difference"] for pair in pairs],
        "consistent_sign": replication["consistent_sign_across_seeds"],
    }, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
