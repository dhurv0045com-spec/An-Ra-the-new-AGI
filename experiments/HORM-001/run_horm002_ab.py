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
from v5_model.core import initialize
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
from v5_identity import HORMONES, HormonalProjection, HormonalState
from v5_identity.attention_patch import HormonalAttentionPatch

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


def _build_batches(seed: int):
    """Deterministic packed batches (tokens, segment_ids) at tiny scale."""

    generator = torch.Generator().manual_seed(seed)
    batches = []
    for update in range(UPDATES):
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

def _run_arm(*, arm: str, seed: int, batches) -> dict[str, object]:
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
        token_budget=UPDATES * TOKENS_PER_UPDATE,
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
            patch.state.values.update(HormonalState.baseline().values)
            outcome = "success" if current.global_update % 2 == 0 else "failure"
            patch.state.appraise(outcome)
            patch.state.decay()
            applied_scale = hormonal.scale(patch.state.vector())
        else:
            applied_scale = 1.0
        report = backend.step(current, batch)
        losses.append(float(backend.last_receipt["loss"]))
        grad_norms.append(float(report.grad_norm_post_clip))
        scales.append(applied_scale)
        return report

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(Path(tmp), state.lineage_id)
        controller = RunController(target_update=UPDATES)
        controller.start()
        started = time.perf_counter()
        final_state = train(
            state=state,
            controller=controller,
            store=store,
            payload_builder=lambda s: production_payloads(backend, state=s),
            backend_step=backend_step,
            updates=UPDATES,
            checkpoint_every=UPDATES,
        )
        elapsed = time.perf_counter() - started

    return {
        "arm": arm,
        "seed": seed,
        "losses": losses,
        "scales": scales,
        "grad_norms": grad_norms,
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "cumulative_tokens": final_state.cumulative_tokens,
        "global_update": final_state.global_update,
        "elapsed_seconds": elapsed,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--replication", action="store_true",
        help="run seeds 707001-707003 and write RESULT_horm002_replication.json")
    args = parser.parse_args()

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
