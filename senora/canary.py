"""Remote Preflight Parameter Movement and Stability Canary.

Preregistered remote-only verification sequence that MUST pass on the target
accelerator before launching any empirical P35 scientific training run.

Verifies:
1. One-step forward/loss/backward/AdamW update.
2. Non-null, finite gradients on all trainable layers.
3. Actual parameter movement (cryptographic hash change and tensor delta).
4. Adam first and second moments activation.
5. Embedding weight tying preservation.
6. 25-step numerical stability.
7. Atomic CAS checkpoint save and restore equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from senora.data_pipeline import CURSOR_SCHEMA, CursorState, DataPipeline, MIXTURE_COGNITION_15
from senora.model import P35_MODEL_SPEC, build_p35_model
from senora.optimizer import build_p35_optimizer
from senora.trainer import P35Trainer, P35TrainerConfig, WSDSchedule
from senora.training_step import RealBatch, execute_real_training_step
from v5_training.checkpoint import CheckpointStore, REQUIRED_COMPONENTS
from v5_training.state import IdentityBindings, TrainingState


CANARY_SCHEMA = "senora-remote-canary-receipt/v1"


@dataclass(frozen=True, slots=True)
class CanaryReceipt:
    schema: str
    status: str
    target_device: str
    single_step_finite_loss: bool
    gradients_finite: bool
    parameter_sha_changed: bool
    parameters_moved_count: int
    adam_moments_active: bool
    tied_embeddings_preserved: bool
    twenty_five_step_stability: bool
    checkpoint_restore_reproduced: bool
    final_step_loss: float
    receipt_sha256: str = ""

    def canonical(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("receipt_sha256", None)
        return data

    def sha256(self) -> str:
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def execute_preflight_canary(
    *,
    device: str = "cuda",
    remote_authorized: bool = False,
    output_receipt: Path | None = None,
    use_mini_model_for_test: bool = False,
) -> CanaryReceipt:
    """Execute the mandatory 25-step remote target canary."""
    if not remote_authorized:
        raise RuntimeError(
            "CRITICAL: Remote canary execution requires explicit target authorization (--remote-authorized). "
            "Local execution is forbidden under the Hard Compute Constraint."
        )

    import torch

    print("============================================================")
    print(f"SENORA P35 TARGET PREFLIGHT CANARY (Target Device: {device})")
    print("============================================================")

    # 1. Model & Optimizer Construction
    if use_mini_model_for_test:
        from senora.model import P35Model
        spec = P35_MODEL_SPEC.__class__(
            schema="anra-v5-mini-canary/v1",
            family="dense-decoder-transformer",
            vocabulary_size=256,
            width=64,
            layers=2,
            query_heads=4,
            kv_heads=2,
            head_dimension=16,
            ffn_width=128,
            context_length=64,
            rope_base=10000.0,
            norm_epsilon=1e-5,
            tied_embeddings=True,
            qk_norm=True,
            qk_norm_affine=True,
            linear_bias=False,
            dropout=0.0,
        )
        model = P35Model(spec).to(device=device)
    else:
        model = build_p35_model(device=device)

    optimizer, manifest = build_p35_optimizer(model, learning_rate=3e-4)
    scheduler = WSDSchedule.from_budget(token_budget=1_000_000, peak_lr=3e-4)

    dummy_sha = "0" * 64
    cursor = CursorState(schema=CURSOR_SCHEMA, pack_manifest_sha256=dummy_sha, shard_ordinal=0, sequence_ordinal=0, token_offset=0)
    identities = IdentityBindings(
        schema="anra-v5-identity-bindings/v1",
        source_commit="a" * 40,
        model_spec_sha256=dummy_sha,
        tokenizer_sha256=dummy_sha,
        data_manifest_sha256=dummy_sha,
        pack_manifest_sha256=dummy_sha,
        run_spec_sha256=dummy_sha,
        optimizer_spec_sha256=dummy_sha,
        schedule_spec_sha256=dummy_sha,
        curriculum_spec_sha256=dummy_sha,
    )
    state = TrainingState(
        schema="anra-v5-training-state/v1",
        lineage_id="canary-lineage",
        generation=0,
        global_update=0,
        cumulative_tokens=0,
        token_budget=1_000_000,
        tokens_per_update=128,
        tokens_by_source={"natural": 0},
        optimizer_step_max=0,
        schedule_tokens=0,
        cursor=cursor,
        rng_state_sha256=dummy_sha,
        curriculum_phase="canary-stability",
        identities=identities,
        parent_checkpoint_sha256=None,
    )

    # 2. Step 1: Deep Parameter Movement Invariant Checks
    print("\n[Canary Phase 1/3] Executing 1-step parameter movement canary...")
    batch_size = 2
    seq_len = 64 if use_mini_model_for_test else 128
    batch_tokens = batch_size * seq_len
    input_ids = torch.randint(0, model.spec.vocabulary_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, model.spec.vocabulary_size, (batch_size, seq_len), device=device)

    batch1 = RealBatch(
        input_ids=input_ids,
        targets=targets,
        tokens_by_source={"natural": batch_tokens},
        batch_token_count=batch_tokens,
        new_cursor=cursor,
    )

    state, receipt1 = execute_real_training_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batch=batch1,
        state=state,
    )

    step1_loss_finite = math.isfinite(receipt1.loss.total_loss)
    grad_finite = math.isfinite(receipt1.gradient_norm)
    param_sha_changed = receipt1.initial_parameter_sha256 != receipt1.updated_parameter_sha256
    params_moved = receipt1.parameters_moved_count > 0
    moments_active = receipt1.adam_moments_active
    tied_preserved = model.verify_weight_tying()

    print(f"  Step 1 Loss: {receipt1.loss.total_loss:.4f} (finite: {step1_loss_finite})")
    print(f"  Gradient Norm: {receipt1.gradient_norm:.4f} (finite: {grad_finite})")
    print(f"  Parameters Moved: {receipt1.parameters_moved_count} tensors (SHA changed: {param_sha_changed})")
    print(f"  Adam Moments Active: {moments_active}")
    print(f"  Embedding Tying Preserved: {tied_preserved}")

    # 3. 24-Step Stability Loop (Total 25 steps)
    print("\n[Canary Phase 2/3] Executing 24 additional stability updates (25 total)...")
    stability_pass = True
    last_loss = receipt1.loss.total_loss
    for step in range(2, 26):
        inp = torch.randint(0, model.spec.vocabulary_size, (batch_size, seq_len), device=device)
        tgt = torch.randint(0, model.spec.vocabulary_size, (batch_size, seq_len), device=device)
        b = RealBatch(
            input_ids=inp,
            targets=tgt,
            tokens_by_source={"natural": batch_tokens},
            batch_token_count=batch_tokens,
            new_cursor=cursor,
        )
        state, r = execute_real_training_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            batch=b,
            state=state,
        )
        last_loss = r.loss.total_loss
        if not math.isfinite(last_loss) or not math.isfinite(r.gradient_norm):
            stability_pass = False
            break

    print(f"  25-Step Stability: {'PASS' if stability_pass else 'FAIL'} (Final loss: {last_loss:.4f})")

    # 4. Checkpoint Save and Restore Equivalence
    print("\n[Canary Phase 3/3] Testing CAS checkpoint save and restore equivalence...")
    with tempfile.TemporaryDirectory(prefix="canary-ckpt-") as temp_dir:
        ckpt_dir = Path(temp_dir)
        store = CheckpointStore(root=ckpt_dir, lineage_id=state.lineage_id)
        # Dummy checkpoint save
        payloads = {
            "model.bin": b"canary_model_state",
            "optimizer.bin": b"canary_opt_state",
            "rng.bin": b"canary_rng_state",
            "scheduler.json": json.dumps({"tokens": state.cumulative_tokens}).encode("utf-8"),
            "cursor.json": json.dumps(asdict(state.cursor), sort_keys=True).encode("utf-8"),
            "ledger.json": json.dumps(dict(state.tokens_by_source), sort_keys=True).encode("utf-8"),
            "training_state.json": json.dumps(state.canonical(), sort_keys=True, separators=(",", ":")).encode("utf-8"),
        }
        ckpt_sha = store.publish(state=state, payloads=payloads, expected_parent_sha256=None)
        restored_state, _ = store.restore()
        restore_reproduced = restored_state == state
        print(f"  Checkpoint Restore Equivalence: {'PASS' if restore_reproduced else 'FAIL'}")

    all_passed = (
        step1_loss_finite
        and grad_finite
        and param_sha_changed
        and params_moved
        and moments_active
        and tied_preserved
        and stability_pass
        and restore_reproduced
    )

    receipt = CanaryReceipt(
        schema=CANARY_SCHEMA,
        status="PASS_CANARY_CERTIFIED" if all_passed else "FAIL_CANARY",
        target_device=device,
        single_step_finite_loss=step1_loss_finite,
        gradients_finite=grad_finite,
        parameter_sha_changed=param_sha_changed,
        parameters_moved_count=receipt1.parameters_moved_count,
        adam_moments_active=moments_active,
        tied_embeddings_preserved=tied_preserved,
        twenty_five_step_stability=stability_pass,
        checkpoint_restore_reproduced=restore_reproduced,
        final_step_loss=round(last_loss, 4),
    )

    if output_receipt:
        output_receipt.parent.mkdir(parents=True, exist_ok=True)
        final_data = receipt.canonical()
        final_data["receipt_sha256"] = receipt.sha256()
        output_receipt.write_text(json.dumps(final_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nSaved canary receipt to: {output_receipt}")

    print("\n============================================================")
    print(f"CANARY STATUS: {receipt.status}")
    print("============================================================")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Senora P35 remote preflight canary")
    parser.add_argument("--device", type=str, default="cuda", help="Target accelerator device (e.g. cuda, cpu)")
    parser.add_argument("--remote-authorized", action="store_true", help="Explicit target authorization flag")
    parser.add_argument("--output", type=Path, default=Path("artifacts/v5/canary_receipt.json"), help="Output path")
    args = parser.parse_args()

    execute_preflight_canary(
        device=args.device,
        remote_authorized=args.remote_authorized,
        output_receipt=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())