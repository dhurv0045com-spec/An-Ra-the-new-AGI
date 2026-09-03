"""Bounded P35 canary through the canonical production training path.

Constructs the exact frozen P35 recipe (16 layers, width 384, 6 query heads,
3 KV heads, FFN 1024) as a real V5 ModelSpec, binds real packed data through
the miniature data path, and runs certified production updates on the target
device (CUDA preferred) with bf16 compute autocast and fp32 reductions.  The
canary records peak VRAM, step time, parameter/moment mutation hashes, and a
checkpoint/restore equivalence proof, all through the same
``ProductionTrainingBackend``/``trainer.train`` path the miniature uses.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize
from v5_training.checkpoint import CheckpointStore
from v5_training.miniature import (
    _load_corpus,
    _load_tokenizer,
    _source_commit,
)
from v5_training.optimizer import build_adamw_optimizer, optimizer_group_receipt
from v5_training.production_backend import (
    PackedBatch,
    ProductionTrainingBackend,
    bounded_warmup_schedule,
    capture_evidence,
    production_payloads,
    restore_production,
)
from v5_training.runner import RunController
from v5_training.streaming import batch_from_window
from v5_data.stream import build_update_stream
from v5_training.state import (
    CURSOR_SCHEMA,
    IDENTITY_SCHEMA,
    CursorState,
    IdentityBindings,
    TrainingState,
)
from v5_training.trainer import train


SCHEMA = "anra-v5-p35-production-canary/v1"
SEED = 47_031
PEAK_LEARNING_RATE = 3e-4
UPDATES = 3

P35_SPEC = ModelSpec(
    schema="anra-v5-model-spec/v1",
    family="dense-decoder-transformer",
    vocabulary_size=24_576,
    width=384,
    layers=16,
    query_heads=6,
    kv_heads=3,
    head_dimension=64,
    ffn_width=1_024,
    context_length=4_096,
    rope_base=10_000.0,
    norm_epsilon=1e-5,
    tied_embeddings=True,
    qk_norm=True,
    qk_norm_affine=True,
    linear_bias=False,
    dropout=0.0,
)


def run_canary(*, repo_root: Path, device_name: str = "cuda", output: Path | None = None) -> dict:
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    repo = repo_root.resolve()
    from v5_data.manifest import build_data_manifest, manifest_sha256
    from v5_data.pack import pack_documents
    from v5_data.stream import build_update_stream
    from v5_training.miniature import SPLITS

    tokenizer, _ = _load_tokenizer(repo)
    documents = _load_corpus(repo, tokenizer)
    manifest, manifest_audit = build_data_manifest(
        documents,
        manifest_id="anra-v5-p35-production-canary",
        tokenizer_sha256=tokenizer.identity.artifact_sha256,
        filter_version="miniature-filter/v1",
        dedup_version="exact-clusters/v1",
        split_salt="anra-v5-miniature/v1",
        split_boundaries=SPLITS,
        count_tokens=lambda text: len(tokenizer.encode(text)),
        contamination_benchmarks={},
    )
    training_ids = {
        record.source_id for record in manifest.sources if record.split == "training"
    }
    packed, pack_audit = pack_documents(
        [
            (document.doc_id, tokenizer.encode(document.text), document.family)
            for document in documents
            if document.doc_id in training_ids
        ],
        bos=2,
        eos=3,
        pad=0,
        sequences_per_shard=4,
    )
    windows = build_update_stream(packed, run_seed=SEED)
    if len(windows) < UPDATES:
        raise ValueError(
            f"P35 canary stream has {len(windows)} exact 4096-token updates; {UPDATES} needed"
        )
    windows = windows[:UPDATES]

    import hashlib

    pack_manifest_sha256 = hashlib.sha256(
        json.dumps([json.loads(shard.payload_bytes()) for shard in packed], sort_keys=True).encode()
    ).hexdigest()
    identities = IdentityBindings(
        schema=IDENTITY_SCHEMA,
        source_commit=_source_commit(repo),
        model_spec_sha256=P35_SPEC.sha256(),
        tokenizer_sha256=tokenizer.identity.artifact_sha256,
        data_manifest_sha256=manifest_sha256(manifest),
        pack_manifest_sha256=pack_manifest_sha256,
        run_spec_sha256=hashlib.sha256(b"p35-production-canary-run/v1").hexdigest(),
        optimizer_spec_sha256=hashlib.sha256(b"adamw-fp32-master/v1").hexdigest(),
        schedule_spec_sha256=hashlib.sha256(b"bounded-warmup/v1").hexdigest(),
        curriculum_spec_sha256=hashlib.sha256(b"canary").hexdigest(),
    )
    state = TrainingState.initial(
        lineage_id="p35-production-canary",
        token_budget=UPDATES * 4096,
        tokens_per_update=4096,
        cursor=CursorState(CURSOR_SCHEMA, pack_manifest_sha256, 0, 0, 0),
        rng_state_sha256="0" * 64,
        curriculum_phase="canary",
        identities=identities,
    )

    torch.manual_seed(SEED)
    model = initialize(P35_SPEC, seed=SEED)
    if device.type == "cuda":
        model = model.to(device)
    optimizer = build_adamw_optimizer(model)
    backend = ProductionTrainingBackend(
        model=model,
        optimizer=optimizer,
        bos_id=2,
        pad_id=0,
        device=device,
        bfloat16_autocast=device.type == "cuda",
        schedule=bounded_warmup_schedule(peak_learning_rate=PEAK_LEARNING_RATE),
    )
    import tempfile

    store = CheckpointStore(Path(tempfile.mkdtemp(prefix="anra-cymek-store-")), state.lineage_id)

    step_times: list[float] = []
    losses: list[float] = []
    grad_norms: list[float] = []

    def backend_step(current: TrainingState):
        ordinal = current.global_update
        batch = batch_from_window(
            windows[ordinal],
            pack_manifest_sha256=pack_manifest_sha256,
            update_ordinal=ordinal,
            device=device,
            torch_module=torch,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        report = backend.step(current, batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - started)
        grad_norms.append(report.grad_norm_post_clip)
        losses.append(float(backend.last_receipt["loss"]))
        return report

    controller = RunController(target_update=UPDATES)
    controller.start()
    started_total = time.perf_counter()
    final_state = train(
        state=state,
        controller=controller,
        store=store,
        payload_builder=lambda s: production_payloads(backend, state=s),
        backend_step=backend_step,
        updates=UPDATES,
        checkpoint_every=UPDATES,
    )
    total_seconds = time.perf_counter() - started_total

    _, payloads = store.restore()
    fresh_model = initialize(P35_SPEC, seed=SEED + 1)
    if device.type == "cuda":
        fresh_model = fresh_model.to(device)
    fresh_optimizer = build_adamw_optimizer(fresh_model)
    fresh_backend = ProductionTrainingBackend(
        model=fresh_model,
        optimizer=fresh_optimizer,
        bos_id=2,
        pad_id=0,
        device=device,
        bfloat16_autocast=device.type == "cuda",
        schedule=bounded_warmup_schedule(peak_learning_rate=PEAK_LEARNING_RATE),
    )
    restore_production(fresh_backend, payloads=payloads)
    live = capture_evidence(backend.model, backend.optimizer, torch=torch)
    resumed = capture_evidence(fresh_backend.model, fresh_backend.optimizer, torch=torch)
    resume_equal = (
        live.parameter_sha256 == resumed.parameter_sha256
        and live.moment_sha256 == resumed.moment_sha256
        and live.optimizer_steps == resumed.optimizer_steps
    )
    if not resume_equal:
        raise ValueError("P35 production canary resume was not hash-equivalent")

    peak_vram_mib = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    executable_parameters = sum(p.numel() for p in backend.model.parameters())
    ladder_claim = 35_414_400
    receipt = {
        "schema": SCHEMA,
        "status": "PASS",
        "classification": "P35_EXECUTED_PRODUCTION_PATH",
        "device": device_name,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "dtype_receipt": {
            "parameters": "float32",
            "compute": "bfloat16 autocast" if device.type == "cuda" else "float32",
            "loss_and_grad_norm": "float32",
            "optimizer_moments": "float32",
        },
        "model": {
            "spec_sha256": P35_SPEC.sha256(),
            "executable_parameters": executable_parameters,
            "training_spec_ladder_claim": ladder_claim,
            "ladder_delta": executable_parameters - ladder_claim,
            "ladder_note": "the executable receipt governs; the ladder prose differs and is recorded as negative evidence",
        },
        "optimizer_group_receipt": optimizer_group_receipt(backend.model, backend.optimizer),
        "updates": UPDATES,
        "tokens": {"per_update": 4096, "total": final_state.cumulative_tokens},
        "losses": losses,
        "grad_norms_post_clip": grad_norms,
        "step_seconds": step_times,
        "total_seconds": total_seconds,
        "peak_vram_mib": peak_vram_mib,
        "parameter_sha256": live.parameter_sha256,
        "moment_sha256": live.moment_sha256,
        "checkpoint": {
            "final_checkpoint_sha256": store.latest_sha256(),
            "resume_hash_equal": resume_equal,
        },
        "data": {
            "documents_bound": len(documents),
            "training_split_documents": len(training_ids),
            "manifest_audit": manifest_audit,
            "pack_audit": pack_audit,
        },
        "limitations": [
            "Bounded canary: proves certified production execution at P35 scale on this device.",
            "Not a learning experiment and not a V5-A authorization.",
        ],
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True).encode()
    ).hexdigest()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=Path("artifacts/cymek/p35_production_canary.json"))
    args = parser.parse_args()
    receipt = run_canary(repo_root=args.repo, device_name=args.device, output=args.output)
    print(json.dumps({"output": str(args.output), "status": receipt["status"]}, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
