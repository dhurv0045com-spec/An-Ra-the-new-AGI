"""Bounded V5-A canary: the real 250,216,960-parameter center, certified.

Constructs the exact frozen V5-A ModelSpec, runs a small number of certified
production updates on the local accelerator with real packed data, publishes
a checkpoint, restores it into fresh objects, and proves hash-equivalent
continuation.  This is an integration and memory/time receipt, not a
learning experiment and not a V5-A launch authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from v5_contracts.model_spec import V5A_250M
from v5_data.manifest import build_data_manifest, manifest_sha256
from v5_data.pack import pack_documents
from v5_model.core import initialize
from v5_training.checkpoint import CheckpointStore
from v5_training.miniature import SPLITS, _load_corpus, _load_tokenizer, _source_commit
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
from v5_training.state import (
    CURSOR_SCHEMA,
    IDENTITY_SCHEMA,
    CursorState,
    IdentityBindings,
    TrainingState,
)
from v5_training.trainer import train


SCHEMA = "anra-v5a-bounded-canary/v1"
SEED = 47_031
PEAK_LEARNING_RATE = 3e-4
UPDATES = 2
BUCKET = 512


def run_canary(*, repo_root: Path, device_name: str = "cuda", output: Path | None = None) -> dict:
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    repo = repo_root.resolve()

    tokenizer, _ = _load_tokenizer(repo)
    documents = _load_corpus(repo, tokenizer)
    manifest, manifest_audit = build_data_manifest(
        documents,
        manifest_id="anra-v5a-bounded-canary",
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
    windows: list[list] = []
    for shard in packed:
        if shard.bucket != BUCKET:
            continue
        for sequence in shard.sequences:
            if -1 not in sequence.segment_ids and len(sequence.tokens) == BUCKET:
                windows.append([sequence])
    if len(windows) < UPDATES:
        raise ValueError(f"V5-A canary stream has {len(windows)} full sequences")
    windows = windows[:UPDATES]

    pack_manifest_sha256 = hashlib.sha256(
        json.dumps([json.loads(shard.payload_bytes()) for shard in packed], sort_keys=True).encode()
    ).hexdigest()
    identities = IdentityBindings(
        schema=IDENTITY_SCHEMA,
        source_commit=_source_commit(repo),
        model_spec_sha256=V5A_250M.sha256(),
        tokenizer_sha256=tokenizer.identity.artifact_sha256,
        data_manifest_sha256=manifest_sha256(manifest),
        pack_manifest_sha256=pack_manifest_sha256,
        run_spec_sha256=hashlib.sha256(b"v5a-bounded-canary-run/v1").hexdigest(),
        optimizer_spec_sha256=hashlib.sha256(b"adamw-fp32-master/v1").hexdigest(),
        schedule_spec_sha256=hashlib.sha256(b"bounded-warmup/v1").hexdigest(),
        curriculum_spec_sha256=hashlib.sha256(b"canary").hexdigest(),
    )
    state = TrainingState.initial(
        lineage_id="v5a-bounded-canary",
        token_budget=UPDATES * BUCKET,
        tokens_per_update=BUCKET,
        cursor=CursorState(CURSOR_SCHEMA, pack_manifest_sha256, 0, 0, 0),
        rng_state_sha256="0" * 64,
        curriculum_phase="canary",
        identities=identities,
    )

    torch.manual_seed(SEED)
    construction_started = time.perf_counter()
    model = initialize(V5A_250M, seed=SEED)
    if device.type == "cuda":
        model = model.to(device)
    optimizer = build_adamw_optimizer(model)
    construction_seconds = time.perf_counter() - construction_started
    backend = ProductionTrainingBackend(
        model=model,
        optimizer=optimizer,
        bos_id=2,
        pad_id=0,
        device=device,
        bfloat16_autocast=device.type == "cuda",
        schedule=bounded_warmup_schedule(peak_learning_rate=PEAK_LEARNING_RATE),
    )
    store_root = repo / "artifacts/cymek/v5a_store"
    store = CheckpointStore(store_root, state.lineage_id)

    step_times: list[float] = []
    losses: list[float] = []
    peak_vram_mib = 0.0

    def backend_step(current: TrainingState):
        nonlocal peak_vram_mib
        ordinal = current.global_update
        window = windows[ordinal]
        tokens = torch.tensor([s.tokens for s in window], dtype=torch.long, device=device)
        segment_ids = torch.tensor(
            [s.segment_ids for s in window], dtype=torch.int32, device=device
        )
        per_source: dict[str, int] = {}
        for sequence in window:
            for index, source in enumerate(sequence.sources):
                count = sum(1 for segment in sequence.segment_ids if segment == index)
                per_source[source] = per_source.get(source, 0) + count
        batch = PackedBatch(
            tokens=tokens,
            segment_ids=segment_ids,
            tokens_by_source=dict(sorted(per_source.items())),
            cursor=CursorState(
                CURSOR_SCHEMA, pack_manifest_sha256, 0, ordinal, BUCKET * (ordinal + 1)
            ),
            rng_state_sha256="0" * 64,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        report = backend.step(current, batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_vram_mib = max(peak_vram_mib, torch.cuda.max_memory_allocated() / (1024 * 1024))
        step_times.append(time.perf_counter() - started)
        losses.append(float(backend.last_receipt["loss"]))
        return report

    controller = RunController(target_update=UPDATES)
    controller.start()
    final_state = train(
        state=state,
        controller=controller,
        store=store,
        payload_builder=lambda s: production_payloads(backend, state=s),
        backend_step=backend_step,
        updates=UPDATES,
        checkpoint_every=UPDATES,
    )

    _, payloads = store.restore()
    fresh_model = initialize(V5A_250M, seed=SEED + 1)
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
        raise ValueError("V5-A canary resume was not hash-equivalent")

    receipt = {
        "schema": SCHEMA,
        "status": "PASS",
        "classification": "V5A_BOUNDED_CANARY_EXECUTED",
        "device": device_name,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "model": {
            "spec_sha256": V5A_250M.sha256(),
            "executable_parameters": sum(p.numel() for p in backend.model.parameters()),
            "expected_parameters": V5A_250M.parameter_receipt().total,
            "single_tied_embedding": True,
        },
        "dtype_receipt": {
            "parameters": "float32",
            "compute": "bfloat16 autocast" if device.type == "cuda" else "float32",
            "loss_and_grad_norm": "float32",
            "optimizer_moments": "float32",
        },
        "construction_seconds": construction_seconds,
        "updates": UPDATES,
        "tokens": {"per_update": BUCKET, "total": final_state.cumulative_tokens},
        "losses": losses,
        "step_seconds": step_times,
        "peak_vram_mib": peak_vram_mib,
        "parameter_sha256": live.parameter_sha256,
        "moment_sha256": live.moment_sha256,
        "optimizer_group_receipt": optimizer_group_receipt(backend.model, backend.optimizer),
        "checkpoint": {
            "final_checkpoint_sha256": store.latest_sha256(),
            "resume_hash_equal": resume_equal,
        },
        "limitations": [
            "Integration and certification receipt only: proves the exact V5-A center constructs, "
            "trains through the certified production path on this device, and resumes hash-equivalently.",
            "Not a learning result. Not a V5-A launch authorization.",
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
    parser.add_argument("--output", type=Path, default=Path("artifacts/cymek/v5a_bounded_canary.json"))
    args = parser.parse_args()
    receipt = run_canary(repo_root=args.repo, device_name=args.device, output=args.output)
    print(json.dumps({"output": str(args.output), "status": receipt["status"]}, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
