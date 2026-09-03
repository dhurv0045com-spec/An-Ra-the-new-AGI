"""End-to-end miniature proof of the entire production V5 path (driver).

Lives in the unrestricted ``anra_v5`` plane because it intentionally joins the
training plane (``v5_training``) with the evaluation plane (``v5_evaluation``)
to prove one continuous production path end to end.  See
``artifacts/cymek/miniature_receipt.json`` for the committed receipt.

    real tokenizer artifact -> real corpus documents -> data manifest
    (dedup, cluster split, contamination scan) -> true packing -> sampler
    -> certified production updates (CE, clip, AdamW, token-indexed LR)
    -> content-addressed checkpoints -> exact restore -> checkpoint-backed
    raw evaluation behind the gold firewall -> task-level evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from v5_contracts.model_spec import ModelSpec
from v5_data.manifest import build_data_manifest, manifest_sha256
from v5_data.pack import pack_documents
from v5_data.stream import build_update_stream
from v5_evaluation.checkpoint_adapter import CheckpointBackedV5Adapter
from v5_evaluation.firewall import (
    CommittedOutput,
    build_evaluator_truth,
    build_visible_tasks,
    score_committed,
)
from v5_model.core import initialize
from v5_training.checkpoint import CheckpointStore
from v5_training.miniature import (
    MINIATURE_EVAL_TASKS,
    SPLITS,
    _canonical_json,
    _load_corpus,
    _load_tokenizer,
    _sha256_file,
    _source_commit,
)
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
from v5_training.streaming import batch_from_window
from v5_training.state import (
    CURSOR_SCHEMA,
    IDENTITY_SCHEMA,
    CursorState,
    IdentityBindings,
    TrainingState,
)
from v5_training.trainer import train


SCHEMA = "anra-v5-miniature-receipt/v1"

SEED = 202_609_03

PEAK_LEARNING_RATE = 3e-4

BUCKET = 512

SEQUENCES_PER_SHARD = 4

MAX_CORPUS_FILES = 48

MAX_DOCUMENT_TOKENS = 6_000

MINI_SPEC = ModelSpec(
    schema="anra-v5-model-spec/v1",
    family="dense-decoder-transformer",
    vocabulary_size=24_576,
    width=64,
    layers=2,
    query_heads=4,
    kv_heads=2,
    head_dimension=16,
    ffn_width=128,
    context_length=4_096,
    rope_base=10_000.0,
    norm_epsilon=1e-5,
    tied_embeddings=True,
    qk_norm=True,
    qk_norm_affine=True,
    linear_bias=False,
    dropout=0.0,
)




def run_miniature(
    *,
    repo_root: Path,
    updates: int = 6,
    device_name: str = "cpu",
    output: Path | None = None,
) -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        return {"schema": SCHEMA, "status": "BLOCKED_TORCH", "reason": str(exc)}
    device = torch.device(device_name)
    repo = repo_root.resolve()

    tokenizer, tokenizer_eval = _load_tokenizer(repo)
    documents = _load_corpus(repo, tokenizer)

    manifest, manifest_audit = build_data_manifest(
        documents,
        manifest_id="anra-v5-miniature",
        tokenizer_sha256=tokenizer.identity.artifact_sha256,
        filter_version="miniature-filter/v1",
        dedup_version="exact-clusters/v1",
        split_salt="anra-v5-miniature/v1",
        split_boundaries=SPLITS,
        count_tokens=lambda text: len(tokenizer.encode(text)),
        contamination_benchmarks={
            task["task_id"]: task["prompt"] for task in MINIATURE_EVAL_TASKS
        },
    )

    training_ids = {
        record.source_id for record in manifest.sources if record.split == "training"
    }
    training_documents = [document for document in documents if document.doc_id in training_ids]
    if not training_documents:
        raise ValueError("no training-split documents survived the cluster split")

    packed, pack_audit = pack_documents(
        [
            (document.doc_id, tokenizer.encode(document.text), document.family)
            for document in training_documents
        ],
        bos=2,
        eos=3,
        pad=0,
        sequences_per_shard=SEQUENCES_PER_SHARD,
    )
    windows = build_update_stream(packed, run_seed=SEED)
    if len(windows) < updates:
        raise ValueError(
            f"miniature stream has {len(windows)} exact 4096-token updates; "
            f"{updates} requested"
        )
    run_updates = min(updates, len(windows))
    windows = windows[:run_updates]
    token_budget = run_updates * 4096

    pack_manifest_sha256 = hashlib.sha256(
        _canonical_json([json.loads(shard.payload_bytes()) for shard in packed])
    ).hexdigest()
    schedule = bounded_warmup_schedule(peak_learning_rate=PEAK_LEARNING_RATE)
    schedule_receipt = {
        "schema": "anra-v5-miniature-schedule/v1",
        "kind": "bounded_warmup",
        "peak_learning_rate": PEAK_LEARNING_RATE,
        "index": "pre-update cumulative real non-padding tokens",
    }
    identities = IdentityBindings(
        schema=IDENTITY_SCHEMA,
        source_commit=_source_commit(repo),
        model_spec_sha256=MINI_SPEC.sha256(),
        tokenizer_sha256=tokenizer.identity.artifact_sha256,
        data_manifest_sha256=manifest_sha256(manifest),
        pack_manifest_sha256=pack_manifest_sha256,
        run_spec_sha256=hashlib.sha256(_canonical_json({"updates": run_updates, "budget": token_budget})).hexdigest(),
        optimizer_spec_sha256=hashlib.sha256(
            _canonical_json({
                "optimizer": "AdamW", "beta1": 0.9, "beta2": 0.95,
                "epsilon": 1e-8, "weight_decay": 0.1, "peak_lr": PEAK_LEARNING_RATE,
            })
        ).hexdigest(),
        schedule_spec_sha256=hashlib.sha256(_canonical_json(schedule_receipt)).hexdigest(),
        curriculum_spec_sha256=hashlib.sha256(b"miniature-uniform").hexdigest(),
    )
    state = TrainingState.initial(
        lineage_id="anra-v5-miniature",
        token_budget=token_budget,
        tokens_per_update=4096,
        cursor=CursorState(CURSOR_SCHEMA, pack_manifest_sha256, 0, 0, 0),
        rng_state_sha256="0" * 64,
        curriculum_phase="miniature",
        identities=identities,
    )

    torch.manual_seed(SEED)
    model = initialize(MINI_SPEC, seed=SEED)
    optimizer = build_adamw_optimizer(model)
    backend = ProductionTrainingBackend(
        model=model,
        optimizer=optimizer,
        bos_id=2,
        pad_id=0,
        device=device,
        schedule=schedule,
    )
    import tempfile

    store = CheckpointStore(Path(tempfile.mkdtemp(prefix="anra-cymek-store-")), state.lineage_id)

    losses: list[float] = []
    receipts: list[dict[str, object]] = []

    def backend_step(current: TrainingState):
        ordinal = current.global_update
        batch = batch_from_window(
            windows[ordinal],
            pack_manifest_sha256=pack_manifest_sha256,
            update_ordinal=ordinal,
            device=device,
            torch_module=torch,
        )
        report = backend.step(current, batch)
        receipts.append(backend.last_receipt)
        losses.append(float(backend.last_receipt["loss"]))
        return report

    controller = RunController(target_update=run_updates)
    controller.start()
    final_state = train(
        state=state,
        controller=controller,
        store=store,
        payload_builder=lambda s: production_payloads(backend, state=s),
        backend_step=backend_step,
        updates=run_updates,
        checkpoint_every=max(1, run_updates // 2),
    )
    if not final_state.complete:
        raise ValueError("miniature run did not reach its frozen token budget")

    restored_state, payloads = store.restore()
    fresh_model = initialize(MINI_SPEC, seed=SEED + 1)
    fresh_optimizer = build_adamw_optimizer(fresh_model)
    fresh_backend = ProductionTrainingBackend(
        model=fresh_model,
        optimizer=fresh_optimizer,
        bos_id=2,
        pad_id=0,
        device=device,
        schedule=schedule,
    )
    restore_production(fresh_backend, payloads=payloads)
    live_evidence = capture_evidence(backend.model, backend.optimizer, torch=torch)
    resumed_evidence = capture_evidence(
        fresh_backend.model, fresh_backend.optimizer, torch=torch
    )
    resume_equal = (
        live_evidence.parameter_sha256 == resumed_evidence.parameter_sha256
        and live_evidence.moment_sha256 == resumed_evidence.moment_sha256
        and live_evidence.optimizer_steps == resumed_evidence.optimizer_steps
    )
    if not resume_equal:
        raise ValueError("miniature resume did not reproduce the live parameter/optimizer state")

    adapter = CheckpointBackedV5Adapter(
        checkpoint_sha256=store.latest_sha256(),
        model_payload=payloads["model.bin"],
        model_spec=MINI_SPEC,
        tokenizer=tokenizer,
        device=device,
        torch_module=torch,
    )
    visible = build_visible_tasks(MINIATURE_EVAL_TASKS)
    truth = build_evaluator_truth(MINIATURE_EVAL_TASKS)
    task_evidence = []
    for task, gold in zip(visible, truth):
        scores = adapter.score_candidates("", task.prompt, list(task.candidates))
        choice = adapter.generate_constrained(task.prompt, list(task.candidates))
        free = adapter.generate_free(task.prompt, max_new_tokens=16)
        committed = CommittedOutput(
            task_id=task.task_id, output=choice, candidate_scores=tuple(scores)
        )
        result = score_committed(committed, task, gold)
        task_evidence.append(
            {
                "task_id": result.task_id,
                "cluster_id": result.cluster_id,
                "family": result.family,
                "split": result.split,
                "difficulty": result.difficulty,
                "raw_output": result.raw_output,
                "gold": result.gold,
                "correct": result.correct,
                "candidate_scores": list(result.candidate_scores or ()),
                "free_generation": free,
                "checkpoint_sha256": store.latest_sha256(),
                "adapter_sha256": adapter.identity.sha256(),
            }
        )

    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "status": "PASS",
        "scope": "end-to-end miniature through the single production path",
        "classification": "END_TO_END_MINIATURE",
        "source_commit": identities.source_commit,
        "identity_bindings": {
            "model_spec_sha256": identities.model_spec_sha256,
            "tokenizer_sha256": identities.tokenizer_sha256,
            "data_manifest_sha256": identities.data_manifest_sha256,
            "pack_manifest_sha256": identities.pack_manifest_sha256,
            "schedule_spec_sha256": identities.schedule_spec_sha256,
            "optimizer_spec_sha256": identities.optimizer_spec_sha256,
        },
        "model": {
            "family": "miniature V5 spec (frozen tokenizer vocabulary)",
            "parameter_count": sum(p.numel() for p in backend.model.parameters()),
            "context_length": BUCKET,
        },
        "tokenizer": {
            "artifact_sha256": tokenizer.identity.artifact_sha256,
            "vocabulary_size": tokenizer.identity.vocabulary_size,
            "tournament_evaluation": tokenizer_eval,
        },
        "data": {
            "documents_bound": len(documents),
            "training_split_documents": len(training_documents),
            "manifest_audit": manifest_audit,
            "pack_audit": pack_audit,
            "exact_updates_available": len(windows),
            "updates_run": run_updates,
            "token_budget": token_budget,
            "cumulative_tokens": final_state.cumulative_tokens,
            "tokens_by_source": dict(final_state.tokens_by_source),
        },
        "training": {
            "losses": losses,
            "final_receipt": receipts[-1],
            "parameter_sha256": live_evidence.parameter_sha256,
            "moment_sha256": live_evidence.moment_sha256,
            "optimizer_step": max(live_evidence.optimizer_steps.values()),
        },
        "checkpoint": {
            "final_checkpoint_sha256": store.latest_sha256(),
            "resume_parameter_hash_equal": resume_equal,
        },
        "evaluation": {
            "adapter": asdict(adapter.identity),
            "tasks": task_evidence,
            "firewall": "VisibleTask/EvaluatorTruth split enforced at projection",
        },
        "limitations": [
            "Miniature scale proves the path and its certification, not learning quality.",
            "Padded pack sequences are excluded from the exact stream; their tokens are recorded in the pack audit.",
            "The bounded warmup schedule is bound by schedule_spec_sha256, not the canonical 5B WSD schedule.",
        ],
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    ).hexdigest()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _source_commit(repo: Path) -> str:
    import subprocess

    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        value = "0" * 40
    return value if len(value) == 40 and all(c in "0123456789abcdef" for c in value) else "0" * 40


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/cymek/miniature_receipt.json"))
    args = parser.parse_args()
    receipt = run_miniature(
        repo_root=args.repo, updates=args.updates, device_name=args.device, output=args.output
    )
    print(json.dumps({"output": str(args.output), "status": receipt["status"]}, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
