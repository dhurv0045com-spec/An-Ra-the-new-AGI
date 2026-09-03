"""Citadel ONE-UPDATE CERTIFICATION of the Cymek production V5 path.

Independent of Cymek's own canaries. Demonstrates or falsifies, from an empty run
directory, that the Cymek stack at the audited commit can:

    consume REAL GENERATED COGNITION documents -> data manifest (dedup/split/scan)
    -> true multi-segment packing -> SAMPLER ORDER -> CURSOR-ADDRESSED MICROBATCH
    -> P35 V5 core -> causal CE -> backward -> clip -> AdamW -> token-indexed LR
    -> training-state advance -> content-addressed checkpoint -> exact restore.

The executed update uses the real v5_data.batch.microbatch + v5_data.pack.sampler_order
path that Cymek's own committed miniature and canaries bypass (they hand-slice windows).
All imported code is the unmodified Cymek tree at the pinned audit SHA; provenance is
recorded in the receipt. Emits ONE_UPDATE.json (certification) and TEN_UPDATE.json
(sanity escalation) — real executions, not unit-test assertions.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

CYMEK_AUDIT_ROOT = Path(r"C:\Users\ankit\.zcode\tmp\audit-cymek2")
CYMEK_AUDIT_SHA = "26a61f6242e9e5c1d1b028b4f8c3c7d26ac0fdc6"
RECEIPT_DIR = Path(__file__).resolve().parent

SEED = 202_609_03
PEAK_LEARNING_RATE = 3e-4
TOKENS_PER_UPDATE = 4_096
SEQUENCES_PER_SHARD = 8  # 8 x 512 = 4096 = exactly one update per full shard
SPLITS = {"training": 0.7, "development": 0.2, "sealed": 0.05, "fresh": 0.05}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def main() -> int:
    t_start = time.time()
    sys.path.insert(0, str(CYMEK_AUDIT_ROOT))

    import torch

    from e0_cognition.training_generators import build_training_examples
    from v5_contracts.model_spec import ModelSpec
    from v5_data.batch import microbatch
    from v5_data.manifest import Document, build_data_manifest, manifest_sha256
    from v5_data.pack import pack_documents, sampler_order
    from v5_model.core import initialize
    from v5_training.checkpoint import CheckpointStore
    from v5_training.miniature import MINIATURE_EVAL_TASKS, _load_tokenizer
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
    from v5_training.state import (
        CURSOR_SCHEMA,
        IDENTITY_SCHEMA,
        CursorState,
        IdentityBindings,
        TrainingState,
    )
    from v5_training.trainer import train

    imported_from = sorted({
        key: Path(getattr(module, "__file__", "")).as_posix()
        for key, module in sorted(sys.modules.items())
        if key.split(".")[0] in {"v5_data", "v5_model", "v5_objectives", "v5_training",
                                 "v5_tokenizer", "v5_contracts", "v5_evaluation", "e0_cognition"}
        and getattr(module, "__file__", None)
    }.values())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": device.type,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_threads": torch.get_num_threads(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cymek_audit_root": CYMEK_AUDIT_ROOT.as_posix(),
        "cymek_audit_sha": CYMEK_AUDIT_SHA,
        "imported_module_files": imported_from,
    }

    # 1. Real generated cognition documents (the data source Cymek's own runs never used)
    examples = build_training_examples(seed=SEED, count=1_200)
    documents = [
        Document(
            doc_id=example.example_id,
            text=f"{example.context}\n{example.query} {example.answer}",
            source_id=example.example_id,
            domain="synthetic-cognition",
            family="verified_cognition",
            authorization_category="first-party-authorized",
            acquired_date=time.strftime("%Y-%m-%d"),
        )
        for example in examples
    ]

    # 2. Real frozen tokenizer artifact (hash-verified against its tournament receipt)
    tokenizer, tokenizer_eval = _load_tokenizer(CYMEK_AUDIT_ROOT)

    # 3. Real data manifest: dedup, cluster-level split, contamination scan
    manifest, manifest_audit = build_data_manifest(
        documents,
        manifest_id="citadel-one-update-cert",
        tokenizer_sha256=tokenizer.identity.artifact_sha256,
        filter_version="citadel-cert-filter/v1",
        dedup_version="exact-clusters/v1",
        split_salt="citadel-one-update-cert/v1",
        split_boundaries=SPLITS,
        count_tokens=lambda text: len(tokenizer.encode(text)),
        contamination_benchmarks={task["task_id"]: task["prompt"] for task in MINIATURE_EVAL_TASKS},
    )
    training_ids = {r.source_id for r in manifest.sources if r.split == "training"}
    training_documents = [d for d in documents if d.doc_id in training_ids]

    # 4. True multi-segment packing
    packed, pack_audit = pack_documents(
        [(d.doc_id, tokenizer.encode(d.text), d.family) for d in training_documents],
        bos=2, eos=3, pad=0, sequences_per_shard=SEQUENCES_PER_SHARD,
    )
    shard_hashes = [shard.sha256() for shard in packed]
    pack_manifest_sha256 = hashlib.sha256(
        _canonical_json([json.loads(shard.payload_bytes()) for shard in packed])
    ).hexdigest()

    # 5. Sampler order + cursor-addressed microbatch walk (the bypassed edge, now executed)
    order = sampler_order(shard_hashes, run_seed=SEED, epoch=0)

    def next_coordinates(position: int, sequence_index: int) -> tuple[int, int]:
        sequence_index += 1
        if sequence_index >= SEQUENCES_PER_SHARD:
            return position + 1, 0
        return position, sequence_index

    def collect_windows(want: int) -> list[dict[str, object]]:
        windows: list[dict[str, object]] = []
        position, sequence_index = 0, 0
        while position < len(order) and len(windows) < want:
            try:
                batch = microbatch(
                    packed, order,
                    shard_ordinal=position, sequence_ordinal=sequence_index,
                    sequences=SEQUENCES_PER_SHARD, pad=0,
                )
            except ValueError:
                position, sequence_index = next_coordinates(position, sequence_index)
                continue
            full = all(len(row) == 512 for row in batch.tokens)
            if full and batch.consumed_real_tokens == TOKENS_PER_UPDATE:
                windows.append({
                    "tokens": batch.tokens,
                    "segment_ids": batch.segment_ids,
                    "tokens_by_source": batch.tokens_by_source,
                    "start": (position, sequence_index),
                    "end": (batch.shard_ordinal, batch.sequence_ordinal),
                })
                position, sequence_index = batch.shard_ordinal, batch.sequence_ordinal
            else:
                position, sequence_index = next_coordinates(position, sequence_index)
        if len(windows) < want:
            raise ValueError(
                f"pack yields only {len(windows)} exact {TOKENS_PER_UPDATE}-real-token "
                f"microbatch windows; {want} required"
            )
        return windows

    def run_phase(label: str, updates: int) -> dict[str, object]:
        windows = collect_windows(updates)
        identities = IdentityBindings(
            schema=IDENTITY_SCHEMA,
            source_commit=CYMEK_AUDIT_SHA,
            model_spec_sha256=chosen_spec.sha256(),
            tokenizer_sha256=tokenizer.identity.artifact_sha256,
            data_manifest_sha256=manifest_sha256(manifest),
            pack_manifest_sha256=pack_manifest_sha256,
            run_spec_sha256=hashlib.sha256(
                _canonical_json({"phase": label, "updates": updates, "budget": updates * TOKENS_PER_UPDATE})
            ).hexdigest(),
            optimizer_spec_sha256=hashlib.sha256(_canonical_json({
                "optimizer": "AdamW", "beta1": 0.9, "beta2": 0.95,
                "epsilon": 1e-8, "weight_decay": 0.1, "peak_lr": PEAK_LEARNING_RATE,
            })).hexdigest(),
            schedule_spec_sha256=hashlib.sha256(_canonical_json({
                "kind": "bounded_warmup", "peak_learning_rate": PEAK_LEARNING_RATE,
                "index": "pre-update cumulative real non-padding tokens",
            })).hexdigest(),
            curriculum_spec_sha256=hashlib.sha256(b"citadel-cert-cognition-only").hexdigest(),
        )
        state = TrainingState.initial(
            lineage_id=f"citadel-cert-{label}",
            token_budget=updates * TOKENS_PER_UPDATE,
            tokens_per_update=TOKENS_PER_UPDATE,
            cursor=CursorState(CURSOR_SCHEMA, pack_manifest_sha256, 0, 0, 0),
            rng_state_sha256="0" * 64,
            curriculum_phase="citadel-certification",
            identities=identities,
        )
        torch.manual_seed(SEED)
        model = initialize(chosen_spec, seed=SEED)
        parameter_count = sum(p.numel() for p in model.parameters())
        optimizer = build_adamw_optimizer(model)
        backend = ProductionTrainingBackend(
            model=model, optimizer=optimizer, bos_id=2, pad_id=0,
            device=device, schedule=bounded_warmup_schedule(peak_learning_rate=PEAK_LEARNING_RATE),
        )
        store_path = RECEIPT_DIR / f"_store_{label}"
        # certification must start from an empty run directory: remove any state left
        # by an aborted prior attempt so the writer fence cannot see a stale parent
        shutil.rmtree(store_path, ignore_errors=True)
        store = CheckpointStore(store_path, f"citadel-cert-{label}")

        losses, grad_norms, step_seconds, ledgers = [], [], [], []
        phase_start = time.time()

        def backend_step(current: TrainingState):
            t0 = time.time()
            window = windows[current.global_update]
            tokens = torch.tensor([list(row) for row in window["tokens"]], dtype=torch.long, device=device)
            segment_ids = torch.tensor([list(row) for row in window["segment_ids"]], dtype=torch.int32, device=device)
            batch = PackedBatch(
                tokens=tokens,
                segment_ids=segment_ids,
                tokens_by_source=dict(window["tokens_by_source"]),
                cursor=CursorState(
                    CURSOR_SCHEMA, pack_manifest_sha256, 0, current.global_update,
                    TOKENS_PER_UPDATE * (current.global_update + 1),
                ),
                rng_state_sha256=hashlib.sha256(f"citadel-cert-rng-{current.global_update}".encode()).hexdigest(),
            )
            report = backend.step(current, batch)
            receipt = backend.last_receipt
            losses.append(float(receipt["loss"]))
            grad_norms.append(float(receipt.get("grad_norm_post_clip", float("nan"))))
            step_seconds.append(round(time.time() - t0, 3))
            ledgers.append({
                "update": current.global_update,
                "tokens_by_source": dict(window["tokens_by_source"]),
                "consumed_real_tokens": int(sum(window["tokens_by_source"].values())),
            })
            return report

        controller = RunController(target_update=updates)
        controller.start()
        final_state = train(
            state=state, controller=controller, store=store,
            payload_builder=lambda s: production_payloads(backend, state=s),
            backend_step=backend_step, updates=updates, checkpoint_every=max(1, updates),
        )

        restored_state, payloads = store.restore()
        fresh_model = initialize(chosen_spec, seed=SEED + 1)
        fresh_backend = ProductionTrainingBackend(
            model=fresh_model, optimizer=build_adamw_optimizer(fresh_model),
            bos_id=2, pad_id=0, device=device,
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

        return {
            "phase": label,
            "model_spec_sha256": chosen_spec.sha256(),
            "executable_parameter_count": int(parameter_count),
            "updates_requested": updates,
            "updates_executed": int(final_state.global_update),
            "cumulative_tokens": int(final_state.cumulative_tokens),
            "state_complete": bool(final_state.complete),
            "losses": losses,
            "grad_norms_post_clip": grad_norms,
            "step_seconds": step_seconds,
            "tokens_per_second": round(
                (updates * TOKENS_PER_UPDATE) / max(sum(step_seconds), 1e-9), 1
            ),
            "phase_wall_seconds": round(time.time() - phase_start, 3),
            "update_ledgers": ledgers,
            "final_update_receipt_sha256": hashlib.sha256(
                _canonical_json(backend.last_receipt)
            ).hexdigest(),
            "final_update_receipt": backend.last_receipt,
            "checkpoint_sha256": store.latest_sha256(),
            "checkpoint_restore_payload_keys": sorted(payloads.keys()),
            "resume_parameter_sha256_equal": live.parameter_sha256 == resumed.parameter_sha256,
            "resume_moment_sha256_equal": live.moment_sha256 == resumed.moment_sha256,
            "resume_optimizer_steps_equal": live.optimizer_steps == resumed.optimizer_steps,
            "resume_hash_equal": resume_equal,
            "final_state_cursor": {
                "shard_ordinal": final_state.cursor.shard_ordinal,
                "sequence_ordinal": final_state.cursor.sequence_ordinal,
                "token_offset": final_state.cursor.token_offset,
            },
        }

    from v5_training.production_canary import P35_SPEC  # exact frozen P35 recipe
    from anra_v5.miniature_run import MINI_SPEC

    chosen_spec = P35_SPEC if len(sys.argv) < 2 or sys.argv[1] == "p35" else MINI_SPEC
    chosen_label = (
        "exact P35 recipe (16L x 384, 6Q/3KV, FFN 1024)"
        if chosen_spec is P35_SPEC
        else "miniature recipe (2L x 64, 4Q/2KV, FFN 128)"
    )

    one = run_phase("one_update", 1)
    ten = run_phase("ten_update", 10)

    certification = {
        "real_tokens_loaded": True,
        "real_cognition_data_source": "e0_cognition.training_generators (e0-train/0.2.0)",
        "real_data_manifest_path_used": True,
        "real_packer_used": True,
        "real_sampler_and_microbatch_used": True,
        "real_cursor_used": True,
        "real_model_used": chosen_label,
        "real_objective_used": "causal_lm_loss via ProductionTrainingBackend",
        "nonzero_gradients_certified": "mechanically enforced by certify_real_update (raises otherwise)",
        "optimizer_stepped": "mechanically enforced (per-parameter step == before+1)",
        "parameters_changed": "mechanically enforced (before/after SHA-256 of every tensor)",
        "training_state_advanced": one["state_complete"] and ten["state_complete"],
        "checkpoint_serialized_and_reloaded": one["resume_hash_equal"] and ten["resume_hash_equal"],
    }

    one_receipt = {
        "schema": "citadel-cymek-one-update-certification/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds_total": round(time.time() - t_start, 3),
        "environment": environment,
        "data_provenance": {
            "generator": "e0_cognition.training_generators.build_training_examples",
            "generator_version": "e0-train/0.2.0",
            "seed": SEED,
            "examples_requested": 1_200,
            "documents_built": len(documents),
            "manifest_id": manifest.manifest_id,
            "data_manifest_sha256": manifest_sha256(manifest),
            "manifest_audit": manifest_audit,
            "training_split_documents": len(training_documents),
            "families_present": sorted({d.family for d in documents}),
            "contamination_benchmarks": sorted({t["task_id"] for t in MINIATURE_EVAL_TASKS}),
            "tokenizer_artifact_sha256": tokenizer.identity.artifact_sha256,
        },
        "pack_provenance": {
            "packer": "v5_data.pack.pack_documents (true multi-segment stream-fill)",
            "sequences_per_shard": SEQUENCES_PER_SHARD,
            "shards": len(packed),
            "pack_audit": pack_audit,
            "pack_manifest_sha256": pack_manifest_sha256,
            "shard_sha256_prefixes": [h[:12] for h in shard_hashes],
            "sampler_order_sha256": hashlib.sha256(
                _canonical_json(order)
            ).hexdigest(),
        },
        "certification_checklist": certification,
        "one_update": one,
        "known_limitations": [
            "Cognition-only corpus: the 65/20/15 production mixture and v5_data.mixture "
            "allocation are NOT exercised by this certification (documents carry one family).",
            "bounded_warmup_schedule (constant LR) used, not the frozen 5B WSD schedule, "
            "matching Cymek's own executed runs.",
            "Documents arrive in memory; no corpus loader exists upstream of the manifest.",
            "Training-text format is context+query+answer per document, a Citadel choice; "
            "Cymek defines no cognition training-document renderer.",
        ],
    }
    ten_receipt = {
        "schema": "citadel-cymek-ten-update-sanity/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "prove training machinery behaves correctly; NO cognition inference",
        "environment": environment,
        "ten_update": ten,
    }

    (RECEIPT_DIR / "ONE_UPDATE.json").write_text(
        json.dumps(one_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (RECEIPT_DIR / "TEN_UPDATE.json").write_text(
        json.dumps(ten_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "one_update": {
            "updates_executed": one["updates_executed"],
            "losses": one["losses"],
            "grad_norms": one["grad_norms_post_clip"],
            "resume_hash_equal": one["resume_hash_equal"],
            "checkpoint_sha256": one["checkpoint_sha256"],
        },
        "ten_update": {
            "updates_executed": ten["updates_executed"],
            "losses": ten["losses"],
            "tokens_per_second": ten["tokens_per_second"],
            "resume_hash_equal": ten["resume_hash_equal"],
        },
        "receipts": [str(RECEIPT_DIR / "ONE_UPDATE.json"), str(RECEIPT_DIR / "TEN_UPDATE.json")],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
