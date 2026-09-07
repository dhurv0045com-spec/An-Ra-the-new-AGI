"""Persistent mirror + eval hooks + XLA adapter + topology map tests (no torch).
Run: python tests/test_v5_durability_contracts.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v5_evaluation.campaign import (  # noqa: E402
    EVAL_CHECKPOINT_TOKENS,
    EvalCheckpointRef,
    EvalDatasetIdentity,
    EvalManifest,
    assert_no_train_eval_collision,
    ingest_eval_receipt,
)
from v5_training.checkpoint import CheckpointStore  # noqa: E402
from v5_training.persistent_store import (  # noqa: E402
    materialize_local,
    mirror_checkpoint,
    read_mirror_pointer,
)
from v5_training.state import (  # noqa: E402
    CURSOR_SCHEMA,
    IDENTITY_SCHEMA,
    CursorState,
    IdentityBindings,
    TrainingState,
)
from v5_training.topology_map import (  # noqa: E402
    pad_replica_batch,
    physical_plan,
    replica_shards,
)
from v5_training.xla_adapter import (  # noqa: E402
    XLAReplicatedBackend,
    require_frozen_topology,
    xla_status,
)


def _sha(name):
    return name * 64


def _identities():
    return IdentityBindings(
        IDENTITY_SCHEMA, "a" * 40, _sha("1"), _sha("2"), _sha("3"),
        _sha("4"), _sha("5"), _sha("6"), _sha("7"), _sha("8"))


def _state(update=1, cumulative=4):
    return TrainingState(
        schema="anra-v5-training-state/v1", lineage_id="m", generation=update,
        global_update=update, cumulative_tokens=cumulative, token_budget=12,
        tokens_per_update=4, tokens_by_source={"t": cumulative},
        optimizer_step_max=update, schedule_tokens=cumulative,
        cursor=CursorState(CURSOR_SCHEMA, _sha("4"), update, update, 0),
        rng_state_sha256=_sha("9"), curriculum_phase="u",
        identities=_identities(), parent_checkpoint_sha256=None)


def _payloads(state):
    from v5_training.checkpoint import _canonical_json
    return {
        "model.bin": f"model-{state.generation}".encode(),
        "optimizer.bin": f"opt-{state.generation}".encode(),
        "scheduler.json": _canonical_json({"schedule_tokens": state.schedule_tokens}),
        "rng.bin": state.rng_state_sha256.encode(),
        "cursor.json": _canonical_json({"schema": state.cursor.schema,
                                        "pack_manifest_sha256": state.cursor.pack_manifest_sha256,
                                        "shard_ordinal": state.cursor.shard_ordinal,
                                        "sequence_ordinal": state.cursor.sequence_ordinal,
                                        "token_offset": state.cursor.token_offset}),
        "ledger.json": _canonical_json(dict(state.tokens_by_source)),
        "training_state.json": _canonical_json(state.canonical()),
    }


def test_mirror_and_materialize_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(Path(tmp) / "local", "m")
        state = _state()
        sha = store.publish(state=state, payloads=_payloads(state),
                            expected_parent_sha256=None)
        receipt = mirror_checkpoint(store, Path(tmp) / "mirror",
                                    checkpoint_sha256=sha)
        assert receipt["status"] == "PERSISTENT_DURABLE"
        assert read_mirror_pointer(Path(tmp) / "mirror", lineage_id="m") == sha
        fresh = CheckpointStore(Path(tmp) / "fresh", "m")
        adopted = materialize_local(Path(tmp) / "mirror", fresh)
        assert adopted == sha
        restored, _ = fresh.restore()
        assert restored == state


def test_interrupted_copy_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(Path(tmp) / "local", "m")
        state = _state()
        sha = store.publish(state=state, payloads=_payloads(state),
                            expected_parent_sha256=None)
        mirror = Path(tmp) / "mirror" / "m"
        staging = mirror / f".staging-{sha}"
        staging.mkdir(parents=True)
        (staging / "partial.bin").write_bytes(b"half")
        assert read_mirror_pointer(Path(tmp) / "mirror", lineage_id="m") is None
        fresh = CheckpointStore(Path(tmp) / "fresh", "m")
        try:
            materialize_local(Path(tmp) / "mirror", fresh)
        except ValueError:
            pass
        else:
            raise AssertionError("pointerless mirror was trusted")


def test_corrupt_mirror_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(Path(tmp) / "local", "m")
        state = _state()
        sha = store.publish(state=state, payloads=_payloads(state),
                            expected_parent_sha256=None)
        mirror_checkpoint(store, Path(tmp) / "mirror", checkpoint_sha256=sha)
        target = Path(tmp) / "mirror" / "m" / sha / "model.bin"
        target.write_bytes(b"corrupt")
        fresh = CheckpointStore(Path(tmp) / "fresh", "m")
        try:
            materialize_local(Path(tmp) / "mirror", fresh)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt mirror was trusted")


def test_stale_and_foreign_pointers_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        mirror = Path(tmp) / "mirror" / "m"
        mirror.mkdir(parents=True)
        (mirror / "MIRROR_HEAD").write_text(json.dumps(
            {"schema": "anra-v5-persistent-mirror/v1", "lineage_id": "m",
             "head_sha256": "c" * 64}), encoding="utf-8")
        assert read_mirror_pointer(Path(tmp) / "mirror", lineage_id="m") == "c" * 64
        fresh = CheckpointStore(Path(tmp) / "fresh", "m")
        try:
            materialize_local(Path(tmp) / "mirror", fresh)
        except ValueError:
            pass
        else:
            raise AssertionError("dangling pointer was trusted")
        (mirror / "MIRROR_HEAD").write_text(json.dumps(
            {"schema": "anra-v5-persistent-mirror/v1", "lineage_id": "other",
             "head_sha256": "c" * 64}), encoding="utf-8")
        try:
            read_mirror_pointer(Path(tmp) / "mirror", lineage_id="m")
        except ValueError as exc:
            assert "lineage" in str(exc).lower()
        else:
            raise AssertionError("foreign pointer was trusted")


def test_eval_collision_rejected():
    try:
        assert_no_train_eval_collision(train_source_ids=["a", "b"],
                                       eval_source_ids=["b", "c"])
    except ValueError as exc:
        assert "COLLISION" in str(exc)
    else:
        raise AssertionError("train/eval collision was allowed")
    assert_no_train_eval_collision(train_source_ids=["a"],
                                   eval_source_ids=["b"])


def test_eval_schedule_and_ingest():
    assert EVAL_CHECKPOINT_TOKENS[0] == 0
    assert EVAL_CHECKPOINT_TOKENS[-1] == 500_000_000
    assert 100_000_000 in EVAL_CHECKPOINT_TOKENS
    ref = EvalCheckpointRef(cumulative_tokens=100_000_000, global_update=763,
                            checkpoint_sha256="d" * 64)
    ref.assert_valid()
    try:
        EvalCheckpointRef(cumulative_tokens=123, global_update=1,
                          checkpoint_sha256="d" * 64).assert_valid()
    except ValueError:
        pass
    else:
        raise AssertionError("off-schedule eval checkpoint was allowed")
    dataset = EvalDatasetIdentity(name="sealed-x", split="sealed",
                                  content_sha256="e" * 64, cases=1024)
    manifest = EvalManifest(schema="anra-v5-campaign-eval/v1", round_id="r1",
                            datasets=(dataset,))
    manifest_sha = manifest.sha256()
    with tempfile.TemporaryDirectory() as tmp:
        receipt = {"checkpoint_sha256": "d" * 64,
                   "eval_manifest_sha256": manifest_sha, "verdict": "GO"}
        import hashlib
        blob = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        receipt["sha256"] = hashlib.sha256(blob).hexdigest()
        digest = ingest_eval_receipt(tmp, receipt)
        assert digest == receipt["sha256"]
        again = ingest_eval_receipt(tmp, receipt)
        assert again == digest
        bad = dict(receipt, verdict="NO_GO")
        try:
            ingest_eval_receipt(tmp, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("tampered receipt was ingested")


def test_promotion_verdict_delegates():
    from v5_promotion.gates import all_pass, evaluate_gates
    gates = evaluate_gates({"f": {"probe": [1, 1]}})
    assert isinstance(all_pass(gates), bool)
    assert set(gates) != set()


def test_topology_map_shapes():
    rows = [[i] for i in range(16)]
    shards = replica_shards(rows, replicas=8)
    assert [len(shard) for shard in shards] == [2] * 8
    assert [row for shard in shards for row in shard] == rows
    try:
        replica_shards(rows, replicas=3)
    except ValueError as exc:
        assert "do not divide" in str(exc)
    else:
        raise AssertionError("indivisible sharding was allowed")
    plan = physical_plan(bucket=512, sequences_global=64, replicas=8,
                         sequences_per_replica=8)
    assert plan["sequences_per_replica"] == 8
    try:
        physical_plan(bucket=512, sequences_global=63, replicas=8,
                      sequences_per_replica=8)
    except ValueError:
        pass
    else:
        raise AssertionError("topology mismatch was certified")
    from v5_training.topology_map import certify_microstep_shape
    frozen = certify_microstep_shape(bucket=512, sequences_global=64,
                                     replicas=8, sequences_per_replica=8)
    assert "partial_tail" not in frozen
    sparse = certify_microstep_shape(bucket=512, sequences_global=70,
                                     replicas=8, sequences_per_replica=8)
    assert sparse["partial_tail"] is True
    assert sparse["sequences_global"] == 70
    padded = pad_replica_batch([[[1]], [[1], [1], [1]]], width=1, pad_row=[0])
    assert padded["rows_per_replica"] == 3
    assert padded["padded_rows_added"] == 2
    assert padded["shards"][0] == [[1], [0], [0]]


def test_xla_fail_closed_without_hardware():
    status = xla_status()
    assert status["status"] in ("TPU_EVIDENCE_REQUIRED",
                                "IMPLEMENTED_PENDING_PRE500M_TPU")
    try:
        require_frozen_topology(4, replicas=8)
    except ValueError as exc:
        assert "8" in str(exc)
    else:
        raise AssertionError("topology mismatch was allowed")
    assert require_frozen_topology(8, replicas=8)["status"] == \
        "IMPLEMENTED_PENDING_PRE500M_TPU"
    adapter = XLAReplicatedBackend(replica_backend=object(), replicas=8)
    adapter_status = adapter.status()
    assert adapter_status["status"] in ("TPU_EVIDENCE_REQUIRED",
                                        "IMPLEMENTED_PENDING_PRE500M_TPU")
    try:
        adapter.all_reduce_sum_gradients(object())
    except (RuntimeError, AttributeError, ImportError, ValueError):
        pass
    else:
        raise AssertionError("XLA collective ran without XLA")


_TESTS = [test_mirror_and_materialize_round_trip,
          test_interrupted_copy_fails_closed,
          test_corrupt_mirror_rejected,
          test_stale_and_foreign_pointers_rejected,
          test_eval_collision_rejected,
          test_eval_schedule_and_ingest,
          test_promotion_verdict_delegates,
          test_topology_map_shapes,
          test_xla_fail_closed_without_hardware]


def main() -> int:
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {exc}", flush=True)
    print(f"{len(_TESTS) - failed}/{len(_TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
