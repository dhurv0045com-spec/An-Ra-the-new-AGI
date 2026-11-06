"""Retained ELENDRIL mirror, crash, and writer-exclusion regressions."""
import json
from pathlib import Path
import shutil
import tempfile

import pytest

from v5_training.checkpoint import CheckpointStore
from v5_training.persistent_store import materialize_local, mirror_checkpoint, read_mirror_pointer
from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState


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


def test_mirror_refuses_corrupt_existing_destination_without_advancing(tmp_path):
    store = CheckpointStore(tmp_path / "local", "m")
    state = _state()
    identity = store.publish(state=state, payloads=_payloads(state), expected_parent_sha256=None)
    mirror_root = tmp_path / "mirror"
    existing = mirror_root / "m" / identity
    shutil.copytree(store.objects / identity, existing)
    (existing / "model.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="mirrored component corrupt"):
        mirror_checkpoint(store, mirror_root, checkpoint_sha256=identity)
    assert read_mirror_pointer(mirror_root, lineage_id="m") is None
    fresh = CheckpointStore(tmp_path / "fresh", "m")
    with pytest.raises(ValueError, match="mirrored component corrupt"):
        materialize_local(mirror_root, fresh, checkpoint_sha256=identity)
    assert not (fresh.objects / identity).exists()
    assert fresh.latest_sha256() is None


def test_mirror_fails_on_explicit_invalid_identity(tmp_path):
    store = CheckpointStore(tmp_path / "local", "m")
    mirror = tmp_path / "mirror"
    for identity in ("", "../other"):
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            mirror_checkpoint(store, mirror, checkpoint_sha256=identity)
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            materialize_local(mirror, store, checkpoint_sha256=identity)
    assert not mirror.exists()


def test_shared_checkpoint_writer_fences_different_generations(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    import v5_training.checkpoint as checkpoint

    store = CheckpointStore(tmp_path, "m")
    first, rival = _state(), _state(update=2, cumulative=8)
    entered, release = Event(), Event()
    write_sync = checkpoint._write_sync

    def pause(path, payload):
        if path.name == "model.bin" and path.parent.name == ".staging-1":
            entered.set()
            assert release.wait(5)
        write_sync(path, payload)

    monkeypatch.setattr(checkpoint, "_write_sync", pause)
    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(store.publish, state=first, payloads=_payloads(first),
                             expected_parent_sha256=None)
        try:
            assert entered.wait(5)
            with pytest.raises(ValueError, match="writer lock unavailable"):
                store.publish(state=rival, payloads=_payloads(rival), expected_parent_sha256=None)
            with pytest.raises(ValueError, match="writer lock unavailable"):
                store.prune(keep=set())
        finally:
            release.set()
        assert writer.result(timeout=5) == store.latest_sha256()
    assert store.restore()[0] == first
