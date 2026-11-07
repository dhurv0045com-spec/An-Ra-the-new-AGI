"""Tiny correctness canaries, never a research training sweep."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from anra_v5.elendril.codec import BOS, EOS, canonical, digest, encode
from anra_v5.elendril.foundation_training import (
    FoundationTrainer, Recipe, StageTransition, TokenSample, legacy_tokenizer_manifest,
)
from anra_v5.elendril.training import MODEL, _dump, _load, configure_cpu
from v5_training.checkpoint import CheckpointStore, InjectedCrash, REQUIRED_COMPONENTS


SPEC = replace(MODEL, width=16, layers=1, query_heads=2, kv_heads=1,
               head_dimension=8, ffn_width=32)
PROVENANCE = {"split": "train", "license": "repository LICENSE", "source": "unit correctness only"}


def rows():
    result = []
    for i, text in enumerate(("A", "AB", "ABC", "ABCD")):
        ids = (BOS, *encode(text + ">Q"), EOS)
        result.append(TokenSample(str(i), ids, (False,) * (len(ids) - 2) + (True, True), str(i),
                                  "missing" if i % 2 else "known"))
    return tuple(result)


def engine(root, **kwargs):
    configure_cpu()
    args = dict(samples=rows(), spec=SPEC, recipe=Recipe(total_updates=4, batch_size=4, micro_batch_size=2),
                source_commit="a" * 40, root=root, lineage="test", data_provenance=PROVENANCE,
                tokenizer_manifest=legacy_tokenizer_manifest())
    args.update(kwargs)
    return FoundationTrainer(**args)


def equal_tensors(left, right):
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)


@pytest.mark.parametrize("objective", ["ce", "action-weighted"])
def test_variable_length_accumulation_matches_full_denominator(tmp_path, objective):
    full = engine(tmp_path / "full", recipe=Recipe(total_updates=2, batch_size=4, micro_batch_size=4,
                                                  objective=objective))
    accum = engine(tmp_path / "accum", recipe=replace(full.recipe, micro_batch_size=1))
    a, b = full.step(), accum.step()
    assert a["supervised_tokens"] == b["supervised_tokens"] == 8
    assert a["weighted_denominator"] == b["weighted_denominator"] == (8 if objective == "ce" else 16)
    assert a["loss"] == pytest.approx(b["loss"], abs=3e-7)
    for p, q in zip(full.model.parameters(), accum.model.parameters()):
        torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
    # Every tied parameter is owned and actually updated once.
    assert all(v["step"].item() == 1 for v in accum.optimizer.state.values())


def test_positive_learnability_and_nonfinite_negative(tmp_path):
    t = engine(tmp_path, recipe=Recipe(total_updates=8, batch_size=4, micro_batch_size=4, lr=.03))
    initial = {name: p.clone() for name, p in t.model.state_dict().items()}
    losses = [t.step()["loss"] for _ in range(8)]
    assert losses[-1] < losses[0] / 2
    assert any(not torch.equal(initial[n], p) for n, p in t.model.state_dict().items())
    bad = engine(tmp_path / "bad")
    with torch.no_grad():
        bad.model.embedding.weight[3, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        bad.step()
    with pytest.raises(ValueError, match="faulted"):
        bad.step()
    assert bad.store.latest_sha256() is None and bad.state.global_update == 0


def test_exact_resume_and_wrong_recipe_tokenizer_rejected(tmp_path):
    t = engine(tmp_path)
    t.step()
    parent = t.save()
    resumed = engine(tmp_path, resume=True)
    assert t.step() == resumed.step()
    equal_tensors(t.model.state_dict(), resumed.model.state_dict())
    assert resumed.state.parent_checkpoint_sha256 == parent
    with pytest.raises(ValueError, match="mismatch"):
        engine(tmp_path, resume=True, recipe=replace(t.recipe, objective="action-weighted"))
    manifest = legacy_tokenizer_manifest()
    manifest["normalizer_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="manifest"):
        engine(tmp_path, resume=True, tokenizer_manifest=manifest)


@pytest.fixture
def committed(tmp_path):
    t = engine(tmp_path / "source")
    t.step(); first = t.save()
    t.step(); second = t.save()
    state, payloads = t.store.restore(second)
    return t, first, second, state, payloads


@pytest.mark.parametrize("stage", [*("after_component:" + n for n in sorted(REQUIRED_COMPONENTS)),
                                  "after_manifest", "after_stage", "after_publish_before_pointer",
                                  "after_pointer_stage", "after_pointer"])
def test_every_crash_boundary_has_no_phantom_publication(tmp_path, committed, stage):
    t, first, second, state, payloads = committed
    store = CheckpointStore(tmp_path / "crash", "test", payload_validator=t.validate_payloads)
    shutil.copytree(t.store.objects / first, store.objects / first)
    store.latest.write_text(first + "\n", encoding="ascii")
    with pytest.raises(InjectedCrash):
        store.publish(state=state, payloads=payloads, expected_parent_sha256=first, inject_crash_at=stage)
    expected = second if stage == "after_pointer" else first
    assert store.latest_sha256() == expected
    restored, _ = store.restore()
    assert restored.global_update == (2 if stage == "after_pointer" else 1)


@pytest.mark.parametrize("component", sorted(REQUIRED_COMPONENTS))
def test_truncation_each_component_fails_without_pointer_fallback(committed, component):
    t, first, second, _, _ = committed
    path = t.store.objects / second / component
    path.write_bytes(path.read_bytes()[:2])
    with pytest.raises(ValueError, match="corrupt"):
        t.store.restore()
    assert t.store.latest_sha256() == second
    assert t.store.restore(first)[0].global_update == 1


def test_decoding_and_counter_failures_rejected_before_publication(tmp_path, committed):
    t, first, second, state, original = committed
    store = CheckpointStore(tmp_path / "new", "test", payload_validator=t.validate_payloads)
    # An internally self-hashed payload is still invalid if its tensor or
    # optimizer counters disagree. Each independent attempt has a new store.
    for index, kind in enumerate(("shape", "counter", "rng", "cursor", "vocab")):
        store = CheckpointStore(tmp_path / str(index), "test", payload_validator=t.validate_payloads)
        payloads = dict(original)
        candidate = replace(state, parent_checkpoint_sha256=None)
        if kind == "shape":
            values = _load(payloads["model.bin"])
            values["embedding.weight"] = values["embedding.weight"][:-1]
            payloads["model.bin"] = _dump(values)
        elif kind == "counter":
            values = _load(payloads["optimizer.bin"])
            next(iter(values["state"].values()))["step"] -= 1
            payloads["optimizer.bin"] = _dump(values)
        elif kind == "rng":
            payloads["rng.bin"] = b"invalid"
        elif kind == "cursor":
            candidate = replace(candidate, cursor=replace(candidate.cursor, sequence_ordinal=0))
            payloads["cursor.json"] = canonical(asdict(candidate.cursor))
        else:
            metadata = json.loads(payloads["scheduler.json"])
            metadata["metadata"]["tokenizer"]["identity"] = "c" * 64
            payloads["scheduler.json"] = canonical(metadata)
        payloads["training_state.json"] = canonical(candidate.canonical())
        with pytest.raises(ValueError):
            store.publish(state=candidate, payloads=payloads, expected_parent_sha256=None)
        assert store.latest_sha256() is None


def test_stale_publish_and_previous_survive_pruning(committed):
    t, first, second, state, payloads = committed
    assert (t.store.lineage_root / "PREVIOUS").read_text().strip() == first
    assert t.store.prune(keep=set()) == []
    with pytest.raises(ValueError, match="stale parent"):
        t.store.publish(state=state, payloads=payloads, expected_parent_sha256=first)
    assert t.store.latest_sha256() == second


def test_copied_directory_resumes_in_fresh_process_for_multiple_updates(tmp_path):
    t = engine(tmp_path / "source")
    t.step(); t.save()
    shutil.copytree(tmp_path / "source", tmp_path / "copied")
    for _ in range(3):
        t.step()
    t.save()
    script = '''from pathlib import Path
import sys
from tests.test_elendril_foundation_training import engine
t=engine(Path(sys.argv[1]),resume=True)
for _ in range(3): t.step()
t.save()
'''
    # tests is not a package in installed environments; use an explicit local
    # module path instead of accidentally importing a third-party tests package.
    script = script.replace("from tests.test_elendril_foundation_training import engine",
        "import runpy\nengine=runpy.run_path('tests/test_elendril_foundation_training.py')['engine']")
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "copied")],
                            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    fresh = engine(tmp_path / "copied", resume=True)
    assert fresh.state.complete and fresh.state.global_update == 4
    equal_tensors(t.model.state_dict(), fresh.model.state_dict())
    for p, q in zip(t.optimizer.state.values(), fresh.optimizer.state.values()):
        equal_tensors(p, q)


def test_stage_migration_resets_optimizer_and_resumes_with_bound_recipe(tmp_path, committed):
    parent, _, checkpoint, _, _ = committed
    recipe = replace(parent.recipe, seed=19)
    probe = engine(tmp_path / "probe", recipe=recipe)
    transition = StageTransition(checkpoint, digest(asdict(parent.recipe)), digest(asdict(recipe)), probe.data_sha)
    child = engine(tmp_path / "child", recipe=recipe, transition=transition, parent_store=parent.store)
    equal_tensors(parent.model.state_dict(), child.model.state_dict())
    assert not child.optimizer.state and child.state.global_update == 0
    child.step(); child.save()
    resumed = engine(tmp_path / "child", recipe=recipe, transition=transition, resume=True)
    assert resumed.state.global_update == 1
    assert child.step() == resumed.step()


def test_curriculum_keeps_complete_epoch_exposure_and_strict_input(tmp_path):
    a = engine(tmp_path / "a", recipe=Recipe(total_updates=2, batch_size=2, micro_batch_size=1))
    b = engine(tmp_path / "b", recipe=replace(a.recipe, sampling="uncertainty-first"))
    assert sorted(a.plan[0] + a.plan[1]) == sorted(b.plan[0] + b.plan[1]) == list(range(4))
    assert all(b.samples[i].category == "missing" for i in b.plan[0])
    with pytest.raises(ValueError, match="empty"):
        engine(tmp_path / "empty", samples=())
    with pytest.raises(ValueError, match="framing"):
        engine(tmp_path / "mask", samples=(replace(rows()[0], eligible=(False,) * len(rows()[0].tokens)),),
               recipe=Recipe(total_updates=1, batch_size=1, micro_batch_size=1))
    with pytest.raises(ValueError, match="license"):
        engine(tmp_path / "leak", data_provenance={"split": "test"})


def test_two_fresh_process_publishers_accept_exactly_one_parent_transition(tmp_path, committed):
    """Race actual OS writer locks using precomputed payloads, with no training.

    Readiness pipes form a deterministic barrier. The go file is created only
    after both processes have loaded the same candidate. No timing-based sleep
    determines who wins; the loser must fail the lock or stale-parent fence.
    """
    from concurrent.futures import ThreadPoolExecutor

    t, first, second, _, _ = committed
    target = tmp_path / "race"
    store = CheckpointStore(target, "test", payload_validator=t.validate_payloads)
    shutil.copytree(t.store.objects / first, store.objects / first)
    store.latest.write_text(first + "\n", encoding="ascii")
    go = tmp_path / "GO"
    script = '''import sys
from pathlib import Path
from v5_training.checkpoint import CheckpointStore
source=CheckpointStore(Path(sys.argv[1]),'test')
target=CheckpointStore(Path(sys.argv[2]),'test')
state,payloads=source.restore(sys.argv[3])
print('READY',flush=True)
assert sys.stdin.readline() == 'GO\\n'
assert Path(sys.argv[5]).read_text() == 'release'
try:
    identity=target.publish(state=state,payloads=payloads,expected_parent_sha256=sys.argv[4])
except ValueError as exc:
    assert 'lock unavailable' in str(exc) or 'stale parent' in str(exc), str(exc)
    print('REJECT '+str(exc),flush=True)
else:
    print('ACCEPT '+identity,flush=True)
'''
    processes = []
    readers = ThreadPoolExecutor(max_workers=2)
    try:
        for _ in range(2):
            processes.append(subprocess.Popen(
                [sys.executable, "-c", script, str(t.store.root), str(target), second, first, str(go)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            ))
        ready = [readers.submit(process.stdout.readline) for process in processes]
        assert [future.result(timeout=45) for future in ready] == ["READY\n", "READY\n"]
        go.write_text("release", encoding="ascii")
        for process in processes:
            process.stdin.write("GO\n")
            process.stdin.flush()
        outputs = [process.communicate(timeout=45) for process in processes]
        assert all(process.returncode == 0 for process in processes), outputs
        lines = [output.strip() for output, _ in outputs]
        assert sum(line == "ACCEPT " + second for line in lines) == 1, lines
        assert sum(line.startswith("REJECT ") for line in lines) == 1, lines
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)
        readers.shutdown(wait=True)
    assert store.latest_sha256() == second
    assert store.restore()[0].global_update == 2
    assert store.restore(first)[0].global_update == 1
    assert (store.lineage_root / "PREVIOUS").read_text().strip() == first


def test_nonempty_scaler_state_cannot_be_published_as_fp32(tmp_path, committed):
    t, _, _, state, original = committed
    candidate = replace(state, parent_checkpoint_sha256=None)
    payloads = dict(original)
    payloads["training_state.json"] = canonical(candidate.canonical())
    scheduler = json.loads(payloads["scheduler.json"])
    scheduler["scaler"] = {"scale": 65536., "growth_factor": 2., "backoff_factor": .5,
                           "growth_interval": 2000, "_growth_tracker": 2}
    payloads["scheduler.json"] = canonical(scheduler)
    store = CheckpointStore(tmp_path / "bad-scaler", "test", payload_validator=t.validate_payloads)
    with pytest.raises(ValueError, match="scaler"):
        store.publish(state=candidate, payloads=payloads, expected_parent_sha256=None)
    assert store.latest_sha256() is None


def test_stage_migration_rejects_self_hashed_dtype_change(tmp_path, committed):
    """A generic byte-valid parent cannot silently cast float64 into FP32."""
    t, _, _, state, original = committed
    candidate = replace(state, parent_checkpoint_sha256=None)
    payloads = dict(original)
    weights = _load(payloads["model.bin"])
    weights["embedding.weight"] = weights["embedding.weight"].double()
    payloads["model.bin"] = _dump(weights)
    payloads["training_state.json"] = canonical(candidate.canonical())
    # Generic store proves checksums only; stage admission must validate dtype.
    parent_store = CheckpointStore(tmp_path / "generic-parent", "test")
    checkpoint = parent_store.publish(state=candidate, payloads=payloads, expected_parent_sha256=None)
    recipe = replace(t.recipe, seed=21)
    probe = engine(tmp_path / "probe-dtype", recipe=recipe)
    transition = StageTransition(checkpoint, digest(asdict(t.recipe)), digest(asdict(recipe)), probe.data_sha)
    with pytest.raises(ValueError, match="dtype"):
        engine(tmp_path / "dtype-child", recipe=recipe, transition=transition, parent_store=parent_store)


def test_utf8_manifest_binds_actual_training_and_exact_resume(tmp_path):
    from anra_v5.elendril.representation import UTF8

    samples = []
    for index, (prompt, answer) in enumerate((("स्थिति?", "हां"), ("🌌?", "é"))):
        framed = UTF8.frame(prompt, answer)
        samples.append(TokenSample(str(index), framed.ids, framed.loss_mask, str(index)))
    kwargs = dict(samples=tuple(samples), spec=replace(SPEC, vocabulary_size=UTF8.vocabulary_size),
                  tokenizer_manifest=UTF8.manifest.record(),
                  recipe=Recipe(total_updates=2, batch_size=2, micro_batch_size=1))
    t = engine(tmp_path / "utf8", **kwargs)
    before = t.model.embedding.weight.detach().clone()
    receipt = t.step()
    assert receipt["supervised_tokens"] == sum(sum(row.eligible) for row in samples)
    assert not torch.equal(before, t.model.embedding.weight)
    checkpoint = t.save()
    state, payloads = t.store.restore(checkpoint)
    metadata = json.loads(payloads["scheduler.json"])["metadata"]
    UTF8.require_manifest(metadata["tokenizer"])
    assert state.identities.tokenizer_sha256 == UTF8.identity
    assert metadata["model"]["vocabulary_size"] == UTF8.vocabulary_size
    resumed = engine(tmp_path / "utf8", resume=True, **kwargs)
    assert t.step() == resumed.step()
    equal_tensors(t.model.state_dict(), resumed.model.state_dict())
    with pytest.raises(ValueError, match="migration"):
        engine(tmp_path / "wrong-vocab", **{**kwargs, "spec": SPEC})
