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
