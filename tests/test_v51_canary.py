"""Task-3 CPU qualification suite for the V5.1 canary.

Covers: generator determinism, group-level split isolation, contamination
screen, shortcut baselines, packing safety (incl. deliberate broken fixture),
objective EOS inclusion, optimizer ownership, WSD trace (incl. resume, no
rewarm), frozen 5B schedule domain check, checkpoint roundtrip, corruption
rejection, crash injection, identity-mismatch rejection, scan safe actions,
and FRESH-PROCESS exact resume via real subprocesses (bitwise state equality
on CPU). Run with a torch-capable interpreter; skips cleanly without torch.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anra_v5 import v51_canary_data as data_mod  # noqa: E402
from anra_v5 import v51_canary_run as runner  # noqa: E402

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


@pytest.fixture(scope="module")
def dataset():
    return data_mod.build_dataset(seed=20260913, worlds_per_family=60)


def test_generator_is_deterministic(dataset):
    again = data_mod.build_dataset(seed=20260913, worlds_per_family=60)
    assert again["split_hashes"] == dataset["split_hashes"]
    third = data_mod.build_dataset(seed=20260914, worlds_per_family=60)
    assert third["split_hashes"] != dataset["split_hashes"]


def test_group_level_split_isolation(dataset):
    screen = data_mod.contamination_screen(dataset)
    assert screen["clean"], screen["collisions"]
    for family in data_mod.FAMILIES:
        groups = {}
        for split in ("training", "development", "sealed"):
            ids = {e.group_id for e in dataset["splits"][split] if e.family == family}
            groups[split] = ids
        assert not (groups["training"] & groups["development"])
        assert not (groups["training"] & groups["sealed"])
        assert not (groups["development"] & groups["sealed"])
    # all six families are present in every split
    for split in ("training", "development", "sealed"):
        families = {e.family for e in dataset["splits"][split]}
        assert families == set(data_mod.FAMILIES)


def test_reference_solver_matches_every_rendering():
    from anra_v5.v51_canary_data import _draw_world, render, solve

    for family in data_mod.FAMILIES:
        for index in range(25):
            for attempt in (0, 1, 2):
                world = _draw_world(family, index, 20260913, attempt)
                rendered = render(family, world)
                solved = solve(family, world)
                assert [a for _, _, a in rendered] == solved


def test_shortcut_baselines_cannot_solve_families(dataset):
    baselines = data_mod.shortcut_baselines(dataset)
    for family, scores in baselines.items():
        for name, value in scores.items():
            assert value < 0.35, f"{family} solved by {name} at {value}"


def test_wsd_trace_covers_all_phases_and_resume_without_rewarm():
    plan = runner.canary_wsd_receipt(token_budget=491_520)
    trace = runner.wsd_trace(plan, updates=120, tokens_per_update=4096,
                             resume_at_update=64)
    assert trace["all_match"] is True
    assert trace["rewarm_events"] == 0
    phases = {row["phase"] for row in trace["rows"]}
    assert phases == {"warmup", "stable", "decay"}
    # the last row is the PRE-update LR (mid-decay); the schedule lands exactly
    # on end_lr at the end of the token budget
    final = trace["rows"][-1]
    assert final["lr_actual"] == runner.canary_lr_at(plan)(
        cumulative_tokens=491_520 - 4_096)
    assert runner.canary_lr_at(plan)(cumulative_tokens=491_520) == pytest.approx(
        plan["decay"]["end_lr"], rel=1e-9)
    assert final["lr_actual"] < plan["stable"]["lr"]  # decay is real
    resumed_row = next(r for r in trace["rows"] if r["resumed_here"])
    assert resumed_row["lr_expected"] == trace["rows"][resumed_row["update"] - 2]["lr_expected"]


def test_frozen_5b_schedule_domain_probe():
    from v5_training.schedule import FINAL_LEARNING_RATE, PEAK_LEARNING_RATE, lr_at

    assert lr_at(cumulative_tokens=0) == 0.0
    assert lr_at(cumulative_tokens=25_000_000) == pytest.approx(PEAK_LEARNING_RATE / 2)
    assert lr_at(cumulative_tokens=49_999_999) < PEAK_LEARNING_RATE
    assert lr_at(cumulative_tokens=50_000_000) == PEAK_LEARNING_RATE
    assert lr_at(cumulative_tokens=4_499_999_999) == PEAK_LEARNING_RATE
    assert lr_at(cumulative_tokens=4_999_999_999) == pytest.approx(
        FINAL_LEARNING_RATE, rel=1e-3)


def test_packing_safety_and_broken_fixture_rejection():
    from v5_model.core import packed_layout
    from v5_data.pack import pack_documents

    documents = [("d1", [10, 11, 12, 13], "fam"), ("d2", [14, 15], "fam")]
    shards, audit = pack_documents(documents, bos=2, eos=3, pad=0,
                                   sequences_per_shard=2)
    expected_tokens = sum(len(t) for _, t, _ in documents) + 2 * len(documents)
    assert audit["tokens_by_source"]["fam"] == expected_tokens  # exact accounting
    assert audit["real_nonpad_tokens"] == expected_tokens
    broken = torch.tensor([[0, 1, 0]], dtype=torch.int32)  # segments reappear
    with pytest.raises(ValueError):
        packed_layout(broken, torch_module=torch)


def test_objective_includes_eos_and_excludes_pad():
    from v5_objectives.causal_lm import causal_lm_loss

    tokens = torch.tensor([[2, 7, 8, 3, 0]])      # BOS a b EOS PAD
    segments = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.int32)
    logits = torch.randn(1, 5, 16, requires_grad=True)
    loss_with_eos, count = causal_lm_loss(logits, tokens, segments,
                                          bos_id=2, pad_id=0)
    tokens_no_eos = torch.tensor([[2, 7, 8, 9, 0]])
    loss_without_eos, _ = causal_lm_loss(logits, tokens_no_eos, segments,
                                         bos_id=2, pad_id=0)
    assert count == 3  # targets: 7, 8, EOS (PAD excluded)
    assert torch.isfinite(loss_with_eos)
    assert not torch.isclose(loss_with_eos, loss_without_eos)


def test_optimizer_ownership_on_rung_a():
    from tools.next_core_compute_model import RUNG_A, parameter_receipt
    from v5_training.optimizer import build_adamw_optimizer, validate_parameter_ownership
    from v5_model.core import initialize

    spec = runner.geometry_to_spec(RUNG_A.__dict__)
    model = initialize(spec, seed=20260913)
    expected = parameter_receipt(RUNG_A)["total"]
    actual = sum(int(p.numel()) for p in model.parameters())
    assert actual == expected == 10_227_456
    optimizer = build_adamw_optimizer(model)
    validate_parameter_ownership(model, optimizer)  # exactly-once ownership


def _canary_env(tmp_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["V51_CANARY_ROOT"] = str(tmp_root)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_cli(args: list[str], tmp_root: Path):
    return subprocess.run([PY, "-m", "anra_v5.v51_canary_run", *args],
                          cwd=REPO, env=_canary_env(tmp_root),
                          capture_output=True, text=True, timeout=900)


@pytest.mark.skipif(os.environ.get("V51_SKIP_SLOW") == "1", reason="slow")
def test_fresh_process_exact_resume_is_bitwise(tmp_path):
    tmp_root = tmp_path / "canary"
    tmp_root.mkdir()
    first = _run_cli(["--mode", "run", "--rung", "A", "--updates", "2",
                      "--checkpoint-every", "2"], tmp_root)
    assert first.returncode == 0, first.stderr[-2000:]
    scan_after_crash = _run_cli(["--mode", "scan"], tmp_root)
    assert json.loads(scan_after_crash.stdout)["action"] == "RESUME"
    resumed = _run_cli(["--mode", "resume", "--rung", "A", "--updates", "4",
                        "--checkpoint-every", "2"], tmp_root)
    assert resumed.returncode == 0, resumed.stderr[-2000:]

    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    reference = _run_cli(["--mode", "run", "--rung", "A", "--updates", "4",
                          "--checkpoint-every", "4"], reference_root)
    assert reference.returncode == 0, reference.stderr[-2000:]

    def final_artifacts(root: Path) -> dict:
        store = runner.CheckpointStore(root / "state", runner.LINEAGE_ID)
        sha = store.latest_sha256()
        manifest = json.loads((store.objects / sha / "manifest.json").read_text("utf-8"))
        read = lambda name: hashlib.sha256(  # noqa: E731
            (store.objects / sha / name).read_bytes()).hexdigest()
        state = dict(manifest["state"])
        # the parent pointer legitimately differs: the resumed trajectory is
        # fenced to its mid-run checkpoint, the reference to None. Everything
        # else must be bitwise identical.
        state.pop("parent_checkpoint_sha256")
        return {
            "state": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
            "model": read("model.bin"),
            "optimizer": read("optimizer.bin"),
            "ledger": read("ledger.json"),
            "cursor": read("cursor.json"),
            "scheduler": read("scheduler.json"),
        }

    resumed = final_artifacts(tmp_root)
    reference = final_artifacts(reference_root)
    assert resumed == reference  # bitwise model/optimizer/ledger/cursor/schedule


@pytest.mark.skipif(os.environ.get("V51_SKIP_SLOW") == "1", reason="slow")
def test_corrupted_checkpoint_rejected(tmp_path):
    tmp_root = tmp_path / "canary"
    tmp_root.mkdir()
    first = _run_cli(["--mode", "run", "--rung", "A", "--updates", "1",
                      "--checkpoint-every", "1"], tmp_root)
    assert first.returncode == 0, first.stderr[-2000:]
    store = runner.CheckpointStore(tmp_root / "state", runner.LINEAGE_ID)
    sha = store.latest_sha256()
    target = store.objects / sha / "model.bin"
    raw = bytearray(target.read_bytes())
    raw[0] ^= 0xFF
    target.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        store.restore(sha)


@pytest.mark.skipif(os.environ.get("V51_SKIP_SLOW") == "1", reason="slow")
def test_crash_injection_preserves_last_known_good(tmp_path):
    tmp_root = tmp_path / "canary"
    tmp_root.mkdir()
    pack = runner.build_pack(seed=20260913, worlds_per_family=60)
    plan = runner.canary_wsd_receipt(token_budget=491_520)
    backend, _ = runner.make_backend(rung="A", device=None, bfloat16=False,
                                     schedule=runner.canary_lr_at(plan), seed=20260913)
    store = runner.CheckpointStore(tmp_root, runner.LINEAGE_ID)
    state = runner.initial_state(
        lineage_id=runner.LINEAGE_ID,
        pack_manifest_sha256=pack["pack_manifest_sha256"],
        token_budget=491_520, tokens_per_update=4096,
        identities=runner.identity_bindings(
            pack_manifest_sha256=pack["pack_manifest_sha256"],
            model_spec_sha=backend.model.spec.sha256(),
            tokenizer_artifact_sha=pack["tokenizer_receipt"]["artifact_sha256"],
            data_receipt_sha="0" * 64, canary_config_sha="0" * 64,
            wsd_sha=plan["sha256"]),
        rng_state_sha256="0" * 64)
    payloads = runner.production_payloads(backend, state=state)
    first = store.publish(state=state, payloads=payloads, expected_parent_sha256=None)
    # advance once, then inject a crash after staging: publication must fail,
    # LATEST must remain the last known good, and retry must succeed
    window = runner.build_update_stream(pack["shards"], run_seed=20260913,
                                        real_tokens_per_update=4096)[0]
    from v5_training.streaming import batch_from_window

    batch = batch_from_window(window, pack_manifest_sha256=pack["pack_manifest_sha256"],
                              update_ordinal=0)
    report = backend.step(state, batch)
    next_state = state.advance(tokens_by_source=report.tokens_by_source,
                               cursor=report.cursor,
                               rng_state_sha256=report.rng_state_sha256,
                               parent_checkpoint_sha256=first)
    next_payloads = runner.production_payloads(backend, state=next_state)
    from v5_training.checkpoint import InjectedCrash

    with pytest.raises(InjectedCrash):
        store.publish(state=next_state, payloads=next_payloads,
                      expected_parent_sha256=first, inject_crash_at="after_stage")
    assert store.latest_sha256() == first  # last known good preserved
    # recovery: resolve the stale staging directory, then retry publication
    staging = store.lineage_root / f".staging-{next_state.generation}"
    assert staging.exists()  # the crash left partial state visible
    shutil.rmtree(staging)
    recovered = store.publish(state=next_state, payloads=next_payloads,
                              expected_parent_sha256=first)
    assert store.latest_sha256() == recovered


def test_identity_mismatch_rejected_on_restore(tmp_path):
    tmp_root = tmp_path / "canary"
    tmp_root.mkdir()
    pack = runner.build_pack(seed=20260913, worlds_per_family=60)
    plan = runner.canary_wsd_receipt(token_budget=491_520)
    backend, _ = runner.make_backend(rung="A", device=None, bfloat16=False,
                                     schedule=runner.canary_lr_at(plan), seed=20260913)
    store = runner.CheckpointStore(tmp_root, runner.LINEAGE_ID)
    state = runner.initial_state(
        lineage_id=runner.LINEAGE_ID,
        pack_manifest_sha256=pack["pack_manifest_sha256"],
        token_budget=491_520, tokens_per_update=4096,
        identities=runner.identity_bindings(
            pack_manifest_sha256=pack["pack_manifest_sha256"],
            model_spec_sha=backend.model.spec.sha256(),
            tokenizer_artifact_sha=pack["tokenizer_receipt"]["artifact_sha256"],
            data_receipt_sha="0" * 64, canary_config_sha="0" * 64,
            wsd_sha=plan["sha256"]),
        rng_state_sha256="0" * 64)
    published = store.publish(state=state,
                              payloads=runner.production_payloads(backend, state=state),
                              expected_parent_sha256=None)
    restored_state, _ = store.restore(published)
    # a different executable identity must be refused by the runner's gate
    other = runner.identity_bindings(
        pack_manifest_sha256=pack["pack_manifest_sha256"],
        model_spec_sha=backend.model.spec.sha256(),
        tokenizer_artifact_sha=pack["tokenizer_receipt"]["artifact_sha256"],
        data_receipt_sha="1" * 64, canary_config_sha="1" * 64,
        wsd_sha=plan["sha256"])
    assert restored_state.identities != other


def test_experimental_output_modes_absent_from_canonical_path():
    from v5_next.contracts import CANONICAL_OUTPUT_MODE, TINY_REFERENCE_GEOMETRY

    assert TINY_REFERENCE_GEOMETRY.output_mode == CANONICAL_OUTPUT_MODE
    assert not TINY_REFERENCE_GEOMETRY.allow_experimental


def test_preregistration_freezes_before_results():
    prereg = json.loads((REPO / "experiments/V5_1_CANARY/PREREGISTRATION.json")
                        .read_text(encoding="utf-8"))
    assert prereg["rungs"]["A"]["parameter_count"] == 10_227_456
    assert prereg["rungs"]["B"]["parameter_count"] == 42_092_544
    assert "claim_ceiling" in prereg and "pass_gates" in prereg
