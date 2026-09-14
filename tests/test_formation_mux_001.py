"""FORMATION-MUX-001 qualification tests (CPU, tiny fixtures; engineering
evidence only — no official science runs locally)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from anra_v5 import formation_mux_model as fxm  # noqa: E402
from anra_v5 import formation_mux_train as train  # noqa: E402
from v5_experiments import formation_mux_protocol as proto  # noqa: E402
from v5_experiments.formation_mux_data import (  # noqa: E402
    build_surface, load_surface, shortcut_screens,
)


# -- data ------------------------------------------------------------------

def test_surface_is_deterministic_and_hash_stable():
    class StubIdentity:
        vocabulary_size = 24576
        special_token_ids = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}

    class StubTokenizer:
        identity = StubIdentity()

        @staticmethod
        def encode(text: str) -> list[int]:
            words = text.split()
            return [hashlib_token(w) for w in words]

    def hashlib_token(word: str) -> int:
        return 128 + (sum(ord(c) for c in word) % 24000)

    first = build_surface(seed=proto.SEED_BUNDLES[0], tokenizer=StubTokenizer())
    second = build_surface(seed=proto.SEED_BUNDLES[0], tokenizer=StubTokenizer())
    assert first["sha256"] == second["sha256"]
    assert first["worst_shortcut_score"] <= 0.5
    counts = {"training": 600, "development": 80, "sealed": 120}
    for split, rows in first["splits"].items():
        assert len(rows) == counts[split]
    r0 = first["splits"]["training"][0]
    assert "r0_prompt_ids" in r0 and "r0_answer_ids" in r0


def test_surface_manifest_fails_closed_on_tamper(tmp_path):
    manifest = build_surface(seed=proto.SEED_BUNDLES[0])
    path = tmp_path / "surface.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    load_surface(path)  # clean load passes
    tampered = dict(manifest)
    tampered["splits"] = dict(manifest["splits"])
    tampered["splits"]["training"] = manifest["splits"]["training"][:-1]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(tampered, indent=1), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_surface(bad)


# -- model treatments --------------------------------------------------------

def test_m0_row_optimizer_matches_torch_adamw_bytes():
    torch.manual_seed(0)
    base = torch.randn(96, 8, dtype=torch.float64)
    grad_sequence = [torch.randn(96, 8, dtype=torch.float64) for _ in range(3)]
    reference = base.clone().requires_grad_(True)
    ref_opt = torch.optim.AdamW([reference], lr=1e-3, betas=(0.9, 0.95),
                                eps=1e-8, weight_decay=0.1)
    rows_param = base.clone()
    row_opt = fxm.EmbeddingRowOptimizer(
        rows_param, trainable_rows={r: 0.1 for r in range(96)}, lr=1e-3,
        betas=(0.9, 0.95), eps=1e-8, torch=torch)
    for grad in grad_sequence:
        reference.grad = grad.clone()
        rows_param.grad = grad.clone()
        ref_opt.step()
        row_opt.step()
        diff = (reference.detach() - rows_param.detach()).abs().max().item()
        scale = max(reference.detach().abs().max().item(), 1e-30)
        assert diff <= 1e-6 * scale, (
            "per-row AdamW must be mathematically identical to torch AdamW; "
            f"observed {diff} (scale {scale})")


def test_m2_frozen_rows_keep_init_bytes():
    torch.manual_seed(1)
    model = fxm.build_model(73011, "M2_EXTRA_FROZEN", torch=torch,
                            device=torch.device("cpu"))
    init = model.embedding.weight.detach().clone()
    optimizers = fxm.make_optimizers(model, "M2_EXTRA_FROZEN", torch=torch,
                                     lr=proto.LR)
    rows = [{"prompt_ids": (260, 271, 300, 301, 272), "answer_ids": (400,)}]
    tokens, segments, eligible, supervised = train.build_batch(
        rows, proto.EXPERIMENT_A, "M2_EXTRA_FROZEN", torch=torch,
        device=torch.device("cpu"))
    from v5_training.production_backend import ProductionTrainingBackend
    from v5_training.state import CursorState
    backend = ProductionTrainingBackend(
        model=model, optimizer=optimizers["view"], bos_id=2, pad_id=0,
        device=torch.device("cpu"), schedule=lambda cumulative_tokens: proto.LR,
        bfloat16_autocast=False, torch_module=torch,
        activation_checkpointing=False)
    embedding = model.embedding.weight
    for update in range(3):
        embedding.grad = torch.zeros_like(embedding)
        ctx = backend.begin_update(type("S", (), {"cumulative_tokens": 0})())
        ctx = backend.accumulate_microstep(
            ctx, tokens=tokens, segment_ids=segments, eligible=eligible,
            tokens_by_source={"mux": supervised}, planned_total=supervised)
        fxm.apply_decay_then_zero_frozen(model, "M2_EXTRA_FROZEN", torch=torch)
        optimizers["rows"].step()
        backend.finish_update(type("S", (), {"cumulative_tokens": 0})(), ctx,
                              planned_total=supervised,
                              cursor=CursorState("s", "d", update + 1, 0, 0))
    after = embedding.detach()
    assert torch.equal(after[list(fxm.EXTRA_ROWS)], init[list(fxm.EXTRA_ROWS)]), \
        "frozen extra rows must keep init bytes"
    assert not torch.equal(after[list(fxm.ACTIVE_ROWS)],
                           init[list(fxm.ACTIVE_ROWS)]), \
        "active rows must move"


def test_m3_masks_denominator_only_during_training():
    torch.manual_seed(2)
    model = fxm.build_model(73011, "M3_EXTRA_FROZEN_MASKED", torch=torch,
                            device=torch.device("cpu"))
    ids = torch.tensor([[2, 260, 300, 272]], dtype=torch.long)
    from v5_model.core import packed_layout
    model.train()
    positions, mask = packed_layout(torch.tensor([[0, 0, 0, 0]]),
                                    torch_module=torch)
    train_logits = model(ids, positions, mask)
    assert bool((train_logits[..., list(fxm.EXTRA_ROWS)] == fxm.MASK_LOGIT_VALUE)
                .all()), "training forward must mask extra rows"
    assert float(train_logits[..., :128].abs().sum()) > 0
    model.eval()
    eval_logits = model(ids, positions, mask)
    assert not bool((eval_logits[..., list(fxm.EXTRA_ROWS)] == fxm.MASK_LOGIT_VALUE)
                    .any()), "evaluation must use the full denominator"


# -- protocol ------------------------------------------------------------------

def test_causal_matrix_contrasts_are_isolated():
    proto.assert_contrast_isolation()  # must not raise


def test_queue_locality_and_totals():
    queue = proto.queue_assignment()
    assert proto.total_official_arms() == 24
    assert len(queue["GPU0"]) == len(queue["GPU1"]) == 12
    # matched-seed GPU locality: each (experiment, bundle) group runs on ONE gpu
    for experiment in (proto.EXPERIMENT_A, proto.EXPERIMENT_B):
        for bundle in proto.SEED_BUNDLES:
            gpus = {gpu for gpu, jobs in queue.items()
                    if any(job["seed_bundle"] == bundle and
                           job["experiment"] == experiment for job in jobs)}
            assert len(gpus) == 1, (experiment, bundle, gpus)


def test_paired_verdict_all_paths():
    bundles = list(proto.SEED_BUNDLES)
    success = proto.paired_verdict(
        deltas={b: 0.08 for b in bundles},
        endpoint_gaps={b: 0.06 for b in bundles})
    assert success["verdict"] == "SUCCESS"
    reverse = proto.paired_verdict(
        deltas={b: -0.08 for b in bundles},
        endpoint_gaps={b: -0.06 for b in bundles})
    assert reverse["verdict"] == "REVERSE_EFFECT"
    null = proto.paired_verdict(
        deltas={b: 0.004 for b in bundles},
        endpoint_gaps={b: 0.004 for b in bundles})
    assert null["verdict"] == "NULL"
    incomplete = proto.paired_verdict(
        deltas={bundles[0]: 0.08, bundles[1]: 0.08},
        endpoint_gaps={bundles[0]: 0.06, bundles[1]: 0.06})
    assert incomplete["verdict"] == "INCONCLUSIVE"
    mixed = proto.paired_verdict(
        deltas={bundles[0]: 0.30, bundles[1]: -0.30,
                bundles[2]: 0.30, bundles[3]: -0.30},
        endpoint_gaps={b: 0.30 for b in bundles})
    assert mixed["verdict"] == "PARTIAL_OR_INTERACTION"


# -- worker contract -------------------------------------------------------------

def _surface_file(tmp_path, tokenizer=None) -> Path:
    path = tmp_path / "surface.json"
    if not path.exists():
        manifest = build_surface(seed=proto.SEED_BUNDLES[0], tokenizer=tokenizer)
        path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_worker_rejects_unregistered_arm(tmp_path):
    surface = _surface_file(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-m", "tools.formation_mux_001_worker",
         "--experiment", "CS-MECH-002", "--arm", "V4096_REVIVAL",
         "--seed-bundle", "73011", "--surface", str(surface),
         "--out", str(tmp_path / "out"), "--device", "cpu"],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert completed.returncode == 4
    assert "GLOBAL_FAILURE" in completed.stdout


def test_worker_rejects_unregistered_bundle(tmp_path):
    surface = _surface_file(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-m", "tools.formation_mux_001_worker",
         "--experiment", "REP-FORM-003A", "--arm", "R0_PRODUCTION_BPE",
         "--seed-bundle", "99999999", "--surface", str(surface),
         "--out", str(tmp_path / "out"), "--device", "cpu"],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert completed.returncode == 4


def test_worker_cpu_tiny_e2e_and_resume(tmp_path):
    """PHASE-1 qualification shape: 2 real updates through the ACTUAL worker
    executable path, then an exact resume to 4 through the same checkpoint
    identity contract."""

    surface = _surface_file(tmp_path)
    out = tmp_path / "out"
    base = [sys.executable, "-m", "tools.formation_mux_001_worker",
            "--experiment", "CS-MECH-002", "--arm", "M2_EXTRA_FROZEN",
            "--seed-bundle", "73011", "--surface", str(surface),
            "--out", str(out), "--device", "cpu", "--engineering-only"]
    first = subprocess.run(base + ["--updates-override", "2"], cwd=REPO,
                           capture_output=True, text=True, timeout=600)
    assert first.returncode == 0, first.stdout[-2000:] + first.stderr[-2000:]
    assert "WORKER_RECEIPT" in first.stdout
    receipt = json.loads(first.stdout.split("WORKER_RECEIPT ", 1)[1]
                         .splitlines()[0])
    assert receipt["engineering_only"] is True
    second = subprocess.run(base + ["--updates-override", "4"], cwd=REPO,
                            capture_output=True, text=True, timeout=600)
    assert second.returncode == 0, second.stdout[-2000:] + second.stderr[-2000:]
    assert "RESUME" in second.stdout, "checkpoint resume must log RESUME"
    result = json.loads((out / "CS-MECH-002" / "M2_EXTRA_FROZEN" / "S1" /
                         "ARM_RESULT.json").read_text())
    assert result["updates"] == 4


# -- coordinator: state, failure policy, sealed firewall, packaging -----------------

def test_write_state_is_locked_and_atomic(tmp_path):
    from tools.formation_mux_001_kaggle_operator import write_state
    state_dir = tmp_path / "state"
    write_state(state_dir, lambda s: {**s, "a": 1})
    write_state(state_dir, lambda s: {**s, "b": 2})
    state = json.loads((state_dir / "CAMPAIGN_STATE.json").read_text())
    assert state["a"] == 1 and state["b"] == 2
    assert not (state_dir / "state.lock").exists()
    assert not (state_dir / "CAMPAIGN_STATE.tmp").exists()


def test_hardware_gate_blocks_non_t4_and_official_requires_two():
    from tools.formation_mux_001_kaggle_operator import (
        hardware_gate, require_official_hardware, HardwareGateError)

    class FakeProperties:
        def __init__(self, total): self.total_memory = total

    class FakeCuda:
        def __init__(self, count, names, vram):
            self._count, self._names, self._vram = count, names, vram
        def is_available(self): return self._count > 0
        def device_count(self): return self._count
        def get_device_name(self, i): return self._names[i]
        def get_device_properties(self, i): return FakeProperties(self._vram[i])

    class Torch1:
        __version__ = "test"
        version = type("V", (), {"cuda": "12.1"})()
        cuda = FakeCuda(1, ["NVIDIA A100"], [80e9])

    gate = hardware_gate(Torch1())
    assert gate["t4_class"] is False and gate["device_count"] == 1
    with pytest.raises(HardwareGateError, match="GPU T4 x2"):
        require_official_hardware(Torch1())

    class Torch2:
        __version__ = "test"
        version = type("V", (), {"cuda": "12.1"})()
        cuda = FakeCuda(2, ["NVIDIA Tesla T4", "NVIDIA Tesla T4"],
                        [15.8e9, 15.8e9])

    assert hardware_gate(Torch2())["t4_class"] is True


def test_sealed_firewall_starting_twice_fails_closed(tmp_path):
    from tools.formation_mux_001_kaggle_operator import (
        set_sealed_marker, GlobalIntegrityError)
    set_sealed_marker(tmp_path, "CS-MECH-002", "NOT_CONSUMED")
    set_sealed_marker(tmp_path, "CS-MECH-002", "STARTED")
    with pytest.raises(GlobalIntegrityError, match="fail closed"):
        set_sealed_marker(tmp_path, "CS-MECH-002", "STARTED")
    set_sealed_marker(tmp_path, "CS-MECH-002", "COMPLETE")  # legal transition


def test_sealed_finalization_refuses_incomplete_arms(tmp_path):
    from tools.formation_mux_001_kaggle_operator import (
        finalize_sealed, write_state, campaign_state)
    import torch as torch_mod
    write_state(tmp_path, lambda s: {**campaign_state(tmp_path),
                                     "arms": {"CS-MECH-002/M0_STANDARD/S1":
                                              "COMPLETE"}})
    results = finalize_sealed(surface=tmp_path / "none.json", out=tmp_path,
                              torch=torch_mod)
    assert results["CS-MECH-002"]["verdict"] == "INCONCLUSIVE"
    assert not (tmp_path / "CS-MECH-002" / "FINAL_RESULT.json").exists()
    state = campaign_state(tmp_path)
    assert state["sealed"]["CS-MECH-002"] == "NOT_CONSUMED"


def test_packaging_bundle(tmp_path):
    from tools.formation_mux_001_kaggle_operator import package_results
    out = tmp_path / "FORMATION_MUX_001"
    out.mkdir()
    (out / "CAMPAIGN_STATE.json").write_text(json.dumps({"status": "PARTIAL_SESSION"}))
    (out / "ENVIRONMENT.json").write_text("{}")
    (out / "CALIBRATION_RECEIPT.json").write_text("{}")
    arm_dir = out / "CS-MECH-002" / "M0_STANDARD" / "S1"
    arm_dir.mkdir(parents=True)
    (arm_dir / "ARM_RESULT.json").write_text(json.dumps({"status": "COMPLETE"}))
    (arm_dir / "FAILURE.json").write_text(json.dumps({"class": "ARM_LOCAL"}))
    bundle = package_results(out, source_commit="deadbeef")
    import zipfile
    with zipfile.ZipFile(bundle["path"]) as zf:
        names = set(zf.namelist())
    assert "README.txt" in names and "SOURCE_COMMIT.txt" in names
    assert "FORMATION_MUX_001/CS-MECH-002/M0_STANDARD/S1/FAILURE.json" in names
    assert Path(bundle["sha256"]) is not None
