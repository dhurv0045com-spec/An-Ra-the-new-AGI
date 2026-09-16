"""Bounded post-training integration coverage, not scientific pilot evidence.

Tiny CPU model, synthetic public rows, zero optimizer steps. Only model geometry,
generation cap and the training entrypoint are substituted; checkpoint writing,
identity-checked reload, diagnostics, lane serialization, pairing and packaging
execute repository code. No tokenizer, downloads, sealed data or campaign launch.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture
def runtime(monkeypatch):
    import torch
    from anra_v5 import formation_mux_train_v2 as base
    from anra_v5 import tie_role_model_v1 as model
    from v5_experiments import tie_role_protocol_v1 as protocol
    from tools import tie_role_pilot_001_colab_v1 as pilot

    # tie_role_train binds shared module globals on import and on every call.
    # Register their original values before import so pytest restores them even
    # when the import itself mutates them.
    for name in ("fxm", "proto", "_make_model_and_optimizers", "_save_checkpoint"):
        monkeypatch.setattr(base, name, getattr(base, name))
    from anra_v5 import tie_role_train_v1 as train

    tiny = replace(
        model.spec(), vocabulary_size=32, width=16, layers=1,
        query_heads=2, kv_heads=1, head_dimension=8, ffn_width=32,
        context_length=64,
    )
    assert tiny.parameter_receipt().total < 10_000
    monkeypatch.setattr(model, "spec", lambda: tiny)
    monkeypatch.setattr(protocol, "MAX_GENERATION_TOKENS", 4)
    # The production saver queries CUDA RNG state even for CPU checkpoints.
    # Keep this fixture entirely CPU-only, including on CUDA-enabled hosts.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        try:
            yield SimpleNamespace(
                torch=torch, device=torch.device("cpu"), base=base,
                model=model, protocol=protocol, train=train, pilot=pilot,
            )
        finally:
            torch.set_num_threads(previous_threads)


def _public():
    rows = [{"family": "identity", "prompt_ids": [4, 5], "answer_ids": [5]}
            for _ in range(80)]
    return {
        "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "splits": {"development": rows},
    }


def _checkpoint(rt, out, public, seed, arm):
    """Write a real, explicitly untrained checkpoint in the pilot lane layout."""
    path = (out / "TRAIN" / f"SEED_{seed}" / arm /
            rt.protocol.EXPERIMENT_A / arm / f"CAL{seed}" / "resume.pt")
    model = rt.model.build_model(seed, arm, torch=rt.torch, device=rt.device)
    optimizers = rt.model.make_optimizers(model, arm, torch=rt.torch, lr=rt.protocol.LR)
    payload = {
        "experiment": rt.protocol.EXPERIMENT_A, "arm": arm,
        "seed_bundle": seed, "data_manifest_sha256": public["sha256"],
        "protocol_sha256": rt.protocol.protocol_sha(rt.protocol.EXPERIMENT_A),
        "engineering_only": True, "updates": 0,
        "formation": rt.base._formation_summary([], rt.protocol.A_ELIGIBLE_FROM_UPDATE),
    }
    rt.base._save_checkpoint(
        path, model=model, optimizers=optimizers, torch=rt.torch, payload=payload,
    )
    return path, payload, rt.pilot._model_sha(model)


def _lane(rt, out, public, seed, arm):
    return rt.pilot._train_lane(
        seed=seed, arm=arm, out=out, public=public, train=rt.train,
        proto=rt.protocol, fxm=rt.model, torch=rt.torch, device=rt.device,
    )


def test_checkpoint_diagnostics_decision_and_package_roundtrip(runtime, monkeypatch, tmp_path):
    rt = runtime
    public = _public()
    results = {}
    checkpoints = {}
    for seed in rt.pilot.SEEDS:
        for arm in (rt.pilot.CONTROL, rt.pilot.TREATMENT):
            path, payload, model_sha = _checkpoint(rt, tmp_path, public, seed, arm)
            results[seed, arm] = payload
            checkpoints[seed, arm] = (path, model_sha)

    calls = []

    def checkpoint_only(**kwargs):
        # The sole training replacement: no train_arm, optimizer step or launch.
        key = (kwargs["seed_bundle"], kwargs["arm"])
        calls.append(key)
        assert kwargs["surface"] is public
        assert kwargs["engineering_only"] is True
        return results[key]

    monkeypatch.setattr(rt.train, "train_arm", checkpoint_only)
    pairs = []
    for seed in rt.pilot.SEEDS:
        lanes = {}
        for arm in (rt.pilot.CONTROL, rt.pilot.TREATMENT):
            path, expected_sha = checkpoints[seed, arm]
            restored = rt.train.load_model_for_evaluation(
                path, experiment=rt.protocol.EXPERIMENT_A, arm=arm,
                seed_bundle=seed, data_manifest_sha256=public["sha256"],
                torch=rt.torch, device=rt.device,
            )
            assert rt.pilot._model_sha(restored) == expected_sha
            assert not restored.training
            assert rt.model.get_gradient_scales(restored) == rt.protocol.gradient_scales(arm)
            del restored
            lane = _lane(rt, tmp_path, public, seed, arm)
            assert lane["updates"] == 0
            assert lane["greedy_identity"]["n"] == 80
            assert lane["teacher_forced_identity"]["target_tokens"] == 160
            for field in ("mean_target_rank", "mean_target_margin", "target_token_accuracy"):
                assert math.isfinite(lane["teacher_forced_identity"][field])
            grad = lane["gradient_diagnostic"]
            assert grad["rows"] == 16
            assert grad["supervised_tokens"] == 32
            assert grad["processed_tokens"] == 80
            assert grad["input_gradient_norm"] > 0
            assert grad["output_gradient_norm"] > 0
            assert all(math.isfinite(value) for value in grad.values())
            lane_file = tmp_path / "TRAIN" / f"SEED_{seed}" / arm / "PILOT_LANE_RESULT.json"
            assert json.loads(lane_file.read_text(encoding="utf-8")) == lane
            lanes[arm] = lane
        pairs.append(rt.pilot._pairwise(lanes[rt.pilot.CONTROL], lanes[rt.pilot.TREATMENT]))
    assert len(calls) == 4 and len(set(calls)) == 4
    decision = rt.pilot._decide(pairs)
    assert decision["decision"] == "DO_NOT_RUN_FULL_TIE_ROLE"
    result = {"engineering_only": True, "training_executed": False,
              "pairs": pairs, **decision}
    rt.pilot._atomic_json(tmp_path / "TIE_ROLE_PILOT_RESULT.json", result)

    # Repack twice to verify archive retry excludes itself and preserves content.
    for _ in range(2):
        packages = rt.pilot._package(tmp_path)
        for item in packages.values():
            archive = Path(item["path"])
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            assert digest == item["sha256"]
            assert archive.stat().st_size == item["bytes"]
            assert Path(str(archive) + ".sha256").read_text().split()[0] == digest
        with zipfile.ZipFile(packages["results"]["path"]) as zf:
            assert zf.testzip() is None
            assert json.loads(zf.read("TIE_ROLE_PILOT_RESULT.json")) == result
            assert not any(name.endswith(("resume.pt", ".zip", ".sha256")) for name in zf.namelist())
            assert sum(name.endswith("PILOT_LANE_RESULT.json") for name in zf.namelist()) == 4
        with zipfile.ZipFile(packages["checkpoints"]["path"]) as zf:
            assert zf.testzip() is None
            assert sum(name.endswith("resume.pt") for name in zf.namelist()) == 4
            for path, _sha in checkpoints.values():
                name = path.relative_to(tmp_path).as_posix()
                assert zf.read(name) == path.read_bytes()
                receipt_name = str(Path(name).with_name("CHECKPOINT_RECEIPT.json")).replace("\\", "/")
                receipt = json.loads(zf.read(receipt_name))
                assert receipt["sha256"] == hashlib.sha256(zf.read(name)).hexdigest()


@pytest.mark.parametrize("field", [
    "experiment", "arm", "seed_bundle", "data_manifest_sha256", "protocol_sha256",
])
def test_checkpoint_identity_mismatch_stops_before_diagnostics(runtime, monkeypatch, tmp_path, field):
    rt = runtime
    public = _public()
    seed, arm = rt.pilot.SEEDS[0], rt.pilot.CONTROL
    path, payload, _sha = _checkpoint(rt, tmp_path, public, seed, arm)
    body = rt.torch.load(path, map_location="cpu", weights_only=False)
    body[field] = "wrong-test-identity"
    rt.torch.save(body, path)
    monkeypatch.setattr(rt.train, "train_arm", lambda **kwargs: payload)

    def forbidden(*args, **kwargs):
        pytest.fail("diagnostics ran despite invalid checkpoint")

    monkeypatch.setattr(rt.pilot, "_greedy", forbidden)
    with pytest.raises(RuntimeError, match=f"checkpoint identity mismatch {field}"):
        _lane(rt, tmp_path, public, seed, arm)
    assert not list(tmp_path.rglob("PILOT_LANE_RESULT.json"))
    assert not list(tmp_path.rglob("*.zip"))


def test_missing_checkpoint_fails_without_result(runtime, monkeypatch, tmp_path):
    rt = runtime
    monkeypatch.setattr(rt.train, "train_arm", lambda **kwargs: {})
    with pytest.raises(RuntimeError, match="checkpoint missing"):
        _lane(rt, tmp_path, _public(), rt.pilot.SEEDS[0], rt.pilot.CONTROL)
    assert not list(tmp_path.rglob("PILOT_LANE_RESULT.json"))
