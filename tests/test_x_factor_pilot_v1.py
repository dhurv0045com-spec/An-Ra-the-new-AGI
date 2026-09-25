from __future__ import annotations

import os
from pathlib import Path

import pytest

from v5_experiments import x_factor_pilot_protocol_v1 as protocol


def test_protocol_is_executable_and_v24576_is_fixed():
    payload = protocol.protocol_payload()
    assert payload["physical_vocabulary"] == 24_576
    assert payload["arms"] == ["TIED_DENSE", "UNTIED_DENSE", "LEV_UNTIED"]
    assert payload["primary_arm"] == "LEV_UNTIED"
    assert payload["x_factor"]["base"] ** payload["x_factor"]["components"] >= 24_576
    assert len(protocol.protocol_sha256()) == 64
    assert protocol.arm_parameterization("TIED_DENSE")["output"] == "tied_dense"
    assert protocol.arm_parameterization("UNTIED_DENSE")["output"] == "independent_dense"
    assert protocol.arm_parameterization("LEV_UNTIED")["structured_code"] is True


def test_positive_control_is_deterministic_and_nontrivial():
    first = protocol.build_positive_control_surface()
    second = protocol.build_positive_control_surface()
    assert first == second
    assert len(first["training"]) == 512
    assert len(first["development"]) == 128
    protocol.validate_positive_control_surface(first)
    mapping = {int(key): int(value) for key, value in first["mapping"].items()}
    assert len(mapping) == 32
    assert all(key != value for key, value in mapping.items())
    assert set(mapping.values()) == set(protocol.CONTROL_OUTPUTS)
    train_keys = {tuple(row["prompt_ids"]) + tuple(row["answer_ids"]) for row in first["training"]}
    dev_keys = {tuple(row["prompt_ids"]) + tuple(row["answer_ids"]) for row in first["development"]}
    assert train_keys.isdisjoint(dev_keys)
    assert all(row["answer_ids"][0] in protocol.CONTROL_OUTPUTS for row in first["training"] + first["development"])


def test_official_rows_use_production_bpe_ids():
    from anra_v5 import x_factor_pilot_train_v1 as train

    row = {
        "prompt_ids": [260, 300],
        "answer_ids": [301],
        "r0_prompt_ids": [2, 100, 101],
        "r0_answer_ids": [102, 3],
    }
    assert train._encode_row(row, "official") == ([2, 100, 101], [102, 3])
    assert train._encode_row(row, "control") == ([2, 260, 300], [301, 3])


def test_positive_control_gate_and_paired_verdict_are_fail_closed():
    passing = {
        arm: {"status": "COMPLETE", "processed_tokens": 100, "supervised_tokens": 50, "data_order_sha256": "a" * 64, "formation": {"endpoint": 0.75, "formation_auc": 0.70}}
        for arm in protocol.ARMS
    }
    assert protocol.positive_control_gate(passing)["status"] == "PASS"
    failing = dict(passing)
    failing[protocol.PRIMARY_ARM] = {"status": "COMPLETE", "formation": {"endpoint": 0.49, "formation_auc": 0.70}}
    assert protocol.positive_control_gate(failing)["status"] == "FAIL"
    auc = {seed: 0.08 for seed in protocol.MODEL_SEEDS}
    endpoint = {seed: 0.12 for seed in protocol.MODEL_SEEDS}
    verdict = protocol.paired_verdict(formation_auc_deltas=auc, sealed_endpoint_gaps=endpoint)
    assert verdict["verdict"] == "SUCCESS"
    assert protocol.paired_verdict(formation_auc_deltas={}, sealed_endpoint_gaps=endpoint)["verdict"] == "INCONCLUSIVE"


def test_factorized_surface_and_worker_command_are_remote_only():
    from tools import x_factor_pilot_001_kaggle_operator_v1 as operator

    command = operator.worker_command(
        mode="official",
        arm=protocol.PRIMARY_ARM,
        seed=protocol.MODEL_SEEDS[0],
        surface=Path("surface.json"),
        out=Path("out"),
    )
    assert "--device" in command and command[command.index("--device") + 1] == "cuda"
    assert "--target-updates" not in command
    class TorchStub:
        __version__ = "stub"

        class version:
            cuda = None

        class cuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def device_count():
                return 0

    assert operator.hardware_receipt(TorchStub)["passed"] is False


def test_colab_operator_uses_one_gpu_without_changing_science():
    from tools import x_factor_pilot_001_colab_operator_v1 as operator

    jobs = operator.one_gpu_jobs()
    assert len(jobs) == len(protocol.ARMS) * len(protocol.MODEL_SEEDS)
    assert all(job["gpu"] == 0 for job in jobs)
    class TorchStub:
        __version__ = "stub"

        class version:
            cuda = None

        class cuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def device_count():
                return 0

    assert operator.single_gpu_receipt(TorchStub)["passed"] is False


def test_model_architecture_and_initial_forward_equivalence():
    import gc
    import torch
    from anra_v5 import x_factor_pilot_model_v1 as model_module
    from v5_model.core import packed_layout

    device = torch.device("cpu")
    tied = model_module.build_model("TIED_DENSE", 17, torch_module=torch, device=device)
    untied = model_module.build_model("UNTIED_DENSE", 17, torch_module=torch, device=device)
    for name, value in tied.state_dict().items():
        if name.startswith("blocks.") or name.startswith("final_norm."):
            torch.testing.assert_close(value, untied.state_dict()[name], rtol=0.0, atol=0.0)
    torch.testing.assert_close(tied.input_layer.embedding.weight, untied.input_layer.embedding.weight, rtol=0.0, atol=0.0)
    torch.testing.assert_close(tied.input_layer.embedding.weight, untied.output_head.weight, rtol=0.0, atol=0.0)
    assert tied.input_layer.embedding.weight.data_ptr() != untied.output_head.weight.data_ptr()
    tokens = torch.tensor([[2, 260, 269, 3]], dtype=torch.long)
    segments = torch.zeros_like(tokens)
    positions, mask = packed_layout(segments, torch_module=torch)
    with torch.no_grad():
        tied_logits = tied(tokens, positions, mask)
        untied_logits = untied(tokens, positions, mask)
    torch.testing.assert_close(tied_logits, untied_logits, rtol=0.0, atol=1e-7)
    assert model_module.parameterization_receipt(tied)["parameter_count"] == 14_163_200
    assert model_module.parameterization_receipt(untied)["parameter_count"] == 20_454_656
    del tied, untied
    gc.collect()
    lev = model_module.build_model("LEV_UNTIED", 17, torch_module=torch, device=device)
    assert lev.embedding is None
    assert lev.output_head.weight.shape == (24_576, 256)
    assert len(lev.input_layer.codebooks) == 3
    with torch.no_grad():
        lev_logits = lev(tokens, positions, mask)
    assert tuple(lev_logits.shape) == (1, 4, 24_576)
    receipt = model_module.parameterization_receipt(lev)
    assert receipt["parameter_count"] == 14_210_048
    assert receipt["output_alias"] is False


@pytest.mark.skipif(os.environ.get("X_FACTOR_SMOKE") != "1", reason="remote trainer smoke only")
def test_remote_trainer_checkpoint_resume(tmp_path):
    import torch
    from anra_v5 import x_factor_pilot_train_v1 as train

    torch.set_num_threads(1)
    surface = protocol.build_positive_control_surface()
    root = tmp_path / "run"
    first = train.train_arm(
        mode="canary",
        arm=protocol.PRIMARY_ARM,
        seed=protocol.CALIBRATION_SEED,
        surface=surface,
        out_dir=root,
        torch=torch,
        device=torch.device("cpu"),
        target_updates=2,
        stop_after=1,
    )
    assert first["status"] == "PARTIAL"
    checkpoint = root / "canary" / protocol.PRIMARY_ARM / protocol.seed_label(protocol.CALIBRATION_SEED) / "resume.pt"
    first_sha = train.file_sha256(checkpoint)
    second = train.train_arm(
        mode="canary",
        arm=protocol.PRIMARY_ARM,
        seed=protocol.CALIBRATION_SEED,
        surface=surface,
        out_dir=root,
        torch=torch,
        device=torch.device("cpu"),
        target_updates=2,
        stop_after=2,
    )
    assert second["status"] == "COMPLETE"
    assert second["resume_count"] == 1
    body = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert body["resume_checkpoint_sha256"] == first_sha
    assert second["resume_checkpoint_sha256"] == train.file_sha256(checkpoint)
    assert second["updates"] == 2
