"""HORM-004 prerequisite tests: session checkpoint/resume + firewall-safe appraisal."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from types import SimpleNamespace

import pytest
import torch

from v5_contracts.model_spec import ModelSpec
from v5_identity import (
    HORMONES,
    HormonalProjection,
    HormonalState,
    appraise_committed,
)
from v5_identity.attention_patch import HormonalAttentionPatch
from v5_model.core import initialize, packed_layout


def _projection(alpha: float) -> HormonalProjection:
    return HormonalProjection(
        weights=tuple(0.1 if n in ("cortisol", "adrenaline") else 0.05 for n in HORMONES),
        bound=0.2,
        raw_alpha=alpha,
    )


def _spec() -> ModelSpec:
    return ModelSpec(
        schema="anra-v5-model-spec/v1", family="dense-decoder-transformer",
        vocabulary_size=64, width=16, layers=2, query_heads=2, kv_heads=1,
        head_dimension=8, ffn_width=32, context_length=16, rope_base=10000.0,
        norm_epsilon=1e-5, tied_embeddings=True, qk_norm=True, qk_norm_affine=True,
        linear_bias=False, dropout=0.0)


def _worked_state() -> HormonalState:
    state = HormonalState.baseline()
    state.appraise("failure")
    state.decay()
    state.appraise("success")
    state.decay()
    return state


class TestSessionCheckpointResume:
    def test_round_trip_is_exact(self) -> None:
        state = _worked_state()
        restored = HormonalState.from_dict(state.to_dict())
        assert restored.vector() == state.vector()
        assert restored.sha256() == state.sha256()

    def test_tampered_payload_refuses_resume(self) -> None:
        payload = _worked_state().to_dict()
        payload["values"] = dict(payload["values"])
        payload["values"]["cortisol"] = 1.5
        with pytest.raises(ValueError, match="hash mismatch"):
            HormonalState.from_dict(payload)

    def test_wrong_schema_or_shape_refuses_resume(self) -> None:
        with pytest.raises(ValueError, match="schema"):
            HormonalState.from_dict({"schema": "other/v1", "values": {}, "sha256": "x"})
        with pytest.raises(ValueError, match="seven hormones"):
            HormonalState.from_dict({
                "schema": "anra-hormonal-state/v1",
                "values": {"dopamine": 0.1}, "sha256": "x"})
        with pytest.raises(ValueError, match="mapping"):
            HormonalState.from_dict(None)

    def test_restored_state_drives_identical_forward(self) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(11)
        model = initialize(_spec(), seed=11).eval()
        tokens = torch.randint(4, 64, (2, 8))
        segment = torch.zeros((2, 8), dtype=torch.int32)
        positions, mask = packed_layout(segment, torch_module=torch)
        projection = _projection(0.5)
        with torch.no_grad():
            patch = HormonalAttentionPatch(model, projection=projection)
            patch.state.values.update(_worked_state().values)
            before_restart = model(tokens, positions, mask)
            saved = patch.state.to_dict()
            patch.restore()
            resumed = HormonalAttentionPatch(model, projection=projection)
            resumed.set_state(HormonalState.from_dict(saved))
            after_restart = model(tokens, positions, mask)
            resumed.restore()
        assert torch.equal(before_restart, after_restart)


class TestLiveAppraisalFirewall:
    def test_scored_correct_true_appraises_success(self) -> None:
        state = HormonalState.baseline()
        label = appraise_committed(
            state, SimpleNamespace(task_id="t1", gold="a", correct=True))
        assert label == "success"
        assert state.values["dopamine"] > 0.10

    def test_scored_correct_false_appraises_failure(self) -> None:
        state = HormonalState.baseline()
        label = appraise_committed(
            state, SimpleNamespace(task_id="t2", gold="b", correct=False))
        assert label == "failure"
        assert state.values["cortisol"] > 0.10

    def test_pre_truth_types_are_rejected(self) -> None:
        state = HormonalState.baseline()
        # VisibleTask shape: prompt + candidates, no `correct`.
        with pytest.raises(ValueError, match="post-truth"):
            appraise_committed(
                state, SimpleNamespace(task_id="t3", prompt="2+2=", candidates=("4",)))
        # CommittedOutput shape: frozen output, no `correct` yet.
        with pytest.raises(ValueError, match="post-truth"):
            appraise_committed(
                state, SimpleNamespace(task_id="t4", output="4"))

    def test_truthy_non_bools_and_missing_ids_are_rejected(self) -> None:
        state = HormonalState.baseline()
        with pytest.raises(ValueError, match="post-truth"):
            appraise_committed(state, SimpleNamespace(task_id="t5", correct=1))
        with pytest.raises(ValueError, match="post-truth"):
            appraise_committed(state, SimpleNamespace(task_id="t6", correct="yes"))
        with pytest.raises(ValueError, match="task id"):
            appraise_committed(state, SimpleNamespace(correct=True))
