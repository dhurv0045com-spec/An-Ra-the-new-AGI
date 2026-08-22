"""Integration tests for the REAL training resume path and runtime invariants.

These test the executable path, not helpers:
- _restore_training_state installs checkpoint weights into the CALLER's model
- pack identity binding (same-pack vs new-pack fail-closed)
- best-checkpoint snapshot preserves the actual best state
- non-idempotent tools execute exactly once; trace == Core input
- family gates prevent blocked families from executing
"""

import pytest
import torch

from training.resume import resolve_pack_horizon


def _make_checkpoint(tmp_path, name: str, marker_delta: float = 0.0):
    from anra_core.config import CANONICAL_CONFIG
    from anra_core.model import AnRaCore
    from anra_core.tokenizer import V4Tokenizer

    model = AnRaCore(CANONICAL_CONFIG)
    state = model.state_dict()
    if marker_delta:
        probe = "blocks.0.norm_1.weight"
        state[probe] = state[probe] + marker_delta
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first_parameter = next(model.parameters())
    optimizer.state[first_parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros_like(first_parameter),
        "exp_avg_sq": torch.zeros_like(first_parameter),
    }
    tokenizer = V4Tokenizer.load_canonical()
    payload = {
        "checkpoint_artifact_class": "full_resume",
        "checkpoint_schema_version": 1,
        "global_step": 20_000,
        "pack_step": 1_000,
        "pack_manifest_sha256": "a" * 64,
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "tokenizer_contract": {
            "available": True,
            **tokenizer.identity(probe_count=500),
        },
        "metrics": {},
    }
    path = tmp_path / name
    torch.save(payload, path)
    return path, state


def test_resume_installs_checkpoint_weights_into_caller_model(tmp_path) -> None:
    """P0 integration proof: the trainer's model must contain the exact
    checkpoint parameters after resume, before any optimizer step."""
    from anra_core.config import CANONICAL_CONFIG
    from anra_core.model import AnRaCore
    from training.resume import RESUME_SAME_PACK as SAME_PACK
    from training.resume import restore_training_state as _restore_training_state

    ckpt_path, ckpt_state = _make_checkpoint(tmp_path, "parent.pt", marker_delta=0.5)
    probe = "blocks.0.norm_1.weight"

    # Fresh randomly-initialized model = what the trainer constructs.
    trainer_model = AnRaCore(CANONICAL_CONFIG)
    assert not torch.equal(trainer_model.state_dict()[probe], ckpt_state[probe])

    optimizer = torch.optim.AdamW(trainer_model.parameters(), lr=1e-3)
    restored = _restore_training_state(
        str(ckpt_path), trainer_model, optimizer,
        mode=SAME_PACK, current_pack_manifest_sha256="a" * 64,
    )

    # The invariant: immediately before the first optimizer step, the trainer's
    # model contains the exact checkpoint parameters.
    for key in ("blocks.0.norm_1.weight", "token_embedding_table.weight"):
        assert torch.equal(
            trainer_model.state_dict()[key], ckpt_state[key]
        ), f"{key} was not restored into the caller's model"
    assert restored.global_step == 20_000
    assert restored.pack_step == 1_000
    assert restored.mode == "same_pack"


def test_same_pack_resume_refuses_different_pack_identity(tmp_path) -> None:
    from anra_core.config import CANONICAL_CONFIG
    from anra_core.model import AnRaCore
    from training.resume import RESUME_SAME_PACK as SAME_PACK
    from training.resume import restore_training_state as _restore_training_state

    ckpt_path, _state = _make_checkpoint(tmp_path, "p.pt")
    model = AnRaCore(CANONICAL_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(RuntimeError, match="different pack"):
        _restore_training_state(
            str(ckpt_path), model, optimizer,
            mode=SAME_PACK, current_pack_manifest_sha256="b" * 64,
        )


def test_same_pack_resume_refuses_unverifiable_pack_step(tmp_path) -> None:
    """pack_step without pack identity is meaningless - fail closed."""
    from anra_core.config import CANONICAL_CONFIG
    from anra_core.model import AnRaCore
    from training.resume import RESUME_SAME_PACK as SAME_PACK
    from training.resume import restore_training_state as _restore_training_state

    ckpt_path, _ = _make_checkpoint(tmp_path, "p.pt")
    payload = torch.load(ckpt_path, weights_only=False)
    del payload["pack_manifest_sha256"]
    torch.save(payload, ckpt_path)

    model = AnRaCore(CANONICAL_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(RuntimeError, match="cannot be verified"):
        _restore_training_state(str(ckpt_path), model, optimizer, mode=SAME_PACK)


def test_new_pack_parent_resets_pack_step(tmp_path) -> None:
    from anra_core.config import CANONICAL_CONFIG
    from anra_core.model import AnRaCore
    from training.resume import RESUME_NEW_PACK_PARENT as NEW_PACK_PARENT
    from training.resume import restore_training_state as _restore_training_state

    ckpt_path, state = _make_checkpoint(tmp_path, "p.pt")
    model = AnRaCore(CANONICAL_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    restored = _restore_training_state(
        str(ckpt_path), model, optimizer, mode=NEW_PACK_PARENT,
    )
    assert restored.pack_step == 0, "new pack must start at schedule position 0"
    # Model weights still restored (it IS the parent).
    assert torch.equal(
        model.state_dict()["blocks.0.norm_1.weight"], state["blocks.0.norm_1.weight"]
    )


def test_step_accounting_global_plus_fresh_pack() -> None:
    horizon = resolve_pack_horizon(
        global_step=20_000, restored_pack_step=0,
        token_budget=330_000_000, tokens_per_step=131_072,
    )
    updates = horizon.updates_remaining
    assert updates == 2_517
    final_global = 20_000 + updates
    assert final_global == 22_517


# --------------------------------------------------------------------------
# Non-idempotent tool: exactly one execution; trace == Core input == candidate
# --------------------------------------------------------------------------


def test_non_idempotent_tool_executes_exactly_once() -> None:
    from connector.experiments.cognitive_credit.case import (
        Attempt,
        DecodePolicy,
        PreparedExecution,
        ToolBehavior,
    )

    counter = {"n": 0}

    def clock() -> str:
        counter["n"] += 1
        return f"VALUE-{counter['n']}"

    attempt = Attempt(
        question="q", plan="p",
        tool=ToolBehavior("clock", available=True, execute=clock),
        decode=DecodePolicy(),
    )
    trace_prompt = PreparedExecution.from_attempt(attempt).prompt   # runtime records
    core_prompt = PreparedExecution.from_attempt(attempt).prompt    # completer consumes

    assert counter["n"] == 1, f"tool executed {counter['n']} times"
    assert "VALUE-1" in trace_prompt
    assert "VALUE-1" in core_prompt
    assert "VALUE-2" not in core_prompt, "Core consumed a second execution"
    assert trace_prompt == core_prompt


def test_tool_output_precedes_answer_boundary() -> None:
    from connector.experiments.cognitive_credit.case import (
        Attempt,
        DecodePolicy,
        PreparedExecution,
        ToolBehavior,
    )

    attempt = Attempt(
        question="What is 20 + 22?", plan="Read the output.",
        tool=ToolBehavior("calc", available=True, execute=lambda: "42"),
        decode=DecodePolicy(),
    )
    prompt = PreparedExecution.from_attempt(attempt).prompt
    assert prompt.index("<tool_output>42</tool_output>") < prompt.index("<answer>")


# --------------------------------------------------------------------------
# Family gating: blocked families produce zero Core executions.
# --------------------------------------------------------------------------


class CountingCompleter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, attempt):
        self.calls += 1
        from connector.experiments.cognitive_credit.case import CompletionResult

        return CompletionResult(texts=("no usable output",), n_executions=1)


def test_blocked_families_execute_zero_cases() -> None:
    from connector.experiments.cognitive_credit.runner import run_experiment
    from connector.experiments.cognitive_credit.suite import FAMILIES

    completer = CountingCompleter()
    runnable = {"missing_knowledge"}  # only one family allowed
    summary = run_experiment(completer, runnable_families=runnable)
    assert summary["not_evaluated_families"] == sorted(set(FAMILIES) - runnable)
    assert summary["runnable_families"] == sorted(runnable)
    assert summary["n_cases"] == 5  # only the runnable family's cases ran
    # Zero executions for blocked families: 5 cases x (1 baseline + 7 arms).
    assert completer.calls == 5 * 8
    # Blocked families never contaminate the metrics denominators.
    assert summary["total_core_executions"] == 5 * 8


# --------------------------------------------------------------------------
# Evidence identity: dirty evaluation requires diff hash; commits separated.
# --------------------------------------------------------------------------


def test_evidence_requires_evaluator_commit_and_diff_hash_when_dirty() -> None:
    from evaluation.evidence import EvidenceIdentity

    base = dict(
        experiment_schema="anra-evidence/v1",
        checkpoint_source_commit="a",
        checkpoint_file_sha256="f" * 64,
        checkpoint_parameter_sha256="p" * 64,
        global_step=100,
        tokenizer_identity="v4_32k",
        architecture_identity="arch",
        execution_profile="cpu",
        decode_policy={"temperature": 0.0},
    )
    with pytest.raises(ValueError, match="evaluation_source_commit"):
        EvidenceIdentity(**base).validate()

    dirty = EvidenceIdentity(evaluation_source_commit="b", evaluation_dirty=True, **base)
    with pytest.raises(ValueError, match="diff hash"):
        dirty.validate()

    ok = EvidenceIdentity(
        evaluation_source_commit="b", evaluation_dirty=True,
        evaluation_diff_sha256="d" * 64, **base,
    )
    ok.validate()  # dirty but reproducible via recorded diff
