"""Regression tests for the boundary fixes and the reference runtime.

Each test pins one bug fixed during the coherence pass:
  1. tokenizer identity is cached (generation hot path was validation-bound);
  2. generate() binds the executor's tokenizer and skips the wasted final pass;
  3. errored intervention arms cannot masquerade as "no intervention helped";
  4. battery outcome labels are real DiagnosisLabel members;
  5. case ids carry no family information;
  6. anra.run(...) executes the full loop and emits verified learning evidence.
"""

from __future__ import annotations

import typing
from pathlib import Path

import pytest

from anra_core.tokenizer import V4Tokenizer
from connector.experiments.cognitive_credit.case import (
    Attempt,
    CompletionResult,
    DiagnosisLabel,
)
from connector.experiments.cognitive_credit.diagnose import (
    OUTCOME_HELPED,
    OUTCOME_NONE,
)
from connector.experiments.cognitive_credit.suite import FAMILIES, build_case

TOKENIZER = Path(__file__).parents[1] / "anra_core" / "assets" / "tokenizer_v4_32k.json"


def test_tokenizer_identity_is_cached_and_immutable() -> None:
    tokenizer = V4Tokenizer.load(TOKENIZER)
    first = tokenizer.identity()
    assert tokenizer._identity_cache, "identity must populate the cache"
    second = tokenizer.identity()
    assert first == second
    # Callers must not be able to poison the cache through the return value.
    first["vocabulary_sha256"] = "tampered"
    assert tokenizer.identity()["vocabulary_sha256"] == second["vocabulary_sha256"]


def test_generate_rejects_foreign_tokenizer() -> None:
    torch = pytest.importorskip("torch")
    from anra_core.config import CoreConfig
    from anra_core.executor import CoreExecutor
    from anra_core.generate import generate
    from anra_core.model import AnRaCore

    config = CoreConfig(
        vocab_size=128, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        head_dim=8, d_ff=64, block_size=32, base_seq_len=32, target_seq_len=32,
        sliding_window=8, full_attention_every=2,
    )
    torch.manual_seed(7)
    executor = CoreExecutor(AnRaCore(config).eval(), tokenizer=V4Tokenizer.load(TOKENIZER))
    other = V4Tokenizer.load(TOKENIZER)
    with pytest.raises(ValueError, match="foreign tokenizer"):
        generate(executor, other, "hello", max_new_tokens=2)


def test_generate_does_not_execute_a_discarded_final_step() -> None:
    torch = pytest.importorskip("torch")
    from anra_core.config import CoreConfig
    from anra_core.executor import CoreExecutor
    from anra_core.generate import generate
    from anra_core.model import AnRaCore

    config = CoreConfig(
        vocab_size=32_768, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        head_dim=8, d_ff=64, block_size=64, base_seq_len=64, target_seq_len=64,
        sliding_window=8, full_attention_every=2,
    )
    torch.manual_seed(7)
    executor = CoreExecutor(AnRaCore(config).eval(), tokenizer=V4Tokenizer.load(TOKENIZER))
    calls = {"steps": 0}
    original = executor.forward_step

    def counting_step(token_ids, *, state=None):
        calls["steps"] += 1
        return original(token_ids, state=state)

    executor.forward_step = counting_step
    generate(executor, executor.tokenizer, "hello world", max_new_tokens=5)
    # 5 tokens generated -> exactly 4 incremental steps; the 5th pass would
    # only produce logits that are discarded.
    assert calls["steps"] == 4


def test_errored_arm_makes_battery_unresolved_not_no_help() -> None:
    """A Core fault in an arm must not be read as a measured non-flip."""
    from connector.experiments.cognitive_credit.runner import run_case

    observed, hidden = build_case("missing_knowledge", 0)

    class FaultyOnArms:
        def __call__(self, attempt: Attempt) -> CompletionResult:
            if attempt is observed.initial_attempt:
                return CompletionResult(texts=("no usable output",), n_executions=1)
            return CompletionResult(texts=(), n_executions=1, error="CoreError")

    result = run_case(observed, hidden, FaultyOnArms())
    assert result.intervention == "unresolved", (
        "errored arms must leave the battery incomplete, not 'nothing helped'"
    )


def test_battery_outcome_labels_are_real_diagnosis_labels() -> None:
    members = set(typing.get_args(DiagnosisLabel))
    assert OUTCOME_HELPED in members
    assert OUTCOME_NONE in members


def test_case_ids_are_content_derived_and_family_blind() -> None:
    import re

    ids = []
    for family in FAMILIES:
        for index in range(5):
            observed, _ = build_case(family, index)
            assert re.fullmatch(r"case-[0-9a-f]{8}", observed.case_id), (
                "ids must be hex digests of the public question text"
            )
            ids.append(observed.case_id)
    assert len(set(ids)) == 20, "distinct questions must yield distinct ids"
    # Stable across rebuilds (content-derived, not family/index-derived).
    observed_again, _ = build_case("missing_knowledge", 0)
    assert observed_again.case_id == ids[0]


def test_reference_runtime_repairs_and_emits_learning_candidate() -> None:
    from connector.runtime import run

    gold = "Lisbon"

    def oracle(attempt: Attempt) -> CompletionResult:
        # Physics: the answer appears only when the fact is in context.
        if "Lisbon" in attempt.knowledge or "Lisbon" in attempt.question:
            return CompletionResult(texts=(gold,), n_executions=1)
        return CompletionResult(texts=("no usable output",), n_executions=1)

    result = run(
        "What is the capital of Portugal?",
        expected=gold,
        knowledge=("The capital of Portugal is Lisbon.",),
        plan_candidates=("State the capital city of Portugal.",),
        complete=oracle,
    )
    assert result.status == "repaired"
    assert result.diagnosis == "missing_knowledge"
    assert result.changed_variable == "knowledge"
    assert result.answer == gold
    assert result.learning_candidate is not None
    assert result.learning_candidate["verified_output"] == gold
    assert any(step.role == "baseline" and not step.verified for step in result.steps)
    assert any(step.role.startswith("intervention:") for step in result.steps)
    # Serialized record is JSON-clean (operational state only).
    import json

    parsed = json.loads(result.to_json())
    assert parsed["status"] == "repaired"
    assert parsed["interventions"], "battery outcomes must be preserved as evidence"


def test_reference_runtime_fails_honestly_when_nothing_helps() -> None:
    from connector.runtime import run

    def never(_attempt: Attempt) -> CompletionResult:
        return CompletionResult(texts=("no usable output",), n_executions=1)

    result = run(
        "What is the capital of Portugal?",
        expected="Lisbon",
        knowledge=("The capital of Portugal is Lisbon.",),
        complete=never,
    )
    assert result.status == "failed"
    assert result.diagnosis == "no_intervention_helped"
    assert result.learning_candidate is None
    assert result.repair_success is None


def test_anra_facade_exposes_run() -> None:
    import anra

    assert callable(anra.run)
    assert anra.RunResult is not None


def test_cuda_incremental_path_does_not_report_device_drift() -> None:
    """torch.device('cuda') != torch.device('cuda:0'); the executor must pin
    its device index or the first forward_step after prefill falsely raises
    'state cache storage device drifted' (CUDA-only trap; CPU is index-less)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from anra_core.config import CoreConfig
    from anra_core.executor import CoreExecutor
    from anra_core.model import AnRaCore

    config = CoreConfig(
        vocab_size=256, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        head_dim=8, d_ff=64, block_size=64, base_seq_len=64, target_seq_len=64,
        sliding_window=8, full_attention_every=2,
    )
    torch.manual_seed(7)
    executor = CoreExecutor(AnRaCore(config).eval(), device="cuda")
    assert executor.device == torch.device("cuda", torch.cuda.current_device())
    state = executor.create_state()
    token = torch.tensor([[3, 10, 20]], dtype=torch.long, device="cuda")
    try:
        executor.prefill(token, state=state)
        executor.forward_step(torch.tensor([[7]], dtype=torch.long, device="cuda"), state=state)
    finally:
        executor.release_state(state)
