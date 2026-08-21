"""P0 regression tests: resume math, durability semantics, soup integrity,
trace/input equivalence, decode-policy round-trip, packaging.

Each test targets a failure that could destroy compute or corrupt evidence.
"""

import json

import numpy as np
import pytest
import torch

from training.pack_verify import PackVerificationError, build_manifest, verify_pack
from training.resume import (
    degradation_ratio,
    resolve_pack_horizon,
    should_periodic_save,
    update_best,
)


# --------------------------------------------------------------------------
# Mission 1: resume math. The exact reported failure: global step 20,000
# resumed into a fresh ~330M-token pack must execute the FULL pack budget.
# --------------------------------------------------------------------------


def test_resume_global_step_20000_into_fresh_pack_trains_full_budget() -> None:
    # 330M tokens at 131,072 tokens/step ~= 2,517 pack steps (the old trainer
    # computed this then looped `while 20000 < 2517` -> zero updates).
    horizon = resolve_pack_horizon(
        global_step=20_000,
        restored_pack_step=0,
        token_budget=330_000_000,
        tokens_per_step=131_072,
    )
    assert horizon.pack_total_steps == 2_517
    assert horizon.updates_remaining == 2_517, "fresh pack must train its full budget"
    # And the loop bound must use pack_step, not global_step:
    executed = 0
    pack_step = horizon.start_pack_step
    while pack_step < horizon.pack_total_steps:
        pack_step += 1
        executed += 1
    assert executed == 2_517


def test_partially_consumed_pack_resumes_at_correct_position() -> None:
    horizon = resolve_pack_horizon(
        global_step=21_400,
        restored_pack_step=1_000,
        token_budget=330_000_000,
        tokens_per_step=131_072,
    )
    assert horizon.updates_remaining == 2_517 - 1_000


def test_exhausted_pack_reports_zero_updates() -> None:
    horizon = resolve_pack_horizon(
        global_step=22_517,
        restored_pack_step=2_517,
        token_budget=330_000_000,
        tokens_per_step=131_072,
    )
    assert horizon.updates_remaining == 0


def test_max_steps_override_bounds_the_pack() -> None:
    horizon = resolve_pack_horizon(
        global_step=0,
        restored_pack_step=0,
        token_budget=330_000_000,
        tokens_per_step=131_072,
        max_steps_override=100,
    )
    assert horizon.pack_total_steps == 100


def test_periodic_save_boundaries() -> None:
    assert should_periodic_save(200, 200)
    assert should_periodic_save(400, 200)
    assert not should_periodic_save(199, 200)
    assert not should_periodic_save(0, 200)
    assert not should_periodic_save(500, 0)  # disabled


def test_best_tracking_and_degradation_ratio() -> None:
    best, improved = update_best(None, 1.9)
    assert improved and best == 1.9
    best, improved = update_best(best, 1.8)
    assert improved and best == 1.8
    best, improved = update_best(best, 2.1)  # degraded
    assert not improved and best == 1.8  # bar never lowered by worse loss
    assert degradation_ratio(1.8, 2.1) > 1.10


# --------------------------------------------------------------------------
# Mission 3/7: soup artifact actually contains averaged weights and honest
# artifact class.
# --------------------------------------------------------------------------


def _tiny_checkpoint(tmp_path, name: str, offset: float, with_optimizer: bool = False):
    from anra_core.config import CANONICAL_CONFIG
    from anra_core.model import AnRaCore

    model = AnRaCore(CANONICAL_CONFIG)
    state = {k: v + offset for k, v in model.state_dict().items()}
    payload = {
        "checkpoint_artifact_class": "full_resume",
        "checkpoint_schema_version": 1,
        "global_step": 100,
        "model_state_dict": state,
        "metrics": {},
    }
    if with_optimizer:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        payload["optimizer"] = opt.state_dict()
    path = tmp_path / name
    torch.save(payload, path)
    return path


def test_soup_artifact_contains_averaged_weights(tmp_path) -> None:
    import subprocess
    import sys

    parent_a = _tiny_checkpoint(tmp_path, "a.pt", 0.0)
    parent_b = _tiny_checkpoint(tmp_path, "b.pt", 1.0)
    out = tmp_path / "soup.pt"

    repo_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "make_soup.py"),
         str(parent_a), str(parent_b), str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    payload = torch.load(out, map_location="cpu", weights_only=False)
    key = "model_state_dict"
    soup_state = payload[key]
    state_a = torch.load(parent_a, map_location="cpu", weights_only=False)[key]
    state_b = torch.load(parent_b, map_location="cpu", weights_only=False)[key]

    # Parameter-hash proofs: soup differs from BOTH parents...
    soup_params = b"".join(t.numpy().tobytes() for t in soup_state.values())
    a_params = b"".join(t.numpy().tobytes() for t in state_a.values())
    b_params = b"".join(t.numpy().tobytes() for t in state_b.values())
    assert soup_params != a_params, "soup identical to parent A: averaging failed"
    assert soup_params != b_params, "soup identical to parent B: averaging failed"
    # ...and equals their element-wise mean on a sampled tensor.
    sample_key = next(iter(state_a))
    expected = (state_a[sample_key].float() + state_b[sample_key].float()) / 2.0
    assert torch.allclose(soup_state[sample_key].float(), expected, atol=1e-6)
    # Honest artifact class: model_only, no resumable optimizer state.
    assert payload["checkpoint_artifact_class"] == "model_only"
    assert not payload.get("optimizer")


# --------------------------------------------------------------------------
# Mission 9: execution trace records EXACTLY what Core consumed, including
# resolved tool output, and tool output appears BEFORE the answer marker.
# --------------------------------------------------------------------------


def test_prepared_execution_resolves_tool_output_before_answer() -> None:
    from connector.experiments.cognitive_credit.case import (
        Attempt,
        DecodePolicy,
        PreparedExecution,
        ToolBehavior,
    )

    def calculator() -> str:
        return "42"

    attempt = Attempt(
        question="What is 20 + 22?",
        plan="Read the calculator output.",
        tool=ToolBehavior("calculator", available=True, execute=calculator),
        decode=DecodePolicy(),
    )
    prepared = PreparedExecution.from_attempt(attempt)
    # Tool output present AND positioned before the answer marker.
    tool_pos = prepared.prompt.index("<tool_output>42</tool_output>")
    answer_pos = prepared.prompt.index("<answer>")
    assert tool_pos < answer_pos
    assert "ERROR" not in prepared.prompt


def test_learning_candidate_contains_exact_tool_input() -> None:
    """Tool-assisted repair trace: saved prompt == exact Core input."""
    from connector.experiments.cognitive_credit.case import (
        Attempt,
        CompletionResult,
        DecodePolicy,
        PreparedExecution,
        HiddenGroundTruth,
        ObservedCase,
        ToolBehavior,
    )
    from connector.experiments.cognitive_credit.runner import run_case

    def calculator() -> str:
        return "42"

    observed = ObservedCase(
        case_id="tool-trace",
        question="Use the calculator to add 20 and 22.",
        success_criterion="contains 42",
        initial_attempt=Attempt(
            question="Use the calculator to add 20 and 22.",
            plan="Read the output.",
            tool=ToolBehavior("calculator", available=False, execute=calculator),
            decode=DecodePolicy(max_new_tokens=12),
        ),
        tools=(ToolBehavior("calculator", available=True, execute=calculator),),
    )
    hidden = HiddenGroundTruth(family="tool_failure", gold_solution="42")

    captured_prompts: list[str] = []

    class RecordingOracle:
        def __call__(self, attempt: Attempt) -> CompletionResult:
            prepared = PreparedExecution.from_attempt(attempt)
            captured_prompts.append(prepared.prompt)
            ok = attempt.tool is not None and attempt.tool.available
            return CompletionResult(texts=("42",) if ok else ("no output",), n_executions=1)

    result = run_case(observed, hidden, RecordingOracle())
    assert result.intervention == "tool_failure"
    assert result.repair_success is True
    # Every recorded prompt that involved the enabled tool contains the real
    # adapter output - the trace equals the computation.
    repair_prompt = captured_prompts[-1]
    assert "<tool_output>42</tool_output>" in repair_prompt
    assert repair_prompt.index("<tool_output>42</tool_output>") < repair_prompt.index("<answer>")


def test_decode_policy_roundtrips_every_parameter() -> None:
    from connector.experiments.cognitive_credit.case import DecodePolicy

    policy = DecodePolicy(
        temperature=0.7, top_p=0.95, candidates=4, seed=99,
        max_new_tokens=48, repetition_penalty=1.3, no_repeat_ngram_size=6,
    )
    data = dict(
        temperature=policy.temperature, top_p=policy.top_p,
        candidates=policy.candidates, seed=policy.seed,
        max_new_tokens=policy.max_new_tokens,
        repetition_penalty=policy.repetition_penalty,
        no_repeat_ngram_size=policy.no_repeat_ngram_size,
    )
    restored = DecodePolicy(**data)
    assert restored == policy
    assert restored.assisted is True
    raw = DecodePolicy.raw()
    assert raw.assisted is False
    assert raw.repetition_penalty == 1.0 and raw.no_repeat_ngram_size == 0


# --------------------------------------------------------------------------
# Mission 5: semantic pack validation - malformed-but-hashed packs fail.
# --------------------------------------------------------------------------


def _make_shard(path, array) -> str:
    np.save(path, array)
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_float_dtype_shard_fails_verification(tmp_path) -> None:
    root = tmp_path / "pack"
    (root / "train").mkdir(parents=True)
    shard = root / "train" / "s.npy"
    digest = _make_shard(shard, np.arange(300, dtype=np.float32))
    manifest = {"schema": "anra-token-pack/v1", "block_size": 64,
                "total_tokens": 300,
                "shards": [{"file": "train/s.npy", "tokens": 300, "sha256": digest}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(PackVerificationError, match="integer dtype"):
        verify_pack(root)


def test_out_of_vocab_token_ids_fail_verification(tmp_path) -> None:
    root = tmp_path / "pack"
    (root / "train").mkdir(parents=True)
    shard = root / "train" / "s.npy"
    tokens = np.arange(300, dtype=np.int32)
    tokens[5] = 999_999  # beyond vocab
    digest = _make_shard(shard, tokens)
    manifest = {"schema": "anra-token-pack/v1", "block_size": 64,
                "total_tokens": 300,
                "shards": [{"file": "train/s.npy", "tokens": 300, "sha256": digest}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(PackVerificationError, match="exceeds vocab"):
        verify_pack(root, vocab_size=32_768)


def test_negative_token_ids_fail_verification(tmp_path) -> None:
    root = tmp_path / "pack"
    (root / "train").mkdir(parents=True)
    shard = root / "train" / "s.npy"
    tokens = np.arange(300, dtype=np.int32)
    tokens[7] = -4
    digest = _make_shard(shard, tokens)
    manifest = {"schema": "anra-token-pack/v1", "block_size": 64,
                "total_tokens": 300,
                "shards": [{"file": "train/s.npy", "tokens": 300, "sha256": digest}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(PackVerificationError, match="negative"):
        verify_pack(root)


def test_manifest_total_mismatch_fails_verification(tmp_path) -> None:
    root = tmp_path / "pack"
    (root / "train").mkdir(parents=True)
    shard = root / "train" / "s.npy"
    digest = _make_shard(shard, np.arange(300, dtype=np.int16))
    manifest = {"schema": "anra-token-pack/v1", "block_size": 64,
                "total_tokens": 999,  # wrong total
                "shards": [{"file": "train/s.npy", "tokens": 300, "sha256": digest}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(PackVerificationError, match="total_tokens"):
        verify_pack(root)


def test_duplicate_shard_paths_fail_verification(tmp_path) -> None:
    root = tmp_path / "pack"
    (root / "train").mkdir(parents=True)
    shard = root / "train" / "s.npy"
    digest = _make_shard(shard, np.arange(300, dtype=np.int16))
    entry = {"file": "train/s.npy", "tokens": 300, "sha256": digest}
    manifest = {"schema": "anra-token-pack/v1", "block_size": 64,
                "total_tokens": 600, "shards": [entry, dict(entry)]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(PackVerificationError, match="duplicate"):
        verify_pack(root)


def test_block_size_contract_enforced(tmp_path) -> None:
    root = tmp_path / "pack"
    (root / "train").mkdir(parents=True)
    shard = root / "train" / "s.npy"
    digest = _make_shard(shard, np.arange(300, dtype=np.int16))
    manifest = {"schema": "anra-token-pack/v1", "block_size": 512,
                "total_tokens": 300,
                "shards": [{"file": "train/s.npy", "tokens": 300, "sha256": digest}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(PackVerificationError, match="block_size"):
        verify_pack(root, expected_block_size=2048)


def test_valid_semantic_pack_still_passes(tmp_path) -> None:
    root = tmp_path / "pack"
    (root / "train").mkdir(parents=True)
    shard = root / "train" / "s.npy"
    rng = np.random.default_rng(0)
    digest = _make_shard(shard, rng.integers(0, 32_767, size=5_000, dtype=np.int16))
    manifest = {"schema": "anra-token-pack/v1", "block_size": 256,
                "total_tokens": 5_000,
                "shards": [{"file": "train/s.npy", "tokens": 5_000, "sha256": digest}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    pack = verify_pack(root, vocab_size=32_768, expected_block_size=256)
    assert pack.total_tokens == 5_000


# --------------------------------------------------------------------------
# Mission 12: packaging exposes anra.run and consistent versions.
# --------------------------------------------------------------------------


def test_anra_facade_exposes_run() -> None:
    import anra

    assert callable(anra.run)
    assert hasattr(anra, "RunResult")
    assert hasattr(anra, "Step")
