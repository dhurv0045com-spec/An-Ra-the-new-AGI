from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


class _Tokenizer:
    vocab_size = 8209
    backend = "test"


class _FakeModel:
    d_model = 1280
    n_layer = 28
    n_head = 16
    n_kv_head = 4
    block_size = 1024

    def to(self, _device):
        return self

    def eval(self):
        return self

    def disable_kv_cache(self):
        self.kv_disabled = True


class _CharacterTokenizer:
    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    @staticmethod
    def decode(ids: list[int]) -> str:
        return "".join(chr(value) for value in ids)


def test_frontier_runtime_refuses_missing_checkpoint(monkeypatch, tmp_path: Path) -> None:
    import generate

    generate._reset_runtime_cache()
    missing = tmp_path / "anra_frontier_500m.pt"
    monkeypatch.setenv("ANRA_MODEL_PROFILE", "frontier")
    monkeypatch.setattr(generate, "FRONTIER_CHECKPOINT", missing)
    monkeypatch.setattr(generate, "_requested_checkpoint_path", lambda: missing)
    monkeypatch.setattr(
        "training.shared_checkpoint.restore_shared_checkpoint",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(FileNotFoundError, match="Frontier runtime requested"):
        generate._load_runtime()

    generate._reset_runtime_cache()


def test_frontier_runtime_uses_frontier_builder(monkeypatch, tmp_path: Path) -> None:
    import generate

    generate._reset_runtime_cache()
    checkpoint = tmp_path / "anra_frontier_500m.pt"
    checkpoint.write_bytes(b"fake")
    calls: list[str] = []

    monkeypatch.setenv("ANRA_MODEL_PROFILE", "frontier")
    monkeypatch.setattr(generate, "FRONTIER_CHECKPOINT", checkpoint)
    monkeypatch.setattr(generate, "_requested_checkpoint_path", lambda: checkpoint)
    monkeypatch.setattr(generate, "load_or_build_v2_tokenizer", lambda: _Tokenizer())

    def build_frontier():
        calls.append("frontier")
        return _FakeModel()

    def build_legacy(**_kwargs):
        calls.append("legacy")
        raise AssertionError("legacy builder must not be used for frontier runtime")

    monkeypatch.setattr(generate, "build_frontier_model", build_frontier)
    monkeypatch.setattr(generate, "build_v2_model", build_legacy)
    monkeypatch.setattr(
        generate,
        "load_checkpoint",
        lambda *_args, **_kwargs: {
            "loaded": True,
            "global_step": 6927,
            "best_loss": 0.3279,
            "sessions_completed": 3,
            "data_profile": "t4-cached",
            "training_data_layout": "conversation_pack_v2",
        },
    )
    monkeypatch.setattr(
        generate,
        "model_summary",
        lambda _model: {
            "parameters": generate.V2_FRONTIER_PARAMETER_COUNT,
            "trainable_parameters": generate.V2_FRONTIER_PARAMETER_COUNT,
        },
    )

    model, tokenizer, loaded, profile, state = generate._load_runtime()

    assert isinstance(model, _FakeModel)
    assert tokenizer.vocab_size == 8209
    assert loaded == checkpoint
    assert profile == "frontier"
    assert state["global_step"] == 6927
    assert calls == ["frontier"]

    generate._reset_runtime_cache()


def test_model_info_exposes_frontier_proof_fields(monkeypatch, tmp_path: Path) -> None:
    import generate

    generate._reset_runtime_cache()
    checkpoint = tmp_path / "anra_frontier_500m.pt"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setenv("ANRA_MODEL_PROFILE", "frontier")
    monkeypatch.setattr(generate, "FRONTIER_CHECKPOINT", checkpoint)
    monkeypatch.setattr(generate, "_requested_checkpoint_path", lambda: checkpoint)
    monkeypatch.setattr(generate, "load_or_build_v2_tokenizer", lambda: _Tokenizer())
    monkeypatch.setattr(generate, "build_frontier_model", lambda: _FakeModel())
    monkeypatch.setattr(
        generate,
        "load_checkpoint",
        lambda *_args, **_kwargs: {
            "loaded": True,
            "global_step": 6927,
            "best_loss": 0.3279,
            "sessions_completed": 3,
        },
    )
    monkeypatch.setattr(
        generate,
        "model_summary",
        lambda _model: {
            "parameters": generate.V2_FRONTIER_PARAMETER_COUNT,
            "trainable_parameters": generate.V2_FRONTIER_PARAMETER_COUNT,
        },
    )

    info = generate.get_model_info()

    assert info["profile"] == "frontier"
    assert info["checkpoint"] == str(checkpoint)
    assert info["param_count"] == generate.V2_FRONTIER_PARAMETER_COUNT
    assert info["block_size"] == 1024
    assert info["checkpoint_state"]["global_step"] == 6927

    generate._reset_runtime_cache()


def test_model_adapter_publishes_readiness_atomically(monkeypatch) -> None:
    import app

    adapter = app.ModelAdapter()
    monkeypatch.setattr(
        app,
        "get_model_info",
        lambda: {
            "checkpoint": "/tmp/anra.pt",
            "checkpoint_sha256": "checkpoint",
            "tokenizer_sha256": "tokenizer",
            "device": "cuda",
            "profile": "frontier",
            "vocab_size": 8209,
            "param_count": 499_167_047,
            "block_size": 1024,
        },
    )

    with pytest.raises(app.HTTPException) as pending:
        adapter.require_ready()
    assert pending.value.status_code == 503

    adapter.load()

    assert adapter.readiness()["stage"] == "ready"
    assert adapter.readiness()["ready"] is True
    assert adapter.info["param_count"] == 499_167_047
    adapter.require_ready()


def test_model_adapter_fails_closed_on_placeholder_metadata(monkeypatch) -> None:
    import app

    adapter = app.ModelAdapter()
    monkeypatch.setattr(app, "get_model_info", lambda: {"profile": "unknown"})

    adapter.load()

    assert adapter.readiness()["stage"] == "failed"
    assert adapter.info == {}
    with pytest.raises(app.HTTPException) as failed:
        adapter.require_ready()
    assert failed.value.detail["error"] == "model_not_ready"


def test_repetition_penalty_moves_repeated_logits_down() -> None:
    import torch
    import generate

    logits = torch.tensor([0.0, 2.0, -2.0])
    adjusted = generate._apply_repetition_penalty(
        logits,
        [1, 2],
        generate.GenerationConfig(repetition_penalty=2.0),
    )
    assert adjusted[1].item() == 1.0
    assert adjusted[2].item() == -4.0


def test_full_system_operator_dispatch_connects_explicit_agent_goal(monkeypatch) -> None:
    import app

    monkeypatch.setattr(
        app,
        "_run_native_agent_goal",
        lambda goal: {"success": True, "output": f"completed:{goal}"},
    )
    result = app._dispatch_full_system_operator("/goal inspect native routing")

    assert result["handled"] is True
    assert result["agent_executed"] is True
    assert result["tool_executed"] is False
    assert "completed:inspect native routing" in result["response"]


def test_request_scoped_runtime_state_isolation_probe() -> None:
    import generate

    esv_keys = set(generate._ESV_STORE)
    ghost_keys = set(generate._GHOST_STORE)
    report = generate.verify_session_state_isolation()

    assert report["verified"] is True
    assert report["generation_serialized"] is True
    assert set(generate._ESV_STORE) == esv_keys
    assert set(generate._GHOST_STORE) == ghost_keys


def test_request_scoped_runtime_isolation_executes_generation_probe(monkeypatch) -> None:
    import generate

    calls: list[str] = []

    def fake_generate(_prompt, _config, *, session_id=None):
        assert session_id is not None
        assert _config.persist_adaptive_state is False
        calls.append(session_id)
        generate._ESV_STORE[session_id] = generate._ESV_STORE[session_id] + 0.01
        generate._GHOST_STORE[session_id]["generated"] = True
        return object()

    monkeypatch.setattr(generate, "generate_traced", fake_generate)
    report = generate.verify_session_state_isolation(probe_generation=True)

    assert report["verified"] is True
    assert report["runtime_generation_probed"] is True
    assert report["generation_state_isolated"] is True
    assert len(calls) == 1


def test_private_promotion_route_starts_background_job(monkeypatch) -> None:
    import asyncio
    import app

    completed: list[bool] = []
    app._PRIVATE_EVAL_TASK = None
    monkeypatch.setattr(app.ADAPTER, "require_ready", lambda: None)
    monkeypatch.setattr(
        app,
        "_run_private_promotion_evaluation",
        lambda: completed.append(True) or {"capability_allowed": False},
    )

    async def scenario():
        result = await app.private_promotion_evaluation_route()
        assert result["status"] == "started"
        assert app._PRIVATE_EVAL_TASK is not None
        await app._PRIVATE_EVAL_TASK

    asyncio.run(scenario())
    assert completed == [True]
    assert "open-review" in app.DEVELOPER_UI_HTML
    assert "run-integration" in app.DEVELOPER_UI_HTML


def test_release_private_promotion_gate_rehashes_heldout_artifact(tmp_path: Path) -> None:
    import app
    from training.eval_v2 import ensure_private_eval_suite

    tasks, metadata = ensure_private_eval_suite(tmp_path)
    report = {
        "capability_allowed": True,
        "task_count": len(tasks),
        "suite_metadata": metadata,
        "model_bundle": {
            "checkpoint_sha256": "checkpoint-hash",
            "tokenizer_sha256": "tokenizer-hash",
            "runtime_source_commit": "runtime-commit",
            "runtime_worktree_clean": True,
        },
    }
    expected = dict(report["model_bundle"])

    assert app._private_promotion_verified(report, expected_bundle=expected) is True
    stale = {**expected, "checkpoint_sha256": "different-checkpoint"}
    assert app._private_promotion_verified(report, expected_bundle=stale) is False
    suite_path = Path(str(metadata["suite_path"]))
    suite_path.write_bytes(suite_path.read_bytes() + b"{}\n")
    assert app._private_promotion_verified(report, expected_bundle=expected) is False


def test_checkpoint_embedded_manifests_restore_with_hash_verification(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import hashlib
    import generate

    payload = b'{"schema_version":3,"shards":[]}'
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(generate, "_get_runtime", lambda: (object(), object(), tmp_path / "x.pt"))
    monkeypatch.setattr(
        generate,
        "_RUNTIME_LOAD_STATE",
        {
            "data_manifests": {"native/manifest.json": digest},
            "data_manifest_payloads": {"native/manifest.json": payload},
        },
    )

    report = generate.restore_embedded_data_manifests(tmp_path / "restored")

    assert report["complete"] is True
    assert report["restored"] == 1
    assert (tmp_path / "restored" / "native" / "manifest.json").read_bytes() == payload

    generate._RUNTIME_LOAD_STATE["data_manifest_payloads"]["native/manifest.json"] = b"tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        generate.restore_embedded_data_manifests(tmp_path / "other")


def test_full_system_integration_evidence_requires_every_bound_check() -> None:
    import app

    bundle = {
        "checkpoint_sha256": "checkpoint",
        "tokenizer_sha256": "tokenizer",
        "runtime_source_commit": "commit",
        "runtime_worktree_clean": True,
    }
    checks = dict.fromkeys(
        (
            "model_and_native_subsystems",
            "ghost_path",
            "evaluation_state_not_persisted",
            "memory",
            "verifier",
            "agent_execution",
            "sandboxed_tool",
            "cognition",
            "capability_graph",
        ),
        True,
    )
    report = {"passed": True, "checks": checks, "model_bundle": bundle}

    assert app._full_system_integration_verified(report, expected_bundle=bundle)
    report["checks"]["memory"] = False
    assert not app._full_system_integration_verified(report, expected_bundle=bundle)


def test_evaluation_generation_executes_ghost_path_without_persisting_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import torch
    import generate
    from anra_brain import CausalTransformerV2

    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        vocab_size = 64
        special_ids = {"<pad>": 0, "<bos>": 1, "<eos>": 2}

        @staticmethod
        def encode(text: str, *, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [3 + (ord(character) % 50) for character in text]

        @staticmethod
        def decode(_ids: list[int]) -> str:
            return "valid complete response with enough words"

    class GhostRecorder:
        writes = 0

        def store(self, *_args, **_kwargs):
            self.writes += 1

    model = CausalTransformerV2(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        block_size=64,
        mod_layers={1},
    ).eval()
    initial_esv = torch.tensor([0.2, -0.1, 0.3])
    model.esv_module.state.copy_(initial_esv)
    recorder = GhostRecorder()
    monkeypatch.setattr(generate, "_get_runtime", lambda: (model, Tokenizer(), tmp_path / "x.pt"))
    monkeypatch.setattr(generate, "_RUNTIME_LOAD_STATE", {"load_report": {"exact_native_load": True}})
    monkeypatch.setattr(generate, "_get_hal", lambda _session_id: None)
    monkeypatch.setattr(generate, "_GHOST_MEMORY", recorder)
    monkeypatch.setattr(generate, "_generation_quality", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(generate, "_language_fragment_detected", lambda _text: False)
    session_id = "nonpersistent_eval"

    trace = generate.generate_traced(
        "H: probe\nANRA:",
        generate.GenerationConfig(
            max_tokens=2,
            mode="full_system",
            persist_adaptive_state=False,
        ),
        session_id=session_id,
    )

    assert trace.subsystem_trace["ghost_executed"] is True
    assert trace.subsystem_trace["adaptive_state_persisted"] is False
    assert session_id not in generate._GHOST_STORE
    assert session_id not in generate._ESV_STORE
    assert recorder.writes == 0
    torch.testing.assert_close(model.esv_module.state, initial_esv)


def test_full_system_probe_executes_real_memory_agent_tool_cognition_and_verifier(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import app

    subsystem_trace = {
        "model_executed": True,
        "mod_executed": True,
        "rim_executed": True,
        "dstp_executed": True,
        "esv_executed": True,
        "hal_executed": True,
        "ghost_executed": True,
        "adaptive_state_persisted": False,
    }
    monkeypatch.setattr(
        app,
        "generate_traced",
        lambda *_args, **_kwargs: SimpleNamespace(
            subsystem_trace=subsystem_trace,
            quality_state="accepted",
            stopped_by="eos",
        ),
    )
    monkeypatch.setattr(app, "clear_session_runtime_state", lambda _session_id: None)
    monkeypatch.setattr(app, "get_model_info", lambda: {})
    monkeypatch.setattr(
        app,
        "_evaluation_model_bundle",
        lambda _info: {"runtime_worktree_clean": True},
    )
    monkeypatch.setattr(app, "OUTPUT_V2_DIR", tmp_path / "output")
    monkeypatch.setattr("anra.anra_paths.OPERATOR_AUDIT_LOG", tmp_path / "operator.jsonl")
    monkeypatch.setenv("ANRA_AGENT_WORKSPACE", str(tmp_path / "workspace"))

    report = app._run_full_system_integration_probe()

    assert report["passed"] is True, report
    assert all(report["checks"].values())


def test_prompt_assembly_preserves_current_message_and_inserts_memory_once() -> None:
    from inference.optimize_context_window import ContextWindowOptimizer

    optimizer = ContextWindowOptimizer(_CharacterTokenizer(), max_context=180)
    result = optimizer.build_optimized_context(
        [("old question", "old answer")],
        [{"content": "remember cobalt"}],
        "current message must remain complete",
        max_new_tokens=32,
        mode="full_system",
    )
    assert result["formatted_prompt"].endswith("H: current message must remain complete\nANRA:")
    assert result["formatted_prompt"].count("remember cobalt") == 1
    assert result["prompt_tokens"] + result["reserved_output_tokens"] + 1 <= 180


def test_prompt_assembly_truncates_old_history_before_current_message() -> None:
    from inference.optimize_context_window import ContextWindowOptimizer

    optimizer = ContextWindowOptimizer(_CharacterTokenizer(), max_context=96)
    result = optimizer.build_optimized_context(
        [("old " * 20, "answer " * 20)],
        [],
        "newest request",
        max_new_tokens=24,
        mode="diagnostic",
    )
    assert "newest request" in result["formatted_prompt"]
    assert result["context_truncated"] is True
