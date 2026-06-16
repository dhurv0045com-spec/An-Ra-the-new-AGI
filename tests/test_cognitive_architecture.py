from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from anra_brain import CausalTransformerV2
from cognition.cdse import CrossDomainSynthesisEngine, StructuralSignature
from cognition.cec import ContinuousExperienceConsolidator
from cognition.cre import CognitiveCausalExtension
from cognition.epistemic_tracker import EpistemicState, EpistemicTracker
from cognition.lhm import ConsentPolicy, LongitudinalHumanModel
from cognition.ssie import FailureEvidence, ScientificSelfImprovementEngine
from cognition.storage import EncryptionUnavailable, SensitiveStateStore
from data.causal_corpus import TARGET_COUNTS, iter_causal_records, validate_records
from evaluation.agi_benchmarks import build_report
from evaluation.promotion import CognitiveExtensionPromotionGate, PromotionDecision
from runtime.cognition_registry import validate_cognition_reachability
from training.preflight import HardwareProfile, run_preflight
from training.causal_trainer import CausalExtensionTrainer
from training.launch_manifest import (
    build_launch_manifest,
    load_and_validate_manifest,
    sign_manifest,
)


def tiny_model() -> CausalTransformerV2:
    torch.manual_seed(7)
    model = CausalTransformerV2(
        vocab_size=32,
        n_embd=16,
        n_head=2,
        n_kv_head=1,
        n_layer=2,
        block_size=8,
        base_seq_len=8,
        target_seq_len=8,
        use_hal=False,
    )
    model.eval()
    return model


def test_zero_gate_preserves_base_logits_and_counts():
    model = tiny_model()
    tokens = torch.randint(0, 32, (2, 8))
    before_count = sum(parameter.numel() for parameter in model.parameters())
    before, _ = model(tokens)
    extension = CognitiveCausalExtension(16, rank=4, integration_layers=(0, 1))
    model.attach_cognitive_extension(extension)
    after, _, evidence = model.forward_cognitive(tokens)
    assert torch.equal(before, after)
    assert model.base_parameter_count() == before_count
    assert model.cognitive_parameter_count() > 0
    assert len(evidence) == 2
    assert float(evidence[-1]["gate"].detach()) == 0.0


def test_causal_router_is_padding_aware():
    extension = CognitiveCausalExtension(8, rank=2)
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    output, evidence = extension.apply_layer(x, 0, attention_mask=mask)
    assert output.shape == x.shape
    assert evidence["routing_logits"].shape == (2, 4)


def test_causal_extension_trainer_executes_all_objectives(tmp_path: Path):
    model = tiny_model()
    extension = CognitiveCausalExtension(16, rank=4, integration_layers=(1,))
    model.attach_cognitive_extension(extension)
    trainer = CausalExtensionTrainer(
        model,
        extension,
        total_steps=4,
        warmup_steps=1,
        cdr_path=str(tmp_path / "cdr.jsonl"),
        optimizer_name="adamw",
        lr=1e-4,
    )
    inputs = torch.randint(0, 32, (2, 8))
    labels = {
        "causal_type": torch.tensor([1, 2]),
        "variable_mask": torch.zeros(2, 8),
        "intervention_mask": torch.zeros(2, 8),
        "has_confounder": torch.tensor([0.0, 1.0]),
        "requires_experiment": torch.tensor([1.0, 1.0]),
        "confidence": torch.tensor([0.9, 0.8]),
        "counterfactual_embedding": torch.randn(2, 4),
        "is_counterfactual": torch.tensor([0.0, 1.0]),
    }
    metrics = trainer.step(
        inputs,
        inputs,
        labels,
        attention_mask=torch.ones(2, 8, dtype=torch.long),
    )
    assert {
        "causal_type",
        "variable_extraction",
        "intervention_extraction",
        "confounder",
        "requires_experiment",
        "counterfactual_consistency",
        "verified_answer",
        "calibration",
        "sparsity",
        "gate",
    } <= metrics.keys()


def test_epistemic_thresholds_staleness_and_calibration(tmp_path: Path):
    tracker = EpistemicTracker(tmp_path / "history.jsonl", tmp_path / "calibration.json")
    state = tracker.assess(
        "A medical claim",
        domain="medical",
        evidence=[{"verified": True, "source_id": "a", "provenance": 1, "score": 1, "recency": 1}],
    )
    assert state.calibrated_conf < EpistemicState.THRESHOLDS["medical"]
    state.expires_at = 0
    assert "stale" in state.caveat().lower()
    assert tracker.calibration_report()["status"] == "insufficient_data"


def test_owner_model_refuses_unencrypted_persistence(tmp_path: Path):
    consent = ConsentPolicy(sensitive_inference=True, persistence=True)
    model = LongitudinalHumanModel(SensitiveStateStore(tmp_path, key="test-key"), consent)
    if model.store.available:
        pytest.skip("cryptography is installed; unavailable-path test is environment-specific")
    with pytest.raises(EncryptionUnavailable):
        model.update(
            name="style",
            value="concise",
            category="style",
            source_session="s1",
            evidence_span="owner said concise",
            confidence=1.0,
            confirmed=True,
        )


def test_cec_is_idempotent_and_quarantines_unverified(tmp_path: Path):
    cec = ContinuousExperienceConsolidator(tmp_path / "cec.json")
    turns = [{"tags": ["training_candidate"], "verified": False, "success": False}]
    first = cec.consolidate("s1", turns, opted_in=True)
    second = cec.consolidate("s1", turns, opted_in=True)
    assert asdict(first) == asdict(second)
    assert first.training_candidates == 0
    assert first.quarantined_candidates == 1
    assert cec.rollback("s1") == 1


def test_ssie_requires_repeated_failure_pattern_and_authorization():
    evidence = [
        FailureEvidence(f"s{i}", "reasoning", "failure", f"h{i}")
        for i in range(3)
    ]
    engine = ScientificSelfImprovementEngine()
    proposal = engine.propose(
        "reasoning",
        evidence,
        base_checkpoint="base",
        tokenizer_hash="tok",
        data_hash="data",
        code_hash="code",
        config_hash="cfg",
        maximum_tokens=1000,
    )
    with pytest.raises(PermissionError):
        engine.record_result(proposal.experiment_id, {"delta": 0.1}, signed=True)
    engine.authorize(proposal.experiment_id, owner_authorized=True)
    engine.record_result(proposal.experiment_id, {"delta": 0.1}, signed=True)
    assert len(engine.cognition_theory()) == 1


def test_cross_domain_outputs_are_candidate_hypotheses():
    engine = CrossDomainSynthesisEngine()
    problem = StructuralSignature(
        "deployment",
        ("components", "load"),
        ("dependency",),
        ("threshold",),
        "minimize outage",
        ("failure transfers load",),
    )
    results = engine.synthesize(problem)
    assert results
    assert all(result.verification_status == "candidate" for result in results)


def test_causal_corpus_exact_contract():
    records = list(iter_causal_records())
    report = validate_records(records)
    assert report["total"] == 7500
    assert report["counts"] == TARGET_COUNTS
    assert all(record.bucket == "symbolic" and record.license for record in records)
    assert report["promotion_grade"] is False


def test_agi_report_does_not_fake_missing_evidence():
    report = build_report({})
    assert report["promotion_ready"] is False
    assert all(item["maturity"] == "insufficient_data" for item in report["results"])


def test_cognitive_promotion_gate_blocks_missing_evidence():
    report = build_report({})
    capability = PromotionDecision(True, {"ibs": True}, {"overall": 0.1}, ())
    decision = CognitiveExtensionPromotionGate().evaluate(
        agi_report=report,
        capability_decision=capability,
        checks={},
    )
    assert not decision.allowed
    assert "a01_causal_accuracy" in decision.reasons
    assert "rollback_artifact" in decision.reasons


def test_t4_frontier_smoke_is_supported_and_old_profiles_are_blocked():
    t4 = HardwareProfile("Tesla T4", True, 16 * 1024**3, 32 * 1024**3, 100 * 1024**3, False)
    frontier = run_preflight("frontier", runtime_class="t4_frontier_smoke", hardware=t4)
    legacy = run_preflight("25m", runtime_class="t4_frontier_smoke", hardware=t4)
    assert frontier.allowed
    assert not legacy.allowed
    assert any("iterate500 supports only" in blocker for blocker in legacy.blockers)


def test_cognition_registry_reachable():
    result = validate_cognition_reachability()
    assert set(result) == {f"C-{index:02d}" for index in range(1, 8)}


def test_signed_launch_manifest_is_enforced(tmp_path: Path, monkeypatch):
    from anra.anra_paths import V3_TOKENIZER_FILE
    import hashlib

    key = "manifest-test-key"
    manifest = build_launch_manifest(
        model_profile="frontier",
        extension_profile="cognition-v1",
        tokenizer_hash=hashlib.sha256(V3_TOKENIZER_FILE.read_bytes()).hexdigest(),
        data_manifests=[],
        stage="smoke",
        optimizer="adamw",
        batch_size=1,
        accumulation=1,
        schedule={"kind": "wsd"},
        seeds=[1301],
        checkpoint_source="",
        expected_tokens=0,
        runtime_estimate_hours=1.0,
        owner_authorized=True,
    )
    path = tmp_path / "launch.json"
    sign_manifest(manifest, path, key=key)
    loaded = load_and_validate_manifest(path, key=key)
    assert loaded["model_profile"] == "frontier"
    loaded["batch_size"] = 2
    path.write_text(__import__("json").dumps(loaded), encoding="utf-8")
    with pytest.raises(PermissionError):
        load_and_validate_manifest(path, key=key)
