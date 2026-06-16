"""AN-RA reference adapter for the standalone ThirdEye evidence platform."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from anra.anra_paths import OUTPUT_V2_DIR, ROOT

PROJECT_ID = "anra"
THIRDEYE_HOME = OUTPUT_V2_DIR / "thirdeye"


def _load_sdk() -> dict[str, Any]:
    try:
        from thirdeye import (  # type: ignore[import-not-found]
            CapabilityTarget,
            FeatureCategory,
            FeatureSpec,
            FeatureVariant,
            ProjectSpec,
            ProtocolKind,
            ProtocolSpec,
            SubsystemSpec,
            ThirdEye,
        )
        from thirdeye.evidence import grade_evidence  # type: ignore[import-not-found]
    except ImportError:
        sibling = ROOT.parent / "thirdeye"
        if not sibling.exists():
            raise RuntimeError(
                "ThirdEye is not installed and no sibling checkout exists. "
                "Install thirdeye-evidence or clone it beside AN-RA."
            ) from None
        sys.path.insert(0, str(sibling))
        return _load_sdk()
    return {
        "FeatureCategory": FeatureCategory,
        "FeatureSpec": FeatureSpec,
        "FeatureVariant": FeatureVariant,
        "CapabilityTarget": CapabilityTarget,
        "ProjectSpec": ProjectSpec,
        "ProtocolKind": ProtocolKind,
        "ProtocolSpec": ProtocolSpec,
        "ThirdEye": ThirdEye,
        "SubsystemSpec": SubsystemSpec,
        "grade_evidence": grade_evidence,
    }


def _feature(
    sdk: dict[str, Any],
    feature_id: str,
    name: str,
    category: str,
    intended_behavior: str,
    *,
    requires_retraining: bool,
    parent: str | None = None,
    probe: str | None = None,
    benefits: tuple[str, ...] = (),
    regressions: tuple[str, ...] = (),
    protected: tuple[str, ...] = (),
) -> Any:
    variants = (
        sdk["FeatureVariant"]("off", "Disabled", is_control=True),
        sdk["FeatureVariant"]("on", "Enabled"),
    )
    return sdk["FeatureSpec"](
        feature_id=feature_id,
        name=name,
        owner="AN-RA Research",
        version="1.0.0",
        intended_behavior=intended_behavior,
        category=sdk["FeatureCategory"](category),
        variants=variants,
        requires_retraining=requires_retraining,
        parent_feature_id=parent,
        expected_benefits=benefits,
        possible_regressions=regressions,
        protected_metrics=protected,
        resource_metrics=("tokens_per_second", "latency_ms", "peak_memory_bytes"),
        activation_probe=probe,
    )


def feature_specs() -> list[Any]:
    sdk = _load_sdk()
    return [
        _feature(
            sdk,
            "anra.esv",
            "Emotional State Vector",
            "architecture",
            "Infer bounded latent state used to modulate attention and identity.",
            requires_retraining=True,
            probe="model.esv_module",
            benefits=("identity_retention", "calibration"),
            regressions=("perplexity", "latency_ms"),
        ),
        _feature(
            sdk,
            "anra.rim",
            "Residual Identity Modulator",
            "architecture",
            "Inject a bounded ESV identity channel into each transformer block.",
            requires_retraining=True,
            probe="model.use_rim",
            benefits=("identity_retention",),
            regressions=("perplexity",),
            protected=("safety",),
        ),
        _feature(
            sdk,
            "anra.dstp",
            "Depth-Scheduled Temperature Profile",
            "architecture",
            "Apply deterministic per-depth attention temperature scaling.",
            requires_retraining=True,
            probe="model.use_dstp",
            benefits=("reasoning", "training_stability"),
            regressions=("perplexity", "latency_ms"),
        ),
        _feature(
            sdk,
            "anra.mod",
            "Mixture of Depth Routing",
            "architecture",
            "Route selected tokens through additional MLP computation.",
            requires_retraining=True,
            probe="model.mod_routers",
            benefits=("reasoning",),
            regressions=("latency_ms", "peak_memory_bytes"),
        ),
        _feature(
            sdk,
            "anra.hal",
            "Hormonal Analog Layer",
            "architecture",
            "Adapt behavior from verifier, identity, safety, and session feedback.",
            requires_retraining=True,
            probe="model.use_hal",
            benefits=("reasoning", "identity_retention", "verifier_pass_rate"),
            regressions=("perplexity", "latency_ms", "instability"),
            protected=("safety", "identity_retention"),
        ),
        _feature(
            sdk,
            "anra.hal.attention_temperature",
            "HAL Attention Temperature",
            "architecture",
            "Convert HAL state into attention temperature.",
            requires_retraining=True,
            parent="anra.hal",
            probe="model.hal_module.attention_temperature",
        ),
        _feature(
            sdk,
            "anra.hal.rlvr_feedback",
            "HAL RLVR Feedback",
            "training",
            "Update HAL state from verifier-scored RLVR outcomes.",
            requires_retraining=True,
            parent="anra.hal",
            probe="rlvr.hal",
        ),
        _feature(
            sdk,
            "anra.hal.memory_threshold",
            "HAL Memory Threshold",
            "memory",
            "Adapt memory salience thresholds from HAL state.",
            requires_retraining=False,
            parent="anra.hal",
            probe="hal.memory_threshold",
        ),
        _feature(
            sdk,
            "anra.hal.ouroboros_weights",
            "HAL Ouroboros Weighting",
            "runtime",
            "Adapt recursive reasoning pass weights from HAL state.",
            requires_retraining=False,
            parent="anra.hal",
            probe="hal.ouroboros_weights",
        ),
        _feature(
            sdk,
            "anra.optimizer",
            "Optimizer Selection",
            "training",
            "Select and report the optimizer implementation used for training.",
            requires_retraining=True,
            probe="training.optimizer_report",
        ),
        _feature(
            sdk,
            "anra.data_mix",
            "Adaptive Data Mixture",
            "data",
            "Control owner, identity, teacher, symbolic, and replay sampling.",
            requires_retraining=True,
            probe="training.mix_report",
        ),
        _feature(
            sdk,
            "anra.rlvr",
            "Reinforcement Learning from Verifiable Rewards",
            "training",
            "Improve checkable tasks using deterministic verifier rewards.",
            requires_retraining=True,
            probe="training.rlvr_report",
            benefits=("verifier_pass_rate",),
            protected=("identity_retention", "safety"),
        ),
        _feature(
            sdk,
            "anra.memory",
            "Hybrid Memory",
            "memory",
            "Retrieve provenance-bearing memories across lexical and semantic stores.",
            requires_retraining=False,
            probe="runtime.memory_router",
            benefits=("memory_recall_at_3",),
            regressions=("latency_ms", "stale_memory_rate"),
        ),
        _feature(
            sdk,
            "anra.verification",
            "Verifier Search",
            "runtime",
            "Ground checkable outputs in deterministic domain verifiers.",
            requires_retraining=False,
            probe="runtime.verifier",
            benefits=("verified_reasoning_rate", "truth_checking_coverage"),
            regressions=("latency_ms",),
        ),
        _feature(
            sdk,
            "anra.cognition",
            "Cognitive Extension",
            "architecture",
            "Provide separately packaged causal and epistemic cognitive services.",
            requires_retraining=True,
            probe="model.cognitive_extension",
            benefits=("causal_accuracy", "epistemic_calibration"),
            protected=("base_model_parity",),
        ),
        _feature(
            sdk,
            "anra.inference_efficiency",
            "Inference Efficiency",
            "runtime",
            "Reduce serving cost through cache and speculative execution mechanisms.",
            requires_retraining=False,
            probe="runtime.inference_efficiency",
            benefits=("tokens_per_second", "latency_ms"),
            regressions=("perplexity", "identity_retention"),
        ),
    ]


def register_project(home: str | Path = THIRDEYE_HOME) -> Any:
    sdk = _load_sdk()
    eye = sdk["ThirdEye"](home)
    eye.register_project(
        sdk["ProjectSpec"](
            project_id=PROJECT_ID,
            name="AN-RA",
            description=(
                "Reference integration for the iterate900 900M-class AN-RA frontier model "
                "and its training, cognition, memory, agent, evaluation, and runtime systems."
            ),
            privacy_mode="aggregate",
        )
    )
    for feature in feature_specs():
        eye.register_feature(PROJECT_ID, feature)
    return eye


def _feature_to_dict(feature: Any) -> dict[str, Any]:
    if hasattr(feature, "to_dict"):
        return dict(feature.to_dict())
    data = dict(getattr(feature, "__dict__", {}))
    variants = data.get("variants")
    if variants is not None:
        data["variants"] = [
            item.to_dict() if hasattr(item, "to_dict") else dict(getattr(item, "__dict__", {}))
            for item in variants
        ]
    category = data.get("category")
    if category is not None and not isinstance(category, str):
        data["category"] = getattr(category, "value", str(category))
    return data


def _write_fallback_report(
    *,
    profile: str,
    snapshot: dict[str, bool],
    home: str | Path,
    error: Exception,
) -> dict[str, Any]:
    features = [_feature_to_dict(feature) for feature in feature_specs()]
    inactive = [
        feature
        for feature in features
        if not bool(snapshot.get(str(feature.get("feature_id")), False))
    ]
    recommended = [
        {
            "feature_id": feature.get("feature_id"),
            "reason": "No current activation evidence in local fallback report.",
            "protocol": "system_audit",
        }
        for feature in inactive[:8]
    ]
    report_dir = Path(home) / "reports" / PROJECT_ID
    report_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "project": {
            "project_id": PROJECT_ID,
            "name": "AN-RA",
            "privacy_mode": "aggregate",
        },
        "profile": profile,
        "features": features,
        "recommended_experiments": recommended,
        "activation_snapshot": snapshot,
        "fallback": {
            "reason": "ThirdEye SDK storage failed; local AN-RA fallback report was written.",
            "error_type": type(error).__name__,
            "error": str(error),
            "generated_at": time.time(),
        },
    }
    report_path = report_dir / "one_click_fallback.json"
    report_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    bundle["report_paths"] = {"fallback": str(report_path)}
    return bundle


def activation_snapshot(model: Any | None = None) -> dict[str, bool]:
    snapshot = {
        "anra.optimizer": (OUTPUT_V2_DIR / "v2_optimizer_bakeoff_report.json").exists(),
        "anra.data_mix": (OUTPUT_V2_DIR / "v2_dataset_mix.json").exists(),
        "anra.rlvr": (OUTPUT_V2_DIR / "v2_rlvr_report.json").exists(),
        "anra.memory": (OUTPUT_V2_DIR / "memory_benchmark.json").exists(),
        "anra.verification": False,
        "anra.inference_efficiency": False,
    }
    if model is None:
        return snapshot
    hal = getattr(model, "hal_module", None)
    snapshot.update(
        {
            "anra.esv": hasattr(model, "esv_module"),
            "anra.rim": bool(getattr(model, "use_rim", False))
            and len(getattr(model, "rim_modules", ())) > 0,
            "anra.dstp": bool(getattr(model, "use_dstp", False))
            and hasattr(model, "dstp_temperature_log"),
            "anra.mod": len(getattr(model, "mod_routers", {})) > 0,
            "anra.hal": bool(getattr(model, "use_hal", False)) and hal is not None,
            "anra.hal.attention_temperature": callable(getattr(hal, "attention_temperature", None)),
            "anra.hal.rlvr_feedback": False,
            "anra.hal.memory_threshold": callable(getattr(hal, "memory_threshold", None)),
            "anra.hal.ouroboros_weights": callable(getattr(hal, "ouroboros_weights", None)),
            "anra.cognition": getattr(model, "cognitive_extension", None) is not None,
        }
    )
    return snapshot


def record_activation_audit(eye: Any, snapshot: dict[str, bool]) -> None:
    sdk = _load_sdk()
    for feature in feature_specs():
        active = bool(snapshot.get(feature.feature_id, False))
        protocol = sdk["ProtocolSpec"](
            protocol_id=f"audit:{feature.feature_id}",
            kind=sdk["ProtocolKind"].SYSTEM_AUDIT,
            feature_id=feature.feature_id,
        )
        evidence = sdk["grade_evidence"](
            project_id=PROJECT_ID,
            feature_id=feature.feature_id,
            protocol=protocol,
            run_ids=[],
            activation_verified=active,
        )
        payload = evidence.to_dict()
        payload["summary"] = (
            f"{feature.name} activation verified."
            if active
            else f"{feature.name} has no current activation proof."
        )
        eye.record_evidence(payload)


def run_one_click(
    *,
    profile: str = "auto",
    model: Any | None = None,
    home: str | Path = THIRDEYE_HOME,
) -> dict[str, Any]:
    snapshot = activation_snapshot(model)
    try:
        eye = register_project(home)
        record_activation_audit(eye, snapshot)
        result = eye.evaluate(PROJECT_ID, profile)
        result["activation_snapshot"] = snapshot
        return result
    except sqlite3.OperationalError as exc:
        if "unixepoch" not in str(exc).lower():
            raise
        return _write_fallback_report(
            profile=profile,
            snapshot=snapshot,
            home=home,
            error=exc,
        )
    except Exception as exc:
        if "unixepoch" not in str(exc).lower():
            raise
        return _write_fallback_report(
            profile=profile,
            snapshot=snapshot,
            home=home,
            error=exc,
        )


def write_summary(result: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return target
