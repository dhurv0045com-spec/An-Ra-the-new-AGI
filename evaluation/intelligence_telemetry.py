"""ThirdEye intelligence telemetry for AN-RA training and evaluation."""

from __future__ import annotations

import os
import time
from typing import Any

from anra.anra_paths import OUTPUT_V2_DIR

from evaluation.thirdeye_adapter import PROJECT_ID, _load_sdk, register_project


def subsystem_specs() -> list[Any]:
    subsystem_spec = _load_sdk()["SubsystemSpec"]

    return [
        subsystem_spec(
            "anra.embeddings",
            "Token Embeddings",
            "AN-RA Research",
            "representation",
            module_patterns=("*token_embedding_table", "*token_embedding"),
            expected_signals=("activation.rms", "gradients.norm", "update.ratio"),
        ),
        subsystem_spec(
            "anra.attention",
            "Attention",
            "AN-RA Research",
            "architecture",
            module_patterns=("*blocks.*.attn",),
            expected_signals=("activation.rms", "activation.saturation", "update.ratio"),
        ),
        subsystem_spec(
            "anra.mlp",
            "Feed-Forward Networks",
            "AN-RA Research",
            "architecture",
            module_patterns=("*blocks.*.mlp",),
            expected_signals=("activation.rms", "activation.sparsity", "update.ratio"),
        ),
        subsystem_spec(
            "anra.normalization",
            "Normalization",
            "AN-RA Research",
            "architecture",
            module_patterns=("*blocks.*.norm_*", "*norm_f"),
            expected_signals=("activation.rms", "activation.std"),
        ),
        subsystem_spec(
            "anra.esv",
            "Emotional State Vector",
            "AN-RA Research",
            "identity",
            module_patterns=("*esv_module", "*esv_module.*"),
            expected_signals=("activation.rms", "gradients.norm", "update.ratio"),
            protected=True,
        ),
        subsystem_spec(
            "anra.rim",
            "Residual Identity Modulation",
            "AN-RA Research",
            "identity",
            module_patterns=("*rim_modules.*",),
            expected_signals=("activation.rms", "gradients.norm", "update.ratio"),
            protected=True,
        ),
        subsystem_spec(
            "anra.mod",
            "Mixture of Depth Routing",
            "AN-RA Research",
            "routing",
            module_patterns=("*mod_routers.*",),
            expected_signals=("activation.sparsity", "gradients.norm", "update.ratio"),
        ),
        subsystem_spec(
            "anra.hal",
            "Hormonal Analog Layer",
            "AN-RA Research",
            "adaptation",
            module_patterns=("*hal_module", "*hal_module.*"),
            expected_signals=("activation.rms", "gradients.norm", "update.ratio"),
            protected=True,
        ),
        subsystem_spec(
            "anra.cognition",
            "Cognitive Extension",
            "AN-RA Research",
            "cognition",
            module_patterns=("*cognitive_extension", "*cognitive_extension.*"),
            expected_signals=("activation.rms", "gradients.norm", "update.ratio"),
        ),
        subsystem_spec(
            "anra.output",
            "Language Model Head",
            "AN-RA Research",
            "behavior",
            module_patterns=("*lm_head",),
            expected_signals=("activation.rms", "activation.saturation", "update.ratio"),
        ),
    ]


class ANRAIntelligenceSession:
    """Low-overhead bridge between an AN-RA training session and ThirdEye."""

    def __init__(self, model: Any, *, sample_every: int = 25) -> None:
        _load_sdk()
        from thirdeye.intelligence import IntelligenceMonitor, PyTorchSubsystemCollector

        self.subsystems = subsystem_specs()
        self.monitor = IntelligenceMonitor(self.subsystems)
        self.hooks = PyTorchSubsystemCollector(
            model,
            self.subsystems,
            self.monitor.collector,
            sample_every=sample_every,
        )
        self.started_at = time.perf_counter()
        self.last_step_at = self.started_at

    def begin_step(self, step: int) -> bool:
        return self.hooks.begin_step(step)

    def record_optimizer_step(
        self,
        *,
        step: int,
        loss: float,
        learning_rate: float,
        gradient_norm: float,
        tokens: int,
    ) -> None:
        now = time.perf_counter()
        elapsed = max(now - self.last_step_at, 1e-9)
        self.last_step_at = now
        subsystem = self.hooks.capture_gradients(learning_rate=learning_rate)
        update_ratios = [value for key, value in subsystem.items() if key.endswith(".update_ratio")]
        self.monitor.collector.record_training_step(
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            update_ratio=(sum(update_ratios) / len(update_ratios) if update_ratios else None),
            tokens_per_second=float(tokens) / elapsed,
        )

    def finalize(
        self,
        *,
        checkpoint_id: str,
        capability_score: float | None = None,
        capability_samples: int = 1,
    ) -> dict[str, Any]:
        capability_target = _load_sdk()["CapabilityTarget"]

        estimate = self.monitor.checkpoint(checkpoint_id)
        if capability_score is not None:
            self.monitor.targets.append(
                capability_target(
                    target_id="anra.compact_eval.overall",
                    value=float(capability_score),
                    checkpoint_id=checkpoint_id,
                    evaluator="anra.run_compact_eval",
                    sample_count=max(1, int(capability_samples)),
                    metadata={"status": "held_out_behavioral_target"},
                )
            )
        eye = register_project()
        self.monitor.persist(eye, PROJECT_ID)
        if capability_score is not None:
            estimate = eye.calibrate_intelligence(
                PROJECT_ID,
                target_id="anra.compact_eval.overall",
                minimum_checkpoints=5,
            )
            eye.record_intelligence_estimate(PROJECT_ID, estimate)
        report = self.monitor.write_report(
            OUTPUT_V2_DIR / "thirdeye" / "reports" / PROJECT_ID / "intelligence.json"
        )
        eye.evaluate(PROJECT_ID, "quick")
        self.hooks.close()
        return {
            "checkpoint_id": checkpoint_id,
            "estimate": estimate.to_dict(),
            "report": str(report),
        }


def create_intelligence_session(model: Any) -> ANRAIntelligenceSession | None:
    if os.environ.get("ANRA_THIRDEYE_INTELLIGENCE", "1") == "0":
        return None
    try:
        return ANRAIntelligenceSession(
            model,
            sample_every=int(os.environ.get("ANRA_THIRDEYE_SAMPLE_EVERY", "25")),
        )
    except Exception as exc:
        print(f"[ThirdEye] Intelligence telemetry unavailable: {exc}", flush=True)
        return None
