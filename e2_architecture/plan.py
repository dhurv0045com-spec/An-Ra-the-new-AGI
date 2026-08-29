"""Build the fail-closed static plan for the E2 P35 architecture screen.

This module performs configuration arithmetic only.  It does not construct a
model or authorize training.  The design separates shape, attention topology,
and context factors so an apparent win cannot silently combine several changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from v5_contracts.model_spec import ModelSpec


SCREEN_TOKENS = 200_000_000
EVALUATION_BOUNDARIES = (50_000_000, 100_000_000, 200_000_000)
SCREEN_SEEDS = (3101,)
FINALIST_SEEDS = (3101, 3102, 3103)


@dataclass(frozen=True, slots=True)
class StaticArm:
    name: str
    group: str
    factors: tuple[tuple[str, str], ...]
    model: ModelSpec

    def assert_valid(self) -> None:
        if self.group not in {"shape", "attention", "context"}:
            raise ValueError("unknown E2 arm group")
        if not self.name or not self.factors:
            raise ValueError("E2 arms require names and explicit factor levels")
        if len(dict(self.factors)) != len(self.factors):
            raise ValueError("duplicate E2 factor names")
        self.model.assert_valid()

    def receipt(self) -> dict[str, Any]:
        self.assert_valid()
        params = self.model.parameter_receipt()
        sequence = self.model.context_length
        projection_forward = (
            2
            * self.model.layers
            * (params.attention_per_layer + params.ffn_per_layer)
            * sequence
        )
        attention_forward = (
            4 * self.model.layers * sequence * sequence * self.model.width
        )
        kv_cache_bf16 = (
            2
            * self.model.layers
            * self.model.kv_heads
            * self.model.head_dimension
            * sequence
            * 2
        )
        return {
            "name": self.name,
            "group": self.group,
            "factors": dict(self.factors),
            "model": self.model.canonical(),
            "model_sha256": self.model.sha256(),
            "parameters": params.as_dict(),
            "parameter_distance_from_35m_pct": (params.total / 35_000_000 - 1) * 100,
            "idealized_6nd_flops_at_screen_tokens": 6 * params.total * SCREEN_TOKENS,
            "forward_flops_per_full_sequence_proxy": projection_forward + attention_forward,
            "attention_fraction_of_forward_proxy": attention_forward
            / (projection_forward + attention_forward),
            "kv_cache_bf16_bytes_per_full_sequence": kv_cache_bf16,
        }


@dataclass(frozen=True, slots=True)
class E2StaticPlan:
    schema: str
    experiment_id: str
    tokenizer_sha256: str | None
    corpus_manifest_sha256: str | None
    model_constructor_sha256: str | None
    screen_tokens: int
    evaluation_boundaries: tuple[int, ...]
    screening_seeds: tuple[int, ...]
    finalist_seeds: tuple[int, ...]
    arms: tuple[StaticArm, ...]

    def _group(self, name: str) -> tuple[StaticArm, ...]:
        return tuple(arm for arm in self.arms if arm.group == name)

    def assert_valid(self) -> None:
        if self.schema != "esoes-e2-static-plan/v1":
            raise ValueError("unexpected E2 plan schema")
        if self.screen_tokens != SCREEN_TOKENS:
            raise ValueError("E2 screen token budget drift")
        if self.evaluation_boundaries != EVALUATION_BOUNDARIES:
            raise ValueError("E2 evaluation boundaries drift")
        if self.screening_seeds != SCREEN_SEEDS or self.finalist_seeds != FINALIST_SEEDS:
            raise ValueError("E2 seed policy drift")
        if len(set(self.finalist_seeds)) != 3:
            raise ValueError("E2 finalists require three distinct seeds")
        if len({arm.name for arm in self.arms}) != len(self.arms):
            raise ValueError("duplicate E2 arm name")
        for arm in self.arms:
            arm.assert_valid()

        shapes = self._group("shape")
        if tuple(arm.name for arm in shapes) != ("deep-narrow", "middle", "wide-shallow"):
            raise ValueError("shape screen requires ordered deep/middle/wide arms")
        totals = [arm.model.parameter_receipt().total for arm in shapes]
        if max(totals) / min(totals) - 1 > 0.01:
            raise ValueError("shape arms must remain within one percent parameters")
        if not (
            shapes[0].model.layers > shapes[1].model.layers > shapes[2].model.layers
            and shapes[0].model.width < shapes[1].model.width < shapes[2].model.width
        ):
            raise ValueError("shape arms do not implement the intended depth/width ordering")
        for arm in shapes:
            if arm.model.kv_heads != arm.model.query_heads or not arm.model.qk_norm:
                raise ValueError("shape screen must hold MHA and QK norm fixed")
            if arm.model.context_length != 2048 or arm.model.vocabulary_size != 24_576:
                raise ValueError("shape screen must hold context and vocabulary fixed")

        attention = self._group("attention")
        if tuple(arm.name for arm in attention) != ("mha-qk", "gqa-qk", "gqa-no-qk"):
            raise ValueError("attention screen requires the preregistered fractional arms")
        fixed = ("vocabulary_size", "width", "layers", "query_heads", "head_dimension", "ffn_width", "context_length")
        for field in fixed:
            if len({getattr(arm.model, field) for arm in attention}) != 1:
                raise ValueError(f"attention screen changed fixed field {field}")
        if not (
            attention[0].model.kv_heads == attention[0].model.query_heads
            and attention[1].model.kv_heads < attention[1].model.query_heads
            and attention[2].model.kv_heads == attention[1].model.kv_heads
            and attention[0].model.qk_norm
            and attention[1].model.qk_norm
            and not attention[2].model.qk_norm
        ):
            raise ValueError("attention factor levels are not isolated")

        context = self._group("context")
        if tuple(arm.name for arm in context) != ("full-2k", "mixed-full-4k"):
            raise ValueError("context screen requires 2k and 4k arms")
        left, right = context
        left_data, right_data = left.model.canonical(), right.model.canonical()
        left_data.pop("context_length")
        right_data.pop("context_length")
        if left_data != right_data or (left.model.context_length, right.model.context_length) != (2048, 4096):
            raise ValueError("context screen changed more than context length")

    def status(self) -> str:
        self.assert_valid()
        if self.tokenizer_sha256 is None or self.corpus_manifest_sha256 is None:
            return "BLOCKED_E1_INPUTS"
        if self.model_constructor_sha256 is None:
            return "BLOCKED_MODEL_IMPLEMENTATION"
        for value in (self.tokenizer_sha256, self.corpus_manifest_sha256, self.model_constructor_sha256):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("E2 dependency hashes must be lowercase SHA-256")
        return "READY_FOR_BOUNDED_P35"

    def as_dict(self) -> dict[str, Any]:
        self.assert_valid()
        payload = {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "status": self.status(),
            "tokenizer_sha256": self.tokenizer_sha256,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "model_constructor_sha256": self.model_constructor_sha256,
            "screen_tokens": self.screen_tokens,
            "evaluation_boundaries": list(self.evaluation_boundaries),
            "screening_seeds": list(self.screening_seeds),
            "finalist_seeds": list(self.finalist_seeds),
            "comparison_policy": {
                "shape": "same token stream; parameter spread <=1%; MHA/QK/2k fixed",
                "attention": "same token stream and dimensions; interpolate outcomes by measured FLOPs",
                "context": "same raw-byte source order; report measured FLOPs and adversarial-position curves",
                "promotion": "worst-family fresh-OOD cognition per measured FLOP with substrate and throughput gates",
            },
            "arms": [arm.receipt() for arm in self.arms],
            "limitations": [
                "Static arithmetic is not target-accelerator throughput.",
                "No model has been constructed or trained by this plan.",
                "The plan remains blocked until E1/corpus and constructor hashes exist.",
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["plan_sha256"] = hashlib.sha256(encoded).hexdigest()
        return payload


def _model(*, width: int, layers: int, ffn: int, kv_heads: int, qk_norm: bool, context: int) -> ModelSpec:
    return ModelSpec(
        schema="anra-v5-p35-model-spec/v1",
        family="dense-decoder-transformer",
        vocabulary_size=24_576,
        width=width,
        layers=layers,
        query_heads=width // 64,
        kv_heads=kv_heads,
        head_dimension=64,
        ffn_width=ffn,
        context_length=context,
        rope_base=10_000.0,
        norm_epsilon=1e-5,
        tied_embeddings=True,
        qk_norm=qk_norm,
        qk_norm_affine=qk_norm,
        linear_bias=False,
        dropout=0.0,
    )


def build_plan(
    *,
    tokenizer_sha256: str | None = None,
    corpus_manifest_sha256: str | None = None,
    model_constructor_sha256: str | None = None,
) -> E2StaticPlan:
    arms = (
        StaticArm("deep-narrow", "shape", (("shape", "deep"),), _model(width=320, layers=24, ffn=768, kv_heads=5, qk_norm=True, context=2048)),
        StaticArm("middle", "shape", (("shape", "middle"),), _model(width=384, layers=16, ffn=896, kv_heads=6, qk_norm=True, context=2048)),
        StaticArm("wide-shallow", "shape", (("shape", "wide"),), _model(width=512, layers=8, ffn=1152, kv_heads=8, qk_norm=True, context=2048)),
        StaticArm("mha-qk", "attention", (("attention", "MHA"), ("qk_norm", "on")), _model(width=384, layers=16, ffn=896, kv_heads=6, qk_norm=True, context=2048)),
        StaticArm("gqa-qk", "attention", (("attention", "GQA-3:1"), ("qk_norm", "on")), _model(width=384, layers=16, ffn=896, kv_heads=2, qk_norm=True, context=2048)),
        StaticArm("gqa-no-qk", "attention", (("attention", "GQA-3:1"), ("qk_norm", "off")), _model(width=384, layers=16, ffn=896, kv_heads=2, qk_norm=False, context=2048)),
        StaticArm("full-2k", "context", (("context", "2k-full"),), _model(width=384, layers=16, ffn=896, kv_heads=2, qk_norm=True, context=2048)),
        StaticArm("mixed-full-4k", "context", (("context", "4k-mixed-full"),), _model(width=384, layers=16, ffn=896, kv_heads=2, qk_norm=True, context=4096)),
    )
    plan = E2StaticPlan(
        schema="esoes-e2-static-plan/v1",
        experiment_id="E2-P35-architecture-screen-v1",
        tokenizer_sha256=tokenizer_sha256,
        corpus_manifest_sha256=corpus_manifest_sha256,
        model_constructor_sha256=model_constructor_sha256,
        screen_tokens=SCREEN_TOKENS,
        evaluation_boundaries=EVALUATION_BOUNDARIES,
        screening_seeds=SCREEN_SEEDS,
        finalist_seeds=FINALIST_SEEDS,
        arms=arms,
    )
    plan.assert_valid()
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256")
    parser.add_argument("--corpus-manifest-sha256")
    parser.add_argument("--model-constructor-sha256")
    args = parser.parse_args()
    plan = build_plan(
        tokenizer_sha256=args.tokenizer_sha256,
        corpus_manifest_sha256=args.corpus_manifest_sha256,
        model_constructor_sha256=args.model_constructor_sha256,
    )
    payload = plan.as_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
