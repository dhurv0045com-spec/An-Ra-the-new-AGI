"""The V5-Next contract: geometry + output-space mode + frozen scientific flags.

Status semantics (docs/cymek/next_core/NEXT_CORE_SPEC.json):
  - full_softmax is the CANONICAL output path (V5-identical likelihood
    semantics); participating_mask / inactive_offset are EXPERIMENT_ONLY
    training treatments mirroring CYR-GPU-014-R1C and the independent
    BRAMASTRA B04 implementation. They require allow_experimental=True,
    are recorded in the contract hash, and must never ship as default.
  - eos_supervised is EVIDENCE-LOCKED (ARK-007R-era BRAMASTRA terminal
    experiment + CITADEL T1D postmortem) and can only be True.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

CANONICAL_OUTPUT_MODE = "full_softmax"
EXPERIMENTAL_OUTPUT_MODES = ("participating_mask", "inactive_offset")

_STATUS_LOCKED = "LOCKED"
_STATUS_EXPERIMENT_ONLY = "EXPERIMENT_ONLY"


@dataclass(frozen=True)
class NextCoreContract:
    vocabulary_size: int
    width: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dimension: int
    ffn_width: int
    context_length: int
    rope_base: float = 10_000.0
    norm_epsilon: float = 1e-5
    output_mode: str = CANONICAL_OUTPUT_MODE
    eos_supervised: bool = True
    allow_experimental: bool = False
    status_overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width != self.query_heads * self.head_dimension:
            raise ValueError("width must equal query_heads * head_dimension")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        if self.head_dimension % 2:
            raise ValueError("head dimension must be even for pairwise RoPE")
        if not self.eos_supervised:
            raise ValueError("eos_supervised is EVIDENCE-LOCKED; it cannot be disabled")
        if self.output_mode not in (CANONICAL_OUTPUT_MODE,) + EXPERIMENTAL_OUTPUT_MODES:
            raise ValueError(f"unknown output mode: {self.output_mode}")
        if self.output_mode != CANONICAL_OUTPUT_MODE and not self.allow_experimental:
            raise ValueError(
                "experimental output modes are EXPERIMENT_ONLY; pass allow_experimental=True")

    # -- geometry component statuses (mirrors NEXT_CORE_SPEC.json) ----------
    def component_statuses(self) -> dict[str, str]:
        statuses = {
            "family": _STATUS_LOCKED,
            "attention_gqa": _STATUS_LOCKED,
            "qk_norm_affine": _STATUS_LOCKED,
            "rmsnorm_prenorm": _STATUS_LOCKED,
            "swiglu_ffn": _STATUS_LOCKED,
            "rope_pairwise": _STATUS_LOCKED,
            "residual_init_1_over_sqrt2L": _STATUS_LOCKED,
            "tied_embeddings": _STATUS_LOCKED,
            "eos_supervised": _STATUS_LOCKED,
            "output_mode": (
                _STATUS_LOCKED if self.output_mode == CANONICAL_OUTPUT_MODE
                else _STATUS_EXPERIMENT_ONLY
            ),
            "vocabulary_size": self.status_overrides.get("vocabulary_size", "BLOCKED"),
        }
        return statuses

    # -- exact parameter accounting (same formula as v5_contracts) ----------
    def parameter_receipt(self) -> dict[str, int]:
        w = self.width
        q_width = self.query_heads * self.head_dimension
        kv_width = self.kv_heads * self.head_dimension
        embedding = self.vocabulary_size * w
        attention = w * q_width + 2 * w * kv_width + q_width * w
        ffn = 3 * w * self.ffn_width
        block_norms = 2 * w
        qk_norms = q_width + kv_width
        block = attention + ffn + block_norms + qk_norms
        all_blocks = self.layers * block
        final_norm = w
        return {
            "embedding": embedding,
            "attention_per_layer": attention,
            "ffn_per_layer": ffn,
            "block_norms_per_layer": block_norms,
            "qk_norms_per_layer": qk_norms,
            "block_total": block,
            "all_blocks": all_blocks,
            "final_norm": final_norm,
            "output_head": 0,
            "total": embedding + all_blocks + final_norm,
        }

    def canonical(self) -> dict[str, object]:
        payload = asdict(self)
        payload["parameter_receipt"] = self.parameter_receipt()
        payload["component_statuses"] = self.component_statuses()
        return payload

    def identity_sha256(self) -> str:
        """Scientific identity hash: any change to geometry OR output-mode OR
        the EOS contract changes this hash, so receipts cannot silently mix
        canonical and experimental configurations."""
        canonical = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


TINY_REFERENCE_GEOMETRY = NextCoreContract(
    vocabulary_size=128,
    width=64,
    layers=2,
    query_heads=4,
    kv_heads=2,
    head_dimension=16,
    ffn_width=128,
    context_length=32,
)


def output_training_logits(
    logits, *, output_mode: str, active_ids: tuple[int, ...], inactive_offset_log_value: float
):
    """Apply an EXPERIMENT_ONLY training-time output treatment to base logits.

    - participating_mask: only ``active_ids`` participate in the training
      softmax (R1C MASK_K family).
    - inactive_offset: all logits remain, inactive rows receive a fixed
      subtraction of ``inactive_offset_log_value`` (R1C OFFSET family).
    The canonical full-softmax path never calls this function.
    """
    if output_mode == CANONICAL_OUTPUT_MODE:
        return logits
    import torch

    mask = torch.zeros_like(logits)
    mask[..., list(active_ids)] = 1.0
    if output_mode == "participating_mask":
        return logits + (1.0 - mask) * torch.finfo(logits.dtype).min
    if output_mode == "inactive_offset":
        return logits - (1.0 - mask) * inactive_offset_log_value
    raise ValueError(f"unknown output mode: {output_mode}")
