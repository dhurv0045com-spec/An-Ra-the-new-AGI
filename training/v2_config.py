from __future__ import annotations

# Canonical An-Ra V4 training contract. Historical constants below exist only
# for forensic readers; MODEL_SIZES exposes exactly one canonical profile.
from dataclasses import dataclass

from training.reproducibility import CANONICAL_TRAINING_SEED

MODEL_LINE = "v4"
TOKENIZER_SCHEMA_VERSION = 4
CHECKPOINT_SCHEMA_VERSION = 9
LEGACY_V2_FRONTIER_PARAMETER_COUNT = 499_167_047
V2_FRONTIER_PARAMETER_COUNT = 499_167_075
V2_FRONTIER_TRANSFORMER_PARAMETER_COUNT = 496_857_600
BASE_VOCAB_SIZE = 8192
CANONICAL_PAD_TOKEN_ID = 0
CANONICAL_UNK_TOKEN_ID = 1
BASE_SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<sep>",
    "<code>",
    "</code>",
    "<think>",
    "</think>",
    "<goal>",
    "<ESV:v>",
    "<ESV:a>",
    "<ESV:d>",
]
DFC_SPECIAL_TOKENS = [
    "<state>",
    "</state>",
    "</goal>",
    "<cons>",
    "</cons>",
    "<hyp>",
    "</hyp>",
    "<act>",
    "</act>",
    "<obs>",
    "</obs>",
    "<verify>",
    "</verify>",
    "<upd>",
    "</upd>",
    "<err>",
    "</err>",
]
CANONICAL_SPECIAL_TOKENS = BASE_SPECIAL_TOKENS + DFC_SPECIAL_TOKENS
CANONICAL_VOCAB_SIZE = BASE_VOCAB_SIZE + len(DFC_SPECIAL_TOKENS)
# V4 has one vocabulary contract. Historical V3/16k construction paths are
# deliberately absent from supported configuration.
TOKENIZER_V4_32K_VOCAB_SIZE = 32_768
CANONICAL_V4_VOCAB_SIZE = TOKENIZER_V4_32K_VOCAB_SIZE
# 499,167,075 + (32,768 - 8,209) * 1,280 (tied-embedding append contract).
V2_FRONTIER_V4_32K_PARAMETER_COUNT = 530_602_595
CANONICAL_MODEL_PROFILE = "anra-v4-180m"
ANRA_V4_MODEL_PARAMETER_COUNT = 181_132_071
ANRA_V4_GROWTH_MODEL_PROFILE = "anra-v4-500m-growth"
ANRA_V4_GROWTH_MODEL_PARAMETER_COUNT = 499_880_031
CANONICAL_FOUNDATION_OPTIMIZER = "adamw"
CANONICAL_FOUNDATION_SCHEDULE = "cosine_with_warmup_v1"


def is_v4_vocab_size(vocab_size: int) -> bool:
    """True only for the canonical V4 vocabulary."""
    return int(vocab_size) == CANONICAL_V4_VOCAB_SIZE


CANONICAL_SPECIAL_TOKEN_IDS = {
    **{token: index for index, token in enumerate(BASE_SPECIAL_TOKENS)},
    **{token: BASE_VOCAB_SIZE + index for index, token in enumerate(DFC_SPECIAL_TOKENS)},
}


@dataclass(frozen=True)
class V2ModelConfig:
    vocab_size: int = CANONICAL_VOCAB_SIZE
    pad_token_id: int = CANONICAL_PAD_TOKEN_ID
    n_embd: int = 512
    n_head: int = 8
    n_kv_head: int = 2
    n_layer: int = 8
    block_size: int = 512
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    d_ff: int | None = None
    rope_base: int = 10_000
    mod_layers: tuple = (2, 4, 6)
    base_seq_len: int = 512
    target_seq_len: int = 2048
    use_hal: bool = False
    use_qk_norm: bool = True
    sliding_window: int = 1024
    full_attention_every: int = 4

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.n_embd <= 0 or self.n_layer <= 0:
            raise ValueError("model vocabulary, width, and depth must be positive")
        if self.n_head <= 0 or self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by positive n_head")
        if self.n_kv_head <= 0 or self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by positive n_kv_head")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.d_ff is not None and (self.d_ff <= 0 or self.d_ff % 64 != 0):
            raise ValueError("d_ff must be positive and divisible by 64")
        if self.block_size <= 0 or self.base_seq_len <= 0 or self.target_seq_len <= 0:
            raise ValueError("context lengths must be positive")
        if len(set(self.mod_layers)) != len(self.mod_layers) or any(
            layer < 0 or layer >= self.n_layer for layer in self.mod_layers
        ):
            raise ValueError("mod_layers must be unique zero-based layer indices")


@dataclass(frozen=True)
class V2FrontierModelConfig(V2ModelConfig):
    n_embd: int = 1280
    n_layer: int = 28
    n_head: int = 16
    n_kv_head: int = 4
    block_size: int = 1024
    vocab_size: int = CANONICAL_VOCAB_SIZE
    mod_layers: tuple = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26)
    base_seq_len: int = 1024
    target_seq_len: int = 1024
    science_ratio: float = 0.30
    action_trace_ratio: float = 0.20
    constraint_ratio: float = 0.20
    cross_domain_ratio: float = 0.10
    identity_ratio: float = 0.10
    base_ratio: float = 0.10
    aux_constraint_loss_weight: float = 0.25
    aux_prediction_loss_weight: float = 0.20
    aux_uncertainty_loss_weight: float = 0.15
    rlvr_tool_reward_weight: float = 1.0
    use_hal: bool = True


@dataclass(frozen=True)
class V2Pilot50ModelConfig(V2ModelConfig):
    """Approximately 58M parameters: cheap scaling-law and pipeline anchor."""

    n_embd: int = 640
    n_layer: int = 12
    n_head: int = 10
    n_kv_head: int = 2
    block_size: int = 2048
    mod_layers: tuple = (3, 5, 7, 9, 11)
    base_seq_len: int = 2048
    target_seq_len: int = 2048
    use_hal: bool = False


@dataclass(frozen=True)
class V2Pilot150ModelConfig(V2ModelConfig):
    """Historical 160M geometry retained for forensic configuration parsing."""

    n_embd: int = 896
    n_layer: int = 18
    n_head: int = 14
    n_kv_head: int = 2
    block_size: int = 2048
    mod_layers: tuple = (4, 6, 8, 10, 12, 14, 16)
    base_seq_len: int = 2048
    target_seq_len: int = 2048
    use_hal: bool = False


@dataclass(frozen=True)
class AnRaV4ModelConfig(V2ModelConfig):
    """The sole active model: 181M parameters, optimized for one T4 session."""

    vocab_size: int = CANONICAL_V4_VOCAB_SIZE
    n_embd: int = 896
    n_layer: int = 18
    n_head: int = 14
    n_kv_head: int = 2
    d_ff: int = 2432
    block_size: int = 2048
    mod_layers: tuple = (4, 6, 8, 10, 12, 14, 16)
    base_seq_len: int = 2048
    target_seq_len: int = 2048
    use_hal: bool = False


@dataclass(frozen=True)
class AnRaV4Growth500MModelConfig(V2ModelConfig):
    """Registered 500M V4 child; construction requires explicit growth opt-in."""

    vocab_size: int = CANONICAL_V4_VOCAB_SIZE
    n_embd: int = 1280
    n_layer: int = 27
    n_head: int = 20
    n_kv_head: int = 2
    d_ff: int = 3456
    block_size: int = 2048
    mod_layers: tuple = tuple(range(4, 27, 2))
    base_seq_len: int = 2048
    target_seq_len: int = 2048
    use_hal: bool = False


@dataclass(frozen=True)
class V2TrainingConfig:
    batch_size: int = 32
    grad_accum_steps: int = 8
    max_grad_norm: float = 1.0
    verified_process_multiplier: float = 1.25
    session_minutes: int = 30
    answer_loss_weight: float = 1.75
    logit_z_loss_weight: float = 1e-4
    teacher_ratio: float = 0.10
    own_ratio: float = 0.65
    identity_ratio: float = 0.15
    symbolic_ratio: float = 0.05
    replay_ratio: float = 0.05
    civ_identity_min_score: float = 0.68
    teacher_examples_target: int = 384
    symbolic_examples_target: int = 192
    max_mixture_examples: int = 16000
    milestone_every_sessions: int = 5
    plateau_window: int = 5
    plateau_delta: float = 0.08
    unified_trainer_overhead_minutes: int = 5


@dataclass(frozen=True)
class V2FrontierTrainingConfig(V2TrainingConfig):
    """
    Training hyperparameters for the 500M frontier model.
    Separate from V2_TRAINING; do not modify V2_TRAINING.
    """

    batch_size: int = 1
    grad_accum_steps: int = 16
    session_minutes: int = 180
    learning_rate: float = 1e-4
    warmup_steps: int = 1_000
    max_steps: int = 50_000
    min_lr: float = 1e-5
    weight_decay: float = 0.05
    max_mixture_examples: int = 4096
    milestone_every_sessions: int = 3
    gradient_checkpointing: bool = True
    use_bfloat16: bool = True
    own_ratio: float = 0.55
    identity_ratio: float = 0.15
    teacher_ratio: float = 0.10
    symbolic_ratio: float = 0.10
    replay_ratio: float = 0.10


@dataclass(frozen=True)
class V2Pilot50TrainingConfig(V2TrainingConfig):
    batch_size: int = 8
    grad_accum_steps: int = 8
    session_minutes: int = 120
    learning_rate: float = 3e-4
    warmup_steps: int = 1_000
    max_steps: int = 50_000
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    max_mixture_examples: int = 16_000
    gradient_checkpointing: bool = False


@dataclass(frozen=True)
class V2Pilot150TrainingConfig(V2TrainingConfig):
    batch_size: int = 4
    grad_accum_steps: int = 8
    session_minutes: int = 180
    learning_rate: float = 2e-4
    warmup_steps: int = 1_000
    max_steps: int = 100_000
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    max_mixture_examples: int = 16_000
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class AnRaV4TrainingConfig(V2Pilot150TrainingConfig):
    """Canonical V4 training settings for seed 1301."""

    batch_size: int = 1
    grad_accum_steps: int = 32
    seed: int = CANONICAL_TRAINING_SEED
    optimizer: str = CANONICAL_FOUNDATION_OPTIMIZER


@dataclass(frozen=True)
class AnRaV4Growth500MTrainingConfig(AnRaV4TrainingConfig):
    """Post-growth stabilization defaults; optimizer state is always new."""

    learning_rate: float = 5e-5
    min_lr: float = 5e-6
    warmup_steps: int = 1_000
    batch_size: int = 1
    grad_accum_steps: int = 64
    gradient_checkpointing: bool = True


V2_MODEL = V2ModelConfig()
V2_FRONTIER = V2FrontierModelConfig()
V2_PILOT_50M = V2Pilot50ModelConfig()
V2_PILOT_150M = V2Pilot150ModelConfig()
V2_TRAINING = V2TrainingConfig()
V2_FRONTIER_TRAINING = V2FrontierTrainingConfig()
V2_PILOT_50M_TRAINING = V2Pilot50TrainingConfig()
V2_PILOT_150M_TRAINING = V2Pilot150TrainingConfig()
ANRA_V4_MODEL = AnRaV4ModelConfig()
ANRA_V4_TRAINING = AnRaV4TrainingConfig()
ANRA_V4_GROWTH_MODEL = AnRaV4Growth500MModelConfig()
ANRA_V4_GROWTH_TRAINING = AnRaV4Growth500MTrainingConfig()
# Backward-compatible aliases for older imports. On iterate500 these point to
# the 500M-class frontier config, not a 1B model.
V2_1B_FRONTIER = V2_FRONTIER
V2_1B_TRAINING = V2_FRONTIER_TRAINING
EXPECTED_TOKENIZER_VOCAB_SIZE = CANONICAL_V4_VOCAB_SIZE
EXPECTED_PAD_TOKEN_ID = CANONICAL_PAD_TOKEN_ID
EXPECTED_SPECIAL_TOKENS = CANONICAL_SPECIAL_TOKENS
EXPECTED_SPECIAL_TOKEN_IDS = CANONICAL_SPECIAL_TOKEN_IDS


def frontier_parameter_count(vocab_size: int = CANONICAL_VOCAB_SIZE) -> int:
    """Parameter contract for append-only vocabulary growth with tied embeddings."""
    if vocab_size < CANONICAL_VOCAB_SIZE:
        raise ValueError("Frontier vocabulary cannot remove canonical token rows")
    count = (
        V2_FRONTIER_PARAMETER_COUNT + (int(vocab_size) - CANONICAL_VOCAB_SIZE) * V2_FRONTIER.n_embd
    )
    pinned_v4_counts = {
        TOKENIZER_V4_32K_VOCAB_SIZE: V2_FRONTIER_V4_32K_PARAMETER_COUNT,
    }
    expected = pinned_v4_counts.get(int(vocab_size))
    if expected is not None and count != expected:
        raise AssertionError(
            f"Frontier V4 parameter contract mismatch at vocab {vocab_size:,}: "
            f"{count:,} != {expected:,}"
        )
    return count


def model_parameter_count(
    config: V2ModelConfig,
    vocab_size: int | None = None,
    *,
    mtp_depth: int = 0,
    moe_routed_experts: int = 0,
) -> int:
    """Exact parameter count for the shared dense/native architecture."""
    vocab = int(vocab_size or config.vocab_size)
    width = int(config.n_embd)
    layers = int(config.n_layer)
    head_dim = width // int(config.n_head)
    kv_width = int(config.n_kv_head) * head_dim
    hidden = int(config.d_ff or ((int(8 / 3 * width) + 63) // 64) * 64)
    per_block = (
        2 * width
        + width * width
        + 2 * width * kv_width
        + width * width
        + 3 * width * hidden
    )
    router_params = len(config.mod_layers) * (width + 4)
    esv_dim = min(64, width)
    esv_predictor = esv_dim * 3 + 3
    rim_params = layers * (width * esv_dim + 1)
    depth_controls = 3 * layers
    base = (
        vocab * width
        + layers * per_block
        + router_params
        + width
        + esv_predictor
        + rim_params
        + depth_controls
    )
    # Each future-token head has one RMSNorm vector and one d_model x
    # d_model projection. Vocabulary projection reuses the tied embedding.
    mtp = max(0, int(mtp_depth)) * (width + width * width)
    routed = max(0, int(moe_routed_experts))
    moe = layers * routed * (3 * width * hidden) + layers * width * routed
    return base + mtp + moe


def model_parameter_breakdown(
    config: V2ModelConfig,
    vocab_size: int | None = None,
    *,
    mtp_depth: int = 0,
    moe_routed_experts: int = 0,
) -> dict[str, int]:
    """Explain the exact installed parameter contract without implying activation.

    Native pilot/control tensors are part of the frozen V4 checkpoint ABI even
    while the dense foundation keeps them disabled and frozen.  Reporting them
    separately prevents the total parameter count from being mistaken for the
    amount of active sparse or cognitive machinery.
    """

    vocab = int(vocab_size or config.vocab_size)
    width = int(config.n_embd)
    layers = int(config.n_layer)
    head_dim = width // int(config.n_head)
    kv_width = int(config.n_kv_head) * head_dim
    hidden = int(config.d_ff or ((int(8 / 3 * width) + 63) // 64) * 64)
    per_block = (
        2 * width
        + width * width
        + 2 * width * kv_width
        + width * width
        + 3 * width * hidden
    )
    dense = vocab * width + layers * per_block + width
    router = len(config.mod_layers) * (width + 4)
    esv = min(64, width) * 3 + 3
    rim = layers * (width * min(64, width) + 1)
    depth_controls = 3 * layers
    installed_native_pilots = router + esv + rim + depth_controls
    mtp = max(0, int(mtp_depth)) * (width + width * width)
    routed = max(0, int(moe_routed_experts))
    moe = layers * routed * (3 * width * hidden) + layers * width * routed
    total = dense + installed_native_pilots + mtp + moe
    expected = model_parameter_count(
        config,
        vocab,
        mtp_depth=mtp_depth,
        moe_routed_experts=moe_routed_experts,
    )
    if total != expected:
        raise AssertionError(f"parameter breakdown drifted: {total:,} != {expected:,}")
    return {
        "dense_parameters": dense,
        "installed_native_pilot_parameters": installed_native_pilots,
        "mtp_parameters": mtp,
        "moe_parameters": moe,
        "total_parameters": total,
    }


IDENTITY_KEYWORDS = [
    "who are you",
    "what are you",
    "what is an-ra",
    "who created you",
    "what is your purpose",
    "i am",
    "an-ra",
    "my purpose",
    "built you",
    "identity",
    "sovereign",
    "consciousness",
    "feelings",
    "self-improvement",
]


TEACHER_REJECT_PATTERNS = [
    "as an ai language model",
    "as a large language model",
    "chatgpt",
    "openai",
    "anthropic",
    "claude",
    "google gemini",
]


V2_REPORT_FILES = {
    "metrics": "v2_session_train_metrics.json",
    "hard_examples": "v2_hard_examples.json",
    "eval_summary": "v2_eval_summary.json",
    "eval_history": "v2_eval_history.jsonl",
    "golden_eval_baseline": "v2_golden_eval_baseline.json",
    "curriculum": "v2_next_session_curriculum.json",
    "run_report": "v2_unified_training_report.json",
    "mix_report": "v2_dataset_mix.json",
    "session_state": "v2_session_state.json",
    "finetune_report": "v2_finetune_report.json",
    "sparse_lora_report": "v2_sparse_lora_report.json",
    "optimizer_bakeoff": "v2_optimizer_bakeoff_report.json",
    "rlvr_report": "v2_rlvr_report.json",
    "gepa_report": "v2_gepa_report.json",
    "improvement_report": "v2_improvement_report.json",
    "audit_report": "v2_audit_report.json",
    "data_ingestion": "v2_data_ingestion_report.json",
    "metrics_snapshot": "metrics/latest.json",
    "ibs_latest": "ibs/latest.json",
    "memory_benchmark": "memory_benchmark.json",
    "cdr_report": "cdr_report.json",
    "growth_report": "model_growth_frontier.json",
    "causal_extension": "cognition/causal_extension_training.json",
    "mix_control": "v2_mix_control.json",
    "validation_history": "v2_validation_history.json",
}


@dataclass(frozen=True)
class ModelProfileRegistration:
    name: str
    status: str
    expected_parameters: int
    parent_profile: str | None = None
    requires_growth_manifest: bool = False
    scratch_training_allowed: bool = False


MODEL_SIZES = {CANONICAL_MODEL_PROFILE: (ANRA_V4_MODEL, ANRA_V4_TRAINING)}
EXPERIMENTAL_MODEL_PROFILES = {
    ANRA_V4_GROWTH_MODEL_PROFILE: (ANRA_V4_GROWTH_MODEL, ANRA_V4_GROWTH_TRAINING)
}
MODEL_PROFILES = {**MODEL_SIZES, **EXPERIMENTAL_MODEL_PROFILES}
MODEL_PROFILE_REGISTRY = {
    CANONICAL_MODEL_PROFILE: ModelProfileRegistration(
        name=CANONICAL_MODEL_PROFILE,
        status="active",
        expected_parameters=ANRA_V4_MODEL_PARAMETER_COUNT,
        scratch_training_allowed=True,
    ),
    ANRA_V4_GROWTH_MODEL_PROFILE: ModelProfileRegistration(
        name=ANRA_V4_GROWTH_MODEL_PROFILE,
        status="pilot",
        expected_parameters=ANRA_V4_GROWTH_MODEL_PARAMETER_COUNT,
        parent_profile=CANONICAL_MODEL_PROFILE,
        requires_growth_manifest=True,
        scratch_training_allowed=False,
    ),
}


def model_profile_registration(name: str) -> ModelProfileRegistration:
    key = str(name).strip().lower()
    if key not in MODEL_PROFILE_REGISTRY:
        raise ValueError(
            f"Unknown model profile {name!r}; expected one of {sorted(MODEL_PROFILE_REGISTRY)}"
        )
    return MODEL_PROFILE_REGISTRY[key]


def resolve_model_profile(
    name: str,
    *,
    allow_experimental: bool = False,
) -> tuple[V2ModelConfig, V2TrainingConfig]:
    key = str(name).strip().lower()
    if key in MODEL_SIZES:
        return MODEL_SIZES[key]
    if key in EXPERIMENTAL_MODEL_PROFILES:
        if not allow_experimental:
            raise ValueError(
                f"Model profile {name!r} is experimental and requires an explicit "
                "validated growth-manifest path; canonical launch defaults cannot select it"
            )
        return EXPERIMENTAL_MODEL_PROFILES[key]
    raise ValueError(f"Unknown model profile {name!r}; expected one of {sorted(MODEL_PROFILES)}")
