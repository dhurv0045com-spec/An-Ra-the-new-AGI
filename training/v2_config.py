from __future__ import annotations

# AN-RA iterate500 branch:
# "frontier" is the only public training profile.
# It is the 500M-class experiment model; the current build has 499,167,047 parameters:
# 1280d, 28L, 16 attention heads, 4 KV heads, 1024 context, HAL enabled.
from dataclasses import dataclass

MODEL_LINE = "v2"
TOKENIZER_SCHEMA_VERSION = 3
CHECKPOINT_SCHEMA_VERSION = 6
V2_FRONTIER_PARAMETER_COUNT = 499_167_047
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
# V4 append-only growth ceilings. 16,384 is the in-session-proven fallback
# (MASTER_UPGRADE fallback route: "V4-32k blocked => V4-16k append => >=1.2x").
# 32,768 is the canonical V4 ceiling (Layer 3): candidates come from the >=50MB
# campaign corpus, IDs 0-8208 immutable, byte + reserved multimodal/control IDs.
TOKENIZER_V4_VOCAB_SIZE = 16_384
TOKENIZER_V4_32K_VOCAB_SIZE = 32_768
CANONICAL_V4_VOCAB_SIZE = TOKENIZER_V4_32K_VOCAB_SIZE
V4_VOCAB_SIZES = (TOKENIZER_V4_VOCAB_SIZE, TOKENIZER_V4_32K_VOCAB_SIZE)
V2_FRONTIER_V4_PARAMETER_COUNT = 509_631_047
# 499,167,047 + (32,768 - 8,209) * 1,280 (tied-embedding append contract).
V2_FRONTIER_V4_32K_PARAMETER_COUNT = 530_602_567


def is_v4_vocab_size(vocab_size: int) -> bool:
    """True when the size is one of the sanctioned append-only V4 ceilings."""
    return int(vocab_size) in V4_VOCAB_SIZES
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
    mod_layers: tuple = (2, 4, 6)
    base_seq_len: int = 512
    target_seq_len: int = 2048
    use_hal: bool = False


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
class V2TrainingConfig:
    batch_size: int = 32
    grad_accum_steps: int = 8
    session_minutes: int = 30
    answer_loss_weight: float = 1.75
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


V2_MODEL = V2ModelConfig()
V2_FRONTIER = V2FrontierModelConfig()
V2_TRAINING = V2TrainingConfig()
V2_FRONTIER_TRAINING = V2FrontierTrainingConfig()
# Backward-compatible aliases for older imports. On iterate500 these point to
# the 500M-class frontier config, not a 1B model.
V2_1B_FRONTIER = V2_FRONTIER
V2_1B_TRAINING = V2_FRONTIER_TRAINING
EXPECTED_TOKENIZER_VOCAB_SIZE = CANONICAL_VOCAB_SIZE
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
        TOKENIZER_V4_VOCAB_SIZE: V2_FRONTIER_V4_PARAMETER_COUNT,
        TOKENIZER_V4_32K_VOCAB_SIZE: V2_FRONTIER_V4_32K_PARAMETER_COUNT,
    }
    expected = pinned_v4_counts.get(int(vocab_size))
    if expected is not None and count != expected:
        raise AssertionError(
            f"Frontier V4 parameter contract mismatch at vocab {vocab_size:,}: "
            f"{count:,} != {expected:,}"
        )
    return count


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


MODEL_SIZES = {
    "frontier": (V2_FRONTIER, V2_FRONTIER_TRAINING),
}
MODEL_PROFILES = MODEL_SIZES


def resolve_model_profile(name: str) -> tuple[V2ModelConfig, V2TrainingConfig]:
    key = str(name).strip().lower()
    if key not in MODEL_SIZES:
        raise ValueError(f"Unknown model profile {name!r}; expected one of {sorted(MODEL_SIZES)}")
    return MODEL_SIZES[key]
