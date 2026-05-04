from __future__ import annotations

from dataclasses import dataclass


MODEL_LINE = "v2"
CANONICAL_VOCAB_SIZE = 8192
CANONICAL_PAD_TOKEN_ID = 0
CANONICAL_UNK_TOKEN_ID = 1
CANONICAL_SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


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


V2_MODEL = V2ModelConfig()
V2_TRAINING = V2TrainingConfig()
EXPECTED_TOKENIZER_VOCAB_SIZE = CANONICAL_VOCAB_SIZE
EXPECTED_PAD_TOKEN_ID = CANONICAL_PAD_TOKEN_ID
EXPECTED_SPECIAL_TOKENS = CANONICAL_SPECIAL_TOKENS


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
    "curriculum": "v2_next_session_curriculum.json",
    "run_report": "v2_unified_training_report.json",
    "mix_report": "v2_dataset_mix.json",
    "session_state": "v2_session_state.json",
    "finetune_report": "v2_finetune_report.json",
    "improvement_report": "v2_improvement_report.json",
    "audit_report": "v2_audit_report.json",
}
