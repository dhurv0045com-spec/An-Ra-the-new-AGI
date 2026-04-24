from __future__ import annotations

from dataclasses import dataclass


MODEL_LINE = "v2"


@dataclass(frozen=True)
class V2ModelConfig:
    vocab_size: int = 4096
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    block_size: int = 256
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0


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
    teacher_examples_target: int = 384
    symbolic_examples_target: int = 192
    max_mixture_examples: int = 16000
    milestone_every_sessions: int = 5
    plateau_window: int = 3
    plateau_delta: float = 0.01


V2_MODEL = V2ModelConfig()
V2_TRAINING = V2TrainingConfig()


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
