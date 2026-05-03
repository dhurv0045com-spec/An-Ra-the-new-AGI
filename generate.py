from __future__ import annotations

import math
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional

import torch

from anra_paths import ROOT, inject_all_paths
from training.v2_runtime import (
    build_v2_model,
    canonical_v2_checkpoint,
    load_checkpoint,
    load_or_build_v2_tokenizer,
    model_summary,
)

inject_all_paths()
logger = logging.getLogger(__name__)

try:
    from anra_paths import inject_all_paths as _inject
    _inject()
    from identity_injector import IdentityInjector as _IdentityInjector
    _IDENTITY_INJECTOR = _IdentityInjector()
except Exception:
    _IDENTITY_INJECTOR = None

try:
    from ghost_memory import GhostMemory as _GhostMemory
    _GHOST_MEMORY = _GhostMemory()
except Exception:
    _GHOST_MEMORY = None


@dataclass
class GenerationConfig:
    strategy: str = "nucleus"
    max_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.92
    beam_width: int = 4
    repetition_penalty: float = 1.15
    repetition_window: int = 64
    stop_strings: list[str] = field(default_factory=list)
    seed: Optional[int] = None


@dataclass
class GenerationTrace:
    output: str
    strategy: str
    tokens_generated: int
    time_ms: float
    entropy_curve: list[float]
    max_prob_curve: list[float]
    stopped_by: str
    repeated_ngrams_detected: bool
    kv_cache_compressed: bool = False
    memory_saved_mb: float = 0.0
    ghost_state_loaded: bool = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_GHOST_STORE: Dict[str, dict] = {}
_ACTIVE_GHOST: Dict[str, object] = {}


def _seed_all(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_runtime():
    tokenizer = load_or_build_v2_tokenizer()
    checkpoint = canonical_v2_checkpoint("ouroboros")
    use_ouroboros = checkpoint.exists()
    model = build_v2_model(vocab_size=tokenizer.vocab_size)
    if use_ouroboros:
        from ouroboros import OuroborosDecoder

        model = OuroborosDecoder(model, n_passes=3)
    model = model.to(DEVICE)
    if not checkpoint.exists():
        checkpoint = canonical_v2_checkpoint("identity")
    if not checkpoint.exists():
        checkpoint = canonical_v2_checkpoint("brain")
    load_checkpoint(model, None, None, None, checkpoint, device=DEVICE, strict=False)
    model.eval()
    return model, tokenizer, checkpoint


_MODEL = None
_TOKENIZER = None
_LOADED_CHECKPOINT = None


def _get_runtime():
    global _MODEL, _TOKENIZER, _LOADED_CHECKPOINT
    if _MODEL is None:
        _MODEL, _TOKENIZER, _LOADED_CHECKPOINT = _load_runtime()
    return _MODEL, _TOKENIZER, _LOADED_CHECKPOINT


def _apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], cfg: GenerationConfig) -> torch.Tensor:
    if cfg.repetition_penalty <= 1.0 or not generated_ids:
        return logits
    adjusted = logits.clone()
    recent = generated_ids[-cfg.repetition_window :]
    for token_id in set(recent):
        adjusted[token_id] /= cfg.repetition_penalty
    return adjusted


def _sample_next_token(logits: torch.Tensor, cfg: GenerationConfig, generated_ids: list[int]) -> tuple[int, float, float]:
    logits = _apply_repetition_penalty(logits, generated_ids, cfg)
    strategy = cfg.strategy.lower()
    temperature = max(cfg.temperature, 1e-6)

    if strategy == "greedy" or temperature < 1e-4:
        probs = torch.softmax(logits, dim=-1)
        next_token = int(torch.argmax(probs).item())
        entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
        return next_token, float(probs[next_token].item()), entropy

    if strategy == "beam":
        strategy = "topk"
    if strategy == "contrastive":
        strategy = "nucleus"

    logits = logits / temperature
    if strategy == "topk":
        top_k = max(1, cfg.top_k)
        values, indices = torch.topk(logits, min(top_k, logits.numel()))
        masked = torch.full_like(logits, float("-inf"))
        masked[indices] = values
        logits = masked
    elif strategy == "nucleus":
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > cfg.top_p
        if cutoff.any():
            first_cut = int(cutoff.nonzero(as_tuple=False)[0].item())
            sorted_logits[first_cut + 1 :] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits[sorted_indices] = sorted_logits

    probs = torch.softmax(logits, dim=-1)
    next_token = int(torch.multinomial(probs, num_samples=1).item())
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
    return next_token, float(probs[next_token].item()), entropy


def _check_stop(text: str, cfg: GenerationConfig) -> tuple[bool, str, str]:
    for stop in cfg.stop_strings:
        if stop and stop in text:
            return True, text.split(stop, 1)[0], "stop_string"
    return False, text, ""


def detect_repetition(text: str) -> dict[str, object]:
    tokens = text.split()
    if len(tokens) < 8:
        return {"repeated_ngrams_detected": False, "ngram": "", "count": 0}
    seen: Dict[str, int] = {}
    for n in (3, 4):
        for idx in range(0, len(tokens) - n + 1):
            gram = " ".join(tokens[idx : idx + n])
            seen[gram] = seen.get(gram, 0) + 1
    repeated = max(seen.items(), key=lambda item: item[1], default=("", 0))
    return {
        "repeated_ngrams_detected": repeated[1] >= 3,
        "ngram": repeated[0],
        "count": repeated[1],
    }


@torch.no_grad()
def generate_traced(prompt: str, cfg: GenerationConfig, *, session_id: str | None = None) -> GenerationTrace:
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    MODEL, TOKENIZER, LOADED_CHECKPOINT = _get_runtime()
    del LOADED_CHECKPOINT
    _seed_all(cfg.seed)

    prompt_ids = TOKENIZER.encode(prompt, add_special_tokens=False)
    ids = [TOKENIZER.bos_token_id] + prompt_ids
    generated_ids = ids[:]
    prompt_token_count = len(ids)
    entropy_curve: list[float] = []
    max_prob_curve: list[float] = []
    stopped_by = "max_tokens"
    ghost_loaded = bool(session_id and session_id in _GHOST_STORE)

    start = time.perf_counter()
    for _ in range(max(0, cfg.max_tokens)):
        x = torch.tensor([generated_ids[-MODEL.block_size :]], dtype=torch.long, device=DEVICE)
        logits, _ = MODEL(x)
        next_id, max_prob, entropy = _sample_next_token(logits[0, -1, :], cfg, generated_ids)
        generated_ids.append(next_id)
        entropy_curve.append(entropy)
        max_prob_curve.append(max_prob)

        if next_id == TOKENIZER.eos_token_id:
            stopped_by = "eos"
            break

        current_text = TOKENIZER.decode(generated_ids)
        hit, trimmed, reason = _check_stop(current_text, cfg)
        if hit:
            generated_ids = TOKENIZER.encode(trimmed, add_special_tokens=False)
            stopped_by = reason
            break

    answer_ids = generated_ids[prompt_token_count:]
    output_text = TOKENIZER.decode(answer_ids).strip()

    if _IDENTITY_INJECTOR is not None:
        try:
            output_text = _IDENTITY_INJECTOR.clean(output_text)
        except Exception as exc:
            logger.warning("Identity injector cleanup failed: %s", exc)

    if session_id:
        _ACTIVE_GHOST["session_id"] = session_id
        _ACTIVE_GHOST["last_output"] = output_text

    repetition = detect_repetition(output_text)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return GenerationTrace(
        output=output_text,
        strategy=cfg.strategy,
        tokens_generated=len(entropy_curve),
        time_ms=elapsed_ms,
        entropy_curve=entropy_curve,
        max_prob_curve=max_prob_curve,
        stopped_by=stopped_by,
        repeated_ngrams_detected=bool(repetition["repeated_ngrams_detected"]),
        kv_cache_compressed=False,
        memory_saved_mb=0.0,
        ghost_state_loaded=ghost_loaded,
    )


def generate(prompt: str, strategy: str = "nucleus", max_tokens: int = 200, **kwargs) -> str:
    cfg = GenerationConfig(strategy=strategy, max_tokens=max_tokens)
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        elif key == "max_new_tokens":
            cfg.max_tokens = int(value)
    return generate_traced(prompt, cfg, session_id=kwargs.get("session_id")).output


def generate_stream(prompt: str, cfg: GenerationConfig) -> Iterator[str]:
    MODEL, TOKENIZER, _ = _get_runtime()
    _seed_all(cfg.seed)
    prompt_ids = TOKENIZER.encode(prompt, add_special_tokens=False)
    generated_ids = [TOKENIZER.bos_token_id] + prompt_ids
    prompt_token_count = len(generated_ids)
    for _ in range(max(0, cfg.max_tokens)):
        x = torch.tensor(
            [generated_ids[-MODEL.block_size:]], dtype=torch.long, device=DEVICE
        )
        with torch.no_grad():
            logits, _ = MODEL(x)
        next_id, _, _ = _sample_next_token(logits[0, -1, :], cfg, generated_ids)
        generated_ids.append(next_id)
        if next_id == TOKENIZER.eos_token_id:
            break
        token_text = TOKENIZER.decode([next_id])
        if token_text:
            yield token_text


def load_ghost_state(session_id: str) -> None:
    _ACTIVE_GHOST.clear()
    if _GHOST_MEMORY is not None:
        try:
            stored = _GHOST_MEMORY.retrieve(session_id) or {}
            _ACTIVE_GHOST.update(stored)
        except Exception:
            _ACTIVE_GHOST.update(_GHOST_STORE.get(session_id, {}))
    else:
        _ACTIVE_GHOST.update(_GHOST_STORE.get(session_id, {}))
    _ACTIVE_GHOST["session_id"] = session_id


def save_ghost_state(session_id: str) -> None:
    _GHOST_STORE[session_id] = dict(_ACTIVE_GHOST)
    if _GHOST_MEMORY is not None:
        try:
            _GHOST_MEMORY.store(session_id, dict(_ACTIVE_GHOST))
        except Exception as exc:
            logger.warning("Ghost state persistence failed for session %s: %s", session_id, exc)


def get_model_info() -> dict[str, object]:
    MODEL, TOKENIZER, LOADED_CHECKPOINT = _get_runtime()
    summary = model_summary(MODEL)
    return {
        "model_line": "v2",
        "checkpoint": str(LOADED_CHECKPOINT),
        "vocab_size": TOKENIZER.vocab_size,
        "param_count": summary["parameters"],
        "trainable_parameters": summary["trainable_parameters"],
        "d_model": getattr(MODEL, "d_model", None),
        "device": str(DEVICE),
        "block_size": MODEL.block_size,
        "tokenizer_backend": getattr(TOKENIZER, "backend", "unknown"),
    }


if __name__ == "__main__":
    prompt = "H: Who are you?\nANRA:"
    trace = generate_traced(prompt, GenerationConfig(max_tokens=60))
    print(trace.output)
