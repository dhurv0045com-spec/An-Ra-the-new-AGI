from __future__ import annotations

import math
import pickle
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from anra_brain import CausalTransformer

# ======================================================================================
# Configuration
# ======================================================================================
CONFIG = {
    "identity_checkpoint": "/content/drive/MyDrive/AnRa/anra_brain_identity.pt",
    "base_checkpoint": "/content/drive/MyDrive/AnRa/anra_brain.pt",
    "local_identity_checkpoint": "anra_brain_identity.pt",
    "local_base_checkpoint": "anra_brain.pt",
    "tokenizer_path": "tokenizer.pkl",
    "n_embd": 256,
    "n_head": 4,
    "n_layer": 4,
    "block_size": 128,
    "default_max_new_tokens": 180,
    "default_temperature": 0.8,
    "default_top_k": 40,
    "default_top_p": 0.92,
    "default_beams": 4,
    "default_penalty": 1.15,
    "default_repetition_window": 128,
    "default_seed": None,
    "contrastive_alpha": 0.6,
    "contrastive_k": 8,
    "diagnostic_window": 64,
}


@dataclass
class GenerationConfig:
    strategy: str = "nucleus"
    max_new_tokens: int = CONFIG["default_max_new_tokens"]
    temperature: float = CONFIG["default_temperature"]
    top_k: int = CONFIG["default_top_k"]
    top_p: float = CONFIG["default_top_p"]
    beams: int = CONFIG["default_beams"]
    alpha: float = CONFIG["contrastive_alpha"]
    contrastive_k: int = CONFIG["contrastive_k"]
    repetition_penalty: float = CONFIG["default_penalty"]
    repetition_window: int = CONFIG["default_repetition_window"]
    seed: Optional[int] = CONFIG["default_seed"]
    stop_strings: Sequence[str] = field(default_factory=list)


@dataclass
class GenerationTrace:
    strategy: str
    prompt: str
    generated_ids: List[int]
    generated_text: str
    elapsed_ms: float
    entropy_curve: List[float] = field(default_factory=list)
    max_prob_curve: List[float] = field(default_factory=list)

    def summary(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "prompt_chars": len(self.prompt),
            "output_chars": len(self.generated_text),
            "elapsed_ms": self.elapsed_ms,
            "entropy_mean": statistics.fmean(self.entropy_curve) if self.entropy_curve else 0.0,
            "max_prob_mean": statistics.fmean(self.max_prob_curve) if self.max_prob_curve else 0.0,
        }


# ======================================================================================
# Initialization
# ======================================================================================

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = _resolve_device()
MODULE_DIR = Path(__file__).resolve().parent


def _checkpoint_candidates() -> List[Tuple[str, Path]]:
    return [
        ("identity", Path(CONFIG["identity_checkpoint"])),
        ("identity", MODULE_DIR / CONFIG["local_identity_checkpoint"]),
        ("base", Path(CONFIG["base_checkpoint"])),
        ("base", MODULE_DIR / CONFIG["local_base_checkpoint"]),
    ]


def _load_tokenizer(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _seed_all(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_model_and_tokenizer():
    tokenizer_path = MODULE_DIR / CONFIG["tokenizer_path"]
    tokenizer = _load_tokenizer(tokenizer_path)
    vocab_size = int(getattr(tokenizer, "vocab_size"))

    model = CausalTransformer(
        vocab_size=vocab_size,
        n_embd=CONFIG["n_embd"],
        n_head=CONFIG["n_head"],
        n_layer=CONFIG["n_layer"],
        block_size=CONFIG["block_size"],
    )

    loaded_checkpoint = "none"
    for ckpt_type, ckpt_path in _checkpoint_candidates():
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=DEVICE)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=False)
            loaded_checkpoint = str(ckpt_path)
            print(f"[generate.py] Loaded {ckpt_type} checkpoint: {ckpt_path}")
            break

    if loaded_checkpoint == "none":
        print("[generate.py] WARNING: no checkpoint found, using random weights")

    model.to(DEVICE).eval()
    return model, tokenizer, loaded_checkpoint


MODEL, TOKENIZER, LOADED_CHECKPOINT = _init_model_and_tokenizer()
VOCAB_SIZE = int(getattr(TOKENIZER, "vocab_size"))
PARAM_COUNT = int(sum(p.numel() for p in MODEL.parameters()))


# ======================================================================================
# Core helpers
# ======================================================================================

def _encode(text: str) -> List[int]:
    return TOKENIZER.encode(text)


def _decode(ids: List[int]) -> str:
    return TOKENIZER.decode(ids)


def _apply_repetition_penalty(logits: torch.Tensor, recent: Sequence[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or not recent:
        return logits
    out = logits.clone()
    for token_id in set(recent):
        out[token_id] = out[token_id] / penalty
    return out


def _entropy_and_maxprob(logits: torch.Tensor) -> Tuple[float, float]:
    probs = F.softmax(logits, dim=-1)
    probs = torch.clamp(probs, 1e-12, 1.0)
    ent = float((-probs * probs.log()).sum().item())
    mx = float(probs.max().item())
    return ent, mx


def _forward_logits(ids: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.tensor([list(ids)[-CONFIG["block_size"] :]], dtype=torch.long, device=DEVICE)
    logits, _ = MODEL(idx)
    return logits[0, -1, :], logits[0]


def _apply_topk(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]))
    masked = torch.full_like(logits, float("-inf"))
    masked[indices] = values
    return masked


def _apply_topp(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative > top_p
    cutoff[1:] = cutoff[:-1].clone()
    cutoff[0] = False
    sorted_logits[cutoff] = float("-inf")
    restored = torch.full_like(logits, float("-inf"))
    restored[sorted_idx] = sorted_logits
    return restored


def _sample_from_logits(logits: torch.Tensor, cfg: GenerationConfig) -> int:
    strategy = cfg.strategy
    adjusted = logits
    if strategy == "greedy":
        return int(torch.argmax(adjusted).item())
    if strategy == "temperature":
        adjusted = adjusted / max(cfg.temperature, 1e-6)
    elif strategy == "topk":
        adjusted = _apply_topk(adjusted, cfg.top_k)
    elif strategy == "nucleus":
        adjusted = _apply_topp(adjusted, cfg.top_p)

    probs = F.softmax(adjusted, dim=-1)
    if torch.isnan(probs).any() or probs.sum().item() <= 0:
        return int(torch.argmax(adjusted).item())
    return int(torch.multinomial(probs, num_samples=1).item())


# ======================================================================================
# Decoders
# ======================================================================================

def _decode_autoregressive(prompt_ids: List[int], cfg: GenerationConfig, trace: GenerationTrace) -> List[int]:
    generated = list(prompt_ids)
    for _ in range(cfg.max_new_tokens):
        logits, _ = _forward_logits(generated)
        logits = _apply_repetition_penalty(logits, generated[-cfg.repetition_window :], cfg.repetition_penalty)
        ent, mx = _entropy_and_maxprob(logits)
        trace.entropy_curve.append(ent)
        trace.max_prob_curve.append(mx)
        next_id = _sample_from_logits(logits, cfg)
        generated.append(next_id)
        if cfg.stop_strings:
            text = _decode(generated[len(prompt_ids) :])
            if any(s and s in text for s in cfg.stop_strings):
                break
    return generated


def _beam_search(prompt_ids: List[int], cfg: GenerationConfig, trace: GenerationTrace) -> List[int]:
    beam = [(0.0, list(prompt_ids))]
    for _ in range(cfg.max_new_tokens):
        candidates = []
        for score, seq in beam:
            logits, _ = _forward_logits(seq)
            ent, mx = _entropy_and_maxprob(logits)
            trace.entropy_curve.append(ent)
            trace.max_prob_curve.append(mx)
            log_probs = F.log_softmax(logits, dim=-1)
            vals, idx = torch.topk(log_probs, k=min(cfg.beams, VOCAB_SIZE))
            for v, i in zip(vals.tolist(), idx.tolist()):
                candidates.append((score + v, seq + [i]))
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[: cfg.beams]
    return beam[0][1]


def _contrastive_search(prompt_ids: List[int], cfg: GenerationConfig, trace: GenerationTrace) -> List[int]:
    generated = list(prompt_ids)
    recent = generated[-32:]
    for _ in range(cfg.max_new_tokens):
        logits, full_logits = _forward_logits(generated)
        logits = _apply_repetition_penalty(logits, generated[-cfg.repetition_window :], cfg.repetition_penalty)
        ent, mx = _entropy_and_maxprob(logits)
        trace.entropy_curve.append(ent)
        trace.max_prob_curve.append(mx)

        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, k=min(cfg.contrastive_k, probs.shape[-1]))
        prev_states = full_logits[-min(len(recent), full_logits.shape[0]) :]

        best_score = -1e9
        best_tok = int(top_idx[0].item())
        for cand_prob, cand_tok in zip(top_probs, top_idx):
            candidate_state = full_logits[-1].clone()
            if prev_states.numel() > 0:
                sim = F.cosine_similarity(candidate_state.unsqueeze(0), prev_states, dim=-1).max().item()
            else:
                sim = 0.0
            score = cfg.alpha * math.log(max(cand_prob.item(), 1e-9)) - (1.0 - cfg.alpha) * sim
            if score > best_score:
                best_score = score
                best_tok = int(cand_tok.item())

        generated.append(best_tok)
        recent.append(best_tok)
        recent = recent[-32:]
        if cfg.stop_strings:
            text = _decode(generated[len(prompt_ids) :])
            if any(s and s in text for s in cfg.stop_strings):
                break
    return generated


# ======================================================================================
# Public API
# ======================================================================================

def _build_config(strategy: str, kwargs: Dict[str, object]) -> GenerationConfig:
    cfg = GenerationConfig(strategy=strategy.lower())
    if "max_new_tokens" in kwargs:
        cfg.max_new_tokens = int(kwargs.pop("max_new_tokens"))
    if "temperature" in kwargs:
        cfg.temperature = float(kwargs.pop("temperature"))
    if "k" in kwargs:
        cfg.top_k = int(kwargs.pop("k"))
    if "p" in kwargs:
        cfg.top_p = float(kwargs.pop("p"))
    if "beams" in kwargs:
        cfg.beams = int(kwargs.pop("beams"))
    if "alpha" in kwargs:
        cfg.alpha = float(kwargs.pop("alpha"))
    if "repetition_penalty" in kwargs:
        cfg.repetition_penalty = float(kwargs.pop("repetition_penalty"))
    if "repetition_window" in kwargs:
        cfg.repetition_window = int(kwargs.pop("repetition_window"))
    if "seed" in kwargs:
        cfg.seed = int(kwargs.pop("seed")) if kwargs["seed"] is not None else None
    if "stop_strings" in kwargs:
        cfg.stop_strings = list(kwargs.pop("stop_strings"))
    if "contrastive_k" in kwargs:
        cfg.contrastive_k = int(kwargs.pop("contrastive_k"))
    return cfg


def generate_with_trace(prompt: str, strategy: str = "nucleus", **kwargs) -> GenerationTrace:
    if not prompt:
        return GenerationTrace(strategy=strategy, prompt=prompt, generated_ids=[], generated_text="", elapsed_ms=0.0)

    run_kwargs = dict(kwargs)
    cfg = _build_config(strategy, run_kwargs)
    _seed_all(cfg.seed)

    prompt_ids = _encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]

    trace = GenerationTrace(strategy=cfg.strategy, prompt=prompt, generated_ids=[], generated_text="", elapsed_ms=0.0)
    t0 = time.perf_counter()

    if cfg.strategy in {"greedy", "temperature", "topk", "nucleus"}:
        out_ids = _decode_autoregressive(prompt_ids, cfg, trace)
    elif cfg.strategy == "beam":
        out_ids = _beam_search(prompt_ids, cfg, trace)
    elif cfg.strategy == "contrastive":
        out_ids = _contrastive_search(prompt_ids, cfg, trace)
    else:
        raise ValueError(f"Unknown strategy '{cfg.strategy}'")

    generated = out_ids[len(prompt_ids) :]
    text = _decode(generated).strip()

    trace.generated_ids = generated
    trace.generated_text = text
    trace.elapsed_ms = (time.perf_counter() - t0) * 1000
    return trace


def generate(prompt: str, strategy: str = "nucleus", **kwargs) -> str:
    trace = generate_with_trace(prompt, strategy=strategy, **kwargs)
    return trace.generated_text


def get_model_info() -> Dict[str, object]:
    return {
        "checkpoint": LOADED_CHECKPOINT,
        "device": str(DEVICE),
        "vocab_size": VOCAB_SIZE,
        "param_count": PARAM_COUNT,
        "block_size": CONFIG["block_size"],
        "default_strategy": "nucleus",
    }


def detect_repetition(text: str, n: int = 3) -> Dict[str, object]:
    if len(text) < n:
        return {"has_repetition": False, "count": 0, "n": n}
    seen = set()
    repeats = 0
    for i in range(len(text) - n + 1):
        g = text[i : i + n]
        if g in seen:
            repeats += 1
        seen.add(g)
    return {"has_repetition": repeats > 0, "count": repeats, "n": n}


if __name__ == "__main__":
    prompt = "H: Tell me who you are.\nANRA:"
    strategies = ["greedy", "temperature", "topk", "nucleus", "beam", "contrastive"]
    print("\n=== An-Ra Strategy Comparison ===")
    print("Model info:", get_model_info())
    for strat in strategies:
        trace = generate_with_trace(prompt, strategy=strat, max_new_tokens=120)
        rep = detect_repetition(trace.generated_text, n=3)
        print(f"\n[{strat}] {trace.elapsed_ms:.1f}ms | repeat={rep['has_repetition']} ({rep['count']})")
        print(trace.generated_text)
        print("diagnostics:", trace.summary())
