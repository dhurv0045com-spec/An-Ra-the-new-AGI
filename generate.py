"""
generate.py — An-Ra Inference Engine
=====================================
Production-grade decoding module for the An-Ra sovereign AI system.
Supports: greedy, temperature, top-k, top-p (nucleus), repetition penalty,
          beam search, and contrastive search (bonus strategy).

Author: An-Ra Build Pipeline
"""

import sys
import math
import pickle
import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────
#  CONFIG — edit these paths, never hardcode below
# ─────────────────────────────────────────────
CONFIG = {
    "model_path":     "anra_brain.pt",
    "tokenizer_path": "tokenizer.pkl",
    "device":         "auto",          # "auto" | "cuda" | "cpu" | "mps"
    "default_max_new_tokens": 200,
    "default_strategy":       "top_p",
}

# ─────────────────────────────────────────────
#  DEVICE RESOLUTION
# ─────────────────────────────────────────────

def resolve_device(preference: str = "auto") -> torch.device:
    """Pick the best available device."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = resolve_device(CONFIG["device"])


# ─────────────────────────────────────────────
#  MODEL + TOKENIZER LOADING
# ─────────────────────────────────────────────

def load_tokenizer(path: str):
    """Load a pickled CharTokenizer (or any tokenizer with encode/decode)."""
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def load_model(path: str, device: torch.device):
    """
    Load An-Ra's CausalTransformer checkpoint.
    Handles both full-model saves and state-dict-only saves gracefully.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Case 1: checkpoint is already the model object
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint

    # Case 2: dict with a 'model' or 'model_state_dict' key
    elif isinstance(checkpoint, dict):
        if "model" in checkpoint:
            model = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            # We need the class to be importable. Try a local import first.
            raise RuntimeError(
                "Checkpoint contains only a state_dict. "
                "Import your CausalTransformer class and call "
                "model.load_state_dict(checkpoint['model_state_dict']) manually, "
                "then pass the model object into the generate() function directly."
            )
        else:
            # Assume the dict itself is a state dict — surface a clear error
            raise RuntimeError(
                "Unrecognised checkpoint format. Keys found: "
                + str(list(checkpoint.keys())[:10])
            )
    else:
        raise RuntimeError(f"Cannot interpret checkpoint of type {type(checkpoint)}")

    model = model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────
#  LOGIT PROCESSORS  (pure functions, composable)
# ─────────────────────────────────────────────

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Divide logits by temperature before softmax.
    temperature → 0  : approaches greedy (peaky distribution)
    temperature = 1  : unmodified model distribution
    temperature > 1  : flatter, more uniform / creative sampling
    Clamps to a safe minimum to prevent division by zero.
    """
    temperature = max(temperature, 1e-8)
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Zero out all logits except the top-k.
    Forces the model to only ever pick among k candidates per step.
    """
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    top_k_values, _ = torch.topk(logits, k)
    threshold = top_k_values[..., -1, None]  # k-th largest value
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling (Holtzman et al., 2020).
    Keep only the smallest set of tokens whose cumulative probability ≥ p.
    More adaptive than top-k: on high-entropy steps it allows more candidates;
    on low-entropy steps it aggressively narrows.
    """
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens whose cumulative prob exceeds p (shift right so we keep the token
    # that pushes us over the threshold)
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    # Scatter back to original ordering
    logits = torch.scatter(logits, 0, sorted_indices, sorted_logits)
    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: List[int],
    penalty: float,
) -> torch.Tensor:
    """
    Penalise tokens that have already appeared in the generated sequence.
    For positive logits  → divide by penalty  (make less likely)
    For negative logits → multiply by penalty (push further negative)
    penalty = 1.0 means no change. Typical range: 1.1 – 1.5.
    This follows the formulation from Keskar et al. (CTRL, 2019).
    """
    if penalty == 1.0 or not generated_ids:
        return logits
    unique_ids = list(set(generated_ids))
    score = logits[unique_ids]
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits[unique_ids] = score
    return logits


def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """
    Min-P sampling (Nguyen et al., 2024) — bonus strategy.
    Removes tokens whose probability is below min_p * P(top_token).
    Scales the threshold relative to the model's confidence at each step,
    making it robust across varying entropy without needing to tune top-k or top-p.
    Outperforms nucleus sampling on instruction-following and coherence benchmarks.
    """
    if min_p <= 0.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max()
    threshold = min_p * max_prob
    return logits.masked_fill(probs < threshold, float("-inf"))


# ─────────────────────────────────────────────
#  SAMPLING KERNEL
# ─────────────────────────────────────────────

def sample_from_logits(logits: torch.Tensor) -> int:
    """Multinomial sample from a (potentially masked) logit vector."""
    probs = F.softmax(logits, dim=-1)
    # Guard: if all logits were -inf (bad config), fall back to argmax
    if not torch.isfinite(probs).any() or probs.sum() < 1e-9:
        return int(logits.argmax())
    return int(torch.multinomial(probs, num_samples=1))


# ─────────────────────────────────────────────
#  AUTOREGRESSIVE FORWARD PASS HELPER
# ─────────────────────────────────────────────

@torch.inference_mode()
def _next_logits(model: torch.nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    """
    Run one forward pass and return the logit vector for the next token.
    Handles models that return (logits,) tuples, raw tensors, or dicts.
    token_ids: shape (1, seq_len)
    Returns: shape (vocab_size,)
    """
    output = model(token_ids)

    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, (tuple, list)):
        logits = output[0]
    elif isinstance(output, dict):
        logits = output.get("logits", output.get("last_hidden_state"))
    else:
        raise RuntimeError(f"Unrecognised model output type: {type(output)}")

    # logits shape: (batch, seq_len, vocab) → take last position
    if logits.dim() == 3:
        logits = logits[0, -1, :]
    elif logits.dim() == 2:
        logits = logits[-1, :]

    return logits.float()  # always fp32 for numerical stability


# ─────────────────────────────────────────────
#  DECODING STRATEGIES
# ─────────────────────────────────────────────

@torch.inference_mode()
def _decode_greedy(
    model, token_ids: torch.Tensor, max_new_tokens: int, eos_id: Optional[int]
) -> List[int]:
    """Argmax at every step. Fast, deterministic, and (for small models) repetitive."""
    generated = []
    ids = token_ids.clone()

    for _ in range(max_new_tokens):
        logits = _next_logits(model, ids)
        next_id = int(logits.argmax())
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)], dim=1)
        if eos_id is not None and next_id == eos_id:
            break

    return generated


@torch.inference_mode()
def _decode_temperature(
    model,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_id: Optional[int],
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    min_p: float = 0.0,
) -> List[int]:
    """
    General-purpose sampler. Applies processors in the recommended order:
    repetition_penalty → temperature → top_k → top_p → min_p → sample.
    Set temperature=1.0, top_k=0, top_p=1.0 for pure temperature sampling.
    All filters compose cleanly.
    """
    generated: List[int] = []
    ids = token_ids.clone()

    for _ in range(max_new_tokens):
        logits = _next_logits(model, ids)

        # 1. Repetition penalty on the raw logits
        all_ids = token_ids[0].tolist() + generated
        logits = apply_repetition_penalty(logits, all_ids, repetition_penalty)

        # 2. Temperature
        logits = apply_temperature(logits, temperature)

        # 3. Top-k
        if top_k > 0:
            logits = apply_top_k(logits, top_k)

        # 4. Nucleus (top-p)
        if top_p < 1.0:
            logits = apply_top_p(logits, top_p)

        # 5. Min-p (bonus)
        if min_p > 0.0:
            logits = apply_min_p(logits, min_p)

        next_id = sample_from_logits(logits)
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)], dim=1)

        if eos_id is not None and next_id == eos_id:
            break

    return generated


@dataclass
class _BeamHypothesis:
    """A single hypothesis tracked during beam search."""
    score: float                     # cumulative log-prob (normalised)
    ids: List[int]                   # generated token ids so far
    done: bool = False

    # Make sortable for heapq (max-heap via negation)
    def __lt__(self, other):
        return self.score > other.score


@torch.inference_mode()
def _decode_beam_search(
    model,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_id: Optional[int],
    num_beams: int = 4,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
) -> List[int]:
    """
    Standard beam search with length normalisation and optional repetition penalty.
    Maintains `num_beams` candidate sequences in parallel and returns the one
    with the highest length-normalised log-probability.

    length_penalty > 1.0  → favours longer sequences
    length_penalty < 1.0  → favours shorter sequences
    length_penalty = 1.0  → raw log-prob sum
    """
    vocab_size = model(token_ids)[0].shape[-1] if hasattr(model(token_ids), '__len__') else None

    # Initialise: one beam with an empty generated sequence
    beams: List[_BeamHypothesis] = [_BeamHypothesis(score=0.0, ids=[])]
    completed: List[_BeamHypothesis] = []

    prefix = token_ids[0].tolist()

    for step in range(max_new_tokens):
        if not beams:
            break

        all_candidates: List[_BeamHypothesis] = []

        for beam in beams:
            if beam.done:
                all_candidates.append(beam)
                continue

            # Build input sequence for this beam
            full_ids = prefix + beam.ids
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=DEVICE)
            logits = _next_logits(model, input_tensor)

            # Apply repetition penalty per-beam
            logits = apply_repetition_penalty(logits, full_ids, repetition_penalty)

            log_probs = F.log_softmax(logits, dim=-1)

            # Expand: keep top num_beams token extensions per existing beam
            topk_vals, topk_ids = torch.topk(log_probs, num_beams)

            for log_p, tok_id in zip(topk_vals.tolist(), topk_ids.tolist()):
                new_ids = beam.ids + [tok_id]
                # Length-normalised score
                norm_len = ((5 + len(new_ids)) / 6) ** length_penalty
                new_score = (beam.score * (len(beam.ids) or 1) + log_p) / norm_len
                candidate = _BeamHypothesis(score=new_score, ids=new_ids)

                if eos_id is not None and tok_id == eos_id:
                    candidate.done = True
                    completed.append(candidate)
                else:
                    all_candidates.append(candidate)

        # Keep top num_beams candidates (excluding completed ones)
        active = [c for c in all_candidates if not c.done]
        active.sort()  # uses __lt__ → highest score first
        beams = active[:num_beams]

        # Early exit if all beams have completed
        if not beams:
            break

    # Select winner from completed + remaining beams
    pool = completed + beams
    if not pool:
        return []
    pool.sort()
    return pool[0].ids


@torch.inference_mode()
def _decode_contrastive(
    model,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_id: Optional[int],
    k: int = 5,
    alpha: float = 0.6,
) -> List[int]:
    """
    Contrastive Search (Su et al., 2022 — NEURIPS).
    BONUS STRATEGY — not in the original spec but recommended.

    At each step:
      1. Select the top-k candidates by model probability.
      2. Score each candidate by: (1-α) * model_prob
                                   - α * max_cosine_sim(candidate, all_prev_hidden_states)
      3. Choose the candidate that maximises this objective.

    The cosine-similarity penalty discourages repetition at the representation level
    (not just the token level), producing significantly more coherent and diverse text
    than any pure sampling strategy — especially for small models like An-Ra that
    tend to collapse into loops.

    alpha: balance between model confidence (0) and anti-repetition (1).
           Recommended: 0.5–0.7 for creative tasks, 0.3–0.5 for factual tasks.
    k:     candidate pool size. Recommended: 4–8.
    """
    generated: List[int] = []
    ids = token_ids.clone()

    # We need access to hidden states, so we probe the model's output structure
    # to find a hidden-state tensor we can use for similarity. If the model only
    # returns logits we fall back to token-level deduplication.
    hidden_states_history: List[torch.Tensor] = []   # (vocab_dim,) per step

    for step in range(max_new_tokens):
        # ── Forward pass ──────────────────────────────────────────────────
        output = model(ids)

        if isinstance(output, (tuple, list)):
            raw_logits = output[0]
            # If model returns hidden states as second element, use them
            hidden_out = output[1] if len(output) > 1 else None
        elif isinstance(output, torch.Tensor):
            raw_logits = output
            hidden_out = None
        else:
            raw_logits = output.get("logits", output.get("last_hidden_state"))
            hidden_out = output.get("hidden_states", None)

        if raw_logits.dim() == 3:
            logits = raw_logits[0, -1, :].float()
        else:
            logits = raw_logits[-1, :].float()

        # ── Get top-k candidate token ids ─────────────────────────────────
        top_k_vals, top_k_ids = torch.topk(F.softmax(logits, dim=-1), k)
        model_probs = top_k_vals  # shape (k,)

        # ── Score candidates ──────────────────────────────────────────────
        if hidden_states_history and hidden_out is None:
            # No hidden states available — use token-level embedding proxy
            # (embed the candidate id using the logit row as a proxy vector)
            best_score = -float("inf")
            best_id = int(top_k_ids[0])

            # Build history matrix from logit rows (vocab_size proxy embeddings)
            # This is an approximation; works reasonably without hidden state access
            for rank, (prob, tok_id) in enumerate(
                zip(model_probs.tolist(), top_k_ids.tolist())
            ):
                tok_id = int(tok_id)
                # Penalise tokens that match recent generated tokens
                recency_penalty = sum(
                    1.0 / (i + 1)
                    for i, prev_id in enumerate(reversed(generated[-k:]))
                    if prev_id == tok_id
                )
                score = (1 - alpha) * prob - alpha * min(recency_penalty, 1.0)
                if score > best_score:
                    best_score = score
                    best_id = tok_id

        else:
            # Use real hidden-state cosine similarity if available
            best_score = -float("inf")
            best_id = int(top_k_ids[0])

            for prob, tok_id in zip(model_probs.tolist(), top_k_ids.tolist()):
                tok_id = int(tok_id)
                if hidden_states_history:
                    # Forward candidate token to get its hidden state
                    cand_input = torch.cat(
                        [ids, torch.tensor([[tok_id]], device=DEVICE)], dim=1
                    )
                    cand_out = model(cand_input)
                    if isinstance(cand_out, (tuple, list)):
                        cand_h = cand_out[0][0, -1, :].float()
                    elif isinstance(cand_out, torch.Tensor):
                        cand_h = cand_out[0, -1, :].float()
                    else:
                        cand_h = cand_out["logits"][0, -1, :].float()

                    cand_h_norm = F.normalize(cand_h.unsqueeze(0), dim=-1)
                    hist_matrix = torch.stack(hidden_states_history, dim=0)  # (t, d)
                    hist_norm = F.normalize(hist_matrix, dim=-1)
                    sims = (hist_norm @ cand_h_norm.T).squeeze()  # (t,)
                    max_sim = float(sims.max()) if sims.numel() > 0 else 0.0

                    score = (1 - alpha) * prob - alpha * max_sim
                else:
                    score = prob

                if score > best_score:
                    best_score = score
                    best_id = tok_id

        next_id = best_id
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)

        if eos_id is not None and next_id == eos_id:
            break

    return generated


# ─────────────────────────────────────────────
#  PUBLIC INTERFACE
# ─────────────────────────────────────────────

def generate(
    prompt: str,
    strategy: str = CONFIG["default_strategy"],
    *,
    model=None,
    tokenizer=None,
    max_new_tokens: int = CONFIG["default_max_new_tokens"],
    # Temperature / sampling kwargs
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    min_p: float = 0.0,
    repetition_penalty: float = 1.15,
    # Beam search kwargs
    num_beams: int = 4,
    length_penalty: float = 1.0,
    # Contrastive search kwargs
    contrastive_k: int = 5,
    contrastive_alpha: float = 0.6,
    # Optional EOS token
    eos_token: Optional[str] = None,
) -> str:
    """
    Generate text from An-Ra given a prompt string.

    Parameters
    ----------
    prompt          : The input string to condition generation on.
    strategy        : One of 'greedy' | 'temperature' | 'top_k' | 'top_p' |
                      'beam' | 'contrastive'.
                      'temperature', 'top_k', and 'top_p' all share the same
                      general sampler — they differ only in which filters are active.
    model           : Pre-loaded model. If None, loads from CONFIG["model_path"].
    tokenizer       : Pre-loaded tokenizer. If None, loads from CONFIG["tokenizer_path"].
    max_new_tokens  : Maximum tokens to generate beyond the prompt.
    temperature     : Sampling temperature. Ignored for greedy and beam.
    top_k           : Keep only top-k tokens. 0 = disabled.
    top_p           : Nucleus probability threshold. 1.0 = disabled.
    min_p           : Min-P threshold (fraction of top token prob). 0.0 = disabled.
    repetition_penalty : Penalty for repeated tokens. 1.0 = none.
    num_beams       : Beam width for beam search.
    length_penalty  : Beam search length normalisation exponent.
    contrastive_k   : Candidate pool size for contrastive search.
    contrastive_alpha: Anti-repetition weight for contrastive search.
    eos_token       : Optional single character / string that terminates generation.

    Returns
    -------
    str : The generated text (prompt not included).
    """
    # ── Load resources if not provided ───────────────────────────────────
    if tokenizer is None:
        tokenizer = load_tokenizer(CONFIG["tokenizer_path"])
    if model is None:
        model = load_model(CONFIG["model_path"], DEVICE)

    # ── Encode prompt ─────────────────────────────────────────────────────
    input_ids: List[int] = tokenizer.encode(prompt)
    if not input_ids:
        raise ValueError("Prompt encodes to an empty token sequence.")
    token_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    # ── EOS token id ─────────────────────────────────────────────────────
    eos_id: Optional[int] = None
    if eos_token is not None:
        eos_encoded = tokenizer.encode(eos_token)
        eos_id = eos_encoded[0] if eos_encoded else None

    # ── Dispatch to strategy ──────────────────────────────────────────────
    strategy = strategy.lower().strip()

    if strategy == "greedy":
        generated_ids = _decode_greedy(model, token_ids, max_new_tokens, eos_id)

    elif strategy == "temperature":
        generated_ids = _decode_temperature(
            model, token_ids, max_new_tokens, eos_id,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
        )

    elif strategy == "top_k":
        generated_ids = _decode_temperature(
            model, token_ids, max_new_tokens, eos_id,
            temperature=temperature,
            top_k=top_k,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
        )

    elif strategy in ("top_p", "nucleus"):
        generated_ids = _decode_temperature(
            model, token_ids, max_new_tokens, eos_id,
            temperature=temperature,
            top_k=0,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
        )

    elif strategy == "combined":
        # Top-k + Top-p + temperature + repetition penalty all active at once.
        # This is a common production configuration (used by GPT-2 and LLaMA demos).
        generated_ids = _decode_temperature(
            model, token_ids, max_new_tokens, eos_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
        )

    elif strategy == "beam":
        generated_ids = _decode_beam_search(
            model, token_ids, max_new_tokens, eos_id,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )

    elif strategy == "contrastive":
        generated_ids = _decode_contrastive(
            model, token_ids, max_new_tokens, eos_id,
            k=contrastive_k,
            alpha=contrastive_alpha,
        )

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Choose from: greedy | temperature | top_k | top_p | combined | beam | contrastive"
        )

    # ── Decode to string ──────────────────────────────────────────────────
    if not generated_ids:
        return ""
    return tokenizer.decode(generated_ids)


# ─────────────────────────────────────────────
#  STANDALONE TEST HARNESS
# ─────────────────────────────────────────────

def _run_demo():
    """
    Test all decoding strategies and print side-by-side results.
    Run with:  python generate.py
    """
    print("=" * 65)
    print("  An-Ra Inference Engine — Strategy Benchmark")
    print(f"  Device : {DEVICE}")
    print("=" * 65)

    # Load once, reuse
    print(f"\nLoading tokenizer from  : {CONFIG['tokenizer_path']}")
    try:
        tokenizer = load_tokenizer(CONFIG["tokenizer_path"])
        print(f"  Vocab size            : {len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else '?'}")
    except FileNotFoundError:
        print(f"  [WARN] tokenizer not found at '{CONFIG['tokenizer_path']}' — aborting demo.")
        return

    print(f"\nLoading model from      : {CONFIG['model_path']}")
    try:
        model = load_model(CONFIG["model_path"], DEVICE)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters            : {param_count:,}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  [WARN] model load failed: {e}")
        return

    PROMPT = "The ancient city"
    MAX_TOKENS = 120

    strategies = [
        ("greedy",       dict()),
        ("temperature",  dict(temperature=0.9, repetition_penalty=1.2)),
        ("top_k",        dict(temperature=0.8, top_k=40, repetition_penalty=1.15)),
        ("top_p",        dict(temperature=0.8, top_p=0.92, repetition_penalty=1.15)),
        ("combined",     dict(temperature=0.75, top_k=50, top_p=0.90,
                              repetition_penalty=1.2, min_p=0.05)),
        ("beam",         dict(num_beams=4, length_penalty=1.2,
                              repetition_penalty=1.1)),
        ("contrastive",  dict(contrastive_k=5, contrastive_alpha=0.6)),
    ]

    print(f'\nPrompt: "{PROMPT}"')
    print(f"Max new tokens: {MAX_TOKENS}\n")

    for strategy, kwargs in strategies:
        print(f"── {strategy.upper()} {'─' * (52 - len(strategy))}")
        try:
            output = generate(
                PROMPT,
                strategy=strategy,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=MAX_TOKENS,
                **kwargs,
            )
            display = output.replace("\n", " ").strip()
            if len(display) > 200:
                display = display[:200] + "…"
            print(f"  {display}")
        except Exception as e:
            print(f"  [ERROR] {e}")
        print()

    print("=" * 65)
    print("  Demo complete. Import generate() into your pipeline.")
    print("=" * 65)


if __name__ == "__main__":
    _run_demo()
