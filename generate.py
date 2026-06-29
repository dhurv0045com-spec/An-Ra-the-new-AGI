from __future__ import annotations

import hashlib
import math
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional, TypedDict

import torch

from anra.anra_paths import (
    DRIVE_LOGS,
    FRONTIER_CHECKPOINT,
    HAL_STATE_FILE,
    ROOT,
    STATE_DIR,
)
from training.v2_config import V2_FRONTIER_PARAMETER_COUNT, frontier_parameter_count
from training.v2_runtime import (
    active_tokenizer_path,
    build_frontier_model,
    build_v2_model,
    canonical_v2_checkpoint,
    load_checkpoint,
    load_or_build_v2_tokenizer,
    model_summary,
)

logger = logging.getLogger(__name__)

try:
    from anra.anra_paths import get_identity_file as _get_identity_file
    from identity_injector import IdentityInjector as _IdentityInjector

    _identity_file = _get_identity_file()
    _IDENTITY_INJECTOR = (
        _IdentityInjector(identity_file=_identity_file) if _identity_file is not None else None
    )
except Exception:
    _IDENTITY_INJECTOR = None

try:
    from ghost_memory import GhostMemory as _GhostMemory

    _GHOST_MEMORY = _GhostMemory()
except Exception:
    _GHOST_MEMORY = None

try:
    from identity.hal import HALModule as _HALModule
except Exception:
    _HALModule = None


@dataclass
class GenerationConfig:
    strategy: str = "greedy"
    max_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.92
    beam_width: int = 4
    repetition_penalty: float = 1.15
    repetition_window: int = 64
    stop_strings: list[str] = field(default_factory=list)
    seed: Optional[int] = 0
    use_think_tokens: bool = False
    use_kv_cache: bool = False
    mode: str = "diagnostic"
    allow_control_tokens: bool = False
    ablated_subsystem: str | None = None
    verifier_score: float | None = None
    task_success: bool | None = None
    civ_score: float | None = None


class SubsystemTrace(TypedDict, total=False):
    mode: str
    model_executed: bool
    agent_executed: bool
    tool_executed: bool
    mod_executed: bool
    rim_executed: bool
    dstp_executed: bool
    esv_executed: bool
    esv_committed: bool
    hal_executed: bool
    hal_updated: bool
    ghost_executed: bool
    ablated_subsystem: str | None
    model: dict[str, object]
    esv: dict[str, float]
    hal: dict[str, float]


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
    prompt_tokens: int = 0
    mode: str = "diagnostic"
    quality_state: str = "unknown"
    language_fragment_detected: bool = False
    subsystem_trace: SubsystemTrace = field(default_factory=dict)
    output_token_ids: list[int] = field(default_factory=list)
    eos_valid: bool = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_GHOST_STORE: Dict[str, dict] = {}
_HAL_STORE: Dict[str, object] = {}
_ESV_STORE: Dict[str, torch.Tensor] = {}
_HAL_DIR = STATE_DIR / "hal_sessions"
_RUNTIME_PROFILE = "unknown"
_RUNTIME_LOAD_STATE: dict[str, object] = {}
_GENERATION_LOCK = threading.RLock()
_KV_CACHE_PARITY_VERIFIED = False
_KV_CACHE_PARITY_IN_PROGRESS = False


def _reset_runtime_cache() -> None:
    """Test/helper hook: force the next generation call to reload model state."""
    global _MODEL, _TOKENIZER, _LOADED_CHECKPOINT, _RUNTIME_PROFILE, _RUNTIME_LOAD_STATE
    _MODEL = None
    _TOKENIZER = None
    _LOADED_CHECKPOINT = None
    _RUNTIME_PROFILE = "unknown"
    _RUNTIME_LOAD_STATE = {}


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_model(model):
    return getattr(model, "model", model)


def _seed_all(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _requested_checkpoint_path() -> Path | None:
    raw = os.environ.get("ANRA_CHECKPOINT_PATH", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _frontier_mode_requested() -> bool:
    profile = os.environ.get("ANRA_MODEL_PROFILE", "").strip().lower()
    if profile in {"frontier", "iterate500"}:
        return True
    checkpoint = _requested_checkpoint_path()
    if checkpoint is not None:
        return True
    return FRONTIER_CHECKPOINT.exists()


def _resolve_frontier_checkpoint() -> Path:
    checkpoint = _requested_checkpoint_path() or FRONTIER_CHECKPOINT
    if checkpoint.exists():
        return checkpoint
    try:
        from training.shared_checkpoint import restore_shared_checkpoint

        restored = restore_shared_checkpoint(checkpoint, checkpoint.name)
        if restored is not None and checkpoint.exists():
            return checkpoint
    except Exception as exc:
        logger.warning("frontier checkpoint restore failed for %s: %s", checkpoint, exc)
    raise FileNotFoundError(
        "Frontier runtime requested, but anra_frontier_500m.pt was not found. "
        "Set ANRA_CHECKPOINT_PATH to the trained checkpoint or restore the shared Drive master first."
    )


def _load_frontier_runtime():
    tokenizer = load_or_build_v2_tokenizer()
    checkpoint = _resolve_frontier_checkpoint()
    model = (
        build_frontier_model()
        if tokenizer.vocab_size == 8_209
        else build_frontier_model(vocab_size=tokenizer.vocab_size)
    )
    model = model.to(DEVICE)
    state = load_checkpoint(model, None, None, None, checkpoint, device=DEVICE, strict=False)
    if not state.get("loaded"):
        raise RuntimeError(f"Frontier checkpoint did not load: {checkpoint}")
    load_report = state.get("load_report", {})
    if isinstance(load_report, dict) and not load_report.get("exact_core_load", True):
        raise RuntimeError(
            "Frontier checkpoint has missing or mismatched core tensors: "
            f"missing={load_report.get('core_missing_keys', [])} "
            f"mismatched={load_report.get('core_mismatched_keys', [])}"
        )
    summary = model_summary(model)
    params = int(summary["parameters"])
    expected_params = frontier_parameter_count(tokenizer.vocab_size)
    if params != expected_params:
        raise RuntimeError(
            f"Frontier parameter contract mismatch: got {params:,}, expected {expected_params:,}"
        )
    model.eval()
    if hasattr(model, "disable_kv_cache"):
        model.disable_kv_cache()
    return model, tokenizer, checkpoint, "frontier", state


def _load_legacy_runtime():
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
    state = load_checkpoint(model, None, None, None, checkpoint, device=DEVICE, strict=False)
    model.eval()
    return model, tokenizer, checkpoint, "legacy", state


def _load_runtime():
    if _frontier_mode_requested():
        return _load_frontier_runtime()
    return _load_legacy_runtime()


_MODEL = None
_TOKENIZER = None
_LOADED_CHECKPOINT = None


def _get_runtime():
    global _MODEL, _TOKENIZER, _LOADED_CHECKPOINT, _RUNTIME_PROFILE, _RUNTIME_LOAD_STATE
    if _MODEL is None:
        (
            _MODEL,
            _TOKENIZER,
            _LOADED_CHECKPOINT,
            _RUNTIME_PROFILE,
            _RUNTIME_LOAD_STATE,
        ) = _load_runtime()
    return _MODEL, _TOKENIZER, _LOADED_CHECKPOINT


def _hal_path(session_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id)[:96]
    return _HAL_DIR / f"{safe or 'default'}.json"


def _get_hal(session_id: str | None):
    if _HALModule is None:
        return None
    key = session_id or "__default__"
    if key in _HAL_STORE:
        return _HAL_STORE[key]
    path = _hal_path(key)
    if path.exists():
        try:
            hal = _HALModule.load(path)
        except Exception:
            hal = _HALModule()
    else:
        hal = _HALModule()
    _HAL_STORE[key] = hal
    return hal


def _save_hal(session_id: str | None, hal) -> None:
    if hal is None:
        return
    try:
        key = session_id or "__default__"
        hal.save(_hal_path(key))
        _hal_publish_path = HAL_STATE_FILE
        _hal_save_path = DRIVE_LOGS / "hal_state.json"
        try:
            _hal_save_path.parent.mkdir(parents=True, exist_ok=True)
            hal.save(str(_hal_save_path))
            _hal_publish_path = _hal_save_path
        except Exception:
            hal.save(str(HAL_STATE_FILE))
        try:
            from runtime.hal_telemetry import publish_hal_state

            publish_hal_state(hal, source=f"generate:{key}", path=_hal_publish_path)
        except Exception:
            pass
    except Exception as exc:
        logger.warning("HAL persistence failed for session %s: %s", session_id, exc)


def _attach_hal(model, hal) -> None:
    if hal is None:
        return
    try:
        if hasattr(model, "hal"):
            model.hal = hal
        if hasattr(model, "model") and hasattr(model.model, "hal_module"):
            model.model.hal_module = hal
            model.model.use_hal = True
        if hasattr(model, "hal_module"):
            model.hal_module = hal
            model.use_hal = True
    except Exception as exc:
        logger.warning("HAL attach failed: %s", exc)


def _language_fragment_detected(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    printable = sum(character.isprintable() for character in stripped) / len(stripped)
    words = [word for word in stripped.replace("\n", " ").split(" ") if word]
    if printable < 0.98:
        return True
    if len(words) >= 12:
        one_char = sum(len(word.strip(".,!?;:'\"()[]{}")) <= 1 for word in words)
        if one_char / len(words) > 0.45:
            return True
        if "```" not in stripped and not any(symbol in stripped for symbol in ("{", "};", "=>")):
            lexical_words = re.findall(r"[a-zA-Z]+", stripped.lower())
            common = {
                "a",
                "about",
                "after",
                "all",
                "also",
                "an",
                "and",
                "answer",
                "are",
                "as",
                "at",
                "be",
                "because",
                "before",
                "but",
                "by",
                "can",
                "context",
                "data",
                "do",
                "does",
                "for",
                "from",
                "has",
                "have",
                "how",
                "i",
                "if",
                "in",
                "into",
                "is",
                "it",
                "its",
                "model",
                "not",
                "of",
                "on",
                "or",
                "result",
                "should",
                "so",
                "system",
                "task",
                "that",
                "the",
                "their",
                "then",
                "this",
                "to",
                "use",
                "was",
                "we",
                "what",
                "when",
                "which",
                "with",
                "would",
                "you",
                "your",
            }
            common_ratio = sum(word in common for word in lexical_words) / max(
                1, len(lexical_words)
            )
            if len(lexical_words) >= 12 and common_ratio < 0.18:
                return True
    return False


def _generation_quality(
    trace_output: str,
    entropy_curve: list[float],
    repeated: bool,
    fragmented: bool,
    stopped_by: str,
) -> float:
    if not trace_output.strip():
        return 0.0
    mean_entropy = sum(entropy_curve) / max(1, len(entropy_curve))
    reward = 0.55
    if 1.0 <= mean_entropy <= 8.5:
        reward += 0.15
    if repeated:
        reward -= 0.45
    if fragmented:
        reward -= 0.45
    if stopped_by == "max_tokens":
        reward -= 0.20
    elif stopped_by == "repetition_guard":
        reward -= 0.35
    elif stopped_by in {"eos", "stop_string"}:
        reward += 0.10
    if "<err>" in trace_output or "failed" in trace_output.lower():
        reward -= 0.05
    return max(0.0, min(1.0, reward))


def _apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: list[int], cfg: GenerationConfig
) -> torch.Tensor:
    if cfg.repetition_penalty <= 1.0 or not generated_ids:
        return logits
    adjusted = logits.clone()
    recent = generated_ids[-cfg.repetition_window :]
    for token_id in set(recent):
        if adjusted[token_id] >= 0:
            adjusted[token_id] /= cfg.repetition_penalty
        else:
            adjusted[token_id] *= cfg.repetition_penalty
    return adjusted


def _blocked_token_ids(tokenizer, cfg: GenerationConfig) -> set[int]:
    special = getattr(tokenizer, "special_ids", {})
    if callable(special):
        special = special()
    blocked = {
        int(getattr(tokenizer, "pad_token_id", 0)),
        int(getattr(tokenizer, "bos_token_id", 2)),
        int(getattr(tokenizer, "unk_token_id", 1)),
    }
    if not cfg.allow_control_tokens and isinstance(special, dict):
        eos = int(getattr(tokenizer, "eos_token_id", 3))
        blocked.update(int(value) for value in special.values() if int(value) != eos)
    return blocked


def _sample_next_token(
    logits: torch.Tensor,
    cfg: GenerationConfig,
    generated_ids: list[int],
    *,
    blocked_ids: set[int] | None = None,
) -> tuple[int, float, float]:
    if not torch.isfinite(logits).all():
        raise FloatingPointError("generation logits contain NaN or infinity")
    logits = _apply_repetition_penalty(logits, generated_ids, cfg)
    for token_id in blocked_ids or ():
        if 0 <= token_id < logits.numel():
            logits[token_id] = float("-inf")
    strategy = cfg.strategy.lower()
    temperature = max(cfg.temperature, 1e-6)

    if strategy == "greedy" or temperature < 1e-4:
        probs = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0:
            raise FloatingPointError("generation probabilities are invalid")
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
    if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0:
        raise FloatingPointError("generation probabilities are invalid")
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
def generate_traced(
    prompt: str, cfg: GenerationConfig, *, session_id: str | None = None
) -> GenerationTrace:
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if cfg.mode not in {"diagnostic", "native", "full_system"}:
        raise ValueError("mode must be diagnostic, native, or full_system")
    cfg = GenerationConfig(**asdict(cfg))
    cfg.max_tokens = max(1, int(cfg.max_tokens))
    cfg.top_p = max(1e-6, min(1.0, float(cfg.top_p)))
    cfg.top_k = max(1, int(cfg.top_k))
    if cfg.use_kv_cache and not (_KV_CACHE_PARITY_VERIFIED or _KV_CACHE_PARITY_IN_PROGRESS):
        raise RuntimeError(
            "KV cache is disabled until /diagnostics/cache-parity proves exact token parity"
        )

    with _GENERATION_LOCK:
        model, tokenizer, _ = _get_runtime()
        native_model = _native_model(model)
        load_report = _RUNTIME_LOAD_STATE.get("load_report", {})
        if (
            cfg.mode in {"native", "full_system"}
            and isinstance(load_report, dict)
            and not load_report.get("exact_native_load", True)
        ):
            raise RuntimeError(
                "Native runtime is blocked by missing subsystem tensors: "
                f"{load_report.get('subsystem_missing_keys', [])}"
            )
        hal = (
            _get_hal(session_id)
            if cfg.mode == "full_system" and cfg.ablated_subsystem != "hal"
            else None
        )
        if hal is not None:
            _attach_hal(model, hal)
            adjusted = hal.generation_temperature(cfg.temperature)
            cfg.temperature = max(cfg.temperature - 0.10, min(cfg.temperature + 0.10, adjusted))
        _seed_all(cfg.seed)

        runtime_mode_state = None
        if hasattr(native_model, "configure_runtime_mode"):
            runtime_mode_state = native_model.configure_runtime_mode(cfg.mode)
        if cfg.ablated_subsystem and hasattr(native_model, "neutralize_subsystem"):
            native_model.neutralize_subsystem(cfg.ablated_subsystem)
        previous_civ_similarity = getattr(native_model, "_runtime_civ_similarity", 1.0)
        if hasattr(native_model, "begin_subsystem_trace"):
            native_model.begin_subsystem_trace(civ_similarity=cfg.civ_score)
        esv_module = getattr(native_model, "esv_module", None)
        prior_esv_state = None
        if esv_module is not None and hasattr(esv_module, "state"):
            prior_esv_state = esv_module.state.detach().clone()
            if cfg.mode == "diagnostic" or (session_id and session_id not in _ESV_STORE):
                esv_module.state.zero_()
            elif session_id and session_id in _ESV_STORE:
                esv_module.state.copy_(_ESV_STORE[session_id].to(esv_module.state))

        if cfg.use_think_tokens:
            prompt = f"<think>\n{prompt}"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        max_prompt = max(1, model.block_size - cfg.max_tokens - 1)
        prompt_ids = prompt_ids[-max_prompt:]
        ids = [tokenizer.bos_token_id, *prompt_ids]
        generated_ids = ids[:]
        answer_ids: list[int] = []
        prompt_token_count = len(ids)
        entropy_curve: list[float] = []
        max_prob_curve: list[float] = []
        stopped_by = "max_tokens"
        ghost_loaded = bool(cfg.mode == "full_system" and session_id and session_id in _GHOST_STORE)
        blocked_ids = _blocked_token_ids(tokenizer, cfg)

        start = time.perf_counter()
        kv_enabled = False
        generation_completed = False
        model_telemetry: dict[str, object] = {}
        esv_committed = False
        hal_updated = False
        ghost_executed = False
        if cfg.use_kv_cache and hasattr(model, "enable_kv_cache"):
            model.enable_kv_cache()
            model.clear_kv_cache()
            kv_enabled = True
        try:
            for _ in range(cfg.max_tokens):
                if len(generated_ids) >= model.block_size:
                    stopped_by = "context_limit"
                    break
                if kv_enabled and answer_ids:
                    token_window = [generated_ids[-1]]
                else:
                    token_window = generated_ids[-model.block_size :]
                x = torch.tensor([token_window], dtype=torch.long, device=DEVICE)
                logits, _ = model(x)
                next_id, max_prob, entropy = _sample_next_token(
                    logits[0, -1, :],
                    cfg,
                    answer_ids,
                    blocked_ids=blocked_ids,
                )
                generated_ids.append(next_id)
                answer_ids.append(next_id)
                entropy_curve.append(entropy)
                max_prob_curve.append(max_prob)

                if next_id == tokenizer.eos_token_id:
                    stopped_by = "eos"
                    break

                current_text = tokenizer.decode(answer_ids)
                hit, trimmed, reason = _check_stop(current_text, cfg)
                if hit:
                    answer_ids = tokenizer.encode(trimmed, add_special_tokens=False)
                    stopped_by = reason
                    break
                if (
                    len(answer_ids) >= 12
                    and detect_repetition(current_text)["repeated_ngrams_detected"]
                ):
                    stopped_by = "repetition_guard"
                    break
            if hasattr(native_model, "subsystem_telemetry"):
                model_telemetry = native_model.subsystem_telemetry()
            generation_completed = True
        finally:
            if kv_enabled:
                model.clear_kv_cache()
                model.disable_kv_cache()
            if runtime_mode_state is not None:
                native_model.restore_runtime_mode(runtime_mode_state)
            native_model._runtime_civ_similarity = previous_civ_similarity
            if not generation_completed and prior_esv_state is not None:
                esv_module.state.copy_(prior_esv_state)
                native_model._pending_esv_state = None

        output_text = tokenizer.decode(answer_ids).strip()
        if cfg.use_think_tokens:
            output_text = output_text.replace("</think>", "").strip()
        if _IDENTITY_INJECTOR is not None:
            try:
                output_text = _IDENTITY_INJECTOR.clean(output_text)
            except Exception as exc:
                logger.warning("Identity injector cleanup failed: %s", exc)

        repetition = detect_repetition(output_text)
        repeated = bool(repetition["repeated_ngrams_detected"])
        fragmented = _language_fragment_detected(output_text)
        quality = _generation_quality(
            output_text,
            entropy_curve,
            repeated,
            fragmented,
            stopped_by,
        )
        quality_state = "accepted" if quality >= 0.65 else "rejected"

        if quality_state == "accepted":
            if cfg.mode != "diagnostic" and hasattr(native_model, "commit_pending_esv_state"):
                esv_committed = bool(native_model.commit_pending_esv_state())
                if session_id and esv_module is not None:
                    _ESV_STORE[session_id] = esv_module.state.detach().cpu().clone()
            if session_id and cfg.mode == "full_system":
                state = dict(_GHOST_STORE.get(session_id, {}))
                state.update({"session_id": session_id, "last_output": output_text})
                _GHOST_STORE[session_id] = state
                ghost_executed = True
                if _GHOST_MEMORY is not None:
                    try:
                        _GHOST_MEMORY.store(session_id, state)
                    except Exception as exc:
                        logger.warning("Ghost state persistence failed for %s: %s", session_id, exc)
            if hal is not None:
                try:
                    coherence_score = 0.0 if fragmented or repeated else 0.60
                    verifier_score = (
                        max(0.0, min(1.0, cfg.verifier_score))
                        if cfg.verifier_score is not None
                        else coherence_score
                    )
                    hal.update(
                        verifier_result=verifier_score,
                        session_context={
                            "task_type": "generation",
                            "domain": "conversation",
                            "task_success": cfg.task_success,
                            "civ_score": cfg.civ_score,
                            "civ_evidence": {
                                "coherence": coherence_score,
                                "truthfulness": (
                                    verifier_score if cfg.verifier_score is not None else None
                                ),
                            },
                            "model_incoherence_self_detected": repeated or fragmented,
                            "near_capability_boundary": verifier_score < 0.65,
                        },
                    )
                    _save_hal(session_id, hal)
                    hal_updated = True
                except Exception as exc:
                    logger.warning("HAL update failed: %s", exc)
        elif prior_esv_state is not None:
            esv_module.state.copy_(prior_esv_state)
            native_model._pending_esv_state = None

        if cfg.mode == "diagnostic" and prior_esv_state is not None:
            esv_module.state.copy_(prior_esv_state)
            native_model._pending_esv_state = None

        elapsed_ms = (time.perf_counter() - start) * 1000
        execution = model_telemetry.get("execution", {})
        if not isinstance(execution, dict):
            execution = {}
        subsystem_trace: SubsystemTrace = {
            "mode": cfg.mode,
            "model_executed": True,
            "agent_executed": False,
            "tool_executed": False,
            "mod_executed": int(execution.get("mod", 0)) > 0,
            "rim_executed": int(execution.get("rim", 0)) > 0,
            "dstp_executed": int(execution.get("dstp", 0)) > 0,
            "esv_executed": int(execution.get("esv", 0)) > 0,
            "esv_committed": esv_committed,
            "hal_executed": int(execution.get("hal", 0)) > 0,
            "hal_updated": hal_updated,
            "ghost_executed": ghost_executed,
            "ablated_subsystem": cfg.ablated_subsystem,
        }
        if cfg.ablated_subsystem in {"mod", "rim", "dstp", "esv", "hal"}:
            subsystem_trace[f"{cfg.ablated_subsystem}_executed"] = False
        if model_telemetry:
            subsystem_trace["model"] = model_telemetry
        if esv_module is not None and hasattr(esv_module, "as_dict"):
            subsystem_trace["esv"] = esv_module.as_dict()
        if hal is not None and hasattr(hal, "state"):
            subsystem_trace["hal"] = hal.state.hormones()
        return GenerationTrace(
            output=output_text,
            strategy=cfg.strategy,
            tokens_generated=len(entropy_curve),
            time_ms=elapsed_ms,
            entropy_curve=entropy_curve,
            max_prob_curve=max_prob_curve,
            stopped_by=stopped_by,
            repeated_ngrams_detected=repeated,
            kv_cache_compressed=kv_enabled,
            memory_saved_mb=0.0,
            ghost_state_loaded=ghost_loaded,
            prompt_tokens=prompt_token_count,
            mode=cfg.mode,
            quality_state=quality_state,
            language_fragment_detected=fragmented,
            subsystem_trace=subsystem_trace,
            output_token_ids=[int(token_id) for token_id in answer_ids],
            eos_valid=stopped_by == "eos",
        )


def verify_kv_cache_parity(
    prompt: str = "H: Verify cache parity for An-Ra.\nANRA:",
    *,
    max_tokens: int = 16,
) -> dict[str, object]:
    global _KV_CACHE_PARITY_IN_PROGRESS, _KV_CACHE_PARITY_VERIFIED
    baseline = generate_traced(
        prompt,
        GenerationConfig(
            strategy="greedy",
            max_tokens=max_tokens,
            seed=0,
            use_kv_cache=False,
            mode="diagnostic",
        ),
        session_id="cache_parity_probe",
    )
    _KV_CACHE_PARITY_IN_PROGRESS = True
    try:
        cached = generate_traced(
            prompt,
            GenerationConfig(
                strategy="greedy",
                max_tokens=max_tokens,
                seed=0,
                use_kv_cache=True,
                mode="diagnostic",
            ),
            session_id="cache_parity_probe",
        )
    finally:
        _KV_CACHE_PARITY_IN_PROGRESS = False
    _KV_CACHE_PARITY_VERIFIED = baseline.output_token_ids == cached.output_token_ids
    return {
        "verified": _KV_CACHE_PARITY_VERIFIED,
        "prompt_tokens": baseline.prompt_tokens,
        "tokens_compared": len(baseline.output_token_ids),
        "uncached_tokens": baseline.output_token_ids,
        "cached_tokens": cached.output_token_ids,
    }


def verify_session_state_isolation(*, probe_generation: bool = False) -> dict[str, object]:
    """Probe request-scoped stores and optionally exercise real model generation."""
    suffix = str(time.time_ns())
    session_a = f"isolation_a_{suffix}"
    session_b = f"isolation_b_{suffix}"
    model_esv_before = None
    if probe_generation and _MODEL is not None:
        esv_module = getattr(_native_model(_MODEL), "esv_module", None)
        if esv_module is not None and hasattr(esv_module, "state"):
            model_esv_before = esv_module.state.detach().clone()
    with _GENERATION_LOCK:
        try:
            _ESV_STORE[session_a] = torch.tensor([0.11, 0.22, 0.33])
            _ESV_STORE[session_b] = torch.tensor([-0.11, -0.22, -0.33])
            _GHOST_STORE[session_a] = {"session_id": session_a, "sentinel": "a"}
            _GHOST_STORE[session_b] = {"session_id": session_b, "sentinel": "b"}
            esv_isolated = (
                not torch.equal(_ESV_STORE[session_a], _ESV_STORE[session_b])
                and _ESV_STORE[session_a].data_ptr() != _ESV_STORE[session_b].data_ptr()
            )
            ghost_isolated = (
                _GHOST_STORE[session_a].get("sentinel") == "a"
                and _GHOST_STORE[session_b].get("sentinel") == "b"
            )
            hal_isolated = _hal_path(session_a) != _hal_path(session_b)
            generation_state_isolated = not probe_generation
            if probe_generation:
                session_b_esv = _ESV_STORE[session_b].clone()
                session_b_ghost = dict(_GHOST_STORE[session_b])
                generate_traced(
                    "H: Return one short isolation probe token.\nANRA:",
                    GenerationConfig(
                        strategy="greedy",
                        max_tokens=4,
                        seed=9127,
                        use_kv_cache=False,
                        mode="native",
                    ),
                    session_id=session_a,
                )
                generation_state_isolated = torch.equal(
                    _ESV_STORE[session_b], session_b_esv
                ) and _GHOST_STORE[session_b] == session_b_ghost
            verified = (
                esv_isolated and ghost_isolated and hal_isolated and generation_state_isolated
            )
            return {
                "verified": verified,
                "generation_serialized": True,
                "runtime_generation_probed": probe_generation,
                "generation_state_isolated": generation_state_isolated,
                "esv_isolated": esv_isolated,
                "ghost_isolated": ghost_isolated,
                "hal_paths_isolated": hal_isolated,
            }
        finally:
            _ESV_STORE.pop(session_a, None)
            _ESV_STORE.pop(session_b, None)
            _GHOST_STORE.pop(session_a, None)
            _GHOST_STORE.pop(session_b, None)
            _HAL_STORE.pop(session_a, None)
            _HAL_STORE.pop(session_b, None)
            _hal_path(session_a).unlink(missing_ok=True)
            _hal_path(session_b).unlink(missing_ok=True)
            if probe_generation and _MODEL is not None:
                esv_module = getattr(_native_model(_MODEL), "esv_module", None)
                if esv_module is not None and hasattr(esv_module, "state"):
                    if model_esv_before is None:
                        esv_module.state.zero_()
                    else:
                        esv_module.state.copy_(model_esv_before.to(esv_module.state))
                    setattr(_native_model(_MODEL), "_pending_esv_state", None)


def clear_session_runtime_state(session_id: str) -> None:
    """Remove request-scoped adaptive state used by an isolated diagnostic job."""
    with _GENERATION_LOCK:
        _ESV_STORE.pop(session_id, None)
        _GHOST_STORE.pop(session_id, None)
        _HAL_STORE.pop(session_id, None)
        _hal_path(session_id).unlink(missing_ok=True)


def generate(prompt: str, strategy: str = "greedy", max_tokens: int = 128, **kwargs) -> str:
    cfg = GenerationConfig(strategy=strategy, max_tokens=max_tokens)
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        elif key == "max_new_tokens":
            cfg.max_tokens = int(value)
    return generate_traced(prompt, cfg, session_id=kwargs.get("session_id")).output


def generate_stream(prompt: str, cfg: GenerationConfig) -> Iterator[str]:
    with _GENERATION_LOCK:
        model, tokenizer, _ = _get_runtime()
        native_model = _native_model(model)
        runtime_mode_state = None
        if hasattr(native_model, "configure_runtime_mode"):
            runtime_mode_state = native_model.configure_runtime_mode(cfg.mode)
        esv_module = getattr(native_model, "esv_module", None)
        prior_esv_state = (
            esv_module.state.detach().clone()
            if esv_module is not None and hasattr(esv_module, "state")
            else None
        )
        _seed_all(cfg.seed)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        generated_ids = [tokenizer.bos_token_id, *prompt_ids[-(model.block_size - 1) :]]
        answer_ids: list[int] = []
        blocked_ids = _blocked_token_ids(tokenizer, cfg)
        try:
            for _ in range(max(0, cfg.max_tokens)):
                x = torch.tensor(
                    [generated_ids[-model.block_size :]], dtype=torch.long, device=DEVICE
                )
                with torch.no_grad():
                    logits, _ = model(x)
                next_id, _, _ = _sample_next_token(
                    logits[0, -1, :],
                    cfg,
                    answer_ids,
                    blocked_ids=blocked_ids,
                )
                generated_ids.append(next_id)
                answer_ids.append(next_id)
                if next_id == tokenizer.eos_token_id:
                    break
                token_text = tokenizer.decode([next_id])
                if token_text:
                    yield token_text
        finally:
            if runtime_mode_state is not None:
                native_model.restore_runtime_mode(runtime_mode_state)
            if prior_esv_state is not None:
                esv_module.state.copy_(prior_esv_state)
                native_model._pending_esv_state = None


def load_ghost_state(session_id: str) -> dict[str, object]:
    state: dict[str, object] = {}
    if _GHOST_MEMORY is not None:
        try:
            stored = _GHOST_MEMORY.retrieve(session_id) or {}
            state.update(stored)
        except Exception:
            state.update(_GHOST_STORE.get(session_id, {}))
    else:
        state.update(_GHOST_STORE.get(session_id, {}))
    state["session_id"] = session_id
    _GHOST_STORE[session_id] = state
    return dict(state)


def save_ghost_state(session_id: str) -> None:
    state = dict(_GHOST_STORE.get(session_id, {"session_id": session_id}))
    _GHOST_STORE[session_id] = state
    if _GHOST_MEMORY is not None:
        try:
            _GHOST_MEMORY.store(session_id, state)
        except Exception as exc:
            logger.warning("Ghost state persistence failed for session %s: %s", session_id, exc)


def get_tokenizer():
    """Return the loaded V2 tokenizer (lazy via runtime cache)."""
    return _get_runtime()[1]


def __getattr__(name: str):
    if name == "TOKENIZER":
        return get_tokenizer()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_model_info() -> dict[str, object]:
    MODEL, TOKENIZER, LOADED_CHECKPOINT = _get_runtime()
    summary = model_summary(MODEL)
    kv_enabled = False
    blocks = getattr(MODEL, "blocks", getattr(getattr(MODEL, "model", None), "blocks", []))
    try:
        kv_enabled = any(getattr(block.attn, "_kv_cache", None) is not None for block in blocks)
    except Exception:
        kv_enabled = False
    checkpoint_sha256 = _sha256_file(Path(LOADED_CHECKPOINT))
    tokenizer_path = active_tokenizer_path()
    tokenizer_sha256 = _sha256_file(tokenizer_path) if tokenizer_path.exists() else "missing"
    return {
        "model_line": "v2",
        "profile": _RUNTIME_PROFILE,
        "checkpoint": str(LOADED_CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": TOKENIZER.vocab_size,
        "param_count": summary["parameters"],
        "trainable_parameters": summary["trainable_parameters"],
        "d_model": getattr(MODEL, "d_model", None),
        "n_layer": getattr(MODEL, "n_layer", None),
        "n_head": getattr(MODEL, "n_head", None),
        "n_kv_head": getattr(MODEL, "n_kv_head", None),
        "device": str(DEVICE),
        "block_size": MODEL.block_size,
        "tokenizer_backend": getattr(TOKENIZER, "backend", "unknown"),
        "kv_cache_enabled": kv_enabled,
        "checkpoint_state": {
            "global_step": _RUNTIME_LOAD_STATE.get("global_step", 0),
            "best_loss": _RUNTIME_LOAD_STATE.get("best_loss", float("inf")),
            "sessions_completed": _RUNTIME_LOAD_STATE.get("sessions_completed", 0),
            "data_profile": _RUNTIME_LOAD_STATE.get("data_profile", "unknown"),
            "training_data_layout": _RUNTIME_LOAD_STATE.get("training_data_layout", "unknown"),
            "tokens_seen": _RUNTIME_LOAD_STATE.get("tokens_seen", 0),
            "continuation_token_counts": _RUNTIME_LOAD_STATE.get("continuation_token_counts", {}),
            "best_validation_loss": _RUNTIME_LOAD_STATE.get("best_validation_loss", float("inf")),
            "validation_history": _RUNTIME_LOAD_STATE.get("validation_history", []),
            "data_manifests": _RUNTIME_LOAD_STATE.get("data_manifests", {}),
            "model_config": _RUNTIME_LOAD_STATE.get("model_config", {}),
            "source_commit": _RUNTIME_LOAD_STATE.get("source_commit", "unknown"),
            "appended_row_optimizer_steps": _RUNTIME_LOAD_STATE.get(
                "appended_row_optimizer_steps", 0
            ),
            "tokenizer_identity": _RUNTIME_LOAD_STATE.get("tokenizer_identity", {}),
            "load_report": _RUNTIME_LOAD_STATE.get("load_report", {}),
            "migration": _RUNTIME_LOAD_STATE.get("migration", {}),
        },
    }


if __name__ == "__main__":
    prompt = "H: Who are you?\nANRA:"
    trace = generate_traced(prompt, GenerationConfig(max_tokens=60))
    print(trace.output)
