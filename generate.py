from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypedDict

import torch
from anra.anra_paths import (
    ACTIVE_RELEASE_MANIFEST,
    ANRA_V4_CHECKPOINT,
    DRIVE_LOGS,
    HAL_STATE_FILE,
    PROMOTED_RELEASE_MANIFEST,
    ROOT,
    STATE_DIR,
)
from engine.feature_flags import is_enabled
from training.v2_config import (
    ANRA_V4_MODEL,
    ANRA_V4_MODEL_PARAMETER_COUNT,
    CANONICAL_MODEL_PROFILE,
    model_parameter_breakdown,
    model_parameter_count,
)
from training.v2_runtime import (
    active_tokenizer_path,
    build_model_for_profile,
    load_checkpoint,
    load_or_build_v2_tokenizer,
    model_summary,
)

logger = logging.getLogger(__name__)

try:
    from anra.anra_paths import get_identity_file as _get_identity_file
    from phase3.identity_45n.identity_injector import IdentityInjector as _IdentityInjector

    _identity_file = _get_identity_file()
    _IDENTITY_INJECTOR = (
        _IdentityInjector(identity_file=_identity_file) if _identity_file is not None else None
    )
except Exception:
    _IDENTITY_INJECTOR = None

try:
    from identity.hal import HALModule as _HALModule
except Exception:
    _HALModule = None

try:
    from identity.civ import ConstitutionalIdentityVector as _CIVVector
except Exception:
    _CIVVector = None  # type: ignore[assignment,misc]


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
    seed: int | None = 0
    use_think_tokens: bool = False
    use_kv_cache: bool = False
    kv_cache_backend: str = "float"
    turboquant_bits: int = 4
    mode: str = "diagnostic"
    allow_control_tokens: bool = False
    ablated_subsystem: str | None = None
    verifier_score: float | None = None
    task_success: bool | None = None
    civ_score: float | None = None
    persist_adaptive_state: bool = True


class SubsystemTrace(TypedDict, total=False):
    mode: str
    model_executed: bool
    agent_executed: bool
    tool_executed: bool
    mod_executed: bool
    rim_executed: bool
    dstp_executed: bool
    esv_executed: bool
    esv_feature_extraction_executed: bool
    esv_committed: bool
    hal_executed: bool
    hal_updated: bool
    adaptive_state_persisted: bool
    ablated_subsystem: str | None
    model: dict[str, object]
    kv_cache: dict[str, object]
    esv: dict[str, float]
    hal: dict[str, float]
    symbolic_verifier: dict[str, object]


@dataclass
class GenerationTrace:
    output: str
    strategy: str
    tokens_generated: int
    time_ms: float
    entropy_curve: list[float]
    max_prob_curve: list[float]
    # Compact full-distribution evidence: probability mass hashed into 16
    # deterministic token-ID bins at every decode step. This catches cache
    # corruption that leaves argmax, entropy, and max probability unchanged.
    distribution_probe_curve: list[list[float]]
    stopped_by: str
    repeated_ngrams_detected: bool
    kv_cache_compressed: bool = False
    kv_cache_backend: str = "none"
    # None means "not measured" (cache disabled); a float is a real measurement.
    # Reporting 0.0 unconditionally presented an unmeasured quantity as data.
    memory_saved_mb: float | None = None
    prompt_tokens: int = 0
    mode: str = "diagnostic"
    quality_state: str = "unknown"
    language_fragment_detected: bool = False
    subsystem_trace: SubsystemTrace = field(default_factory=dict)
    output_token_ids: list[int] = field(default_factory=list)
    eos_valid: bool = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_HAL_STORE: dict[str, object] = {}
_ESV_STORE: dict[str, torch.Tensor] = {}
_HAL_DIR = STATE_DIR / "hal_sessions"
_CIV_STORE: dict[str, object] = {}
_CIV_DIR = STATE_DIR / "civ_sessions"
_RUNTIME_PROFILE = "unknown"
_RUNTIME_LOAD_STATE: dict[str, object] = {}
_GENERATION_LOCK = threading.RLock()
_RUNTIME_LOAD_LOCK = threading.RLock()
_KV_CACHE_PARITY_VERIFIED = False
_KV_CACHE_PARITY_IN_PROGRESS = False
_TURBOQUANT_VERIFIED_BITS: set[int] = set()
_TURBOQUANT_BITS_IN_PROGRESS: set[int] = set()


def _reset_runtime_cache() -> None:
    """Test/helper hook: force the next generation call to reload model state."""
    global _MODEL, _TOKENIZER, _LOADED_CHECKPOINT, _RUNTIME_PROFILE, _RUNTIME_LOAD_STATE
    _MODEL = None
    _TOKENIZER = None
    _LOADED_CHECKPOINT = None
    _RUNTIME_PROFILE = "unknown"
    _RUNTIME_LOAD_STATE = {}


def unload_runtime() -> None:
    """Release the resident V4 model and reclaim local GPU cache.

    This is deliberately a serving lifecycle operation, not a checkpoint
    mutation.  The source checkpoint remains untouched; a later request can
    load the same verified artifact again.
    """

    global _MODEL, _TOKENIZER, _LOADED_CHECKPOINT, _RUNTIME_PROFILE, _RUNTIME_LOAD_STATE
    with _GENERATION_LOCK, _RUNTIME_LOAD_LOCK:
        _MODEL = None
        _TOKENIZER = None
        _LOADED_CHECKPOINT = None
        _RUNTIME_PROFILE = "unknown"
        _RUNTIME_LOAD_STATE = {}
        _ESV_STORE.clear()
        _HAL_STORE.clear()
        _CIV_STORE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with suppress(RuntimeError):
                # IPC collection is unavailable on some valid CUDA runtimes.
                torch.cuda.ipc_collect()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_model(model: object) -> object:
    return getattr(model, "model", model)


def _model_device(model: object) -> torch.device:
    """Resolve the resident model device instead of assuming the global default."""

    native_model = _native_model(model)
    parameters = getattr(native_model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            pass
    return DEVICE


def _seed_all(seed: int | None) -> None:
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


def _configure_active_release(checkpoint: Path) -> dict[str, object]:
    """Select small release-side artifacts without changing the Colab cell."""
    for manifest_path in (ACTIVE_RELEASE_MANIFEST, PROMOTED_RELEASE_MANIFEST):
        if not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        metadata = payload.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        tokenizer_value = payload.get("tokenizer") or metadata.get("tokenizer")
        if tokenizer_value and not os.environ.get("ANRA_TOKENIZER_PATH"):
            tokenizer_path = Path(str(tokenizer_value)).expanduser()
            if not tokenizer_path.is_absolute():
                tokenizer_path = (ROOT / tokenizer_path).resolve()
            if tokenizer_path.is_file():
                os.environ["ANRA_TOKENIZER_PATH"] = str(tokenizer_path)
        payload["manifest_path"] = str(manifest_path)
        payload["requested_checkpoint"] = str(checkpoint)
        return payload
    return {}


def _frontier_mode_requested() -> bool:
    """Compatibility name: the runtime is now always canonical V4."""
    return True


def _resolve_frontier_checkpoint() -> Path:
    checkpoint = _requested_checkpoint_path() or ANRA_V4_CHECKPOINT
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
        "Canonical V4 runtime requested, but anra_v4_180m.pt was not found. "
        "Set ANRA_CHECKPOINT_PATH to a trained V4 checkpoint. Legacy checkpoint "
        "fallback is disabled."
    )


def _load_frontier_runtime() -> tuple[object, object, Path, str, dict[str, object]]:
    checkpoint = _resolve_frontier_checkpoint()
    active_release = _configure_active_release(checkpoint)
    tokenizer = load_or_build_v2_tokenizer()
    model = build_model_for_profile(
        CANONICAL_MODEL_PROFILE, vocab_size=tokenizer.vocab_size
    )
    model = model.to(DEVICE)
    state = load_checkpoint(
        model,
        None,
        None,
        None,
        checkpoint,
        device=DEVICE,
        checkpoint_device=torch.device("cpu"),
        strict=False,
    )
    state["active_release"] = active_release
    if not state.get("loaded"):
        raise RuntimeError(f"Frontier checkpoint did not load: {checkpoint}")
    saved_model_config = state.get("model_config", {})
    approved = (
        saved_model_config.get("approved_subsystems", [])
        if isinstance(saved_model_config, dict)
        else []
    )
    if not isinstance(approved, list):
        raise RuntimeError("Checkpoint approved_subsystems must be a list")
    # Validate and retain the trained recipe, then return the resident model to
    # dense mode. Per-request runtime modes may activate only this recipe.
    if not hasattr(model, "configure_subsystems"):
        if approved:
            raise RuntimeError("Checkpoint approves native subsystems but model cannot gate them")
    else:
        model.configure_subsystems(approved)
        model.configure_subsystems((), approve=False)
    load_report = state.get("load_report", {})
    if isinstance(load_report, dict) and not load_report.get("exact_core_load", True):
        raise RuntimeError(
            "Frontier checkpoint has missing or mismatched core tensors: "
            f"missing={load_report.get('core_missing_keys', [])} "
            f"mismatched={load_report.get('core_mismatched_keys', [])}"
        )
    summary = model_summary(model)
    params = int(summary["parameters"])
    expected_params = model_parameter_count(ANRA_V4_MODEL, tokenizer.vocab_size)
    if expected_params != ANRA_V4_MODEL_PARAMETER_COUNT:
        raise RuntimeError(
            f"Canonical V4 parameter definition drifted: {expected_params:,} != "
            f"{ANRA_V4_MODEL_PARAMETER_COUNT:,}"
        )
    if params != expected_params:
        raise RuntimeError(
            f"Frontier parameter contract mismatch: got {params:,}, expected {expected_params:,}"
        )
    model.eval()
    if hasattr(model, "disable_kv_cache"):
        model.disable_kv_cache()
    return model, tokenizer, checkpoint, CANONICAL_MODEL_PROFILE, state


def _load_legacy_runtime() -> tuple[object, object, Path, str, dict[str, object]]:
    raise RuntimeError(
        "Legacy runtime loading was removed. Use explicit checkpoint-forensics tools "
        "for V3 artifacts; serving accepts only canonical V4."
    )


def _load_runtime() -> tuple[object, object, Path, str, dict[str, object]]:
    return _load_frontier_runtime()


_MODEL = None
_TOKENIZER = None
_LOADED_CHECKPOINT = None


def _get_runtime() -> tuple[object, object, Path | None]:
    global _MODEL, _TOKENIZER, _LOADED_CHECKPOINT, _RUNTIME_PROFILE, _RUNTIME_LOAD_STATE
    with _RUNTIME_LOAD_LOCK:
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


_SYMBOLIC_BRIDGE_STATE: dict[str, object] = {}


def _symbolic_bridge_module() -> object | None:
    """Lazy-load the 45Q bridge; sympy import costs seconds, pay it once."""
    if "module" not in _SYMBOLIC_BRIDGE_STATE:
        try:
            from phase3.symbolic_bridge_45q import symbolic_bridge as bridge

            _SYMBOLIC_BRIDGE_STATE["module"] = bridge
        except Exception as exc:
            logger.warning("Symbolic bridge unavailable: %s", exc)
            _SYMBOLIC_BRIDGE_STATE["module"] = None
    return _SYMBOLIC_BRIDGE_STATE["module"]


def _extract_user_message(prompt: str) -> str:
    """Last human turn from the canonical `H: ...\\nANRA:` prompt format."""
    tail = prompt.rsplit("H:", 1)[-1]
    return tail.split("\nANRA:", 1)[0].strip()


def _normalized_symbolic_text(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def _symbolic_verify(prompt: str, output_text: str) -> dict[str, object] | None:
    """DFC check: derive the symbolic answer independently, test the output.

    Returns None when the message is not a math/logic task or the bridge is
    unavailable; a report with ``score: None`` when the bridge cannot verify
    the task; and a binary score when a verified reference answer exists.
    Confidence must be earned by checking, never asserted.
    """
    bridge = _symbolic_bridge_module()
    if bridge is None:
        return None
    message = _extract_user_message(prompt)
    if not message:
        return None
    try:
        detection = bridge.detect(message)
        mode_name = getattr(detection.mode, "name", str(detection.mode))
        if mode_name not in {"MATH", "LOGIC"}:
            return None
        result = bridge.query(message)
        verdict = getattr(result.verdict, "name", str(result.verdict))
        expected = str(result.answer_text or "").strip()
        report: dict[str, object] = {
            "mode": mode_name,
            "verdict": verdict,
            "expected": expected,
            "score": None,
        }
        if verdict == "VERIFIED" and expected:
            from verification import DEFAULT_VERIFIER_REGISTRY

            checked = DEFAULT_VERIFIER_REGISTRY.verify(
                "symbolic_output",
                {"expected": expected, "response": output_text, "mode": mode_name},
            )
            report["score"] = float(checked.score)
            report["reason"] = str(checked.reason)
        return report
    except Exception as exc:
        logger.warning("Symbolic verification failed: %s", exc)
        return None


def _civ_path(session_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id)[:96]
    return _CIV_DIR / f"{safe or 'default'}.json"


def _get_civ(session_id: str | None) -> object | None:
    """Per-session constitutional identity vector, persisted across restarts."""
    if _CIVVector is None:
        return None
    key = session_id or "__default__"
    if key in _CIV_STORE:
        return _CIV_STORE[key]
    path = _civ_path(key)
    if path.exists():
        try:
            civ = _CIVVector.load(path)
        except Exception:
            civ = _CIVVector()
    else:
        civ = _CIVVector()
    _CIV_STORE[key] = civ
    return civ


def _save_civ(session_id: str | None, civ: object) -> None:
    if civ is None:
        return
    try:
        civ.save(_civ_path(session_id or "__default__"))
    except Exception as exc:
        logger.warning("CIV persistence failed for session %s: %s", session_id, exc)


def _get_hal(session_id: str | None) -> object | None:
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


def _save_hal(session_id: str | None, hal: object) -> None:
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


def _attach_hal(model: object, hal: object) -> None:
    """Attach HAL to external wrappers without enabling attention control."""
    if hal is None:
        return
    try:
        if hasattr(model, "hal"):
            model.hal = hal
        if hasattr(model, "model") and hasattr(model.model, "hal_module"):
            model.model.hal_module = hal
        if hasattr(model, "hal_module"):
            model.hal_module = hal
    except Exception as exc:
        logger.warning("HAL attach failed: %s", exc)


def language_fragment_detected(text: str) -> bool:
    """Public surface-coherence check shared by generation and evaluation."""
    return _language_fragment_detected(text)


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


def _blocked_token_ids(tokenizer: object, cfg: GenerationConfig) -> set[int]:
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
    seen: dict[str, int] = {}
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
    if cfg.kv_cache_backend not in {"float", "turboquant"}:
        raise ValueError("kv_cache_backend must be 'float' or 'turboquant'")
    if cfg.turboquant_bits not in {4, 8}:
        raise ValueError("turboquant_bits must be 4 or 8")
    if cfg.use_kv_cache:
        if cfg.kv_cache_backend == "float" and not (
            _KV_CACHE_PARITY_VERIFIED or _KV_CACHE_PARITY_IN_PROGRESS
        ):
            raise RuntimeError(
                "KV cache is disabled until /diagnostics/cache-parity proves "
                "exact float-cache parity"
            )
        if (
            cfg.kv_cache_backend == "turboquant"
            and cfg.turboquant_bits not in _TURBOQUANT_VERIFIED_BITS
            and cfg.turboquant_bits not in _TURBOQUANT_BITS_IN_PROGRESS
        ):
            raise RuntimeError(
                "TurboQuant cache is disabled until /diagnostics/cache-parity "
                f"passes its {cfg.turboquant_bits}-bit behavior gate"
            )

    with _GENERATION_LOCK:
        model, tokenizer, _ = _get_runtime()
        native_model = _native_model(model)
        runtime_device = _model_device(model)
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
        # The router must see a measured identity score, never a silent constant.
        # Diagnostic mode stays neutral (subsystems are off); otherwise an
        # explicit operator-provided civ_score wins, and the session's own
        # evidence-updated CIV profile is the default measurement.
        civ = None
        measured_civ_score = cfg.civ_score
        if cfg.mode != "diagnostic":
            civ = _get_civ(session_id)
            if measured_civ_score is None and civ is not None:
                try:
                    measured_civ_score = float(civ.score())
                except Exception as exc:
                    logger.warning("CIV scoring failed for session %s: %s", session_id, exc)
        if hasattr(native_model, "begin_subsystem_trace"):
            native_model.begin_subsystem_trace(civ_similarity=measured_civ_score)
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
        distribution_probe_curve: list[list[float]] = []
        stopped_by = "max_tokens"
        blocked_ids = _blocked_token_ids(tokenizer, cfg)

        start = time.perf_counter()
        kv_enabled = False
        generation_completed = False
        model_telemetry: dict[str, object] = {}
        cache_telemetry: dict[str, object] = {}
        esv_committed = False
        hal_updated = False
        if cfg.use_kv_cache and hasattr(model, "enable_kv_cache"):
            model.enable_kv_cache(
                backend=cfg.kv_cache_backend,
                turboquant_bits=cfg.turboquant_bits,
            )
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
                x = torch.tensor([token_window], dtype=torch.long, device=runtime_device)
                logits, _ = model(x)
                raw_probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
                probe = torch.zeros(16, dtype=torch.float32, device=raw_probs.device)
                probe.scatter_add_(
                    0,
                    torch.arange(raw_probs.numel(), device=raw_probs.device) % 16,
                    raw_probs,
                )
                distribution_probe_curve.append(probe.cpu().tolist())
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
            if kv_enabled and hasattr(native_model, "kv_cache_telemetry"):
                cache_telemetry = native_model.kv_cache_telemetry()
            generation_completed = True
        finally:
            if kv_enabled:
                model.clear_kv_cache()
                model.disable_kv_cache()
            if hasattr(native_model, "end_subsystem_trace"):
                native_model.end_subsystem_trace()
            if runtime_mode_state is not None:
                native_model.restore_runtime_mode(runtime_mode_state)
            native_model._runtime_civ_similarity = previous_civ_similarity
            if not generation_completed and prior_esv_state is not None:
                esv_module.state.copy_(prior_esv_state)
                native_model._pending_esv_state = None

        output_text = tokenizer.decode(answer_ids).strip()
        if cfg.use_think_tokens:
            output_text = output_text.replace("</think>", "").strip()
        if is_enabled("identity") and _IDENTITY_INJECTOR is not None:
            try:
                # The injector's real surface is clean_response(); the old
                # .clean() call could never run because the module never
                # imported, which hid the missing method.
                output_text = _IDENTITY_INJECTOR.clean_response(output_text)
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

        # Falsification pass: on symbolic tasks the answer is checkable, so
        # check it. The report lands in the trace whether or not the output
        # was accepted; the score feeds HAL/CIV truthfulness only when no
        # operator-supplied verifier already scored this generation.
        symbolic_report: dict[str, object] | None = None
        if cfg.mode == "full_system":
            symbolic_report = _symbolic_verify(prompt, output_text)
        symbolic_score = symbolic_report.get("score") if symbolic_report else None

        if quality_state == "accepted":
            if (
                cfg.persist_adaptive_state
                and cfg.mode != "diagnostic"
                and hasattr(native_model, "commit_pending_esv_state")
            ):
                esv_committed = bool(native_model.commit_pending_esv_state())
                if session_id and esv_module is not None:
                    _ESV_STORE[session_id] = esv_module.state.detach().cpu().clone()
            # HAL and CIV adapt from verified evidence, so their coherence
            # input must be the real per-generation quality signal (entropy
            # band, repetition, fragmentation, stop reason), not a flat
            # constant. Detected incoherence still hard-zeros the score.
            coherence_score = 0.0 if fragmented or repeated else float(quality)
            externally_verified = cfg.verifier_score is not None or symbolic_score is not None
            if cfg.verifier_score is not None:
                verifier_score = max(0.0, min(1.0, cfg.verifier_score))
            elif symbolic_score is not None:
                verifier_score = float(symbolic_score)
            else:
                verifier_score = coherence_score
            if hal is not None and cfg.persist_adaptive_state:
                try:
                    hal.update(
                        verifier_result=verifier_score,
                        session_context={
                            "task_type": "generation",
                            "domain": "conversation",
                            "task_success": cfg.task_success,
                            "civ_score": measured_civ_score,
                            "civ_evidence": {
                                "coherence": coherence_score,
                                "truthfulness": (
                                    verifier_score if externally_verified else None
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
            if civ is not None and cfg.persist_adaptive_state and cfg.mode != "diagnostic":
                try:
                    # The identity profile drifts only on measured evidence:
                    # coherence is always computed; truthfulness only when an
                    # operator verifier or the symbolic falsification pass
                    # actually scored this generation.
                    civ_evidence = {"coherence": coherence_score}
                    if externally_verified:
                        civ_evidence["truthfulness"] = verifier_score
                    civ.update(civ_evidence)
                    _save_civ(session_id, civ)
                except Exception as exc:
                    logger.warning("CIV update failed: %s", exc)
        elif prior_esv_state is not None:
            esv_module.state.copy_(prior_esv_state)
            native_model._pending_esv_state = None

        if (
            (cfg.mode == "diagnostic" or not cfg.persist_adaptive_state)
            and prior_esv_state is not None
        ):
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
            "esv_feature_extraction_executed": int(execution.get("esv_features", 0)) > 0,
            "esv_committed": esv_committed,
            "hal_executed": hal is not None or int(execution.get("hal", 0)) > 0,
            "hal_updated": hal_updated,
            "adaptive_state_persisted": bool(
                cfg.persist_adaptive_state
                and (esv_committed or hal_updated)
            ),
            "ablated_subsystem": cfg.ablated_subsystem,
        }
        if model_telemetry:
            subsystem_trace["model"] = model_telemetry
        if cache_telemetry:
            subsystem_trace["kv_cache"] = cache_telemetry
        if esv_module is not None and hasattr(esv_module, "as_dict"):
            subsystem_trace["esv"] = esv_module.as_dict()
        if hal is not None and hasattr(hal, "state"):
            subsystem_trace["hal"] = hal.state.hormones()
        if symbolic_report is not None:
            subsystem_trace["symbolic_verifier"] = symbolic_report
        return GenerationTrace(
            output=output_text,
            strategy=cfg.strategy,
            tokens_generated=len(entropy_curve),
            time_ms=elapsed_ms,
            entropy_curve=entropy_curve,
            max_prob_curve=max_prob_curve,
            distribution_probe_curve=distribution_probe_curve,
            stopped_by=stopped_by,
            repeated_ngrams_detected=repeated,
            kv_cache_compressed=bool(
                kv_enabled and cfg.kv_cache_backend == "turboquant"
            ),
            kv_cache_backend=cfg.kv_cache_backend if kv_enabled else "none",
            memory_saved_mb=(
                float(cache_telemetry.get("memory_saved_bytes", 0)) / 1024**2
                if kv_enabled and cfg.kv_cache_backend == "turboquant"
                else None
            ),
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

    def _curves_close(first: list[float], second: list[float]) -> bool:
        return len(first) == len(second) and all(
            abs(a - b) <= 1e-3 for a, b in zip(first, second, strict=True)
        )

    token_parity = baseline.output_token_ids == cached.output_token_ids
    # Token equality alone is blind to cache corruption that shifts logits
    # without flipping the greedy argmax (measured: a stale cached token moved
    # logits by 0.117 while every sampled token still matched). The per-step
    # entropy and max-probability curves expose distribution-level divergence.
    distribution_parity = _curves_close(
        baseline.entropy_curve, cached.entropy_curve
    ) and _curves_close(baseline.max_prob_curve, cached.max_prob_curve)
    probe_parity = len(baseline.distribution_probe_curve) == len(
        cached.distribution_probe_curve
    ) and all(
        _curves_close(first, second)
        for first, second in zip(
            baseline.distribution_probe_curve,
            cached.distribution_probe_curve,
            strict=True,
        )
    )
    distribution_parity = distribution_parity and probe_parity
    _KV_CACHE_PARITY_VERIFIED = token_parity and distribution_parity
    return {
        "verified": _KV_CACHE_PARITY_VERIFIED,
        "token_parity": token_parity,
        "distribution_parity": distribution_parity,
        "prompt_tokens": baseline.prompt_tokens,
        "tokens_compared": len(baseline.output_token_ids),
        "uncached_tokens": baseline.output_token_ids,
        "cached_tokens": cached.output_token_ids,
    }


def verify_turboquant_cache(
    prompt: str = "H: Verify compressed cache behavior for An-Ra.\nANRA:",
    *,
    max_tokens: int = 16,
    bits: int | str = "auto",
    max_distribution_delta: float = 0.025,
    max_relative_mse: float = 0.08,
    minimum_compression_ratio: float = 3.0,
) -> dict[str, object]:
    """Compare the compressed pilot with uncached inference and gate enablement.

    Unlike the exact float cache, lossy compression cannot require bit-identical
    probabilities. It must preserve greedy tokens on the probe, remain within a
    declared distribution tolerance, meet a distortion ceiling, and prove
    physical storage reduction.
    """

    if bits == "auto":
        attempts = []
        for candidate_bits in (4, 8):
            report = verify_turboquant_cache(
                prompt,
                max_tokens=max_tokens,
                bits=candidate_bits,
                max_distribution_delta=max_distribution_delta,
                max_relative_mse=max_relative_mse,
                minimum_compression_ratio=minimum_compression_ratio,
            )
            attempts.append(report)
            if report["verified"]:
                return {
                    **report,
                    "requested_bits": "auto",
                    "selected_bits": candidate_bits,
                    "attempts": attempts,
                }
        return {
            **attempts[-1],
            "requested_bits": "auto",
            "selected_bits": None,
            "attempts": attempts,
        }
    if bits not in {4, 8}:
        raise ValueError("TurboQuant verification supports 'auto', 4, or 8 bits")
    selected_bits = int(bits)

    baseline = generate_traced(
        prompt,
        GenerationConfig(
            strategy="greedy",
            max_tokens=max_tokens,
            seed=0,
            use_kv_cache=False,
            mode="diagnostic",
        ),
        session_id="turboquant_parity_probe",
    )
    _TURBOQUANT_BITS_IN_PROGRESS.add(selected_bits)
    try:
        compressed = generate_traced(
            prompt,
            GenerationConfig(
                strategy="greedy",
                max_tokens=max_tokens,
                seed=0,
                use_kv_cache=True,
                kv_cache_backend="turboquant",
                turboquant_bits=selected_bits,
                mode="diagnostic",
            ),
            session_id="turboquant_parity_probe",
        )
    finally:
        _TURBOQUANT_BITS_IN_PROGRESS.discard(selected_bits)

    def _maximum_delta(
        first: list[list[float]],
        second: list[list[float]],
    ) -> float:
        if len(first) != len(second):
            return float("inf")
        return max(
            (
                abs(a - b)
                for first_step, second_step in zip(first, second, strict=True)
                for a, b in zip(first_step, second_step, strict=True)
            ),
            default=0.0,
        )

    kv_report = compressed.subsystem_trace.get("kv_cache", {})
    if not isinstance(kv_report, dict):
        kv_report = {}
    token_parity = baseline.output_token_ids == compressed.output_token_ids
    distribution_delta = _maximum_delta(
        baseline.distribution_probe_curve,
        compressed.distribution_probe_curve,
    )
    relative_mse = float(kv_report.get("max_relative_mse", float("inf")))
    compression_ratio = float(kv_report.get("compression_ratio", 0.0))
    verified = bool(
        token_parity
        and distribution_delta <= max_distribution_delta
        and relative_mse <= max_relative_mse
        and compression_ratio >= minimum_compression_ratio
    )
    if verified:
        _TURBOQUANT_VERIFIED_BITS.add(selected_bits)
    else:
        _TURBOQUANT_VERIFIED_BITS.discard(selected_bits)
    return {
        "verified": verified,
        "backend": "turboquant",
        "bits": selected_bits,
        "selected_bits": selected_bits if verified else None,
        "verified_bits": sorted(_TURBOQUANT_VERIFIED_BITS),
        "paper_complete": False,
        "qjl_fused": False,
        "token_parity": token_parity,
        "maximum_distribution_delta": distribution_delta,
        "maximum_distribution_delta_allowed": max_distribution_delta,
        "max_relative_mse": relative_mse,
        "max_relative_mse_allowed": max_relative_mse,
        "compression_ratio": compression_ratio,
        "minimum_compression_ratio": minimum_compression_ratio,
        "tokens_compared": len(baseline.output_token_ids),
        "baseline_tokens": baseline.output_token_ids,
        "compressed_tokens": compressed.output_token_ids,
        "cache": kv_report,
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
            esv_isolated = (
                not torch.equal(_ESV_STORE[session_a], _ESV_STORE[session_b])
                and _ESV_STORE[session_a].data_ptr() != _ESV_STORE[session_b].data_ptr()
            )
            hal_isolated = _hal_path(session_a) != _hal_path(session_b)
            generation_state_isolated = not probe_generation
            if probe_generation:
                session_b_esv = _ESV_STORE[session_b].clone()
                generate_traced(
                    "H: Return one short isolation probe token.\nANRA:",
                    GenerationConfig(
                        strategy="greedy",
                        max_tokens=4,
                        seed=9127,
                        use_kv_cache=False,
                        mode="native",
                        persist_adaptive_state=False,
                    ),
                    session_id=session_a,
                )
                generation_state_isolated = torch.equal(
                    _ESV_STORE[session_b], session_b_esv
                )
            verified = (
                esv_isolated and hal_isolated and generation_state_isolated
            )
            return {
                "verified": verified,
                "generation_serialized": True,
                "runtime_generation_probed": probe_generation,
                "generation_state_isolated": generation_state_isolated,
                "esv_isolated": esv_isolated,
                "hal_paths_isolated": hal_isolated,
            }
        finally:
            _ESV_STORE.pop(session_a, None)
            _ESV_STORE.pop(session_b, None)
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
                    _native_model(_MODEL)._pending_esv_state = None


def clear_session_runtime_state(session_id: str) -> None:
    """Remove request-scoped adaptive state used by an isolated diagnostic job."""
    with _GENERATION_LOCK:
        _ESV_STORE.pop(session_id, None)
        _HAL_STORE.pop(session_id, None)
        _hal_path(session_id).unlink(missing_ok=True)
        _CIV_STORE.pop(session_id, None)
        _civ_path(session_id).unlink(missing_ok=True)


def generate(
    prompt: str,
    strategy: str = "greedy",
    max_tokens: int = 128,
    **kwargs: object,
) -> str:
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
        runtime_device = _model_device(model)
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
                    [generated_ids[-model.block_size :]],
                    dtype=torch.long,
                    device=runtime_device,
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


def get_tokenizer() -> object:
    """Return the loaded V2 tokenizer (lazy via runtime cache)."""
    return _get_runtime()[1]


def restore_embedded_data_manifests(root: str | Path) -> dict[str, object]:
    """Restore exact corpus manifests embedded in a future native checkpoint."""
    _get_runtime()
    expected = _RUNTIME_LOAD_STATE.get("data_manifests", {})
    payloads = _RUNTIME_LOAD_STATE.get("data_manifest_payloads", {})
    if not isinstance(expected, dict) or not isinstance(payloads, dict):
        return {"available": False, "restored": 0, "verified": 0, "missing": []}
    destination_root = Path(root).resolve()
    restored = 0
    verified = 0
    missing: list[str] = []
    for name, expected_digest in expected.items():
        relative = Path(str(name))
        target = (destination_root / relative).resolve()
        if destination_root != target and destination_root not in target.parents:
            raise ValueError(f"Checkpoint manifest path escapes destination root: {name}")
        payload = payloads.get(name)
        if not isinstance(payload, (bytes, bytearray)):
            missing.append(str(name))
            continue
        raw = bytes(payload)
        digest = hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(digest, str(expected_digest)):
            raise ValueError(f"Embedded checkpoint manifest hash mismatch: {name}")
        if target.is_file() and hmac.compare_digest(
            _sha256_file(target),
            str(expected_digest),
        ):
            verified += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(raw)
        temporary.replace(target)
        restored += 1
        verified += 1
    return {
        "available": bool(payloads),
        "restored": restored,
        "verified": verified,
        "expected": len(expected),
        "missing": missing,
        "complete": bool(expected) and verified == len(expected) and not missing,
    }


def __getattr__(name: str) -> object:
    if name == "TOKENIZER":
        return get_tokenizer()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_model_info() -> dict[str, object]:
    model, tokenizer, loaded_checkpoint = _get_runtime()
    summary = model_summary(model)
    parameter_breakdown = model_parameter_breakdown(ANRA_V4_MODEL, tokenizer.vocab_size)
    kv_enabled = False
    blocks = getattr(model, "blocks", getattr(getattr(model, "model", None), "blocks", []))
    try:
        kv_enabled = any(getattr(block.attn, "_kv_cache", None) is not None for block in blocks)
    except Exception:
        kv_enabled = False
    checkpoint_sha256 = _sha256_file(Path(loaded_checkpoint))
    tokenizer_path = active_tokenizer_path()
    tokenizer_sha256 = _sha256_file(tokenizer_path) if tokenizer_path.exists() else "missing"
    return {
        "model_line": "v2",
        "profile": _RUNTIME_PROFILE,
        "checkpoint": str(loaded_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": tokenizer.vocab_size,
        "param_count": summary["parameters"],
        "parameter_breakdown": parameter_breakdown,
        "trainable_parameters": summary["trainable_parameters"],
        "d_model": getattr(model, "d_model", None),
        "n_layer": getattr(model, "n_layer", None),
        "n_head": getattr(model, "n_head", None),
        "n_kv_head": getattr(model, "n_kv_head", None),
        "device": str(_model_device(model)),
        "block_size": model.block_size,
        "tokenizer_backend": getattr(tokenizer, "backend", "unknown"),
        "kv_cache_enabled": kv_enabled,
        "checkpoint_state": {
            "global_step": _RUNTIME_LOAD_STATE.get("global_step", 0),
            "best_loss": _RUNTIME_LOAD_STATE.get("best_loss", float("inf")),
            "best_training_loss": _RUNTIME_LOAD_STATE.get(
                "best_training_loss",
                _RUNTIME_LOAD_STATE.get("best_loss", float("inf")),
            ),
            "best_validation_loss": _RUNTIME_LOAD_STATE.get(
                "best_validation_loss",
                float("inf"),
            ),
            "best_answer_validation_loss": _RUNTIME_LOAD_STATE.get(
                "best_answer_validation_loss",
                float("inf"),
            ),
            "loss_semantics": _RUNTIME_LOAD_STATE.get("loss_semantics", {}),
            "sessions_completed": _RUNTIME_LOAD_STATE.get("sessions_completed", 0),
            "data_profile": _RUNTIME_LOAD_STATE.get("data_profile", "unknown"),
            "training_data_layout": _RUNTIME_LOAD_STATE.get("training_data_layout", "unknown"),
            "tokens_seen": _RUNTIME_LOAD_STATE.get("tokens_seen", 0),
            "continuation_token_counts": _RUNTIME_LOAD_STATE.get("continuation_token_counts", {}),
            "validation_history": _RUNTIME_LOAD_STATE.get("validation_history", []),
            "data_manifests": _RUNTIME_LOAD_STATE.get("data_manifests", {}),
            "model_config": _RUNTIME_LOAD_STATE.get("model_config", {}),
            "sft": _RUNTIME_LOAD_STATE.get("sft", {}),
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
