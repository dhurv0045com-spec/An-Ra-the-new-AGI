"""
LLM Bridge — Central Model Hub for An-Ra AGI
=============================================

Connects ALL subsystems to the Phase 1 Neural Network:
  - Phase 2: Agent Loop (45k), Memory (45J), Training (45I), Self-Improvement (45l)
  - Phase 3: Ouroboros recursive depth (45O), Ghost Memory (45P), Identity (45N)

Singleton — the model is loaded once and shared everywhere.
Every subsystem imports:  from llm_bridge import get_llm_bridge
"""
import sys
import asyncio
from pathlib import Path
from typing import Optional, Callable

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for d in ["core", "config", "history/production (45H)", "tokenizer"]:
    p = PROJECT_ROOT / d
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class LLMBridge:
    """
    Central model hub. Loads Phase 1 LanguageModel once, exposes:
      - generate()       → text generation (sync)
      - agenerate()      → text generation (async)
      - model_fn()       → callable for 45J memory summarization
      - raw_model        → underlying Decoder for Phase 3 wrapping
      - d_model          → model dimensionality for Phase 3 gates
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMBridge, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None):
        if self._initialized:
            return

        print("[LLMBridge] Initializing Phase 1 Model...")
        if not config_path:
            config_path = str(PROJECT_ROOT / "config" / "tiny.yaml")

        from core.model import LanguageModel
        self.lm = LanguageModel(config_path=config_path, checkpoint=checkpoint_path)

        # Expose raw decoder for Phase 3 (Ouroboros wrapping, etc.)
        self.raw_decoder = self.lm._model
        self.d_model = self.lm.cfg.model.d_model
        self.vocab_size = self.lm.cfg.model.vocab_size

        self._initialized = True
        print(f"[LLMBridge] Phase 1 LanguageModel ready. "
              f"d_model={self.d_model}, vocab={self.vocab_size}, "
              f"params={self.lm.num_parameters:,}")

    # ── Text Generation ─────────────────────────────────────────────────────

    def generate(self, prompt: str, max_new_tokens: int = 200, **kwargs) -> str:
        """Synchronous text generation."""
        generated_full = self.lm.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        if generated_full.startswith(prompt):
            return generated_full[len(prompt):].strip()
        return generated_full.strip()

    async def agenerate(self, prompt: str, max_new_tokens: int = 200, **kwargs) -> str:
        """Asynchronous text generation (non-blocking for agent loop)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, max_new_tokens, **kwargs)
        )

    # ── Callable Interface for 45J Memory ───────────────────────────────────

    def model_fn(self, prompt: str) -> str:
        """
        Simple callable for subsystems that need a (str→str) function.
        Used by 45J MemoryExtractor, MemoryConsolidator, etc.
        """
        return self.generate(prompt, max_new_tokens=150)

    # ── Status ──────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "initialized": self._initialized,
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
            "parameters": self.lm.num_parameters if self._initialized else 0,
        }


# ── Global accessor ─────────────────────────────────────────────────────────

_default_bridge: Optional[LLMBridge] = None

def get_llm_bridge(config_path: Optional[str] = None,
                   checkpoint_path: Optional[str] = None) -> LLMBridge:
    """Get or create the global LLMBridge singleton."""
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = LLMBridge(config_path=config_path,
                                    checkpoint_path=checkpoint_path)
    return _default_bridge


def get_model_fn() -> Callable[[str], str]:
    """Get a simple (str→str) callable for subsystems like 45J memory."""
    return get_llm_bridge().model_fn
