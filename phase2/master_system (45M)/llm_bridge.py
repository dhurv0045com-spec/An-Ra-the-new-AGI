"""LLMBridge: central singleton around the canonical An-Ra mainline runtime."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate import MODEL, TOKENIZER, generate, get_model_info


class LLMBridge:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, checkpoint_path: Optional[str] = None):
        del checkpoint_path
        if self._initialized:
            return

        info = get_model_info()
        self.model = MODEL
        self.tokenizer = TOKENIZER
        self.raw_decoder = MODEL
        self.d_model = int(info.get("d_model") or getattr(MODEL, "d_model", 0) or 0)
        self.vocab_size = int(info.get("vocab_size") or getattr(TOKENIZER, "vocab_size", 0) or 0)
        self.num_parameters = int(info.get("param_count") or 0)
        self.loaded_checkpoint = str(info.get("checkpoint", ""))
        self._initialized = True
        print(
            f"[LLMBridge] loaded {Path(self.loaded_checkpoint).name} | "
            f"d_model={self.d_model} vocab={self.vocab_size}",
        )

    def generate(self, prompt: str, max_new_tokens: int = 200, **kwargs) -> str:
        return generate(prompt, max_tokens=max_new_tokens, **kwargs)

    async def agenerate(self, prompt: str, max_new_tokens: int = 200, **kwargs) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs),
        )

    def model_fn(self, prompt: str) -> str:
        return self.generate(prompt, max_new_tokens=150)

    def status(self) -> dict:
        return {
            "initialized": self._initialized,
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
            "parameters": self.num_parameters,
            "checkpoint": self.loaded_checkpoint,
        }


_default_bridge: Optional[LLMBridge] = None


def get_llm_bridge(config_path: Optional[str] = None, checkpoint_path: Optional[str] = None) -> LLMBridge:
    del config_path
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = LLMBridge(checkpoint_path=checkpoint_path)
    return _default_bridge


def get_model_fn() -> Callable[[str], str]:
    return get_llm_bridge().model_fn
