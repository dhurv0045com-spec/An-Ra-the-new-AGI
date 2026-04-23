"""LLMBridge: central singleton around the trained CausalTransformer."""
from __future__ import annotations

import asyncio
import pickle
import sys
from pathlib import Path
from typing import Callable, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anra_brain import CausalTransformer


class LLMBridge:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, checkpoint_path: Optional[str] = None):
        if self._initialized:
            return
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tok_path = PROJECT_ROOT / "tokenizer.pkl"
        if not tok_path.exists():
            raise FileNotFoundError(f"tokenizer missing: {tok_path}")
        with tok_path.open("rb") as f:
            self.tokenizer = pickle.load(f)

        self.model = CausalTransformer(
            vocab_size=self.tokenizer.vocab_size,
            n_embd=256,
            n_head=4,
            n_layer=4,
            block_size=128,
        ).to(self.device)

        candidates = [
            Path(checkpoint_path) if checkpoint_path else None,
            PROJECT_ROOT / "anra_brain_identity.pt",
            PROJECT_ROOT / "anra_brain.pt",
        ]
        chosen = None
        for c in candidates:
            if c and c.exists():
                chosen = c
                break
        if chosen is None:
            raise FileNotFoundError("No checkpoint found. Expected anra_brain_identity.pt or anra_brain.pt")

        state = torch.load(chosen, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.raw_decoder = self.model
        self.d_model = self.model.d_model
        self.vocab_size = self.tokenizer.vocab_size
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
        self.loaded_checkpoint = str(chosen)
        self._initialized = True
        print(f"[LLMBridge] loaded {chosen.name} | d_model={self.d_model} vocab={self.vocab_size}")

    def _sample(self, logits: torch.Tensor, temperature: float = 0.8, top_k: int = 40) -> int:
        logits = logits / max(temperature, 1e-6)
        if top_k > 0:
            v, i = torch.topk(logits, min(top_k, logits.numel()))
            masked = torch.full_like(logits, float("-inf"))
            masked[i] = v
            logits = masked
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 200, **kwargs) -> str:
        ids = self.tokenizer.encode(prompt)
        if not ids:
            ids = [0]
        temperature = float(kwargs.get("temperature", 0.8))
        top_k = int(kwargs.get("top_k", 40))
        for _ in range(max_new_tokens):
            idx = torch.tensor([ids[-128:]], dtype=torch.long, device=self.device)
            logits, _ = self.model(idx)
            nxt = self._sample(logits[0, -1, :], temperature=temperature, top_k=top_k)
            ids.append(nxt)
        return self.tokenizer.decode(ids[len(self.tokenizer.encode(prompt)):]).strip()

    async def agenerate(self, prompt: str, max_new_tokens: int = 200, **kwargs) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs))

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
