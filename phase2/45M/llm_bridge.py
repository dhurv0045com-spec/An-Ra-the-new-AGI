"""
LLM Bridge - Connects Phase 2 Autonomous Loop to Phase 1 Neural Network Inference
"""
import sys
import asyncio
from pathlib import Path
from typing import Optional

# Add project root and major component directories to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for d in ["history/45H", "core"]:
    p = PROJECT_ROOT / d
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.model import LanguageModel

class LLMBridge:
    """
    Singleton bridge exposing Phase 1's LanguageModel to Phase 2.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMBridge, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None, checkpoint_path: Optional[str] = None):
        if self._initialized:
            return
            
        print("[LLMBridge] Initializing Phase 1 Model...")
        # Default config fallback
        if not config_path:
            config_path = str(PROJECT_ROOT / "config" / "tiny.yaml")
            
        self.lm = LanguageModel(config_path=config_path, checkpoint=checkpoint_path)
        self._initialized = True
        print("[LLMBridge] Phase 1 LanguageModel ready.")

    def generate(self, prompt: str, max_new_tokens: int = 200, **kwargs) -> str:
        """Synchronous text generation."""
        # Use LanguageModel's built-in generate method
        generated_full = self.lm.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        # Extract just the new tokens (remove prompt)
        if generated_full.startswith(prompt):
            return generated_full[len(prompt):].strip()
        return generated_full.strip()

    async def agenerate(self, prompt: str, max_new_tokens: int = 200, **kwargs) -> str:
        """Asynchronous text generation to prevent blocking the Agent Loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.generate(prompt, max_new_tokens, **kwargs)
        )

# Create a global default instance to be easily imported
_default_bridge = None

def get_llm_bridge(config_path: Optional[str] = None, checkpoint_path: Optional[str] = None) -> LLMBridge:
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = LLMBridge(config_path=config_path, checkpoint_path=checkpoint_path)
    return _default_bridge
