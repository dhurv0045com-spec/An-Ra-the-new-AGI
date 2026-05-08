from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "fine_tuning (45I)" / "finetune" / "lora.py"
_SPEC = importlib.util.spec_from_file_location("_anra_fine_tuning_45i_lora", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load {_PATH}")
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _mod
_SPEC.loader.exec_module(_mod)

LoRALayer = _mod.LoRALayer
LoRAManager = _mod.LoRAManager
PyTorchLoRAManager = _mod.PyTorchLoRAManager
LoRALinear = _mod.LoRALinear

__all__ = ["LoRALayer", "LoRAManager", "PyTorchLoRAManager", "LoRALinear"]

