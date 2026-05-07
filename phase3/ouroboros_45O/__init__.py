from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "ouroboros (45O)"
_SPEC = importlib.util.spec_from_file_location("phase3.ouroboros_45O.ouroboros", _ROOT / "ouroboros.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Unable to load phase3/ouroboros (45O)/ouroboros.py")
ouroboros = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ouroboros
_SPEC.loader.exec_module(ouroboros)

__all__ = ["ouroboros"]
