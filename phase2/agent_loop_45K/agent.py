from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "agent_loop (45k)"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PATH = _ROOT / "agent_main.py"
_SPEC = importlib.util.spec_from_file_location("_anra_agent_loop_45k_agent", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load {_PATH}")
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _mod
_SPEC.loader.exec_module(_mod)

Agent = _mod.Agent
AgentLoop = _mod.AgentLoop

__all__ = ["Agent", "AgentLoop"]

