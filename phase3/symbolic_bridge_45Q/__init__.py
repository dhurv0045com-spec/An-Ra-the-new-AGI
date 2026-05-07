from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "symbolic_bridge (45Q)"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from . import domain_verifiers

__all__ = ["domain_verifiers"]
