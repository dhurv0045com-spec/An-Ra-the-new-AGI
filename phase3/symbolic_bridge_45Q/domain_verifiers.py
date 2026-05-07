from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "symbolic_bridge (45Q)" / "domain_verifiers.py"
_SPEC = importlib.util.spec_from_file_location("_anra_symbolic_bridge_45q_domain_verifiers", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Unable to load phase3/symbolic_bridge (45Q)/domain_verifiers.py")
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _mod
_SPEC.loader.exec_module(_mod)

VerificationResult = _mod.VerificationResult
verify_qiskit = _mod.verify_qiskit
verify_rdkit = _mod.verify_rdkit
verify_constraint_json = _mod.verify_constraint_json
verify_citation_grounding = _mod.verify_citation_grounding
verify_verilog = _mod.verify_verilog
