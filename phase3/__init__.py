"""An-Ra phase 3 subsystem package."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_phase3_aliases() -> None:
    ouro_pkg_name = "phase3.ouroboros_45O"
    if ouro_pkg_name not in sys.modules:
        pkg = types.ModuleType(ouro_pkg_name)
        pkg.__path__ = []
        ouro = _load_module(f"{ouro_pkg_name}.ouroboros", _ROOT / "ouroboros (45O)" / "ouroboros.py")
        pkg.ouroboros = ouro
        sys.modules[ouro_pkg_name] = pkg

    sym_pkg_name = "phase3.symbolic_bridge_45Q"
    if sym_pkg_name not in sys.modules:
        pkg = types.ModuleType(sym_pkg_name)
        pkg.__path__ = []
        domain_verifiers = _load_module(
            f"{sym_pkg_name}.domain_verifiers",
            _ROOT / "symbolic_bridge (45Q)" / "domain_verifiers.py",
        )
        pkg.domain_verifiers = domain_verifiers
        sys.modules[sym_pkg_name] = pkg


_install_phase3_aliases()
