"""ARK-020 V4 A1.3 capability-template contract guard.

Engineering-only compatibility repair around the frozen scientific runner.

Several frozen V4 call sites pass the complete template table ``tt`` into
``cap_*_metrics`` even though those helpers were written for one capability's
flat template mapping. On a real campaign this reaches ``render_binding`` with
``t == {A: ..., B: ..., C: ..., D: ...}`` and fails at ``t['p']`` before main
training begins.

This overlay accepts either representation and deterministically selects the
requested capability template. It does not change task construction, examples,
metrics, thresholds, treatments, seeds, or verdicts.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import ark020_v4_durability as A1

_REVISION = "A1.3"
_EXPECTED_KEYS = {
    "A": frozenset(("p", "m", "s", "q", "t")),
    "B": frozenset(("p", "m", "s", "q", "t")),
    "C": frozenset(("p", "g", "s", "k", "q", "e")),
    "D": frozenset(("p", "m", "s", "q", "t")),
}


def template_for_capability(cap: str, t: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a validated flat template mapping for ``cap``.

    ``t`` may already be the flat mapping, or may be the campaign-wide
    ``{A, B, C, D}`` table. Ambiguous/malformed shapes fail closed with a useful
    error instead of surfacing a late ``KeyError('p')`` from task rendering.
    """
    if cap not in _EXPECTED_KEYS:
        raise RuntimeError(f"unknown ARK-020 capability {cap!r}")
    if not isinstance(t, Mapping):
        raise RuntimeError(f"template object for {cap} is not a mapping: {type(t).__name__}")

    required = _EXPECTED_KEYS[cap]
    keys = set(t.keys())
    if required.issubset(keys):
        return t

    nested = t.get(cap)
    if isinstance(nested, Mapping) and required.issubset(set(nested.keys())):
        return nested

    raise RuntimeError(
        f"template contract mismatch for capability {cap}: required {sorted(required)}, "
        f"top-level keys={sorted(map(str, keys))}"
    )


def _install_identity_binding(R) -> None:
    if getattr(A1, "_ARK020_V4_A13_IDENTITY_INSTALLED", False):
        return
    base = A1.executable_identity

    def executable_identity(runner) -> dict[str, Any]:
        x = base(runner)
        p = runner.HERE / "ark020_v4_template_guard.py"
        if not p.exists():
            raise RuntimeError("A1.3 template guard source missing")
        x = dict(x)
        x["files"] = dict(x["files"])
        x["files"]["capability_template_guard"] = A1._sha256(p)
        x["implementation_revision"] = _REVISION
        return x

    A1.executable_identity = executable_identity
    A1._ARK020_V4_A13_IDENTITY_INSTALLED = True


def install(R) -> None:
    """Install template-normalizing metric wrappers exactly once."""
    if getattr(R, "_ARK020_V4_A13_TEMPLATE_GUARD_INSTALLED", False):
        return

    originals = {
        "control": R.cap_control_metrics,
        "validation": R.cap_validation_metrics,
        "sealed": R.cap_sealed_metrics,
    }

    def control(m, cap, t, tasks, d):
        return originals["control"](m, cap, template_for_capability(cap, t), tasks, d)

    def validation(m, cap, t, tasks, d):
        return originals["validation"](m, cap, template_for_capability(cap, t), tasks, d)

    def sealed(m, cap, t, tasks, d):
        return originals["sealed"](m, cap, template_for_capability(cap, t), tasks, d)

    R.cap_control_metrics = control
    R.cap_validation_metrics = validation
    R.cap_sealed_metrics = sealed
    _install_identity_binding(R)
    R._ARK020_V4_A13_TEMPLATE_GUARD_INSTALLED = True
