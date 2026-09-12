"""ARK-020 V4 A1.2 device-consistency guard.

Engineering-only compatibility repair. The frozen scientific runner's exact-resume
helper accepted an explicit device but its internal _advance() re-selected a device
from global CUDA availability. On a CUDA host, a CPU-directed smoke therefore mixed
a CPU model with CUDA inputs. Production CUDA runs happened to be internally
consistent, but the function contract was false and Colab exposed it.

This overlay preserves the scientific protocol and makes _advance() derive its device
from the actual restored model. It also binds this source into the immutable A1
executable identity receipt.
"""
from __future__ import annotations

from typing import Any

import ark020_v4_durability as A1

_REVISION = "A1.2"


def _install_identity_binding(R) -> None:
    if getattr(A1, "_ARK020_V4_A12_IDENTITY_INSTALLED", False):
        return
    base = A1.executable_identity

    def executable_identity(runner) -> dict[str, Any]:
        x = base(runner)
        p = runner.HERE / "ark020_v4_device_guard.py"
        if not p.exists():
            raise RuntimeError("A1.2 device guard source missing")
        x = dict(x)
        x["files"] = dict(x["files"])
        x["files"]["device_consistency_guard"] = A1._sha256(p)
        x["implementation_revision"] = _REVISION
        return x

    A1.executable_identity = executable_identity
    A1._ARK020_V4_A12_IDENTITY_INSTALLED = True


def install(R) -> None:
    """Install the device-consistent exact-resume advance helper exactly once."""
    if getattr(R, "_ARK020_V4_A12_DEVICE_GUARD_INSTALLED", False):
        return

    def _advance(st, tasks, tt, bufs, dose_b, start: int, stop: int) -> list:
        rows = []
        names = R.pnames(st["m"])
        try:
            d = next(st["m"].parameters()).device
        except StopIteration as exc:
            raise RuntimeError("cannot determine exact-resume model device") from exc

        for step in range(start, stop + 1):
            rep = ([("A", tt["A"], tasks["sem"]["A"]["train"])]
                   if step % 2 == 0 else [])
            r = R.mixed_update(
                st["m"], st["o"], st["sc"], bufs["train"], "B", tt["B"],
                tasks["sem"]["B"]["train"], st["b_seed"], step, dose_b, rep, None,
                d, names, tag="resume-smoke",
            )
            st["phase_step"] = step
            st["global_step"] += 1
            st["cnt"]["replay_slots"] += r["replay_slots"]
            st["stream_receipts"].append({
                "step": step,
                "real_starts_sha256": r["real_starts_sha256"],
            })
            rows.append((
                step,
                r["replay_slots"],
                r["real_starts_sha256"],
                round(r["loss"], 10),
            ))
        return rows

    R._advance = _advance
    _install_identity_binding(R)
    R._ARK020_V4_A12_DEVICE_GUARD_INSTALLED = True
