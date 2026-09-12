"""Cross-environment ARK-020 test bootstrap.

Keeps the inherited V4 suite semantically identical while installing the A1.2
engineering device-consistency overlay before test modules execute. This makes the
explicit-CPU exact-resume regression behave the same on CPU CI and CUDA-equipped
Colab hosts.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V4_DIR = ROOT / "experiments" / "ARK-020-V4"
ARK019_DIR = ROOT / "experiments" / "ARK-019"
ARK018_DIR = ROOT / "experiments" / "ARK-018"

for p in (str(V4_DIR), str(ARK019_DIR), str(ARK018_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import ark020_v4_device_guard as DEVICE_GUARD  # noqa: E402
import run_ark020_v4 as R  # noqa: E402

DEVICE_GUARD.install(R)
