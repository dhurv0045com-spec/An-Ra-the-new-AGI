"""
Root conftest.py - pytest configuration for AN-RA.

This file ensures the project root is always on sys.path for every test
without any individual test file needing sys.path manipulation.

All imports in tests use package paths (from anra_brain import ..., etc.)
without any sys.path.append or sys.path.insert calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path exactly once.
# This replaces all sys.path.append calls in individual test files.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
