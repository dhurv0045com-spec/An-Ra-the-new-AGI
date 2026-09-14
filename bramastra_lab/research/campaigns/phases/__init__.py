"""K8 phase executors (D4): real production modules behind worker dispatch.

Each phase exposes execute(job, *, ops=None) with one typed job input and
typed result. Production ops perform real model work on GPU; tests inject
deterministic doubles that record calls without optimizer updates. Every
executor returns failure for missing evidence with actual counts (never
hardcoded success, never zero-work inheritance).
"""
from __future__ import annotations

from bramastra_lab.research.campaigns.phases.types import JobInput, PhaseResult

PHASE_EXECUTORS = {}


def register(phase: str):
    def decorator(fn):
        PHASE_EXECUTORS[phase] = fn
        return fn

    return decorator
