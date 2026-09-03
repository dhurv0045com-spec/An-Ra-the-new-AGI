"""Cognition readiness instrument package (Triquetra direction change).

Measures whether a checkpoint is strong enough for an experiment to be
interpretable BEFORE diagnosing failure causes. Never rescues weak cores.
"""

from .ladder import RUNGS, gen_tasks
from .identifiability import check_identifiability, required_n_mcnemar
from .gate import run_gate
from .schemas import (
    CausalResponseProfile,
    PredictionBeforeInterventionRecord,
    commit_prediction,
)

__all__ = ["RUNGS", "gen_tasks", "check_identifiability", "required_n_mcnemar",
           "run_gate", "CausalResponseProfile",
           "PredictionBeforeInterventionRecord", "commit_prediction"]
