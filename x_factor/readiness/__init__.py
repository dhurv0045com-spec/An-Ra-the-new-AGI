"""Cognition readiness instrument package (Triquetra direction change).

Measures whether a checkpoint is strong enough for an experiment to be
interpretable BEFORE diagnosing failure causes. Never rescues weak cores.

v1 (gate.py, identifiability.check_identifiability): historical, superseded.
  v1 semantics are frozen; do not use v1 for new qualification claims.
v2 (status.py, readiness_v2.py, frontier.py, canaries.py): current contract.
"""

from .ladder import RUNGS, gen_tasks
from .identifiability import check_identifiability, required_n_mcnemar
from .gate import run_gate
from .schemas import (
    CausalResponseProfile,
    PredictionBeforeInterventionRecord,
    commit_prediction,
)
from .status import (
    assess_identifiability,
    chance_report,
    classify_capability,
    wilson,
)
from .frontier import check_frontier
from .canaries import CANARIES, canary_rule, gen_canary
from .readiness_v2 import (
    decide_readiness,
    legal_headroom,
    power_gate,
    response_diversity,
    x0_permitted,
    x1_permitted,
)

__all__ = ["RUNGS", "gen_tasks", "check_identifiability", "required_n_mcnemar",
           "run_gate", "CausalResponseProfile",
           "PredictionBeforeInterventionRecord", "commit_prediction",
           "assess_identifiability", "chance_report", "classify_capability",
           "wilson", "check_frontier", "CANARIES", "canary_rule",
           "gen_canary", "decide_readiness", "legal_headroom", "power_gate",
           "response_diversity", "x0_permitted", "x1_permitted"]
