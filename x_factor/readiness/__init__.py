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
from .pipeline import check_replication, e5_dup, e5_sham, e7_sel, run_readiness_v2
from .replication import build_evidence, check_evidence
from .assistance import ASSISTANCE, LAYERS, LEGAL_ARMS
from .adapters import (
    V4Adapter,
    V5Adapter,
    V5A_EXPECT,
    UnsupportedSubject,
    assert_v5_tokenizer_identity,
)

__all__ = ["RUNGS", "gen_tasks", "check_identifiability", "required_n_mcnemar",
           "run_gate", "CausalResponseProfile",
           "PredictionBeforeInterventionRecord", "commit_prediction",
           "assess_identifiability", "chance_report", "classify_capability",
           "wilson", "check_frontier", "CANARIES", "canary_rule",
           "gen_canary", "decide_readiness", "legal_headroom", "power_gate",
           "response_diversity", "x0_permitted", "x1_permitted",
           "check_replication", "e5_dup", "e5_sham", "e7_sel",
           "run_readiness_v2", "build_evidence", "check_evidence",
           "ASSISTANCE", "LAYERS", "LEGAL_ARMS", "V4Adapter", "V5Adapter",
           "V5A_EXPECT", "UnsupportedSubject", "assert_v5_tokenizer_identity"]
