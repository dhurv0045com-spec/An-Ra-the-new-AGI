"""Separate X3 experimental package testing pointer relation tracking after X2 completion."""

from .factorial import (
    X3_SYMBOLS,
    build_x3_paired_graph_factorial,
    build_x3_x2_crosswalk,
    eval_x3_shortcut_baselines,
    independent_walk_x3_graph,
    render_x3_prompt,
)
from .interventions import (
    evaluate_x3_checkpoint_suite,
    run_x3_pointer_interchange_intervention,
    score_x3_campaign,
)
from .runner import run_gated_x3_campaign

__all__ = [
    "X3_SYMBOLS",
    "build_x3_paired_graph_factorial",
    "build_x3_x2_crosswalk",
    "eval_x3_shortcut_baselines",
    "evaluate_x3_checkpoint_suite",
    "independent_walk_x3_graph",
    "render_x3_prompt",
    "run_gated_x3_campaign",
    "run_x3_pointer_interchange_intervention",
    "score_x3_campaign",
]
