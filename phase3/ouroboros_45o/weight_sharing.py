"""
weight_sharing.py — Phase 3 | Component 45O
Weight Sharing: verifying and documenting the parameter efficiency of Ouroboros.

This is not a complex module. Its job is clarity:
  - Formally verify that all passes share the same weights
  - Count exactly what was added
  - Make the efficiency gain measurable and auditable

The innovation in Ouroboros is NOT clever mathematics.
It is discipline: the refusal to add parameters when recursion suffices.
This file enforces and documents that discipline.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


def verify_weight_sharing(ouroboros_model) -> Dict[str, bool]:
    """
    Confirms that Ouroboros passes share the base model's weights
    and have not accidentally introduced per-pass copies.

    Args:
        ouroboros_model: an OuroborosDecoder instance

    Returns:
        dict with boolean checks, all should be True
    """
    base_params = dict(ouroboros_model.model.named_parameters())
    checks      = {}

    # Every base-model parameter should appear exactly once in state_dict
    state = ouroboros_model.state_dict()
    for name, param in base_params.items():
        full_name = f"model.{name}"
        if full_name in state:
            checks[f"param_shared_{name[:30]}"] = True
        else:
            checks[f"param_missing_{name[:30]}"] = False

    # Ouroboros should not have duplicated any base param under a new key
    # Check: no parameter named "pass_0_*", "pass_1_*" etc (would indicate per-pass copies)
    duplicated_keys = [k for k in state.keys() if any(
        f"pass_{i}_" in k for i in range(ouroboros_model.n_passes)
    )]
    checks["no_per_pass_weight_copies"] = len(duplicated_keys) == 0

    # The only new keys should be pass_gates and blend_weights
    expected_new_keys = {"pass_gates", "blend_weights"}
    actual_new_keys   = {
        k for k in state.keys()
        if not k.startswith("model.")
    }
    checks["only_expected_new_params"] = actual_new_keys == expected_new_keys

    return checks


def parameter_audit(ouroboros_model) -> Dict[str, int]:
    """
    Precise count of every parameter category in the Ouroboros model.
    Designed to make the efficiency claim auditable to the exact integer.

    Returns a dict with:
      base_model_params      — parameters from the wrapped transformer
      pass_gates_params      — n_passes × d_model
      blend_weights_params   — n_passes
      total_new_params       — sum of the two above
      total_params           — everything
      equivalent_layer_cost  — approx params of a single new transformer layer
                               at the same d_model (for comparison)
    """
    d_model  = ouroboros_model.d_model
    n_passes = ouroboros_model.n_passes

    base_params    = sum(p.numel() for p in ouroboros_model.model.parameters())
    gate_params    = ouroboros_model.pass_gates.numel()      # n_passes × d_model
    weight_params  = ouroboros_model.blend_weights.numel()   # n_passes
    total_new      = gate_params + weight_params
    total          = base_params + total_new

    # A single transformer layer at d_model dimensions costs roughly:
    #   4 weight matrices in self-attention: 4 × d_model² = 4 × 256² = 262,144
    #   2 weight matrices in FFN (4x expansion): 2 × d_model × 4d_model = 524,288
    #   biases add ~1% more
    # Total ≈ 6 × d_model² = 393,216 for d_model=256
    one_layer_cost = 6 * d_model * d_model

    return {
        "base_model_params"    : base_params,
        "pass_gates_params"    : gate_params,
        "blend_weights_params" : weight_params,
        "total_new_params"     : total_new,
        "total_params"         : total,
        "one_layer_equivalent" : one_layer_cost,
        "ouroboros_vs_layer"   : f"{total_new} vs {one_layer_cost} ({100*total_new/one_layer_cost:.2f}% of one layer)",
        "overhead_pct"         : f"{100 * total_new / base_params:.4f}%",
    }


def parameter_budget_report(ouroboros_model) -> str:
    """
    Human-readable report of parameter efficiency.
    Intended for the progress report and README.
    """
    audit = parameter_audit(ouroboros_model)
    lines = [
        "═══════════════════════════════════════",
        "  OUROBOROS PARAMETER BUDGET REPORT",
        "═══════════════════════════════════════",
        f"  Base model parameters : {audit['base_model_params']:>12,}",
        f"  Pass gates added      : {audit['pass_gates_params']:>12,}  (n_passes × d_model)",
        f"  Blend weights added   : {audit['blend_weights_params']:>12,}  (n_passes scalars)",
        f"  Total new parameters  : {audit['total_new_params']:>12,}",
        f"  ─────────────────────────────────────",
        f"  Total parameters      : {audit['total_params']:>12,}",
        f"  Overhead              : {audit['overhead_pct']}",
        f"",
        f"  Compare to adding one transformer layer:",
        f"  {audit['ouroboros_vs_layer']}",
        "═══════════════════════════════════════",
    ]
    return "\n".join(lines)


def compute_pass_contribution(ouroboros_model, hidden_states_list) -> Dict[str, float]:
    """
    After a forward pass, compute how much each pass contributed
    to the final accumulated hidden state.

    This is a diagnostic tool to verify that all passes are active
    (none collapsed to zero contribution) and that the blend is meaningful.

    Args:
        ouroboros_model: OuroborosDecoder instance
        hidden_states_list: list of per-pass hidden states (from _last_pass_hiddens)

    Returns:
        dict mapping pass name to its fractional contribution
    """
    import torch.nn.functional as F

    blend = F.softmax(ouroboros_model.blend_weights, dim=0).detach()

    result = {}
    for i, (h, w) in enumerate(zip(hidden_states_list, blend)):
        magnitude   = h.norm().item()
        contribution = (w.item() * magnitude)
        result[f"pass_{i+1}_weight"]       = round(w.item(), 4)
        result[f"pass_{i+1}_hidden_norm"]  = round(magnitude, 4)
        result[f"pass_{i+1}_contribution"] = round(contribution, 4)

    return result
