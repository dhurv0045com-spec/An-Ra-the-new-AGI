"""V5 training objectives: CE launch default plus a gated challenger."""

from .causal_lm import causal_lm_loss
from .query_swap import query_swap_loss

__all__ = ["causal_lm_loss", "query_swap_loss"]
