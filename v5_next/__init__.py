"""V5-Next: the evidence-selected next-Core contract layer.

The Task-2 analysis (docs/cymek/next_core/) concludes that the strongest
currently justified next Core is V5's architecture with an explicitly frozen
contract layer and experiment-gated output-space paths. This package adds ONLY:

  1. a contract object whose canonical hash makes the scientific configuration
     visible in receipts (full-softmax is canonical; masked/offset training
     treatments are EXPERIMENT_ONLY and can never become default silently);
  2. a reference builder that instantiates the real v5_model core from the
     contract and mechanically asserts the EOS, tied-weight, and receipt
     contracts on tiny CPU fixtures.

It deliberately does NOT duplicate or fork the V5 training stack.
"""
from .contracts import (
    CANONICAL_OUTPUT_MODE,
    EXPERIMENTAL_OUTPUT_MODES,
    NextCoreContract,
    TINY_REFERENCE_GEOMETRY,
    output_training_logits,
)
from .reference import build_reference_model, checkpoint_round_trip

__all__ = [
    "CANONICAL_OUTPUT_MODE",
    "EXPERIMENTAL_OUTPUT_MODES",
    "NextCoreContract",
    "TINY_REFERENCE_GEOMETRY",
    "build_reference_model",
    "checkpoint_round_trip",
    "output_training_logits",
]
