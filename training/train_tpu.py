"""DEPRECATED TPU trainer - do not use.

This module contained a historically dangerous resume implementation that
silently discarded loaded checkpoint weights (the resume helper built a new
model internally and returned only metadata). The canonical TPU trainer is:

    python -m training.train_xla

Importing this module succeeds but every symbol access fails loudly, so
stale tooling cannot silently "work" against the deprecated path.
"""

from __future__ import annotations

import sys

DEPRECATION_NOTICE = """
======================================================================
training.train_tpu is DEPRECATED and must not be used.

Its resume implementation had a P0 defect (loaded checkpoint weights
were silently discarded). The canonical TPU trainer is:

    python -m training.train_xla

Update notebooks/scripts to the canonical entrypoint.
======================================================================
"""

_PUBLIC_GUARD = {"ResumeMode", "_restore_training_state", "_save_checkpoint",
                 "run", "main", "ShardWindowDataset", "resolve_pack_horizon"}


def __getattr__(name: str):
    # Module dunders (__path__, __all__, ...) resolve normally; anything that
    # looks like an API use of the deprecated trainer raises.
    if not name.startswith("__") and name in _PUBLIC_GUARD or name.isidentifier() and not name.startswith("__"):
        raise RuntimeError(
            f"training.train_tpu.{name} is deprecated and non-functional.\n"
            "Use training.train_xla (the canonical TPU trainer)."
        )
    raise AttributeError(name)


if __name__ == "__main__":
    print(DEPRECATION_NOTICE, file=sys.stderr)
    sys.exit(2)
