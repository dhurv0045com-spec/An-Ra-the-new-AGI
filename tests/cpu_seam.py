"""CPU seam for production-entry tests and dry runs (lowest device seam).

The production entry receives device, torch_module and xb as injected
parameters, so on CPU the seam only supplies the torch device - everything
above it (backend certification, checkpoint transactions, state machine,
manifests) runs unmodified. No torch_xla required.
"""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def cpu_seams():
    import torch

    yield torch, torch.device("cpu")
