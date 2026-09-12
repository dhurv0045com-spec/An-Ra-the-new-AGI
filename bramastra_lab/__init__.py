"""Minimal components for the bounded BRAMASTRA experiments.

Torch-backed exports are lazy (PEP 562) so lightweight entry points such as
the research CLI never load a backend or start CUDA as a side effect of
``import bramastra_lab``.
"""
from typing import Any

_TORCH_EXPORTS = ("BramastraModel", "ModelConfig", "TransformerDecoder", "parameter_count")

__all__ = [*_TORCH_EXPORTS]


def __getattr__(name: str) -> Any:
    if name in _TORCH_EXPORTS:
        from . import model as _model

        return getattr(_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
