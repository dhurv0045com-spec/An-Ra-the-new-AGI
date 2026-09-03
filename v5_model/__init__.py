"""Production V5 model constructor: validated spec in, tensor core out."""

from .attention import build_attention
from .block import build_block, build_rmsnorm
from .config import ModelConfig, from_spec
from .core import (
    assert_receipt,
    assert_single_embedding,
    initialize,
    packed_layout,
    parameter_receipt,
)
from .embedding import assert_tied_storage, build_embedding
from .initialize import NORMAL_STD, initialize_module, residual_output_std

__all__ = [
    "NORMAL_STD",
    "ModelConfig",
    "assert_receipt",
    "assert_single_embedding",
    "assert_tied_storage",
    "build_attention",
    "build_block",
    "build_embedding",
    "build_rmsnorm",
    "from_spec",
    "initialize",
    "initialize_module",
    "packed_layout",
    "parameter_receipt",
    "residual_output_std",
]
