"""Backend-aware activation checkpointing for deterministic training regions."""

from __future__ import annotations

from typing import Any, Callable


def activation_checkpoint(
    function: Callable[..., Any],
    *args: Any,
    torch_module: Any = None,
    device_type: str | None = None,
) -> Any:
    """Recompute an RNG-free function in backward using the backend's API.

    PyTorch/XLA currently requires its own reentrant checkpoint implementation;
    the upstream non-reentrant implementation raises on XLA tensors. CPU and
    CUDA use upstream non-reentrant checkpointing. Callers must keep checkpointed
    functions deterministic because RNG state preservation is disabled.
    """

    if torch_module is None:
        import torch as torch_module

    if device_type is None:
        tensor_device_types = [
            value.device.type for value in args
            if isinstance(value, torch_module.Tensor)
        ]
        device_type = next(
            (kind for kind in tensor_device_types if kind != "cpu"),
            tensor_device_types[0] if tensor_device_types else "cpu",
        )
    if device_type == "xla":
        try:
            from torch_xla.utils.checkpoint import checkpoint
        except ImportError as error:
            raise RuntimeError(
                "XLA activation checkpointing requires torch_xla.utils.checkpoint"
            ) from error
        return checkpoint(
            function, *args, use_reentrant=True, preserve_rng_state=False,
        )

    from torch.utils.checkpoint import checkpoint

    return checkpoint(
        function, *args, use_reentrant=False, preserve_rng_state=False,
    )


__all__ = ["activation_checkpoint"]
