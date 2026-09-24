from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from v5_checkpointing import activation_checkpoint


def test_xla_checkpoint_dispatch_uses_torch_xla_reentrant_api() -> None:
    calls: list[dict[str, object]] = []
    checkpoint_module = ModuleType("torch_xla.utils.checkpoint")

    def fake_checkpoint(function, *args, **kwargs):
        calls.append(kwargs)
        return function(*args)

    checkpoint_module.checkpoint = fake_checkpoint
    utils_module = ModuleType("torch_xla.utils")
    xla_module = ModuleType("torch_xla")
    utils_module.checkpoint = checkpoint_module
    xla_module.utils = utils_module
    modules = {
        "torch_xla": xla_module,
        "torch_xla.utils": utils_module,
        "torch_xla.utils.checkpoint": checkpoint_module,
    }

    with patch.dict(sys.modules, modules):
        value = activation_checkpoint(
            lambda tensor: tensor + 1,
            torch.tensor(4.0),
            torch_module=torch,
            device_type="xla",
        )

    assert float(value.item()) == 5.0
    assert calls == [{"use_reentrant": True, "preserve_rng_state": False}]
