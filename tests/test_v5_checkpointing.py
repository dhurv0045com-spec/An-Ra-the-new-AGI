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



@pytest.mark.parametrize("lineage", ["", ".", "..", "a/b", "a:b", "NUL", "name.", "name "])
def test_checkpoint_store_rejects_unsafe_lineage(tmp_path, lineage):
    from v5_training.checkpoint import CheckpointStore

    with pytest.raises(ValueError, match="safe single path component"):
        CheckpointStore(tmp_path / "store", lineage)
    assert not (tmp_path / "store").exists()


def test_checkpoint_explicit_empty_identity_fails_closed(tmp_path):
    from v5_training.checkpoint import CheckpointStore

    store = CheckpointStore(tmp_path, "safe")
    store.latest.write_text("a" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        store.restore("")
