from __future__ import annotations

import torch

from runtime.safe_load import ensure_torch_serialization_modules, safe_torch_load


def test_safe_load_rebinds_torch_utils_for_full_resume_payload(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "full-resume.pt"
    expected = {
        "global_step": 10_200,
        "model": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer": {"step": torch.tensor(10_200)},
    }
    torch.save(expected, checkpoint)
    monkeypatch.delattr(torch, "_utils", raising=False)

    loaded = safe_torch_load(checkpoint, map_location="cpu")

    assert isinstance(loaded, dict)
    assert loaded["global_step"] == 10_200
    assert torch.equal(loaded["model"]["weight"], expected["model"]["weight"])
    assert hasattr(torch, "_utils")


def test_serialization_module_contract_is_available() -> None:
    ensure_torch_serialization_modules()

    assert callable(torch._utils._rebuild_tensor_v2)
