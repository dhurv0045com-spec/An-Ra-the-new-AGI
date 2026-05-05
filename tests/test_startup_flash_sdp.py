from __future__ import annotations

import pytest

from startup_checks import assert_flash_sdp_ready


def test_startup_warns_when_flash_sdp_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.backends.cuda.is_flash_attention_available", lambda: True)
    monkeypatch.setattr("torch.backends.cuda.flash_sdp_enabled", lambda: False)

    with pytest.warns(UserWarning, match=r"flash_sdp_enabled\(\) is False"):
        assert_flash_sdp_ready("anra.py")
