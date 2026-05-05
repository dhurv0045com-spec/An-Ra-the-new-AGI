from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from training.dynamic_regret import DynamicRegretScheduler


def test_dynamic_regret_besbes_gur_zeevi_formula() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=3e-4)
    scheduler = DynamicRegretScheduler(
        optimizer,
        eta_base=3e-4,
        min_lr=1e-5,
        max_lr=3e-3,
        warmup_sessions=0,
        min_multiplier=0.0,
    )

    assert scheduler.current_lr() == 3e-4
    scheduler.session_start(1.0)
    lr = scheduler.session_end(0.15, 12_500)

    expected = 3e-4 * ((0.85 / 12_500) ** (1 / 3))
    assert math.isclose(lr, expected, rel_tol=1e-2)
    assert 1e-5 <= lr <= 3e-3
    assert optimizer.param_groups[0]["lr"] == lr


def test_dynamic_regret_t_zero_returns_base_lr() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=1e-4)
    scheduler = DynamicRegretScheduler(optimizer, eta_base=3e-4)
    assert scheduler.current_lr() == 3e-4


def test_regret_state_path_is_on_drive():
    """REGRET_STATE must point to Drive, not local state/."""
    from anra_paths import REGRET_STATE, DRIVE_V3_DIR

    assert str(DRIVE_V3_DIR) in str(REGRET_STATE), (
        f"REGRET_STATE={REGRET_STATE} is not under DRIVE_V3_DIR={DRIVE_V3_DIR}"
    )


def test_build_brain_imports_REGRET_STATE():
    """build_brain.py must import and use REGRET_STATE from anra_paths."""
    from anra_paths import ROOT

    src = (ROOT / "scripts" / "build_brain.py").read_text(encoding="utf-8")
    assert "REGRET_STATE" in src, (
        "build_brain.py must import REGRET_STATE from anra_paths and use it for save/load"
    )
    assert "ROOT / \"state\" / \"regret_state" not in src and \
           "ROOT/'state'/'regret_state" not in src, (
        "build_brain.py must save to REGRET_STATE (Drive), not ROOT/state/regret_state.json"
    )
