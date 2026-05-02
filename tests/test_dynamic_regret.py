from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from training.dynamic_regret import DynamicRegretScheduler


def test_dynamic_regret_besbes_gur_zeevi_formula() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=3e-4)
    scheduler = DynamicRegretScheduler(optimizer, eta_base=3e-4, min_lr=1e-5, max_lr=3e-3)

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
