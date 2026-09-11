from __future__ import annotations

import pytest


def test_r1c_optimizer_wrapper_matches_canonical_frozen_adamw_values():
    torch = pytest.importorskip("torch")
    from anra_v5 import cyr_gpu014_r1c_run_v2 as compat
    from v5_training import optimizer as canonical

    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.LayerNorm(4))
    opt = compat._compatible_build_optimizer(model, torch=torch)

    assert opt.defaults["betas"] == (canonical.BETA1, canonical.BETA2) == (0.9, 0.95)
    assert opt.defaults["eps"] == canonical.EPSILON == 1e-8
    assert canonical.WEIGHT_DECAY == 0.1
    assert opt.param_groups
    assert all(float(g["lr"]) == pytest.approx(float(compat.frozen.base.CYR11_HIGH_LR)) for g in opt.param_groups)


def test_r1c_wrapper_patches_only_optimizer_boundary():
    from anra_v5 import cyr_gpu014_r1c_run_v2 as compat
    assert compat.frozen.build_optimizer is compat._compatible_build_optimizer
    assert compat.main.__module__ == "anra_v5.cyr_gpu014_r1c_run_v2"
