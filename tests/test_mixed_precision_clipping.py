from __future__ import annotations

import torch

from training.mixed_precision import MixedPrecisionTrainer


def test_clip_gradients_matches_torch_on_cpu() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = MixedPrecisionTrainer(device=torch.device("cpu"), enabled=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])

    norm = trainer.clip_gradients(model, optimizer, max_norm=1.0)

    assert norm == 5.0
    assert torch.allclose(model.weight.grad, torch.tensor([[0.6, 0.8]]))


def test_amp_step_uses_shared_unscale_before_clip_path() -> None:
    source = __import__("training.mixed_precision", fromlist=["amp_step"])
    text = open(source.__file__, encoding="utf-8").read()

    assert "grad_norm = mp.clip_gradients(model, optimizer, max_grad_norm)" in text
