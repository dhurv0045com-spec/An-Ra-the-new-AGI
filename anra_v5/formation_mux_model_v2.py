"""FORMATION-MUX-001 Amendment-1 model treatments.

Scientific repairs relative to the failed pre-execution S1 design:
- shared/active rows are exactly 0..4095;
- extra rows are exactly 4096..24575;
- all CS-MECH arms keep physical V=24576;
- frozen-row gradient masking happens before the production backend global clip;
- the row-aware embedding AdamW step is executed by the optimizer view *after*
  the backend clips the whole model, never manually before the clip.
"""

from __future__ import annotations

from typing import Any, Mapping

from v5_contracts.model_spec import ModelSpec

PHYSICAL_VOCAB = 24_576
SHARED_VOCAB = 4_096
ACTIVE_ROWS = tuple(range(0, SHARED_VOCAB))
EXTRA_ROWS = tuple(range(SHARED_VOCAB, PHYSICAL_VOCAB))
ARMS = (
    "M0_STANDARD",
    "M1_EXTRA_NO_DECAY",
    "M2_EXTRA_FROZEN",
    "M3_EXTRA_FROZEN_MASKED",
)
MASK_LOGIT_VALUE = -1.0e4


def spec() -> ModelSpec:
    return ModelSpec(
        schema="anra-v5-model-spec/v1",
        family="dense-decoder-transformer",
        vocabulary_size=PHYSICAL_VOCAB,
        width=256,
        layers=8,
        query_heads=4,
        kv_heads=2,
        head_dimension=64,
        ffn_width=1024,
        context_length=1024,
        rope_base=10_000.0,
        norm_epsilon=1e-5,
        tied_embeddings=True,
        qk_norm=True,
        qk_norm_affine=True,
        linear_bias=False,
        dropout=0.0,
    )


class EmbeddingRowOptimizer:
    """Row-selective decoupled AdamW for the tied [V, d] matrix."""

    def __init__(
        self,
        parameter: Any,
        *,
        trainable_rows: Mapping[int, float],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        torch: Any = None,
    ) -> None:
        torch = torch or __import__("torch")
        self.torch = torch
        self.parameter = parameter
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        rows = sorted(int(r) for r in trainable_rows)
        if not rows or rows[0] < 0 or rows[-1] >= parameter.shape[0]:
            raise ValueError("trainable rows must be nonempty and in range")
        self.row_index = torch.tensor(rows, dtype=torch.long, device=parameter.device)
        self.decay = torch.tensor(
            [float(trainable_rows[r]) for r in rows],
            dtype=parameter.dtype,
            device=parameter.device,
        )
        self.exp_avg = torch.zeros_like(parameter[self.row_index])
        self.exp_avg_sq = torch.zeros_like(parameter[self.row_index])
        self.step_count = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "lr": self.lr,
            "exp_avg": self.exp_avg,
            "exp_avg_sq": self.exp_avg_sq,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.step_count = int(state["step_count"])
        self.lr = float(state.get("lr", self.lr))
        self.exp_avg = state["exp_avg"].to(self.exp_avg.device)
        self.exp_avg_sq = state["exp_avg_sq"].to(self.exp_avg_sq.device)

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)

    def step(self) -> None:
        torch = self.torch
        grad = self.parameter.grad
        if grad is None:
            raise ValueError("embedding row optimizer requires a gradient")
        with torch.no_grad():
            self.step_count += 1
            beta1, beta2 = self.betas
            bias_c1 = 1.0 - beta1 ** self.step_count
            bias_c2 = 1.0 - beta2 ** self.step_count
            row_grad = grad.index_select(0, self.row_index)
            rows_view = self.parameter.index_select(0, self.row_index)
            self.exp_avg.lerp_(row_grad, 1.0 - beta1)
            self.exp_avg_sq.mul_(beta2).addcmul_(
                row_grad, row_grad, value=1.0 - beta2
            )
            denom = (self.exp_avg_sq.sqrt() / bias_c2 ** 0.5).add_(self.eps)
            step_size = self.lr / bias_c1
            updated = rows_view.detach().clone()
            updated.mul_(1.0 - self.lr * self.decay.unsqueeze(1))
            updated.addcdiv_(self.exp_avg, denom, value=-step_size)
            self.parameter.index_copy_(0, self.row_index, updated)


def build_model(seed: int, arm: str, *, torch: Any, device: Any) -> Any:
    from v5_model.core import initialize

    if arm not in ARMS:
        raise ValueError(f"unknown CS-MECH-002 arm {arm}")
    model = initialize(spec(), int(seed), torch_module=torch).to(device)
    if arm == "M3_EXTRA_FROZEN_MASKED":
        original = model.forward

        def forward(self, *args, **kwargs):
            logits = original(*args, **kwargs)
            if self.training:
                logits = logits.clone()
                logits[..., SHARED_VOCAB:] = MASK_LOGIT_VALUE
            return logits

        import types

        model.forward = types.MethodType(forward, model)
    return model


class _OptimizerStateView(Mapping):
    def __init__(self, inner: Any, rows: Any, embedding: Any) -> None:
        self._inner = inner
        self._rows = rows
        self._embedding = embedding

    def __getitem__(self, parameter: Any) -> dict[str, Any]:
        if parameter is self._embedding:
            return {
                "step": self._rows.step_count,
                "exp_avg": self._rows.exp_avg,
                "exp_avg_sq": self._rows.exp_avg_sq,
            }
        return self._inner[parameter]

    def get(self, parameter: Any, default: Any = None) -> dict[str, Any]:
        try:
            return self[parameter]
        except KeyError:
            return {} if default is None else default

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)


class ArmOptimizerView:
    """Full-model optimizer facade used by ProductionTrainingBackend.

    The backend owns zero_grad -> backward -> global clip -> step. This view's
    step performs both the canonical main AdamW step and the row-aware tied
    matrix step *after* clipping.
    """

    def __init__(self, main: Any, rows: EmbeddingRowOptimizer, embedding: Any) -> None:
        self.main = main
        self.rows = rows
        self.embedding = embedding
        self._row_param_group = {
            "params": [embedding],
            "lr": float(rows.lr),
            "weight_decay": 0.0,
            "row_optimizer_owned": True,
        }
        self.defaults = dict(main.defaults)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [*self.main.param_groups, self._row_param_group]

    @property
    def state(self) -> Any:
        return _OptimizerStateView(self.main.state, self.rows, self.embedding)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.main.zero_grad(set_to_none=set_to_none)
        self.embedding.grad = None if set_to_none else self.torch.zeros_like(self.embedding)

    @property
    def torch(self) -> Any:
        return self.rows.torch

    def step(self, closure: Any = None) -> None:
        # ProductionTrainingBackend has already applied the single global clip.
        self.rows.set_lr(float(self.param_groups[-1]["lr"]))
        self.main.step(closure)
        self.rows.step()


def _trainable_row_decay(arm: str) -> dict[int, float]:
    shared = {row: 0.1 for row in ACTIVE_ROWS}
    if arm == "M0_STANDARD":
        shared.update({row: 0.1 for row in EXTRA_ROWS})
    elif arm == "M1_EXTRA_NO_DECAY":
        shared.update({row: 0.0 for row in EXTRA_ROWS})
    elif arm in ("M2_EXTRA_FROZEN", "M3_EXTRA_FROZEN_MASKED"):
        pass
    else:
        raise ValueError(f"unknown arm {arm}")
    return shared


def make_optimizers(model: Any, arm: str, *, torch: Any, lr: float) -> dict[str, Any]:
    from v5_training.optimizer import build_adamw_optimizer

    embedding = model.embedding.weight
    main = build_adamw_optimizer(model, torch_module=torch, lr=lr)
    for group in main.param_groups:
        group["params"] = [p for p in group["params"] if p is not embedding]
    rows = EmbeddingRowOptimizer(
        embedding,
        trainable_rows=_trainable_row_decay(arm),
        lr=lr,
        torch=torch,
    )
    view = ArmOptimizerView(main, rows, embedding)
    return {"main": main, "rows": rows, "view": view}


def mask_frozen_gradients_before_clip(model: Any, arm: str) -> None:
    """Remove frozen-row gradients before the one production global clip."""

    if arm not in ("M2_EXTRA_FROZEN", "M3_EXTRA_FROZEN_MASKED"):
        return
    grad = model.embedding.weight.grad
    if grad is None:
        raise RuntimeError("embedding gradient missing before frozen-row mask")
    grad[SHARED_VOCAB:] = 0.0


def assert_latent_ids_are_shared(ids: list[int] | tuple[int, ...]) -> None:
    if ids and (min(ids) < 0 or max(ids) >= SHARED_VOCAB):
        raise RuntimeError("latent scientific token escaped shared 0..4095 region")
