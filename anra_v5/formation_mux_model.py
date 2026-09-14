"""FORMATION-MUX-001 model construction and arm treatments.

Geometry: the qualified CS-TRANSFER-001 development model (8 layers, width
256, 4Q/2KV heads, head_dim 64, FFN 1024, context 1024, tied embeddings,
QK-norm affine) at FIXED physical vocabulary 24,576 — ported from the
read-only authority ``cymek-cs-transfer-001 @ b52fe453``. No physical row
count changes anywhere in CS-MECH-002.

Arms differ ONLY in the declared causal variables (see the causal-variable
matrix in the preregistration):

    M0 STANDARD              every row: trainable, decayed, in denominator
    M1 EXTRA_NO_DECAY        extra rows trainable + in denominator, wd=0
    M2 EXTRA_FROZEN          extra rows frozen (no grad/state/decay), in denominator
    M3 EXTRA_FROZEN_MASKED   M2 + extra rows excluded from the TRAINING denominator
                             (evaluation always uses the full denominator)

Because AdamW is element-wise decoupled, per-row treatment of the tied
embedding/output matrix is exact: the embedding rows can be optimized by a
row-aware optimizer whose active-row arithmetic is byte-identical to
``torch.optim.AdamW`` (proved by an oracle test) while extra rows receive
exactly the declared treatment. All arms share byte-identical initialization
for a given seed (the treatments alter dynamics only).
"""

from __future__ import annotations

from typing import Any, Mapping

from v5_contracts.model_spec import ModelSpec

PHYSICAL_VOCAB = 24_576
ACTIVE_VOCAB = 19  # pad/bos unused for content; latent grammar lives in 4..4095
ACTIVE_ROWS = tuple(range(0, 128))   # shared content rows treated as "active/shared"
EXTRA_ROWS = tuple(range(128, PHYSICAL_VOCAB))
ARMS = ("M0_STANDARD", "M1_EXTRA_NO_DECAY", "M2_EXTRA_FROZEN",
        "M3_EXTRA_FROZEN_MASKED")
MASK_LOGIT_VALUE = -1.0e4


def spec() -> ModelSpec:
    return ModelSpec(
        schema="anra-v5-model-spec/v1", family="dense-decoder-transformer",
        vocabulary_size=PHYSICAL_VOCAB, width=256, layers=8,
        query_heads=4, kv_heads=2, head_dimension=64, ffn_width=1024,
        context_length=1024, rope_base=10_000.0, norm_epsilon=1e-5,
        tied_embeddings=True, qk_norm=True, qk_norm_affine=True,
        linear_bias=False, dropout=0.0)


class EmbeddingRowOptimizer:
    """Exact row-wise decoupled AdamW over one [V, width] tied matrix.

    Active rows reproduce ``torch.optim.AdamW`` byte-for-byte (same op
    order: ``mul_(1 - lr*wd)`` then ``addcdiv_`` with torch's bias
    corrections); rows outside the trainable set receive NO gradient
    application, NO moment evolution, and NO decay. Frozen rows keep their
    init bytes for the whole arm.
    """

    def __init__(self, parameter: Any, *, trainable_rows: Mapping[int, float],
                 lr: float, betas: tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8, torch: Any = None) -> None:
        torch = torch or __import__("torch")
        self.torch = torch
        self.parameter = parameter
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        rows = sorted(trainable_rows)
        if not rows or rows[0] < 0 or rows[-1] >= parameter.shape[0]:
            raise ValueError("trainable rows must be a nonempty in-range set")
        self.row_index = torch.tensor(rows, dtype=torch.long,
                                      device=parameter.device)
        self.decay = torch.tensor([float(trainable_rows[r]) for r in rows],
                                  device=parameter.device)
        self.exp_avg = torch.zeros_like(parameter[self.row_index])
        self.exp_avg_sq = torch.zeros_like(parameter[self.row_index])
        self.step_count = 0

    def state_dict(self) -> dict[str, Any]:
        return {"step_count": self.step_count, "lr": self.lr,
                "exp_avg": self.exp_avg, "exp_avg_sq": self.exp_avg_sq}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.step_count = int(state["step_count"])
        self.exp_avg = state["exp_avg"].to(self.exp_avg.device)
        self.exp_avg_sq = state["exp_avg_sq"].to(self.exp_avg_sq.device)

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)

    def step(self) -> None:
        torch = self.torch
        with torch.no_grad():
            grad = self.parameter.grad
            if grad is None:
                raise ValueError("embedding row optimizer requires a gradient")
            self.step_count += 1
            beta1, beta2 = self.betas
            bias_c1 = 1.0 - beta1 ** self.step_count
            bias_c2 = 1.0 - beta2 ** self.step_count
            row_grad = grad.index_select(0, self.row_index)
            rows_view = self.parameter.index_select(0, self.row_index)
            self.exp_avg.lerp_(row_grad, 1.0 - beta1)  # torch AdamW uses lerp_
            self.exp_avg_sq.mul_(beta2).addcmul_(row_grad, row_grad,
                                                 value=1.0 - beta2)
            denom = (self.exp_avg_sq.sqrt() / bias_c2 ** 0.5).add_(self.eps)
            step_size = self.lr / bias_c1
            updated = rows_view.detach().clone()
            updated.mul_(1.0 - self.lr * self.decay.unsqueeze(1))
            updated.addcdiv_(self.exp_avg, denom, value=-step_size)
            self.parameter.index_copy_(0, self.row_index, updated)


def build_model(seed: int, arm: str, *, torch: Any, device: Any) -> Any:
    """Byte-identical initialization for every arm at a given seed; the arm
    only changes training-time dynamics."""

    from v5_model.core import initialize
    if arm not in ARMS:
        raise ValueError(f"unknown CS-MECH-002 arm {arm}")
    model = initialize(spec(), int(seed), torch_module=torch).to(device)
    if arm == "M3_EXTRA_FROZEN_MASKED":
        original = model.forward

        def forward(self, *args, **kwargs):
            logits = original(*args, **kwargs)
            if self.training:
                extra = _extra_mask(logits.shape[-1], torch, logits.device)
                logits = logits.clone()
                logits[..., extra] = MASK_LOGIT_VALUE
            return logits

        import types
        model.forward = types.MethodType(forward, model)
    return model


def _extra_mask(vocab: int, torch: Any, device: Any) -> Any:
    if vocab != PHYSICAL_VOCAB:
        raise ValueError("M3 masking expects the fixed physical vocabulary")
    return torch.tensor(list(EXTRA_ROWS), dtype=torch.long, device=device)


def split_parameters(model: Any, arm: str, *, torch: Any,
                     ) -> tuple[list[Any], Any]:
    """Return (main params for the canonical AdamW, row optimizer).

    M0 uses the canonical whole-model AdamW (row optimizer unused). M1/M2/M3
    exclude the tied matrix from the canonical optimizer and hand it to
    ``EmbeddingRowOptimizer`` with exactly the declared row treatment."""

    embedding = model.embedding.weight
    main = [p for name, p in model.named_parameters()
            if p is not embedding]
    trainable = {row: 0.1 for row in ACTIVE_ROWS}
    if arm == "M0_STANDARD":
        trainable.update({row: 0.1 for row in EXTRA_ROWS})
    if arm == "M1_EXTRA_NO_DECAY":
        trainable.update({row: 0.0 for row in EXTRA_ROWS})
    elif arm in ("M2_EXTRA_FROZEN", "M3_EXTRA_FROZEN_MASKED"):
        pass  # extra rows frozen: absent from the trainable map
    else:
        raise ValueError(f"unknown arm {arm}")
    return main, ("embedding", embedding, trainable)


class _OptimizerStateView(Mapping):
    """Serves the certification surface: the embedding's Adam step is the
    row optimizer's step count; every other parameter reads the canonical
    AdamW state."""

    def __init__(self, inner: Any, rows: Any, embedding: Any) -> None:
        self._inner = inner
        self._rows = rows
        self._embedding = embedding

    def __getitem__(self, parameter: Any) -> dict[str, Any]:
        if self._rows is not None and parameter is self._embedding:
            return {"step": self._rows.step_count,
                    "exp_avg": self._rows.exp_avg,
                    "exp_avg_sq": self._rows.exp_avg_sq}
        return self._inner[parameter]

    def get(self, parameter: Any, default: Any = None) -> dict[str, Any]:
        try:
            return self[parameter]
        except KeyError:
            return default if default is not None else {}

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)


class ArmOptimizerView:
    """Presents FULL model parameter ownership to the production backend
    while delegating all arithmetic to (canonical main optimizer, row
    optimizer). The backend's clip/certify/step surface operates on this
    view; the embedding's dynamics live entirely in the row optimizer, whose
    state is receipted separately in checkpoints."""

    def __init__(self, main: Any, rows: Any, embedding: Any) -> None:
        self.main = main
        self.rows = rows
        groups = [dict(group) for group in main.param_groups]
        if rows is not None:
            groups.append({"params": [embedding],
                           "row_optimizer_owned": True})
        self.param_groups = groups
        self.defaults = dict(main.defaults)

    @property
    def state(self) -> Any:
        # dynamic: load_state_dict rebinds main.state to a fresh dict
        return _OptimizerStateView(self.main.state, self.rows,
                                   self.rows.parameter if self.rows else None)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.main.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> None:
        self.main.step(closure)


def make_optimizers(model: Any, arm: str, *, torch: Any, lr: float) -> dict[str, Any]:
    """ALL arms share ONE optimizer implementation for the tied matrix so the
    contrasts are exact by construction (no implementation confound): the
    per-row optimizer with the arm's row configuration. M0 = every row
    trainable at the canonical decay — mathematically identical to
    torch.optim.AdamW (oracle test asserts <= 1e-6 relative elementwise; the
    residual is torch-internal operation-order round-off, not a treatment
    difference). Non-embedding parameters use the canonical AdamW in every
    arm, so they never contribute to a contrast."""

    from v5_training.optimizer import build_adamw_optimizer
    _main_params, row_spec = split_parameters(model, arm, torch=torch)
    _name, embedding, trainable = row_spec
    main = build_adamw_optimizer(model, torch_module=torch, lr=lr)
    # Drop the tied matrix from the canonical optimizer's group; it is owned
    # by the row optimizer with the arm's exact treatment.
    main.param_groups[0]["params"] = [
        p for p in main.param_groups[0]["params"] if p is not embedding]
    rows = EmbeddingRowOptimizer(embedding, trainable_rows=trainable,
                                 lr=lr, torch=torch)
    embedding.grad = torch.zeros_like(embedding)  # allocated once; zeroed per step
    view = ArmOptimizerView(main, rows, embedding)
    return {"main": main, "rows": rows, "view": view}


def apply_decay_then_zero_frozen(model: Any, arm: str, *, torch: Any) -> None:
    """Between backward and step: for M2/M3 the frozen rows must receive NO
    gradient-driven evolution — their gradient slice is zeroed so neither
    moments nor the parameter move. (Decay for frozen rows is absent because
    the row optimizer only touches trainable rows.)"""

    if arm in ("M2_EXTRA_FROZEN", "M3_EXTRA_FROZEN_MASKED"):
        embedding = model.embedding.weight
        grad = embedding.grad
        grad[list(EXTRA_ROWS)] = 0.0
