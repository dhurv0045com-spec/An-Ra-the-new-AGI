"""Canonical, reversible parameter-extension contract for An-Ra.

The V4 checkpoint remains immutable.  Capability adapters contain only
LoRA/DoRA parameters plus a hash-bound manifest naming the exact base model,
tokenizer, target modules, and source code.  Loading is weights-only, strict,
and reversible; detaching restores the original ``nn.Linear`` modules.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - canonical PyTorch alias

CAPABILITY_ADAPTER_SCHEMA_VERSION = 1
_HASH_LENGTH = 64


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_hash(value: str, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != _HASH_LENGTH or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dora: bool = False,
    ) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        if self.rank <= 0 or self.rank > min(base.in_features, base.out_features):
            raise ValueError("adapter rank must fit the wrapped linear layer")
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features, **factory))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank, **factory))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.magnitude = nn.Parameter(base.weight.detach().norm(dim=1)) if dora else None

    @property
    def dora(self) -> bool:
        return self.magnitude is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.magnitude is not None:
            direction = self.base.weight.detach() + self.scale * (self.lora_b @ self.lora_a)
            row_norm = direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
            adapted_weight = direction * (self.magnitude[:, None] / row_norm)
            return F.linear(x, adapted_weight, self.base.bias)
        base_output = self.base(x)
        delta = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale
        return base_output + delta


def attach_candidate_adapters(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dora: bool = False,
    predicate: Callable[[str, nn.Linear], bool] | None = None,
    target_modules: tuple[str, ...] | None = None,
) -> list[str]:
    """Freeze the base and attach adapters to an explicit, reproducible target set."""

    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise RuntimeError("model already has an active capability adapter")
    if int(rank) <= 0 or not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise ValueError("adapter rank and alpha must be positive")
    requested = set(target_modules or ())
    eligible: list[tuple[str, nn.Linear]] = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if requested and module_name not in requested:
            continue
        if predicate is not None and not predicate(module_name, module):
            continue
        if int(rank) > min(module.in_features, module.out_features):
            if requested:
                continue
            # Broad recipes skip tiny projections instead of leaving a partially
            # wrapped model when their requested rank cannot fit.
            continue
        eligible.append((module_name, module))
    eligible_names = {name for name, _ in eligible}
    if requested and requested != eligible_names:
        missing = sorted(requested - eligible_names)
        raise ValueError(f"adapter target modules are absent or ineligible: {missing}")
    if not eligible:
        raise ValueError("adapter recipe selected no eligible linear modules")

    original_trainability = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad = False
    attached: list[str] = []
    try:
        for module_name, module in eligible:
            parent_name, _, child_name = module_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(
                parent,
                child_name,
                LoRALinear(module, rank=rank, alpha=alpha, dora=dora),
            )
            attached.append(module_name)
    except Exception:
        detach_candidate_adapters(model)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = original_trainability[name]
        raise
    return attached


def detach_candidate_adapters(model: nn.Module) -> tuple[str, ...]:
    """Remove every active adapter without merging it into immutable base weights."""

    detached: list[str] = []
    for module_name, module in reversed(list(model.named_modules())):
        if not isinstance(module, LoRALinear):
            continue
        parent_name, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, module.base)
        detached.append(module_name)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return tuple(reversed(detached))


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.endswith((".lora_a", ".lora_b", ".magnitude"))
    }


def load_adapter_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith((".lora_a", ".lora_b", ".magnitude"))
    }
    if set(current) != set(state):
        raise ValueError(
            "adapter state keys do not match attached modules: "
            f"missing={sorted(set(current) - set(state))}, "
            f"unexpected={sorted(set(state) - set(current))}"
        )
    with torch.no_grad():
        for name, parameter in current.items():
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise ValueError(f"adapter tensor shape mismatch at {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


@dataclass(frozen=True)
class CapabilityAdapterSpec:
    capability_id: str
    kind: str
    base_model_profile: str
    base_checkpoint_sha256: str
    tokenizer_sha256: str
    rank: int
    alpha: float
    target_modules: tuple[str, ...]
    trainable_parameters: int
    source_commit: str
    created_at: float

    def validate(self) -> None:
        if not self.capability_id or self.kind not in {"lora", "dora"}:
            raise ValueError("capability adapter requires an id and kind lora/dora")
        if not self.base_model_profile or not self.target_modules:
            raise ValueError("capability adapter must bind a model and target modules")
        _validated_hash(self.base_checkpoint_sha256, "base checkpoint hash")
        _validated_hash(self.tokenizer_sha256, "tokenizer hash")
        if self.rank <= 0 or self.alpha <= 0.0 or self.trainable_parameters <= 0:
            raise ValueError("capability adapter dimensions must be positive")


def save_capability_adapter(
    model: nn.Module,
    path: str | Path,
    *,
    capability_id: str,
    base_model_profile: str,
    base_checkpoint_sha256: str,
    tokenizer_sha256: str,
    source_commit: str,
) -> dict[str, object]:
    modules = tuple(
        name for name, module in model.named_modules() if isinstance(module, LoRALinear)
    )
    if not modules:
        raise ValueError("cannot save a model with no attached capability adapter")
    wrappers = [model.get_submodule(name) for name in modules]
    first = wrappers[0]
    assert isinstance(first, LoRALinear)
    if any(
        not isinstance(module, LoRALinear)
        or module.rank != first.rank
        or module.alpha != first.alpha
        or module.dora != first.dora
        for module in wrappers
    ):
        raise ValueError("one capability artifact requires one adapter recipe")
    state = adapter_state_dict(model)
    spec = CapabilityAdapterSpec(
        capability_id=str(capability_id),
        kind="dora" if first.dora else "lora",
        base_model_profile=str(base_model_profile),
        base_checkpoint_sha256=_validated_hash(
            base_checkpoint_sha256, "base checkpoint hash"
        ),
        tokenizer_sha256=_validated_hash(tokenizer_sha256, "tokenizer hash"),
        rank=first.rank,
        alpha=first.alpha,
        target_modules=modules,
        trainable_parameters=sum(value.numel() for value in state.values()),
        source_commit=str(source_commit or "unknown"),
        created_at=time.time(),
    )
    spec.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        {
            "schema_version": CAPABILITY_ADAPTER_SCHEMA_VERSION,
            "spec": asdict(spec),
            "state_dict": state,
        },
        temporary,
    )
    temporary.replace(target)
    manifest: dict[str, object] = {
        "schema_version": CAPABILITY_ADAPTER_SCHEMA_VERSION,
        "artifact": str(target),
        "artifact_sha256": sha256_file(target),
        "spec": asdict(spec),
        "state_shapes": {name: list(value.shape) for name, value in sorted(state.items())},
    }
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    return manifest


def load_capability_adapter(
    model: nn.Module,
    path: str | Path,
    *,
    expected_base_model_profile: str,
    expected_base_checkpoint_sha256: str,
    expected_tokenizer_sha256: str,
) -> CapabilityAdapterSpec:
    target = Path(path)
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    if not target.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("capability artifact and manifest are both required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_sha256") != sha256_file(target):
        raise ValueError("capability artifact hash does not match its manifest")
    payload = torch.load(target, map_location="cpu", weights_only=True)
    if int(payload.get("schema_version", 0)) != CAPABILITY_ADAPTER_SCHEMA_VERSION:
        raise ValueError("unsupported capability adapter schema")
    raw_spec = payload.get("spec")
    if not isinstance(raw_spec, dict):
        raise ValueError("capability adapter has no typed spec")
    spec = CapabilityAdapterSpec(
        **{
            **raw_spec,
            "target_modules": tuple(raw_spec.get("target_modules", ())),
        }
    )
    spec.validate()
    if spec.base_model_profile != expected_base_model_profile:
        raise ValueError("capability adapter model profile mismatch")
    if spec.base_checkpoint_sha256 != _validated_hash(
        expected_base_checkpoint_sha256, "expected base checkpoint hash"
    ):
        raise ValueError("capability adapter base checkpoint mismatch")
    if spec.tokenizer_sha256 != _validated_hash(
        expected_tokenizer_sha256, "expected tokenizer hash"
    ):
        raise ValueError("capability adapter tokenizer mismatch")
    manifest_spec = manifest.get("spec")
    if isinstance(manifest_spec, dict):
        manifest_spec = {
            **manifest_spec,
            "target_modules": tuple(manifest_spec.get("target_modules", ())),
        }
    if manifest_spec != asdict(spec):
        raise ValueError("capability artifact spec differs from its manifest")

    detach_candidate_adapters(model)
    attach_candidate_adapters(
        model,
        rank=spec.rank,
        alpha=spec.alpha,
        dora=spec.kind == "dora",
        target_modules=spec.target_modules,
    )
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        detach_candidate_adapters(model)
        raise ValueError("capability adapter has no state dictionary")
    try:
        load_adapter_state(model, state)
    except Exception:
        detach_candidate_adapters(model)
        raise
    return spec
