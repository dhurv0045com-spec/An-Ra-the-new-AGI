from __future__ import annotations

import hashlib
import json
from typing import Any

from v5_experiments import x_factor_pilot_protocol_v1 as protocol


def _factor_ids(token_ids: Any, *, base: int, components: int, torch: Any) -> Any:

    values = token_ids.long()
    digits: list[Any] = []
    for _ in range(components):
        digits.append(values.remainder(base))
        values = torch.div(values, base, rounding_mode="floor")
    if bool((values != 0).any().item()):
        raise ValueError("token id exceeds the factorized code capacity")
    return torch.stack(digits, dim=-1)


def _define_classes(torch: Any) -> tuple[Any, Any, Any]:
    class DenseInput(torch.nn.Module):
        def __init__(self, vocabulary: int, width: int) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(vocabulary, width)

        def forward(self, token_ids: Any) -> Any:
            return self.embedding(token_ids)

    class LeviathanInput(torch.nn.Module):
        def __init__(self, vocabulary: int, width: int, base: int, components: int, seed_width: int, heads: int) -> None:
            super().__init__()
            if base ** components < vocabulary:
                raise ValueError("factorized code capacity is smaller than the vocabulary")
            self.vocabulary = int(vocabulary)
            self.width = int(width)
            self.base = int(base)
            self.components = int(components)
            self.seed_width = int(seed_width)
            self.heads = int(heads)
            self.codebooks = torch.nn.ModuleList(
                [torch.nn.Embedding(base, seed_width) for _ in range(components)]
            )
            self.input_norm = torch.nn.LayerNorm(seed_width)
            self.head_seed = torch.nn.ModuleList(
                [torch.nn.Linear(seed_width, seed_width, bias=False) for _ in range(heads)]
            )
            self.head_output = torch.nn.ModuleList(
                [torch.nn.Linear(seed_width, width, bias=False) for _ in range(heads)]
            )

        def forward(self, token_ids: Any) -> Any:
            digits = _factor_ids(
                token_ids,
                base=self.base,
                components=self.components,
                torch=torch,
            )
            seed = None
            for index, codebook in enumerate(self.codebooks):
                component = codebook(digits[..., index])
                seed = component if seed is None else seed + component
            seed = seed / (float(self.components) ** 0.5)
            seed = self.input_norm(seed)
            output = None
            for seed_layer, output_layer in zip(self.head_seed, self.head_output):
                candidate = output_layer(torch.nn.functional.silu(seed_layer(seed)))
                output = candidate if output is None else output + candidate
            return output

    class XFactorModel(torch.nn.Module):
        def __init__(self, arm: str, seed: int, *, torch_module: Any) -> None:
            super().__init__()
            if arm not in protocol.ARMS:
                raise ValueError(f"unknown arm: {arm}")
            self.arm = arm
            self.seed = int(seed)
            self.vocabulary = protocol.PHYSICAL_VOCAB
            self.width = protocol.WIDTH
            self.context_length = protocol.CONTEXT_LENGTH
            self.input_mode = "dense" if arm != "LEV_UNTIED" else "lev_factorized_continuous"
            self.output_mode = "tied" if arm == "TIED_DENSE" else "independent"
            from v5_model.config import ModelConfig
            from v5_model.block import build_block, build_rmsnorm

            config = ModelConfig(
                vocabulary_size=protocol.PHYSICAL_VOCAB,
                width=protocol.WIDTH,
                layers=protocol.LAYERS,
                query_heads=protocol.QUERY_HEADS,
                kv_heads=protocol.KV_HEADS,
                head_dimension=protocol.HEAD_DIMENSION,
                ffn_width=protocol.FFN_WIDTH,
                context_length=protocol.CONTEXT_LENGTH,
                rope_base=10_000.0,
                norm_epsilon=1e-5,
                qk_norm=True,
                qk_norm_affine=True,
                qk_norm_epsilon=1e-6,
            )
            self.config = config
            if arm == "LEV_UNTIED":
                self.input_layer = LeviathanInput(
                    protocol.PHYSICAL_VOCAB,
                    protocol.WIDTH,
                    base=protocol_payload()["base"],
                    components=protocol_payload()["components"],
                    seed_width=protocol_payload()["seed_width"],
                    heads=protocol_payload()["heads"],
                )
            else:
                self.input_layer = DenseInput(protocol.PHYSICAL_VOCAB, protocol.WIDTH)
            self.blocks = torch_module.nn.ModuleList(
                [build_block(config, torch_module=torch_module) for _ in range(config.layers)]
            )
            self.final_norm = build_rmsnorm(
                config.width, epsilon=config.norm_epsilon, torch_module=torch_module
            )
            if arm == "TIED_DENSE":
                self.output_head = None
            else:
                self.output_head = torch_module.nn.Linear(
                    config.width, config.vocabulary_size, bias=False
                )
            self._reference_state = None

        @property
        def embedding(self) -> Any:
            return self.input_layer.embedding if self.arm != "LEV_UNTIED" else None

        def load_reference(self, state: dict[str, Any], dense_weight: Any) -> None:
            backbone = {
                key: value for key, value in state.items() if not key.startswith("embedding.")
            }
            missing, unexpected = self.load_state_dict(backbone, strict=False)
            if unexpected:
                raise ValueError(f"unexpected reference parameters: {unexpected}")
            expected_missing = {
                key for key in self.state_dict() if key.startswith(("input_layer.", "output_head."))
            }
            if set(missing) != expected_missing:
                raise ValueError(f"reference parameter mismatch: missing={missing}")
            if self.embedding is not None:
                with torch.no_grad():
                    self.embedding.weight.copy_(dense_weight)
            if self.output_head is not None:
                with torch.no_grad():
                    self.output_head.weight.copy_(dense_weight)

        def forward(self, token_ids: Any, positions: Any, mask: Any, use_activation_checkpointing: bool = False) -> Any:
            if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.context_length:
                raise ValueError("token ids must be [batch, length] within native context")
            hidden = self.input_layer(token_ids)
            for block in self.blocks:
                if use_activation_checkpointing and self.training:
                    hidden = torch.utils.checkpoint.checkpoint(
                        block, hidden, positions, mask, use_reentrant=False
                    )
                else:
                    hidden = block(hidden, positions, mask)
            hidden = self.final_norm(hidden)
            weight = self.embedding.weight if self.output_head is None else self.output_head.weight
            return torch.nn.functional.linear(hidden, weight)

    return DenseInput, LeviathanInput, XFactorModel


def protocol_payload() -> dict[str, Any]:
    return protocol.protocol_payload()["x_factor"]


def build_model(arm: str, seed: int, *, torch_module: Any, device: Any) -> Any:
    from anra_v5 import formation_mux_model_v2 as legacy
    from v5_model.core import initialize

    DenseInput, LeviathanInput, XFactorModel = _define_classes(torch_module)
    reference = initialize(legacy.spec(), int(seed), torch_module=torch_module)
    reference_state = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    dense_weight = reference_state["embedding.weight"].detach().clone()
    with torch_module.random.fork_rng():
        torch_module.manual_seed(int(seed) + 0x5EED)
        model = XFactorModel(arm, int(seed), torch_module=torch_module)
        model.load_reference(reference_state, dense_weight)
        if arm == "LEV_UNTIED":
            for name, parameter in model.input_layer.named_parameters():
                if parameter.ndim >= 2:
                    torch_module.nn.init.normal_(parameter, mean=0.0, std=0.02)
                elif name.endswith("bias"):
                    torch_module.nn.init.zeros_(parameter)
                else:
                    torch_module.nn.init.ones_(parameter)
            for layer in model.input_layer.head_output:
                torch_module.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            with torch_module.no_grad():
                sample = torch_module.arange(4_096, dtype=torch_module.long)
                current_scale = model.input_layer(sample).float().std().clamp_min(1e-6)
                scale = 0.02 / current_scale
                for layer in model.input_layer.head_output:
                    layer.weight.mul_(scale)
    model.to(device)
    assert_parameterization(model)
    return model


def parameterization_receipt(model: Any) -> dict[str, Any]:
    counts = {name: int(parameter.numel()) for name, parameter in model.named_parameters()}
    total = sum(counts.values())
    if model.arm == "TIED_DENSE":
        if model.output_head is not None or model.embedding is None:
            raise ValueError("tied arm storage invariant failed")
        output_alias = True
    else:
        if model.output_head is None:
            raise ValueError("untied arm output head missing")
        if model.arm == "UNTIED_DENSE" and model.output_head.weight.data_ptr() == model.embedding.weight.data_ptr():
            raise ValueError("untied dense storage is aliased")
        output_alias = False
    return {
        "schema": "anra.x-factor-parameterization/v1",
        "arm": model.arm,
        "input_mode": model.input_mode,
        "output_mode": model.output_mode,
        "parameter_names": counts,
        "parameter_count": total,
        "trainable_parameter_count": total,
        "output_alias": output_alias,
        "physical_vocabulary": int(model.vocabulary),
        "width": int(model.width),
        "protocol_sha256": protocol.protocol_sha256(),
    }


def assert_parameterization(model: Any) -> None:
    receipt = parameterization_receipt(model)
    if receipt["physical_vocabulary"] != protocol.PHYSICAL_VOCAB:
        raise ValueError("physical vocabulary changed")
    if model.arm == "TIED_DENSE":
        if model.embedding.weight.shape != (protocol.PHYSICAL_VOCAB, protocol.WIDTH):
            raise ValueError("tied embedding shape mismatch")
    elif model.arm == "UNTIED_DENSE":
        if model.embedding.weight.shape != (protocol.PHYSICAL_VOCAB, protocol.WIDTH):
            raise ValueError("dense input shape mismatch")
        if model.output_head.weight.shape != (protocol.PHYSICAL_VOCAB, protocol.WIDTH):
            raise ValueError("untied output shape mismatch")
    else:
        if model.embedding is not None or len(model.input_layer.codebooks) != 3:
            raise ValueError("Leviathan input invariant failed")
        if model.output_head.weight.shape != (protocol.PHYSICAL_VOCAB, protocol.WIDTH):
            raise ValueError("Leviathan output shape mismatch")


def model_state_sha(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def architecture_id(arm: str) -> str:
    payload = {
        "protocol_sha256": protocol.protocol_sha256(),
        "arm": arm,
        "parameterization": protocol.arm_parameterization(arm),
        "model": protocol.protocol_payload()["model"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
