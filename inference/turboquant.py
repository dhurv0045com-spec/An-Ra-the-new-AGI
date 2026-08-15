"""Device-resident, bit-packed TurboQuant KV-cache pilot.

This is an evidence-gated inference pilot inspired by TurboQuant
(arXiv:2504.19874), not a claim of reproducing the paper's fused QJL attention
kernel.  It implements the part An-Ra can validate honestly today:

* deterministic randomized Walsh-Hadamard rotation;
* online normal-approximation Lloyd-Max scalar quantization;
* real 4-bit nibble packing (or 8-bit storage);
* FP16 vector norms and bounded, device-resident cache buffers;
* measured distortion, physical bytes, and compression ratio;
* transparent dequantization before PyTorch SDPA.

The paper's inner-product-optimal path applies QJL to the residual inside the
attention estimator.  The retired implementation incorrectly treated QJL sign
bits as a vector-space correction and stored every "4-bit" code in a full byte.
This pilot deliberately omits QJL until An-Ra has a fused query-aware kernel.

Primary reference: https://arxiv.org/abs/2504.19874
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch


class TurboQuantError(RuntimeError):
    """Raised when the compressed cache contract cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class TurboQuantConfig:
    """Configuration for the reversible KV-cache pilot."""

    bits: Literal[4, 8] = 4
    seed: int = 1301
    scale_dtype: torch.dtype = torch.float16
    minimum_compression_ratio: float = 3.0

    def __post_init__(self) -> None:
        if self.bits not in {4, 8}:
            raise ValueError("TurboQuant pilot supports only packed 4-bit or 8-bit codes")
        if self.seed < 0:
            raise ValueError("TurboQuant seed must be non-negative")
        if self.scale_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("TurboQuant scale dtype must be a floating-point storage dtype")
        if self.minimum_compression_ratio <= 1.0:
            raise ValueError("minimum_compression_ratio must exceed 1")

    @property
    def levels(self) -> int:
        return 1 << self.bits


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _fwht(tensor: torch.Tensor) -> torch.Tensor:
    """Normalized fast Walsh-Hadamard transform over the final dimension."""

    width = int(tensor.shape[-1])
    if not _is_power_of_two(width):
        raise TurboQuantError(
            f"Walsh-Hadamard rotation requires a power-of-two head dimension, got {width}"
        )
    output = tensor
    block = 1
    while block < width:
        shaped = output.reshape(*output.shape[:-1], -1, block * 2)
        left = shaped[..., :block]
        right = shaped[..., block:]
        output = torch.cat((left + right, left - right), dim=-1).reshape_as(output)
        block *= 2
    return output / math.sqrt(width)


@lru_cache(maxsize=2)
def _normal_lloyd_max(levels: int, *, iterations: int = 80) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic scalar Lloyd-Max codebook for N(0, 1).

    Random rotation makes individual coordinates close to normal in high
    dimensions.  The paper uses dimension-aware Beta codebooks; the normal
    approximation is explicit pilot debt and is recorded in telemetry.
    """

    dtype = torch.float64
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )
    probabilities = (torch.arange(levels, dtype=dtype) + 0.5) / levels
    centroids = normal.icdf(probabilities)
    sqrt_two_pi = math.sqrt(2.0 * math.pi)

    for _ in range(iterations):
        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        lower = torch.cat(
            (torch.tensor([-torch.inf], dtype=dtype), boundaries),
        )
        upper = torch.cat(
            (boundaries, torch.tensor([torch.inf], dtype=dtype)),
        )
        lower_cdf = normal.cdf(lower)
        upper_cdf = normal.cdf(upper)
        lower_pdf = torch.where(
            torch.isfinite(lower),
            torch.exp(-0.5 * lower.square()) / sqrt_two_pi,
            torch.zeros_like(lower),
        )
        upper_pdf = torch.where(
            torch.isfinite(upper),
            torch.exp(-0.5 * upper.square()) / sqrt_two_pi,
            torch.zeros_like(upper),
        )
        mass = (upper_cdf - lower_cdf).clamp_min(torch.finfo(dtype).eps)
        updated = (lower_pdf - upper_pdf) / mass
        if torch.max(torch.abs(updated - centroids)).item() < 1e-12:
            centroids = updated
            break
        centroids = updated

    boundaries = (centroids[:-1] + centroids[1:]) * 0.5
    return centroids.float(), boundaries.float()


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype is not torch.uint8:
        raise TypeError("4-bit packing requires uint8 code indices")
    width = int(codes.shape[-1])
    if width % 2:
        codes = torch.nn.functional.pad(codes, (0, 1))
    low = codes[..., 0::2]
    high = codes[..., 1::2]
    return low | (high << 4)


def _unpack_nibbles(packed: torch.Tensor, width: int) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack((low, high), dim=-1).flatten(-2)[..., :width]


class TorchTurboQuantCache:
    """Compressed per-layer K/V cache used by An-Ra's actual attention path."""

    algorithm = "turboquant-pilot/hadamard-lloyd-max-v1"

    def __init__(
        self,
        *,
        num_kv_heads: int,
        max_seq_len: int,
        d_head: int,
        config: TurboQuantConfig | None = None,
    ) -> None:
        if num_kv_heads <= 0 or max_seq_len <= 0 or d_head <= 0:
            raise ValueError(
                "KV heads, maximum sequence length, and head dimension must be positive"
            )
        if not _is_power_of_two(d_head):
            raise ValueError("TurboQuant pilot requires a power-of-two head dimension")
        self.num_kv_heads = int(num_kv_heads)
        self.max_seq_len = int(max_seq_len)
        self.d_head = int(d_head)
        self.config = config or TurboQuantConfig()
        self.current_len = 0
        self.total_tokens_seen = 0
        self._batch_size: int | None = None
        self._device: torch.device | None = None
        self._k_codes: torch.Tensor | None = None
        self._v_codes: torch.Tensor | None = None
        self._k_norms: torch.Tensor | None = None
        self._v_norms: torch.Tensor | None = None
        self._codebook_cpu, self._boundaries_cpu = _normal_lloyd_max(
            self.config.levels
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        signs = torch.randint(0, 2, (self.d_head,), generator=generator)
        self._signs_cpu = signs.mul(2).sub(1).to(torch.float32)
        self._last_relative_mse = 0.0
        self._max_relative_mse = 0.0
        self._updates = 0
        self._source_dtype_bytes = 2

    @property
    def position(self) -> int:
        return self.total_tokens_seen

    @property
    def packed_width(self) -> int:
        return (self.d_head + 1) // 2 if self.config.bits == 4 else self.d_head

    def _ensure_storage(self, tensor: torch.Tensor) -> None:
        batch, heads, _tokens, width = tensor.shape
        if heads != self.num_kv_heads or width != self.d_head:
            raise TurboQuantError(
                "KV tensor geometry does not match the compressed-cache contract"
            )
        if self._batch_size is not None:
            if batch != self._batch_size or tensor.device != self._device:
                raise TurboQuantError(
                    "TurboQuant cache cannot change batch size or device without reset"
                )
            return
        self._batch_size = int(batch)
        self._device = tensor.device
        code_shape = (
            batch,
            self.num_kv_heads,
            self.max_seq_len,
            self.packed_width,
        )
        norm_shape = (batch, self.num_kv_heads, self.max_seq_len, 1)
        self._k_codes = torch.empty(code_shape, dtype=torch.uint8, device=tensor.device)
        self._v_codes = torch.empty_like(self._k_codes)
        self._k_norms = torch.empty(
            norm_shape,
            dtype=self.config.scale_dtype,
            device=tensor.device,
        )
        self._v_norms = torch.empty_like(self._k_norms)

    def _rotation_material(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signs = self._signs_cpu.to(device=tensor.device, dtype=torch.float32)
        codebook = self._codebook_cpu.to(device=tensor.device)
        boundaries = self._boundaries_cpu.to(device=tensor.device)
        return signs, codebook, boundaries

    def _quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        signs, codebook, boundaries = self._rotation_material(tensor)
        working = tensor.detach().to(torch.float32)
        norms = torch.linalg.vector_norm(working, dim=-1, keepdim=True).clamp_min(1e-8)
        rotated = _fwht(working * signs)
        standardized = rotated * math.sqrt(self.d_head) / norms
        codes = torch.bucketize(standardized, boundaries).to(torch.uint8)
        packed = _pack_nibbles(codes) if self.config.bits == 4 else codes

        stored_norms = norms.to(self.config.scale_dtype)
        reconstructed_rotated = (
            codebook[codes.long()]
            * stored_norms.to(torch.float32)
            / math.sqrt(self.d_head)
        )
        reconstructed = _fwht(reconstructed_rotated) * signs
        error = (working - reconstructed).square().sum(dim=-1)
        denominator = working.square().sum(dim=-1).clamp_min(1e-12)
        relative_mse = float((error / denominator).mean().item())
        return packed, stored_norms, relative_mse

    def _dequantize(
        self,
        packed: torch.Tensor,
        norms: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        signs, codebook, _boundaries = self._rotation_material(norms)
        codes = (
            _unpack_nibbles(packed, self.d_head)
            if self.config.bits == 4
            else packed
        )
        rotated = (
            codebook[codes.long()]
            * norms.to(torch.float32)
            / math.sqrt(self.d_head)
        )
        return (_fwht(rotated) * signs).to(dtype)

    @staticmethod
    def _shift_left(tensor: torch.Tensor, *, used: int, amount: int) -> None:
        remaining = max(0, used - amount)
        if remaining:
            tensor[:, :, :remaining].copy_(tensor[:, :, amount:used].clone())

    def _make_room(self, new_tokens: int) -> None:
        overflow = max(0, self.current_len + new_tokens - self.max_seq_len)
        if overflow <= 0:
            return
        if overflow >= self.current_len:
            self.current_len = 0
            return
        assert self._k_codes is not None
        assert self._v_codes is not None
        assert self._k_norms is not None
        assert self._v_norms is not None
        for tensor in (self._k_codes, self._v_codes, self._k_norms, self._v_norms):
            self._shift_left(tensor, used=self.current_len, amount=overflow)
        self.current_len -= overflow

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key.shape != value.shape or key.ndim != 4:
            raise TurboQuantError("K and V must have identical [batch, heads, tokens, dim] shape")
        self._ensure_storage(key)
        tokens_seen = int(key.shape[2])
        if key.shape[2] > self.max_seq_len:
            key = key[:, :, -self.max_seq_len :, :]
            value = value[:, :, -self.max_seq_len :, :]
        new_tokens = int(key.shape[2])
        self._make_room(new_tokens)
        start = self.current_len
        end = start + new_tokens
        k_codes, k_norms, k_error = self._quantize(key)
        v_codes, v_norms, v_error = self._quantize(value)
        assert self._k_codes is not None
        assert self._v_codes is not None
        assert self._k_norms is not None
        assert self._v_norms is not None
        self._k_codes[:, :, start:end].copy_(k_codes)
        self._v_codes[:, :, start:end].copy_(v_codes)
        self._k_norms[:, :, start:end].copy_(k_norms)
        self._v_norms[:, :, start:end].copy_(v_norms)
        self.current_len = end
        # RoPE position is absolute even when the bounded cache evicts history.
        self.total_tokens_seen += tokens_seen
        self._source_dtype_bytes = key.element_size()
        self._last_relative_mse = (k_error + v_error) * 0.5
        self._max_relative_mse = max(
            self._max_relative_mse,
            self._last_relative_mse,
        )
        self._updates += 1
        return (
            self._dequantize(
                self._k_codes[:, :, :end],
                self._k_norms[:, :, :end],
                dtype=key.dtype,
            ),
            self._dequantize(
                self._v_codes[:, :, :end],
                self._v_norms[:, :, :end],
                dtype=value.dtype,
            ),
        )

    def reset(self) -> None:
        self.current_len = 0
        self.total_tokens_seen = 0
        self._last_relative_mse = 0.0
        self._max_relative_mse = 0.0
        self._updates = 0

    def memory_report(self) -> dict[str, object]:
        batch = int(self._batch_size or 0)
        tokens = self.current_len
        bytes_per_stored_vector = self.packed_width + self.config.scale_dtype.itemsize
        occupied_compressed = (
            2 * batch * self.num_kv_heads * tokens * bytes_per_stored_vector
        )
        occupied_uncompressed = (
            2
            * batch
            * self.num_kv_heads
            * tokens
            * self.d_head
            * self._source_dtype_bytes
        )
        allocated_compressed = (
            2
            * batch
            * self.num_kv_heads
            * self.max_seq_len
            * bytes_per_stored_vector
        )
        equivalent_uncompressed_capacity = (
            2
            * batch
            * self.num_kv_heads
            * self.max_seq_len
            * self.d_head
            * self._source_dtype_bytes
        )
        ratio = (
            equivalent_uncompressed_capacity / allocated_compressed
            if allocated_compressed
            else 0.0
        )
        return {
            "algorithm": self.algorithm,
            "paper_complete": False,
            "qjl_fused": False,
            "bits": self.config.bits,
            "tokens_cached": tokens,
            "cache_capacity_tokens": self.max_seq_len,
            "compressed_bytes": allocated_compressed,
            "uncompressed_bytes": equivalent_uncompressed_capacity,
            "occupied_compressed_bytes": occupied_compressed,
            "occupied_uncompressed_bytes": occupied_uncompressed,
            "compression_ratio": ratio,
            "memory_saved_bytes": max(
                0,
                equivalent_uncompressed_capacity - allocated_compressed,
            ),
            "last_relative_mse": self._last_relative_mse,
            "max_relative_mse": self._max_relative_mse,
            "updates": self._updates,
        }


# Compatibility name used by the component registry and older import surfaces.
CompressedKVCache = TorchTurboQuantCache


def health_check() -> dict[str, object]:
    """Run a real bit-packing and round-trip health probe."""

    try:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1301)
        cache = TorchTurboQuantCache(
            num_kv_heads=2,
            max_seq_len=32,
            d_head=64,
            config=TurboQuantConfig(bits=4),
        )
        key = torch.randn(1, 2, 16, 64, generator=generator, dtype=torch.float16)
        value = torch.randn(1, 2, 16, 64, generator=generator, dtype=torch.float16)
        output_key, output_value = cache.update(key, value)
        report = cache.memory_report()
        healthy = (
            output_key.shape == key.shape
            and output_value.shape == value.shape
            and float(report["compression_ratio"]) >= 3.0
            and float(report["last_relative_mse"]) < 0.08
        )
        return {
            "status": "ok" if healthy else "degraded",
            "module": "turboquant",
            **report,
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "module": "turboquant",
            "reason": f"{type(exc).__name__}: {exc}",
        }
