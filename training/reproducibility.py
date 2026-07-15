"""Canonical randomness and deterministic-resume contract for An-Ra training.

A seed is an address for one stochastic run, not a quality setting.  This
module makes that address useful by binding every RNG owned by the foundation
trainer and by serializing their states at optimizer boundaries.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

CANONICAL_TRAINING_SEED = 1301
RNG_CONTRACT_VERSION = 1
DETERMINISM_MODE = "reproducible_same_stack_v1"


def validate_seed(seed: int) -> int:
    value = int(seed)
    if value < 0 or value > 2**32 - 1:
        raise ValueError("training seed must be in [0, 2**32-1]")
    return value


@dataclass(frozen=True)
class SeedReport:
    seed: int
    contract_version: int
    determinism_mode: str
    python_hash_seed: str
    python_hash_seed_matches: bool
    deterministic_algorithms: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cuda_seeded: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def seed_everything(seed: int = CANONICAL_TRAINING_SEED) -> SeedReport:
    """Seed the complete foundation stack and select reproducible CUDA policy.

    ``PYTHONHASHSEED`` only takes effect at interpreter startup.  The unified
    launcher exports it before starting the phase trainer; direct invocations
    still record whether the current process inherited the right value.
    """

    value = validate_seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(value))
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    cuda_seeded = bool(torch.cuda.is_available())
    if cuda_seeded:
        torch.cuda.manual_seed_all(value)

    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    hash_seed = os.environ.get("PYTHONHASHSEED", "")
    return SeedReport(
        seed=value,
        contract_version=RNG_CONTRACT_VERSION,
        determinism_mode=DETERMINISM_MODE,
        python_hash_seed=hash_seed,
        python_hash_seed_matches=hash_seed == str(value),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=bool(getattr(torch.backends.cudnn, "benchmark", False)),
        cudnn_deterministic=bool(getattr(torch.backends.cudnn, "deterministic", False)),
        cuda_seeded=cuda_seeded,
    )


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""

    del worker_id
    worker_seed = int(torch.initial_seed() % 2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_data_generator(seed: int = CANONICAL_TRAINING_SEED) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(validate_seed(seed))
    return generator


def _numpy_state_to_payload(state: tuple[Any, ...]) -> dict[str, object]:
    return {
        "bit_generator": str(state[0]),
        "keys": np.asarray(state[1], dtype=np.uint32).tolist(),
        "position": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _numpy_state_from_payload(payload: object) -> tuple[Any, ...]:
    if not isinstance(payload, dict):
        raise TypeError("NumPy RNG state must be a mapping")
    return (
        str(payload["bit_generator"]),
        np.asarray(payload["keys"], dtype=np.uint32),
        int(payload["position"]),
        int(payload["has_gauss"]),
        float(payload["cached_gaussian"]),
    )


def capture_rng_states(
    *, data_generator: torch.Generator | None = None
) -> dict[str, object]:
    """Capture a weights-only-safe optimizer-boundary RNG snapshot."""

    return {
        "contract_version": RNG_CONTRACT_VERSION,
        "determinism_mode": DETERMINISM_MODE,
        "python": random.getstate(),
        "numpy": _numpy_state_to_payload(np.random.get_state()),
        "torch": torch.get_rng_state().cpu(),
        "cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
        "data_generator": data_generator.get_state().cpu()
        if data_generator is not None
        else None,
    }


def restore_rng_states(
    payload: object,
    *,
    data_generator: torch.Generator | None = None,
) -> dict[str, object]:
    """Restore a complete snapshot or fail instead of silently reseeding."""

    if not isinstance(payload, dict):
        raise TypeError("checkpoint RNG state must be a mapping")
    version = int(payload.get("contract_version", 0))
    if version != RNG_CONTRACT_VERSION:
        raise ValueError(
            f"unsupported RNG contract version {version}; expected {RNG_CONTRACT_VERSION}"
        )
    if str(payload.get("determinism_mode", "")) != DETERMINISM_MODE:
        raise ValueError("checkpoint determinism mode differs from the active trainer")

    python_state = payload.get("python")
    torch_state = payload.get("torch")
    if not isinstance(python_state, tuple) or not isinstance(torch_state, torch.Tensor):
        raise TypeError("checkpoint is missing Python or Torch RNG state")
    random.setstate(python_state)
    np.random.set_state(_numpy_state_from_payload(payload.get("numpy")))
    torch.set_rng_state(torch_state.cpu())

    cuda_states = payload.get("cuda", [])
    if torch.cuda.is_available():
        if not isinstance(cuda_states, list) or not cuda_states:
            raise ValueError("CUDA resume requires checkpointed CUDA RNG states")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])

    generator_state = payload.get("data_generator")
    if data_generator is not None:
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("checkpoint is missing the DataLoader generator state")
        data_generator.set_state(generator_state.cpu())

    return {
        "contract_version": version,
        "determinism_mode": DETERMINISM_MODE,
        "python": True,
        "numpy": True,
        "torch": True,
        "cuda": bool(torch.cuda.is_available()),
        "data_generator": data_generator is not None,
    }
