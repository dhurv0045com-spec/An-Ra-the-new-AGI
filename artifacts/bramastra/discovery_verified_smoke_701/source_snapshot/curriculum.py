"""Training-only demonstrations; exact teacher never runs inside evaluation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
import random

import numpy as np
import torch

from .worlds import RuleWorld, inputs


_TENSOR_FIELDS = ("observations", "lengths", "targets", "labels", "legal", "gains")


def demonstration_digest(tensors) -> str:
    """Hash the names, shapes, dtypes, and bytes of all training tensors."""
    digest = hashlib.sha256()
    for name, tensor in zip(_TENSOR_FIELDS, tensors):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def teacher_gains(table: np.ndarray, target: int, legal: np.ndarray) -> np.ndarray:
    """Expected target entropy reduction under a uniform compatible prior."""
    if table.ndim != 2 or not len(table):
        raise ValueError("teacher posterior must contain a world")
    values = table.astype(np.float64, copy=False)
    pt = values[:, target].mean()
    pa = values.mean(axis=0)
    joint = (values * values[:, target, None]).mean(axis=0)
    given_one = np.divide(joint, pa, out=np.zeros_like(pa), where=pa > 0)
    given_zero = np.divide(pt - joint, 1 - pa, out=np.zeros_like(pa), where=pa < 1)
    conditional = pa * binary_entropy(given_one) + (1 - pa) * binary_entropy(given_zero)
    gains = np.maximum(0, binary_entropy(np.asarray(pt)) - conditional)
    return np.where(legal, gains, 0).astype(np.float32)


@dataclass
class Demonstrations:
    observations: torch.Tensor
    lengths: torch.Tensor
    targets: torch.Tensor
    labels: torch.Tensor
    legal: torch.Tensor
    gains: torch.Tensor
    manifest: dict

    def __len__(self) -> int:
        return len(self.labels)


def build_demonstrations(worlds: list[RuleWorld], *, episodes: int, budget: int,
                         seed: int, teacher_fraction: float = 0.5) -> Demonstrations:
    if not worlds or episodes <= 0 or budget < 0 or not 0 <= teacher_fraction <= 1:
        raise ValueError("invalid demonstration configuration")
    bits = worlds[0].bits
    candidates = np.asarray(inputs(bits), dtype=np.float32)
    if budget > len(candidates) - 2:
        raise ValueError("budget must leave a legal action for all training states")
    if any(w.bits != bits for w in worlds):
        raise ValueError("mixed world widths")
    table = np.asarray([w.table for w in worlds], dtype=np.uint8)
    rng = random.Random(seed)
    obs, lengths, targets, labels, masks, all_gains = [], [], [], [], [], []
    for _ in range(episodes):
        world_index = rng.randrange(len(worlds))
        truth = table[world_index]
        target = rng.randrange(len(candidates))
        legal = np.ones(len(candidates), dtype=bool)
        legal[target] = False
        compatible = np.ones(len(worlds), dtype=bool)
        history = np.zeros((budget, bits + 1), dtype=np.float32)
        for t in range(budget + 1):
            gains = teacher_gains(table[compatible], target, legal)
            obs.append(history.copy())
            lengths.append(t)
            targets.append(candidates[target])
            labels.append(truth[target])
            masks.append(legal.copy())
            all_gains.append(gains)
            if t == budget:
                break
            if rng.random() < teacher_fraction and gains.max() > 1e-7:
                choices = np.flatnonzero(legal & np.isclose(gains, gains.max(), atol=1e-7)).tolist()
            else:
                choices = np.flatnonzero(legal).tolist()
            action = rng.choice(choices)
            label = truth[action]
            history[t] = np.concatenate((candidates[action], [label]))
            legal[action] = False
            compatible &= table[:, action] == label
    identity = hashlib.sha256("\n".join(w.world_id for w in worlds).encode()).hexdigest()
    raw = [torch.from_numpy(np.asarray(x)) for x in (obs, lengths, targets, labels, masks, all_gains)]
    tensors = (raw[0].float(), raw[1].long(), raw[2].float(), raw[3].float(), raw[4].bool(), raw[5].float())
    manifest = {"world_ids_sha256": identity, "worlds": len(worlds), "episodes": episodes,
                "states": len(labels), "budget": budget, "seed": seed,
                "teacher_fraction": teacher_fraction, "teacher_scope": "training worlds only",
                "tensor_sha256": demonstration_digest(tensors)}
    return Demonstrations(*tensors, manifest)
