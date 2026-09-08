"""Matched one-step and exact depth-two teaching data.

The exact teacher is privileged and training-only.  The adapter deliberately
uses the discovery ``Demonstrations`` contract so both arms run through the
same model and trainer implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

import numpy as np
import torch

from bramastra_lab.discovery.curriculum import (
    Demonstrations,
    demonstration_digest,
    teacher_gains,
)
from bramastra_lab.discovery.worlds import RuleWorld, inputs


def depth_two_scores(
    table: np.ndarray,
    target: int,
    legal: np.ndarray,
    *,
    query_cost: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Return exact Q2 scores and the number of evaluated posterior branches.

    Q2(a) is immediate target-entropy gain less the first query cost, plus the
    expected best nonnegative net gain of one further legal query.  Allowing a
    zero-valued stop option makes positive query cost explicit.  Illegal,
    target, and already-used actions are represented by zero and never searched.
    """
    if table.ndim != 2 or not len(table):
        raise ValueError("teacher posterior must contain a world")
    if not isinstance(target, (int, np.integer)) or isinstance(target, (bool, np.bool_)):
        raise ValueError("target must be an integer action index")
    if target < 0 or target >= table.shape[1]:
        raise ValueError("target is out of range")
    if legal.shape != (table.shape[1],) or legal.dtype != np.bool_:
        raise ValueError("legal mask shape mismatch")
    if legal[target]:
        raise ValueError("scored target must be explicitly masked")
    if not np.isfinite(query_cost) or query_cost < 0:
        raise ValueError("query_cost must be finite and nonnegative")
    one = teacher_gains(table, target, legal).astype(np.float64)
    scores = np.zeros(table.shape[1], dtype=np.float64)
    branches = 0
    for action in np.flatnonzero(legal):
        future = 0.0
        for observation in (0, 1):
            branch = table[table[:, action] == observation]
            if not len(branch):
                continue
            next_legal = legal.copy()
            next_legal[action] = False
            second = teacher_gains(branch, target, next_legal).astype(np.float64)
            best_net = max(0.0, float(second[next_legal].max()) - query_cost) if next_legal.any() else 0.0
            future += len(branch) / len(table) * best_net
            branches += 1
        scores[action] = max(0.0, float(one[action]) - query_cost + future)
    return scores.astype(np.float32), branches


def stratified_semantic_split(worlds: list[RuleWorld]) -> dict[str, list[RuleWorld]]:
    """Stable family-stratified 60/20/20 split with nonempty feasible groups."""
    if not worlds:
        raise ValueError("worlds cannot be empty")
    result = {"train": [], "dev": [], "test": []}
    families = sorted({world.family for world in worlds})
    for family in families:
        members = sorted(
            (world for world in worlds if world.family == family),
            key=lambda w: hashlib.sha256(bytes(w.table)).digest(),
        )
        if len(members) < 3:
            raise ValueError(f"family {family} needs at least three semantic worlds")
        n_dev = max(1, round(0.2 * len(members)))
        n_test = max(1, round(0.2 * len(members)))
        n_train = len(members) - n_dev - n_test
        if n_train < 1:
            raise ValueError(f"family {family} cannot populate all splits")
        result["train"].extend(members[:n_train])
        result["dev"].extend(members[n_train:n_train + n_dev])
        result["test"].extend(members[n_train + n_dev:])
    for values in result.values():
        values.sort(key=lambda w: (w.family, w.world_id))
    return result


@dataclass(frozen=True)
class PairedTeachingData:
    one_step: Demonstrations
    depth_two: Demonstrations
    diagnostics: dict


def _make_data(tensors: tuple[torch.Tensor, ...], base_manifest: dict, teacher: str) -> Demonstrations:
    manifest = {**base_manifest, "teacher": teacher, "tensor_sha256": demonstration_digest(tensors)}
    return Demonstrations(*tensors, manifest)


def build_paired_teaching_data(
    worlds: list[RuleWorld], *, episodes: int, budget: int, seed: int,
    query_cost: float = 0.0,
) -> PairedTeachingData:
    """Create paired datasets whose histories differ only in teacher scores.

    Histories follow one frozen uniform-legal distribution.  World, target,
    observation, label, legal mask, row order, and update sampling are therefore
    identical between arms; the final ``gains`` tensor is the sole difference.
    """
    if not worlds or episodes <= 0 or budget < 1:
        raise ValueError("invalid demonstration configuration")
    bits = worlds[0].bits
    if any(world.bits != bits for world in worlds):
        raise ValueError("mixed world widths")
    candidates = np.asarray(inputs(bits), dtype=np.float32)
    if budget > len(candidates) - 2:
        raise ValueError("budget must leave a legal action at every labelled state")
    table = np.asarray([world.table for world in worlds], dtype=np.uint8)
    rng = random.Random(seed)
    obs, lengths, targets, labels, masks = [], [], [], [], []
    one_scores, two_scores = [], []
    branch_count = 0
    for _ in range(episodes):
        world_index = rng.randrange(len(worlds))
        target = rng.randrange(len(candidates))
        truth = table[world_index]
        legal = np.ones(len(candidates), dtype=bool)
        legal[target] = False
        compatible = np.ones(len(worlds), dtype=bool)
        history = np.zeros((budget, bits + 1), dtype=np.float32)
        for step in range(budget + 1):
            posterior = table[compatible]
            remaining = budget - step
            raw_one = teacher_gains(posterior, target, legal)
            one = np.where(legal, np.maximum(0.0, raw_one - query_cost), 0.0).astype(np.float32)
            if remaining == 0:
                one = np.zeros_like(one)
                two = np.zeros_like(one)
                searched = 0
            elif remaining == 1:
                # A depth-two controller cannot teach a preparatory action when
                # only one real inquiry remains. Both controls reduce to net IG.
                two = one.copy()
                searched = 0
            else:
                two, searched = depth_two_scores(
                    posterior, target, legal, query_cost=query_cost
                )
            obs.append(history.copy()); lengths.append(step); targets.append(candidates[target])
            labels.append(truth[target]); masks.append(legal.copy())
            one_scores.append(one); two_scores.append(two); branch_count += searched
            if step == budget:
                break
            action = rng.choice(np.flatnonzero(legal).tolist())
            value = truth[action]
            history[step] = np.concatenate((candidates[action], [value]))
            legal[action] = False
            compatible &= table[:, action] == value
    common_np = (obs, lengths, targets, labels, masks)
    raw = [torch.from_numpy(np.asarray(value)) for value in common_np]
    common = (raw[0].float(), raw[1].long(), raw[2].float(), raw[3].float(), raw[4].bool())
    one_tensors = (*common, torch.from_numpy(np.asarray(one_scores)).float())
    two_tensors = (*common, torch.from_numpy(np.asarray(two_scores)).float())
    identity = hashlib.sha256("\n".join(w.world_id for w in worlds).encode()).hexdigest()
    base = {"schema": "bramastra-inquiry-demonstrations/v1", "world_ids_sha256": identity,
            "worlds": len(worlds), "episodes": episodes, "states": len(labels),
            "budget": budget, "seed": seed, "history_policy": "uniform_legal",
            "query_cost": query_cost, "teacher_scope": "training worlds only"}
    one_data = _make_data(one_tensors, base, "one_step_target_entropy")
    two_data = _make_data(two_tensors, base, "exact_depth_two_target_entropy")
    one_array, two_array = np.asarray(one_scores), np.asarray(two_scores)
    legal_array = np.asarray(masks)
    informative_one = one_array.max(axis=1) > 1e-7
    informative_two = two_array.max(axis=1) > 1e-7
    both_informative = informative_one & informative_two
    one_best = np.isclose(one_array, one_array.max(axis=1, keepdims=True), atol=1e-7) & legal_array
    two_best = np.isclose(two_array, two_array.max(axis=1, keepdims=True), atol=1e-7) & legal_array
    diagnostics = {
        "states": len(labels), "q2_posterior_branches_evaluated": branch_count,
        "one_step_informative_fraction": float((one_array.max(axis=1) > 1e-7).mean()),
        "depth_two_informative_fraction": float((two_array.max(axis=1) > 1e-7).mean()),
        "one_step_mean_best_score": float(one_array.max(axis=1).mean()),
        "depth_two_mean_best_score": float(two_array.max(axis=1).mean()),
        "legal_fraction": float(legal_array.mean()),
        "teacher_target_agreement_fraction": (
            float((one_best[both_informative] & two_best[both_informative]).any(axis=1).mean())
            if both_informative.any() else None
        ),
        "both_informative_states": int(both_informative.sum()),
        "zero_remaining_both_zero": bool(
            np.allclose(one_array[budget::budget + 1], 0.0)
            and np.allclose(two_array[budget::budget + 1], 0.0)
        ),
        "one_remaining_labels_equal": bool(
            np.allclose(one_array[budget - 1::budget + 1], two_array[budget - 1::budget + 1])
        ),
    }
    return PairedTeachingData(one_data, two_data, diagnostics)
