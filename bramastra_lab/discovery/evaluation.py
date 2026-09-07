"""Batched evaluation: policies get only public queries and observed labels."""
from __future__ import annotations

import hashlib
import random

import torch

from .learner import Investigator
from .worlds import Episode, RuleWorld, inputs


def evaluate(model: Investigator, worlds: list[RuleWorld], *, policies: tuple[str, ...],
             budgets: tuple[int, ...], targets_per_world: int = 4, seed: int = 17,
             chunk_size: int = 128) -> list[dict]:
    allowed = {"random", "coverage", "learned", "uncertainty", "no_memory"}
    if not worlds or not policies or set(policies) - allowed or not budgets or min(budgets) < 0:
        raise ValueError("invalid evaluation configuration")
    bits = model.config.bits
    if max(budgets) >= 2 ** bits or not 1 <= targets_per_world <= 2 ** bits:
        raise ValueError("invalid target count or query budget")
    candidates = torch.tensor(inputs(bits), dtype=torch.float32)
    device = next(model.parameters()).device
    candidates = candidates.to(device)
    cases = []
    for world in worlds:
        if world.bits != bits:
            raise ValueError("model/world width mismatch")
        identity = int(hashlib.sha256(f"{seed}:{world.world_id}".encode()).hexdigest()[:16], 16)
        targets = random.Random(identity).sample(range(2 ** bits), targets_per_world)
        cases.extend((world, target, identity + target) for target in targets)
    coverage = list(dict.fromkeys([1 << i for i in range(bits)] + list(range(2 ** bits))))
    rows = []
    model.eval()
    with torch.no_grad():
        for policy in policies:
            for offset in range(0, len(cases), chunk_size):
                batch = cases[offset:offset + chunk_size]
                episodes = [Episode(w, target=t, budget=max(budgets)) for w, t, _ in batch]
                rngs = [random.Random(s) for _, _, s in batch]
                history = torch.zeros(len(batch), max(budgets), bits + 1, device=device)
                target_features = candidates[[t for _, t, _ in batch]]
                actions = [[] for _ in batch]
                for step in range(max(budgets) + 1):
                    lengths = torch.full((len(batch),), 0 if policy == "no_memory" else step,
                                         dtype=torch.long, device=device)
                    state = model.encode(history, lengths)
                    probabilities = torch.sigmoid(model.predict(state, target_features)).cpu().tolist()
                    if step in budgets:
                        for i, (world, target, _) in enumerate(batch):
                            label = world.table[target]  # Examiner-only scoring, never passed to the model.
                            probability = probabilities[i]
                            rows.append({"world_id": world.world_id, "family": world.family,
                                         "target": target, "policy": policy, "budget": step,
                                         "correct": int(probability >= 0.5) == label,
                                         "probability": probability, "label": label, "queries": step,
                                         "actions": list(actions[i])})
                    if step == max(budgets):
                        break
                    legal = torch.zeros(len(batch), len(candidates), dtype=torch.bool, device=device)
                    for i, episode in enumerate(episodes):
                        legal[i, list(episode.legal_actions())] = True
                    if policy == "learned":
                        scores = model.select(state, target_features, candidates, legal)
                        selected = scores.argmax(dim=-1).cpu().tolist()
                    elif policy == "uncertainty":
                        expanded = state[:, None].expand(-1, len(candidates), -1).reshape(-1, state.shape[-1])
                        queries = candidates[None].expand(len(batch), -1, -1).reshape(-1, bits)
                        logits = model.predict(expanded, queries).reshape(len(batch), -1)
                        selected = (-logits.abs()).masked_fill(~legal, -1e9).argmax(dim=-1).cpu().tolist()
                    elif policy in {"coverage", "no_memory"}:
                        selected = [next(a for a in coverage if a in e.legal_actions()) for e in episodes]
                    else:
                        selected = [r.choice(e.legal_actions()) for r, e in zip(rngs, episodes)]
                    for i, (episode, action) in enumerate(zip(episodes, selected)):
                        value = episode.observe(action)
                        history[i, step, :bits] = candidates[action]
                        history[i, step, bits] = value
                        actions[i].append(action)
    return rows
