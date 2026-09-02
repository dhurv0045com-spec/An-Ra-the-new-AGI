"""Interventional cognitive geometry: neutral observations, latent worlds,
prospective prediction, falsification regimes, active selection.

Design law: the learner NEVER sees human-named factor sensors. The hidden
causal state h (missing-factor mask) passes through a fixed random mixing
transform A plus noise to produce neutral observables o. No dimension of o
maps one-to-one onto a factor. All worlds share the outcome physics; they
differ in how o relates to h and how families relate to interventions —
which is what makes them falsification regimes rather than demonstrations.

Pure deterministic Python + numpy. Software/mechanism evidence only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

INTERVENTIONS = ("NO_CHANGE", "RETRIEVAL_HELP", "BINDING_SUPPORT",
                 "DECOMPOSITION", "FULL_REPLAY")
COSTS = {"NO_CHANGE": 0, "RETRIEVAL_HELP": 1, "BINDING_SUPPORT": 1,
         "DECOMPOSITION": 2, "FULL_REPLAY": 4}
N_FACTORS = 3
D_OBS = 6
COVERAGE = {
    "NO_CHANGE": frozenset(),
    "RETRIEVAL_HELP": frozenset({"retrieve"}),
    "BINDING_SUPPORT": frozenset({"bind"}),
    "DECOMPOSITION": frozenset({"compose"}),
    "FULL_REPLAY": frozenset({"retrieve", "bind", "compose"}),
}
FAMILIES = ("ledger", "gazetteer", "telemetry", "manifest", "atlas", "cipher")


def _mixing(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Small magnitudes keep A@h inside the observable range: a
    # saturating mixing transform would collapse distinct causal
    # states onto identical observations (information destruction).
    A = rng.uniform(0.15, 0.45, size=(D_OBS, N_FACTORS))
    return A * rng.choice((-1.0, 1.0), size=A.shape)


def _observations(A: np.ndarray, required: frozenset[str], noise: float,
                  rng: random.Random, squash: bool,
                  factor_order: tuple[str, ...]) -> np.ndarray:
    h = np.array([1.0 if f in required else 0.0 for f in factor_order])
    z = A @ h + np.array([rng.gauss(0, noise) for _ in range(A.shape[0])])
    if squash:
        z = 1.0 / (1.0 + np.exp(-z))
    return np.clip(z, 0.0, 1.0)


@dataclass(slots=True)
class World:
    name: str
    A: np.ndarray
    rows: list[dict]
    families: tuple[str, ...]
    meta: dict


def make_world(name: str, seed: int, n_tasks: int, *, noise: float = 0.06,
               squash: bool = False, random_outcomes: bool = False,
               interaction: bool = False, extra_factor: bool = False,
               A: np.ndarray | None = None,
               factor_order: tuple[str, str, str] =
               ("retrieve", "bind", "compose"),
               required_sets: list[frozenset[str]] | None = None) -> World:
    rng = random.Random(seed)
    # Observation noise comes from a DEDICATED stream: requirement sampling
    # must not shift the noise applied to identical latent states (this is
    # what makes the factor-permutation invariance exact).
    noise_rng = random.Random(seed + 999_999)
    A = _mixing(seed) if A is None else A
    rows = []
    for i in range(n_tasks):
        if required_sets is not None:
            required = required_sets[i]  # externally fixed (invariance tests)
        else:
            k = rng.choice((1, 1, 2, 2, 3))
            required = frozenset(rng.sample(list(factor_order), k))
        if extra_factor and rng.random() < 0.4:
            required = frozenset(set(required) | {"quantum"})
        family = FAMILIES[rng.randrange(len(FAMILIES))]
        o = _observations(A, required & set(factor_order), noise, noise_rng,
                          squash, factor_order)
        if random_outcomes:
            gold = [rng.randint(0, 1) for _ in INTERVENTIONS]
            gold[0] = 0  # NO_CHANGE never repairs a failing task
        else:
            gold = []
            for iv in INTERVENTIONS:
                ok = COVERAGE[iv] >= required
                if interaction and ok and {"retrieve", "compose"} <= required:
                    ok = False  # non-additive block: the pair jointly fails
                gold.append(int(ok))
        rows.append({"o": o, "family": family, "required": required,
                     "gold": gold, "index": i})
    return World(name, A, rows, tuple(FAMILIES),
                 {"noise": noise, "squash": squash, "seed": seed,
                  "factor_order": factor_order})


def shortcut_correlated_world(name: str, seed: int, n_tasks: int, *,
                              dev_correlation: float,
                              fresh_correlation: float) -> tuple[World, World]:
    """WORLD C: family -> best intervention strongly correlated in
    development, recombined in fresh; latent truth unchanged."""
    dev = make_world(f"{name}-dev", seed, n_tasks)
    fresh = make_world(f"{name}-fresh", seed + 1, n_tasks)
    rng = random.Random(seed + 2)

    def best_of(row: dict) -> str:
        for iv, gold in zip(INTERVENTIONS, row["gold"]):
            if gold == 1:
                return iv
        return "NO_CHANGE"

    def relabel(world: World, correlation: float) -> None:
        for row in world.rows:
            matching = FAMILIES[INTERVENTIONS.index(best_of(row))]
            if rng.random() < correlation:
                row["family"] = matching
            else:
                others = [f for f in FAMILIES if f != matching]
                row["family"] = others[rng.randrange(len(others))]

    relabel(dev, dev_correlation)
    relabel(fresh, fresh_correlation)
    return dev, fresh


def effective_rank(matrix: np.ndarray) -> float:
    """Participation ratio of the CENTERED singular-value spectrum: a shared
    column offset is not causal structure."""
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    s = np.linalg.svd(matrix, compute_uv=False)
    energy = s ** 2
    return float((energy.sum() ** 2) / (energy ** 2).sum())


def conjunction_basis(O: np.ndarray) -> np.ndarray:
    """Neutral nonlinear basis: raw dims + pairwise products + pairwise mins
    + triple mins. Lets a linear head represent CONJUNCTIVE outcome structure
    (an intervention repairs iff several factors are missing) without ever
    naming a factor. Family-blind by construction."""
    n, d = O.shape
    cols = [O[:, i] for i in range(d)]
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(O[:, i] * O[:, j])
            cols.append(np.minimum(O[:, i], O[:, j]))
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                cols.append(np.minimum(np.minimum(O[:, i], O[:, j]), O[:, k]))
    return np.column_stack(cols)


class ProspectivePredictor:
    """Ridge head o_x -> response row R_x (predicted BEFORE outcomes).
    use_conjunction=False is the pure linear geometry — the honest baseline
    that fails on conjunctive/interaction worlds."""

    def __init__(self, use_conjunction: bool = True) -> None:
        self.use_conjunction = use_conjunction
        self.W: np.ndarray | None = None
        self.center: np.ndarray | None = None

    def _design(self, world: World) -> np.ndarray:
        O = np.stack([row["o"] for row in world.rows]) - self.center
        return conjunction_basis(O) if self.use_conjunction else O

    def fit(self, world: World, lam: float = 1e-2) -> "ProspectivePredictor":
        O = np.stack([row["o"] for row in world.rows])
        self.center = O.mean(axis=0)
        D = self._design(world)
        R = np.stack([np.array(row["gold"], dtype=float) for row in world.rows])
        self.W = np.linalg.solve(D.T @ D + lam * np.eye(D.shape[1]), D.T @ R)
        return self

    def predict(self, world: World) -> np.ndarray:
        return self._design(world) @ self.W

    @staticmethod
    def errors(pred: np.ndarray, world: World) -> np.ndarray:
        R = np.stack([np.array(row["gold"], dtype=float) for row in world.rows])
        return np.abs(pred - R).mean(axis=1)


def policy_from_prediction(pred_row: np.ndarray, cost_weight: float = 0.05) -> str:
    best, best_value = INTERVENTIONS[0], -1e18
    for i, name in enumerate(INTERVENTIONS):
        value = pred_row[i] - cost_weight * COSTS[name]
        if value > best_value:
            best, best_value = name, value
    return best


def active_select(predictors: list[ProspectivePredictor], world: World,
                  row_index: int, *, beta: float = 0.5,
                  cost_weight: float = 0.05) -> tuple[str, float]:
    """value(I) = mean predicted repair + beta * committee disagreement
    (epistemic uncertainty / expected information) - cost."""
    preds = np.stack([p.predict(world)[row_index] for p in predictors])
    mean = preds.mean(axis=0)
    disagreement = preds.std(axis=0)

    def z(v):
        sd = float(v.std())
        return (v - v.mean()) / sd if sd > 1e-12 else v * 0.0
    # Scale-free combination: repair value and information gain compete in
    # standardized units, so beta has a consistent meaning across worlds.
    zm, zd = z(mean), z(disagreement)
    values = [zm[i] + beta * zd[i] - cost_weight * COSTS[name]
              for i, name in enumerate(INTERVENTIONS)]
    best = max(zip(values, INTERVENTIONS))[1]
    return best, float(disagreement.max())
