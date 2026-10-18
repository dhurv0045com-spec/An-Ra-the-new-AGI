from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping

from .contracts import PUBLIC_SCHEMA_ID, SPLITS, SuiteManifest, content_hash
from .rules import CATALOG, CATALOG_BY_STRUCTURE, STRUCTURE_SPLITS, RuleSpec, catalog_digest


GENERATOR_VERSION = "anra-discovery-gym/generator/v2"
SUITE_SCHEMA = "anra-discovery-gym/suite-manifest/v2"
WORLD_SCHEMA = "anra-discovery-gym/world-manifest/v2"
RENDERINGS = (
    ("letters", ("alpha", "beta", "gamma")),
    ("roles", ("source", "relay", "target")),
    ("numeric", ("unit0", "unit1", "unit2")),
)
FULL_AFFORDANCE = ("read", "probe", "intervene", "submit", "stop")
NO_INTERVENTION_AFFORDANCE = ("read", "probe", "submit", "stop")
AFFORDANCE_BY_SPLIT = {
    "train": FULL_AFFORDANCE,
    "development": FULL_AFFORDANCE,
    "confirmation": FULL_AFFORDANCE,
    "transfer": NO_INTERVENTION_AFFORDANCE,
}
ALL_CONTEXTS = tuple(((index >> 2) & 1, (index >> 1) & 1, index & 1) for index in range(8))


@dataclass(frozen=True, slots=True)
class WorldManifest:
    schema: str
    world_id: str
    world_id_basis_sha256: str
    split: str
    structure_id: str
    rendering_id: str
    variable_names: tuple[str, ...]
    target_context_id: str
    probe_contexts: tuple[tuple[int, ...], ...]
    target_candidate_contexts: tuple[tuple[int, ...], ...]
    intervention_contexts: tuple[tuple[int, ...], ...]
    generation_inputs_sha256: str
    rule_semantic_sha256: str
    target_assignment_sha256: str
    public_schema_id: str
    action_costs: Mapping[str, int]
    legal_action_kinds: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "world_id": self.world_id,
            "world_id_basis_sha256": self.world_id_basis_sha256,
            "split": self.split,
            "structure_id": self.structure_id,
            "rendering_id": self.rendering_id,
            "variable_names": list(self.variable_names),
            "target_context_id": self.target_context_id,
            "probe_contexts": [list(item) for item in self.probe_contexts],
            "target_candidate_contexts": [list(item) for item in self.target_candidate_contexts],
            "intervention_contexts": [list(item) for item in self.intervention_contexts],
            "generation_inputs_sha256": self.generation_inputs_sha256,
            "rule_semantic_sha256": self.rule_semantic_sha256,
            "target_assignment_sha256": self.target_assignment_sha256,
            "public_schema_id": self.public_schema_id,
            "action_costs": dict(self.action_costs),
            "legal_action_kinds": list(self.legal_action_kinds),
        }


@dataclass(frozen=True, slots=True)
class GeneratedWorld:
    world_id: str
    split: str
    structure_id: str
    rendering_id: str
    variable_names: tuple[str, ...]
    rule: RuleSpec
    target_assignment: tuple[int, int, int]
    probe_contexts: tuple[tuple[int, int, int], ...]
    intervention_contexts: tuple[tuple[int, int, int], ...]
    legal_action_kinds: tuple[str, ...]
    generation_inputs: Mapping[str, object]
    manifest: WorldManifest

    @property
    def hidden_answer(self) -> int:
        return self.rule.evaluate(self.target_assignment)

    def evaluator_record(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.to_dict(),
            "rule": {
                "structure_id": self.rule.structure_id,
                "variant": self.rule.variant,
                "parents": list(self.rule.parents),
                "threshold": self.rule.threshold,
                "flip": self.rule.flip,
                "swap_branches": self.rule.swap_branches,
                "semantic_sha256": self.rule.semantic_sha256,
            },
            "target_assignment": list(self.target_assignment),
            "legal_action_kinds": list(self.legal_action_kinds),
            "hidden_answer": self.hidden_answer,
        }


def _split_seed(base_seed: int, split: str) -> int:
    offset = {"train": 0, "development": 100000, "confirmation": 200000, "transfer": 300000}[split]
    return base_seed + offset


def world_id_basis(*, split: str, seed: int, index: int) -> dict[str, object]:
    return {
        "generator_version": GENERATOR_VERSION,
        "split": split,
        "seed": seed,
        "index": index,
    }


def world_id_for(*, split: str, seed: int, index: int) -> tuple[str, str]:
    basis = world_id_basis(split=split, seed=seed, index=index)
    basis_hash = content_hash(basis)
    return f"dgw-{basis_hash[:20]}", basis_hash


def _make_world(split: str, seed: int, index: int) -> GeneratedWorld:
    structures = STRUCTURE_SPLITS[split]
    structure_id = structures[index % len(structures)]
    variants = CATALOG_BY_STRUCTURE[structure_id]
    variant_index = (index // len(structures)) % len(variants)
    rule = variants[variant_index]
    rendering_id, variable_names = RENDERINGS[(index + SPLITS.index(split)) % len(RENDERINGS)]
    legal_action_kinds = AFFORDANCE_BY_SPLIT[split]
    world_seed = seed + index * 7919
    rng = random.Random(world_seed)
    intervention_anchor = tuple(rng.choice(ALL_CONTEXTS))
    probe_contexts = tuple(
        sorted(
            [intervention_anchor]
            + [
                tuple(1 - value if position == flip_axis else value for position, value in enumerate(intervention_anchor))
                for flip_axis in range(3)
            ]
        )
    )
    target_candidate_contexts = tuple(
        sorted(item for item in ALL_CONTEXTS if item not in probe_contexts)
    )
    intervention_contexts = (intervention_anchor,)
    target_assignment = tuple(rng.choice(target_candidate_contexts))
    generation_inputs = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "split": split,
        "index": index,
        "structure_id": structure_id,
        "variant": rule.variant,
        "rendering_id": rendering_id,
        "legal_action_kinds": list(legal_action_kinds),
        "target_assignment": list(target_assignment),
        "probe_contexts": [list(item) for item in probe_contexts],
        "target_candidate_contexts": [list(item) for item in target_candidate_contexts],
        "intervention_contexts": [list(item) for item in intervention_contexts],
        "catalog_sha256": catalog_digest(),
    }
    inputs_hash = content_hash(generation_inputs)
    world_id, world_id_basis_sha256 = world_id_for(split=split, seed=seed, index=index)
    action_costs = {"read": 1, "probe": 2, "intervene": 3, "submit": 1, "stop": 0}
    manifest = WorldManifest(
        schema=WORLD_SCHEMA,
        world_id=world_id,
        world_id_basis_sha256=world_id_basis_sha256,
        split=split,
        structure_id=structure_id,
        rendering_id=rendering_id,
        variable_names=variable_names,
        target_context_id="current-state",
        probe_contexts=probe_contexts,
        target_candidate_contexts=target_candidate_contexts,
        intervention_contexts=intervention_contexts,
        generation_inputs_sha256=inputs_hash,
        rule_semantic_sha256=rule.semantic_sha256,
        target_assignment_sha256=content_hash(list(target_assignment)),
        public_schema_id=PUBLIC_SCHEMA_ID,
        action_costs=action_costs,
        legal_action_kinds=legal_action_kinds,
    )
    return GeneratedWorld(
        world_id=world_id,
        split=split,
        structure_id=structure_id,
        rendering_id=rendering_id,
        variable_names=variable_names,
        rule=rule,
        target_assignment=target_assignment,
        probe_contexts=probe_contexts,
        intervention_contexts=intervention_contexts,
        legal_action_kinds=legal_action_kinds,
        generation_inputs=generation_inputs,
        manifest=manifest,
    )


def generate_worlds(split: str, *, seed: int, count: int) -> tuple[GeneratedWorld, ...]:
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    if count <= 0:
        raise ValueError("world count must be positive")
    return tuple(_make_world(split, _split_seed(seed, split), index) for index in range(count))


def generate_suite(*, seed: int = 1701, worlds_per_split: int = 6) -> dict[str, tuple[GeneratedWorld, ...]]:
    if worlds_per_split < 2:
        raise ValueError("each split needs at least two worlds")
    suite = {split: generate_worlds(split, seed=seed, count=worlds_per_split) for split in SPLITS}
    world_ids = [world.world_id for worlds in suite.values() for world in worlds]
    if len(world_ids) != len(set(world_ids)):
        raise ValueError("world identity collision")
    return suite


def suite_manifest(suite: Mapping[str, tuple[GeneratedWorld, ...]], *, seed: int) -> SuiteManifest:
    if set(suite) != set(SPLITS):
        raise ValueError("suite must contain every declared split")
    worlds = {
        split: [world.manifest.to_dict() for world in suite[split]]
        for split in SPLITS
    }
    split_identities = {split: content_hash(worlds[split]) for split in SPLITS}
    structures = {
        split: sorted({world.structure_id for world in suite[split]})
        for split in SPLITS
    }
    affordances = {
        split: sorted({",".join(world.legal_action_kinds) for world in suite[split]})
        for split in SPLITS
    }
    renderings = {
        split: sorted({world.rendering_id for world in suite[split]})
        for split in SPLITS
    }
    variable_roles = {
        split: sorted({",".join(world.variable_names) for world in suite[split]})
        for split in SPLITS
    }
    return SuiteManifest(
        schema=SUITE_SCHEMA,
        generator_version=GENERATOR_VERSION,
        seed=seed,
        split_seeds={split: _split_seed(seed, split) for split in SPLITS},
        worlds=worlds,
        split_identities=split_identities,
        structure_ids=structures,
        affordance_ids=affordances,
        rendering_ids=renderings,
        variable_role_ids=variable_roles,
        catalog_sha256=catalog_digest(),
    )


def public_descriptor(world: GeneratedWorld) -> dict[str, object]:
    descriptor: dict[str, object] = {
        "world_id": world.world_id,
        "variable_names": list(world.variable_names),
        "target_context_id": world.manifest.target_context_id,
        "probe_contexts": [list(item) for item in world.probe_contexts],
        "legal_action_kinds": list(world.legal_action_kinds),
        "belief_space_id": catalog_digest(),
    }
    if "intervene" in world.legal_action_kinds:
        descriptor["intervention_contexts"] = [list(item) for item in world.intervention_contexts]
    return descriptor
