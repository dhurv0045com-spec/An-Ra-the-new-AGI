"""Typed hierarchical goal planning and workflow compilation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Callable, Iterable


@dataclass(frozen=True)
class VerificationRule:
    verifier: str
    criterion: str


@dataclass
class MissionNode:
    node_id: str
    title: str
    objective: str
    level: int
    constraints: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    expected_artifacts: tuple[str, ...] = ()
    verification: tuple[VerificationRule, ...] = ()
    retry_budget: int = 3
    recovery: str = "replan"
    children: list["MissionNode"] = field(default_factory=list)

    def walk(self) -> Iterable["MissionNode"]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class MissionTree:
    goal: str
    root: MissionNode
    success_criteria: tuple[str, ...] = ()

    def validate(self, max_depth: int = 5, max_workflow: int = 10) -> None:
        nodes = list(self.root.walk())
        if max(node.level for node in nodes) > max_depth:
            raise ValueError(f"Mission tree exceeds maximum depth {max_depth}.")
        leaves = [node for node in nodes if not node.children]
        if len(leaves) > max_workflow:
            raise ValueError(f"Mission tree exceeds maximum workflow length {max_workflow}.")
        ids = {node.node_id for node in nodes}
        if len(ids) != len(nodes):
            raise ValueError("Mission node IDs must be unique.")
        for node in nodes:
            missing = set(node.dependencies) - ids
            if missing:
                raise ValueError(f"Node {node.node_id} has missing dependencies: {missing}")
            if node.retry_budget < 0:
                raise ValueError(f"Node {node.node_id} has a negative retry budget.")
            if any(not value.strip() for value in node.constraints):
                raise ValueError(f"Node {node.node_id} has an empty constraint.")
            if any(not value.strip() for value in node.expected_artifacts):
                raise ValueError(f"Node {node.node_id} has an empty expected artifact.")
        for leaf in leaves:
            if not leaf.verification:
                raise ValueError(f"Leaf {leaf.node_id} has no verification rule.")
            if any(not rule.verifier.strip() or not rule.criterion.strip() for rule in leaf.verification):
                raise ValueError(f"Leaf {leaf.node_id} has an invalid verification rule.")
        graph = {node.node_id: set(node.dependencies) for node in nodes}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError(f"Mission dependency cycle includes {node_id}.")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in graph[node_id]:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in graph:
            visit(node_id)

    def to_dict(self) -> dict[str, object]:
        return {"goal": self.goal, "root": asdict(self.root), "success_criteria": self.success_criteria}


@dataclass(frozen=True)
class WorkflowStep:
    skill: str
    parameters: dict[str, object]
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    timeout_seconds: float
    recovery: str
    source_node: str


class HierarchicalGoalPlanner:
    def __init__(self, max_depth: int = 5, max_workflow: int = 10) -> None:
        self.max_depth = int(max_depth)
        self.max_workflow = int(max_workflow)

    def make_node(
        self,
        *,
        title: str,
        objective: str,
        level: int,
        **kwargs,
    ) -> MissionNode:
        digest = hashlib.sha256(f"{title}:{objective}:{level}".encode("utf-8")).hexdigest()[:12]
        return MissionNode(node_id=f"mission-{digest}", title=title, objective=objective, level=level, **kwargs)

    def decompose(
        self,
        goal: str,
        decomposer: Callable[[str, int], MissionNode],
        *,
        success_criteria: tuple[str, ...] = (),
    ) -> MissionTree:
        root = decomposer(goal, self.max_depth)
        tree = MissionTree(goal=goal, root=root, success_criteria=success_criteria)
        tree.validate(self.max_depth, self.max_workflow)
        return tree

    def compile_workflow(
        self,
        tree: MissionTree,
        skill_mapper: Callable[[MissionNode], WorkflowStep],
    ) -> list[WorkflowStep]:
        tree.validate(self.max_depth, self.max_workflow)
        leaves = [node for node in tree.root.walk() if not node.children]
        return [skill_mapper(node) for node in leaves]

    def backtrack(
        self,
        tree: MissionTree,
        failed_node_id: str,
        revise: Callable[[MissionNode], MissionNode],
    ) -> MissionTree:
        def replace(node: MissionNode) -> MissionNode:
            if node.node_id == failed_node_id:
                if node.retry_budget <= 0:
                    raise RuntimeError(f"Retry budget exhausted for {failed_node_id}.")
                revised = revise(node)
                revised.retry_budget = node.retry_budget - 1
                return revised
            node.children = [replace(child) for child in node.children]
            return node

        tree.root = replace(tree.root)
        tree.validate(self.max_depth, self.max_workflow)
        return tree
