"""Grounded cognitive workspace (M19).

Typed goal frame, evidence ledger, belief set, acyclic subgoal graph, pending
commitments and capability estimates over episode-local aliases. Rendering
exposes aliases only; provenance (task ids, splits, evaluator verdicts) never
enters rendered tokens. Belief confidence cannot change epistemic status.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from bramastra_lab.research.experience.trajectory import PROVENANCE_ONLY_FIELDS
from bramastra_lab.research.cognition.beliefs import (
    BELIEF_STATUSES,
    Belief,
    BeliefError,
    EvidenceRecord,
    corroboration_roots_for,
    reference_finite_support_update,
    unique_corroboration_roots,
)

OPERATION_VERBS = frozenset({"RETRIEVE", "PREDICT", "COMPARE", "DECOMPOSE", "DERIVE",
                             "QUERY", "EXECUTE", "VERIFY", "REVISE", "ABSTAIN", "SUBMIT"})
SUBGOAL_STATUSES = frozenset({"pending", "verified", "failed", "unknown", "cancelled"})
POOL_ELIGIBLE_FOR_WORKSPACE = frozenset({"training", "controller"})
COGNITION_PRIVATE_FIELDS = PROVENANCE_ONLY_FIELDS | frozenset({
    "task_semantic_id", "source", "collection_policy", "family",
    "mechanism_cluster", "trainable", "pool", "pair_group_id",
    "model_checkpoint",
})

RENDER_BUDGET_DEFAULT = 512


class WorkspaceError(ValueError):
    """A workspace operation violated its contract."""


@dataclass
class Subgoal:
    subgoal_id: str
    parent: str | None
    description: str
    dependencies: tuple[str, ...]
    check: str                       # public predicate/test reference
    status: str = "pending"
    estimated_cost: float = 0.0


@dataclass
class PendingCommitment:
    commitment_id: str
    verb: str
    expected_return: str
    deadline_step: int
    status: str = "open"             # open | completed | cancelled


@dataclass
class CapabilityEstimate:
    operation: str
    family: str
    recent_validated_performance: float
    source_pool: str                 # must be training/controller to render
    model_checkpoint: str


@dataclass
class CognitiveWorkspace:
    """Episode-scoped cognitive state. Reset at the protocol boundary."""

    goal: Mapping[str, Any]
    success_predicate: str
    budget: int
    evidence: dict[str, EvidenceRecord] = field(default_factory=dict)
    beliefs: dict[str, Belief] = field(default_factory=dict)
    subgoals: dict[str, Subgoal] = field(default_factory=dict)
    commitments: dict[str, PendingCommitment] = field(default_factory=dict)
    capability_estimates: list[CapabilityEstimate] = field(default_factory=list)
    step_counter: int = 0
    _alias_counter: int = 0

    # -- aliases --------------------------------------------------------------

    def next_alias(self, prefix: str) -> str:
        self._alias_counter += 1
        return f"{prefix}{self._alias_counter}"

    # -- evidence -------------------------------------------------------------

    def admit_evidence(self, content: Mapping[str, Any], *,
                       ancestry: Sequence[str] = (),
                       reliability: float = 1.0) -> EvidenceRecord:
        alias = self.next_alias("e")
        record = EvidenceRecord(alias=alias, content=dict(content),
                                ancestry=tuple(ancestry) or (alias,),
                                reliability=reliability)
        self.evidence[alias] = record
        return record

    def retract_evidence(self, alias: str) -> None:
        if alias not in self.evidence:
            raise WorkspaceError(f"unknown evidence alias {alias!r}")
        self.evidence[alias] = EvidenceRecord(
            alias=alias, content=self.evidence[alias].content,
            ancestry=self.evidence[alias].ancestry, status="retracted",
            reliability=self.evidence[alias].reliability,
            conflict_state=self.evidence[alias].conflict_state)

    def set_evidence_conflict_state(self, alias: str, state: str) -> None:
        """Mirror deterministic temporal-conflict analysis into typed state.

        Admission/retraction is orthogonal to contradiction/supersession:
        conflict states remain visible in the audit ledger but are not new
        admitted evidence for Bayesian revision.
        """
        if alias not in self.evidence:
            raise WorkspaceError(f"unknown evidence alias {alias!r}")
        if state not in {"active", "conflicting", "superseded"}:
            raise WorkspaceError(f"unknown evidence conflict state {state!r}")
        record = self.evidence[alias]
        if record.status != "admitted":
            raise WorkspaceError("retracted evidence cannot be reclassified")
        self.evidence[alias] = EvidenceRecord(
            alias=record.alias, content=record.content,
            ancestry=record.ancestry, status=record.status,
            reliability=record.reliability, conflict_state=state)

    # -- beliefs --------------------------------------------------------------

    def propose_belief(self, proposition: Mapping[str, Any], *,
                       status: str = "hypothesis",
                       evidence_aliases: Sequence[str] = ()) -> Belief:
        if status not in BELIEF_STATUSES:
            raise WorkspaceError(f"belief status must be one of {sorted(BELIEF_STATUSES)}")
        for alias in evidence_aliases:
            if alias not in self.evidence:
                raise WorkspaceError(f"belief references missing evidence {alias!r}")
        alias = self.next_alias("b")
        belief = Belief(alias=alias, proposition=dict(proposition), status=status,
                        support={}, evidence_aliases=tuple(evidence_aliases))
        self.beliefs[alias] = belief
        return belief

    def revise_belief(self, belief_alias: str, likelihood: Mapping[str, float],
                      evidence_aliases: Sequence[str]) -> dict[str, float]:
        """Apply one source-observation update to a finite-support belief.

        Revisions are idempotent by evidence ancestry: replaying the same
        observation or a derived copy cannot amplify its likelihood. Each
        call accepts exactly one new independent source root; callers with
        several observations must submit them as separate updates so each
        source's reliability is applied explicitly.
        """
        if belief_alias not in self.beliefs:
            raise WorkspaceError(f"unknown belief {belief_alias!r}")
        belief = self.beliefs[belief_alias]
        admitted: list[EvidenceRecord] = []
        seen_aliases: set[str] = set()
        for alias in evidence_aliases:
            record = self.evidence.get(alias)
            if record is None:
                raise WorkspaceError(f"revision references missing evidence {alias!r}")
            if record.status == "admitted" and alias not in seen_aliases:
                admitted.append(record)
                seen_aliases.add(alias)
        if not admitted:
            raise WorkspaceError("revision requires at least one admitted evidence record")

        # Resolve lineage against the complete workspace so derived copies
        # still point to their original source even when that source alias was
        # not repeated in this call.
        prior_records = [self.evidence[alias]
                         for alias in belief.evidence_aliases
                         if alias in self.evidence
                         and self.evidence[alias].status == "admitted"]
        prior_roots = unique_corroboration_roots(
            prior_records, universe=self.evidence)
        incoming_roots: dict[str, list[EvidenceRecord]] = {}
        for record in admitted:
            for root in corroboration_roots_for(record, self.evidence):
                incoming_roots.setdefault(root, []).append(record)
        new_roots = sorted(set(incoming_roots) - prior_roots)
        if len(new_roots) > 1:
            raise WorkspaceError(
                "revise one independent evidence root per call; split this "
                "update so each observation's likelihood and reliability "
                "remain auditable")

        cited_aliases = tuple(sorted(set(belief.evidence_aliases)
                                     | {record.alias for record in admitted}))
        if not new_roots:
            # Preserve the new alias in the audit trail without changing the
            # posterior: a duplicate/corroborating copy is not new evidence.
            self.beliefs[belief_alias] = Belief(
                alias=belief.alias, proposition=belief.proposition,
                status=belief.status, support=dict(belief.support),
                evidence_aliases=cited_aliases,
                conflicting_aliases=tuple(sorted(set(belief.conflicting_aliases))))
            return dict(belief.support)

        source_root = new_roots[0]
        root_records = incoming_roots[source_root]
        source_record = self.evidence.get(source_root)
        reliability = (source_record.reliability
                       if source_record is not None
                       and source_record.status == "admitted"
                       else min(record.reliability for record in root_records))
        support = dict(belief.support) or {
            h: 1.0 / max(1, len(likelihood)) for h in likelihood}
        if not support:
            raise WorkspaceError("belief revision has no hypotheses to update")
        if set(likelihood) != set(support):
            raise WorkspaceError(
                "likelihood hypotheses must exactly match the belief support")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
               for value in likelihood.values()):
            raise WorkspaceError("likelihood values must be finite probabilities in [0, 1]")

        # Reliability is the probability that this source's stated likelihood
        # model is trustworthy. The complement contributes a hypothesis-
        # independent likelihood equal to the model's mean likelihood. At
        # reliability 1 this is ordinary Bayes; lower values temper the update
        # without letting one low-trust zero likelihood erase a hypothesis.
        baseline = sum(float(value) for value in likelihood.values()) / len(likelihood)
        effective_likelihood = {
            hypothesis: reliability * float(value) + (1.0 - reliability) * baseline
            for hypothesis, value in likelihood.items()}
        next_support, status = reference_finite_support_update(
            support, effective_likelihood)
        if status == "MODEL_MISMATCH":
            # Retain the evidence and the prior belief; flag the mismatch.
            self.beliefs[belief_alias] = Belief(
                alias=belief.alias, proposition=belief.proposition,
                status="unresolved", support=belief.support,
                evidence_aliases=cited_aliases,
                conflicting_aliases=tuple(sorted(
                    set(belief.conflicting_aliases)
                    | {record.alias for record in root_records})))
            raise BeliefError(
                "MODEL_MISMATCH: support/likelihood assumptions are inconsistent "
                "with the observation; evidence retained, broader hypothesis "
                "proposal required")
        new_status = belief.status
        lead = _leading(next_support)
        prior_lead = _leading(support)
        if lead != prior_lead or len(next_support) > 1 and min(
                next_support.values()) > max(support.values()):
            new_status = "unresolved" if belief.status in ("contradicted",) else belief.status
        lowered = next_support.get(prior_lead, 0.0) < support.get(prior_lead, 0.0)
        if lowered:
            new_status = "contradicted" if belief.status == "observed_report" else belief.status
        conflicting_aliases = set(belief.conflicting_aliases)
        if prior_lead is not None and any(
                float(effective_likelihood.get(prior_lead, 0.0))
                < float(effective_likelihood.get(hypothesis, 0.0))
                for hypothesis in support if hypothesis != prior_lead):
            conflicting_aliases.update(record.alias for record in root_records)
        self.beliefs[belief_alias] = Belief(
            alias=belief_alias, proposition=belief.proposition, status=new_status,
            support=next_support,
            evidence_aliases=cited_aliases,
            conflicting_aliases=tuple(sorted(conflicting_aliases)))
        return next_support

    def contradict_belief(self, belief_alias: str) -> None:
        """Explicit contradiction marks status; confidence fields cannot
        promote a hypothesis to observed evidence."""
        if belief_alias not in self.beliefs:
            raise WorkspaceError(f"unknown belief {belief_alias!r}")
        belief = self.beliefs[belief_alias]
        self.beliefs[belief_alias] = Belief(
            alias=belief_alias, proposition=belief.proposition,
            status="contradicted", support=belief.support,
            evidence_aliases=belief.evidence_aliases,
            conflicting_aliases=belief.conflicting_aliases)

    # -- subgoal DAG ----------------------------------------------------------

    def add_subgoal(self, subgoal_id: str, *, description: str, check: str,
                    parent: str | None = None,
                    dependencies: Sequence[str] = (),
                    estimated_cost: float = 0.0) -> Subgoal:
        if subgoal_id in self.subgoals:
            raise WorkspaceError(f"subgoal {subgoal_id!r} already exists")
        if parent is not None and parent not in self.subgoals:
            raise WorkspaceError(f"subgoal parent {parent!r} does not exist")
        deps = tuple(dependencies)
        for dep in deps:
            if dep not in self.subgoals:
                raise WorkspaceError(f"dependency {dep!r} does not exist")
        # Cycle rejection: adding id with these dependencies must not create
        # a path back to id.
        if subgoal_id in deps or self._reaches(subgoal_id, deps):
            raise WorkspaceError("cyclic subgoal dependency rejected")
        subgoal = Subgoal(subgoal_id=subgoal_id, parent=parent,
                          description=description, dependencies=deps, check=check,
                          estimated_cost=estimated_cost)
        self.subgoals[subgoal_id] = subgoal
        return subgoal

    def _reaches(self, target: str, start_dependencies: Sequence[str]) -> bool:
        stack = list(start_dependencies)
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            subgoal = self.subgoals.get(current)
            if subgoal is not None:
                stack.extend(subgoal.dependencies)
                if subgoal.parent is not None:
                    stack.append(subgoal.parent)
        return False

    def complete_subgoal(self, subgoal_id: str, *, verified: bool | None,
                         cancelled: bool = False) -> Subgoal:
        """Completion is evidence-bearing: only a public predicate/test or an
        admitted observation verifies. Model text saying 'done' is not it."""
        if subgoal_id not in self.subgoals:
            raise WorkspaceError(f"unknown subgoal {subgoal_id!r}")
        if cancelled:
            status = "cancelled"
        elif verified is True:
            status = "verified"
        elif verified is False:
            status = "failed"
        else:
            status = "unknown"
        self.subgoals[subgoal_id].status = status
        return self.subgoals[subgoal_id]

    # -- capability estimates -------------------------------------------------

    def add_capability_estimate(self, estimate: CapabilityEstimate) -> None:
        if estimate.source_pool not in POOL_ELIGIBLE_FOR_WORKSPACE:
            raise WorkspaceError(
                f"capability estimate from pool {estimate.source_pool!r} is not "
                "eligible for the model-visible workspace; query/confirmation/"
                "sealed summaries stay examiner sidecars")
        if not 0.0 <= estimate.recent_validated_performance <= 1.0:
            raise WorkspaceError("capability estimate must lie in [0, 1]")
        self.capability_estimates.append(estimate)

    # -- rendering / persistence ---------------------------------------------

    def rendered_view(self, budget: int = RENDER_BUDGET_DEFAULT) -> dict[str, Any]:
        """Return a bounded JSON-byte view: evidence links, no private provenance."""
        if (not isinstance(budget, int) or isinstance(budget, bool)
                or budget < 1):
            raise WorkspaceError("render budget must be a positive integer")
        view = {
            "goal": dict(self.goal),
            "success_predicate": self.success_predicate,
            "remaining_budget": max(0, self.budget - self.step_counter),
            "evidence": [{"alias": record.alias, "content": dict(record.content),
                          "status": (record.conflict_state
                                     if record.status == "admitted"
                                     else record.status)}
                         for record in self.evidence.values()],
            "beliefs": [{"alias": belief.alias,
                         "proposition": dict(belief.proposition),
                         "status": belief.status,
                         "support": {k: round(belief.support[k], 6)
                                     for k in sorted(belief.support)},
                         "evidence_aliases": list(belief.evidence_aliases),
                         "conflicting_aliases": list(belief.conflicting_aliases)}
                        for belief in self.beliefs.values()],
            "subgoals": [{"id": s.subgoal_id, "parent": s.parent,
                          "description": s.description, "status": s.status,
                          "dependencies": list(s.dependencies), "check": s.check}
                         for s in self.subgoals.values()],
            "commitments": [{"id": c.commitment_id, "verb": c.verb,
                             "expected_return": c.expected_return,
                             "deadline_step": c.deadline_step,
                             "status": c.status}
                            for c in self.commitments.values()],
            "capability_estimates": [
                {"operation": e.operation, "family": e.family,
                 "performance": round(e.recent_validated_performance, 4)}
                for e in self.capability_estimates],
        }
        _reject_provenance(view)
        if len(_stable_size(view)) <= budget:
            return view

        # Keep the stable goal frame, active commitments, and current beliefs
        # when possible. Evict complete low-priority capability estimates,
        # then oldest evidence and completed planning state. Report every
        # omission. Never return a render that still exceeds its declared
        # serialized JSON byte budget.
        omitted: dict[str, int] = {}
        view = dict(view)
        for key in ("capability_estimates", "evidence", "subgoals",
                    "commitments", "beliefs"):
            while view[key] and len(_stable_size(view)) > budget:
                view[key].pop(0)
                omitted[key] = omitted.get(key, 0) + 1
                view["omitted"] = dict(omitted)
        if len(_stable_size(view)) > budget:
            raise WorkspaceError(
                "goal frame and omission receipt exceed render budget; "
                "increase the budget or shorten the goal")
        return view

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": dict(self.goal), "success_predicate": self.success_predicate,
            "budget": self.budget,
            "evidence": [{"alias": r.alias, "content": dict(r.content),
                          "ancestry": list(r.ancestry), "status": r.status,
                          "reliability": r.reliability,
                          "conflict_state": r.conflict_state}
                         for r in self.evidence.values()],
            "beliefs": [b.__dict__ | {"support": dict(b.support)} for b in self.beliefs.values()],
            "subgoals": [vars(s) | {"dependencies": list(s.dependencies)}
                         for s in self.subgoals.values()],
            "commitments": [vars(c) for c in self.commitments.values()],
            "capability_estimates": [vars(e) for e in self.capability_estimates],
            "step_counter": self.step_counter,
            "_alias_counter": self._alias_counter,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CognitiveWorkspace":
        workspace = cls(goal=dict(raw["goal"]), success_predicate=raw["success_predicate"],
                        budget=raw["budget"])
        for record in raw["evidence"]:
            workspace.evidence[record["alias"]] = EvidenceRecord(
                alias=record["alias"], content=record["content"],
                ancestry=tuple(record["ancestry"]), status=record["status"],
                reliability=record["reliability"],
                conflict_state=record.get("conflict_state", "active"))
        for belief in raw["beliefs"]:
            workspace.beliefs[belief["alias"]] = Belief(
                alias=belief["alias"], proposition=belief["proposition"],
                status=belief["status"], support=dict(belief["support"]),
                evidence_aliases=tuple(belief.get("evidence_aliases", ())),
                conflicting_aliases=tuple(belief.get("conflicting_aliases", ())))
        for subgoal in raw["subgoals"]:
            workspace.subgoals[subgoal["subgoal_id"]] = Subgoal(
                subgoal_id=subgoal["subgoal_id"], parent=subgoal.get("parent"),
                description=subgoal["description"],
                dependencies=tuple(subgoal["dependencies"]), check=subgoal["check"],
                status=subgoal["status"], estimated_cost=subgoal.get("estimated_cost", 0.0))
        for commitment in raw["commitments"]:
            workspace.commitments[commitment["commitment_id"]] = PendingCommitment(
                commitment_id=commitment["commitment_id"], verb=commitment["verb"],
                expected_return=commitment["expected_return"],
                deadline_step=commitment["deadline_step"], status=commitment["status"])
        for estimate in raw["capability_estimates"]:
            workspace.capability_estimates.append(CapabilityEstimate(**estimate))
        workspace.step_counter = raw["step_counter"]
        workspace._alias_counter = raw["_alias_counter"]
        return workspace


def _leading(support: Mapping[str, float]) -> str | None:
    if not support:
        return None
    return max(support, key=lambda key: support[key])


def _stable_size(view: Mapping[str, Any]) -> str:
    import json

    return json.dumps(view, sort_keys=True, default=str)


def _reject_provenance(value: Any, path: str = "view") -> None:
    """Fail closed when internal audit fields are embedded in model-visible data."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in COGNITION_PRIVATE_FIELDS:
                raise WorkspaceError(
                    f"provenance-only field {key!r} cannot enter rendered cognition "
                    f"state at {path}")
            _reject_provenance(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_provenance(child, f"{path}[{index}]")
