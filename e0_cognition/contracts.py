"""Typed causal contracts for E0 cases and counterfactual pairs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable


class Split(str, Enum):
    DEVELOPMENT = "development"
    SEALED = "sealed"
    FRESH = "fresh"


class PairKind(str, Enum):
    QUERY_SWAP = "query_swap"
    RELEVANT_FACT_SWAP = "relevant_fact_swap"
    IRRELEVANT_FACT_SWAP = "irrelevant_fact_swap"
    ORDER_PERMUTATION = "order_permutation"
    STATE_SWAP = "state_swap"


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    relation: str
    target: str


@dataclass(frozen=True, slots=True)
class HiddenTruth:
    """Evaluator-only fields. Never include these in model-facing text."""

    relevant_fact_indices: tuple[int, ...]
    distractor_fact_indices: tuple[int, ...]
    graph: tuple[GraphEdge, ...]
    operation_trace: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CausalCase:
    case_id: str
    family: str
    split: Split
    domain: str
    template_id: str
    seed: int
    facts: tuple[str, ...]
    query: str
    answer: str
    candidates: tuple[str, ...]
    difficulty: tuple[tuple[str, int], ...]
    provenance: tuple[tuple[str, str], ...]
    hidden: HiddenTruth

    def context(self) -> str:
        return "\n".join(self.facts)

    def prompt(self) -> str:
        return f"{self.context()}\nQuestion: {self.query}\nAnswer:"

    def model_view(self) -> dict[str, str]:
        """The complete information allowed to reach a model adapter."""

        return {"context": self.context(), "query": self.query, "prompt": self.prompt()}

    def canonical(self, *, include_hidden: bool = True) -> dict[str, Any]:
        data = asdict(self)
        data["split"] = self.split.value
        if not include_hidden:
            data.pop("answer", None)
            data.pop("candidates", None)
            data.pop("hidden", None)
        return data

    def content_sha256(self) -> str:
        payload = json.dumps(
            self.canonical(include_hidden=True), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CausalPair:
    pair_id: str
    kind: PairKind
    base: CausalCase
    changed: CausalCase

    def assert_contract(self) -> None:
        """Verify the serialized pair changed exactly the intended variable."""

        if self.base.split != self.changed.split:
            raise AssertionError("a causal pair cannot cross splits")
        if self.base.case_id == self.changed.case_id:
            raise AssertionError("pair members require distinct identities")
        if self.base.candidates != self.changed.candidates:
            raise AssertionError("a causal pair must keep its scoring candidates fixed")
        if self.base.template_id != self.changed.template_id or self.base.domain != self.changed.domain:
            raise AssertionError("a causal pair must keep template and domain fixed")

        same_facts = self.base.facts == self.changed.facts
        same_multiset = Counter(self.base.facts) == Counter(self.changed.facts)
        same_query = self.base.query == self.changed.query
        same_answer = self.base.answer == self.changed.answer
        changed_fact_indices = tuple(
            i
            for i, (left, right) in enumerate(zip(self.base.facts, self.changed.facts))
            if left != right
        )
        if len(self.base.facts) != len(self.changed.facts):
            changed_fact_indices = (-1,)

        if self.kind is PairKind.QUERY_SWAP:
            if not same_facts or same_query or same_answer:
                raise AssertionError("query swap must preserve facts and change query+answer")
        elif self.kind is PairKind.RELEVANT_FACT_SWAP:
            if not same_query or same_answer or len(changed_fact_indices) != 1:
                raise AssertionError("relevant-fact swap must change one fact and the answer")
            if changed_fact_indices[0] not in self.base.hidden.relevant_fact_indices:
                raise AssertionError("changed fact is not evaluator-declared relevant")
        elif self.kind is PairKind.IRRELEVANT_FACT_SWAP:
            if not same_query or not same_answer or len(changed_fact_indices) != 1:
                raise AssertionError("irrelevant-fact swap must change one fact but preserve answer")
            if changed_fact_indices[0] not in self.base.hidden.distractor_fact_indices:
                raise AssertionError("changed fact is not evaluator-declared irrelevant")
        elif self.kind is PairKind.ORDER_PERMUTATION:
            if not same_multiset or same_facts or not same_query or not same_answer:
                raise AssertionError("permutation must reorder identical facts only")
        elif self.kind is PairKind.STATE_SWAP:
            if not same_query or same_answer or len(changed_fact_indices) != 1:
                raise AssertionError("state swap must change one state fact and the answer")
            if changed_fact_indices[0] not in self.base.hidden.relevant_fact_indices:
                raise AssertionError("state swap did not change the relevant update")
        else:  # pragma: no cover - Enum exhaustiveness guard
            raise AssertionError(f"unsupported pair kind: {self.kind}")


@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    schema: str
    generator_version: str
    split: Split
    cases: tuple[CausalCase, ...]
    pairs: tuple[CausalPair, ...]

    def assert_valid(self) -> None:
        if not self.cases:
            raise AssertionError("suite cannot be empty")
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise AssertionError("duplicate case ids")
        if any(case.split != self.split for case in self.cases):
            raise AssertionError("case split disagrees with suite split")
        hashes = [case.content_sha256() for case in self.cases]
        if len(hashes) != len(set(hashes)):
            raise AssertionError("duplicate canonical cases")
        known = set(ids)
        for pair in self.pairs:
            if pair.base.case_id not in known or pair.changed.case_id not in known:
                raise AssertionError("pair member missing from suite")
            pair.assert_contract()

    def sha256(self) -> str:
        self.assert_valid()
        payload = {
            "schema": self.schema,
            "generator_version": self.generator_version,
            "split": self.split.value,
            "cases": [case.canonical(include_hidden=True) for case in self.cases],
            "pairs": [
                {
                    "pair_id": pair.pair_id,
                    "kind": pair.kind.value,
                    "base": pair.base.case_id,
                    "changed": pair.changed.case_id,
                }
                for pair in self.pairs
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def family_histogram(self) -> dict[str, int]:
        return dict(sorted(Counter(case.family for case in self.cases).items()))


def assert_split_disjoint(suites: Iterable[EvaluationSuite]) -> None:
    """Enforce disjoint symbols, relations, templates, domains and case hashes."""

    suites = tuple(suites)
    for i, left in enumerate(suites):
        left_templates = {case.template_id for case in left.cases}
        left_domains = {case.domain for case in left.cases}
        left_hashes = {case.content_sha256() for case in left.cases}
        left_graph_tokens = {
            token
            for case in left.cases
            for edge in case.hidden.graph
            for token in (edge.source, edge.relation, edge.target)
        }
        for right in suites[i + 1 :]:
            if left.split == right.split:
                raise AssertionError("disjointness check expects different splits")
            checks = {
                "template": left_templates & {case.template_id for case in right.cases},
                "domain": left_domains & {case.domain for case in right.cases},
                "case hash": left_hashes & {case.content_sha256() for case in right.cases},
                "graph token": left_graph_tokens
                & {
                    token
                    for case in right.cases
                    for edge in case.hidden.graph
                    for token in (edge.source, edge.relation, edge.target)
                },
            }
            collisions = {name: values for name, values in checks.items() if values}
            if collisions:
                raise AssertionError(f"split leakage: {collisions}")
