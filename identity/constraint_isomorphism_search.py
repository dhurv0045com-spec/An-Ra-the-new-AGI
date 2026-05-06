from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import time
from typing import Any


AXES = ("state", "operators", "invariants")
WEIGHTS = {"state": 0.2, "operators": 0.3, "invariants": 0.5}


@dataclass
class DomainSignature:
    name: str
    state: set[str] = field(default_factory=set)
    operators: set[str] = field(default_factory=set)
    invariants: set[str] = field(default_factory=set)

    @classmethod
    def from_mapping(cls, name: str, mapping: dict[str, Any]) -> "DomainSignature":
        return cls(
            name=name,
            state=_token_set(mapping.get("state", mapping.get("variables", []))),
            operators=_token_set(mapping.get("operators", mapping.get("actions", []))),
            invariants=_token_set(mapping.get("invariants", mapping.get("constraints", []))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": sorted(self.state),
            "operators": sorted(self.operators),
            "invariants": sorted(self.invariants),
        }


@dataclass
class AnalogyCandidate:
    domain_a: str
    domain_b: str
    axis_scores: dict[str, float]
    score: float
    valid: bool
    shared: dict[str, list[str]]
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


KNOWN_SIGNATURES: dict[str, DomainSignature] = {
    "quantum_transpilation": DomainSignature(
        name="quantum_transpilation",
        state={"qubits", "coupling_graph", "depth", "gate_sequence", "layout"},
        operators={"swap", "route", "compose", "transpile", "commute"},
        invariants={"unitary_equivalence", "graph_depth", "local_connectivity", "gate_equivalence"},
    ),
    "electrical_routing": DomainSignature(
        name="electrical_routing",
        state={"nets", "routing_graph", "wire_length", "layers", "congestion"},
        operators={"swap", "route", "place", "reroute", "transform"},
        invariants={"electrical_equivalence", "graph_depth", "connectivity", "wire_equivalence"},
    ),
    "protein_folding": DomainSignature(
        name="protein_folding",
        state={"residues", "conformation", "energy_landscape", "contacts", "solvent"},
        operators={"fold", "rotate", "minimize", "dock", "anneal"},
        invariants={"free_energy_minimum", "kinetic_traps", "steric_constraints", "bond_lengths"},
    ),
    "nanotech_self_assembly": DomainSignature(
        name="nanotech_self_assembly",
        state={"monomers", "configuration", "energy_landscape", "interfaces", "medium"},
        operators={"assemble", "rotate", "minimize", "dock", "anneal"},
        invariants={"free_energy_minimum", "kinetic_traps", "steric_constraints", "binding_geometry"},
    ),
    "qec_syndrome": DomainSignature(
        name="qec_syndrome",
        state={"syndrome_bits", "code_graph", "error_budget", "paths", "stabilizers"},
        operators={"decode", "match", "propagate", "measure", "correct"},
        invariants={"path_analysis", "error_budget", "parity_consistency", "logical_equivalence"},
    ),
    "fpga_timing_closure": DomainSignature(
        name="fpga_timing_closure",
        state={"timing_paths", "netlist_graph", "slack_budget", "paths", "registers"},
        operators={"place", "route", "propagate", "measure", "retime"},
        invariants={"path_analysis", "error_budget", "slack_consistency", "functional_equivalence"},
    ),
}


def _token_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        raw = value.replace(",", " ").replace("/", " ").replace("-", "_").split()
    else:
        raw = []
        for item in value:
            raw.extend(str(item).replace(",", " ").replace("/", " ").replace("-", "_").split())
    return {token.strip().lower() for token in raw if token.strip()}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


class ConstraintIsomorphismSearch:
    def __init__(self, signatures: dict[str, DomainSignature] | None = None, threshold: float = 0.65) -> None:
        self.signatures = dict(KNOWN_SIGNATURES)
        if signatures:
            self.signatures.update(signatures)
        self.threshold = float(threshold)

    def signature_for(self, name: str, override: dict[str, Any] | DomainSignature | None = None) -> DomainSignature:
        if isinstance(override, DomainSignature):
            return override
        if isinstance(override, dict):
            return DomainSignature.from_mapping(name, override)
        if name in self.signatures:
            return self.signatures[name]
        tokens = _token_set(name)
        return DomainSignature(name=name, state=tokens, operators=set(), invariants=tokens)

    def compare(
        self,
        domain_a: str,
        domain_b: str,
        *,
        signature_a: dict[str, Any] | DomainSignature | None = None,
        signature_b: dict[str, Any] | DomainSignature | None = None,
    ) -> AnalogyCandidate:
        a = self.signature_for(domain_a, signature_a)
        b = self.signature_for(domain_b, signature_b)
        axis_scores = {
            "state": jaccard(a.state, b.state),
            "operators": jaccard(a.operators, b.operators),
            "invariants": jaccard(a.invariants, b.invariants),
        }
        score = sum(axis_scores[axis] * WEIGHTS[axis] for axis in AXES)
        shared = {
            "state": sorted(a.state & b.state),
            "operators": sorted(a.operators & b.operators),
            "invariants": sorted(a.invariants & b.invariants),
        }
        return AnalogyCandidate(
            domain_a=a.name,
            domain_b=b.name,
            axis_scores={key: round(value, 4) for key, value in axis_scores.items()},
            score=round(score, 4),
            valid=score > self.threshold,
            shared=shared,
        )

    def store_confirmed_analogy(self, candidate: AnalogyCandidate, epg_path: str | Path) -> bool:
        if not candidate.valid:
            return False
        try:
            from memory.experimental_proof_graph import ExperimentalProofGraph

            epg = ExperimentalProofGraph(epg_path)
            epg.add_node("CROSS_DOMAIN_ANALOGY", candidate.to_dict())
            return True
        except Exception:
            path = Path(epg_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            if path.exists():
                try:
                    rows = json.loads(path.read_text(encoding="utf-8")).get("nodes", [])
                except Exception:
                    rows = []
            rows.append({"node_type": "CROSS_DOMAIN_ANALOGY", "content": candidate.to_dict()})
            path.write_text(json.dumps({"nodes": rows}, indent=2), encoding="utf-8")
            return True


def compare_domains(
    domain_a: str,
    domain_b: str,
    *,
    signature_a: dict[str, Any] | DomainSignature | None = None,
    signature_b: dict[str, Any] | DomainSignature | None = None,
    threshold: float = 0.65,
) -> AnalogyCandidate:
    return ConstraintIsomorphismSearch(threshold=threshold).compare(
        domain_a,
        domain_b,
        signature_a=signature_a,
        signature_b=signature_b,
    )
