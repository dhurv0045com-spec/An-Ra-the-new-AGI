from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from pathlib import Path
from typing import Any


NODE_TYPES = {"HYPOTHESIS", "ACTION", "OBSERVATION", "CORRECTION", "MEMORY", "CROSS_DOMAIN_ANALOGY"}
EDGE_TYPES = {"produces", "falsifies", "updates", "stores", "replays"}


def _stable_id(prefix: str, payload: str) -> str:
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:14]}"


@dataclass
class EPGNode:
    node_type: str
    content: dict[str, Any]
    node_id: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.node_type = self.node_type.upper()
        if self.node_type not in NODE_TYPES:
            raise ValueError(f"invalid EPG node type: {self.node_type}")
        if not self.node_id:
            self.node_id = _stable_id(self.node_type.lower(), json.dumps(self.content, sort_keys=True))


@dataclass
class EPGEdge:
    source: str
    target: str
    relation: str
    evidence: dict[str, Any] = field(default_factory=dict)
    edge_id: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.relation not in EDGE_TYPES:
            raise ValueError(f"invalid EPG edge type: {self.relation}")
        if not self.edge_id:
            self.edge_id = _stable_id("edge", f"{self.source}|{self.relation}|{self.target}")


class ExperimentalProofGraph:
    """JSON-first graph of experiments, observations, corrections, and replay memory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.nodes: dict[str, EPGNode] = {}
        self.edges: dict[str, EPGEdge] = {}
        if self.path.exists():
            self.load()

    def add_node(self, node_type: str, content: dict[str, Any]) -> EPGNode:
        node = EPGNode(node_type=node_type, content=content)
        self.nodes[node.node_id] = node
        self.save()
        return node

    def add_edge(self, source: str, target: str, relation: str, evidence: dict[str, Any] | None = None) -> EPGEdge:
        if source not in self.nodes:
            raise KeyError(f"missing source node: {source}")
        if target not in self.nodes:
            raise KeyError(f"missing target node: {target}")
        edge = EPGEdge(source=source, target=target, relation=relation, evidence=evidence or {})
        self.edges[edge.edge_id] = edge
        self.save()
        return edge

    def record_experiment(
        self,
        *,
        hypothesis: dict[str, Any],
        action: dict[str, Any],
        observation: dict[str, Any],
        correction: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
    ) -> dict[str, EPGNode]:
        h = self.add_node("HYPOTHESIS", hypothesis)
        a = self.add_node("ACTION", action)
        o = self.add_node("OBSERVATION", observation)
        self.add_edge(h.node_id, a.node_id, "produces")
        self.add_edge(a.node_id, o.node_id, "produces")
        nodes = {"hypothesis": h, "action": a, "observation": o}
        if correction is not None:
            c = self.add_node("CORRECTION", correction)
            relation = "falsifies" if observation.get("passed") is False else "updates"
            self.add_edge(o.node_id, c.node_id, relation)
            nodes["correction"] = c
        if memory is not None:
            m = self.add_node("MEMORY", memory)
            self.add_edge(o.node_id, m.node_id, "stores")
            nodes["memory"] = m
        return nodes

    def find_failed_corrected_by(self, query: str) -> list[dict[str, Any]]:
        needle = query.lower()
        out = []
        for edge in self.edges.values():
            if edge.relation != "falsifies":
                continue
            src = self.nodes.get(edge.source)
            dst = self.nodes.get(edge.target)
            if not src or not dst:
                continue
            blob = json.dumps(src.content).lower() + " " + json.dumps(dst.content).lower()
            if needle and needle not in blob:
                continue
            out.append({"failure": asdict(src), "correction": asdict(dst), "edge": asdict(edge)})
        return out

    def export_failure_replay(self) -> list[dict[str, Any]]:
        rows = []
        for item in self.find_failed_corrected_by(""):
            failure = item["failure"]["content"]
            correction = item["correction"]["content"]
            rows.append(
                {
                    "template": "FAILURE_REPLAY",
                    "failed_attempt": failure,
                    "correction": correction,
                    "text": (
                        f"<obs>{json.dumps(failure, sort_keys=True)}</obs>\n"
                        f"<err>{failure.get('error', failure.get('reason', 'verifier failed'))}</err>\n"
                        f"<upd>{json.dumps(correction, sort_keys=True)}</upd>"
                    ),
                }
            )
        return rows

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges.values()],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.nodes = {row["node_id"]: EPGNode(**row) for row in data.get("nodes", [])}
        self.edges = {row["edge_id"]: EPGEdge(**row) for row in data.get("edges", [])}
