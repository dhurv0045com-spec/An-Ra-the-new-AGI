"""
knowledge/graph.py — Step 9: Personal Knowledge Graph
Nodes: people, projects, goals, preferences, facts, events.
Edges: how everything connects.
Grows automatically. Queryable. Exportable.
"""

import json
import time
import re
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict


@dataclass
class GraphNode:
    id: str
    type: str           # person, project, goal, preference, skill, location, concept, event
    label: str          # human-readable name
    properties: Dict[str, Any] = field(default_factory=dict)
    importance: float = 3.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    mention_count: int = 0

    def touch(self):
        self.mention_count += 1
        self.updated_at = time.time()


@dataclass
class GraphEdge:
    id: str
    source: str         # node id
    target: str         # node id
    relation: str       # "works_on", "knows", "uses", "wants", "located_in", etc.
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)  # memory ids that support this edge


# ─────────────────────────────────────────────
# Node type detection
# ─────────────────────────────────────────────

NODE_PATTERNS = {
    "person": [
        r"\b([A-Z][a-z]+ (?:[A-Z][a-z]+)?)\b(?=\s+(?:is|was|works|said|told))",
        r"(?:my|her|his|their)\s+(?:friend|colleague|boss|partner|wife|husband|son|daughter)\s+([A-Z][a-z]+)",
    ],
    "project": [
        r"(?:project|app|product|startup|company|tool|system|platform|website|service)\s+(?:called|named)?\s*[\"']?([A-Z][a-zA-Z0-9\s]+)[\"']?",
        r"(?:building|creating|developing|working on)\s+(?:a|an|the)?\s+([A-Z][a-zA-Z0-9\s]+)",
    ],
    "skill": [
        r"\b(Python|JavaScript|TypeScript|Rust|Go|Java|C\+\+|Ruby|Swift|Kotlin|SQL|"
        r"React|Vue|Angular|FastAPI|Django|Flask|PyTorch|TensorFlow|Docker|Kubernetes|"
        r"AWS|GCP|Azure|PostgreSQL|MongoDB|Redis|GraphQL|REST|Machine Learning|"
        r"Deep Learning|NLP|Data Science|DevOps|MLOps)\b",
    ],
    "location": [
        r"(?:in|from|based in|living in|located in)\s+([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*)",
    ],
    "goal": [
        r"(?:want to|trying to|plan to|goal is to|hoping to)\s+(.{5,50}?)(?:\.|,|$)",
    ],
    "concept": [
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",  # capitalized concepts
    ],
}

RELATION_PATTERNS = [
    ("works_on",     r"(?:working on|building|developing|created|maintains?)"),
    ("uses",         r"(?:using|use|uses|works? with|built with|written in)"),
    ("knows",        r"(?:know|knows|met|friend|colleague|works? with)"),
    ("wants",        r"(?:want|wants|trying to|goal is|hoping to|plan to)"),
    ("located_in",   r"(?:live|lives|based|located|from)"),
    ("expert_in",    r"(?:expert|experienced|good at|proficient|specialized)"),
    ("learning",     r"(?:learning|studying|practicing|improving)"),
    ("part_of",      r"(?:part of|member of|works? at|employed by)"),
]


class KnowledgeGraph:
    """
    Personal knowledge graph for one user.
    Stored as JSON for portability.
    Thread-safe reads; writes require explicit save.
    """

    def __init__(self, graph_path: str, user_id: str = "default"):
        self.graph_path = Path(graph_path)
        self.user_id = user_id
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._load()

    # ── Node operations ───────────────────────

    def upsert_node(self, type: str, label: str,
                     properties: Optional[Dict] = None,
                     importance: float = 3.0) -> GraphNode:
        """Add node or update if exists (by type+label)."""
        node_id = self._node_id(type, label)
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.touch()
            if properties:
                node.properties.update(properties)
            node.importance = max(node.importance, importance)
        else:
            node = GraphNode(
                id=node_id, type=type, label=label,
                properties=properties or {},
                importance=importance,
            )
            self.nodes[node_id] = node
        return node

    def upsert_edge(self, source_label: str, source_type: str,
                     relation: str,
                     target_label: str, target_type: str,
                     weight: float = 1.0,
                     evidence_id: Optional[str] = None) -> GraphEdge:
        """Add or strengthen edge between two nodes."""
        src = self.upsert_node(source_type, source_label)
        tgt = self.upsert_node(target_type, target_label)
        edge_id = f"{src.id}_{relation}_{tgt.id}"

        if edge_id in self.edges:
            edge = self.edges[edge_id]
            edge.weight += weight
            if evidence_id:
                edge.evidence.append(evidence_id)
        else:
            edge = GraphEdge(
                id=edge_id, source=src.id, target=tgt.id,
                relation=relation, weight=weight,
                evidence=[evidence_id] if evidence_id else [],
            )
            self.edges[edge_id] = edge
            self._adjacency[src.id].add(tgt.id)
            self._adjacency[tgt.id].add(src.id)

        return edge

    def get_node(self, label: str, type: Optional[str] = None) -> Optional[GraphNode]:
        if type:
            return self.nodes.get(self._node_id(type, label))
        # Search by label
        label_lower = label.lower()
        for node in self.nodes.values():
            if node.label.lower() == label_lower:
                return node
        return None

    def get_neighbors(self, node_id: str,
                       relation: Optional[str] = None,
                       max_hops: int = 1) -> List[GraphNode]:
        """Return connected nodes (BFS)."""
        visited = {node_id}
        frontier = {node_id}
        result = []

        for _ in range(max_hops):
            next_frontier = set()
            for nid in frontier:
                for neighbor_id in self._adjacency.get(nid, set()):
                    if neighbor_id not in visited:
                        # Check relation filter
                        if relation:
                            edge_id_fwd = f"{nid}_{relation}_{neighbor_id}"
                            edge_id_bwd = f"{neighbor_id}_{relation}_{nid}"
                            if edge_id_fwd not in self.edges and edge_id_bwd not in self.edges:
                                continue
                        visited.add(neighbor_id)
                        next_frontier.add(neighbor_id)
                        if neighbor_id in self.nodes:
                            result.append(self.nodes[neighbor_id])
            frontier = next_frontier

        return result

    # ── Graph-building from memory ─────────────

    def extract_and_add(self, text: str, memory_id: Optional[str] = None):
        """Parse text and add nodes/edges to the graph."""
        self._extract_user_node(text, memory_id)
        self._extract_skills(text, memory_id)
        self._extract_projects(text, memory_id)
        self._extract_goals(text, memory_id)
        self._extract_locations(text, memory_id)
        self._extract_people(text, memory_id)

    def _extract_user_node(self, text: str, memory_id=None):
        """User is always the central node."""
        user_node = self.upsert_node("person", "User",
                                      importance=5.0,
                                      properties={"is_self": True})

        # Name extraction
        name_match = re.search(
            r"(?:my name is|i(?:'m| am) called|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            text, re.IGNORECASE
        )
        if name_match:
            name = name_match.group(1)
            user_node.properties["name"] = name
            user_node.label = name

    def _extract_skills(self, text: str, memory_id=None):
        user_node = self.upsert_node("person", "User", importance=5.0)
        for pattern_str in NODE_PATTERNS["skill"]:
            for match in re.finditer(pattern_str, text, re.IGNORECASE):
                skill = match.group(1)
                self.upsert_edge(
                    user_node.label, "person",
                    "uses", skill, "skill",
                    evidence_id=memory_id
                )

    def _extract_projects(self, text: str, memory_id=None):
        user_node = self.upsert_node("person", "User", importance=5.0)
        for pattern_str in NODE_PATTERNS["project"]:
            for match in re.finditer(pattern_str, text, re.IGNORECASE):
                project = match.group(1).strip()
                if len(project) < 2 or len(project) > 50:
                    continue
                self.upsert_edge(
                    user_node.label, "person",
                    "works_on", project, "project",
                    evidence_id=memory_id
                )

    def _extract_goals(self, text: str, memory_id=None):
        user_node = self.upsert_node("person", "User", importance=5.0)
        for match in re.finditer(
            r"(?:want to|trying to|plan to|goal is to)\s+(.{5,60}?)(?:\.|,|\n|$)",
            text, re.IGNORECASE
        ):
            goal_text = match.group(1).strip()
            if len(goal_text) < 5:
                continue
            goal_label = goal_text[:40]
            self.upsert_edge(
                user_node.label, "person",
                "wants", goal_label, "goal",
                evidence_id=memory_id
            )

    def _extract_locations(self, text: str, memory_id=None):
        user_node = self.upsert_node("person", "User", importance=5.0)
        for match in re.finditer(
            r"(?:live|lives|based|located|from)\s+(?:in|at)?\s+([A-Z][a-z]+(?:[,\s]+[A-Z][a-z]+)*)",
            text, re.IGNORECASE
        ):
            loc = match.group(1).strip()
            if len(loc) < 2:
                continue
            self.upsert_edge(
                user_node.label, "person",
                "located_in", loc, "location",
                evidence_id=memory_id
            )

    def _extract_people(self, text: str, memory_id=None):
        user_node = self.upsert_node("person", "User", importance=5.0)
        for match in re.finditer(
            r"(?:my|her|his)\s+(?:friend|colleague|partner|boss|manager)\s+([A-Z][a-z]+)",
            text, re.IGNORECASE
        ):
            person = match.group(1)
            self.upsert_edge(
                user_node.label, "person",
                "knows", person, "person",
                evidence_id=memory_id
            )

    # ── Query interface ───────────────────────

    def query(self, subject: str) -> Dict:
        """What do I know about X?"""
        node = self.get_node(subject)
        if not node:
            return {"found": False, "subject": subject}

        neighbors = self.get_neighbors(node.id, max_hops=2)
        edges = [e for e in self.edges.values()
                 if e.source == node.id or e.target == node.id]

        return {
            "found": True,
            "node": {"id": node.id, "type": node.type, "label": node.label,
                     "properties": node.properties, "importance": node.importance,
                     "mentions": node.mention_count},
            "connections": [
                {"relation": e.relation,
                 "node": self.nodes.get(
                     e.target if e.source == node.id else e.source
                 ).label if (e.target if e.source == node.id else e.source) in self.nodes else "?",
                 "weight": e.weight}
                for e in edges
            ],
            "neighbors": [n.label for n in neighbors[:10]],
        }

    def get_user_summary(self, user_id: str = "default") -> str:
        """Compact text summary of what we know about the user."""
        user_node = self.get_node("User", "person")
        if not user_node:
            return ""

        lines = []
        name = user_node.properties.get("name", "User")

        # Group edges by relation
        by_relation: Dict[str, List[str]] = defaultdict(list)
        for edge in self.edges.values():
            if edge.source == user_node.id:
                target = self.nodes.get(edge.target)
                if target:
                    by_relation[edge.relation].append(target.label)

        if "uses" in by_relation:
            techs = ", ".join(by_relation["uses"][:8])
            lines.append(f"Technologies: {techs}")
        if "works_on" in by_relation:
            projs = ", ".join(by_relation["works_on"][:5])
            lines.append(f"Projects: {projs}")
        if "wants" in by_relation:
            goals = "; ".join(by_relation["wants"][:3])
            lines.append(f"Goals: {goals}")
        if "located_in" in by_relation:
            lines.append(f"Location: {', '.join(by_relation['located_in'][:2])}")
        if "knows" in by_relation:
            people = ", ".join(by_relation["knows"][:5])
            lines.append(f"People mentioned: {people}")

        if not lines:
            return ""
        header = f"Known about {name}:"
        return header + "\n" + "\n".join(f"  • {l}" for l in lines)

    def export_json(self) -> str:
        """Export graph as JSON for visualization."""
        return json.dumps({
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
            "stats": {
                "n_nodes": len(self.nodes),
                "n_edges": len(self.edges),
                "user_id": self.user_id,
            }
        }, indent=2)

    def export_dot(self) -> str:
        """Export as GraphViz DOT format."""
        lines = ["digraph memory {", '  rankdir=LR;',
                 '  node [shape=box style=filled];']
        type_colors = {
            "person": "lightblue", "project": "lightgreen",
            "skill": "lightyellow", "goal": "lightsalmon",
            "location": "lightpink", "concept": "white",
        }
        for node in self.nodes.values():
            color = type_colors.get(node.type, "white")
            label = node.label.replace('"', "'")
            lines.append(f'  "{node.id}" [label="{label}" fillcolor="{color}"];')
        for edge in self.edges.values():
            lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{edge.relation}"];')
        lines.append("}")
        return "\n".join(lines)

    def stats(self) -> Dict:
        type_counts: Dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            type_counts[n.type] += 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "by_type": dict(type_counts),
            "top_nodes": sorted(
                [(n.label, n.mention_count) for n in self.nodes.values()],
                key=lambda x: -x[1]
            )[:10],
        }

    # ── Persistence ───────────────────────────

    def save(self):
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
            "edges": {eid: asdict(e) for eid, e in self.edges.items()},
        }
        with open(self.graph_path, "w") as f:
            json.dump(data, f)

    def _load(self):
        if not self.graph_path.exists():
            return
        try:
            with open(self.graph_path) as f:
                data = json.load(f)
            for nid, nd in data.get("nodes", {}).items():
                self.nodes[nid] = GraphNode(**nd)
            for eid, ed in data.get("edges", {}).items():
                self.edges[eid] = GraphEdge(**ed)
                self._adjacency[ed["source"]].add(ed["target"])
                self._adjacency[ed["target"]].add(ed["source"])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def _node_id(self, type: str, label: str) -> str:
        return f"{type}:{label.lower().replace(' ', '_')}"
