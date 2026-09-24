#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    from .validate_evidence_ledger import (
        CANONICAL_METADATA_FILES,
        DEFAULT_LEDGER,
        EXPECTED_IMPORT_COUNT,
        EXPECTED_LEDGER_COUNT,
        EXPECTED_STATUS_ENUM,
        EXPECTED_MANIFEST_SHA256,
        GitRepository,
        MANIFEST_RELATIVE,
        add_problem,
        commit_tokens,
        load_json,
        require_keys,
        require_string,
        require_string_list,
        status_counts,
        validate_full_commit,
        validate_ledger,
        validate_manifest,
        working_tree_file,
    )
except ImportError:
    from validate_evidence_ledger import (
        CANONICAL_METADATA_FILES,
        DEFAULT_LEDGER,
        EXPECTED_IMPORT_COUNT,
        EXPECTED_LEDGER_COUNT,
        EXPECTED_STATUS_ENUM,
        EXPECTED_MANIFEST_SHA256,
        GitRepository,
        MANIFEST_RELATIVE,
        add_problem,
        commit_tokens,
        load_json,
        require_keys,
        require_string,
        require_string_list,
        status_counts,
        validate_full_commit,
        validate_ledger,
        validate_manifest,
        working_tree_file,
    )

ROOT = Path(__file__).resolve().parents[1]
MODEL_SCHEMA = "anra.research-decision-model/v1"
BELIEF_SCHEMA = "anra.belief-registry/v1"
ARCHITECTURE_SCHEMA = "anra.architecture-decision-ledger/v1"
DEPENDENCY_SCHEMA = "anra.research-dependency-graph/v1"
TREE_SCHEMA = "anra.next-experiment-decision-tree/v1"
EXPECTED_BELIEF_STATUS = [
    "STRONGLY_SUPPORTED", "SUPPORTED", "WEAKLY_SUPPORTED", "OPEN", "CONTESTED",
    "CONTRADICTED", "NOT_TESTED",
]
EXPECTED_BELIEF_CLASSES = {
    "ARCHITECTURE", "BINDING", "CAUSAL_DIAGNOSIS", "COGNITION_DATA", "COMPOSITION",
    "CONTINUAL_LEARNING", "DATA_MIXTURE", "EVALUATION_VALIDITY", "GENERALIZATION",
    "GUARDIAN_CONTROL", "INTERFERENCE", "INVARIANCE", "LEARNING_RATE", "OBJECTIVE",
    "OPTIMIZER", "OUTPUT_COMPETITION", "PLASTICITY", "QK_NORMALIZATION", "REPLAY",
    "REPRESENTATION", "RETENTION", "SCALING_TO_500M", "SCHEDULE", "TERMINATION",
    "TOKENIZATION",
}
EXPECTED_ARCHITECTURE_STATUS = [
    "LOCKED_BY_EVIDENCE", "DEFAULT_BASELINE", "PROVISIONAL",
    "BLOCKED_ON_EXPERIMENT", "REJECTED", "UNJUSTIFIED",
]
EXPECTED_NODE_TYPES = [
    "UNRESOLVED_RESULT", "DECISION", "GATE", "EXPERIMENT", "TERMINAL",
]
EXPECTED_GRAPH_STATUSES = {
    "COMPLETE", "FIRED", "COMPLETE_BUT_FLOOR_LIMITED", "PARTIAL_IN_PROGRESS",
    "NEXT", "QUEUED_AFTER_PREFLIGHT", "BLOCKED",
    "REJECTED_FOR_TESTED_PRODUCTION_REMEDY", "BLOCKED_ON_BASELINE_SENSITIVITY",
    "OPEN", "DO_NOT_RUN", "DEVELOPMENT_READY_DEFERRED", "BLOCKED_BY_EXTERNAL",
    "PENDING_BLOCKED", "ENGINEERING_ONLY_COMPLETED_PARTIAL",
    "ENGINEERING_ONLY_NOT_RUN", "PREREGISTERED_EXECUTION_BLOCKED", "SUPERSEDED_NO_EXECUTION",
    "PROPOSED", "BLOCKED_UPSTREAM", "NO_SCIENCE_UNLOCK",
}
EXPECTED_EDGE_RELATIONS = {
    "fires", "informs", "extends_partial", "discriminates", "prerequisite",
    "feeds", "required_for", "authorizes", "would_strengthen_or_reverse", "supersedes",
}
EXPECTED_UNKNOWN_STATUS = {"OPEN", "ANSWERED_NO"}
EXPECTED_CANDIDATE_STATUS = {
    "complete", "next", "queued-after-preflight", "required-parallel",
    "blocked-by-corpus-and-geometry", "parallel-zero-gpu", "opportunistic",
    "designed", "embedded", "blocked-by-ARK020-readiness", "DO_NOT_RUN",
    "deferred", "blocked", "zero-gpu", "completed-engineering-partial",
    "engineering-not-run", "superseded-by-preregistered-design", "preregistered-execution-blocked",
}
EXPECTED_KILL_STATUS = {
    "FIRED", "OPEN", "PARTIAL_OR_INTERACTION", "BLOCKED",
    "OPEN_SUCCESSOR_REQUIRED",
}
EXPECTED_MODEL_KEYS = {
    "schema", "generated", "phase", "historical_snapshot",
    "consolidation_source_manifest", "consolidation_manifest_sha256",
    "pre_consolidation_head", "phase1_basis", "consolidation_basis",
    "evidence_snapshot_delta_since_phase1", "beliefs_ref", "evidence_refs",
    "unknowns", "dependencies_ref", "candidate_experiments", "scoring_function",
    "ranking", "ranking_sensitivity", "blocked_decisions", "kill_criteria_ref",
    "kill_criteria", "kill_criteria_detail", "decision_tree_ref",
    "authorization_boundary", "invariants",
}
EXPECTED_BELIEF_KEYS = {
    "schema", "generated", "phase", "historical_snapshot_date",
    "consolidation_source_manifest", "consolidation_manifest_sha256",
    "pre_consolidation_head", "evidence_snapshot", "status_enum", "beliefs",
}
EXPECTED_ARCHITECTURE_KEYS = {
    "schema", "generated", "phase", "historical_snapshot_date",
    "consolidation_source_manifest", "consolidation_manifest_sha256",
    "pre_consolidation_head", "audience", "authorization_boundary", "status_enum",
    "state_definitions", "decisions",
}
EXPECTED_DEPENDENCY_KEYS = {
    "schema", "generated", "phase", "historical_snapshot_date",
    "consolidation_source_manifest", "consolidation_manifest_sha256",
    "pre_consolidation_head", "node_types", "nodes", "edges", "critical_paths",
}
EXPECTED_TREE_KEYS = {
    "schema", "generated", "phase", "historical_snapshot_date",
    "consolidation_source_manifest", "consolidation_manifest_sha256",
    "pre_consolidation_head", "usage", "root", "current_evidence_state", "nodes",
}
ALLOWED_TREE_NODE_FIELDS = {
    "question", "branches", "action", "next", "prohibitions", "label", "rule",
    "path_record", "belief_updates", "decision_updates",
}
EXPECTED_GRAPH_PATH_NAMES = {
    "representation-critical-path", "formation-mux-floor-branch",
    "continual-learning-critical-path", "scale-critical-path",
    "engineering-only-branch",
}
EXPECTED_TREE_STATE_KEYS = {
    "R1C", "K01", "CS-TRANSFER-001", "FORMATION-MUX-001-S5-V8",
    "FORMATION-MUX-001-V12-FRONTIER-PARTIAL", "ROLE-TRANSFER-001", "ARK-020-V4", "K8", "TPU",
}
HEX_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


def exact_top_level(value: Any, keys: set[str], label: str, problems: list[str]) -> bool:
    return require_keys(value, keys, label, problems, exact=True)


def validate_pre_consolidation_head(
    value: Any, label: str, repository: GitRepository, problems: list[str]
) -> None:
    fields = {"branch", "ref", "commit", "label"}
    if not require_keys(value, fields, label, problems, exact=True):
        return
    for key in ("branch", "ref", "label"):
        require_string(value[key], f"{label}.{key}", problems)
    validate_full_commit(value["commit"], f"{label}.commit", repository, problems)


def validate_common_metadata(
    value: Any,
    schema: str,
    label: str,
    root: Path,
    actual_manifest_hash: str,
    repository: GitRepository,
    problems: list[str],
) -> None:
    if not isinstance(value, dict):
        add_problem(problems, f"{label}: expected object")
        return
    if value.get("schema") != schema:
        add_problem(problems, f"{label}: wrong schema {value.get('schema')!r}")
    if value.get("phase") != 3:
        add_problem(problems, f"{label}: phase must be 3")
    if "generated" in value:
        require_string(value["generated"], f"{label}.generated", problems)
    if "historical_snapshot_date" in value:
        require_string(value["historical_snapshot_date"], f"{label}.historical_snapshot_date", problems)
    if value.get("consolidation_source_manifest") != MANIFEST_RELATIVE:
        add_problem(problems, f"{label}: wrong consolidation source manifest")
    if value.get("consolidation_manifest_sha256") != actual_manifest_hash:
        add_problem(problems, f"{label}: consolidation manifest hash does not match actual bytes")
    if "pre_consolidation_head" in value:
        validate_pre_consolidation_head(value["pre_consolidation_head"], f"{label}.pre_consolidation_head", repository, problems)
    if not isinstance(value.get("consolidation_manifest_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", actual_manifest_hash):
        add_problem(problems, f"{label}: invalid actual manifest hash")


def unique_ids(
    values: list[Any], key: str, label: str, problems: list[str], expected_count: int | None = None
) -> set[str]:
    ids: list[str] = []
    if not isinstance(values, list):
        add_problem(problems, f"{label}: expected list")
        return set()
    if expected_count is not None and len(values) != expected_count:
        add_problem(problems, f"{label}: expected {expected_count} records, found {len(values)}")
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            add_problem(problems, f"{label}[{index}]: expected object")
            continue
        item_id = value.get(key)
        if not isinstance(item_id, str) or not item_id.strip() or not HEX_ID.fullmatch(item_id):
            add_problem(problems, f"{label}[{index}].{key}: invalid ID")
            continue
        ids.append(item_id)
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        add_problem(problems, f"{label}: duplicate IDs {duplicates}")
    return set(ids)


def validate_reference_list(
    values: Any,
    known: set[str],
    label: str,
    problems: list[str],
    required: bool = True,
) -> None:
    if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
        add_problem(problems, f"{label}: expected list of non-empty strings")
        return
    if required and not values:
        add_problem(problems, f"{label}: must not be empty")
    if len(values) != len(set(values)):
        add_problem(problems, f"{label}: contains duplicates")
    for value in values:
        if value not in known:
            add_problem(problems, f"{label}: unresolved reference {value}")


def load_documents(root: Path, problems: list[str]) -> dict[str, Any]:
    names = {
        "ledger": DEFAULT_LEDGER,
        "beliefs": "docs/research/BELIEF_REGISTRY.json",
        "architecture": "docs/research/ARCHITECTURE_DECISION_LEDGER.json",
        "model": "docs/research/RESEARCH_DECISION_MODEL.json",
        "dependency": "docs/research/RESEARCH_DEPENDENCY_GRAPH.json",
        "tree": "docs/research/NEXT_EXPERIMENT_DECISION_TREE.json",
    }
    documents: dict[str, Any] = {}
    for key, relative in names.items():
        path = root / relative
        try:
            documents[key] = load_json(path)
        except (OSError, ValueError) as exc:
            documents[key] = {}
            add_problem(problems, f"{key}: cannot parse {relative}: {exc}")
    return documents


def validate_manifest_hash_and_ledger(
    root: Path, repository: GitRepository, documents: dict[str, Any], problems: list[str]
) -> tuple[str, set[str]]:
    manifest_path = root / MANIFEST_RELATIVE
    actual_hash = ""
    manifest_paths: set[str] = set()
    try:
        manifest_bytes = manifest_path.read_bytes()
        actual_hash = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = load_json(manifest_path)
        manifest_paths = validate_manifest(manifest, manifest_path, root, repository, problems)
    except (OSError, ValueError) as exc:
        add_problem(problems, f"manifest: cannot parse {MANIFEST_RELATIVE}: {exc}")
    if actual_hash != EXPECTED_MANIFEST_SHA256:
        add_problem(problems, f"manifest: actual SHA-256 {actual_hash or '<missing>'} does not match expected value")
    ledger = documents.get("ledger")
    if isinstance(ledger, dict):
        validate_ledger(ledger, root, repository, manifest_paths, problems)
    else:
        add_problem(problems, "ledger: top-level JSON must be an object")
    return actual_hash, manifest_paths


def validate_belief_registry(
    beliefs: Any,
    ledger_ids: set[str],
    architecture_ids: set[str],
    root: Path,
    actual_hash: str,
    repository: GitRepository,
    problems: list[str],
) -> set[str]:
    if not exact_top_level(beliefs, EXPECTED_BELIEF_KEYS, "belief registry", problems):
        return set()
    validate_common_metadata(beliefs, BELIEF_SCHEMA, "belief registry", root, actual_hash, repository, problems)
    if beliefs.get("status_enum") != EXPECTED_BELIEF_STATUS:
        add_problem(problems, "belief registry.status_enum: values or order do not match schema")
    snapshot = beliefs.get("evidence_snapshot")
    snapshot_fields = {
        "phase1_branch", "phase1_completion_sha", "snapshot_date", "snapshot_note",
        "consolidation_source_manifest", "pre_consolidation_head_commit",
    }
    if not require_keys(snapshot, snapshot_fields, "belief registry.evidence_snapshot", problems, exact=True):
        pass
    else:
        require_string(snapshot["phase1_branch"], "belief registry.evidence_snapshot.phase1_branch", problems)
        validate_full_commit(snapshot["phase1_completion_sha"], "belief registry.evidence_snapshot.phase1_completion_sha", repository, problems)
        require_string(snapshot["snapshot_date"], "belief registry.evidence_snapshot.snapshot_date", problems)
        require_string(snapshot["snapshot_note"], "belief registry.evidence_snapshot.snapshot_note", problems)
        if snapshot["consolidation_source_manifest"] != MANIFEST_RELATIVE:
            add_problem(problems, "belief registry.evidence_snapshot: wrong consolidation manifest")
        pre_head = beliefs.get("pre_consolidation_head")
        pre_commit = pre_head.get("commit") if isinstance(pre_head, dict) else None
        if snapshot["pre_consolidation_head_commit"] != pre_commit:
            add_problem(problems, "belief registry.evidence_snapshot: pre-consolidation commit disagrees")
        validate_full_commit(snapshot["pre_consolidation_head_commit"], "belief registry.evidence_snapshot.pre_consolidation_head_commit", repository, problems)
    rows = beliefs.get("beliefs")
    belief_ids = unique_ids(rows, "belief_id", "belief registry.beliefs", problems, 32)
    fields = {
        "belief_id", "belief_class", "statement", "current_status", "prior_justification",
        "supporting_experiments", "contradicting_experiments", "dependent_design_decisions",
        "plausible_alternatives", "confidence", "what_would_falsify_it", "cheapest_falsifier",
        "value_of_resolving",
    }
    if isinstance(rows, list):
        for index, belief in enumerate(rows):
            if not isinstance(belief, dict) or not require_keys(belief, fields, f"belief registry.beliefs[{index}]", problems, exact=True):
                continue
            belief_id = belief["belief_id"]
            for key in ("belief_class", "statement", "prior_justification", "confidence", "what_would_falsify_it", "cheapest_falsifier", "value_of_resolving"):
                require_string(belief[key], f"belief {belief_id}.{key}", problems)
            if belief["current_status"] not in EXPECTED_BELIEF_STATUS:
                add_problem(problems, f"belief {belief_id}: invalid current status {belief['current_status']!r}")
            if not isinstance(belief["belief_class"], str) or belief["belief_class"] not in EXPECTED_BELIEF_CLASSES:
                add_problem(problems, f"belief {belief_id}: invalid belief class {belief['belief_class']!r}")
            validate_reference_list(belief["supporting_experiments"], ledger_ids, f"belief {belief_id}.supporting_experiments", problems, required=False)
            validate_reference_list(belief["contradicting_experiments"], ledger_ids, f"belief {belief_id}.contradicting_experiments", problems, required=False)
            validate_reference_list(belief["dependent_design_decisions"], architecture_ids, f"belief {belief_id}.dependent_design_decisions", problems, required=False)
            if not isinstance(belief["plausible_alternatives"], list) or any(not isinstance(item, str) or not item for item in belief["plausible_alternatives"]):
                add_problem(problems, f"belief {belief_id}.plausible_alternatives: expected string list")
    return belief_ids


def validate_architecture_ledger(
    architecture: Any,
    ledger_ids: set[str],
    unknown_ids: set[str],
    dependency_ids: set[str],
    root: Path,
    actual_hash: str,
    repository: GitRepository,
    problems: list[str],
) -> set[str]:
    if not exact_top_level(architecture, EXPECTED_ARCHITECTURE_KEYS, "architecture ledger", problems):
        return set()
    validate_common_metadata(architecture, ARCHITECTURE_SCHEMA, "architecture ledger", root, actual_hash, repository, problems)
    require_string(architecture.get("audience"), "architecture ledger.audience", problems)
    require_string(architecture.get("authorization_boundary"), "architecture ledger.authorization_boundary", problems)
    if architecture.get("status_enum") != EXPECTED_ARCHITECTURE_STATUS:
        add_problem(problems, "architecture ledger.status_enum: values or order do not match schema")
    definitions = architecture.get("state_definitions")
    if not isinstance(definitions, dict) or set(definitions) != set(EXPECTED_ARCHITECTURE_STATUS) or any(not isinstance(v, str) or not v for v in definitions.values()):
        add_problem(problems, "architecture ledger.state_definitions: keys and values do not match status enum")
    decisions = architecture.get("decisions")
    ids = unique_ids(decisions, "id", "architecture ledger.decisions", problems, 33)
    fields = {"id", "component", "decision", "status", "evidence_refs", "rationale", "blocked_by", "would_change_if"}
    if isinstance(decisions, list):
        for index, decision in enumerate(decisions):
            if not isinstance(decision, dict) or not require_keys(decision, fields, f"architecture ledger.decisions[{index}]", problems, exact=True):
                continue
            decision_id = decision["id"]
            for key in ("component", "decision", "rationale", "would_change_if"):
                require_string(decision[key], f"architecture decision {decision_id}.{key}", problems)
            if decision["status"] not in EXPECTED_ARCHITECTURE_STATUS:
                add_problem(problems, f"architecture decision {decision_id}: invalid status")
            if decision["status"] == "BLOCKED_ON_EXPERIMENT" and decision["blocked_by"] is None:
                add_problem(problems, f"architecture decision {decision_id}: blocked status has no blocked_by token")
            validate_reference_list(decision["evidence_refs"], ledger_ids, f"architecture decision {decision_id}.evidence_refs", problems, required=False)
            blocked_by = decision["blocked_by"]
            if blocked_by is not None:
                if not isinstance(blocked_by, str) or not blocked_by:
                    add_problem(problems, f"architecture decision {decision_id}.blocked_by: invalid token")
                else:
                    tokens = blocked_by.split("+")
                    if any(not token or token not in unknown_ids and token not in dependency_ids for token in tokens):
                        add_problem(problems, f"architecture decision {decision_id}.blocked_by: unresolved token {blocked_by}")
    return ids


def validate_dependency_graph(
    graph: Any,
    ledger_ids: set[str],
    root: Path,
    actual_hash: str,
    repository: GitRepository,
    problems: list[str],
) -> set[str]:
    if not exact_top_level(graph, EXPECTED_DEPENDENCY_KEYS, "dependency graph", problems):
        return set()
    validate_common_metadata(graph, DEPENDENCY_SCHEMA, "dependency graph", root, actual_hash, repository, problems)
    if graph.get("node_types") != EXPECTED_NODE_TYPES:
        add_problem(problems, "dependency graph.node_types: values or order do not match schema")
    nodes = graph.get("nodes")
    ids = unique_ids(nodes, "id", "dependency graph.nodes", problems, 31)
    if isinstance(nodes, list):
        for index, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            required = {"id", "type", "label", "status"}
            if not require_keys(node, required, f"dependency graph.nodes[{index}]", problems):
                continue
            if set(node) - (required | {"evidence_refs", "note"}):
                add_problem(problems, f"dependency graph.nodes[{index}]: unexpected fields")
            if node["type"] not in EXPECTED_NODE_TYPES:
                add_problem(problems, f"dependency graph node {node['id']}: invalid type")
            require_string(node["label"], f"dependency graph node {node['id']}.label", problems)
            require_string(node["status"], f"dependency graph node {node['id']}.status", problems)
            if node["status"] not in EXPECTED_GRAPH_STATUSES:
                add_problem(problems, f"dependency graph node {node['id']}: invalid status")
            if "evidence_refs" in node:
                validate_reference_list(node["evidence_refs"], ledger_ids, f"dependency graph node {node['id']}.evidence_refs", problems, required=False)
            if "note" in node:
                require_string(node["note"], f"dependency graph node {node['id']}.note", problems)
    edges = graph.get("edges")
    if not isinstance(edges, list) or not edges:
        add_problem(problems, "dependency graph.edges: expected non-empty list")
    else:
        for index, edge in enumerate(edges):
            label = f"dependency graph.edges[{index}]"
            if not isinstance(edge, dict) or not require_keys(edge, {"from", "to", "relation"}, label, problems):
                continue
            if set(edge) - {"from", "to", "relation", "note"}:
                add_problem(problems, f"{label}: unexpected fields")
            if edge["from"] not in ids:
                add_problem(problems, f"{label}.from: unknown node {edge['from']!r}")
            if edge["to"] not in ids:
                add_problem(problems, f"{label}.to: unknown node {edge['to']!r}")
            if edge["from"] == edge["to"]:
                add_problem(problems, f"{label}: self-edge")
            require_string(edge["relation"], f"{label}.relation", problems)
            if edge["relation"] not in EXPECTED_EDGE_RELATIONS:
                add_problem(problems, f"{label}.relation: unknown relation kind")
            if "note" in edge:
                require_string(edge["note"], f"{label}.note", problems)
    paths = graph.get("critical_paths")
    path_names: set[str] = set()
    if not isinstance(paths, list) or not paths:
        add_problem(problems, "dependency graph.critical_paths: expected non-empty list")
    else:
        for index, path in enumerate(paths):
            label = f"dependency graph.critical_paths[{index}]"
            if not isinstance(path, dict) or not require_keys(path, {"name", "chain", "note"}, label, problems, exact=True):
                continue
            name = path["name"]
            if name in path_names:
                add_problem(problems, f"dependency graph.critical_paths: duplicate name {name}")
            path_names.add(name)
            require_string(name, f"{label}.name", problems)
            validate_reference_list(path["chain"], ids, f"{label}.chain", problems)
            if isinstance(path["chain"], list) and all(isinstance(item, str) for item in path["chain"]) and len(path["chain"]) != len(set(path["chain"])):
                add_problem(problems, f"{label}.chain: contains duplicate nodes")
            require_string(path["note"], f"{label}.note", problems)
    if path_names != EXPECTED_GRAPH_PATH_NAMES:
        add_problem(problems, "dependency graph.critical_paths: current path set does not match phase-3 state")
    return ids


def validate_tree(
    tree: Any,
    belief_ids: set[str],
    architecture_ids: set[str],
    root: Path,
    actual_hash: str,
    repository: GitRepository,
    problems: list[str],
) -> set[str]:
    if not exact_top_level(tree, EXPECTED_TREE_KEYS, "decision tree", problems):
        return set()
    validate_common_metadata(tree, TREE_SCHEMA, "decision tree", root, actual_hash, repository, problems)
    require_string(tree.get("usage"), "decision tree.usage", problems)
    root_id = tree.get("root")
    nodes = tree.get("nodes")
    if not isinstance(root_id, str) or not root_id:
        add_problem(problems, "decision tree.root: expected non-empty string")
        root_id = ""
    if not isinstance(nodes, dict) or not nodes:
        add_problem(problems, "decision tree.nodes: expected non-empty object")
        return set()
    node_ids = set(nodes)
    if root_id not in node_ids:
        add_problem(problems, f"decision tree.root: unknown node {root_id!r}")
    state = tree.get("current_evidence_state")
    if not isinstance(state, dict) or set(state) != EXPECTED_TREE_STATE_KEYS or any(not isinstance(v, str) or not v for v in state.values()):
        add_problem(problems, "decision tree.current_evidence_state: keys or values do not match phase-3 state")
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for node_id, node in nodes.items():
        label = f"decision tree node {node_id}"
        if not isinstance(node, dict):
            add_problem(problems, f"{label}: expected object")
            continue
        if set(node) - ALLOWED_TREE_NODE_FIELDS:
            add_problem(problems, f"{label}: unexpected fields")
        has_question = "question" in node
        has_action = "action" in node
        if has_question and has_action:
            add_problem(problems, f"{label}: cannot have both question and action")
        if has_question:
            require_string(node["question"], f"{label}.question", problems)
        elif has_action:
            require_string(node["action"], f"{label}.action", problems)
        elif "branches" not in node or not ("label" in node or "rule" in node):
            add_problem(problems, f"{label}: expected question, action, or labelled branch state")
        if "branches" in node:
            branches = node["branches"]
            if not isinstance(branches, list) or not branches:
                add_problem(problems, f"{label}.branches: expected non-empty list")
            else:
                for index, branch in enumerate(branches):
                    branch_label = f"{label}.branches[{index}]"
                    if not isinstance(branch, dict) or not require_keys(branch, {"if", "then"}, branch_label, problems):
                        continue
                    if set(branch) - {"if", "then", "note"}:
                        add_problem(problems, f"{branch_label}: unexpected fields")
                    require_string(branch["if"], f"{branch_label}.if", problems)
                    target = branch["then"]
                    if target not in node_ids:
                        add_problem(problems, f"{branch_label}.then: unknown node {target!r}")
                    else:
                        adjacency[node_id].add(target)
                    if "note" in branch:
                        require_string(branch["note"], f"{branch_label}.note", problems)
        if "next" in node:
            if node["next"] not in node_ids:
                add_problem(problems, f"{label}.next: unknown node {node['next']!r}")
            else:
                adjacency[node_id].add(node["next"])
        if "prohibitions" in node and (not isinstance(node["prohibitions"], list) or any(not isinstance(x, str) or not x for x in node["prohibitions"])):
            add_problem(problems, f"{label}.prohibitions: expected string list")
        if "belief_updates" in node:
            updates = node["belief_updates"]
            if not isinstance(updates, dict) or any(key not in belief_ids or not isinstance(value, str) or not value for key, value in updates.items()):
                add_problem(problems, f"{label}.belief_updates: unresolved belief or invalid value")
        if "decision_updates" in node:
            updates = node["decision_updates"]
            if not isinstance(updates, dict) or any(key not in architecture_ids or not isinstance(value, str) or not value for key, value in updates.items()):
                add_problem(problems, f"{label}.decision_updates: unresolved decision or invalid value")
        if "path_record" in node and (not isinstance(node["path_record"], list) or any(not isinstance(x, str) or not x for x in node["path_record"])):
            add_problem(problems, f"{label}.path_record: expected string list")
    reachable: set[str] = set()
    stack = [root_id] if root_id in node_ids else []
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        stack.extend(adjacency[current] - reachable)
    if node_ids - reachable:
        add_problem(problems, f"decision tree: unreachable nodes {sorted(node_ids - reachable)}")
    return node_ids


def validate_model(
    model: Any,
    ledger: Any,
    beliefs: Any,
    architecture: Any,
    dependency: Any,
    tree: Any,
    root: Path,
    actual_hash: str,
    repository: GitRepository,
    problems: list[str],
) -> None:
    if not exact_top_level(model, EXPECTED_MODEL_KEYS, "decision model", problems):
        return
    validate_common_metadata(model, MODEL_SCHEMA, "decision model", root, actual_hash, repository, problems)
    ledger_rows = ledger.get("experiments", []) if isinstance(ledger, dict) else []
    ledger_by_id = {
        row.get("id"): row for row in ledger_rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    ledger_ids = set(ledger_by_id)
    belief_ids = {row.get("belief_id") for row in beliefs.get("beliefs", []) if isinstance(row, dict) and isinstance(row.get("belief_id"), str)} if isinstance(beliefs, dict) else set()
    architecture_ids = {row.get("id") for row in architecture.get("decisions", []) if isinstance(row, dict) and isinstance(row.get("id"), str)} if isinstance(architecture, dict) else set()
    dependency_ids = {row.get("id") for row in dependency.get("nodes", []) if isinstance(row, dict) and isinstance(row.get("id"), str)} if isinstance(dependency, dict) else set()
    belief_rows = beliefs.get("beliefs", []) if isinstance(beliefs, dict) else []
    ledger_status = {row.get("id"): row.get("status") for row in ledger_rows if isinstance(row, dict)}
    bad_positive_statuses = {"IMPLEMENTED_NOT_EXECUTED", "NOT_TESTED", "SPECULATIVE"}
    for belief in belief_rows:
        if not isinstance(belief, dict) or belief.get("current_status") not in {"STRONGLY_SUPPORTED", "SUPPORTED"}:
            continue
        belief_id = belief.get("belief_id", "<missing>")
        for source in belief.get("supporting_experiments", []) if isinstance(belief.get("supporting_experiments"), list) else []:
            if ledger_status.get(source) in bad_positive_statuses:
                add_problem(problems, f"belief {belief_id}: cites {source} as positive evidence despite status {ledger_status[source]}")
    for key in ("historical_snapshot",):
        value = model.get(key)
        fields = {"date", "phase", "branch", "basis_head", "ledger_entries", "note"}
        if not require_keys(value, fields, f"decision model.{key}", problems, exact=True):
            continue
        require_string(value["date"], f"decision model.{key}.date", problems)
        if value["phase"] != 2:
            add_problem(problems, f"decision model.{key}.phase: expected historical phase 2")
        require_string(value["branch"], f"decision model.{key}.branch", problems)
        validate_full_commit(value["basis_head"], f"decision model.{key}.basis_head", repository, problems)
        if value["ledger_entries"] != 78:
            add_problem(problems, f"decision model.{key}.ledger_entries: expected 78")
        require_string(value["note"], f"decision model.{key}.note", problems)
    phase1 = model.get("phase1_basis")
    phase1_fields = {"branch", "completion_sha", "ledger", "ledger_entries", "historical_ledger_entries", "current_ledger_entries", "historical"}
    if not require_keys(phase1, phase1_fields, "decision model.phase1_basis", problems, exact=True):
        phase1 = {}
    else:
        require_string(phase1["branch"], "decision model.phase1_basis.branch", problems)
        validate_full_commit(phase1["completion_sha"], "decision model.phase1_basis.completion_sha", repository, problems)
        if phase1["ledger"] != DEFAULT_LEDGER:
            add_problem(problems, "decision model.phase1_basis.ledger: wrong ledger path")
        if phase1["ledger_entries"] != 78 or phase1["historical_ledger_entries"] != 78 or phase1["current_ledger_entries"] != EXPECTED_LEDGER_COUNT:
            add_problem(problems, "decision model.phase1_basis: ledger counts do not reconcile")
        if phase1["historical"] is not True:
            add_problem(problems, "decision model.phase1_basis.historical: expected true")
    consolidation = model.get("consolidation_basis")
    consolidation_fields = {"date", "ledger", "ledger_entries", "source_manifest", "pre_consolidation_head_commit"}
    if not require_keys(consolidation, consolidation_fields, "decision model.consolidation_basis", problems, exact=True):
        consolidation = {}
    else:
        require_string(consolidation["date"], "decision model.consolidation_basis.date", problems)
        if consolidation["ledger"] != DEFAULT_LEDGER:
            add_problem(problems, "decision model.consolidation_basis.ledger: wrong ledger path")
        if consolidation["ledger_entries"] != len(ledger_rows) or consolidation["ledger_entries"] != EXPECTED_LEDGER_COUNT:
            add_problem(problems, "decision model.consolidation_basis.ledger_entries: does not match JSON ledger")
        if consolidation["source_manifest"] != MANIFEST_RELATIVE:
            add_problem(problems, "decision model.consolidation_basis.source_manifest: wrong path")
        pre_head = model.get("pre_consolidation_head")
        pre_commit = pre_head.get("commit") if isinstance(pre_head, dict) else None
        if consolidation["pre_consolidation_head_commit"] != pre_commit:
            add_problem(problems, "decision model.consolidation_basis: pre-consolidation commit disagrees")
        validate_full_commit(consolidation["pre_consolidation_head_commit"], "decision model.consolidation_basis.pre_consolidation_head_commit", repository, problems)
    require_string_list(model.get("evidence_snapshot_delta_since_phase1"), "decision model.evidence_snapshot_delta_since_phase1", problems)
    for key in ("beliefs_ref", "dependencies_ref", "decision_tree_ref", "kill_criteria_ref"):
        value = model.get(key)
        require_string(value, f"decision model.{key}", problems)
        if isinstance(value, str):
            relative = value.split(" ", 1)[0]
            if key == "beliefs_ref" and "(B01-B32)" not in value:
                add_problem(problems, "decision model.beliefs_ref: current belief range is missing")
            if not working_tree_file(root, relative):
                add_problem(problems, f"decision model.{key}: referenced file is missing: {relative}")
    evidence_refs = model.get("evidence_refs")
    if not require_keys(evidence_refs, {"note", "primary_sources"}, "decision model.evidence_refs", problems, exact=True):
        evidence_refs = {}
    else:
        require_string(evidence_refs["note"], "decision model.evidence_refs.note", problems)
        validate_reference_list(evidence_refs["primary_sources"], ledger_ids, "decision model.evidence_refs.primary_sources", problems)
    unknowns = model.get("unknowns")
    unknown_ids = unique_ids(unknowns, "id", "decision model.unknowns", problems, 11)
    unknown_fields = {"id", "status", "question", "discriminating_experiment", "decisions_blocked", "tier", "evidence_refs"}
    allowed_unknown_fields = unknown_fields | {"answer", "no_gpu"}
    unknown_rows = unknowns if isinstance(unknowns, list) else []
    candidates = model.get("candidate_experiments")
    candidate_ids = unique_ids(candidates, "id", "decision model.candidate_experiments", problems, 18)
    candidate_fields = {"id", "status", "scope", "gpu_hours", "scores", "value", "unlocks", "kill_criterion", "notes"}
    candidate_rows = candidates if isinstance(candidates, list) else []
    model_blocked = model.get("blocked_decisions")
    model_blocked_fields = {"decision", "blocked_by_unknown"}
    for index, unknown in enumerate(unknown_rows):
        label = f"decision model.unknowns[{index}]"
        if not isinstance(unknown, dict) or not require_keys(unknown, unknown_fields, label, problems):
            continue
        if set(unknown) - allowed_unknown_fields:
            add_problem(problems, f"{label}: unexpected fields")
        unknown_id = unknown["id"]
        if unknown["status"] not in EXPECTED_UNKNOWN_STATUS:
            add_problem(problems, f"unknown {unknown_id}: invalid status")
        require_string(unknown["question"], f"unknown {unknown_id}.question", problems)
        require_string(unknown["discriminating_experiment"], f"unknown {unknown_id}.discriminating_experiment", problems)
        if unknown["discriminating_experiment"] not in candidate_ids:
            add_problem(problems, f"unknown {unknown_id}: discriminating experiment is not a candidate")
        if unknown["tier"] not in {1, 2, 3}:
            add_problem(problems, f"unknown {unknown_id}: invalid tier")
        if "no_gpu" in unknown and not isinstance(unknown["no_gpu"], bool):
            add_problem(problems, f"unknown {unknown_id}.no_gpu: expected boolean")
        if unknown["status"] == "ANSWERED_NO" and "answer" not in unknown:
            add_problem(problems, f"unknown {unknown_id}: answered unknown lacks answer")
        if "answer" in unknown:
            require_string(unknown["answer"], f"unknown {unknown_id}.answer", problems)
        validate_reference_list(unknown["decisions_blocked"], architecture_ids, f"unknown {unknown_id}.decisions_blocked", problems, required=False)
        validate_reference_list(unknown["evidence_refs"], ledger_ids, f"unknown {unknown_id}.evidence_refs", problems, required=False)
    known_kill_ids = model.get("kill_criteria") if isinstance(model.get("kill_criteria"), list) else []
    for index, candidate in enumerate(candidate_rows):
        label = f"decision model.candidate_experiments[{index}]"
        if not isinstance(candidate, dict) or not require_keys(candidate, candidate_fields, label, problems, exact=True):
            continue
        candidate_id = candidate["id"]
        if candidate["status"] not in EXPECTED_CANDIDATE_STATUS:
            add_problem(problems, f"candidate {candidate_id}: invalid status")
        for key in ("scope", "kill_criterion", "notes"):
            require_string(candidate[key], f"candidate {candidate_id}.{key}", problems)
        kill_text = candidate["kill_criterion"] if isinstance(candidate["kill_criterion"], str) else ""
        if not isinstance(candidate["gpu_hours"], (int, float)) or isinstance(candidate["gpu_hours"], bool) or not math.isfinite(candidate["gpu_hours"]) or candidate["gpu_hours"] < 0:
            add_problem(problems, f"candidate {candidate_id}.gpu_hours: invalid value")
        scores = candidate["scores"]
        if not isinstance(scores, dict) or set(scores) != {"info_gain", "architecture_impact", "dependency_unlocks", "p_discriminate", "cost"}:
            add_problem(problems, f"candidate {candidate_id}.scores: invalid schema")
        else:
            for key, value in scores.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                    add_problem(problems, f"candidate {candidate_id}.scores.{key}: invalid value")
            if not isinstance(scores["cost"], (int, float)) or not 1 <= scores["cost"] <= 5:
                add_problem(problems, f"candidate {candidate_id}.scores.cost: outside 1..5")
            for key in ("info_gain", "architecture_impact", "dependency_unlocks"):
                if not isinstance(scores[key], int) or isinstance(scores[key], bool) or not 0 <= scores[key] <= 5:
                    add_problem(problems, f"candidate {candidate_id}.scores.{key}: expected integer 0..5")
            if not isinstance(scores["cost"], int) or isinstance(scores["cost"], bool) or not 1 <= scores["cost"] <= 5:
                add_problem(problems, f"candidate {candidate_id}.scores.cost: expected integer 1..5")
            if not isinstance(scores["p_discriminate"], (int, float)) or isinstance(scores["p_discriminate"], bool) or not 0 <= scores["p_discriminate"] <= 1:
                add_problem(problems, f"candidate {candidate_id}.scores.p_discriminate: outside 0..1")
        if not isinstance(candidate["value"], (int, float)) or isinstance(candidate["value"], bool) or not math.isfinite(candidate["value"]) or candidate["value"] < 0:
            add_problem(problems, f"candidate {candidate_id}.value: invalid value")
        elif candidate["value"] >= 3 and not candidate["unlocks"]:
            add_problem(problems, f"candidate {candidate_id}: high-value candidate has no unlock targets")
        if isinstance(candidate["scores"], dict) and candidate["scores"].get("cost", 0) >= 4 and not kill_text.strip():
            add_problem(problems, f"candidate {candidate_id}: expensive candidate has no kill criterion")
        validate_reference_list(candidate["unlocks"], architecture_ids | unknown_ids, f"candidate {candidate_id}.unlocks", problems, required=False)
        for token in re.findall(r"\bK\d{2}\b", kill_text):
            if token not in known_kill_ids:
                add_problem(problems, f"candidate {candidate_id}: unknown kill criterion {token}")
    scoring = model.get("scoring_function")
    if not require_keys(scoring, {"formula", "scales", "caveat"}, "decision model.scoring_function", problems, exact=True):
        pass
    else:
        for key in ("formula", "scales", "caveat"):
            require_string(scoring[key], f"decision model.scoring_function.{key}", problems)
    ranking = model.get("ranking")
    if not require_keys(ranking, {"top_three", "basis", "completed_not_ranked"}, "decision model.ranking", problems, exact=True):
        ranking = {}
    else:
        validate_reference_list(ranking["top_three"], candidate_ids, "decision model.ranking.top_three", problems)
        validate_reference_list(ranking["completed_not_ranked"], candidate_ids, "decision model.ranking.completed_not_ranked", problems)
        require_string(ranking["basis"], "decision model.ranking.basis", problems)
    sensitivity = model.get("ranking_sensitivity")
    if not require_keys(sensitivity, {"method", "result"}, "decision model.ranking_sensitivity", problems, exact=True):
        pass
    else:
        require_string(sensitivity["method"], "decision model.ranking_sensitivity.method", problems)
        if not isinstance(sensitivity["result"], dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in sensitivity["result"].items()):
            add_problem(problems, "decision model.ranking_sensitivity.result: invalid result map")
    if not isinstance(model_blocked, list) or not model_blocked:
        add_problem(problems, "decision model.blocked_decisions: expected non-empty list")
    else:
        decisions_seen: list[str] = []
        for index, blocked in enumerate(model_blocked):
            label = f"decision model.blocked_decisions[{index}]"
            if not isinstance(blocked, dict) or not require_keys(blocked, model_blocked_fields, label, problems):
                continue
            if set(blocked) - (model_blocked_fields | {"note"}):
                add_problem(problems, f"{label}: unexpected fields")
            decision = blocked["decision"]
            decisions_seen.append(decision)
            if decision not in architecture_ids:
                add_problem(problems, f"{label}.decision: unknown architecture decision {decision}")
            if not isinstance(blocked["blocked_by_unknown"], str) or not blocked["blocked_by_unknown"]:
                add_problem(problems, f"{label}.blocked_by_unknown: expected token string")
            else:
                tokens = blocked["blocked_by_unknown"].split("+")
                if any(token not in unknown_ids for token in tokens):
                    add_problem(problems, f"{label}.blocked_by_unknown: unresolved unknown token")
            if "note" in blocked:
                require_string(blocked["note"], f"{label}.note", problems)
        if len(decisions_seen) != len(set(decisions_seen)):
            add_problem(problems, "decision model.blocked_decisions: duplicate decisions")
    kill_ids = model.get("kill_criteria")
    if not require_string_list(kill_ids, "decision model.kill_criteria", problems):
        kill_ids = []
    elif any(not re.fullmatch(r"K\d{2}", value) for value in kill_ids):
        add_problem(problems, "decision model.kill_criteria: invalid criterion ID")
    details = model.get("kill_criteria_detail")
    unique_ids(details, "id", "decision model.kill_criteria_detail", problems)
    if isinstance(details, list):
        for index, detail in enumerate(details):
            if not isinstance(detail, dict) or not require_keys(detail, {"id", "status", "effect"}, f"decision model.kill_criteria_detail[{index}]", problems, exact=True):
                continue
            if detail["id"] not in kill_ids:
                add_problem(problems, f"kill criterion detail {detail['id']}: not listed in kill_criteria")
            if detail["status"] not in EXPECTED_KILL_STATUS:
                add_problem(problems, f"kill criterion detail {detail['id']}: invalid status")
            require_string(detail["effect"], f"kill criterion detail {detail['id']}.effect", problems)
    invariants = model.get("invariants")
    if not require_string_list(invariants, "decision model.invariants", problems):
        pass
    require_string(model.get("authorization_boundary"), "decision model.authorization_boundary", problems)


def validate_markdown_ledger(path: Path, ledger: dict[str, Any], problems: list[str]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        add_problem(problems, f"human evidence ledger: cannot read {path}: {exc}")
        return
    status_heading = next((i for i, line in enumerate(lines) if line.strip() == "## Status counts"), None)
    index_heading = next((i for i, line in enumerate(lines) if line.strip() == "## Complete experiment evidence index"), None)
    if status_heading is None or index_heading is None:
        add_problem(problems, "human evidence ledger: required status/index headings are missing")
        return
    status_end = next((i for i in range(status_heading + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    parsed_counts: dict[str, int] = {}
    for line in lines[status_heading + 1:status_end]:
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] not in EXPECTED_STATUS_ENUM:
            continue
        value = cells[1].replace(",", "")
        if not re.fullmatch(r"\d+", value):
            add_problem(problems, f"human evidence ledger: invalid count for {cells[0]}")
        elif cells[0] in parsed_counts:
            add_problem(problems, f"human evidence ledger: duplicate count for {cells[0]}")
        else:
            parsed_counts[cells[0]] = int(value)
    expected_counts = status_counts(ledger)
    if set(parsed_counts) != set(EXPECTED_STATUS_ENUM):
        add_problem(problems, "human evidence ledger: status table does not contain the exact status enum")
    if parsed_counts != expected_counts:
        add_problem(problems, f"human evidence ledger: status counts disagree with JSON: {parsed_counts} != {expected_counts}")
    total_match = re.search(r"\*\*(\d+) ledger entries", "\n".join(lines[status_heading:status_end]))
    if total_match is None or int(total_match.group(1)) != len(ledger.get("experiments", [])):
        add_problem(problems, "human evidence ledger: declared total does not match JSON row count")
    index_end = next((i for i in range(index_heading + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    index_rows: dict[str, tuple[str, str]] = {}
    for line in lines[index_heading + 1:index_end]:
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in {"ID", "---"} or set(cells[0]) <= {"-", ":"}:
            continue
        experiment_id = cells[0].strip("*` ")
        if not experiment_id:
            continue
        if experiment_id in index_rows:
            add_problem(problems, f"human evidence ledger: duplicate experiment index row {experiment_id}")
        else:
            index_rows[experiment_id] = (cells[1].strip("*` "), cells[2].strip("*` "))
    json_rows = {row.get("id"): row for row in ledger.get("experiments", []) if isinstance(row, dict)}
    if set(index_rows) != set(json_rows):
        missing = sorted(set(json_rows) - set(index_rows))
        extra = sorted(set(index_rows) - set(json_rows))
        add_problem(problems, f"human evidence ledger: experiment index mismatch; missing={missing}, extra={extra}")
    for experiment_id, (status, replication) in index_rows.items():
        row = json_rows.get(experiment_id)
        if row is not None and (status != row.get("status") or replication != row.get("replication")):
            add_problem(problems, f"human evidence ledger: index metadata mismatch for {experiment_id}")
    if len(index_rows) != EXPECTED_LEDGER_COUNT:
        add_problem(problems, f"human evidence ledger: expected {EXPECTED_LEDGER_COUNT} index rows, found {len(index_rows)}")


def validate_current_invariants(
    ledger: dict[str, Any],
    beliefs: dict[str, Any],
    architecture: dict[str, Any],
    model: dict[str, Any],
    dependency: dict[str, Any],
    tree: dict[str, Any],
    problems: list[str],
) -> None:
    ledger_rows = ledger.get("experiments", []) if isinstance(ledger, dict) else []
    candidate_rows = model.get("candidate_experiments", []) if isinstance(model, dict) else []
    dependency_rows = dependency.get("nodes", []) if isinstance(dependency, dict) else []
    belief_rows = beliefs.get("beliefs", []) if isinstance(beliefs, dict) else []
    if not isinstance(ledger_rows, list):
        ledger_rows = []
    if not isinstance(candidate_rows, list):
        candidate_rows = []
    if not isinstance(dependency_rows, list):
        dependency_rows = []
    if not isinstance(belief_rows, list):
        belief_rows = []
    ledger_by_id = {row.get("id"): row for row in ledger_rows if isinstance(row, dict)}
    candidate_by_id = {row.get("id"): row for row in candidate_rows if isinstance(row, dict)}
    graph_by_id = {row.get("id"): row for row in dependency_rows if isinstance(row, dict)}
    belief_by_id = {row.get("belief_id"): row for row in belief_rows if isinstance(row, dict)}
    state = tree.get("current_evidence_state", {}) if isinstance(tree, dict) else {}
    expected_statuses = {
        "CYR-GPU-014-R1C": "CONTRADICTED",
        "CS-TRANSFER-001": "INCONCLUSIVE",
        "FORMATION-MUX-001-S5-V8": "INCONCLUSIVE",
        "FORMATION-MUX-001-V12-FRONTIER-PARTIAL": "IN_PROGRESS",
        "ROLE-TRANSFER-001": "NOT_TESTED",
        "BRAMASTRA-K8-20260922": "INCONCLUSIVE",
        "ARK-020-V4": "IMPLEMENTED_NOT_EXECUTED",
        "GANDIVA-TPU-100M-PREFLIGHT": "NOT_TESTED",
    }
    for experiment_id, expected in expected_statuses.items():
        row = ledger_by_id.get(experiment_id)
        if row is None or row.get("status") != expected:
            add_problem(problems, f"phase-3 invariant: {experiment_id} must be {expected}")
    result_expectations = {
        "CYR-GPU-014-R1C": ("COMPLETE 24/24", "SOFTMAX_COMPETITION_NOT_SUFFICIENT"),
        "CS-TRANSFER-001": ("COMPLETE", "PARTIAL_OR_INTERACTION"),
        "FORMATION-MUX-001-S5-V8": ("COMPLETE 24/24", "near the zero baseline"),
        "FORMATION-MUX-001-V12-FRONTIER-PARTIAL": ("2/24", "partial"),
        "ROLE-TRANSFER-001": ("PREREGISTERED_EXECUTION_BLOCKED", "no trainer", "scientific outcome"),
        "BRAMASTRA-K8-20260922": ("COMPLETED_ENGINEERING_PARTIAL",),
        "ARK-020-V4": ("DO_NOT_RUN", "NOT_EXECUTED"),
        "GANDIVA-TPU-100M-PREFLIGHT": ("No TPU", "qualification"),
    }
    for experiment_id, needles in result_expectations.items():
        result = str(ledger_by_id.get(experiment_id, {}).get("result", ""))
        for needle in needles:
            if needle.lower() not in result.lower():
                add_problem(problems, f"phase-3 invariant: {experiment_id} result lacks {needle}")
    candidate_expectations = {
        "EXEC-R1C": "complete",
        "CS-TRANSFER-001": "complete",
        "FMUX-CONTROL-METRIC-PREFLIGHT": "next",
        "TIED-ROW-GEOMETRY-WD-001": "superseded-by-preregistered-design",
        "ROLE-TRANSFER-001": "preregistered-execution-blocked",
        "CORPUS-REGEN": "required-parallel",
        "ARK-020-EXECUTION": "DO_NOT_RUN",
        "BRAMASTRA-K8-20260922": "completed-engineering-partial",
        "GANDIVA-TPU-100M-PREFLIGHT": "engineering-not-run",
    }
    for candidate_id, expected in candidate_expectations.items():
        if candidate_by_id.get(candidate_id, {}).get("status") != expected:
            add_problem(problems, f"phase-3 invariant: candidate {candidate_id} must be {expected}")
    graph_expectations = {
        "N-R1C": "COMPLETE",
        "N-K01": "FIRED",
        "N-CTRANSFER": "COMPLETE",
        "N-FMUX-S5": "COMPLETE_BUT_FLOOR_LIMITED",
        "N-FMUX-V12": "PARTIAL_IN_PROGRESS",
        "N-FMUX-RECOVERY": "BLOCKED",
        "N-ROLE-TRANSFER": "PREREGISTERED_EXECUTION_BLOCKED",
        "N-TIED-PLACEHOLDER": "SUPERSEDED_NO_EXECUTION",
        "N-K8": "ENGINEERING_ONLY_COMPLETED_PARTIAL",
        "N-ARK020": "DO_NOT_RUN",
        "N-TPU": "ENGINEERING_ONLY_NOT_RUN",
        "N-500M-AUTH": "BLOCKED",
        "N-CORE500": "PENDING_BLOCKED",
    }
    for node_id, expected in graph_expectations.items():
        if graph_by_id.get(node_id, {}).get("status") != expected:
            add_problem(problems, f"phase-3 invariant: graph node {node_id} must be {expected}")
    tree_expectations = {
        "R1C": ("COMPLETE", "SOFTMAX_COMPETITION_NOT_SUFFICIENT"),
        "K01": "FIRED",
        "CS-TRANSFER-001": ("COMPLETE", "PARTIAL_OR_INTERACTION"),
        "FORMATION-MUX-001-S5-V8": "COMPLETE_BUT_FLOOR_LIMITED",
        "FORMATION-MUX-001-V12-FRONTIER-PARTIAL": ("IN_PROGRESS", "2/24"),
        "ROLE-TRANSFER-001": ("PREREGISTERED_EXECUTION_BLOCKED", "no trainer", "sealed evaluation", "result"),
        "ARK-020-V4": ("DO_NOT_RUN", "NOT_EXECUTED"),
        "K8": "ENGINEERING_ONLY_COMPLETED_PARTIAL",
        "TPU": "ENGINEERING_ONLY_NOT_RUN",
    }
    for key, expected in tree_expectations.items():
        value = str(state.get(key, ""))
        if isinstance(expected, tuple):
            if any(needle not in value for needle in expected):
                add_problem(problems, f"phase-3 invariant: decision tree {key} is not reconciled")
        elif expected not in value:
            add_problem(problems, f"phase-3 invariant: decision tree {key} is not reconciled")
    detail_rows = model.get("kill_criteria_detail", []) if isinstance(model, dict) else []
    if not isinstance(detail_rows, list):
        detail_rows = []
    detail = {row.get("id"): row for row in detail_rows if isinstance(row, dict)}
    if detail.get("K01", {}).get("status") != "FIRED":
        add_problem(problems, "phase-3 invariant: K01 must be FIRED")
    if belief_by_id.get("B05", {}).get("current_status") != "CONTRADICTED":
        add_problem(problems, "phase-3 invariant: B05 must be CONTRADICTED")
    if belief_by_id.get("B07", {}).get("current_status") != "OPEN":
        add_problem(problems, "phase-3 invariant: B07 must remain OPEN")
    expected_top_three = ["FMUX-CONTROL-METRIC-PREFLIGHT", "ROLE-TRANSFER-001", "CORPUS-REGEN"]
    ranking = model.get("ranking") if isinstance(model, dict) else {}
    if not isinstance(ranking, dict) or ranking.get("top_three") != expected_top_three:
        add_problem(problems, "phase-3 invariant: top_three is not the exact required sequence")
    boundary = str(model.get("authorization_boundary", "")).lower()
    required_boundary_terms = ("no candidate or decision", "production", "pre500m", "250m", "500m", "cognition", "agi")
    if any(term not in boundary for term in required_boundary_terms):
        add_problem(problems, "phase-3 invariant: model authorization boundary is incomplete")
    architecture_boundary = str(architecture.get("authorization_boundary", "")).lower()
    if any(term not in architecture_boundary for term in ("no row authorizes", "production", "pre500m", "250m", "500m", "cognition", "agi")):
        add_problem(problems, "phase-3 invariant: architecture authorization boundary is incomplete")
    candidate_rows = model.get("candidate_experiments", []) if isinstance(model, dict) else []
    if not isinstance(candidate_rows, list):
        candidate_rows = []
    for candidate in candidate_rows:
        if not isinstance(candidate, dict):
            continue
        status = str(candidate.get("status", "")).upper()
        if "AUTHORIZED" in status or "APPROVED" in status:
            add_problem(problems, f"phase-3 invariant: candidate {candidate.get('id')} claims authorization")
    dependency_rows = dependency.get("nodes", []) if isinstance(dependency, dict) else []
    if not isinstance(dependency_rows, list):
        dependency_rows = []
    for node in dependency_rows:
        if isinstance(node, dict) and str(node.get("status", "")).upper() in {"AUTHORIZED", "APPROVED", "PRODUCTION_AUTHORIZED"}:
            add_problem(problems, f"phase-3 invariant: graph node {node.get('id')} claims authorization")
    if "false" not in str(ledger_by_id.get("ARK-020-V4", {}).get("result", "")).lower():
        add_problem(problems, "phase-3 invariant: ARK-020 authorization flags are not explicitly false")


def validate_documents(root: Path) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    repository = GitRepository(root)
    documents = load_documents(root, problems)
    actual_hash, manifest_paths = validate_manifest_hash_and_ledger(root, repository, documents, problems)
    ledger = documents.get("ledger", {})
    beliefs = documents.get("beliefs", {})
    architecture = documents.get("architecture", {})
    model = documents.get("model", {})
    dependency = documents.get("dependency", {})
    tree = documents.get("tree", {})
    ledger_ids = {
        row.get("id") for row in ledger.get("experiments", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    } if isinstance(ledger, dict) else set()
    dependency_ids = validate_dependency_graph(dependency, ledger_ids, root, actual_hash, repository, problems)
    unknown_ids = {
        row.get("id") for row in model.get("unknowns", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    } if isinstance(model, dict) else set()
    architecture_ids = validate_architecture_ledger(architecture, ledger_ids, unknown_ids, dependency_ids, root, actual_hash, repository, problems)
    belief_ids = validate_belief_registry(beliefs, ledger_ids, architecture_ids, root, actual_hash, repository, problems)
    validate_model(model, ledger, beliefs, architecture, dependency, tree, root, actual_hash, repository, problems)
    validate_tree(tree, belief_ids, architecture_ids, root, actual_hash, repository, problems)
    if isinstance(ledger, dict):
        validate_markdown_ledger(root / "docs/research/EXPERIMENT_EVIDENCE_LEDGER.md", ledger, problems)
    validate_current_invariants(ledger, beliefs, architecture, model, dependency, tree, problems)
    return problems, {
        "ledger": ledger,
        "beliefs": beliefs,
        "architecture": architecture,
        "model": model,
        "dependency": dependency,
        "tree": tree,
        "manifest_paths": manifest_paths,
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    problems, documents = validate_documents(ROOT)
    ledger = documents["ledger"]
    model = documents["model"]
    dependency = documents["dependency"]
    print(
        f"decision model: {len(model.get('candidate_experiments', [])) if isinstance(model, dict) else 0} candidates, "
        f"{len(model.get('unknowns', [])) if isinstance(model, dict) else 0} unknowns, "
        f"{len(model.get('blocked_decisions', [])) if isinstance(model, dict) else 0} blocked decisions, "
        f"{len(documents['beliefs'].get('beliefs', [])) if isinstance(documents['beliefs'], dict) else 0} beliefs, "
        f"{len(documents['architecture'].get('decisions', [])) if isinstance(documents['architecture'], dict) else 0} architecture decisions"
    )
    print(
        f"evidence ledger: {len(ledger.get('experiments', [])) if isinstance(ledger, dict) else 0} rows; "
        f"dependency graph: {len(dependency.get('nodes', [])) if isinstance(dependency, dict) else 0} nodes, "
        f"{len(dependency.get('edges', [])) if isinstance(dependency, dict) else 0} edges; "
        f"manifest: {len(documents['manifest_paths'])} imported paths"
    )
    if problems:
        print(f"DECISION MODEL VALIDATION FAILED ({len(problems)} problems):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("DECISION MODEL VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
