#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = "docs/research/EXPERIMENT_EVIDENCE_LEDGER.json"
MANIFEST_RELATIVE = "docs/research/EVIDENCE_SOURCE_MANIFEST_2026-09-24.json"
EXPECTED_MANIFEST_SHA256 = "c164c735ec4e628a6311fe4c52117c8b58370321df61b00d4976c11769f0215f"
EXPECTED_IMPORT_COUNT = 71
EXPECTED_IMPORTED_JSON_COUNT = 36
EXPECTED_LEDGER_COUNT = 90
EXPECTED_LEDGER_SCHEMA = "anra.evidence-ledger/v1"
EXPECTED_MANIFEST_SCHEMA = "anra.evidence-source-manifest/v1"
EXPECTED_FIELD_ORDER = [
    "id", "program", "branch", "commit", "question", "treatment", "control",
    "model", "data", "runtime", "seeds", "status", "evidence_class",
    "replication", "metrics", "result", "supported_claim",
    "unsupported_claim", "confounds", "artifacts", "supersedes",
    "superseded_by", "relations", "next_discriminating_experiment",
]
EXPECTED_STATUS_ENUM = [
    "DEMONSTRATED", "SUPPORTED", "IMPLEMENTED_NOT_EXECUTED", "IN_PROGRESS",
    "SPECULATIVE", "CONTRADICTED", "INVALIDATED", "SUPERSEDED",
    "INCONCLUSIVE", "NOT_TESTED",
]
EXPECTED_CLASS_ENUM = ["ENGINEERING_EVIDENCE", "SCIENTIFIC_EVIDENCE", "BOTH", "NEITHER"]
EXPECTED_REPLICATION_ENUM = ["R0", "R1", "R2", "R3", "R4"]
EXPECTED_RELATION_ENUM = [
    "EXACT_DUPLICATE", "NEAR_DUPLICATE", "REPLICATION", "EXTENSION",
    "ORTHOGONAL", "CONFLICTING",
]
EXPECTED_BRANCH_HEADS = {
    "Arkenstone", "BRAMASTRA", "arkenstone-ark020-v4", "arkenstone-astra",
    "citadel", "codex/arkenstone-improvements", "core-exp", "core-frozen-v4",
    "cymek-500m-readiness", "esoes", "eval-integrity-001", "iterate500",
    "iterate900", "main", "triquetra",
}
HISTORICAL_COMMIT_FIELDS = {
    "CYR-GPU-006-smoke": "301f5f2 (stash), untracked parent 87ea5d6",
    "SENORA-P35-CMS1-CAD": "30a8fa7 (unreachable; rooted at esoes tip 85f44b7)",
}
CANONICAL_METADATA_FILES = (
    "docs/research/EXPERIMENT_EVIDENCE_LEDGER.json",
    "docs/research/BELIEF_REGISTRY.json",
    "docs/research/ARCHITECTURE_DECISION_LEDGER.json",
    "docs/research/RESEARCH_DECISION_MODEL.json",
    "docs/research/RESEARCH_DEPENDENCY_GRAPH.json",
    "docs/research/NEXT_EXPERIMENT_DECISION_TREE.json",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX40_TOKEN = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{40})(?![0-9a-fA-F])")
SHORT_HEX_TOKEN = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{7,39})(?![0-9a-fA-F])")
PLAIN_PATH = re.compile(r"^[A-Za-z0-9_./-]+$")


class GitRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._commit_cache: dict[str, bool] = {}
        self._tree_cache: dict[str, dict[str, tuple[str, int, str]] | None] = {}
        self._hash_cache: dict[str, str | None] = {}

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args], cwd=self.root, capture_output=True, text=True,
                encoding="utf-8", errors="replace", check=False,
            )
        except OSError as exc:
            return subprocess.CompletedProcess(args, 1, "", str(exc))

    def commit_exists(self, commit: str) -> bool:
        value = commit.lower()
        if value not in self._commit_cache:
            result = self.run("cat-file", "-e", f"{value}^{{commit}}")
            self._commit_cache[value] = result.returncode == 0
        return self._commit_cache[value]

    def tree(self, commit: str) -> dict[str, tuple[str, int, str]] | None:
        value = commit.lower()
        if value in self._tree_cache:
            return self._tree_cache[value]
        if not self.commit_exists(value):
            self._tree_cache[value] = None
            return None
        result = self.run("ls-tree", "-r", "-l", value)
        if result.returncode != 0:
            self._tree_cache[value] = None
            return None
        entries: dict[str, tuple[str, int, str]] = {}
        for line in result.stdout.splitlines():
            try:
                metadata, path = line.split("\t", 1)
                mode, object_type, object_id, size = metadata.split()
                entries[path] = (object_id, -1 if size == "-" else int(size), object_type)
            except (ValueError, TypeError):
                self._tree_cache[value] = None
                return None
        self._tree_cache[value] = entries
        return entries

    def tree_contains(self, commit: str, path: str) -> bool:
        tree = self.tree(commit)
        return tree is not None and path in tree

    def hash_file(self, path: str) -> str | None:
        if path in self._hash_cache:
            return self._hash_cache[path]
        result = self.run("hash-object", "--", path)
        value = result.stdout.strip().lower() if result.returncode == 0 else None
        self._hash_cache[path] = value
        return value

    def ref_commit(self, ref: str) -> str | None:
        result = self.run("rev-parse", "--verify", f"{ref}^{{commit}}")
        value = result.stdout.strip().lower() if result.returncode == 0 else ""
        return value or None


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=True,
    )
    return result.stdout


def commit_tokens(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return list(dict.fromkeys(token.lower() for token in HEX40_TOKEN.findall(value)))


def commit_is_syntactically_valid(commit_value: Any) -> bool:
    return bool(commit_tokens(commit_value))


def historical_commit_reference_valid(
    experiment_id: str, commit_value: Any, repository: GitRepository
) -> bool:
    if HISTORICAL_COMMIT_FIELDS.get(experiment_id) != commit_value:
        return False
    if experiment_id == "SENORA-P35-CMS1-CAD":
        tokens = SHORT_HEX_TOKEN.findall(commit_value)
        return bool(tokens) and all(repository.commit_exists(token) for token in tokens)
    return "stash" in commit_value and "untracked parent" in commit_value


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=unique_json_object,
            parse_constant=reject_json_constant,
        )


def add_problem(problems: list[str], message: str) -> None:
    if message not in problems:
        problems.append(message)


def require_keys(
    value: Any, required: set[str], label: str, problems: list[str], exact: bool = False
) -> bool:
    if not isinstance(value, dict):
        add_problem(problems, f"{label}: expected object")
        return False
    missing = sorted(required - set(value))
    if missing:
        add_problem(problems, f"{label}: missing fields {missing}")
    if exact:
        extra = sorted(set(value) - required)
        if extra:
            add_problem(problems, f"{label}: unexpected fields {extra}")
    return not missing and (not exact or not (set(value) - required))


def require_string(value: Any, label: str, problems: list[str], nonempty: bool = True) -> bool:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        add_problem(problems, f"{label}: expected non-empty string")
        return False
    return True


def require_string_list(value: Any, label: str, problems: list[str], unique: bool = True) -> bool:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        add_problem(problems, f"{label}: expected list of non-empty strings")
        return False
    if unique and len(value) != len(set(value)):
        add_problem(problems, f"{label}: contains duplicates")
        return False
    return True


def safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and value not in {".", ".."} and ".." not in path.parts


def working_tree_file(root: Path, relative: str) -> Path | None:
    if not safe_relative_path(relative):
        return None
    candidate = (root / Path(*PurePosixPath(relative).parts)).resolve()
    root = root.resolve()
    if candidate == root or root not in candidate.parents:
        return None
    return candidate if candidate.is_file() else None


def validate_full_commit(
    value: Any, label: str, repository: GitRepository, problems: list[str]
) -> bool:
    if not isinstance(value, str) or not HEX40.fullmatch(value.lower()):
        add_problem(problems, f"{label}: expected one full 40-hex commit SHA")
        return False
    if not repository.commit_exists(value):
        add_problem(problems, f"{label}: commit object is not available locally: {value}")
        return False
    return True


def validate_sha256(value: Any, label: str, problems: list[str]) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value.lower()):
        add_problem(problems, f"{label}: expected lowercase full SHA-256")
        return False
    return True


def validate_manifest_metadata(root: Path, actual_hash: str, problems: list[str]) -> None:
    seen: set[str] = set()
    research = root / "docs" / "research"
    if not research.is_dir():
        add_problem(problems, "canonical research directory is missing")
        return
    for path in sorted(research.glob("*.json")):
        try:
            data = load_json(path)
        except (OSError, ValueError) as exc:
            add_problem(problems, f"canonical metadata parse failed: {path.relative_to(root)}: {exc}")
            continue
        if isinstance(data, dict) and "consolidation_manifest_sha256" in data:
            relative = path.relative_to(root).as_posix()
            seen.add(relative)
            if data["consolidation_manifest_sha256"] != actual_hash:
                add_problem(problems, f"{relative}: consolidation manifest hash does not match bytes")
    expected = set(CANONICAL_METADATA_FILES)
    missing = sorted(expected - seen)
    if missing:
        add_problem(problems, f"canonical metadata hash declarations missing: {missing}")


def validate_manifest(
    manifest: Any,
    manifest_path: Path,
    root: Path,
    repository: GitRepository,
    problems: list[str] | None = None,
) -> set[str]:
    if problems is None:
        problems = []
    valid_paths: set[str] = set()
    required = {
        "schema", "generated_date", "phase", "purpose", "target",
        "evidence_precedence", "evidence_classes", "status_boundaries",
        "source_refs", "import_groups", "imported_paths",
        "external_only_artifacts", "reference_urls_present_in_imported_evidence",
        "explicit_exclusions", "claim_ceilings", "distinctions", "caveats",
        "verification",
    }
    if not require_keys(manifest, required, "manifest", problems, exact=True):
        return valid_paths
    if manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA:
        add_problem(problems, f"manifest: wrong schema {manifest.get('schema')!r}")
    if manifest.get("phase") != 3:
        add_problem(problems, "manifest: phase must be 3")
    require_string(manifest.get("generated_date"), "manifest.generated_date", problems)
    require_string(manifest.get("purpose"), "manifest.purpose", problems)
    if not isinstance(manifest.get("target"), dict):
        add_problem(problems, "manifest.target: expected object")
    else:
        target = manifest["target"]
        target_required = {
            "branch", "branch_ref", "base_ref", "base_commit", "base_note",
            "pre_consolidation_tip_ref", "pre_consolidation_tip_commit",
            "pre_consolidation_tip_label", "pre_consolidation_origin_tip_ref",
            "pre_consolidation_origin_tip_commit", "pre_consolidation_origin_tip_label",
            "origin_main_tip_at_import", "consolidation_state", "import_operation",
        }
        require_keys(target, target_required, "manifest.target", problems, exact=True)
        commit_keys = (
            "base_commit", "pre_consolidation_tip_commit",
            "pre_consolidation_origin_tip_commit", "origin_main_tip_at_import",
        )
        for key in commit_keys:
            if key in target:
                validate_full_commit(target[key], f"manifest.target.{key}", repository, problems)
    precedence = manifest.get("evidence_precedence")
    if not isinstance(precedence, list) or not precedence:
        add_problem(problems, "manifest.evidence_precedence: expected non-empty list")
    else:
        ranks: list[int] = []
        for index, item in enumerate(precedence):
            if not isinstance(item, dict) or not require_keys(item, {"rank", "class", "rule"}, f"manifest.evidence_precedence[{index}]", problems):
                continue
            if not isinstance(item["rank"], int) or item["rank"] < 1:
                add_problem(problems, f"manifest.evidence_precedence[{index}].rank: invalid rank")
            else:
                ranks.append(item["rank"])
            require_string(item["class"], f"manifest.evidence_precedence[{index}].class", problems)
            require_string(item["rule"], f"manifest.evidence_precedence[{index}].rule", problems)
        if ranks != list(range(1, len(ranks) + 1)):
            add_problem(problems, "manifest.evidence_precedence: ranks are not contiguous")
    for key in ("evidence_classes", "status_boundaries", "claim_ceilings"):
        value = manifest.get(key)
        if not isinstance(value, dict) or not value:
            add_problem(problems, f"manifest.{key}: expected non-empty object")
        elif any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in value.items()):
            add_problem(problems, f"manifest.{key}: values must be non-empty strings")
    source_refs = manifest.get("source_refs")
    source_required = {"review", "origin_branches", "archive_tags", "other_tags_considered", "symbolic_refs"}
    if not require_keys(source_refs, source_required, "manifest.source_refs", problems, exact=True):
        source_refs = {}
    review = source_refs.get("review")
    if not isinstance(review, dict) or not require_keys(
        review,
        {"all_current_origin_branches_reviewed", "relevant_archive_tags_reviewed", "historical_cutoff", "review_through", "rule"},
        "manifest.source_refs.review", problems, exact=True,
    ):
        add_problem(problems, "manifest.source_refs.review: invalid review record")
    else:
        for key in ("all_current_origin_branches_reviewed", "relevant_archive_tags_reviewed"):
            if not isinstance(review[key], bool):
                add_problem(problems, f"manifest.source_refs.review.{key}: expected boolean")
    for collection in ("origin_branches", "archive_tags", "other_tags_considered"):
        values = source_refs.get(collection)
        if not isinstance(values, list) or not values:
            add_problem(problems, f"manifest.source_refs.{collection}: expected non-empty list")
            continue
        for index, item in enumerate(values):
            label = f"manifest.source_refs.{collection}[{index}]"
            if not isinstance(item, dict):
                add_problem(problems, f"{label}: expected object")
                continue
            require_string(item.get("ref"), f"{label}.ref", problems)
            commit_key = "commit" if "commit" in item else "tip_commit"
            if commit_key in item:
                validate_full_commit(item[commit_key], f"{label}.{commit_key}", repository, problems)
            if "reviewed" in item and not isinstance(item["reviewed"], bool):
                add_problem(problems, f"{label}.reviewed: expected boolean")
            require_string(item.get("decision"), f"{label}.decision", problems)
    symbolic_refs = source_refs.get("symbolic_refs")
    if not isinstance(symbolic_refs, list) or not symbolic_refs:
        add_problem(problems, "manifest.source_refs.symbolic_refs: expected non-empty list")
    else:
        for index, item in enumerate(symbolic_refs):
            label = f"manifest.source_refs.symbolic_refs[{index}]"
            if not isinstance(item, dict) or not require_keys(item, {"ref", "resolves_to", "reviewed", "decision", "coverage_class", "reason"}, label, problems, exact=True):
                continue
            for key in ("ref", "resolves_to", "decision", "coverage_class", "reason"):
                require_string(item[key], f"{label}.{key}", problems)
            if not isinstance(item["reviewed"], bool):
                add_problem(problems, f"{label}.reviewed: expected boolean")
    groups = manifest.get("import_groups")
    group_by_id: dict[str, dict[str, Any]] = {}
    if not isinstance(groups, list) or not groups:
        add_problem(problems, "manifest.import_groups: expected non-empty list")
        groups = []
    for index, group in enumerate(groups):
        label = f"manifest.import_groups[{index}]"
        fields = {"id", "source_ref", "source_commit", "decision", "path_count", "reason"}
        if not require_keys(group, fields, label, problems, exact=True):
            continue
        group_id = group["id"]
        if not isinstance(group_id, str) or not group_id.strip():
            add_problem(problems, f"{label}.id: expected non-empty string")
            continue
        if group_id in group_by_id:
            add_problem(problems, f"manifest.import_groups: duplicate id {group_id}")
        group_by_id[group_id] = group
        require_string(group["source_ref"], f"{label}.source_ref", problems)
        validate_full_commit(group["source_commit"], f"{label}.source_commit", repository, problems)
        require_string(group["decision"], f"{label}.decision", problems)
        if not isinstance(group["path_count"], int) or isinstance(group["path_count"], bool) or group["path_count"] < 1:
            add_problem(problems, f"{label}.path_count: expected positive integer")
        require_string(group["reason"], f"{label}.reason", problems)
    imported = manifest.get("imported_paths")
    if not isinstance(imported, list):
        add_problem(problems, "manifest.imported_paths: expected list")
        imported = []
    if len(imported) != EXPECTED_IMPORT_COUNT:
        add_problem(problems, f"manifest.imported_paths: expected exactly {EXPECTED_IMPORT_COUNT}, found {len(imported)}")
    seen_paths: set[str] = set()
    seen_folded_paths: set[str] = set()
    group_counts: dict[str, int] = {}
    record_fields = {
        "path", "source_ref", "source_commit", "blob", "size_bytes",
        "evidence_class", "status_boundary", "import_group",
    }
    for index, record in enumerate(imported):
        label = f"manifest.imported_paths[{index}]"
        if not require_keys(record, record_fields, label, problems, exact=True):
            continue
        path = record["path"]
        if not safe_relative_path(path):
            add_problem(problems, f"{label}.path: unsafe or non-relative path")
            continue
        if path in seen_paths or path.casefold() in seen_folded_paths:
            add_problem(problems, f"manifest.imported_paths: duplicate path {path}")
        seen_paths.add(path)
        seen_folded_paths.add(path.casefold())
        require_string(record["source_ref"], f"{label}.source_ref", problems)
        if not validate_full_commit(record["source_commit"], f"{label}.source_commit", repository, problems):
            continue
        if not HEX40.fullmatch(str(record["blob"]).lower()):
            add_problem(problems, f"{label}.blob: expected full 40-hex blob ID")
        if not isinstance(record["size_bytes"], int) or isinstance(record["size_bytes"], bool) or record["size_bytes"] < 0:
            add_problem(problems, f"{label}.size_bytes: expected non-negative integer")
        if not isinstance(record["evidence_class"], str) or not isinstance(manifest.get("evidence_classes"), dict) or record["evidence_class"] not in manifest["evidence_classes"]:
            add_problem(problems, f"{label}.evidence_class: unknown value {record['evidence_class']!r}")
        if not isinstance(record["status_boundary"], str) or not isinstance(manifest.get("status_boundaries"), dict) or record["status_boundary"] not in manifest["status_boundaries"]:
            add_problem(problems, f"{label}.status_boundary: unknown value {record['status_boundary']!r}")
        group = group_by_id.get(record["import_group"]) if isinstance(record["import_group"], str) else None
        if group is None:
            add_problem(problems, f"{label}.import_group: unknown group {record['import_group']!r}")
        else:
            if record["source_ref"] != group["source_ref"] or record["source_commit"] != group["source_commit"]:
                add_problem(problems, f"{label}: source ref or commit disagrees with import group")
            group_counts[record["import_group"]] = group_counts.get(record["import_group"], 0) + 1
        current = working_tree_file(root, path)
        if current is None:
            add_problem(problems, f"{label}: imported path is absent from the working tree: {path}")
        else:
            actual_size = current.stat().st_size
            if actual_size != record["size_bytes"]:
                add_problem(problems, f"{label}: working-tree byte size mismatch")
            actual_blob = repository.hash_file(path)
            if actual_blob != str(record["blob"]).lower():
                add_problem(problems, f"{label}: working-tree Git blob mismatch")
        tree = repository.tree(record["source_commit"])
        if tree is None:
            add_problem(problems, f"{label}: source commit tree cannot be inspected")
        else:
            source_entry = tree.get(path)
            if source_entry is None:
                add_problem(problems, f"{label}: path is absent from source commit")
            else:
                source_blob, source_size, object_type = source_entry
                if object_type != "blob" or source_blob != str(record["blob"]).lower() or source_size != record["size_bytes"]:
                    add_problem(problems, f"{label}: source commit blob or size mismatch")
        valid_paths.add(path)
    for group_id, group in group_by_id.items():
        if group_counts.get(group_id, 0) != group["path_count"]:
            add_problem(problems, f"manifest.import_groups[{group_id}]: path_count does not match imported paths")
    if set(group_counts) != set(group_by_id):
        add_problem(problems, "manifest.imported_paths: every import group must have paths")
    external = manifest.get("external_only_artifacts")
    if not isinstance(external, list):
        add_problem(problems, "manifest.external_only_artifacts: expected list")
        external = []
    for index, record in enumerate(external):
        label = f"manifest.external_only_artifacts[{index}]"
        if not isinstance(record, dict):
            add_problem(problems, f"{label}: expected object")
            continue
        for key in ("artifact", "kind", "location_or_url", "hash_status", "imported", "reason"):
            if key not in record:
                add_problem(problems, f"{label}: missing {key}")
        if record.get("imported") is not False:
            add_problem(problems, f"{label}.imported: must be false")
        if "referenced_by" in record:
            require_string_list(record["referenced_by"], f"{label}.referenced_by", problems)
        sha = record.get("sha256")
        if isinstance(sha, str):
            if not re.fullmatch(r"[0-9a-f]{7,64}", sha.lower()):
                add_problem(problems, f"{label}.sha256: invalid hexadecimal hash")
        elif isinstance(sha, dict):
            for name, value in sha.items():
                if not isinstance(name, str) or not name or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{7,64}", value):
                    add_problem(problems, f"{label}.sha256.{name}: invalid hexadecimal hash")
        elif sha is not None:
            add_problem(problems, f"{label}.sha256: expected string, object, or null")
    urls = manifest.get("reference_urls_present_in_imported_evidence")
    if not isinstance(urls, list) or any(not isinstance(item, dict) for item in urls):
        add_problem(problems, "manifest.reference_urls_present_in_imported_evidence: invalid list")
    else:
        for index, item in enumerate(urls):
            label = f"manifest.reference_urls_present_in_imported_evidence[{index}]"
            if not require_keys(item, {"url", "purpose", "imported_artifact"}, label, problems, exact=True):
                continue
            if not isinstance(item["url"], str) or not item["url"].startswith(("http://", "https://")):
                add_problem(problems, f"{label}.url: expected URL")
            if item["imported_artifact"] is not False:
                add_problem(problems, f"{label}.imported_artifact: must be false")
    verification = manifest.get("verification")
    if not isinstance(verification, dict):
        add_problem(problems, "manifest.verification: expected object")
    else:
        source = verification.get("source_blob_comparison")
        if not isinstance(source, dict) or source.get("expected_paths") != EXPECTED_IMPORT_COUNT or source.get("passed") != EXPECTED_IMPORT_COUNT or source.get("failed") != 0:
            add_problem(problems, f"manifest.verification.source_blob_comparison: not finalized at {EXPECTED_IMPORT_COUNT}/{EXPECTED_IMPORT_COUNT}")
        parsed = verification.get("json_parse")
        if not isinstance(parsed, dict) or parsed.get("expected_files") != EXPECTED_IMPORTED_JSON_COUNT or parsed.get("parsed") != EXPECTED_IMPORTED_JSON_COUNT or parsed.get("errors") != 0:
            add_problem(problems, f"manifest.verification.json_parse: not finalized at {EXPECTED_IMPORTED_JSON_COUNT}/{EXPECTED_IMPORTED_JSON_COUNT}")
        manifest_parse = verification.get("manifest_json_parse")
        if not isinstance(manifest_parse, dict) or manifest_parse.get("status") != "passed":
            add_problem(problems, "manifest.verification.manifest_json_parse: not passed")
        diff_check = verification.get("git_diff_check")
        if not isinstance(diff_check, dict) or diff_check.get("status") != "passed":
            add_problem(problems, "manifest.verification.git_diff_check: not passed")
    actual_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else ""
    if actual_hash != EXPECTED_MANIFEST_SHA256:
        add_problem(problems, f"manifest: SHA-256 is {actual_hash or '<missing>'}, expected {EXPECTED_MANIFEST_SHA256}")
    validate_manifest_metadata(root, actual_hash, problems)
    return valid_paths


def validate_ledger(
    ledger: Any,
    root: Path,
    repository: GitRepository,
    manifest_paths: set[str] | None = None,
    problems: list[str] | None = None,
) -> None:
    if problems is None:
        problems = []
    manifest_paths = manifest_paths or set()
    required = {
        "schema", "generated", "phase", "generated_by", "synthesis_branch",
        "basis_head", "historical_snapshot_date", "consolidation_source_manifest",
        "consolidation_manifest_sha256", "pre_consolidation_head",
        "evidence_authority", "field_order", "evidence_status_enum",
        "evidence_class_enum", "replication_enum", "relation_enum",
        "branch_heads_at_audit", "experiments",
    }
    if not require_keys(ledger, required, "ledger", problems, exact=True):
        return
    if ledger.get("schema") != EXPECTED_LEDGER_SCHEMA:
        add_problem(problems, f"ledger: wrong schema {ledger.get('schema')!r}")
    if ledger.get("phase") != 3:
        add_problem(problems, "ledger: phase must be 3")
    for key in ("generated", "historical_snapshot_date"):
        require_string(ledger.get(key), f"ledger.{key}", problems)
    for key in ("generated_by", "synthesis_branch", "evidence_authority"):
        require_string(ledger.get(key), f"ledger.{key}", problems)
    if ledger.get("consolidation_source_manifest") != MANIFEST_RELATIVE:
        add_problem(problems, f"ledger: consolidation source manifest must be {MANIFEST_RELATIVE}")
    validate_full_commit(ledger.get("basis_head"), "ledger.basis_head", repository, problems)
    validate_sha256(ledger.get("consolidation_manifest_sha256"), "ledger.consolidation_manifest_sha256", problems)
    pre = ledger.get("pre_consolidation_head")
    if not require_keys(pre, {"branch", "ref", "commit", "label"}, "ledger.pre_consolidation_head", problems, exact=True):
        pass
    else:
        require_string(pre["branch"], "ledger.pre_consolidation_head.branch", problems)
        require_string(pre["ref"], "ledger.pre_consolidation_head.ref", problems)
        validate_full_commit(pre["commit"], "ledger.pre_consolidation_head.commit", repository, problems)
        require_string(pre["label"], "ledger.pre_consolidation_head.label", problems)
    if ledger.get("field_order") != EXPECTED_FIELD_ORDER:
        add_problem(problems, "ledger.field_order: expected the exact 24-field order")
    enum_values = {
        "evidence_status_enum": EXPECTED_STATUS_ENUM,
        "evidence_class_enum": EXPECTED_CLASS_ENUM,
        "replication_enum": EXPECTED_REPLICATION_ENUM,
        "relation_enum": EXPECTED_RELATION_ENUM,
    }
    for key, expected in enum_values.items():
        if ledger.get(key) != expected:
            add_problem(problems, f"ledger.{key}: enum values or order do not match the schema")
    branches = ledger.get("branch_heads_at_audit")
    frozen_commits: list[str] = []
    if not isinstance(branches, dict):
        add_problem(problems, "ledger.branch_heads_at_audit: expected object")
    else:
        actual_names = {key for key in branches if not key.startswith("_")}
        if actual_names != EXPECTED_BRANCH_HEADS:
            add_problem(problems, "ledger.branch_heads_at_audit: branch set does not match the frozen audit set")
        for name, commit in branches.items():
            if name.startswith("_"):
                require_string(commit, f"ledger.branch_heads_at_audit.{name}", problems)
                continue
            label = f"ledger.branch_heads_at_audit.{name}"
            if validate_full_commit(commit, label, repository, problems):
                frozen_commits.append(commit)
                if repository.tree(commit) is None:
                    add_problem(problems, f"{label}: commit tree cannot be inspected")
    experiments = ledger.get("experiments")
    if not isinstance(experiments, list):
        add_problem(problems, "ledger.experiments: expected list")
        return
    if len(experiments) != EXPECTED_LEDGER_COUNT:
        add_problem(problems, f"ledger.experiments: expected exactly {EXPECTED_LEDGER_COUNT}, found {len(experiments)}")
    ids: list[str] = []
    valid_ids: set[str] = set()
    for index, experiment in enumerate(experiments):
        label = f"ledger.experiments[{index}]"
        if not require_keys(experiment, set(EXPECTED_FIELD_ORDER), label, problems, exact=True):
            continue
        if list(experiment) != EXPECTED_FIELD_ORDER:
            add_problem(problems, f"{label}: fields are not in the declared order")
        experiment_id = experiment.get("id")
        if not isinstance(experiment_id, str) or not experiment_id.strip() or any(ch.isspace() for ch in experiment_id):
            add_problem(problems, f"{label}.id: invalid experiment ID")
            experiment_id = f"<invalid-{index}>"
        ids.append(experiment_id)
        valid_ids.add(experiment_id)
        for key in ("program", "branch", "commit", "question", "treatment", "control", "model", "data", "runtime", "result", "supported_claim", "unsupported_claim", "next_discriminating_experiment"):
            require_string(experiment.get(key), f"{label}.{key}", problems)
        if experiment.get("status") not in EXPECTED_STATUS_ENUM:
            add_problem(problems, f"{experiment_id}: invalid status {experiment.get('status')!r}")
        if experiment.get("evidence_class") not in EXPECTED_CLASS_ENUM:
            add_problem(problems, f"{experiment_id}: invalid evidence class {experiment.get('evidence_class')!r}")
        if experiment.get("replication") not in EXPECTED_REPLICATION_ENUM:
            add_problem(problems, f"{experiment_id}: invalid replication {experiment.get('replication')!r}")
        if not isinstance(experiment.get("seeds"), (list, dict)):
            add_problem(problems, f"{experiment_id}: seeds must be a list or object")
        if not isinstance(experiment.get("metrics"), dict):
            add_problem(problems, f"{experiment_id}: metrics must be an object")
        for key in ("confounds", "artifacts", "supersedes", "superseded_by", "relations"):
            if key == "relations":
                if not isinstance(experiment.get(key), list):
                    add_problem(problems, f"{experiment_id}: relations must be a list")
            elif not require_string_list(experiment.get(key), f"{experiment_id}.{key}", problems, unique=True):
                pass
        if experiment.get("status") == "IMPLEMENTED_NOT_EXECUTED" and experiment.get("metrics") != {}:
            add_problem(problems, f"{experiment_id}: IMPLEMENTED_NOT_EXECUTED must have empty metrics")
        if experiment.get("status") in {"DEMONSTRATED", "SUPPORTED"} and not experiment.get("artifacts"):
            add_problem(problems, f"{experiment_id}: demonstrated or supported claim has no artifacts")
        tokens = commit_tokens(experiment.get("commit"))
        if not tokens:
            if not historical_commit_reference_valid(experiment_id, experiment.get("commit"), repository):
                add_problem(problems, f"{experiment_id}: commit field has no full 40-hex SHA token")
        for token in tokens:
            if not repository.commit_exists(token):
                add_problem(problems, f"{experiment_id}: commit object is not available locally: {token}")
        for field in ("supersedes", "superseded_by"):
            for reference in experiment.get(field, []) if isinstance(experiment.get(field), list) else []:
                if reference == experiment_id:
                    add_problem(problems, f"{experiment_id}: {field} cannot reference itself")
                if reference not in valid_ids and reference not in {e.get("id") for e in experiments if isinstance(e, dict)}:
                    add_problem(problems, f"{experiment_id}: {field} reference does not resolve: {reference}")
        relations = experiment.get("relations")
        if isinstance(relations, list):
            for relation_index, relation in enumerate(relations):
                relation_label = f"{experiment_id}.relations[{relation_index}]"
                if not isinstance(relation, dict) or not require_keys(relation, {"id", "relation"}, relation_label, problems):
                    continue
                if set(relation) - {"id", "relation", "note"}:
                    add_problem(problems, f"{relation_label}: unexpected fields")
                if relation.get("id") == experiment_id:
                    add_problem(problems, f"{relation_label}: relation cannot reference itself")
                if relation.get("id") not in {e.get("id") for e in experiments if isinstance(e, dict)}:
                    add_problem(problems, f"{relation_label}: relation reference does not resolve: {relation.get('id')!r}")
                if relation.get("relation") not in EXPECTED_RELATION_ENUM:
                    add_problem(problems, f"{relation_label}: invalid relation kind {relation.get('relation')!r}")
                if "note" in relation:
                    require_string(relation["note"], f"{relation_label}.note", problems)
    duplicate_ids = sorted({value for value in ids if ids.count(value) > 1})
    if duplicate_ids:
        add_problem(problems, f"ledger: duplicate experiment IDs: {duplicate_ids}")
    id_set = set(ids)
    if len(id_set) != len(ids):
        add_problem(problems, "ledger: experiment IDs are not unique")
    for experiment in experiments:
        if not isinstance(experiment, dict):
            continue
        experiment_id = experiment.get("id", "<invalid>")
        for artifact in experiment.get("artifacts", []) if isinstance(experiment.get("artifacts"), list) else []:
            if not isinstance(artifact, str) or not artifact.strip():
                continue
            if not PLAIN_PATH.fullmatch(artifact):
                continue
            if not safe_relative_path(artifact):
                add_problem(problems, f"{experiment_id}: unsafe artifact path {artifact}")
                continue
            if working_tree_file(root, artifact) is not None or artifact in manifest_paths:
                continue
            if not any(repository.tree_contains(commit, artifact) for commit in frozen_commits):
                add_problem(problems, f"{experiment_id}: artifact path does not resolve in worktree, manifest, or frozen source trees: {artifact}")


def status_counts(ledger: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for experiment in ledger.get("experiments", []):
        if isinstance(experiment, dict):
            status = experiment.get("status", "?")
            counts[status] = counts.get(status, 0) + 1
    return counts


def print_ledger_result(path: str, ledger: dict[str, Any], problems: list[str]) -> None:
    print(f"ledger: {path}")
    print(f"experiments: {len(ledger.get('experiments', [])) if isinstance(ledger.get('experiments', []), list) else 0}")
    counts = status_counts(ledger)
    for status in EXPECTED_STATUS_ENUM:
        if counts.get(status):
            print(f"  {status}: {counts[status]}")
    unknown = sorted(set(counts) - set(EXPECTED_STATUS_ENUM))
    if unknown:
        print(f"  UNKNOWN-STATUS: {unknown}")
    if problems:
        print(f"\nVALIDATION FAILED ({len(problems)} problems):")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nVALIDATION PASSED")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    ledger_name = arguments[0] if arguments else DEFAULT_LEDGER
    root = ROOT
    ledger_path = Path(ledger_name)
    if not ledger_path.is_absolute():
        ledger_path = root / ledger_path
    problems: list[str] = []
    ledger: Any = {}
    try:
        ledger = load_json(ledger_path)
    except (OSError, ValueError) as exc:
        add_problem(problems, f"ledger: cannot parse {ledger_path}: {exc}")
    repository = GitRepository(root)
    manifest_paths: set[str] = set()
    if isinstance(ledger, dict) and ledger.get("consolidation_source_manifest") == MANIFEST_RELATIVE:
        manifest_path = root / MANIFEST_RELATIVE
        try:
            manifest = load_json(manifest_path)
            manifest_paths = validate_manifest(manifest, manifest_path, root, repository, problems)
        except (OSError, ValueError) as exc:
            add_problem(problems, f"manifest: cannot parse {manifest_path}: {exc}")
    else:
        add_problem(problems, f"ledger: cannot resolve finalized manifest {MANIFEST_RELATIVE}")
    if isinstance(ledger, dict):
        validate_ledger(ledger, root, repository, manifest_paths, problems)
    else:
        add_problem(problems, "ledger: top-level JSON must be an object")
    print_ledger_result(str(ledger_path), ledger if isinstance(ledger, dict) else {}, problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
