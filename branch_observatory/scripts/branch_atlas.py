from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from atlas_lib import (
    STATUS_NAMES,
    AtlasError,
    BlobReader,
    GitRepository,
    SCHEMA_VERSION,
    TOOL_VERSION,
    activity_for_branch,
    atomic_write,
    build_branch,
    build_documents,
    citation_for,
    default_config,
    is_document_candidate,
    load_config,
    parse_datetime,
    project_activity,
    safe_json,
    scrub_url,
    slugify_ref,
    stock_metrics,
    utc_now,
    validate_snapshot,
    write_json,
    write_text,
)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AtlasError(f"Could not read {path}: {error}") from error
    return value if isinstance(value, dict) else None


def output_paths(output: Path) -> dict[str, Path]:
    return {
        "latest": output / "data" / "latest_snapshot.json",
        "history_dir": output / "data" / "history" / "snapshots",
        "daily": output / "data" / "history" / "daily_metrics.csv",
        "latest_report": output / "reports" / "latest.md",
        "branch_index": output / "reports" / "branch_index.md",
        "branches_dir": output / "reports" / "branches",
        "specs": output / "reports" / "specs_index.md",
        "project_map": output / "reports" / "project_map.md",
        "comparisons": output / "reports" / "comparisons.md",
        "quality": output / "reports" / "data_quality.md",
        "verification": output / "reports" / "verification.json",
    }


def ensure_output(output: Path) -> dict[str, Path]:
    paths = output_paths(output)
    for path in [output, paths["history_dir"], paths["branches_dir"]]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def ref_groups(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for ref in refs:
        if ref.get("commit_sha"):
            groups[ref["commit_sha"]].append(ref["name"])
    return [{"commit_sha": sha, "refs": sorted(names), "ref_count": len(names)} for sha, names in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))]


def remote_without_local(refs: list[dict[str, Any]]) -> list[str]:
    local = {item["short_name"] for item in refs if item["type"] == "local_head"}
    result = []
    for item in refs:
        if item["type"] != "remote_tracking" or item.get("symbolic_target"):
            continue
        short = item["short_name"]
        branch = short[len("origin/"):] if short.startswith("origin/") else short.split("/", 1)[-1]
        if branch not in local:
            result.append(item["name"])
    return sorted(result)


def local_without_upstream(refs: list[dict[str, Any]]) -> list[str]:
    return sorted(item["name"] for item in refs if item["type"] == "local_head" and not item.get("upstream"))


def upstream_gone(repository: GitRepository, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in refs:
        if item["type"] != "local_head" or not item.get("upstream"):
            continue
        found = False
        for candidate in (f"refs/remotes/{item['upstream']}", f"refs/heads/{item['upstream']}", item["upstream"]):
            if repository.resolve_commit(candidate):
                found = True
                break
        if not found:
            result.append({"ref": item["name"], "upstream": item["upstream"], "status": "gone_or_unavailable"})
    return result


def diff_file_totals(repository: GitRepository, base: str, tip: str, config: dict[str, Any]) -> dict[str, Any]:
    exclusions = [":(exclude).git.deleted-remnants-*/**", ":(exclude)branch_observatory/**", ":(exclude)**/*.pt", ":(exclude)**/*.pth", ":(exclude)**/*.safetensors", ":(exclude)**/*.bin", ":(exclude)**/*.zip"]
    output = repository.run(["diff", "--numstat", "--find-renames", base, tip, "--", *exclusions], False, 300).stdout
    totals = {"files_changed": 0, "source_lines_added": 0, "source_lines_removed": 0, "docs_lines_added": 0, "docs_lines_removed": 0, "test_lines_added": 0, "test_lines_removed": 0, "paths": []}
    for raw in output.splitlines():
        fields = raw.decode("utf-8", errors="replace").split("\t", 2)
        if len(fields) != 3 or fields[0] == "-" or fields[1] == "-":
            continue
        path = fields[2].split("=>", 1)[-1].strip().strip("{}")
        try:
            added, removed = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        totals["files_changed"] += 1
        if len(totals["paths"]) < 100:
            totals["paths"].append(path)
        lower = path.lower()
        if Path(lower).suffix in {".md", ".markdown", ".rst", ".adoc", ".txt"} or "docs/" in lower or "blueprint/" in lower:
            totals["docs_lines_added"] += added
            totals["docs_lines_removed"] += removed
        elif "test" in Path(lower).name or "/tests/" in lower:
            totals["test_lines_added"] += added
            totals["test_lines_removed"] += removed
        elif Path(lower).suffix in {".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".c", ".h", ".cc", ".cpp", ".hpp", ".rs", ".go", ".java", ".kt", ".swift", ".sh", ".ps1", ".sql", ".html", ".css"}:
            totals["source_lines_added"] += added
            totals["source_lines_removed"] += removed
    return totals


def build_comparisons(repository: GitRepository, branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for branch in branches:
        if branch.get("commit_sha"):
            unique.setdefault(branch["commit_sha"], branch)
    representatives = [unique[sha] for sha in sorted(unique)]
    delta_cache: dict[tuple[str, str], dict[str, Any]] = {}
    result = []
    for index, left in enumerate(representatives):
        for right in representatives[index + 1:]:
            merge = repository.merge_base(left["commit_sha"], right["commit_sha"])
            record = {"left_ref": left["ref"], "right_ref": right["ref"], "left_sha": left["commit_sha"], "right_sha": right["commit_sha"], "merge_base_sha": merge, "comparable": merge is not None, "comparability_reason": "shared local merge-base and common classification policy" if merge else "no local merge-base"}
            if merge:
                left_key = (merge, left["commit_sha"])
                right_key = (merge, right["commit_sha"])
                if left_key not in delta_cache:
                    delta_cache[left_key] = diff_file_totals(repository, merge, left["commit_sha"], {})
                if right_key not in delta_cache:
                    delta_cache[right_key] = diff_file_totals(repository, merge, right["commit_sha"], {})
                left_delta = delta_cache[left_key]
                right_delta = delta_cache[right_key]
                record.update({"left_commits_from_merge_base": repository.count_range(merge, left["commit_sha"]), "right_commits_from_merge_base": repository.count_range(merge, right["commit_sha"]), "left_delta": left_delta, "right_delta": right_delta, "specs_changed_left": [path for path in left_delta.get("paths", []) if is_document_candidate(path)][:50], "specs_changed_right": [path for path in right_delta.get("paths", []) if is_document_candidate(path)][:50]})
            result.append(record)
    return result


def previous_delta(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if not previous:
        return {"previous_snapshot_id": None, "refs_added": [], "refs_removed": [], "refs_moved": [], "worktree_status_changes": [], "documents_added": [], "documents_removed": [], "documents_changed": [], "report_stale_reasons": ["first snapshot; no prior comparison exists"]}
    old_refs = {item["name"]: item.get("commit_sha") for item in previous.get("refs", [])}
    new_refs = {item["name"]: item.get("commit_sha") for item in current.get("refs", [])}
    moved = [{"ref": name, "from_sha": old_refs[name], "to_sha": new_refs[name]} for name in sorted(set(old_refs) & set(new_refs)) if old_refs[name] != new_refs[name]]
    old_worktrees = {item["path"]: item for item in previous.get("worktrees", [])}
    new_worktrees = {item["path"]: item for item in current.get("worktrees", [])}
    worktree_changes = []
    for path in sorted(set(old_worktrees) | set(new_worktrees)):
        if path not in old_worktrees:
            worktree_changes.append({"path": path, "change": "added"})
        elif path not in new_worktrees:
            worktree_changes.append({"path": path, "change": "removed"})
        else:
            old_status = old_worktrees[path].get("status", {})
            new_status = new_worktrees[path].get("status", {})
            old_sig = (old_worktrees[path].get("head"), old_worktrees[path].get("branch"), old_status.get("user_dirty"), old_status.get("status_digest"), old_status.get("generated_overlay_count"))
            new_sig = (new_worktrees[path].get("head"), new_worktrees[path].get("branch"), new_status.get("user_dirty"), new_status.get("status_digest"), new_status.get("generated_overlay_count"))
            if old_sig != new_sig:
                worktree_changes.append({"path": path, "change": "status_or_head_changed"})
    old_docs = {(item.get("ref"), item.get("path")): item.get("blob_sha") for item in previous.get("documents", [])}
    new_docs = {(item.get("ref"), item.get("path")): item.get("blob_sha") for item in current.get("documents", [])}
    added = [{"ref": key[0], "path": key[1]} for key in sorted(set(new_docs) - set(old_docs))]
    removed = [{"ref": key[0], "path": key[1]} for key in sorted(set(old_docs) - set(new_docs))]
    changed = [{"ref": key[0], "path": key[1], "from_blob": old_docs[key], "to_blob": new_docs[key]} for key in sorted(set(old_docs) & set(new_docs)) if old_docs[key] != new_docs[key]]
    stale = []
    if previous.get("schema_version") != current.get("schema_version"):
        stale.append("schema version changed")
    if previous.get("tool", {}).get("source_sha256") != current.get("tool", {}).get("source_sha256"):
        stale.append("tool source hash changed")
    return {"previous_snapshot_id": previous.get("snapshot_id"), "refs_added": sorted(set(new_refs) - set(old_refs)), "refs_removed": sorted(set(old_refs) - set(new_refs)), "refs_moved": moved, "worktree_status_changes": worktree_changes, "documents_added": added, "documents_removed": removed, "documents_changed": changed, "report_stale_reasons": stale}


def build_snapshot(repository: GitRepository, config: dict[str, Any], capture: dt.datetime, previous: dict[str, Any] | None) -> dict[str, Any]:
    refs = repository.refs()
    worktrees = parse_worktrees(repository, repository.worktrees())
    base_ref = config.get("base_ref", "refs/remotes/origin/main")
    base_sha = repository.resolve_commit(base_ref)
    stock_cache: dict[str, dict[str, Any]] = {}
    diff_cache: dict[str, dict[str, Any]] = {}
    metadata_cache: dict[str, dict[str, Any]] = {}
    reader = BlobReader(repository)
    branches = []
    documents = []
    warnings = []
    try:
        for ref in refs:
            if not ref.get("is_branch_ref"):
                continue
            if not ref.get("commit_sha"):
                warnings.append(f"Branch ref could not be resolved to a commit: {ref.get('name')}")
                continue
            branch, branch_documents = build_branch(repository, ref, base_sha, worktrees, stock_cache, diff_cache, metadata_cache, config, capture, reader)
            branches.append(branch)
            documents.extend(branch_documents)
    finally:
        reader.close()
    submodules = repository.submodules(base_sha)
    if submodules:
        warnings.append(f"Submodule entries detected in the base tree: {len(submodules)}; they are reported separately and not combined with parent source counts.")
    else:
        warnings.append("No submodule entries were found in the selected base tree; untracked nested repositories were not recursively crawled.")
    if any(not item.get("status", {}).get("readable") for item in worktrees):
        warnings.append("At least one linked worktree status could not be read; its path is retained as inaccessible.")
    if any(not item.get("commit_resolves", True) for item in refs):
        warnings.append("One or more refs could not be resolved to a local commit object; see ref inventory.")
    warnings.append("Remote refs were not fetched or refreshed; this snapshot uses only locally present Git objects and refs.")
    local_timestamp = capture.astimezone()
    main_source = Path(__file__).read_bytes()
    library_path = Path(__file__).with_name("atlas_lib.py")
    library_source = library_path.read_bytes()
    combined_source = hashlib.sha256(main_source + library_source).hexdigest()
    snapshot: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": capture.strftime("%Y%m%dT%H%M%S%fZ"),
        "capture": {"timestamp_utc": capture.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"), "local_timestamp": local_timestamp.isoformat(timespec="seconds"), "local_timezone": local_timestamp.tzname() or "unknown", "local_utc_offset": local_timestamp.strftime("%z"), "remote_refreshed": False, "remote_operations": []},
        "tool": {"name": "branch_atlas", "version": TOOL_VERSION, "source_sha256": combined_source, "source_files": {"branch_atlas.py": hashlib.sha256(main_source).hexdigest(), "atlas_lib.py": hashlib.sha256(library_source).hexdigest()}, "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"},
        "repository": {**repository.identity(), "remotes": repository.remotes()},
        "base_selection": {"ref": base_ref, "sha": base_sha, "rationale": config.get("base_selection", {}).get("rationale"), "ambiguity": config.get("base_selection", {}).get("ambiguity"), "selection_method": "configured ref; branch-specific upstream merge-base is recorded per dossier"},
        "scope": {"ref_count": len(refs), "branch_ref_count": sum(1 for item in refs if item.get("is_branch_ref")), "local_head_count": sum(1 for item in refs if item.get("type") == "local_head"), "remote_tracking_count": sum(1 for item in refs if item.get("type") == "remote_tracking"), "tag_count": sum(1 for item in refs if item.get("type") == "tag"), "worktree_count": len(worktrees), "inaccessible_worktree_count": sum(1 for item in worktrees if not item.get("status", {}).get("readable")), "repository_count": 1 + len(submodules), "submodules": submodules, "remote_without_local": remote_without_local(refs), "local_without_upstream": local_without_upstream(refs), "upstreams_gone": upstream_gone(repository, refs), "uninspectable_refs": [item["name"] for item in refs if not item.get("commit_resolves")], "excluded_data": ["ignored and untracked contents not in committed trees", "dependency/cache/vendor paths in config excluded_paths", "large blobs above max_text_blob_bytes: metadata counted, content not loaded", "external artifacts, weights, datasets, and deleted history not present in local Git"]},
        "refs": refs,
        "ref_groups": ref_groups(refs),
        "worktrees": worktrees,
        "branches": branches,
        "documents": documents,
        "comparisons": build_comparisons(repository, branches),
        "warnings": warnings,
    }
    snapshot["project_activity"] = project_activity(repository, branches, capture, config, diff_cache, metadata_cache)
    snapshot["previous_snapshot_delta"] = previous_delta(previous, snapshot)
    snapshot["goal_recommendations"] = config.get("goals", [])
    snapshot["validation"] = validate_snapshot(repository, snapshot)
    return snapshot


def parse_worktrees(repository: GitRepository, worktrees: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**item, "status": repository.status_for_path(item["path"])} for item in worktrees]


def md(value: Any) -> str:
    return str(value if value is not None else "N/A").replace("|", "\\|").replace("\n", " ")


def citation(evidence: dict[str, Any]) -> str:
    location = f"{evidence.get('ref')}:{evidence.get('path')}"
    if evidence.get("section"):
        location += f" § {evidence['section']}"
    if evidence.get("blob_sha"):
        location += f" @ {evidence['blob_sha'][:12]}"
    if evidence.get("citation_status") != "resolved":
        location += " [unresolved]"
    return location


def render_latest(snapshot: dict[str, Any]) -> str:
    capture = snapshot["capture"]
    scope = snapshot["scope"]
    branches = sorted(snapshot.get("branches", []), key=lambda item: item.get("commit", {}).get("committer_date") or "", reverse=True)
    delta = snapshot.get("previous_snapshot_delta", {})
    lines = [
        "# An-Ra Branch Observatory", "",
        f"Generated from local Git objects and repository documents at `{capture['timestamp_utc']}` ({capture.get('local_timezone', 'timezone unavailable')}).", "",
        "## Executive summary", "",
        f"- **What exists:** {scope['ref_count']} refs, {scope['branch_ref_count']} branch refs, {scope['worktree_count']} linked worktrees, and {scope['repository_count']} repository scope. No fetch, pull, or remote refresh was performed.",
        "- **Strongest completed evidence:** Arkenstone has the densest executed, matched, audited bounded evidence; Citadel is the strongest independent audit authority; neither establishes AGI.",
        "- **Implemented but unexecuted:** several branches explicitly contain implemented or locally verified paths whose own records say the decisive run is pending, blocked, incomplete, or externally held.",
        f"- **Most recent activity:** `{branches[0]['ref'] if branches else 'UNKNOWN'}` is the latest visible branch tip by committer date; this is not a quality ranking.",
        "- **Reliable metrics:** committed-tree stock, blob/document identities, ref/worktree inventory, ancestry, commit metadata, and first-parent diffs are mechanically measured. Daily stock replay is labeled as an estimate when merge and side-branch flow overlap.",
        "- **LOC limitation:** branches with large uninspected source blobs are explicitly marked loc-limited in the branch index; their counts are not treated as directly comparable.",
        "- **Three caveats:** no single branch is authoritative for every program; code/tests/receipts/prose have different ceilings; deleted history, prior dirty states, and external artifacts cannot be reconstructed.", "",
        "## Goal-specific reading", "", "| Goal | Recommended ref | Reason | Confidence |", "|---|---|---|---|",
    ]
    for goal in snapshot.get("goal_recommendations", []):
        lines.append(f"| `{md(goal.get('title'))}` | `{md(goal.get('recommended_ref'))}` | {md(goal.get('rationale'))} | {md(goal.get('confidence'))} |")
    lines.extend(["", "A recommendation is a reading recommendation, not a declaration that one branch is globally best. Branches serving different purposes are not directly comparable.", "", "## Current tips", "", "| Ref | Tip | Committer date | Subject | Dossier |", "|---|---|---|---|---|"])
    for branch in branches:
        lines.append(f"| `{md(branch['ref'])}` | `{md(branch.get('commit_sha'))}` | {md(branch.get('commit', {}).get('committer_date'))} | {md(branch.get('commit', {}).get('subject'))} | [open](branches/{slugify_ref(branch['ref'])}.md) |")
    lines.extend(["", "## Scope and base", "", f"- Base: `{snapshot['base_selection']['ref']}` at `{snapshot['base_selection'].get('sha')}`.", f"- Selection: {snapshot['base_selection'].get('rationale')}", f"- Ambiguity: {snapshot['base_selection'].get('ambiguity')}", f"- Remote refresh: `{capture.get('remote_refreshed')}`.", "", "## Since previous snapshot", "", f"- Previous: `{delta.get('previous_snapshot_id') or 'none'}`.", f"- Refs added/removed/moved: {len(delta.get('refs_added', []))} / {len(delta.get('refs_removed', []))} / {len(delta.get('refs_moved', []))}.", f"- Worktree changes: {len(delta.get('worktree_status_changes', []))}; documents added/removed/changed: {len(delta.get('documents_added', []))} / {len(delta.get('documents_removed', []))} / {len(delta.get('documents_changed', []))}.", "", "## Navigation", "", "- [Branch index](branch_index.md)", "- [Specifications and evidence index](specs_index.md)", "- [Project map](project_map.md)", "- [Goal-specific comparisons](comparisons.md)", "- [Data quality](data_quality.md)", "- [Methodology](../methodology.md)", "", "Direct repository facts are identified by ref, commit, path, blob, receipt, or Git command output. Interpretations are labeled as recommendations or profile judgments. Missing evidence remains UNKNOWN."])
    return "\n".join(lines)


def render_index(snapshot: dict[str, Any]) -> str:
    scope = snapshot["scope"]
    lines = ["# Branch index", "", f"Captured `{snapshot['capture']['timestamp_utc']}`. Every accessible non-symbolic local and remote-tracking branch has a dossier; symbolic `origin/HEAD` remains an inventory alias.", "", f"- Refs: {scope['ref_count']}; branch refs: {scope['branch_ref_count']}; local heads: {scope['local_head_count']}; remote-tracking: {scope['remote_tracking_count']}; tags: {scope['tag_count']}.", f"- Worktrees: {scope['worktree_count']}; inaccessible: {scope['inaccessible_worktree_count']}.", f"- Remote-tracking without local branch: {', '.join(f'`{item}`' for item in scope['remote_without_local']) or 'none'}.", f"- Local without upstream: {', '.join(f'`{item}`' for item in scope['local_without_upstream']) or 'none'}.", "", "## Branches", "", "| Ref | Tip | Family | Base ahead / behind | Upstream ahead / behind | Flags | Worktree | Dossier |", "|---|---|---|---:|---:|---|---|---|"]
    for branch in snapshot.get("branches", []):
        flags = []
        if branch.get("divergent_from_base"):
            flags.append("divergent")
        if branch.get("stale_looking"):
            flags.append("stale-looking")
        if branch.get("incomparable"):
            flags.append("incomparable")
        if any(status.get("user_dirty") for status in branch.get("dirty_overlay", [])):
            flags.append("dirty")
        if branch.get("base", {}).get("upstream_behind"):
            flags.append("upstream-behind")
        if branch.get("base", {}).get("upstream_ahead"):
            flags.append("upstream-ahead")
        if (branch.get("loc") or {}).get("loc_comparison_limited"):
            flags.append("loc-limited")
        if not flags:
            flags.append("none")
        worktree = ", ".join(f"`{item}`" for item in branch.get("worktree_paths", [])) or "none"
        base = branch.get("base", {})
        lines.append(f"| `{md(branch['ref'])}` | `{md(branch.get('commit_sha'))}` | {md(branch.get('profile', {}).get('family'))} | {md(base.get('ahead'))} / {md(base.get('behind'))} | {md(base.get('upstream_ahead'))} / {md(base.get('upstream_behind'))} | {md(', '.join(flags))} | {md(worktree)} | [open](branches/{slugify_ref(branch['ref'])}.md) |")
    lines.extend(["", "## Duplicate tip groups", ""])
    for group in snapshot.get("ref_groups", []):
        if group.get("ref_count", 0) > 1:
            lines.append(f"- `{group['commit_sha']}`: {', '.join(f'`{item}`' for item in group['refs'])}")
    lines.extend(["", "Ahead/behind is relative to the recorded comparison base and is not a scientific score. A branch can be ahead because it contains experiments, generated artifacts, infrastructure, or stale documentation."])
    return "\n".join(lines)


def render_dossier(branch: dict[str, Any]) -> str:
    profile = branch.get("mission_and_soul", {})
    loc = branch.get("loc") or {}
    activity = branch.get("activity", {})
    lines = [f"# {branch['ref']}", "", f"**Type:** `{md(branch.get('ref_type'))}`  ", f"**Tip:** `{md(branch.get('commit_sha'))}`  ", f"**Commit:** {md(branch.get('commit', {}).get('subject'))}  ", f"**Committer date:** {md(branch.get('commit', {}).get('committer_date'))}  ", f"**Family:** {md(profile.get('kind'))} / `{md(branch.get('profile', {}).get('family'))}`  ", f"**Captured:** `{branch.get('capture_timestamp')}`", "", "## Identity and custody", "", f"- Upstream: `{md(branch.get('upstream') or 'none')}` ({md(branch.get('upstream_status'))}); upstream divergence ahead/behind: {md(branch.get('base', {}).get('upstream_ahead'))}/{md(branch.get('base', {}).get('upstream_behind'))}.", f"- Base: `{md(branch.get('base', {}).get('base_sha'))}` by `{md(branch.get('base', {}).get('method'))}`; ahead/behind {md(branch.get('base', {}).get('ahead'))}/{md(branch.get('base', {}).get('behind'))}.", f"- Worktrees: {', '.join(f'`{item}`' for item in branch.get('worktree_paths', [])) or 'none'}; dirty state is an overlay, not committed tip content.", f"- Unique commits: {md(activity.get('branch_unique_commit_count', 0))}; first/last UTC: {md(activity.get('first_branch_unique_commit_date'))} / {md(activity.get('last_branch_unique_commit_date'))}; days since last: {md(activity.get('days_since_last_branch_unique_commit'))}.", "", "## Mission and soul", "", f"**Problem:** {md(profile.get('mission'))}", "", f"**Thesis/design approach:** {md(profile.get('central_thesis'))}", "", f"**Role in An-Ra:** {md(profile.get('role_in_program'))}", "", f"**Unique contribution:** {md(profile.get('unique_contribution'))}", "", f"**Strongest evidence-backed result:** {md(profile.get('strongest_evidence_backed_result'))}", "", f"**Unresolved question:** {md(profile.get('most_important_unresolved_question'))}", "", f"**Falsifier/failure condition:** {md(profile.get('clearest_falsifier_or_failure_condition'))}", "", f"**Read first:** {', '.join(f'`{item}`' for item in profile.get('read_first', [])) or 'none'}", "", "These fields are reviewed interpretation grounded in the profile citations; they do not upgrade the status labels.", "", "## Status labels", "", "| Label | State | Confidence | Evidence |", "|---|---|---|---|"]
    for status in branch.get("status", {}).get("labels", []):
        anchors = "; ".join(citation(item) for item in status.get("evidence", [])) or status.get("basis", "")
        lines.append(f"| {md(status.get('status'))} | {md(status.get('state'))} | {md(status.get('confidence'))} | {md(anchors)} |")
    flow = activity.get("cumulative_flow", {})
    lines.extend(["", "## Committed tip stock", "", f"- Production source lines: **{md(loc.get('source_lines'))}**; tests: **{md(loc.get('test_lines'))}**; docs: **{md(loc.get('documentation_lines'))}**; notebook raw lines/cells: **{md(loc.get('notebook_lines'))} / {md(loc.get('notebook_cells'))}**.", f"- Tracked blob files/bytes: **{md(loc.get('tracked_blob_files'))} / {md(loc.get('tracked_bytes'))}**; binary files/bytes: **{md(loc.get('binary_files'))} / {md(loc.get('binary_bytes'))}**.", f"- Exact base-to-tip source stock change: **{md(activity.get('stock_delta', {}).get('absolute_source_line_change'))}** lines ({md(activity.get('stock_delta', {}).get('source_size_percent_change'))}%).", f"- Excluded samples: {len(loc.get('excluded_paths', []))}; large/uninspected samples: {len(loc.get('large_or_uninspected', []))}; source-category large files: {md(loc.get('large_source_files_uninspected'))}.", f"- LOC comparison limited by uninspected large source files: **{md(loc.get('loc_comparison_limited'))}**.", "", "### Production source language stock", "", "| Language | Source lines |", "|---|---:|"])
    for language, count in sorted((loc.get("source_language_lines") or loc.get("language_lines") or {}).items()):
        lines.append(f"| {md(language)} | {count:,} |")
    lines.extend(["", "## Change history", "", f"- History: `{md(activity.get('history_status'))}`.", f"- Commit flow added/removed/net/churn: {md(flow.get('source_lines_added'))} / {md(flow.get('source_lines_removed'))} / {md(flow.get('source_lines_added', 0) - flow.get('source_lines_removed', 0))} / {md(flow.get('source_lines_churn'))}.", f"- Merge first-parent added/removed: {md(activity.get('merge_first_parent_flow', {}).get('source_lines_added'))} / {md(activity.get('merge_first_parent_flow', {}).get('source_lines_removed'))}.", f"- Recent windows: `{md(json.dumps(activity.get('recent', {}), sort_keys=True))}`.", "", "Daily rows and formulas are in `../../data/latest_snapshot.json` and `../../data/history/daily_metrics.csv`.", "", "## Specifications, plans, and evidence", "", f"This tip indexes {len(branch.get('specifications_and_documents', []))} relevant documents. The complete record is in [specs_index.md](../specs_index.md).", "", "| Path | Type | Authority | Date | Summary |", "|---|---|---|---|---|"])
    for document in branch.get("specifications_and_documents", [])[:40]:
        lines.append(f"| `{md(document.get('path'))}` | {md(document.get('type'))} | {md(document.get('authority_status'))} | {md(document.get('document_date'))} | {md(document.get('summary'))} |")
    if len(branch.get("specifications_and_documents", [])) > 40:
        lines.append(f"| … | … | … | … | {len(branch['specifications_and_documents']) - 40} more in complete index |")
    lines.extend(["", "## Authority and contradictions", ""])
    notes = profile.get("authority_notes", [])
    lines.extend([f"- {md(note)}" for note in notes] or ["- No branch-specific authority note was configured; treat unclear documents as non-authoritative."])
    lines.extend(["", "## Claim ceiling", "", "Do not infer scientific success from implementation, tests, LOC, model size, benchmark-shaped files, or recent commits. Use the status labels and cited receipts. This dossier is an audit aid, not an experiment record or merge recommendation."])
    return "\n".join(lines)


def render_specs(snapshot: dict[str, Any]) -> str:
    lines = ["# Specifications and evidence index", "", f"Complete index generated at `{snapshot['capture']['timestamp_utc']}`. Each record is tied to a ref, commit, path, and blob; summaries are bounded extracts.", "", "| Ref | Path | Blob | Type | Authority | Date | Summary | Sections | Conflict signals | Claim support |", "|---|---|---|---|---|---|---|---|---|---|"]
    for document in snapshot.get("documents", []):
        sections = "; ".join(document.get("sections", [])[:8])
        conflicts = "; ".join(document.get("conflicts_or_corrections", [])[:3])
        lines.append(f"| `{md(document.get('ref'))}` | `{md(document.get('path'))}` | `{md(document.get('blob_sha'))}` | {md(document.get('type'))} | {md(document.get('authority_status'))} | {md(document.get('document_date'))} | {md(document.get('summary'))} | {md(sections)} | {md(conflicts)} | {md(document.get('claim_support'))} |")
    lines.extend(["", "## Index rules", "", "- Authority is a document-level classification, not automatic truth; `canonical` and `current` mean the document claims that role.", "- Superseded, historical, draft, and unclear records remain visible; no filename is silently promoted.", "- Receipt/test references are recorded as references. Executable support requires a matching blob and independently inspectable result; missing external artifacts stay explicit."])
    return "\n".join(lines)


def render_map(snapshot: dict[str, Any], config: dict[str, Any]) -> str:
    lines = ["# Project map", "", "This combines Git ancestry measurements with explicitly interpretive program families. Family labels are not encoded by Git and are not automatic authority.", "", "## Branch families", "", "| Family | Purpose | Refs | Evidence owner |", "|---|---|---|---|"]
    for family in config.get("families", []):
        lines.append(f"| **{md(family.get('name'))}** | {md(family.get('purpose'))} | {md(', '.join(f'`{item}`' for item in family.get('refs', [])))} | {md(family.get('evidence_owner'))} |")
    lines.extend(["", "## Purpose and dependency diagram", "", "```mermaid", "graph TD"])
    ids = {}
    for family in config.get("families", []):
        family_id = re.sub(r"[^A-Za-z0-9_]", "_", family.get("id", family.get("name", "family")))
        ids[family.get("id", family.get("name"))] = family_id
        lines.append(f'  {family_id}["{family.get("name", family_id)}"]')
    for family in config.get("families", []):
        family_id = ids[family.get("id", family.get("name"))]
        for parent in family.get("depends_on", []):
            lines.append(f"  {ids.get(parent, re.sub(r'[^A-Za-z0-9_]', '_', parent))} --> {family_id}")
    lines.extend(["```", "", "## Shared conclusions", "", "- Implementation, local verification, hardware qualification, bounded execution, replication, and broad scientific claims are separate labels.", "- Complete-answer termination, held-out boundaries, shortcut/contamination controls, custody, and negative results recur across the records.", "- External corpus, target topology, credentials, sealed fixtures, and complete checkpoint custody remain external blockers in multiple ledgers.", "", "## Live disagreements and corrections", "", "- Historical prose and immutable receipts can disagree with newer ledgers; the correction record must be read with the receipt.", "- Prelaunch/readiness language is scoped and does not authorize production training or AGI claims.", "- Ahead/behind counts measure history, not scientific quality.", "", "## Largest unknowns", "", "- Whether missing external corpora, target topology, sealed custody, and checkpoint trees can be obtained without changing frozen protocols.", "- Which bounded development findings replicate on fresh tasks and subjects.", "- Which implementation and evidence branches can be compared without mixing purposes or incompatible histories.", "", "## Glossary", "", "| Term | Definition | Source basis |", "|---|---|---|"])
    for item in config.get("glossary", []):
        lines.append(f"| **{md(item.get('term'))}** | {md(item.get('definition'))} | {md(item.get('source'))} |")
    return "\n".join(lines)


def score(profile: dict[str, Any], dimension: str) -> str:
    value = (profile.get("scores") or {}).get(dimension)
    if value is None:
        return "unknown"
    if isinstance(value, dict):
        return str(value.get("label", "unknown"))
    return str(value)


def render_comparisons(snapshot: dict[str, Any], config: dict[str, Any]) -> str:
    dimensions = ["goal_clarity", "specification_completeness", "implementation_completeness", "reproducibility", "evaluation_quality", "provenance_and_custody", "hardware_qualification", "scientific_result_maturity", "negative_evidence_handling", "operational_readiness", "unresolved_blockers", "stale_instruction_risk", "unique_contribution"]
    lines = ["# Goal-specific comparisons", "", "There is no universal branch league table. A score is meaningful only inside a stated goal and family. `unknown` is not zero.", "", "## Rubric", "", "- 5 = strong, directly evidenced and current for the stated scope; 4 = good with bounded caveats; 3 = mixed/partial; 2 = substantial blocker or stale authority; 1 = weak or contradicted; `unknown` = not evidenced.", "- Scientific maturity is not implementation completeness. A branch may be operationally useful and scientifically negative, or promising and unexecuted.", "- Scores are reviewed profile judgments, never LOC, commits, model size, benchmark, or last-modified proxies.", ""]
    for goal in config.get("goals", []):
        lines.extend([f"## {md(goal.get('title'))}", "", f"**Question:** {md(goal.get('question'))}", "", f"**Recommendation:** `{md(goal.get('recommended_ref'))}` (confidence: {md(goal.get('confidence'))}).", "", f"**Why:** {md(goal.get('rationale'))}", "", f"**Tradeoffs:** {md(goal.get('tradeoff'))}", "", "| Candidate | " + " | ".join(item.replace("_", " ") for item in dimensions) + " |", "|---" + "|---" * len(dimensions) + "|"])
        for ref in goal.get("candidate_refs", []):
            branch = next((item for item in snapshot.get("branches", []) if item.get("ref") == ref), None)
            profile = branch.get("profile", {}) if branch else {}
            lines.append(f"| `{md(ref)}` | " + " | ".join(score(profile, item) for item in dimensions) + " |")
        lines.extend(["", "**Evidence anchors:**"])
        for item in goal.get("evidence", []):
            lines.append(f"- `{md(item.get('ref'))}:{md(item.get('path'))}`" + (f" § {md(item.get('section'))}" if item.get("section") else ""))
        if goal.get("resolution_if_no_winner"):
            lines.extend(["", f"**Smallest resolving comparison:** {md(goal['resolution_if_no_winner'])}"])
        lines.append("")
    lines.extend(["## Additional decision views", "", "- **Data integrity:** use Citadel's evidence ledger/protocol for contamination and shortcut audit; use the Cyhex integrity audit for custody and receipt verification. Neither is production promotion authority.", "- **Formation-Mux recovery:** use the Cymek-beta recovery runbook and session report; the required original checkpoint tree is external and the campaign remains partial.", "- **Architecture decision:** use the CS-TRANSFER-001 decision for its bounded vocabulary conclusion and retain its explicit prohibition on production-scale authorization.", "", "## Non-comparisons", "", "Do not compare an evidence archive, training branch, operator notebook, and Core runtime as candidates for the same scientific claim. Their valid comparison is functional: custody, implementation, execution, or decision authority."])
    return "\n".join(lines)


def render_quality(snapshot: dict[str, Any], validation: dict[str, Any], verification: dict[str, Any] | None = None) -> str:
    lines = ["# Data quality and validation", "", f"Status: **{validation.get('status')}** at `{snapshot['capture']['timestamp_utc']}`.", "", "| Check | Status | Details |", "|---|---|---|"]
    for check in validation.get("checks", []):
        lines.append(f"| `{md(check.get('check'))}` | **{md(check.get('status'))}** | `{md(json.dumps(check.get('details'), sort_keys=True))}` |")
    lines.extend(["", "## Validation method", "", "- JSON required fields, local object resolution, daily percentage recomputation, commit deduplication, dirty-overlay separation, category totals, and report-citation resolution are checked without experiments.", "- Branch LOC is read from committed trees; worktree changes are separate. Observatory-generated files in the dedicated worktree are labeled generated overlay.", "- Two scans against unchanged refs/worktree state are compared after removing capture timestamps and tool capture metadata; sorted refs, paths, classifications, and atomic writes make the remaining content deterministic.", "- `make lint` and `make typecheck` could not run because `make` is unavailable; direct `ruff` and `mypy` modules are also unavailable in the runtime. No lint/typecheck success is inferred.", "- No training, inference, GPU/TPU work, benchmark, sealed evaluation, fetch, merge, reset, clean, stash, or branch-switch command is part of the tool.", "", "## Limitations", "", "- The first snapshot cannot recover old dirty overlays, deleted commits, external receipts, or prior working-tree states.", "- Commit dates are metadata; daily source stock rows are estimates when merge/side-branch flow overlaps.", "- Diff flow excludes the tracked deleted-remnant object tree and model/binary payload paths listed in the tool implementation; their stock metadata remains visible where committed.", "- Large artifacts, weights, datasets, binary files, and ignored trees are not recursively loaded.", "- Document authority and mission profiles are human-reviewed interpretation, not automatic truth.", "", "## Warnings", ""])
    lines.extend([f"- {md(item)}" for item in snapshot.get("warnings", [])])
    if verification:
        lines.extend(["", "## Last verify", "", "```json", json.dumps(verification, indent=2, sort_keys=True), "```"])
    return "\n".join(lines)


def history_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.json")) if directory.exists() else []


def write_daily_csv(path: Path, snapshots: list[dict[str, Any]]) -> None:
    fields = ["snapshot_id", "scope", "ref", "utc_day", "source_lines_added", "source_lines_removed", "source_lines_net", "source_lines_churn", "source_size_start_of_day", "source_size_end_of_day", "source_growth_percent", "commit_count", "files_changed", "docs_lines_added", "docs_lines_removed", "test_lines_added", "test_lines_removed", "notebook_changes", "generated_artifact_changes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for snapshot in snapshots:
            for branch in snapshot.get("branches", []):
                for row in branch.get("activity", {}).get("daily", []):
                    writer.writerow({"snapshot_id": snapshot.get("snapshot_id"), "scope": "branch", "ref": branch.get("ref"), **{field: row.get(field) for field in fields[3:]}})
            for row in snapshot.get("project_activity", {}).get("daily", []):
                writer.writerow({"snapshot_id": snapshot.get("snapshot_id"), "scope": "project_deduplicated", "ref": "PROJECT", **{field: row.get(field) for field in fields[3:]}})
    os.replace(temporary, path)


def write_outputs(snapshot: dict[str, Any], output: Path, config: dict[str, Any], append_history: bool, previous: dict[str, Any] | None) -> None:
    paths = ensure_output(output)
    write_json(paths["latest"], snapshot)
    snapshots = [load_json(path) for path in history_files(paths["history_dir"])]
    snapshots = [item for item in snapshots if item]
    if append_history:
        name = f"{snapshot['snapshot_id']}.json"
        target = paths["history_dir"] / name
        suffix = 1
        while target.exists():
            target = paths["history_dir"] / f"{snapshot['snapshot_id']}-{suffix}.json"
            suffix += 1
        write_json(target, snapshot)
    if not any(item.get("snapshot_id") == snapshot.get("snapshot_id") for item in snapshots):
        snapshots.append(snapshot)
    write_daily_csv(paths["daily"], snapshots)
    write_text(paths["latest_report"], render_latest(snapshot))
    write_text(paths["branch_index"], render_index(snapshot))
    for branch in snapshot.get("branches", []):
        write_text(paths["branches_dir"] / f"{slugify_ref(branch['ref'])}.md", render_dossier(branch))
    write_text(paths["specs"], render_specs(snapshot))
    write_text(paths["project_map"], render_map(snapshot, config))
    write_text(paths["comparisons"], render_comparisons(snapshot, config))
    write_text(paths["quality"], render_quality(snapshot, snapshot.get("validation", {})))


def command_scan(args: argparse.Namespace, append_history: bool) -> int:
    output = Path(args.output).resolve()
    repository_path = Path(args.repo).resolve()
    config_path = Path(args.config).resolve() if args.config else output / "config.json"
    config = load_config(config_path)
    repository = GitRepository(repository_path)
    previous = load_json(output / "data" / "latest_snapshot.json")
    snapshot = build_snapshot(repository, config, utc_now(), previous)
    write_outputs(snapshot, output, config, append_history, previous)
    result = {"command": "refresh" if append_history else "scan", "snapshot_id": snapshot["snapshot_id"], "output": output.as_posix(), "validation": snapshot["validation"], "refs": snapshot["scope"]["ref_count"], "branch_dossiers": len(snapshot["branches"]), "worktrees": snapshot["scope"]["worktree_count"], "remote_refreshed": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if snapshot["validation"]["status"] != "FAIL" else 1


def command_verify(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    repository_path = Path(args.repo).resolve()
    config_path = Path(args.config).resolve() if args.config else output / "config.json"
    latest = load_json(output / "data" / "latest_snapshot.json")
    if not latest:
        raise AtlasError("No latest snapshot exists; run refresh first")
    config = load_config(config_path)
    repository = GitRepository(repository_path)
    current = build_snapshot(repository, config, utc_now(), latest)
    delta = current["previous_snapshot_delta"]
    stale_reasons = list(delta.get("report_stale_reasons", []))
    if current.get("validation", {}).get("status") == "FAIL":
        stale_reasons.append("current snapshot validation failed")
    report_files = [output / "reports" / "latest.md", output / "reports" / "branch_index.md", output / "reports" / "specs_index.md", output / "reports" / "project_map.md", output / "reports" / "comparisons.md", output / "reports" / "data_quality.md"]
    missing_report_files = [path.relative_to(output).as_posix() for path in report_files if not path.exists()]
    report_stale = bool(missing_report_files or stale_reasons or any(delta.get(key) for key in ("refs_added", "refs_removed", "refs_moved", "worktree_status_changes", "documents_added", "documents_removed", "documents_changed")))
    result = {"status": "STALE" if report_stale else "CURRENT", "latest_snapshot_id": latest.get("snapshot_id"), "current_snapshot_id": current.get("snapshot_id"), "schema_match": latest.get("schema_version") == current.get("schema_version"), "tool_source_match": latest.get("tool", {}).get("source_sha256") == current.get("tool", {}).get("source_sha256"), "delta": delta, "validation": current.get("validation"), "reports_stale": report_stale, "missing_generated_report_files": missing_report_files, "generated_report_files": sorted(path.relative_to(output).as_posix() for path in report_files if path.exists())}
    write_json(output_paths(output)["verification"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["validation"].get("status") != "FAIL" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Read-only An-Ra branch and evidence observatory")
    value.add_argument("command", choices=["scan", "refresh", "verify"])
    value.add_argument("--repo", default=".", help="Repository path; no ref-changing Git operation is performed")
    value.add_argument("--output", default=str(Path(__file__).resolve().parents[1]), help="Observatory output directory")
    value.add_argument("--config", default="", help="Optional config.json path")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "scan":
            return command_scan(args, False)
        if args.command == "refresh":
            return command_scan(args, True)
        return command_verify(args)
    except AtlasError as error:
        print(f"branch_atlas: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
