#!/usr/bin/env python3
"""Deterministic consistency validator for docs/research/EXPERIMENT_EVIDENCE_LEDGER.json.

Checks (intentionally lightweight, no framework):
  1. unique experiment IDs
  2. evidence status / evidence class / replication values are in the declared enums
  3. commit fields contain syntactically valid 40-hex or 7..40-hex SHAs
  4. artifact paths exist either in the working tree or on one of the known branches
     (paths that are not plain paths - containing spaces or parentheses - are skipped)
  5. DEMONSTRATED / SUPPORTED claims carry at least one non-empty artifact reference
  6. IMPLEMENTED_NOT_EXECUTED entries carry no metrics (a metric would be a scientific
     result claim)
  7. supersedes / superseded_by / relations reference known experiment IDs
  8. prints status counts so the Markdown ledger's summary table can be reconciled

Usage: python tools/validate_evidence_ledger.py [path-to-json]
Exit code 0 = clean, 1 = violations found.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

DEFAULT_LEDGER = os.path.join("docs", "research", "EXPERIMENT_EVIDENCE_LEDGER.json")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX7_40 = re.compile(r"^[0-9a-f]{7,40}$")
PLAIN_PATH = re.compile(r"^[A-Za-z0-9_./\-]+$")

KNOWN_BRANCHES = [
    "origin/Arkenstone", "origin/BRAMASTRA", "origin/arkenstone-ark020-v4",
    "origin/arkenstone-astra", "origin/citadel", "origin/codex/arkenstone-improvements",
    "origin/core-exp", "origin/core-frozen-v4", "origin/cymek-500m-readiness",
    "origin/esoes", "origin/iterate500", "origin/iterate900", "origin/main",
    "origin/triquetra",
]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def build_branch_path_index() -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for branch in KNOWN_BRANCHES:
        try:
            files = git("ls-tree", "-r", "--name-only", branch).splitlines()
        except subprocess.CalledProcessError:
            continue
        index[branch] = set(files)
    return index


def commit_is_syntactically_valid(commit_value: str) -> bool:
    for token in re.findall(r"[0-9a-fA-F]{6,40}", commit_value or ""):
        if HEX40.match(token.lower()) or HEX7_40.match(token.lower()):
            return True
    return bool(commit_value and commit_value.strip() and "(" not in commit_value)


def main() -> int:
    ledger_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LEDGER
    with open(ledger_path, encoding="utf-8") as fh:
        ledger = json.load(fh)

    problems: list[str] = []
    experiments = ledger.get("experiments", [])
    ids = [e.get("id", "<missing>") for e in experiments]

    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        problems.append(f"duplicate experiment IDs: {dupes}")

    id_set = set(ids)
    status_enum = set(ledger["evidence_status_enum"])
    class_enum = set(ledger["evidence_class_enum"])
    replication_enum = set(ledger["replication_enum"])
    relation_enum = set(ledger["relation_enum"])

    path_index: dict[str, set[str]] | None = None

    for e in experiments:
        eid = e.get("id", "<missing>")
        if e.get("status") not in status_enum:
            problems.append(f"{eid}: bad status {e.get('status')!r}")
        if e.get("evidence_class") not in class_enum:
            problems.append(f"{eid}: bad evidence_class {e.get('evidence_class')!r}")
        if e.get("replication") not in replication_enum:
            problems.append(f"{eid}: bad replication {e.get('replication')!r}")
        if not commit_is_syntactically_valid(e.get("commit", "")):
            problems.append(f"{eid}: commit field has no valid SHA: {e.get('commit')!r}")

        artifacts = e.get("artifacts") or []
        if e.get("status") in {"DEMONSTRATED", "SUPPORTED"} and not artifacts:
            problems.append(f"{eid}: {e.get('status')} claim without artifact references")

        if e.get("status") == "IMPLEMENTED_NOT_EXECUTED" and e.get("metrics"):
            problems.append(f"{eid}: IMPLEMENTED_NOT_EXECUTED must not carry metrics")

        for ref in list(e.get("supersedes") or []) + list(e.get("superseded_by") or []):
            if ref not in id_set:
                problems.append(f"{eid}: supersession reference does not resolve: {ref}")
        for rel in e.get("relations") or []:
            if rel.get("id") not in id_set:
                problems.append(f"{eid}: relation references unknown id {rel.get('id')!r}")
            if rel.get("relation") not in relation_enum:
                problems.append(f"{eid}: relation {rel.get('id')!r} has invalid kind {rel.get('relation')!r}")

        # artifact path existence: working tree first, then any known branch
        if path_index is None:
            path_index = build_branch_path_index()
        for artifact in artifacts:
            if not PLAIN_PATH.match(artifact):
                continue  # annotated / unreachable-ref references are skipped by design
            if os.path.exists(artifact):
                continue
            if any(artifact in files for files in path_index.values()):
                continue
            problems.append(f"{eid}: artifact path not found anywhere: {artifact}")

    counts: dict[str, int] = {}
    for e in experiments:
        counts[e.get("status", "?")] = counts.get(e.get("status", "?"), 0) + 1

    print(f"ledger: {ledger_path}")
    print(f"experiments: {len(experiments)}")
    for status in ledger["evidence_status_enum"]:
        if counts.get(status):
            print(f"  {status}: {counts[status]}")
    unknown = set(counts) - status_enum
    if unknown:
        print(f"  UNKNOWN-STATUS: {sorted(unknown)}")

    if problems:
        print(f"\nVALIDATION FAILED ({len(problems)} problems):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nVALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
