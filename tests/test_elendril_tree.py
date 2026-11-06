"""Fail closed on drift of the archived tree or its retained evidence."""

import json
from pathlib import Path

from tools.audit_tree import archive_inventory, git


def test_archive_inventory_matches_git_and_pruned_paths_stay_archived():
    recorded = json.loads(Path("docs/TREE_INVENTORY.json").read_bytes())
    assert recorded == archive_inventory()
    assert git("rev-parse", "elendril-pre-v3^{commit}").decode().strip() == recorded["archive_commit"]
    active = set(git("ls-files", "-z").decode().split("\0"))
    for path, entry in recorded["files"].items():
        if entry["action"] in {"KEEP", "REIMPLEMENT"}:
            assert path in active, path
        else:
            assert path not in active, path


def test_unique_elendril_evidence_remains_at_identical_git_blobs():
    historical = archive_inventory()
    current = {}
    for row in git("ls-files", "--stage", "-z", "--", "artifacts/elendril").split(b"\0"):
        if row:
            metadata, path = row.split(b"\t", 1)
            _, digest, stage = metadata.decode().split()
            assert stage == "0"
            current[path.decode()] = digest
    for path, entry in historical["files"].items():
        if path.startswith("artifacts/elendril/"):
            assert current[path] == entry["git_object"], path
