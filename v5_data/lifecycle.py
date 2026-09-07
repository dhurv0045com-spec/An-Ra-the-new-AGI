"""Dataset lifecycle states for the production data path.

The supply audit (`materialize_first_party`) reports honest availability;
this machine tracks dataset readiness through explicit states so no missing
state can collapse into RUNNABLE. Transitions are append-only history with
content-bound receipts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


LIFECYCLE_SCHEMA = "anra-v5-dataset-lifecycle/v1"

STATES = ("DECLARED", "MATERIALIZED", "VERIFIED", "DEDUPED", "QUALIFIED",
          "TOKENIZED", "PACKED", "RUNNABLE")

_FORWARD = {name: (STATES[index + 1] if index + 1 < len(STATES) else None)
            for index, name in enumerate(STATES)}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


@dataclass
class DatasetLifecycle:
    """Append-only lifecycle tracker for one dataset lineage."""

    lineage_id: str
    state: str = "DECLARED"
    history: list = field(default_factory=list)

    def assert_valid(self) -> None:
        if not self.lineage_id or "/" in self.lineage_id or "\\" in self.lineage_id:
            raise ValueError("lifecycle lineage id must be a safe single component")
        if self.state not in STATES:
            raise ValueError(f"unknown lifecycle state: {self.state}")

    def advance(self, *, evidence_sha256: str, note: str = "") -> "DatasetLifecycle":
        """Move exactly one state forward with content-bound evidence."""

        self.assert_valid()
        if len(evidence_sha256) != 64 or any(
                c not in "0123456789abcdef" for c in evidence_sha256):
            raise ValueError("lifecycle evidence must be a lowercase SHA-256")
        following = _FORWARD[self.state]
        if following is None:
            raise ValueError("a RUNNABLE dataset cannot advance further")
        self.history.append({"from": self.state, "to": following,
                             "evidence_sha256": evidence_sha256, "note": note})
        self.state = following
        return self

    def receipt(self) -> dict[str, object]:
        self.assert_valid()
        body: dict[str, object] = {
            "schema": LIFECYCLE_SCHEMA,
            "lineage_id": self.lineage_id,
            "state": self.state,
            "history": [dict(item) for item in self.history],
        }
        body["sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
        return body


def require_runnable(lifecycle: DatasetLifecycle | None) -> None:
    """Fail closed unless an explicitly tracked dataset reached RUNNABLE."""

    if lifecycle is None:
        raise ValueError("production ingestion requires a tracked dataset lifecycle")
    lifecycle.assert_valid()
    if lifecycle.state != "RUNNABLE":
        raise ValueError(
            f"dataset {lifecycle.lineage_id} is {lifecycle.state}, not RUNNABLE")


__all__ = ["STATES", "DatasetLifecycle", "LIFECYCLE_SCHEMA", "require_runnable"]
