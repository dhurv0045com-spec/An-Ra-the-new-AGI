from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any

from anra_paths import ROOT, SELF_MOD_AUDIT_LOG


def _under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def sovereignty_audit_change(file_path: str | Path, new_content: str, *, reason: str = "") -> dict[str, Any]:
    path = Path(file_path)
    allowed = True
    findings: list[str] = []
    if _under(path, ROOT / "history"):
        allowed = False
        findings.append("history is immutable")
    if path.suffix == ".pt":
        allowed = False
        findings.append("checkpoint mutation is blocked")
    if path.suffix == ".py":
        try:
            ast.parse(new_content, filename=str(path))
        except SyntaxError as exc:
            allowed = False
            findings.append(f"syntax error: {exc}")

    event = {
        "ts": time.time(),
        "event_type": "SELF_MOD_AUDIT",
        "component": "self_modification",
        "action": "approve" if allowed else "reject",
        "file": str(path),
        "reason": reason,
        "findings": findings,
    }
    try:
        SELF_MOD_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SELF_MOD_AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
    except Exception:
        pass
    try:
        from sovereignty.logger import audit_event

        audit_event("self_modification", "SELF_MOD_AUDIT", event["action"], details=event)
    except Exception:
        pass
    return {"allowed": allowed, "event": event}

