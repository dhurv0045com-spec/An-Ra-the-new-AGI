"""Shared structured logging for An-Ra components.

Provides TRACE/DEBUG/INFO/WARN/ERROR/AUDIT levels, a required event envelope,
append-only audit logging for self-modification + file mutations, and
session lifecycle hooks (rotation + Drive sync) intended to be orchestrated by
L0.2.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from anra_paths import DRIVE_DIR, STATE_DIR

TRACE_LEVEL_NUM = 5
AUDIT_LEVEL_NUM = 25
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
logging.addLevelName(AUDIT_LEVEL_NUM, "AUDIT")

DEFAULT_LOG_DIR = STATE_DIR / "logs"
AUDIT_LOG = DEFAULT_LOG_DIR / "AUDIT_LOG.jsonl"
SESSION_LOG = DEFAULT_LOG_DIR / "session.log"


class SharedLogger(logging.Logger):
    def trace(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(TRACE_LEVEL_NUM):
            self._log(TRACE_LEVEL_NUM, msg, args, **kwargs)

    def audit(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(AUDIT_LEVEL_NUM):
            self._log(AUDIT_LEVEL_NUM, msg, args, **kwargs)


logging.setLoggerClass(SharedLogger)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _envelope(level: str, event_type: str, component: str, action: str,
              actor: str = "system", message: str = "", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ts": _iso_now(),
        "level": level,
        "event_type": event_type,
        "component": component,
        "action": action,
        "actor": actor,
        "message": message,
        "details": details or {},
        "pid": os.getpid(),
    }


def _append_jsonl(path: Path, entry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def setup_shared_logging(log_dir: Path = DEFAULT_LOG_DIR, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(TRACE_LEVEL_NUM)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fh = logging.FileHandler(log_dir / "service.log", encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setLevel(getattr(logging, level.upper(), logging.INFO))
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def get_shared_logger(name: str) -> SharedLogger:
    return logging.getLogger(name)  # type: ignore[return-value]


def emit_event(logger: SharedLogger, level: str, event_type: str, component: str, action: str,
               actor: str = "system", message: str = "", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    env = _envelope(level, event_type, component, action, actor, message, details)
    log_line = json.dumps(env, ensure_ascii=False)
    lvl = level.upper()
    if lvl == "TRACE":
        logger.trace(log_line)
    elif lvl == "DEBUG":
        logger.debug(log_line)
    elif lvl == "INFO":
        logger.info(log_line)
    elif lvl == "WARN":
        logger.warning(log_line)
    elif lvl == "ERROR":
        logger.error(log_line)
    elif lvl == "AUDIT":
        logger.audit(log_line)
    else:
        logger.info(log_line)
    return env


def emit_audit_event(logger: SharedLogger, event_type: str, component: str, action: str,
                     actor: str = "system", message: str = "", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    env = emit_event(logger, "AUDIT", event_type, component, action, actor, message, details)
    _append_jsonl(AUDIT_LOG, env)
    return env


def log_self_modification(logger: SharedLogger, component: str, target: str, diff_ref: str, actor: str = "system") -> Dict[str, Any]:
    return emit_audit_event(
        logger,
        event_type="SELF_MODIFICATION",
        component=component,
        action="modify_code",
        actor=actor,
        message=f"Self-modification applied to {target}",
        details={"target": target, "diff_ref": diff_ref},
    )


def log_file_mutation(logger: SharedLogger, component: str, path: str, mutation: str, actor: str = "system") -> Dict[str, Any]:
    return emit_audit_event(
        logger,
        event_type="FILE_MUTATION",
        component=component,
        action=mutation,
        actor=actor,
        message=f"File mutation {mutation}: {path}",
        details={"path": path, "mutation": mutation},
    )


@dataclass
class L02SessionLogManager:
    """L0.2-managed session hooks for log rotation and Drive sync."""

    session_id: str
    log_dir: Path = DEFAULT_LOG_DIR
    drive_root: Path = DRIVE_DIR

    def _session_log_path(self) -> Path:
        return self.log_dir / f"session_{self.session_id}.log"

    def rotate_for_new_session(self) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if SESSION_LOG.exists():
            archived = self._session_log_path()
            SESSION_LOG.replace(archived)
        return SESSION_LOG

    def sync_session_end_to_drive(self) -> Optional[Path]:
        if not self.drive_root.exists():
            return None
        out_dir = self.drive_root / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        src = self._session_log_path() if self._session_log_path().exists() else SESSION_LOG
        if not src.exists():
            return None
        dst = out_dir / src.name
        dst.write_bytes(src.read_bytes())
        return dst
