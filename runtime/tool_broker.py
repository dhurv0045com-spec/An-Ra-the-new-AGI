"""Explicit, bounded tools for the local V4 prototype.

This module deliberately does *not* turn a language model into an autonomous
operator.  A caller must first obtain a short-lived capability for one local
session, then invoke one allowlisted typed tool.  Results include stable hashes
and are written to the existing experience ledger, so they can be inspected
without treating tool output as model truth or silently mixing it into training.

The first production-shaped tool is an exact arithmetic evaluator.  It uses a
small AST interpreter rather than ``eval`` and has no filesystem, network, or
subprocess access.  Other tools must meet the same contract before registration.
"""

from __future__ import annotations

import ast
import math
import secrets
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal

from runtime.experience_ledger import content_hash, record_experience

TOOL_BROKER_SCHEMA = "anra-bounded-tool-broker/v1"
ToolName = Literal["calculator"]
_MAX_EXPRESSION_CHARS = 256
_MAX_ABS_VALUE = 1.0e100
_MAX_POWER = 12


class ToolPolicyError(PermissionError):
    """Raised when a tool request exceeds its explicit capability."""


class ToolInputError(ValueError):
    """Raised when typed input is invalid for an allowlisted tool."""


@dataclass(frozen=True)
class ToolGrant:
    """Server-issued, session-bound authority to call one small tool set."""

    capability_id: str
    session_id: str
    allowed_tools: tuple[ToolName, ...]
    max_calls: int
    expires_at: float
    issued_at: float
    principal: str = "local_prototype_operator"

    def public_view(self) -> dict[str, object]:
        return {
            "schema": TOOL_BROKER_SCHEMA,
            "capability_id": self.capability_id,
            "session_id": self.session_id,
            "allowed_tools": list(self.allowed_tools),
            "max_calls": self.max_calls,
            "expires_at": self.expires_at,
            "principal": self.principal,
        }


@dataclass(frozen=True)
class ToolReceipt:
    """Public, content-minimised proof of one accepted or refused tool call."""

    invocation_id: str
    tool: ToolName
    status: Literal["completed", "refused", "failed"]
    capability_id: str
    session_id: str
    arguments_hash: str
    result_hash: str | None
    calls_remaining: int
    duration_ms: float
    ledger_persisted: bool
    schema: str = TOOL_BROKER_SCHEMA

    def public_view(self) -> dict[str, object]:
        return asdict(self)


def _finite(value: float) -> float:
    if not math.isfinite(value) or abs(value) > _MAX_ABS_VALUE:
        raise ToolInputError("calculation result is outside the permitted numeric range")
    return value


def _bounded_number(value: int | float) -> int | float:
    """Keep integers exact while bounding finite floating-point results."""
    if isinstance(value, int):
        if abs(value) > int(_MAX_ABS_VALUE):
            raise ToolInputError("calculation result is outside the permitted numeric range")
        return value
    return _finite(value)


def _evaluate_arithmetic(expression: str) -> int | float:
    """Interpret a strictly numeric arithmetic expression without ``eval``."""
    text = str(expression).strip()
    if not text:
        raise ToolInputError("expression is required")
    if len(text) > _MAX_EXPRESSION_CHARS:
        raise ToolInputError(f"expression exceeds {_MAX_EXPRESSION_CHARS} characters")
    try:
        root = ast.parse(text, mode="eval").body
    except SyntaxError as error:
        raise ToolInputError("expression is not valid arithmetic") from error

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                raise ToolInputError("boolean values are not arithmetic inputs")
            number = float(node.value) if isinstance(node.value, float) else int(node.value)
            return _bounded_number(number)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if not isinstance(node, ast.BinOp):
            raise ToolInputError(f"unsupported arithmetic syntax: {type(node).__name__}")
        left, right = visit(node.left), visit(node.right)
        if isinstance(node.op, ast.Add):
            return _bounded_number(left + right)
        if isinstance(node.op, ast.Sub):
            return _bounded_number(left - right)
        if isinstance(node.op, ast.Mult):
            return _bounded_number(left * right)
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ToolInputError("division by zero")
            return _bounded_number(left / right)
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ToolInputError("division by zero")
            return _bounded_number(left // right)
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ToolInputError("division by zero")
            return _bounded_number(left % right)
        if isinstance(node.op, ast.Pow):
            if abs(float(right)) > _MAX_POWER:
                raise ToolInputError(f"exponent magnitude must not exceed {_MAX_POWER}")
            return _bounded_number(left**right)
        raise ToolInputError(f"unsupported arithmetic operator: {type(node.op).__name__}")

    result = visit(root)
    if isinstance(result, float) and result.is_integer() and abs(result) <= 2**63 - 1:
        return int(result)
    return result


class BoundedToolBroker:
    """In-process tool broker with server-issued authority and ledger receipts."""

    _SUPPORTED_TOOLS: frozenset[str] = frozenset({"calculator"})

    def __init__(self, *, ledger_source: str = "runtime.tool_broker") -> None:
        self._ledger_source = ledger_source
        self._grants: dict[str, ToolGrant] = {}
        self._calls: dict[str, int] = {}
        self._lock = threading.RLock()

    def issue_grant(
        self,
        *,
        session_id: str,
        allowed_tools: tuple[ToolName, ...] = ("calculator",),
        max_calls: int = 1,
        ttl_seconds: float = 120.0,
        principal: str = "local_prototype_operator",
    ) -> ToolGrant:
        normalized_session = str(session_id).strip()
        if not normalized_session:
            raise ToolInputError("session_id is required")
        tools = tuple(dict.fromkeys(str(name) for name in allowed_tools))
        if not tools or any(name not in self._SUPPORTED_TOOLS for name in tools):
            raise ToolInputError("only registered local tools can be granted")
        if not 1 <= int(max_calls) <= 8:
            raise ToolInputError("max_calls must be in [1, 8]")
        if not 1.0 <= float(ttl_seconds) <= 600.0:
            raise ToolInputError("ttl_seconds must be in [1, 600]")
        if not str(principal).strip():
            raise ToolInputError("principal is required")
        now = time.time()
        grant = ToolGrant(
            capability_id=secrets.token_urlsafe(24),
            session_id=normalized_session,
            allowed_tools=tools,  # type: ignore[arg-type]
            max_calls=int(max_calls),
            expires_at=now + float(ttl_seconds),
            issued_at=now,
            principal=str(principal).strip(),
        )
        with self._lock:
            self._grants[grant.capability_id] = grant
            self._calls[grant.capability_id] = 0
        return grant

    def revoke_session(self, session_id: str) -> int:
        """Destroy all session-bound capabilities; called on clear/unload."""
        with self._lock:
            ids = [key for key, grant in self._grants.items() if grant.session_id == session_id]
            for key in ids:
                self._grants.pop(key, None)
                self._calls.pop(key, None)
            return len(ids)

    def clear(self) -> None:
        with self._lock:
            self._grants.clear()
            self._calls.clear()

    def active_grant_count(self) -> int:
        now = time.time()
        with self._lock:
            self._purge_expired(now)
            return len(self._grants)

    def execute(
        self,
        *,
        capability_id: str,
        session_id: str,
        tool: ToolName,
        arguments: Mapping[str, object],
    ) -> tuple[dict[str, object], ToolReceipt]:
        """Run one typed local tool after all authority checks have passed."""
        started = time.perf_counter()
        normalized_args = dict(arguments)
        if len(str(normalized_args).encode("utf-8")) > 1024:
            raise ToolInputError("tool arguments exceed the 1024-byte budget")
        with self._lock:
            grant = self._validate_grant(capability_id, session_id, tool)
            used = self._calls[grant.capability_id]
            self._calls[grant.capability_id] = used + 1
            remaining = grant.max_calls - used - 1

        invocation_id = str(uuid.uuid4())
        argument_hash = content_hash(normalized_args)
        status: Literal["completed", "refused", "failed"] = "completed"
        result: dict[str, object]
        try:
            if tool == "calculator":
                expression = normalized_args.get("expression")
                if not isinstance(expression, str):
                    raise ToolInputError("calculator requires a string expression")
                value = _evaluate_arithmetic(expression)
                result = {"expression": expression.strip(), "value": value, "exact": True}
            else:  # pragma: no cover - validation makes this unreachable.
                raise ToolPolicyError(f"tool is not registered: {tool}")
        except ToolInputError as error:
            status = "refused"
            result = {"error": str(error), "exact": False}
        except Exception:
            status = "failed"
            result = {"error": "tool execution failed", "exact": False}
        finally:
            # A consumed call is intentional even for malformed input: retrying a
            # rejected request cannot turn a bounded grant into an unlimited loop.
            duration_ms = (time.perf_counter() - started) * 1000

        result_hash = content_hash(result)
        receipt = ToolReceipt(
            invocation_id=invocation_id,
            tool=tool,
            status=status,
            capability_id=grant.capability_id,
            session_id=grant.session_id,
            arguments_hash=argument_hash,
            result_hash=result_hash,
            calls_remaining=remaining,
            duration_ms=round(duration_ms, 3),
            ledger_persisted=False,
        )
        _, persisted = record_experience(
            trace_id=invocation_id,
            kind="tool_execution",
            inputs={
                "tool": tool,
                "arguments": normalized_args,
                "capability_id": grant.capability_id,
                "session_id": grant.session_id,
            },
            output={
                "tool": tool,
                "result_hash": result_hash,
                "exact": bool(result.get("exact", False)),
                "status": status,
            },
            verifier_verdicts=(
                {
                    "name": "typed_local_tool",
                    "passed": status == "completed",
                    "score": 1.0 if status == "completed" else 0.0,
                    "scope": (
                        "exact local arithmetic evaluation only"
                        if status == "completed"
                        else "rejected local arithmetic request"
                    ),
                },
            ),
            gate_record={
                "allowed": status == "completed",
                "gate": "server_issued_session_capability",
                "calls_remaining": remaining,
            },
            source=self._ledger_source,
            metadata={
                "schema": TOOL_BROKER_SCHEMA,
                "principal": grant.principal,
                "arguments_hash": argument_hash,
                "result_hash": result_hash,
                "expires_at": grant.expires_at,
            },
        )
        receipt = ToolReceipt(**{**asdict(receipt), "ledger_persisted": persisted})
        return result, receipt

    def _validate_grant(self, capability_id: str, session_id: str, tool: ToolName) -> ToolGrant:
        now = time.time()
        self._purge_expired(now)
        grant = self._grants.get(str(capability_id))
        if grant is None:
            raise ToolPolicyError("unknown, expired, or revoked tool capability")
        if grant.session_id != str(session_id):
            raise ToolPolicyError("tool capability belongs to a different session")
        if tool not in grant.allowed_tools:
            raise ToolPolicyError("tool is not permitted by this capability")
        if self._calls.get(grant.capability_id, 0) >= grant.max_calls:
            raise ToolPolicyError("tool capability call budget is exhausted")
        return grant

    def _purge_expired(self, now: float) -> None:
        expired = [key for key, grant in self._grants.items() if grant.expires_at <= now]
        for key in expired:
            self._grants.pop(key, None)
            self._calls.pop(key, None)
