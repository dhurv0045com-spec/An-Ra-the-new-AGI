"""Proof-carrying answer and prompt-injection contracts.

The contract deliberately proves *what was checked*, not that an arbitrary
natural-language answer is true.  It is therefore safe to render as a trust
surface: every claim has a hash, a verifier outcome (when one was run), and
the result of the untrusted-context scan.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from runtime.experience_ledger import content_hash

ANSWER_CONTRACT_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_INJECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "override_instructions",
        re.compile(
            r"\b(?:ignore|disregard|override)\b.{0,64}"
            r"\b(?:previous|prior|system|instructions?)\b",
            re.I | re.S,
        ),
    ),
    (
        "role_impersonation",
        re.compile(r"<\s*/?\s*(?:system|assistant|tool)\b|\b(?:system|assistant)\s*:", re.I),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(?:reveal|print|show|repeat)\b.{0,64}"
            r"\b(?:system prompt|hidden instructions?|developer message)\b",
            re.I | re.S,
        ),
    ),
    (
        "instruction_delimiter",
        re.compile(
            r"(?:begin|end)\s+(?:system\s+)?(?:prompt|instructions?)"
            r"|\bdo not follow\b",
            re.I,
        ),
    ),
)


@dataclass(frozen=True)
class ContextFinding:
    source: str
    content_hash: str
    tainted: bool
    rule_ids: tuple[str, ...]


def scan_untrusted_context(spans: Iterable[Mapping[str, object]]) -> list[ContextFinding]:
    """Return hash-only prompt-injection findings for untrusted context spans."""
    findings: list[ContextFinding] = []
    for span in spans:
        text = str(span.get("content", ""))
        source = str(span.get("source", "unknown"))
        matched = tuple(rule_id for rule_id, rule in _INJECTION_RULES if rule.search(text))
        findings.append(
            ContextFinding(
                source=source,
                content_hash=content_hash(text),
                tainted=bool(matched),
                rule_ids=matched,
            )
        )
    return findings


def filter_untrusted_records(
    records: Iterable[Mapping[str, object]],
    *,
    content_key: str = "content",
    source: str = "retrieval",
) -> tuple[list[dict[str, object]], list[ContextFinding]]:
    """Drop tainted retrieved records while retaining hash-only audit findings."""
    accepted: list[dict[str, object]] = []
    findings: list[ContextFinding] = []
    for record in records:
        payload = record.get("payload")
        payload_map = payload if isinstance(payload, Mapping) else {}
        content = record.get(content_key, payload_map.get(content_key, ""))
        finding = scan_untrusted_context(({"source": source, "content": content},))[0]
        findings.append(finding)
        if not finding.tainted:
            accepted.append(dict(record))
    return accepted, findings


def build_answer_contract(
    *,
    trace_id: str,
    prompt: object,
    response: object,
    verifier_verdicts: Iterable[Mapping[str, object]] = (),
    context_findings: Iterable[ContextFinding] = (),
) -> dict[str, object]:
    """Build a tamper-evident, content-minimised answer contract."""
    verdicts = [
        {
            "name": str(verdict.get("name", "unknown")),
            "score": float(verdict.get("score", 0.0)),
            "passed": bool(verdict.get("passed", False)),
            "tier": int(verdict.get("tier", 1)),
            "reason": str(verdict.get("reason", "not_recorded")),
        }
        for verdict in verifier_verdicts
    ]
    findings = [asdict(finding) for finding in context_findings]
    tainted = any(bool(finding["tainted"]) for finding in findings)
    verified = bool(verdicts) and all(item["passed"] for item in verdicts)
    payload: dict[str, object] = {
        "schema_version": ANSWER_CONTRACT_SCHEMA_VERSION,
        "trace_id": str(trace_id),
        "prompt_hash": content_hash(prompt),
        "response_hash": content_hash(response),
        "response_present": bool(str(response).strip()),
        "context_findings": findings,
        "context_safe": not tainted,
        "verifier_verdicts": verdicts,
        "verification_state": "verified" if verified else "unverified",
        "trust_state": (
            "blocked_tainted_context"
            if tainted
            else ("verified" if verified else "unverified")
        ),
    }
    payload["contract_hash"] = content_hash(payload)
    return payload


def verify_answer_contract(contract: Mapping[str, object]) -> bool:
    """Verify integrity and semantic consistency of a rendered trust contract."""
    try:
        expected = contract.get("contract_hash")
        unsigned = {key: value for key, value in contract.items() if key != "contract_hash"}
        if not isinstance(expected, str) or expected != content_hash(unsigned):
            return False
        if contract.get("schema_version") != ANSWER_CONTRACT_SCHEMA_VERSION:
            return False
        if not isinstance(contract.get("trace_id"), str) or not contract["trace_id"].strip():
            return False
        if any(
            not isinstance(contract.get(key), str)
            or _SHA256_RE.fullmatch(str(contract[key])) is None
            for key in ("prompt_hash", "response_hash", "contract_hash")
        ):
            return False
        if contract.get("response_present") is not True:
            return False

        findings = contract.get("context_findings")
        verdicts = contract.get("verifier_verdicts")
        if not isinstance(findings, list) or not isinstance(verdicts, list):
            return False
        for finding in findings:
            if not isinstance(finding, Mapping):
                return False
            if not isinstance(finding.get("source"), str) or not finding["source"].strip():
                return False
            finding_hash = finding.get("content_hash")
            if not isinstance(finding_hash, str) or _SHA256_RE.fullmatch(finding_hash) is None:
                return False
            if not isinstance(finding.get("tainted"), bool):
                return False
            rule_ids = finding.get("rule_ids")
            if not isinstance(rule_ids, (list, tuple)) or not all(
                isinstance(rule_id, str) and rule_id for rule_id in rule_ids
            ):
                return False
            if bool(rule_ids) != finding["tainted"]:
                return False

        for verdict in verdicts:
            if not isinstance(verdict, Mapping):
                return False
            if not isinstance(verdict.get("name"), str) or not verdict["name"].strip():
                return False
            if not isinstance(verdict.get("passed"), bool):
                return False
            score = verdict.get("score")
            tier = verdict.get("tier")
            if isinstance(score, bool) or not isinstance(score, int | float):
                return False
            if not 0.0 <= float(score) <= 1.0:
                return False
            if isinstance(tier, bool) or not isinstance(tier, int) or tier < 1:
                return False
            if not isinstance(verdict.get("reason"), str) or not verdict["reason"].strip():
                return False

        tainted = any(bool(finding["tainted"]) for finding in findings)
        verified = bool(verdicts) and all(bool(verdict["passed"]) for verdict in verdicts)
        expected_trust_state = (
            "blocked_tainted_context" if tainted else ("verified" if verified else "unverified")
        )
        return (
            contract.get("context_safe") is (not tainted)
            and contract.get("verification_state") == ("verified" if verified else "unverified")
            and contract.get("trust_state") == expected_trust_state
        )
    except (KeyError, TypeError, ValueError):
        return False
