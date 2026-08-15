"""Built-in verifiers registered independently behind the shared protocol."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verification.registry import VerifierRequest, register_verifier


@dataclass(frozen=True)
class BuiltinVerificationResult:
    score: float
    tier: int
    reason: str
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


def _hierarchy(request: VerifierRequest) -> Any:
    if request.context is not None:
        return request.context
    from training.verifier import VerifierHierarchy

    return VerifierHierarchy()


def _domain_result(result: object) -> BuiltinVerificationResult:
    return BuiltinVerificationResult(
        score=float(getattr(result, "score", 0.0)),
        tier=1,
        reason=str(getattr(result, "reason", "domain_verifier_no_reason")),
        stdout=str(getattr(result, "stdout", "")),
        stderr=str(getattr(result, "stderr", "")),
        return_code=int(getattr(result, "return_code", 0)),
    )


@register_verifier("code")
def verify_code(request: VerifierRequest) -> BuiltinVerificationResult:
    return _hierarchy(request).verify_code(
        str(request.payload.get("code", "")),
        str(request.payload.get("test_code", "")),
    )


@register_verifier("math")
def verify_math(request: VerifierRequest) -> BuiltinVerificationResult:
    return _hierarchy(request).verify_math(
        str(request.payload.get("expression", "")),
        str(request.payload.get("expected", "")),
    )


def _verify_exact(request: VerifierRequest) -> BuiltinVerificationResult:
    return _hierarchy(request).verify_exact(
        str(request.payload.get("response", request.payload.get("expression", ""))),
        str(request.payload.get("expected", "")),
        request.name,
    )


for _exact_name in ("logic", "date", "measurement"):
    register_verifier(_exact_name)(_verify_exact)


@register_verifier("file_state")
def verify_file_state(request: VerifierRequest) -> BuiltinVerificationResult:
    check_fn = request.payload.get("check_fn")
    return _hierarchy(request).verify_file_state(check_fn if callable(check_fn) else lambda: False)


@register_verifier("instruction")
def verify_instruction(request: VerifierRequest) -> BuiltinVerificationResult:
    return _hierarchy(request).verify_instruction(
        str(request.payload.get("response", "")),
        str(request.payload.get("pattern", ".*")),
    )


@register_verifier("open_ended", aliases=("open", "general"))
def verify_open_ended(request: VerifierRequest) -> BuiltinVerificationResult:
    return _hierarchy(request).verify_open_ended(
        str(request.payload.get("task", "")),
        str(request.payload.get("response", "")),
    )


@register_verifier("symbolic_output")
def verify_symbolic_output(request: VerifierRequest) -> BuiltinVerificationResult:
    expected = re.sub(r"\s+", "", str(request.payload.get("expected", "")).lower())
    response = re.sub(r"\s+", "", str(request.payload.get("response", "")).lower())
    matched = bool(expected) and expected in response
    return BuiltinVerificationResult(
        1.0 if matched else 0.0,
        1,
        "symbolic_output_matched" if matched else "symbolic_output_mismatch",
    )


@register_verifier("dfc_format")
def verify_dfc_format(request: VerifierRequest) -> BuiltinVerificationResult:
    tags = tuple(
        request.payload.get(
            "tags",
            (
                "[GOAL]",
                "[CONSTRAINT]",
                "[HYPOTHESIS]",
                "[ACTION]",
                "[RESULT]",
                "[VERIFY]",
                "[UPDATE]",
            ),
        )
    )
    text = str(request.payload.get("text", ""))
    positions = [text.find(str(tag)) for tag in tags]
    valid = all(position >= 0 for position in positions) and positions == sorted(positions)
    return BuiltinVerificationResult(
        1.0 if valid else 0.0,
        1,
        "dfc_format_valid" if valid else "dfc_format_invalid",
    )


@register_verifier("gepa_candidate")
def verify_gepa_candidate(request: VerifierRequest) -> BuiltinVerificationResult:
    evidence = request.payload.get("evidence_trace_ids", ())
    predicted = request.payload.get("predicted_delta", {})
    proposed_text = str(request.payload.get("proposed_text", "")).strip()
    owner_gate = request.payload.get("owner_approval_required") is True
    evidence_ok = isinstance(evidence, (list, tuple)) and bool(evidence)
    prediction_ok = isinstance(predicted, Mapping) and "eval_success" in predicted
    valid = evidence_ok and prediction_ok and bool(proposed_text) and owner_gate
    return BuiltinVerificationResult(
        1.0 if valid else 0.0,
        2,
        "gepa_candidate_evidence_complete" if valid else "gepa_candidate_evidence_incomplete",
    )


def _load_domain(name: str) -> Any:
    from phase3.symbolic_bridge_45q import domain_verifiers

    return getattr(domain_verifiers, name)


@register_verifier("qiskit")
def verify_qiskit(request: VerifierRequest) -> BuiltinVerificationResult:
    return _domain_result(
        _load_domain("verify_qiskit")(
            request.payload.get("qasm", request.payload.get("code", "")),
            request.payload.get("target_topology", "linear_nn"),
        )
    )


@register_verifier("rdkit")
def verify_rdkit(request: VerifierRequest) -> BuiltinVerificationResult:
    return _domain_result(
        _load_domain("verify_rdkit")(
            request.payload.get(
                "smiles",
                request.payload.get("molecule", request.payload.get("response", "")),
            )
        )
    )


@register_verifier("verilog", aliases=("verilator",))
def verify_verilog(request: VerifierRequest) -> BuiltinVerificationResult:
    return _domain_result(
        _load_domain("verify_verilog")(
            request.payload.get("verilog", request.payload.get("code", "")),
            request.payload.get("testbench", request.payload.get("test_code", "")),
        )
    )


@register_verifier("constraint_json")
def verify_constraint_json(request: VerifierRequest) -> BuiltinVerificationResult:
    return _domain_result(
        _load_domain("verify_constraint_json")(
            request.payload.get("constraints", {}), request.payload.get("candidate", {})
        )
    )


@register_verifier("citation_grounding")
def verify_citation_grounding(request: VerifierRequest) -> BuiltinVerificationResult:
    memory_nodes = request.payload.get("memory_nodes")
    retriever = request.payload.get("retriever")
    if retriever is not None and hasattr(retriever, "search"):
        from retrieval import RetrievalQuery

        hits = retriever.search(
            RetrievalQuery(
                text=str(request.payload.get("claim", request.payload.get("response", ""))),
                limit=int(request.payload.get("retrieval_limit", 8)),
            )
        )
        memory_nodes = [
            {
                "id": hit.id,
                "content": hit.text,
                "score": hit.score,
                "metadata": dict(hit.metadata),
            }
            for hit in hits
        ]
    return _domain_result(
        _load_domain("verify_citation_grounding")(
            request.payload.get("claim", request.payload.get("response", "")),
            request.payload.get("epg_path"),
            memory_nodes,
        )
    )


@register_verifier("cross_domain_analogy")
def verify_cross_domain_analogy(request: VerifierRequest) -> BuiltinVerificationResult:
    return _domain_result(
        _load_domain("verify_cross_domain_analogy")(
            request.payload.get("domain_a", ""),
            request.payload.get("domain_b", ""),
            signature_a=request.payload.get("signature_a"),
            signature_b=request.payload.get("signature_b"),
            epg_path=request.payload.get("epg_path"),
        )
    )
