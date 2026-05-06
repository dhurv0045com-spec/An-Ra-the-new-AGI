from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anra_paths import ROOT, STATE_DIR, get_v2_checkpoint, inject_all_paths

inject_all_paths()

from identity.falsification_ledger import FalsificationLedger
from identity.hal import HALModule
from memory.experimental_proof_graph import ExperimentalProofGraph

try:
    from domain_verifiers import verify_constraint_json, verify_qiskit
except Exception:
    verify_constraint_json = verify_qiskit = None


PROMPT = (
    "Design a 4-qubit quantum circuit that measures a biological oscillation signal. "
    "Topology: linear nearest-neighbor. Constraint: depth <= 20. Power budget: 20mW."
)


FALLBACK_OUTPUT = """
<hyp>A 4-qubit linear nearest-neighbor circuit can encode a biological oscillation phase with depth <= 20 and power <= 20mW.</hyp>
<cons>{"constraints":[{"name":"qubits","op":"==","value":4},{"name":"depth","op":"<=","value":20},{"name":"topology","op":"==","value":"linear_nn"},{"name":"power_mw","op":"<=","value":20}]}</cons>
<act>OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
rz(0.25) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];</act>
<obs>{"qubits":4,"depth":6,"topology":"linear_nn","power_mw":18}</obs>
"""


def _model_output() -> tuple[str, str]:
    checkpoint = get_v2_checkpoint("ouroboros")
    if not checkpoint.exists():
        checkpoint = get_v2_checkpoint("identity")
    if not checkpoint.exists():
        checkpoint = get_v2_checkpoint("brain")
    if not checkpoint.exists():
        return FALLBACK_OUTPUT, "no checkpoint found; using scaffolded candidate"
    try:
        from generate import GenerationConfig, generate_traced

        trace = generate_traced(PROMPT, GenerationConfig(max_tokens=240, temperature=0.75), session_id="demo")
        return trace.output or FALLBACK_OUTPUT, f"checkpoint used: {checkpoint}"
    except Exception as exc:
        return FALLBACK_OUTPUT, f"generation failed ({exc}); using scaffolded candidate"


def _extract_qasm(text: str) -> str:
    match = re.search(r"OPENQASM\s+2\.0;.*?(?=</act>|<obs>|<upd>|$)", text, flags=re.I | re.S)
    if match:
        return match.group(0).strip()
    return FALLBACK_OUTPUT.split("<act>", 1)[1].split("</act>", 1)[0].strip()


def _extract_json_between(text: str, tag: str) -> dict[str, Any]:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S)
    if not match:
        return {}
    raw = match.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                return {}
    return {}


def _extract_claims(text: str) -> list[str]:
    claims = re.findall(r"<hyp>(.*?)</hyp>", text, flags=re.S)
    if not claims:
        claims = [PROMPT]
    return [re.sub(r"\s+", " ", claim).strip() for claim in claims if claim.strip()]


def _result_dict(result: Any) -> dict[str, Any]:
    if result is None:
        return {"score": 0.0, "verified": False, "label": "UNKNOWN", "reason": "verifier unavailable"}
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return {
        "score": getattr(result, "score", 0.0),
        "verified": getattr(result, "verified", False),
        "label": getattr(result, "label", "UNKNOWN"),
        "reason": getattr(result, "reason", ""),
        "properties": getattr(result, "properties", {}),
    }


def _candidate_from_obs(obs: dict[str, Any]) -> dict[str, Any]:
    if "result" in obs and isinstance(obs["result"], dict):
        return dict(obs["result"])
    return dict(obs)


def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    epg = ExperimentalProofGraph(STATE_DIR / "demo_epg.json")
    ledger = FalsificationLedger(STATE_DIR / "demo_falsification_ledger.json")
    hal = HALModule()

    model_text, generation_status = _model_output()
    dfc_text = (
        f'<bos><task domain="quantum_biology" type="scientific_investigation">{PROMPT}</task>'
        f"{model_text}<eos>"
    )

    qasm = _extract_qasm(dfc_text)
    constraints = _extract_json_between(dfc_text, "cons")
    candidate = _candidate_from_obs(_extract_json_between(dfc_text, "obs"))

    qiskit_result = _result_dict(verify_qiskit(qasm, "linear_nn") if verify_qiskit is not None else None)
    constraint_result = _result_dict(
        verify_constraint_json(constraints, candidate)
        if verify_constraint_json is not None and constraints
        else None
    )
    results = [qiskit_result, constraint_result]
    mean_score = sum(float(r.get("score", 0.0) or 0.0) for r in results) / max(1, len(results))
    hal.update(
        verifier_result=mean_score,
        session_context={
            "domain": "quantum_biology",
            "task_type": "scientific_investigation",
            "conflicting_constraints_detected": any(not r.get("verified") for r in results),
            "near_capability_boundary": mean_score < 0.5,
        },
    )

    nodes = epg.record_experiment(
        hypothesis={"prompt": PROMPT, "claims": _extract_claims(dfc_text)},
        action={"qasm": qasm, "constraints": constraints, "candidate": candidate},
        observation={"qiskit": qiskit_result, "constraints": constraint_result, "passed": all(r.get("verified") for r in results)},
        correction={
            "repair": "If verification fails, repair topology/depth/power claims first, then rerun qiskit and constraint_json.",
            "hal_state": hal.state.hormones(),
        },
        memory={"dfc_trace": dfc_text[:4000], "generation_status": generation_status},
    )

    for claim in _extract_claims(dfc_text):
        status = "VERIFIED" if all(r.get("verified") for r in results) else "UNKNOWN"
        if any(str(r.get("label", "")).upper() == "FALSIFIED" for r in results):
            status = "FALSIFIED"
        ledger.append(
            claim,
            status=status,
            confidence=max(0.0, min(1.0, mean_score)),
            evidence=[{"qiskit": qiskit_result}, {"constraints": constraint_result}],
            would_be_false_if="Qiskit rejects the circuit or constraint_json reports an unsatisfied topology, depth, qubit, or power constraint.",
            next_verifier="verify_qiskit + verify_constraint_json",
        )

    print("AN-RA DFC Scientific Investigation Demo")
    print(f"Generation: {generation_status}")
    print(f"EPG nodes: {', '.join(f'{k}={v.node_id}' for k, v in nodes.items())}")
    print(f"Ledger entries: {len(ledger.records)}")
    print(f"Qiskit: {qiskit_result.get('label')} score={qiskit_result.get('score')} reason={qiskit_result.get('reason')}")
    print(f"Constraints: {constraint_result.get('label')} score={constraint_result.get('score')} reason={constraint_result.get('reason')}")
    print(f"HAL: {json.dumps(hal.state.hormones(), sort_keys=True)}")
    print(f"Trace: {STATE_DIR / 'demo_epg.json'}")
    print(f"Ledger: {STATE_DIR / 'demo_falsification_ledger.json'}")


if __name__ == "__main__":
    main()
