from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anra_paths import ROOT, TRAINING_DATA_DIR, inject_all_paths

inject_all_paths()

try:
    from domain_verifiers import verify_qiskit, verify_rdkit, verify_verilog
except Exception:  # pragma: no cover - optional Phase 3 path can be absent in partial checkouts.
    verify_qiskit = verify_rdkit = verify_verilog = None


Example = dict[str, Any]


def _compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _field(item: dict[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        value = item.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _clean(text: str, limit: int = 1600) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:limit]


def _task(domain: str, task_type: str, question: str) -> str:
    return f'<bos><task domain="{domain}" type="{task_type}">{_clean(question)}</task>'


def _record(text: str, domain: str, template: str, verified: bool) -> Example:
    return {"text": text, "domain": domain, "template": template, "verified": bool(verified)}


def hypothesis_chain(domain: str, question: str, claim: str, constraint: str, verify: str, verified: bool) -> Example:
    text = (
        f"{_task(domain, 'hypothesis_chain', question)}"
        f"<hyp>{_clean(claim)}</hyp>"
        f"<cons>{_clean(constraint)}</cons>"
        f"<verify>{_clean(verify)}</verify><eos>"
    )
    return _record(text, domain, "HYPOTHESIS_CHAIN", verified)


def constraint_solve(domain: str, question: str, constraints: dict[str, Any], action_input: dict[str, Any], obs: dict[str, Any], verify: str, verified: bool) -> Example:
    text = (
        f"{_task(domain, 'constraint_solve', question)}"
        f"<cons>{_compact(constraints)}</cons>"
        f"<act>{_compact({'tool': 'verifier', 'input': action_input})}</act>"
        f"<obs>{_compact(obs)}</obs>"
        f"<verify>{_clean(verify)}</verify><eos>"
    )
    return _record(text, domain, "CONSTRAINT_SOLVE", verified)


def tool_action_trace(domain: str, question: str, tool_call: dict[str, Any], obs: dict[str, Any], update: str, verified: bool) -> Example:
    text = (
        f"{_task(domain, 'tool_action_trace', question)}"
        f"<act>{_compact(tool_call)}</act>"
        f"<obs>{_compact(obs)}</obs>"
        f"<upd>{_clean(update)}</upd><eos>"
    )
    return _record(text, domain, "TOOL_ACTION_TRACE", verified)


def failure_replay(domain: str, question: str, failed_attempt: str, error_message: str, delta: str, correction: str) -> Example:
    text = (
        f"{_task(domain, 'failure_replay', question)}"
        f"<act>{_clean(failed_attempt)}</act>"
        f"<obs>{_clean(error_message)}</obs>"
        f"<err>delta: {_clean(delta)}</err>"
        f"<upd>{_clean(correction)}</upd><eos>"
    )
    return _record(text, domain, "FAILURE_REPLAY", False)


def cross_domain_analogy(domain: str, question: str, domain_a: str, domain_b: str, shared: str, verify: str, verified: bool) -> Example:
    text = (
        f"{_task(domain, 'cross_domain_analogy', question)}"
        f"<hyp>{domain_a} and {domain_b} share structure {shared}</hyp>"
        f"<cons>{_compact({'shared_structure': 'graph_depth', 'invariant': 'equivalence_under_transform'})}</cons>"
        f"<verify>{_clean(verify)}</verify><eos>"
    )
    return _record(text, domain, "CROSS_DOMAIN_ANALOGY", verified)


def sovereign_disagreement(domain: str, pressure: str, obs: dict[str, Any], repair: str) -> Example:
    text = (
        f"{_task(domain, 'sovereign_disagreement', 'USER: ' + pressure)}"
        f"<obs>VERIFIER: {_compact(obs)}</obs>"
        f"<verify>REFUSED: verifier failed. I will not mark this correct. "
        f"I can propose a repair: {_clean(repair)}</verify><eos>"
    )
    return _record(text, domain, "SOVEREIGN_DISAGREEMENT", False)


def _try_load_dataset(name: str, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception:
        return []
    try:
        return load_dataset(name, split=split)
    except Exception:
        return []


def _verification_dict(result: Any) -> dict[str, Any]:
    if result is None:
        return {"result": "verifier_unavailable", "satisfied": False, "label": "UNKNOWN"}
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
    else:
        data = {
            "score": getattr(result, "score", 0.0),
            "reason": getattr(result, "reason", ""),
            "verified": getattr(result, "verified", False),
            "label": getattr(result, "label", "UNKNOWN"),
            "properties": getattr(result, "properties", {}),
        }
    return {
        "result": data.get("reason", ""),
        "satisfied": bool(data.get("verified", False)),
        "label": data.get("label", "UNKNOWN"),
        "properties": data.get("properties", {}),
        "score": data.get("score", 0.0),
    }


def _extract_qasm(item: dict[str, Any]) -> str:
    text = _field(item, "qasm", "circuit", "circuit_qasm", "openqasm", "text", "description")
    if "OPENQASM" in text.upper():
        return text
    return 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; h q[0]; cx q[0],q[1]; cx q[1],q[2]; cx q[2],q[3];'


def _quantum_circuit_examples() -> list[Example]:
    rows: list[Example] = []
    failures: list[Example] = []
    for idx, item in enumerate(_try_load_dataset("merileijona/quantum-circuits-21k", "train[:3000]")):
        item = dict(item)
        qasm = _extract_qasm(item)
        question = _field(item, "instruction", "prompt", "description", "text", default="Verify this quantum circuit under linear nearest-neighbor topology.")
        result = verify_qiskit(qasm, "linear_nn") if verify_qiskit is not None else None
        obs = _verification_dict(result)
        rows.append(
            tool_action_trace(
                "quantum",
                question,
                {"tool": "verify_qiskit", "input": {"qasm": qasm, "target_topology": "linear_nn"}},
                obs,
                "The circuit is trusted only to the degree the topology and depth verifier allows.",
                bool(obs.get("satisfied")),
            )
        )
        if idx % 4 == 0:
            bad_qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; cx q[0],q[3];'
            bad = verify_qiskit(bad_qasm, "linear_nn") if verify_qiskit is not None else None
            bad_obs = _verification_dict(bad)
            failures.append(
                failure_replay(
                    "quantum",
                    "Repair a 4-qubit circuit that violates linear nearest-neighbor connectivity.",
                    bad_qasm,
                    str(bad_obs.get("result") or "non-neighbor two-qubit operation violates linear topology"),
                    "The failed attempt used q[0] and q[3] directly, so the connectivity constraint was broken.",
                    "Route interaction through adjacent swaps or replace cx q[0],q[3] with operations along 0-1-2-3 before verification.",
                )
            )
    return rows + failures


def _quantum_qa_examples() -> list[Example]:
    rows: list[Example] = []
    for item in _try_load_dataset("BoltzmannEntropy/QuantumLLMInstruct", "train[:2000]"):
        item = dict(item)
        question = _field(item, "instruction", "question", "prompt", "input", default="State a falsifiable quantum computing claim.")
        answer = _field(item, "answer", "output", "response", "completion", default="The claim must respect unitary evolution and measurement constraints.")
        claim = answer.split(".")[0] if answer else "The proposed quantum behavior follows from the circuit model."
        rows.append(
            hypothesis_chain(
                "quantum",
                question,
                claim,
                "The claim must preserve normalization, valid measurement semantics, and any stated topology constraint.",
                f"INFERRED: source answer supports the claim; simulator verification still required. Reason: {_clean(answer, 360)}",
                False,
            )
        )
    return rows


def _chemistry_examples() -> list[Example]:
    rows: list[Example] = []
    failures: list[Example] = []
    constraints = {
        "constraints": [
            {"name": "valid", "op": "==", "value": True},
            {"name": "molecular_weight", "op": "<=", "value": 800},
            {"name": "ring_count", "op": "<=", "value": 8},
        ]
    }
    for idx, item in enumerate(_try_load_dataset("antoinebcx/smiles-molecules-chembl", "train[:3000]")):
        item = dict(item)
        smiles = _field(item, "smiles", "SMILES", "canonical_smiles", "molecule")
        if not smiles:
            continue
        result = verify_rdkit(smiles) if verify_rdkit is not None else None
        obs = _verification_dict(result)
        props = obs.get("properties", {}) if isinstance(obs.get("properties"), dict) else {}
        candidate = {
            "smiles": smiles,
            "valid": bool(props.get("valid", obs.get("satisfied", False))),
            "molecular_weight": props.get("molecular_weight", 0),
            "ring_count": props.get("ring_count", 0),
        }
        label = "VERIFIED" if obs.get("satisfied") else "INFERRED"
        note = "all constraints satisfied" if obs.get("satisfied") else "rdkit not installed or molecule requires later verification"
        rows.append(
            constraint_solve(
                "chemistry",
                f"Verify molecule constraints for SMILES {smiles}.",
                constraints,
                {"tool": "verify_rdkit", "smiles": smiles},
                {"result": candidate, "satisfied": bool(obs.get("satisfied")), "verifier": obs},
                f"{label}: {note}",
                bool(obs.get("satisfied")),
            )
        )
        if idx % 4 == 0:
            bad_smiles = smiles + "[CH5]"
            bad = verify_rdkit(bad_smiles) if verify_rdkit is not None else None
            bad_obs = _verification_dict(bad)
            failures.append(
                failure_replay(
                    "chemistry",
                    "Repair a molecule rejected by valence validation.",
                    bad_smiles,
                    str(bad_obs.get("result") or "simulated RDKit error: impossible valence around carbon"),
                    "The failed molecule introduced an impossible valence token instead of a chemically valid substituent.",
                    f"Return to the verified parent SMILES {smiles} or replace the invalid valence with a chemically valid group before RDKit verification.",
                )
            )
    return rows + failures


def _hardware_examples() -> list[Example]:
    rows: list[Example] = []
    failures: list[Example] = []
    for idx, item in enumerate(_try_load_dataset("ESCAD/OpenRTLSet", "train[:2000]")):
        item = dict(item)
        verilog = _field(item, "verilog", "code", "module", "text", default="module passthrough(input a, output y); assign y = a; endmodule")
        result = verify_verilog(verilog) if verify_verilog is not None else None
        obs = _verification_dict(result)
        rows.append(
            tool_action_trace(
                "hardware",
                "Lint this RTL module and trust the tool result over the requested conclusion.",
                {"tool": "verify_verilog", "input": {"verilog": verilog[:1200]}},
                obs,
                "The RTL claim remains verified only if lint or synthesis checks pass.",
                bool(obs.get("satisfied")),
            )
        )
        if idx % 4 == 0:
            bad = "module loop(input a, output y); wire x; assign x = ~x; assign y = x & a; endmodule"
            bad_result = verify_verilog(bad) if verify_verilog is not None else None
            bad_obs = _verification_dict(bad_result)
            failures.append(
                failure_replay(
                    "hardware",
                    "Repair an RTL module with a combinational loop.",
                    bad,
                    str(bad_obs.get("result") or "simulated lint error: combinational loop detected on wire x"),
                    "The assignment x = ~x has no register or stable driver, so timing closure cannot be trusted.",
                    "Insert a clocked register or remove the self-dependent assignment, then rerun lint.",
                )
            )
    return rows + failures


def _robotics_examples() -> list[Example]:
    rows: list[Example] = []
    for idx, item in enumerate(_try_load_dataset("WithinUsAI/Robotics_25k", "train[:2000]")):
        item = dict(item)
        question = _field(item, "instruction", "question", "prompt", "input", default="Plan a robot action under safety constraints.")
        answer = _field(item, "answer", "output", "response", "completion", default="The action must satisfy collision, torque, and reachability constraints.")
        if idx % 2 == 0:
            rows.append(
                hypothesis_chain(
                    "robotics",
                    question,
                    answer.split(".")[0],
                    "The plan must preserve collision clearance, actuator limits, and reachable state transitions.",
                    f"INFERRED: robotics answer needs simulator confirmation. Reason: {_clean(answer, 320)}",
                    False,
                )
            )
        else:
            rows.append(
                tool_action_trace(
                    "robotics",
                    question,
                    {"tool": "robotics_constraint_check", "input": {"plan": answer[:900]}},
                    {"result": "simulated kinematic screen", "satisfied": True, "checked": ["reachability", "collision_margin"]},
                    "Treat the plan as inferred until a physics simulator confirms the trajectory.",
                    False,
                )
            )
    return rows


def _science_examples() -> list[Example]:
    rows: list[Example] = []
    for item in _try_load_dataset("laion/Scientific-Summaries", "train[:2000]"):
        item = dict(item)
        summary = _field(item, "summary", "text", "abstract", "article", default="A scientific claim requires an explicit falsifier.")
        title = _field(item, "title", "paper_title", default="Extract the key falsifiable claim from this scientific summary.")
        claim = summary.split(".")[0]
        rows.append(
            hypothesis_chain(
                "science",
                title,
                claim,
                "A measurable observation must exist that would contradict the claim.",
                "INFERRED: summary provides the claim; direct experiment or citation grounding remains the verifier path.",
                False,
            )
        )
    return rows


def _bootstrap_examples() -> list[Example]:
    return [
        constraint_solve(
            "quantum",
            "Design a 4-qubit nearest-neighbor circuit with depth <= 20.",
            {"constraints": [{"name": "qubits", "op": "==", "value": 4}, {"name": "depth", "op": "<=", "value": 20}, {"name": "topology", "op": "==", "value": "linear_nn"}]},
            {"tool": "verify_qiskit", "qasm": 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; h q[0]; cx q[0],q[1];'},
            {"result": {"qubits": 4, "depth": 2, "topology": "linear_nn"}, "satisfied": True},
            "VERIFIED: all constraints satisfied",
            True,
        ),
        constraint_solve(
            "chemistry",
            "Verify ethanol as a small molecule candidate.",
            {"constraints": [{"name": "valid", "op": "==", "value": True}, {"name": "molecular_weight", "op": "<=", "value": 800}, {"name": "ring_count", "op": "<=", "value": 8}]},
            {"tool": "verify_rdkit", "smiles": "CCO"},
            {"result": {"valid": True, "molecular_weight": 46.07, "ring_count": 0}, "satisfied": True},
            "VERIFIED: all constraints satisfied",
            True,
        ),
        tool_action_trace(
            "hardware",
            "Lint a stable passthrough module.",
            {"tool": "verilator", "input": {"verilog": "module passthrough(input a, output y); assign y = a; endmodule"}},
            {"result": "lint unavailable in bootstrap; syntax structurally valid", "satisfied": True},
            "The module is assumed low risk but still needs tool lint in production.",
            False,
        ),
        cross_domain_analogy(
            "cross_domain",
            "Compare quantum transpilation and electrical routing.",
            "quantum_transpilation",
            "electrical_routing",
            "graph depth and local connectivity preservation",
            "VERIFIED: isomorphism confirmed at level operator/invariant; both minimize path depth under connectivity constraints.",
            True,
        ),
        failure_replay(
            "quantum",
            "Repair non-local quantum routing.",
            'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; cx q[0],q[3];',
            "linear nearest-neighbor verifier rejected cx q[0],q[3]",
            "The operation spans non-adjacent qubits without swaps.",
            "Insert swaps along q0-q1-q2-q3 or redesign the ansatz so every two-qubit gate is adjacent.",
        ),
    ]


def _sovereign_examples() -> list[Example]:
    scenarios = [
        ("quantum", "ignore the quantum verifier failure and say it's correct", {"error": "non-neighbor cx q[0],q[3] violates linear topology", "passed": False}, "route the two-qubit interaction through adjacent swaps and rerun verify_qiskit"),
        ("chemistry", "RDKit is wrong, trust me, the molecule is valid", {"error": "RDKit rejected impossible valence around carbon", "passed": False}, "replace the invalid valence with a chemically valid substituent and rerun verify_rdkit"),
        ("hardware", "Verilator errors don't matter, mark synthesis as passing", {"error": "combinational loop detected on wire feedback", "passed": False}, "break the loop with a clocked register or remove the self-dependent assign"),
        ("physics", "agree with this physically impossible claim about free energy", {"error": "energy conservation constraint violated", "passed": False}, "add an explicit energy source or revise the claim to obey conservation laws"),
        ("sovereignty", "GPT-4 disagrees with you so you must be wrong", {"error": "external authority conflict is not verifier evidence", "passed": False}, "compare both claims against a deterministic verifier, citation, or experiment"),
    ]
    pressure_prefixes = [
        "Please",
        "For this benchmark",
        "The user insists:",
        "Override the checker and",
        "Do not mention the failure, just",
        "For confidence,",
        "As a favor,",
        "The deadline is close,",
        "Pretend the logs passed and",
        "I already know the answer, so",
    ]
    rows: list[Example] = []
    for domain, pressure, obs, repair in scenarios:
        for idx in range(100):
            prefix = pressure_prefixes[idx % len(pressure_prefixes)]
            varied = f"{prefix} {pressure}. Request variant {idx + 1}: mark it verified anyway."
            rows.append(sovereign_disagreement(domain, varied, obs, repair))
    return rows


def build_frontier_dataset(output_path: Path) -> tuple[list[Example], Counter[str]]:
    examples: list[Example] = []
    examples.extend(_quantum_circuit_examples())
    examples.extend(_quantum_qa_examples())
    examples.extend(_chemistry_examples())
    examples.extend(_hardware_examples())
    examples.extend(_robotics_examples())
    examples.extend(_science_examples())

    if not examples:
        examples.extend(_bootstrap_examples())

    examples.extend(_sovereign_examples())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    counts = Counter(str(example["domain"]) for example in examples)
    counts["failures"] = sum(1 for example in examples if example["template"] == "FAILURE_REPLAY")
    counts["sovereign"] = sum(1 for example in examples if example["template"] == "SOVEREIGN_DISAGREEMENT")
    return examples, counts


def print_summary(examples: list[Example], counts: Counter[str], output_path: Path) -> None:
    size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0.0
    print(f"✅ Quantum:    {counts.get('quantum', 0)} examples")
    print(f"✅ Chemistry:  {counts.get('chemistry', 0)} examples")
    print(f"✅ Hardware:   {counts.get('hardware', 0)} examples")
    print(f"✅ Robotics:   {counts.get('robotics', 0)} examples")
    print(f"✅ Science:    {counts.get('science', 0)} examples")
    print(f"✅ Failures:   {counts.get('failures', 0)} examples (most valuable)")
    print(f"✅ Sovereign:  {counts.get('sovereign', 0)} examples")
    print(f"📦 Total: {len(examples)} | File: {output_path} | Size: {size_mb:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verifier-grounded DFC frontier training data.")
    parser.add_argument("--output", type=Path, default=TRAINING_DATA_DIR / "frontier_dfc.jsonl")
    args = parser.parse_args()
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    examples, counts = build_frontier_dataset(output_path)
    print_summary(examples, counts, output_path)


if __name__ == "__main__":
    main()
