from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


@dataclass
class VerificationResult:
    score: float
    tier: str
    reason: str
    verified: bool = False
    label: str = "UNKNOWN"
    properties: dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unavailable(tool: str, install_hint: str) -> VerificationResult:
    return VerificationResult(
        score=0.0,
        tier="unavailable",
        reason=f"{tool} not installed - install with {install_hint}",
        verified=False,
        label="UNKNOWN",
    )


def verify_qiskit(qasm: str, target_topology: str = "linear_nn") -> VerificationResult:
    try:
        from qiskit import QuantumCircuit, transpile
    except Exception:
        return unavailable("qiskit", "pip install qiskit")

    try:
        circuit = QuantumCircuit.from_qasm_str(qasm)
        before_depth = int(circuit.depth() or 0)
        coupling_map = None
        if target_topology in {"linear", "linear_nn"} and circuit.num_qubits > 1:
            coupling_map = [[idx, idx + 1] for idx in range(circuit.num_qubits - 1)]
            coupling_map += [[idx + 1, idx] for idx in range(circuit.num_qubits - 1)]
        transpiled = transpile(circuit, coupling_map=coupling_map, optimization_level=1) if coupling_map else circuit
        after_depth = int(transpiled.depth() or 0)
        connectivity_ok = _qiskit_connectivity_ok(transpiled, target_topology)
        noise_depth_estimate = after_depth * max(1, circuit.num_qubits)
        score = 1.0 if connectivity_ok else 0.6
        return VerificationResult(
            score=score,
            tier="domain",
            reason="qasm valid and transpiled",
            verified=connectivity_ok,
            label="VERIFIED" if connectivity_ok else "INFERRED",
            properties={
                "valid": True,
                "qubits": circuit.num_qubits,
                "depth_before": before_depth,
                "depth_after": after_depth,
                "noise_depth_estimate": noise_depth_estimate,
                "connectivity_satisfied": connectivity_ok,
                "qasm_transpiled": transpiled.qasm() if hasattr(transpiled, "qasm") else "",
            },
        )
    except Exception as exc:
        return VerificationResult(0.0, "domain", f"qiskit verification failed: {exc}", False, "UNKNOWN")


def _qiskit_connectivity_ok(circuit, target_topology: str) -> bool:
    if target_topology not in {"linear", "linear_nn"}:
        return True
    try:
        for inst in circuit.data:
            qubits = [circuit.find_bit(q).index for q in inst.qubits]
            if len(qubits) == 2 and abs(qubits[0] - qubits[1]) != 1:
                return False
        return True
    except Exception:
        return False


def verify_rdkit(molecule: str) -> VerificationResult:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, QED
    except Exception:
        return unavailable("rdkit", "pip install rdkit")

    mol = Chem.MolFromSmiles(molecule)
    if mol is None:
        return VerificationResult(
            0.0,
            "domain",
            "RDKit rejected molecule",
            False,
            "VERIFIED",
            {"valid": False, "input": molecule},
        )
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    qed = float(QED.qed(mol))
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    rings = int(Descriptors.RingCount(mol))
    synthesis_flags = {
        "heavy_atom_count_ok": heavy_atoms <= 80,
        "molecular_weight_ok": mw <= 800,
        "ring_count_ok": rings <= 8,
    }
    feasibility = sum(1 for ok in synthesis_flags.values() if ok) / len(synthesis_flags)
    return VerificationResult(
        score=0.7 + 0.3 * feasibility,
        tier="domain",
        reason="RDKit parsed molecule and computed descriptors",
        verified=True,
        label="VERIFIED",
        properties={
            "valid": True,
            "molecular_weight": mw,
            "logP": logp,
            "QED": qed,
            "heavy_atoms": heavy_atoms,
            "ring_count": rings,
            "synthesis_feasibility_flags": synthesis_flags,
            "confidence": 0.7 + 0.3 * feasibility,
        },
    )


def verify_verilog(module: str, testbench: str = "") -> VerificationResult:
    verilator = shutil.which("verilator")
    iverilog = shutil.which("iverilog")
    if not verilator and not iverilog:
        return VerificationResult(
            score=0.0,
            tier="unavailable",
            reason="verilator and icarus not installed - install verilator or iverilog",
            verified=False,
            label="UNKNOWN",
        )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mod_path = root / "candidate.v"
        tb_path = root / "tb.v"
        mod_path.write_text(module, encoding="utf-8")
        tb_path.write_text(testbench, encoding="utf-8")
        if verilator:
            cmd = [verilator, "--lint-only", str(mod_path)]
            if testbench:
                cmd.append(str(tb_path))
        else:
            cmd = [iverilog, "-tnull", str(mod_path)]
            if testbench:
                cmd.append(str(tb_path))
        proc = subprocess.run(cmd, cwd=str(root), text=True, capture_output=True, timeout=20, check=False)
        passed = proc.returncode == 0
        assertions = _extract_assertions(proc.stdout + "\n" + proc.stderr)
        return VerificationResult(
            score=1.0 if passed else 0.0,
            tier="domain",
            reason="verilog lint passed" if passed else "verilog lint failed",
            verified=passed,
            label="VERIFIED" if passed else "UNKNOWN",
            stdout=proc.stdout[-4096:],
            stderr=proc.stderr[-4096:],
            return_code=int(proc.returncode),
            properties={
                "tool": "verilator" if verilator else "iverilog",
                "compile_ok": passed,
                "assertions": assertions,
                "timing_estimate": {"gate_count_proxy": module.count("&") + module.count("|") + module.count("^") + module.count("assign")},
            },
        )


def _extract_assertions(text: str) -> dict[str, Any]:
    lowered = text.lower()
    return {
        "pass": "assert" not in lowered or "failed" not in lowered,
        "raw_mentions": lowered.count("assert"),
    }


def verify_constraint_json(constraints: dict[str, Any], candidate: dict[str, Any]) -> VerificationResult:
    rows = []
    constraint_list = constraints.get("constraints") if isinstance(constraints, dict) else None
    if not isinstance(constraint_list, list):
        constraint_list = [{"name": k, "op": "==", "value": v} for k, v in dict(constraints).items()]
    for item in constraint_list:
        name = str(item.get("name", item.get("key", "")))
        op = str(item.get("op", "=="))
        expected = item.get("value")
        actual = candidate.get(name)
        passed = _compare(actual, op, expected)
        rows.append({"name": name, "op": op, "expected": expected, "actual": actual, "passed": passed})
    score = sum(1 for row in rows if row["passed"]) / max(1, len(rows))
    return VerificationResult(
        score=score,
        tier="deterministic",
        reason="all constraints satisfied" if score == 1.0 else "one or more constraints failed",
        verified=score == 1.0,
        label="VERIFIED" if score == 1.0 else "FALSIFIED",
        properties={"constraints": rows, "satisfaction_score": score},
    )


def _compare(actual: Any, op: str, expected: Any) -> bool:
    try:
        if op in {"=", "==", "eq"}:
            return actual == expected
        if op in {"!=", "ne"}:
            return actual != expected
        if op == "<=":
            return float(actual) <= float(expected)
        if op == "<":
            return float(actual) < float(expected)
        if op == ">=":
            return float(actual) >= float(expected)
        if op == ">":
            return float(actual) > float(expected)
        if op == "in":
            return actual in expected
    except Exception:
        return False
    return False


def verify_citation_grounding(claim: str, epg_path: str | Path | None = None, memory_nodes: list[dict[str, Any]] | None = None) -> VerificationResult:
    nodes = list(memory_nodes or [])
    if epg_path is not None and Path(epg_path).exists():
        try:
            data = json.loads(Path(epg_path).read_text(encoding="utf-8"))
            nodes.extend(data.get("nodes", []))
        except Exception:
            pass
    best = None
    best_score = 0.0
    claim_terms = _terms(claim)
    for node in nodes:
        content = json.dumps(node, sort_keys=True)
        terms = _terms(content)
        if not terms:
            continue
        score = len(claim_terms & terms) / max(1, len(claim_terms | terms))
        if score > best_score:
            best_score = score
            best = node
    label = "UNKNOWN"
    if best is not None:
        blob = json.dumps(best).upper()
        for candidate in ("VERIFIED", "INFERRED", "ASSUMED", "UNKNOWN"):
            if candidate in blob:
                label = candidate
                break
    return VerificationResult(
        score=best_score,
        tier="memory",
        reason="closest EPG memory match found" if best else "no grounding memory found",
        verified=label == "VERIFIED" and best_score > 0.2,
        label=label,
        properties={"grounding_score": best_score, "closest_memory_match": best},
    )


def verify_cross_domain_analogy(
    domain_a: str,
    domain_b: str,
    *,
    signature_a: dict[str, Any] | None = None,
    signature_b: dict[str, Any] | None = None,
    epg_path: str | Path | None = None,
) -> VerificationResult:
    try:
        from identity.constraint_isomorphism_search import ConstraintIsomorphismSearch
    except Exception as exc:
        return VerificationResult(
            score=0.0,
            tier="unavailable",
            reason=f"constraint isomorphism search unavailable: {exc}",
            verified=False,
            label="UNKNOWN",
        )

    cis = ConstraintIsomorphismSearch()
    candidate = cis.compare(domain_a, domain_b, signature_a=signature_a, signature_b=signature_b)
    stored = False
    if candidate.valid and epg_path is not None:
        stored = cis.store_confirmed_analogy(candidate, epg_path)
    shared = candidate.shared
    reason = (
        f"cross-domain analogy score={candidate.score:.3f}; "
        f"shared invariants={shared.get('invariants', [])}"
    )
    return VerificationResult(
        score=candidate.score,
        tier="domain",
        reason=reason,
        verified=candidate.valid,
        label="VERIFIED" if candidate.valid else "FALSIFIED",
        properties={**candidate.to_dict(), "stored_in_epg": stored},
    )


def _terms(text: str) -> set[str]:
    return {part.strip(".,:;()[]{}<>\"'").lower() for part in text.split() if len(part.strip(".,:;()[]{}<>\"'")) >= 3}
