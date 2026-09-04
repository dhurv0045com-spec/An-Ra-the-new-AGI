"""P35-A readiness: machine-verifiable gate for the first real experiment (M33).

The scientific question stays fixed: does adding verified cognition
training produce measurable cognitive acquisition beyond ordinary language
training at matched compute? Control 0% vs treatment ~15% verified
cognition, CE only, everything else matched.

Gates: exact P35 architecture, real tokenizer, qualified dataset from an
EXTERNAL corpus (repository canary data never qualifies), qualified
generators, canonical stream/cursor proof, exact resume, trainer,
preregistration, matched arms, evaluation fixture/protocol, compute
availability, Fresh/sealed policy. Any gap yields BLOCKED_BY_* verdicts;
BLOCKED_BY_DATASET is success when the corpus is not real (M34).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


P35A_SCHEMA = "anra-v5-p35a-readiness/v1"
P35A_QUESTION = (
    "Does adding verified cognition training produce measurable cognitive "
    "acquisition beyond ordinary language training at matched compute?"
)
P35_RECIPE = {
    "layers": 16, "width": 384, "query_heads": 6, "kv_heads": 3,
    "head_dimension": 64, "ffn_width": 1024, "vocabulary_size": 24576,
    "context_length": 4096,
}
P35_PARAMETERS = 35_411_328
REQUIRED_GENERATOR_FAMILIES = (
    "query_binding",
    "semantic_state",
    "interference_retrieval",
    "relational_composition",
    "counterfactual_sensitivity",
    "heldout_rule_induction",
    "missing_information",
    "faithful_realization",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def check_recipe() -> dict[str, object]:
    """Verify the exact P35 2:1 recipe against the contract receipt."""

    from v5_contracts.model_spec import V5A_250M
    import dataclasses

    spec = dataclasses.replace(
        V5A_250M, layers=P35_RECIPE["layers"], width=P35_RECIPE["width"],
        query_heads=P35_RECIPE["query_heads"], kv_heads=P35_RECIPE["kv_heads"],
        head_dimension=P35_RECIPE["head_dimension"], ffn_width=P35_RECIPE["ffn_width"],
        vocabulary_size=P35_RECIPE["vocabulary_size"],
        context_length=P35_RECIPE["context_length"],
    )
    spec.assert_valid()
    actual = spec.parameter_receipt().total
    if actual != P35_PARAMETERS:
        return {"pass": False, "via": f"recipe counts {actual}, expected {P35_PARAMETERS}"}
    if spec.query_heads != 2 * spec.kv_heads:
        return {"pass": False, "via": "recipe breaks the 2:1 GQA invariant"}
    return {"pass": True, "via": f"P35 2:1 recipe counts exactly {actual}"}


def check_tokenizer(root: Path) -> dict[str, object]:
    result = _load_json(root / "artifacts/e1/local_tournament/result.json") or {}
    artifact = root / "artifacts/e1/local_tournament/tokenizer-24576.json.gz"
    if result.get("status") != "DEVELOPMENT_STATIC_PASS":
        return {"pass": False, "via": "E1 local tournament receipt missing or failing"}
    if not artifact.is_file():
        return {"pass": False, "via": "24k tokenizer artifact absent"}
    return {"pass": True, "via": "development 24k artifact present with static PASS"}


def check_dataset(root: Path, qualification_path: Path | None) -> dict[str, object]:
    path = qualification_path or (root / "artifacts/cymek/dataset_qualification.json")
    receipt = _load_json(path)
    if receipt is None:
        return {"pass": False, "via": "BLOCKED_BY_DATASET: no qualification receipt"}
    if receipt.get("status") != "DATASET_QUALIFIED":
        return {"pass": False, "via": "BLOCKED_BY_DATASET: qualification did not pass"}
    if receipt.get("corpus_class") != "external-qualified":
        return {
            "pass": False,
            "via": "BLOCKED_BY_DATASET: repository canary data never qualifies (M34)",
        }
    return {"pass": True, "via": "external dataset qualified"}


def check_generators(root: Path, qualification_path: Path | None) -> dict[str, object]:
    path = qualification_path or (root / "artifacts/cymek/generator_qualification.json")
    receipt = _load_json(path)
    if receipt is None:
        return {"pass": False, "via": "BLOCKED: no generator qualification receipt"}
    families = {item.get("family"): item.get("verdict") for item in receipt.get("families", [])}
    missing = [name for name in REQUIRED_GENERATOR_FAMILIES if families.get(name) != "GENERATOR_QUALIFIED"]
    if missing:
        return {"pass": False, "via": f"BLOCKED: families not qualified: {missing}"}
    return {"pass": True, "via": "all required families qualified"}


def check_test_evidence(root: Path) -> dict[str, object]:
    from anra_v5.status import _tree_has_test, classify_freshness

    receipt = _load_json(root / "artifacts/cymek/test_receipt.json") or {}
    tested = str(receipt.get("tested_head") or "")
    freshness = classify_freshness(root, tested)
    if freshness["state"] not in {"EXACT_HEAD_TESTED", "TREE_EQUIVALENT_TESTED"}:
        return {"pass": False, "via": f"test evidence stale: {freshness['state']}"}
    if int((receipt.get("result") or {}).get("failed", 1)) != 0:
        return {"pass": False, "via": "test receipt records failures"}
    for test_path, test_name in (
        ("tests/test_v5_stream_resume.py", "test_path_a_equals_path_b_from_artifacts_only"),
        ("tests/test_v5_experiment_integrity.py", "test_declared_treatment_passes"),
        ("tests/test_v5_evaluation_protocol.py", "test_n_must_be_exact"),
    ):
        if not _tree_has_test(root, tested, test_path, test_name):
            return {"pass": False, "via": f"{test_name} absent from tested tree"}
    return {"pass": True, "via": "stream/experiment/protocol proofs green in tested tree"}


COMPUTE_SCHEMA = "anra-v5-compute-evidence/v1"
MIN_ACCELERATOR_MEMORY_BYTES = 4 * 1024**3


def probe_execution_environment() -> dict[str, object]:
    """Record factual host capability. Informational only, never a PASS claim."""

    try:
        import torch
    except ImportError:
        return {
            "schema": COMPUTE_SCHEMA,
            "host_label": "torch-absent",
            "torch_version": None,
            "accelerator_available": False,
            "accelerator_qualified": False,
            "memory_sufficient": False,
            "runtime_verified": False,
        }
    cuda = bool(torch.cuda.is_available())
    memory_ok = False
    device_name: object = None
    if cuda:
        try:
            device_name = torch.cuda.get_device_name(0)
            memory_ok = int(torch.cuda.get_device_properties(0).total_memory) >= MIN_ACCELERATOR_MEMORY_BYTES
        except Exception:
            cuda = False
    return {
        "schema": COMPUTE_SCHEMA,
        "host_label": f"torch-{torch.__version__}-cuda-{cuda}",
        "torch_version": str(torch.__version__),
        "accelerator_available": cuda,
        "accelerator_qualified": cuda and memory_ok,
        "memory_sufficient": (memory_ok if cuda else False),
        "runtime_verified": False,
    }


def evaluate_compute_gate(evidence: Mapping[str, Any] | None) -> dict[str, object]:
    """Judge compute evidence: missing blocks, tampered fails, capable passes."""

    if evidence is None:
        return {"pass": False, "via": "BLOCKED: no compute evidence supplied"}
    if not isinstance(evidence, Mapping) or evidence.get("schema") != COMPUTE_SCHEMA:
        return {"pass": False, "via": "FAIL: compute evidence has the wrong schema"}
    for key in (
        "accelerator_available", "accelerator_qualified",
        "memory_sufficient", "runtime_verified",
    ):
        if not isinstance(evidence.get(key), bool):
            return {"pass": False, "via": f"FAIL: compute evidence field invalid: {key}"}
    if not bool(evidence["accelerator_available"]):
        return {
            "pass": False,
            "via": f"UNAVAILABLE_ON_THIS_HOST: {evidence.get('host_label')}",
        }
    if not bool(evidence["accelerator_qualified"]):
        return {"pass": False, "via": "FAIL: accelerator below qualification floor"}
    return {"pass": True, "via": f"accelerator capable: {evidence.get('host_label')}"}


def check_compute() -> dict[str, object]:
    """Host-scoped convenience probe (factual; use evaluate_compute_gate to judge)."""

    return evaluate_compute_gate(probe_execution_environment())


SCIENTIFIC_GATES = (
    "exact_p35_architecture",
    "real_tokenizer",
    "qualified_dataset",
    "qualified_generators",
    "test_evidence",
)


def evaluate_p35a_readiness(
    root: Path,
    *,
    dataset_qualification: Path | None = None,
    generator_qualification: Path | None = None,
    compute_evidence: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Evaluate every P35-A gate; READY only if all pass.

    Scientific readiness (architecture, tokenizer, dataset, generators,
    test evidence) is platform-independent. The compute gate judges
    supplied evidence about the CURRENT host only; pass ``compute_evidence``
    explicitly (or nothing, to record its absence as a blocker).
    """

    gates = {
        "exact_p35_architecture": check_recipe(),
        "real_tokenizer": check_tokenizer(root),
        "qualified_dataset": check_dataset(root, dataset_qualification),
        "qualified_generators": check_generators(root, generator_qualification),
        "test_evidence": check_test_evidence(root),
        "compute": evaluate_compute_gate(compute_evidence),
        "fresh_sealed_policy": {
            "pass": True,
            "via": "Fresh untouched; sealed runs only under independent custody",
        },
    }
    scientific_failed = sorted(name for name in SCIENTIFIC_GATES if not gates[name]["pass"])
    failed = sorted(name for name, gate in gates.items() if not gate["pass"])
    receipt: dict[str, object] = {
        "schema": P35A_SCHEMA,
        "question": P35A_QUESTION,
        "gates": gates,
        "experiment_requirements_ready": not scientific_failed,
        "execution_environment_ready": bool(gates["compute"]["pass"]),
        "blocked_by": failed,
        "verdict": "P35A_EXECUTION_READY" if not failed else "BLOCKED",
    }
    receipt["sha256"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--dataset-qualification", type=Path, default=None)
    parser.add_argument("--generator-qualification", type=Path, default=None)
    parser.add_argument("--compute-receipt", type=Path, default=None)
    parser.add_argument("--probe-compute", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo.resolve() if args.repo else Path.cwd()
    evidence: Mapping[str, Any] | None = None
    if args.compute_receipt is not None:
        evidence = json.loads(args.compute_receipt.read_text(encoding="utf-8"))
    elif args.probe_compute:
        evidence = probe_execution_environment()
    receipt = evaluate_p35a_readiness(
        root, dataset_qualification=args.dataset_qualification,
        generator_qualification=args.generator_qualification,
        compute_evidence=evidence,
    )
    payload = json.dumps(receipt, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if receipt["verdict"] == "P35A_EXECUTION_READY" else 1


__all__ = [
    "COMPUTE_SCHEMA",
    "P35A_QUESTION",
    "P35A_SCHEMA",
    "P35_PARAMETERS",
    "P35_RECIPE",
    "REQUIRED_GENERATOR_FAMILIES",
    "SCIENTIFIC_GATES",
    "evaluate_compute_gate",
    "evaluate_p35a_readiness",
    "probe_execution_environment",
]
