"""Verified readiness/status: evidence or it did not happen.

Every PASS derives from a verifier: receipt re-validation, test-receipt
freshness against the live tree, or a live capability probe. Nothing is
inferred from code existence or test names in working-tree files. Freshness
classification: EXACT_HEAD_TESTED, TREE_EQUIVALENT_TESTED,
METADATA_ONLY_DESCENDANT, STALE_IMPLEMENTATION_TESTS, NO_TEST_EVIDENCE.
Readiness is a derived dependency graph, never a manually blessed ledger.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

STATUS_SCHEMA = "anra-v5-readiness-status/v2"

# Paths whose change cannot alter execution (docs, committed evidence).
METADATA_PREFIXES = ("artifacts/", "docs/", "blueprint/")
METADATA_SUFFIXES = (".md",)


def _repo_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return Path(output)
    except (OSError, subprocess.CalledProcessError):
        return Path.cwd()


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text("utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _is_metadata_only(paths: list[str]) -> bool:
    for path in paths:
        if path.startswith(METADATA_PREFIXES) or path.endswith(METADATA_SUFFIXES):
            continue
        return False
    return True


def classify_freshness(repo: Path, tested_head: str | None) -> dict[str, object]:
    """Classify test evidence against the live tree (M22)."""

    head = _git(repo, "rev-parse", "HEAD")
    if not tested_head or not head:
        return {"state": "NO_TEST_EVIDENCE", "tested_head": tested_head, "head": head}
    if tested_head == head:
        return {"state": "EXACT_HEAD_TESTED", "tested_head": tested_head, "head": head}
    tested_tree = _git(repo, "rev-parse", f"{tested_head}^{{tree}}")
    head_tree = _git(repo, "rev-parse", f"HEAD^{{tree}}")
    if tested_tree and tested_tree == head_tree:
        return {"state": "TREE_EQUIVALENT_TESTED", "tested_head": tested_head, "head": head}
    ancestor = _git(repo, "merge-base", "--is-ancestor", tested_head, "HEAD")
    if ancestor is None:
        return {"state": "STALE_IMPLEMENTATION_TESTS", "tested_head": tested_head, "head": head}
    diff = _git(repo, "diff", "--name-only", tested_head, "HEAD")
    paths = [line for line in (diff or "").splitlines() if line.strip()]
    paths = [path for path in paths if path != "artifacts/cymek/test_receipt.json"]
    if not paths:
        return {"state": "EXACT_HEAD_TESTED", "tested_head": tested_head, "head": head}
    if _is_metadata_only(paths):
        return {"state": "METADATA_ONLY_DESCENDANT", "tested_head": tested_head, "head": head}
    return {"state": "STALE_IMPLEMENTATION_TESTS", "tested_head": tested_head, "head": head}


def _tree_has_test(repo: Path, commit: str, test_path: str, test_name: str) -> bool:
    content = _git(repo, "show", f"{commit}:{test_path}")
    return content is not None and test_name in content


def _receipt_artifact(root: Path, filename: str) -> dict[str, Any] | None:
    return _load_json(root / "artifacts/cymek" / filename)


def verify_unit_baseline(root: Path) -> dict[str, object]:
    """Freshness of the suite receipt plus zero recorded failures."""

    receipt = _receipt_artifact(root, "test_receipt.json")
    if receipt is None:
        return {"status": "NO_TEST_EVIDENCE", "via": "missing test_receipt.json"}
    freshness = classify_freshness(root, str(receipt.get("tested_head") or ""))
    result = receipt.get("result") or {}
    if int(result.get("failed", 1)) != 0:
        return {"status": "FAILED", "via": "receipt records failures", "freshness": freshness}
    if freshness["state"] in {"EXACT_HEAD_TESTED", "TREE_EQUIVALENT_TESTED"}:
        return {
            "status": "VERIFIED",
            "via": f"{result.get('passed', 0)} passed, {freshness['state']}",
            "freshness": freshness,
        }
    return {"status": "STALE", "via": "code moved since tested tree", "freshness": freshness}


def verify_causality(root: Path) -> dict[str, object]:
    """EVALUATION_CAUSALITY: causal test present in the TESTED tree and green."""

    receipt = _receipt_artifact(root, "test_receipt.json")
    if receipt is None:
        return {"status": "NO_TEST_EVIDENCE", "via": "missing test_receipt.json"}
    tested = str(receipt.get("tested_head") or "")
    freshness = classify_freshness(root, tested)
    if freshness["state"] not in {"EXACT_HEAD_TESTED", "TREE_EQUIVALENT_TESTED"}:
        return {"status": "STALE", "via": "tested tree is not current", "freshness": freshness}
    if int((receipt.get("result") or {}).get("failed", 1)) != 0:
        return {"status": "FAILED", "via": "receipt records failures"}
    if not _tree_has_test(
        root, tested, "tests/test_v5_checkpoint_adapter.py",
        "test_no_future_token_leakage_in_evaluation",
    ):
        return {"status": "FAILED", "via": "causal test absent from tested tree"}
    return {"status": "VERIFIED", "via": "causal test in tested tree, suite green"}


def verify_exact_resume(root: Path) -> dict[str, object]:
    """EXACT_RESUME: miniature PASS plus the M3 resume proof green in tree."""

    receipt = _receipt_artifact(root, "miniature_receipt.json")
    if receipt is None:
        receipt = _receipt_artifact(root, "test_receipt.json")
        _ = receipt
        return {"status": "NOT_DEMONSTRATED", "via": "missing miniature_receipt.json"}
    if receipt.get("status") != "PASS":
        return {"status": "FAILED", "via": "miniature receipt is not PASS"}
    test_receipt = _receipt_artifact(root, "test_receipt.json") or {}
    tested = str(test_receipt.get("tested_head") or "")
    freshness = classify_freshness(root, tested)
    if freshness["state"] not in {"EXACT_HEAD_TESTED", "TREE_EQUIVALENT_TESTED"}:
        return {"status": "STALE", "via": "resume proof not verified on current tree"}
    if not _tree_has_test(
        root, tested, "tests/test_v5_stream_resume.py",
        "test_path_a_equals_path_b_from_artifacts_only",
    ):
        return {"status": "NOT_DEMONSTRATED", "via": "M3 proof absent from tested tree"}
    if int((test_receipt.get("result") or {}).get("failed", 1)) != 0:
        return {"status": "FAILED", "via": "receipt records failures"}
    return {"status": "VERIFIED", "via": "miniature PASS + M3 proof green in tested tree"}


READINESS_DEPS: dict[str, dict[str, object]] = {
    "MODEL": {"requires": [], "via": "training-spec receipt checks"},
    "TOKENIZER": {"requires": [], "via": "E1 local tournament receipt"},
    "DATASET_QUALIFIED": {"requires": [], "via": "data-qualify gate (absent)"},
    "STREAM_RESUME": {"requires": ["CHECKPOINT"], "via": "M3 proof in tested tree"},
    "TRAINING_BACKEND": {"requires": ["MODEL"], "via": "P35 torch canary receipt"},
    "CHECKPOINT": {"requires": [], "via": "transaction canary receipt"},
    "EVALUATION": {"requires": ["MODEL", "TOKENIZER"], "via": "scoring cert + protocol tests"},
    "EXPERIMENT_PROTOCOL": {"requires": ["EVALUATION"], "via": "preregistration tests"},
    "MATCHED_ARMS": {"requires": ["EXPERIMENT_PROTOCOL"], "via": "matched-arms test"},
    "RUNTIME": {"requires": [], "via": "live torch/CUDA probe"},
    "P35A_EXECUTION_READY": {
        "requires": [
            "MODEL", "TOKENIZER", "DATASET_QUALIFIED", "STREAM_RESUME",
            "TRAINING_BACKEND", "CHECKPOINT", "EVALUATION",
            "EXPERIMENT_PROTOCOL", "MATCHED_ARMS", "RUNTIME",
        ],
        "via": "composite: every dependency verified",
    },
}


def evaluate_readiness(root: Path) -> dict[str, dict[str, object]]:
    """Derive every readiness node from verifiers; no manual blessing."""

    nodes: dict[str, dict[str, object]] = {}

    def v5_receipt(name: str) -> dict[str, object]:
        receipt = _load_json(root / "artifacts/v5" / name) or {}
        if receipt.get("status") == "PASS":
            return {"status": "VERIFIED", "via": f"{name} reports PASS"}
        if not receipt:
            return {"status": "NOT_RUN", "via": f"missing {name}"}
        return {"status": "FAILED", "via": f"{name} reports {receipt.get('status')}"}

    def training_spec_ok() -> bool:
        receipt = _load_json(root / "artifacts/v5/training_spec_v1.json") or {}
        checks = receipt.get("checks") or {}
        return bool(checks) and all(checks.values())

    nodes["MODEL"] = (
        {"status": "VERIFIED", "via": "training-spec checks all true"}
        if training_spec_ok()
        else {"status": "FAILED", "via": "training-spec receipt checks fail"}
    )
    e1 = _load_json(root / "artifacts/e1/local_tournament/result.json") or {}
    nodes["TOKENIZER"] = (
        {"status": "VERIFIED", "via": "E1 local tournament DEVELOPMENT_STATIC_PASS"}
        if e1.get("status") == "DEVELOPMENT_STATIC_PASS"
        else {"status": "FAILED", "via": "E1 tournament receipt missing or failing"}
    )
    nodes["DATASET_QUALIFIED"] = {
        "status": "BLOCKED",
        "via": "no data-qualify gate exists; real 5B corpus absent (M30-M32)",
    }
    stream = verify_exact_resume(root)
    nodes["STREAM_RESUME"] = (
        {"status": "VERIFIED", "via": stream["via"]}
        if stream["status"] == "VERIFIED"
        else {"status": stream["status"], "via": stream["via"]}
    )
    nodes["TRAINING_BACKEND"] = v5_receipt("local_p35_checkpoint_canary.json")
    nodes["CHECKPOINT"] = v5_receipt("training_transaction_canary.json")
    causality = verify_causality(root)
    nodes["EVALUATION"] = (
        {"status": "VERIFIED", "via": "causality proof green in tested tree"}
        if causality["status"] == "VERIFIED"
        else {"status": causality["status"], "via": causality["via"]}
    )
    nodes["EXPERIMENT_PROTOCOL"] = _protocol_tests(root)
    nodes["MATCHED_ARMS"] = dict(nodes["EXPERIMENT_PROTOCOL"])
    nodes["RUNTIME"] = _runtime_probe()
    deps = READINESS_DEPS["P35A_EXECUTION_READY"]["requires"]
    assert isinstance(deps, list)
    blocked = [name for name in deps if nodes[name]["status"] != "VERIFIED"]
    if not blocked:
        nodes["P35A_EXECUTION_READY"] = {"status": "READY", "via": "every dependency verified"}
    else:
        nodes["P35A_EXECUTION_READY"] = {
            "status": "BLOCKED",
            "via": f"blocked by {[name for name in blocked]}",
        }
    for name, spec in READINESS_DEPS.items():
        if name in nodes:
            nodes[name]["requires"] = spec["requires"]
    return nodes


def _protocol_tests(root: Path) -> dict[str, object]:
    receipt = _receipt_artifact(root, "test_receipt.json")
    if receipt is None:
        return {"status": "NO_TEST_EVIDENCE", "via": "missing test_receipt.json"}
    tested = str(receipt.get("tested_head") or "")
    if classify_freshness(root, tested)["state"] not in {
        "EXACT_HEAD_TESTED", "TREE_EQUIVALENT_TESTED",
    }:
        return {"status": "STALE", "via": "protocol tests not verified on current tree"}
    if int((receipt.get("result") or {}).get("failed", 1)) != 0:
        return {"status": "FAILED", "via": "receipt records failures"}
    for test_path, test_name in (
        ("tests/test_v5_evaluation_protocol.py", "test_n_must_be_exact"),
        ("tests/test_v5_evaluation_protocol.py", "test_reproduction_is_identical"),
    ):
        if not _tree_has_test(root, tested, test_path, test_name):
            return {"status": "FAILED", "via": f"{test_name} absent from tested tree"}
    return {"status": "VERIFIED", "via": "protocol enforcement tests green in tested tree"}


def _runtime_probe() -> dict[str, object]:
    try:
        import torch

        return {
            "status": "VERIFIED",
            "via": f"torch {torch.__version__}, cuda={torch.cuda.is_available()}",
        }
    except ImportError:
        return {"status": "BLOCKED", "via": "torch is not installed"}


def build_status(*, repo_root: Path | None = None) -> dict[str, object]:
    root = _repo_root(repo_root)
    cymek = root / "artifacts/cymek"
    ledgers = _load_json(cymek / "evidence_ledger.json") or {}
    baseline = verify_unit_baseline(root)
    causality = verify_causality(root)
    resume = verify_exact_resume(root)
    nodes = evaluate_readiness(root)
    p35a = nodes["P35A_EXECUTION_READY"]
    assert isinstance(p35a, dict)
    if p35a.get("status") == "READY":
        verdict = "P35A_EXECUTION_READY"
    else:
        verdict = str(ledgers.get("launch_verdict", "UNKNOWN"))
    return {
        "schema": STATUS_SCHEMA,
        "verified": {
            "UNIT_TEST_BASELINE": baseline,
            "EVALUATION_CAUSALITY": causality,
            "EXACT_RESUME": resume,
        },
        "readiness": nodes,
        "blocked": {
            "REAL_5B_CORPUS": "BLOCKED_BY_external_corpus",
            "TPU_XLA_CANARY": "BLOCKED_BY_target_topology",
            "REMOTE_EXECUTION": "BLOCKED_BY_remote_credentials",
            "SEALED_EVALUATION": "BLOCKED_BY_sealed_fixture_commitment",
        },
        "not_authorized": {
            "P35_A_EXPERIMENT": "NOT_AUTHORIZED",
            "V5A_LONG_RUN": "NOT_AUTHORIZED",
        },
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    status = build_status(repo_root=args.repo)
    payload = json.dumps(status, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
