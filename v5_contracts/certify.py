"""Emit the executable 250M model/run contract receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_spec import V5A_250M
from .run_spec import V5A_RUN_CENTER


def build_certificate() -> dict[str, object]:
    parameters = V5A_250M.parameter_receipt()
    run = V5A_RUN_CENTER.receipt(V5A_250M)
    checks = {
        "target_within_half_percent": abs(parameters.total - 250_000_000) / 250_000_000 < 0.005,
        "head_dimensions_exact": V5A_250M.width == V5A_250M.query_heads * V5A_250M.head_dimension,
        "gqa_groups_integral": V5A_250M.query_heads % V5A_250M.kv_heads == 0,
        "token_budget_near_twenty_per_parameter": 19.5 <= run["tokens_per_parameter"] <= 20.5,
        "token_budget_terminates_exactly": (
            run["full_size_updates"] * V5A_RUN_CENTER.tokens_per_update
            + run["final_update_tokens"]
            == V5A_RUN_CENTER.token_budget
            and run["termination_policy"] == "exact-final-partial-update-no-overshoot"
        ),
        "main_training_authorized": False,
    }
    return {
        "schema": "anra-v5-implementation-contract-certificate/v1",
        "status": "PASS" if all(value for key, value in checks.items() if key != "main_training_authorized") else "FAIL",
        "scope": "configuration and infrastructure contracts; no model or trainer instantiated",
        "model": V5A_250M.canonical(),
        "model_sha256": V5A_250M.sha256(),
        "parameters": parameters.as_dict(),
        "run": run,
        "checks": checks,
        "contract_schemas": {
            "model": "anra-v5-model-spec/v1",
            "run": "anra-v5-run-spec/v2",
            "source_data_pack": "anra-v5-*-manifest/v1",
            "checkpoint": "anra-v5-checkpoint/v1",
            "checkpoint_transaction": "anra-v5-checkpoint-transaction/v1",
            "training_state": "anra-v5-training-state/v1",
            "evaluation": "anra-v5-evaluation/v1",
            "durability": "anra-v5-durability/v1",
            "promotion": "anra-v5-promotion/v2",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/v5/implementation_contract.json"))
    args = parser.parse_args()
    certificate = build_certificate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": certificate["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if certificate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
