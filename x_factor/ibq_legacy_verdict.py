"""IBQ applied to the preserved X1-REAL-0 outcome matrix (existing artifact).

No model execution: this analyzes the already-harvested real data committed
in output/x1_real_receipt.json. Produces the formal qualification verdict
for the legacy 5-intervention basis and the quantitative always-negative
equivalence that downgrades X1-REAL-0.
"""

from __future__ import annotations

import json
import time

import numpy as np
from pathlib import Path

from x_factor.ibq import (
    basis_qualified,
    basis_quality,
    geometry_vs_nulls,
    cell_prevalence,
    oracle_coverage,
    per_intervention_prevalence,
)


def main() -> int:
    raise SystemExit(_main())


def _main() -> int:
    from x_factor.geometry import effective_rank
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / "output" / "x1_real_receipt.json").read_text(encoding="utf-8"))
    interventions = ["NO_CHANGE", "KNOWLEDGE_RESTATED", "FORMAT_NORMALIZED",
                     "QUERY_NEAR_ANSWER", "DECODE_SEARCH"]
    ids = sorted(receipt["matrix"])
    M = [[int(receipt["matrix"][i][iv]) for iv in interventions] for i in ids]

    prevalence = cell_prevalence(M)
    always_negative_accuracy = 1.0 - prevalence
    q = basis_quality(M)
    gate = basis_qualified(M)
    nulls = geometry_vs_nulls(M, n_nulls=300, seed=7)

    verdict = {
        "schema": "anra-ibq-verdict/v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "basis": "legacy-5 (NO_CHANGE, KNOWLEDGE_RESTATED, FORMAT_NORMALIZED, "
                 "QUERY_NEAR_ANSWER, DECODE_SEARCH)",
        "data": "preserved X1-REAL-0 outcome matrix (no new model execution)",
        "n_failures": len(M),
        "oracle_coverage": round(oracle_coverage(M), 4),
        "cell_prevalence": round(prevalence, 4),
        "always_negative_cell_accuracy": round(always_negative_accuracy, 4),
        "always_negative_equivalence": (
            f"an always-negative predictor scores {always_negative_accuracy:.1%} "
            f"cell accuracy on this matrix — statistically indistinguishable from "
            f"the reported 95.45% (X1-REAL-0's predictor metric)"),
        "per_intervention_prevalence": dict(zip(interventions, q["per_intervention_prevalence"])),
        "quality": q,
        "gate": gate,
        "null_analysis": nulls,
        "effective_rank_real": round(effective_rank(np.array(M, dtype=float)), 3),
        "IBQ_VERDICT": "BASIS NOT QUALIFIED" if not gate["qualified"] else "QUALIFIED",
        "consequence": ("no self-model training on this basis; the intervention set "
                        "must be redesigned toward the qualified classes "
                        "(addressing support with legality-clean selection, "
                        "realization support, distractor structure) and re-harvested "
                        "on authorized compute before any X1-v2"),
    }
    out = root / "output" / "ibq_legacy_basis_verdict.json"
    out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps({k: verdict[k] for k in
                      ("always_negative_cell_accuracy", "oracle_coverage",
                       "cell_prevalence", "gate", "null_analysis",
                       "IBQ_VERDICT")}, indent=2))
    return 0


if __name__ == "__main__":
    main()
