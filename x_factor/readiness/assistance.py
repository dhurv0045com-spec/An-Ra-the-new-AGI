"""Legal assistance taxonomy + capability layers (Mission 16/17/18).

TWO orthogonal axes — never mix them:

  ASSISTANCE (what external help was given):
    A0 surface-preserving / formatting (no semantic selection)
    A1 candidate restriction (output space narrowed, no selection made)
    A2 answer-blind query-conditioned context transformation
       (e.g. visible-query fact duplication; recency/shape changed)
    A3 external fact selection/decomposition
       (CONNECTOR_EXTERNAL_LOOKUP_ASSIST: E7 isolates the matched fact;
       if it works, credit the Connector lookup + Core reduced-context use,
       NEVER "native binding")
    A4 oracle evaluator assistance (gold supplied; ceiling only)

  LAYER (where the measured capability lives):
    NATIVE_CORE        no external semantic selection
    CONNECTOR_ASSISTED answer-blind external computation
    ORACLE_ASSISTED    evaluator truth supplied

The internalization frontier compares Native vs Connector vs Oracle on the
same family. Training should move tasks ORACLE_ONLY -> CONNECTOR_ELICITABLE
-> RAW_NATIVE; the taxonomy tells us which move happened.
"""

from __future__ import annotations

ASSISTANCE = {
    "A0": "surface-preserving/formatting",
    "A1": "candidate restriction",
    "A2": "answer-blind query-conditioned context transformation",
    "A3": "CONNECTOR_EXTERNAL_LOOKUP_ASSIST (external fact selection)",
    "A4": "oracle evaluator assistance (ceiling only)",
}

LAYERS = ("NATIVE_CORE", "CONNECTOR_ASSISTED", "ORACLE_ASSISTED")

# Legal arms used by the readiness pipeline, with assistance classes.
LEGAL_ARMS = {
    "raw": {"assistance": None, "layer": "NATIVE_CORE"},
    "e5dup": {"assistance": "A2", "layer": "CONNECTOR_ASSISTED"},
    "e5sham": {"assistance": "A2", "layer": "CONNECTOR_ASSISTED",
               "note": "matched control: same transformation, wrong target"},
    "e7sel": {"assistance": "A3", "layer": "CONNECTOR_ASSISTED",
              "note": "CONNECTOR_EXTERNAL_LOOKUP_ASSIST: success credits "
                      "Connector lookup + reduced-context use, never native binding"},
    "oracle": {"assistance": "A4", "layer": "ORACLE_ASSISTED",
               "note": "ORACLE_REALIZATION_CEILING, not binding evidence"},
}
