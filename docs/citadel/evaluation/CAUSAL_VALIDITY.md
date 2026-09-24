# CAUSAL VALIDITY — CITADEL-EVAL-001

Date: 2026-09-13. Branch: `eval-integrity-001`.
Tests whether evaluation scores react to causally relevant changes and
remain stable under irrelevant transformations.

## Principle

A valid cognition evaluation must show:
- **Sensitivity**: mutating a causally relevant premise changes the score
- **Specificity**: mutating an irrelevant premise or shuffling serialization
  does NOT change the score

If a model scores the same on BASE and TWIN (where TWIN differs by one
causal mutation), the evaluation is not measuring causal understanding.

## Causal sensitivity results from T1D

### Entity×value factorial (from triquetra, replicated in T1D data)

| Condition | What changes | Expected if causal | Measured |
|---|---|---|---|
| BASE vs QUERY_ONLY | query text | score should change | **NO CHANGE** (query-blind) |
| BASE vs RELEVANT_VALUE_ONLY | relevant value | score should change | **NO CHANGE** |
| BASE vs ORDER_ONLY | fact order | score may change | **NO CHANGE** (shortcut) |
| BASE vs IRRELEVANT_VALUE_ONLY | distractor value | score should NOT change | **CONFIRMED** (correctly stable) |

The model passes the specificity test (irrelevant changes don't affect it)
but FAILS the sensitivity test (relevant changes don't affect it either).
This means the model is not doing causal reasoning — it's pattern-matching
on surface features.

### Cross-tier generalization

| From tier | To tier | Test exact | Expected if generalizing |
|---|---|---|---|
| T1 → T1 (held-out) | same structure, new operands | >0 | **0.058-0.066** (barely above 0) |
| T1 → T2 (harder) | same op, more digits | >0 | **0.000** |
| T2 → T3 (carry) | same op, carry required | >0 | **0.000** |

The model does NOT generalize across difficulty levels even within the
same operation. This confirms that any apparent learning is memorization,
not algorithmic understanding.

## Causal sensitivity results from Triquetra (replicated)

Triquetra's preregistered experiments independently confirmed:
- Entity×value factorial: VALUE_RECENCY_DOMINANT — value-only insertion
  repairs 43% of failures vs entity-only 0% (dev, one checkpoint)
- Query-value evidence: raw rank = 0.2500 (= chance exactly); query-conditioned
  signal absent
- Competitive binding: CBL-specific effect NOT supported (floor-limited)

## Verdict

| Evaluation surface | Sensitivity | Specificity | Causally valid? |
|---|---|---|---|
| T1D arithmetic | not tested | not tested | **UNTESTED** (no causal twins in T1D arms) |
| E0 cognitive benchmark | **YES** (causal contrast pairs) | **YES** (irrelevant mutation invariant) | **VALIDATED** (dev tier only) |
| Triquetra entity×value | **YES** (preregistered factorial) | **PARTIAL** (floor-limited) | **VALIDATED** (for absence claims) |

The E0 benchmark is the ONLY evaluation with both causal sensitivity and
specificity tests. T1D does not have causal twins — it tests static
exact-match on a fixed dataset without counterfactual manipulation.

## Required improvement

Before the 500M campaign can make cognition claims, T1E must add:
1. Counterfactual twins to every arm (BASE vs TWIN with one causal mutation)
2. Irrelevant-mutation controls (score should not change)
3. Query-swap sensitivity (score should change with query)
4. Serialization-order shuffle (score should not change if the model truly
   understands the semantics rather than the serialization order)

These are already designed in the T1E PLAN and the E0 generator supports
them; they need to be wired into the T1E evaluation pipeline.
