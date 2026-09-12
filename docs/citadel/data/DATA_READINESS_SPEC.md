# DATA READINESS SPEC — Citadel acceptance contract (CITADEL-DATA-001)

## Overview
Fail-closed audit of a candidate training corpus. Twelve independent gates.
OVERALL PASS requires every REQUIRED gate = PASS. Missing evidence = INCONCLUSIVE or FAIL.

## Gates
| Gate | Type | What it checks |
|---|---|---|
| PROVENANCE | REQUIRED | Source identity, version, content hashes |
| SPLIT_INTEGRITY | REQUIRED | Train/dev/test/sealed disjointness; causal twins don't straddle |
| EXACT_DEDUP | REQUIRED | No exact duplicate content |
| NEAR_DEDUP | REQUIRED | Near-duplicate detection (3-gram Jaccard >= 0.85) |
| CONTAMINATION | REQUIRED | No eval content in train (exact + n-gram) |
| SHORTCUT_RESISTANCE | REQUIRED | Trivial baselines don't approach reference solver |
| MIXTURE_MATCH | REQUIRED | Consumed tokens match declared mixture within tolerance |
| TOKENIZER_ACCOUNTING | REQUIRED | Token counts consistent with text lengths |
| PACKING_SAFETY | REQUIRED | Segment isolation, valid boundaries, no cross-doc attention |
| COGNITION_COVERAGE | REQUIRED | Every family has train gen + eval gen + solver + shortcut audit |
| DIVERSITY | REQUIRED | Template saturation, unique template count, dominance check |
| REPRODUCIBILITY | REQUIRED | Regenerable from manifest + seeds + generator identity |

## Verdict
- PASS: all gates PASS
- FAIL: any gate FAIL
- INCONCLUSIVE: evidence missing for a required gate (never a silent skip)

## Boundary
This gate certifies mechanical and scientific-precondition gates only.
It does NOT certify that training will produce intelligence, that the
architecture is sufficient, or that the model will generalize to AGI.
