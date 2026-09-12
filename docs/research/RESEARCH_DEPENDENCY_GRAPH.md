# RESEARCH DEPENDENCY GRAPH

**Phase 2 · 2026-09-13.** Machine source: [`RESEARCH_DEPENDENCY_GRAPH.json`](RESEARCH_DEPENDENCY_GRAPH.json) (schema `anra.research-dependency-graph/v1`). Nodes are unresolved results, decisions, gates, experiments, terminals; edges carry `discriminates / informs / required_for / feeds / prerequisite / strengthens_or_reverses / authorizes`.

## Critical path 1 — representation → Core spec (the science spine)

```
R1C execution (frozen; ~22 T4-h; launcher rebind first)
  ↓ discriminates
representation/output-space mechanism decision
  ↓ informs
tied vs controlled output-head / softmax-partition treatment
  ↓ informs
production vocabulary size + numeric atomization
  ↓ feeds
500M Core specification
```

Side condition: `CS-TRANSFER-001` (transfer probe) feeds the vocabulary decision — without it, any class-space-based choice is arithmetic-micro-scoped.

## Critical path 2 — continual learning → internalization

```
ARK-019 V4 bundle recovery + byte audit (zero GPU if bundle exists)
  ↓ discriminates
Guardian-vs-static validity (prevention + efficiency)
  ↓ prerequisite
ARK-021 retention-vs-reacquisition
  ↓ feeds
internalize-controller decision (currently BLOCKED)
```

ARK-020 execution strengthens or reverses Guardian validity; its *interpretation* depends on the V4 audit even though its *execution* does not.

## Critical path 3 — scale authorization

```
production 5B corpus materialized + qualified   [BLOCKED_BY_EXTERNAL]
top-level entry point + canonical schedule executed once   [OPEN engineering]
  ↓ both required for
PRE500M green decision   ← representation decision joins here
  ↓ authorizes
500M Core specification
```

NEW blocker from the snapshot delta: the arithmetic/cognition generator surface itself (`CITADEL-DATA-001`: latest-position shortcut 1.000/tier, 530 verbatim cross-split docs, 13.5% duplication, supply 0.004× of 500M) must be regenerated and re-screened before it feeds CS-TRANSFER-001's production-tokenizer task arm or any future cognition data.

## Critical path 4 — measurement validity

```
CIT-C0 extended scorer screen (no GPU)
  ↓ discriminates
certified scoring mode (production_scoring_mode != null)
  ↓ required_for
mechanism studies / assisted cognition comparisons
```

Independent track: a checkpoint passing readiness-v2 (subject qualification) is the second prerequisite for mechanism instrumentation.

## Decision-dependency rules (for the architecture agent)

1. No vocabulary/tied-output choice may be justified while `N-REPR-DECISION` is blocked — the only legal states are "default baseline" or "provisional".
2. No internalization of replay/controller logic while `N-GUARDIAN-VALIDITY` is open.
3. No assisted-scoring cognition claim while `N-SCORER` is blocked; candidate-free primary evaluation is unaffected.
4. No PRE500M green while the corpus gate, entry-point gate, and representation decision are unresolved.
5. Any new arithmetic experiment requires the regenerated corpus surface (post `CITADEL-DATA-001`).
