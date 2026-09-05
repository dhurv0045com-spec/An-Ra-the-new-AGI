# MECHANISM TOURNAMENT (Arkenstone)

Verdicts: UNTESTED / TESTING / FAILED / TENTATIVE / SUPPORTED / REPLICATED /
NOVEL_CANDIDATE / PROMOTED_TO_CORE / REJECTED / SUPERSEDED / ALREADY_KNOWN /
REPRODUCTION_ONLY / EXTENSION_ONLY. Failed mechanisms stay here forever.

| ID | Mechanism | Novelty | Prior status | Hypothesis (observed failure -> why it should help) | Verdict |
|----|-----------|---------|--------------|------------------------------------------------------|---------|
| M-001 | Lift-off dose mapping (measurement instrument) | NEW_MEASUREMENT | No branch measured lift-off dose | T1/T1C train-exact ~0 everywhere -> first establish WHERE lift-off exists at all | SUPPORTED (ARK-001 executed: lift-off 200-400 steps; grokking transition found on T2) |
| M-002 | Curriculum (easy->hard tiers) | ALREADY_KNOWN (designed) | citadel T1D arm B, unexecuted | multi-digit mixed exposure too hard -> staged lift-off | UNTESTED (await ARK-001 boundary measurement) |
| M-003 | Micro-teacher rows (digit/subproblem supervision) | ALREADY_KNOWN (designed) | citadel T1D arm C, unexecuted | answer CE hides algorithmic structure -> teacher rows | UNTESTED |
| M-004 | Vocab reduction for symbolic tasks | EXTENSION | T1D arm E designed, never run | dead-vocab embedding dilutes capacity -> compact vocab | REJECTED_AT_MICRO_SCALE (ARK-001: byte-vocab lift-off identical to compact; revisit only at P35 scale) |
| M-005 | Query-swap auxiliary objective (lambda 0.05/0.15) | ALREADY_KNOWN (esoes/cymek, fail-closed, never run) | challenger only | binding failures -> query-conditioning pressure | UNTESTED |
| M-006 | Recurrent/universal blocks, adaptive depth, memory tokens | ALREADY_KNOWN families | no branch tested any | unknown until lift-off exists to measure against | PARKED (no capability to move yet) |
| M-007 | Interference-pair binding generator (pair-preserving v2) | REPRODUCTION_ONLY (cymek) | cymek 3e1b8b2, qualification test-asserted | n/a (data, not mechanism) | REPLICATED (Arkenstone red-team receipt: CONSISTENT, 3 seeds + trained escalation baseline) |
