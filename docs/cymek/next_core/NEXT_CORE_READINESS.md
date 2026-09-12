# NEXT CORE READINESS

**Verdict: `READY_FOR_CANARY`** — development-scale canary of the V5.1 contract layer only. Machine gate: [`NEXT_CORE_READINESS.json`](NEXT_CORE_READINESS.json).

## Gate states

| Gate | State |
|---|---|
| architecture specification | SATISFIED |
| reference implementation (v5_next; 12/12 CPU tests) | SATISFIED |
| compute model validation (250,216,960 reproduced; 8/8 tests) | SATISFIED |
| spec validator | SATISFIED |
| output-space resolution | OPEN (R1C) |
| corpus readiness | **FAILING** (no corpus; tiered surface retired) |
| evaluation readiness | **FAILING** (Citadel NOT_READY) |
| schedule execution (WSD once) | OPEN |
| scale authorization | **NOT_JUSTIFIED** |

## What the canary is and is not

The canary: instantiate the V5.1 contract geometry at a development rung (byte-6M or ~40M), train through the production path on screened synthetic fixtures, and verify receipt/evaluation/checkpoint contracts end to end.

It does **not** authorize: 500M scheduling, production-corpus claims, cognition promotion, output-space adoption, or any superiority claim. `READY_FOR_500M` is not issued and cannot be issued until the six blockers in the JSON gate clear (R1C, transfer probe, corpus, schedule execution, ~97M formation replication, Citadel green). The gate fails closed: every SATISFIED state is re-verified, not assumed, at evaluation time.
