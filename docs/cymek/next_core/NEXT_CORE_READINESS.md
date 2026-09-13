# NEXT CORE READINESS

**Verdict: `READY_FOR_CANARY`** — development-scale canary of the V5.1 contract layer only. Machine gate: [`NEXT_CORE_READINESS.json`](NEXT_CORE_READINESS.json).

## Gate states

| Gate | State |
|---|---|
| architecture specification | SATISFIED |
| reference implementation (v5_next; 12/12 CPU tests) | SATISFIED |
| compute model validation (250,216,960 reproduced; 8/8 tests) | SATISFIED |
| spec validator | SATISFIED |
| output-space mechanism | **R1C SATISFIED / NEGATIVE** — inactive-softmax competition is not sufficient |
| physical vocabulary/output geometry | OPEN — real V4096↔V24576 transfer still required |
| corpus readiness | **FAILING** (no corpus; tiered surface retired) |
| evaluation readiness | **FAILING** (Citadel NOT_READY) |
| schedule execution (WSD once) | OPEN |
| scale authorization | **NOT_JUSTIFIED** |

## R1C update

`CYR-GPU-014-R1C` completed all 24/24 arms on four matched seeds. The preregistered `MASK_4096 - FULL_24576` formation-AUC gaps were `[-0.139792, -0.242215, -0.019377, -0.040138]`, mean `-0.110381`; functional and structural primary tests both return unsupported / not sufficient. Source bundle SHA-256: `2a9359e49f792f962774e42b4e5825b93fb7c91ddcad8f530c1c6f16c6f9bf9e`. See [`R1C_POSTRUN_UPDATE.md`](R1C_POSTRUN_UPDATE.md).

Architecture consequence: do **not** promote masked-output Candidate B. Candidate A's canonical full-softmax path remains the conservative canary default. This is not proof that physical 24,576 is optimal: R1C held the physical matrix fixed, so actual vocabulary/output geometry remains blocked on `CS-TRANSFER-001`.

## What the canary is and is not

The canary: instantiate the V5.1 contract geometry at a development rung (byte-6M or ~40M), train through the production path on screened synthetic fixtures, and verify receipt/evaluation/checkpoint contracts end to end.

It does **not** authorize: 500M scheduling, production-corpus claims, cognition promotion, output-space adoption, or any superiority claim. `READY_FOR_500M` is not issued and cannot be issued until the remaining blockers clear: physical class-space transfer, production corpus, canonical WSD execution, ~97M multi-seed formation replication, and Citadel PRE500M green. The gate fails closed: every SATISFIED state is re-verified, not assumed, at evaluation time.
