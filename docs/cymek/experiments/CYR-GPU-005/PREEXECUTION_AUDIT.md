# CYR-GPU-005 — PRE-EXECUTION AUDIT

Date: 2026-09-08. Branch: `cymek-500m-readiness` @ `e64a148` (audit base).
Auditor: Cymek research/engineering agent (independent pass, live code).

## 1. CYR-GPU-004 runner audit (`experiments/ARK-007/run_v4.py`)

| ID | Finding | Evidence | Severity | Disposition in 005 |
|----|---------|----------|----------|--------------------|
| A1 | HIGH/LOW are independent acquisitions from init; LOW active DURING learning | `run_arm` trains acquisition+continuation per `(seed, lr)`; main() invoked per LR | FATAL | Shared-parent fork contract: ONE acquisition per seed, four forks restore the same bytes |
| A2 | G90 snapshot never restored; optimizer "copy" aliases live state | `snap_model`/`snap_opt` written, never loaded; shallow dict | FATAL | Byte-exact parent checkpoint + `PARENT_EQUIVALENCE.json` equality test before first update |
| A3 | No future-stream identity across arms | per-run `torch.randint` generators | CRITICAL | Pre-generated full stream (prefix+tail), per-arm consumed-batch SHA equality receipt |
| A4 | `MANIFEST_SHA` hardcoded; rows never verified | constant at module foot, copied into receipt | MAJOR | Manifest SHA computed from actual serialized rows; `assert_manifest_sha` fail-closed |
| A5 | EOS=3 hardcoded | `answers + [3]`, stop on `(0, 3)` | MAJOR | Special IDs read from the frozen artifact, cross-checked against `TokenizerIdentity` |
| A6 | Silent CPU fallback | `device = cuda if available else cpu` | MAJOR | Full mode raises without CUDA (test-exercised) |
| A7 | Steps define exposure; `acq_tokens` decorative | loop bounds are steps | MAJOR | Every loop stops on cumulative ACTUAL real tokens; receipts carry target/consumed |
| A8 | One `test` set both controls training and carries the claim | G90 + retention on same rows | MAJOR | Four roles: DEV_CONTROLLER (controls), DEV_MEASUREMENT (passive), SEALED_RESERVED (never consulted), train probe (M99) |
| A9 | 6L/256w proxy labelled "P35-proxy" | spec in `main()` | MAJOR | Canonical proxy registry; `assert_proxy_in_registry` refuses mislabeled scale (P35 = 35,411,328 verified) |
| A10 | Test set also used for G90 controller metric with no split firewall | — | MAJOR | Split manifest + leak audit (ordered-row holdout, distribution matching) |
| A11 | No timebox deadline; no INCONCLUSIVE semantics | unbounded loops | MINOR | One absolute campaign deadline; TIMEBOXed arms cannot be compared |
| A12 | No displacement/moment diagnostics (near-freezing invisible) | — | MINOR | Red-team ledger per evaluation: displacement, relative displacement, Adam moment norms, exposure split |
| A13 | Notebooks ran mutable branch HEAD | clone without checkout | MINOR | Two-commit freeze; CELL 0 checks out commit A and verifies file hashes against an external prereg copy |

## 2. Production XLA accumulation audit (`v5_training/production_entry.py`)

**CONFIRMED at line 941 (audit base).** `xla_adapter.all_reduce_sum_gradients()`
was called INSIDE the per-microstep loop over `windows`. With R replicas and
4 microsteps the accumulated buffer after the loop holds
`R³·S₁ + R²·S₂ + R·S₃ + S₄` instead of `S₁+S₂+S₃+S₄` — early microstep
gradients over-counted by powers of the replica count.

**Fixed in this cycle**: the collective now fires ONCE at the accumulation
boundary (after the final microstep, before `finish_update`'s single
clip/step). Guarded by `tests/test_v5_xla_accumulation_oracle.py` (AST
guard) and the distributed CPU oracle
(`v5_experiments/xla_accumulation_oracle.py`) with the negative regression
(proof that per-microstep reduction FAILS the oracle). Status stays
`IMPLEMENTED_PENDING_PRE500M_TPU` — CPU equivalence is not TPU evidence.

## 3. Cross-branch evidence re-audit (remote state wins)

- **Arkenstone @ `origin/Arkenstone` = `fc3e689`** (newer than the
  `4911b84` recorded in agent.md): ARK-007R replication (LOW 1e-5: 0/12
  collapse; HIGH 1e-3: 9/12; risk difference −0.75); ARK-009 transfer
  gate NOT qualified (diagnostic confounded query swap with fact-order
  reversal); ARK-010 recovery (HIGH 8/9 vs LOW 2/9); Discovery-V6
  receipts 14/14 hash-valid, no failure receipt; ARK-011 RESULT present
  (agent.md said UNEXECUTED — stale).
- **BRAMASTRA @ `origin/BRAMASTRA` = `4655733`** (newer than `90ee31a`):
  `chief_d02_horizon_audit_20260908` (implementation verification, 12
  terminal rows, no training); `discovery_dev_701` parent/child
  comparisons with CIs.
- Consequence for 005: arms follow the replicated trajectory
  (HIGH for acquisition/recovery, LOW for retention after confirmed
  capability; FIXED_TIME + HYST test the untested switching policies);
  the transfer family uses query-only variants to avoid the ARK-009
  confound and reports `NOT_INFORMATIVE` at zero event rate rather than
  manufacturing a failure.

## 4. Notebook audit (`notebooks/CYR-GPU-004.ipynb`)

Standalone fork cell defined its own undefined helpers (`detect_g90`,
`minutes_left` — NameError at runtime, D01/D02 of the v4 audit) and
duplicated architecture classes. CYR-GPU-005's notebook contains NO
science logic (enforced by `tests/test_v5_cyr_gpu005_notebook.py`).
