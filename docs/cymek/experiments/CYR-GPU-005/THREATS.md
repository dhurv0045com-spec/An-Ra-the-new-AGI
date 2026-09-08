# CYR-GPU-005 — THREATS TO VALIDITY

Written before execution; revisited by the section-62 self-red-team at
packaging. Each threat lists its mechanical mitigation.

| # | Threat | Mitigation |
|---|--------|------------|
| T1 | Forks diverge before the first update (restore bug, optimizer aliasing) | `PARENT_EQUIVALENCE.json` byte-hashes model+optimizer state of all four forks and fails the campaign on inequality; checkpoints store bytes, not references |
| T2 | Arms consume different future data (hidden per-arm reseeding) | The stream is pre-generated with SHAs; each arm records the SHAs of batches it ACTUALLY consumed; equality across arms is asserted from those receipts, not assumed |
| T3 | LOW "wins" by near-freezing and is sold as consolidation | Displacement/relative-displacement/Adam-moment ledger on every evaluation; DECISION must carry the near-freezing reading if displacement ≈ 0 |
| T4 | G90 on teacher-forced metrics (candidate leakage) | All gates score free generation with valid stops only; teacher forcing is not computed |
| T5 | Controller/measurement contamination | DEV_CONTROLLER is the only set that can steer LR; DEV_MEASUREMENT never controls; SEALED_RESERVED is scored once, after the decision |
| T6 | Nominal token accounting (steps, capacity, assumed widths) | Every loop stops on cumulative ACTUAL real tokens; receipts carry target/consumed/supervised/executed/padded; backend certifies the ledger per update |
| T7 | Manifest claimed by constant (CYR-GPU-004 defect) | Manifest SHA computed from the exact serialized rows; `assert_manifest_sha` re-derives and fails closed |
| T8 | Canonical-pair "leak" panic on a commutative task | Holdout axis preregistered as the ordered question; closure reported as a declared property with counts; ordered-row crossing still fails the audit |
| T9 | Mislabeled proxy scale (CYR-GPU-004's "P35-proxy") | Registry parameter bands verified against built models in tests; `assert_proxy_in_registry` refuses mismatched scale |
| T10 | EOS/PAD ids hardcoded and silently wrong for the artifact | IDs read from artifact `added_tokens`, cross-checked against `TokenizerIdentity`; mismatch raises |
| T11 | Timebox silently promoted to a win | TIMEBOXed/starved arms are excluded from comparisons; verdict rules return INCONCLUSIVE with reasons |
| T12 | Wall clock kills a campaign mid-update with no resume | Research checkpoints after the parent, every 25 updates, and at each arm end; one absolute deadline with packaging margin |
| T13 | Silent CPU fallback masquerading as GPU evidence | Full mode raises without CUDA; test-exercised; smoke receipts are labeled smoke |
| T14 | Plumbing override leaks into science | Override exists only in smoke mode, stamped `PLUMBING_SMOKE_ONLY` in receipts; full mode refuses it structurally (tested) |
| T15 | XLA accumulation bug resurfaces | AST guard: exactly one all-reduce per logical update, outside the microstep loop; mathematical oracle incl. negative regression; status stays pending TPU |
| T16 | One seed mistaken for replication | Decision requires the full contract; parents are reported individually; unqualified parents fork nothing |
| T17 | Evaluation dominates the wall clock | Batched generation; controller/measurement cadence comes from the resolver; sealed set touched once |
| T18 | Post-freeze code drift | Two-commit freeze; CELL 0 verifies HEAD and every bound file hash against the external prereg copy; drift means SUPERSEDED (section 45) |
| T19 | Bundle loses partial results on failure | Packaging is in a `finally` path; FAILURE.json carries stage/seed/arm/exception/tokens/last-checkpoint; partial receipts survive |
| T20 | Operator runs a mutable-HEAD notebook | Notebook checks out the frozen executable and verifies hashes before anything runs |
