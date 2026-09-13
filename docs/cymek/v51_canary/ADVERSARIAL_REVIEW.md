# ADVERSARIAL FINAL QUALIFICATION (canary, §56)

Hostile pass over the canary qualification, question by question, with the holes found and their fixes.

| Question | Finding | Action |
|---|---|---|
| Did any test bypass production code? | No — the runner drives `ProductionTrainingBackend.step`, `certify_update`, `CheckpointStore.publish/restore`, real `v5_data` packing/streaming, real tokenizer artifact. The only new code is the contract/wrapper layer. | verified by reading the runner imports |
| Did data leak across splits? | No — zero exact/normalized/group collisions; group-level cuts; global freshness set (a cross-FAMILY collision was caught and the generator fixed). | screens re-run at prepare |
| Did a trivial heuristic solve the task? | Three near-misses caught pre-training (0.583 binding positional; 0.583 termination constant; 0.367 residual) — each repaired by construction; final worst baseline 0.267 < 0.35. | SHORTCUT_AUDIT.md |
| Did resume reuse in-memory state? | No — genuine `subprocess` boundaries; restore exclusively from durable bytes; bitwise model/optimizer equality proven. | RESUME_AUDIT.md |
| Did the scheduler restart after restore? | No — zero rewarm events; the resume row continues the token-indexed position. | WSD_AUDIT.md |
| Are token counts actual or inferred? | Actual: window ledgers cross-checked against `TrainingState` and `consumed_real_tokens` per update. | TRAINING receipt |
| Did the optimizer miss parameters? | No — `validate_parameter_ownership` exactly-once at construction; `assert_live_ownership` every update. | suite |
| Did the test ever exercise clipping? | Yes — post-clip norm 1.0 during warmup (certificate engaged), relaxing to 0.30 late in decay. | WSD_AUDIT table |
| Did we verify actual runtime dtype? | Yes — CUDA preflight recorded the precision contract receipt (FP32 params/moments/reductions, BF16 autocast) from the live runtime, not config. | PREFLIGHT_GPU receipt |
| Did sealed test influence development? | No tuning loop touched sealed. NOTE: sealed was scored twice — once per full execution — but the two executions are **bitwise identical** (model.bin sha proof), so the second scoring could not have fed anything back; recorded as non-invalidating with the proof. | FINALIZATION + readiness JSON |
| Did an EXPERIMENT_ONLY output path become canonical? | No — constructor gate + identity-hash divergence + canonical-path purity test. | suite |
| Is a partial checkpoint accepted? | No — injected crash preserves LATEST; staging must be resolved; corrupted bytes fail closed. | suite |
| Could a stale Drive checkpoint silently load? | No — identity bindings (source commit, tokenizer, data/pack/config/schedule hashes) are compared at resume; drift → FAIL_CLOSED. | runner gate |
| Could different code produce the same receipt? | Receipts embed the executable commit + generator source sha; blob hashes verified at launcher checkout. | notebook cell 3 |
| Are we calling a synthetic formation score cognition? | No — the claim ceiling names it an integration/formation instrument; Triquetra owns cognition qualification. | CANARY_SPEC/README |
| Are we certifying a GPU we never used? | No — the local RTX 4050 preflight is labeled bounded-smoke evidence; the substantive Rung-B verdict is deferred to operator T4 execution. | GPU preflight receipt note |
| Did BRAMASTRA mechanisms leak into V5.1? | No — contract-level overlap only (EOS, fail-closed checkpoints, recomputed scoring), each independently evidenced. | BRAMASTRA_EXECUTION_LESSONS.md |
| Did we accidentally modify R1C? | No — R1C files untouched; the 24ca7f3 launcher-pin repair (made by the operator on cymek-500m-readiness) was inspected and deliberately NOT ported; no Drive-root overlap. | delta audit in EVIDENCE_INPUT |

## Defects found in MY OWN execution layer by this review (root-caused, fixed, regression-covered)

1. **Tracer LR labeling** — rows labeled post-update vs the frozen pre-update contract (the applied LR sequence was correct). Fixed; execution 2 superseded execution 1 with bitwise-identical weights as proof.
2. **Verdict mapping** — formation-gate failures were initially mapped to CANARY_FAIL_ENGINEERING. Fixed to the §48 taxonomy: mechanics all-pass + formation failure = **CANARY_FAIL_FORMATION**.
3. **Device placement** — model never moved to CUDA for preflight. Fixed.
4. **Ambient RNG** — process-start generator nondeterminism broke bitwise resume. Root-caused and pinned.

## The honest bottom line

The canary **fails formation**: identity/state_order did not learn (0.000), binding sits at its shortcut boundary (0.203 vs 0.225 baseline), and only termination clearly learned (0.313). The mechanical system it runs on is exactly what Task 3 set out to prove — and is now proven — but §31 is explicit: formation failure means **investigate before scaling**, and the designed canary-v2 (≈3-epoch budget crossing the delayed-generalization region) must run before any larger claim. The strongest permitted statement is the claim ceiling in FINALIZATION.json, and even its formation clause is only partially satisfied (termination family).
