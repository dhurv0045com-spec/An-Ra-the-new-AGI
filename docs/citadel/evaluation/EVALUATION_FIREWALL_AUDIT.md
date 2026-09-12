# EVALUATION FIREWALL AUDIT — CITADEL-EVAL-001

Date: 2026-09-13. Branch: `eval-integrity-001`.

## What the firewall is supposed to prevent

1. Training code reading sealed evaluation labels
2. Diagnosis code reading sealed labels
3. Curriculum generation reading sealed labels
4. Promotion logic consuming mutable results
5. Results modifying thresholds after the fact
6. Evaluation generators drifting after checkpoint selection
7. Answer/gold leakage through metadata

## Audit findings

### FW-1: Training code reading sealed labels
**Status: PASS (structurally enforced)**
- `v5_training.trainer` has no import path to `e0_cognition.evaluation_generators`
- `v5_data` split assignment is hash-based; sealed rows get one split, not training
- `production_entry.py` contamination gate scans against benchmark prompts
- However: `contamination_benchmarks` defaults to empty dict in production calls

### FW-2: Diagnosis code reading sealed labels
**Status: PASS (type-enforced on Cymek side)**
- `v5_evaluation.firewall.VisibleTask` structurally cannot carry gold (TypeError on gold kwarg)
- `EvaluatorTruth` joins a frozen `CommittedOutput` only via `score_committed` with task-id match
- **Caveat**: type-level discipline in one process, not a process boundary

### FW-3: Curriculum reading sealed labels
**Status: PASS (namespace isolation)**
- Training generators use `train.causal.*` prefix; evaluation uses `dev.*` / `sealed.*`
- `assert_training_eval_disjoint()` checks template-prefix collision
- However: the 173 dev + 357 test cross-split text overlaps were found via
  exact-hash comparison, NOT caught by the template-prefix check

### FW-4: Promotion consuming immutable receipts
**Status: PASS (structurally enforced)**
- `v5_promotion.gates` requires detached signature + hash chain
- `TrainingState.identities` binds model/tokenizer/data/pack/runtime SHAs
- Chronology-based promotion rejected (must use quality-based selection)
- However: no real promotion decision has ever been made (no production run)

### FW-5: Results modifying thresholds
**Status: NO EVIDENCE FOUND**
- Git history searched for threshold changes after result publication
- T1D thresholds were preregistered in PLAN.md before execution
- No evidence of post-hoc threshold adjustment in the git log
- The `build_next_500m_decision` and `build_decision` gates are
  fail-closed with frozen threshold sets

### FW-6: Evaluation generator drift
**Status: AMBIGUOUS**
- Generator version is recorded (`e0-eval/0.4.0`, `tiered-arith/1.0`)
- CI byte-compares receipts against regenerated output
- However: no hash-based lock prevents a future commit from changing
  generator code while keeping the same version string
- The `RUNTIME_AMENDMENT_001.md` protocol (freeze → verify → execute)
  mitigates but does not eliminate this risk

### FW-7: Answer leakage through metadata
**Status: PARTIAL**
- `CausalCase.model_view()` excludes gold, candidates, graph, traces, seeds
- Tiered data rows contain the answer in the text (`"12 + 9 = 21"` — answer IS in the text)
- For arithmetic this is by design (the model must learn to compute)
- For cognition probes, answers appear in metadata but not in model_view
- **Risk**: metadata answer leakage if a metadata field is accidentally
  included in the prompt or the model learns to read metadata channels

---

## Git history search results

Searched `origin/citadel`, `origin/cymek`, `origin/triquetra`,
`origin/Arkenstone`, `origin/BRAMASTRA` for:

| Pattern | Found? | Details |
|---|---|---|
| test changed after failure | YES | T1C parser bug fixed after T1C ran (hotfix commit `f23bbee`); T1D source_commit bug fixed after T1D failed |
| threshold moved | NO | No threshold changes found in git history |
| generator changed | YES | E0 generator 0.3.0 → 0.4.0 (shortcut repair, documented); tiered_data v1.0 unchanged since creation |
| result reinterpreted | YES | Arkenstone ARK-004A precursor claim INVERTED by own reanalysis (VERDICT B+C); properly documented |
| fresh set reusing template | YES | Tiered data dev/test share operand ranges with train within T0/T1 (by design, disclosed) |

## Conclusion

The firewall is structurally sound at the type/namespace level. The main
risks are:
1. Metadata answer leakage (partial mitigation via model_view)
2. Generator version drift without hash lock (mitigated by RUNTIME_AMENDMENT)
3. Cross-split text overlap (found by the data gate, not by the template check)

No case of deliberate threshold manipulation, result fabrication, or
sealed-label leakage was found in the git history.
