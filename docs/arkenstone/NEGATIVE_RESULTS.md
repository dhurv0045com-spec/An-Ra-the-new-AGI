# NEGATIVE RESULTS (preserved permanently)

## Inherited (executed on TPU, receipts in citadel)

| Result | Evidence | What it rules out |
|---|---|---|
| Whole-row CE: loss 10.10->2.85, exact 0/500 | citadel T1 | whole-row masking is not the lever |
| Answer-only CE: loss ->1.90, exact 0/500 | citadel T1C arm B | answer masking is not the lever |
| Rich 6.5M-row corpus: exact 0/500 | citadel T1C A/B | corpus size is not the lever at 4M tokens |
| Narrow 4k-row pool: train exact 6/500 only | citadel T1C arm C | narrowness alone does not create instance-fitting |
| 2.3x scale (1.6M->3.7M): exact 0/500 | citadel T1C arm D | modest scale alone is not the lever |
| Copy-first-operand heuristic 2.7% null | citadel T1C | tasks not trivially shortcut at answer level |

## Arkenstone

| Result | Evidence | What it rules out |
|---|---|---|
| (ARK-001 results recorded in EXPERIMENT_LOG when complete) | experiments/ARK-001/RESULT.json | — |
| H-FLOOR at micro scale: REFUTED | ARK-001 (all arms lift off, 200-400 steps) | optimization/capacity pathology is NOT the universal explanation of the citadel anomaly |
| H-REPR (vocab) at micro scale: REFUTED | ARK-001 (T1-BYTE == T1-COMPACT, lift-off 200) | dead-vocab embedding is not a first-order variable for symbolic micro-learning |
| ARK-001 harness bug (self-caught): ByteVocab.encode prepended PAD not BOS to answers -> loss 0.0 / exact 0.0 artifact | caught by the impossible signature, fixed, arm rerun; superseded artifact kept in git history | eval decode paths must share the answer-encoding contract |
| T2 wall-box cut the grokking transition mid-flight (0.365 and climbing at box) | ARK-001 RESULT.json trajectory | preregistered 12-min boxes are too short for OOD-transition readouts; ARK-002a extends |
| Curriculum (25% T1 stage): delays T2 memorization, zero OOD at box | ARK-003 arm B | easy-first staging does not accelerate this transition at micro scale |
| Aligned digit-decomposition teacher: no acceleration at equal wall budget | ARK-003 arm C (vs D control, both null) | aligned supervision did not shorten the post-memorization delay within budget; compute handicap documented — step-matched rerun still open |
| Counterfactual locality co-emerges with OOD (tentative positive) | ARK-003 arm A trajectory | behavioral factorization signature tracks the transition (single arm/snapshot — tentative) |
| M99 (memorization speed) does NOT predict G90 timing (rho 0.00) | ARK-004A | memorization and generalization are decoupled phenomena |
| Post-G90 instability: seed 101 collapsed 1.0->0.188 after sustained G90 | ARK-004A | the generalized state is not automatically stable; retention is a first-class objective |
| (BRAMASTRA, inherited) Citadel T1C had an UNTESTED stopping requirement: no answer-terminator token was ever supervised; all arms ended MAX_TOKENS:1000, so exact-match 0/500 conflates computation with termination | BRAMASTRA docs/bramastra/EVIDENCE.md static contract audit + their EOS experiment (0/32 -> 32/32) | T1C's 'formatting ruled out' claim was too strong; Citadel T1D must confirm EOS supervision before its arms are interpretable |
| Weight-decay removal does not prevent post-G90 decay (identical trajectory to control) | ARK-005 arm C | H-WD not supported at micro scale |
| EMA-0.999 consolidation does not prevent post-G90 decay (identical trajectory to control) | ARK-005 arm D | simple consolidation machinery is insufficient; the decay is in the training dynamics, not the weights' recency |
| LR-decay x0.1 delays but does not prevent post-G90 collapse (single decaying seed) | ARK-005 arm B | H-LR weakly supported; stronger LR intervention untested |
| LR 1e-4 (10x reduction from 1e-3) is NOT sufficient to prevent collapse — threshold is lower | ARK-006 | the decay requires a >=100x LR reduction to prevent, not just a 10x reduction |
