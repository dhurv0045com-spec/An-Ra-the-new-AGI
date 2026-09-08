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
| H-FLOOR at micro scale: REFUTED | ARK-001 | optimization/capacity pathology is NOT the universal explanation of the citadel anomaly |
| H-REPR (vocab) at micro scale: REFUTED | ARK-001 | dead-vocab embedding is not a first-order variable for symbolic micro-learning |
| ARK-001 ByteVocab answer-encoding bug | caught by impossible loss/exact signature; fixed and rerun | eval decode paths must share answer-encoding contract |
| T2 wall-box cut transition mid-flight | ARK-001 | short boxes are inadequate for delayed OOD transition claims |
| Curriculum delays T2 memorization and yields zero OOD in box | ARK-003 | easy-first staging does not accelerate this transition at micro scale |
| Aligned digit-decomposition teacher did not accelerate at equal wall budget | ARK-003 | aligned supervision did not shorten the delay in the tested box |
| M99 does not predict G90 timing | ARK-004A | memorization and generalization timing are decoupled |
| Post-G90 instability exists | ARK-004A onward | generalized state is not automatically stable |
| Weight-decay removal does not prevent decay | ARK-005 | H-WD not supported at micro scale |
| EMA-0.999 does not prevent decay | ARK-005 | simple weight averaging is insufficient here |
| LR 1e-4 is not sufficient to prevent collapse | ARK-006 | 10x LR reduction is insufficient in this regime |
| High-LR decay is not universal or necessarily permanent | ARK-007/007R/010 | HIGH is a risk factor, not deterministic permanent forgetting |
| Immediate LOW is not an effective general recovery intervention after collapse | ARK-010: HIGH recovery 8/9 vs LOW 2/9 | rule `collapse -> lower LR immediately` is falsified on Micro T2 |
| ARK-009 strict transfer qualification not met | ARK-009 | non-arithmetic LR-retention transfer remained untested there |
| ARK-009 composite query-swap diagnostic not causally isolated | V5 runner changed query + order together | low score cannot be attributed specifically to query conditioning |
| Exact best recovery-switch threshold is not identified | ARK-012: threshold aliasing and non-monotonic selected-source screen | do not promote 0.85 or 0.90 as a general threshold |
| LOW LR does not preserve old T2 under 12k no-replay T3-only training | ARK-013 | same-task LOW protection is not a general solution to cross-task interference |
| ARK-013 cannot establish plasticity cost/frontier because HIGH never acquired T3 and ADAPTIVE never switched | ARK-013 | no adaptive-frontier claim from this run |
| ARK-014 non-arithmetic LR transfer screen had zero failures in both arms | ARK-014: HIGH 0/3, LOW 0/3 | cannot infer protection or failure of transfer from a no-event regime |
