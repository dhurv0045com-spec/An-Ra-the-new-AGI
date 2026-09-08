# Cross-branch evidence audit (Arkenstone 933d4f3, BRAMASTRA 90ee31a, readiness HEAD)

Independent audit for CYR-GPU-001/002. Raw evidence wins over prompt hypotheses.
Scale context: Arkenstone Micro ≈ 0.8M params CPU/T4; BRAMASTRA smoke ≈
117k params CPU; Cymek proxies in this tournament ≈ 4–35M on Colab GPU.

## 2026-09-08 delta re-audit
- Arkenstone 4911b84 → 933d4f3: ARK-012/013/014 + V6 runtime + discovery
  campaign are all PREREGISTERED/UNEXECUTED (runners exist, zero result
  receipts). ARK-011 still UNEXECUTED. No new evidence; prior decisions
  stand. ARK-014's orthogonal factorial plan corroborates our factorial
  design (independent convergence, not copying).
- BRAMASTRA unchanged at 90ee31a: terminal mechanics re-confirmed from
  raw manifests (matched answer weight, +9600 terminal targets, 2 seeds);
  query-blind numbers re-confirmed (62/128 vs 64/64 baselines, 0/64 both);
  mixture/teaching/full-replay still UNEXECUTED. Prior decisions stand.

## Findings

| Finding | Branch@SHA | Experiment | Receipt | Seeds | Metric / effect | Strongest alternative | Confound | Scale/HW | Transfer | Relevance | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LOW LR protects post-G90 retention (HIGH 9/12 vs LOW 0/12 collapse) | ARK@4911b84 | ARK-007R | RESULT.json:15-28 | 3 acq × 4 forks | risk-diff −0.75, discordant 9/0 | near-freezing (Δθ 0.379 vs 0.008) | nesting (12 forks/3 models); clean otherwise | Micro/T4 | NOT_DEMONSTRATED | core | TEST_IN_CYMEK |
| HIGH LR reacquires from collapse (8/9 vs 2/9) | ARK@4911b84 | ARK-010 | RESULT.json:9-19 | 9 nested sources | sustained G90 8/9 vs 2/9 | LOW leaves degraded state | nesting; collapse=episode not forgetting | Micro/CUDA | none | core | TEST_IN_CYMEK |
| HIGH→LOW hysteretic switch policy | ARK@4911b84 | ARK-011 | NONE (notebook never run) | — | prereg only | — | — | — | — | core | REPLICATE_FIRST (as CYR paradigm, adapted) |
| Orthogonal query/order transfer | ARK@4911b84 | ARK-009 | RESULT.json:11 (acq only) | 2 | swap 0.31–0.38, qual bar correctly blocked forks | composite swap confound | query AND order changed together | Micro/CUDA | UNEXECUTED | gate | REPLICATE_FIRST (factorial as gate) |
| EOS supervision repairs stopping (0/32→32/32) | BRA@90ee31a | terminal_dev | manifest+result ×2 seeds | 2 | exact+stop +1.0 | extra gradient magnitude | matched init/batches/updates/answer-weight | 117k/CPU | REJECTED/NULL | contract | ADOPT_MECHANICAL_CONTRACT |
| ~50% accuracy coexists with zero query-sensitivity | BRA@90ee31a | binding_diversity | analysis.json | 1(+recomputed) | both-correct 0/64, same-answer 62/64 | — | exploratory, unmatched worlds contrast | CPU | — | core | TEST_IN_CYMEK |
| Counterfactual-grouped minibatches | BRA@90ee31a | B0 proposal | PREREG only | — | — | — | — | — | — | core | TEST_IN_CYMEK (same as above) |
| Vocab reduction accelerates lift-off | ARK@4911b84 | ARK-001 | ANALYSIS.md | 1 | no accel | leakage inflates OOD | overlaps, wall-box | Micro/CPU | none | — | REJECT_FOR_NOW |
| Easy→hard curriculum | ARK@4911b84 | ARK-003 | ANALYSIS.md | 1 | delays mem, OOD 0 | wall-time handicap | unmatched steps/tokens | Micro/CPU | none | — | REJECT_FOR_NOW |
| Naive EMA / wd-removal fix instability | ARK@4911b84 | ARK-005 | ANALYSIS.md | 2 | NULL | ceiling seed | post-hoc trigger impl | Micro/GPU | none | — | REJECT_FOR_NOW |
| 10x LR reduction sufficient | ARK@4911b84 | ARK-005/006 | ANALYSIS.md | 1–2 | insufficient/single-seed threshold | freezing | single seed, reused control | Micro | none | diagnostic | RETAIN_AS_DIAGNOSTIC |
| Micro-teacher effect | ARK@4911b84 | ARK-003 | ANALYSIS.md | 1 | confounded null | wall handicap | unmatched | Micro | none | — | REJECT_FOR_NOW |
| Memorization timing predicts generalization | ARK@4911b84 | ARK-004A | ANALYSIS/REANALYSIS | 4 | rho 0.00 | — | LOO unspecified | Micro/GPU | none | — | REJECT_FOR_NOW |
| Precursor selectivity predicts G90 | ARK@4911b84 | ARK-004A | ANALYSIS | 4 | fails 3/4, =LATER | lookup-phase sensitivity | inverted prose | Micro/GPU | none | marker | RETAIN_AS_DIAGNOSTIC |
| Learned discovery beats random | BRA@90ee31a | discovery_dev | manifest/result | 1 | n.s. (+0.034 CI crosses 0) | privileged teacher imitation | entangled seeds, 1 round | 35k/CPU | — | — | PARK |
| Replay rescues retention (this regime) | BRA@90ee31a | discovery replay | comparison.json | 1 | NULL/REJECTED (−72.9pp conjunction) | reset-Adam children | not full resume | CPU | — | — | PARK |
| B0/B1 full, T1D, P35-A | BRA@90ee31a | B0/B1/T1D/P35-A | PREREG/DRAFT only | — | — | — | — | — | — | — | PARK |
| COLAB-001 program replication | ARK@4911b84 | COLAB | PROGRAM_SUMMARY.json | multi | 5 exps / 84 min T4 | — | — | T4 | replicated | method | ADOPT_MECHANICAL_CONTRACT (notebook pattern) |

Status legend: ARK-011 = UNEXECUTED (prereg only). ARK-009 retention = UNEXECUTED.
Nothing above Micro scale is demonstrated anywhere; nothing is TPU evidence.

## Decisions for CYR-GPU-001
1. H1 pair-preserving minibatches → query control (BRAMASTRA UNEXECUTED
   proposal; cheapest decisive test; primary both-correct + blind gap).
2. H2 hysteretic HIGH→LOW with fixed-time control (ARK-011 paradigm, first
   execution; gated on measured instability; displacement tracked to answer
   the freezing alternative).
3. H3 displacement-matched intermediate LR (separates LR from displacement;
   diagnostic value even if H2 fails).
4. H4 mixture screen: DEFERRED (lower value than H1–H3; stated, not dropped).
5. External candidates: NONE admitted (branch-derived hypotheses strictly
   more informative; one literature check recorded in PLAN.md).
6. Complete-answer contract: ADOPT (verify in Cymek, test deterministically).
