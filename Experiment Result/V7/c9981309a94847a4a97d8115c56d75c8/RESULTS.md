# V7 result: training order matters; composition remains weak

Owner run `c9981309a94847a4a97d8115c56d75c8`, source `413a3eccf4344ef2a39051b870e92feb6271b769`. All nine registered runs completed four epochs: 3,084,800 parameters, 2,368 updates, 75,776 supervised tokens and 8,914,404 unpadded input positions per run. This matches exposure, not exact padded compute. No model was retrained during this analysis.

## Correctness and custody

The supplied final status says **COMPLETE**, delivery **PASS**, both GPU canaries and recovery checks **PASS**. The supplied analysis ZIP exactly matches its SHA-256: `3d83558f735dd958381f7f80d79b379edb5fb6b5ab47b48accf65cfbdfbda3a2`. ZIP contents and source/manifest identities were checked locally. Independent recorded-action replay passed **158 records**, using matching, audited procedural datasets retained from preparation. This verifies recorded actions/outcomes, not fresh neural inference.

The full checkpoint ZIP was not supplied: its integrity/recovery result is reported by the owner receipt, not independently rechecked here. Cross-Python entropy reduction differed in 124 derived fields by at most **1.33e-15**. A documented tolerance was applied only to entropy in copies for comparison; original evidence and strict raw-action checks were preserved.

## Confirmation results

Each cell is correct episodes out of the **same 112 confirmation worlds**. Means average three model seeds; these are not 336 independent worlds.

| Method | Seed 7 | Seed 19 | Seed 41 | Mean accuracy |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 91/112 | 68/112 | 48/112 | 61.6% |
| Blocked | 83/112 | 93/112 | 94/112 | **80.4%** |
| Interleaved | 44/112 | 94/112 | 55/112 | 57.4% |

Blocked improves over uniform in seeds 19 and 41, but loses in seed 7. It is the strongest baseline candidate here, not a universally better algorithm or a statistically established general result. Interleaving does not provide a reliable improvement.

## What worked

- All nine final models queried **32/32 query-needed cases** and correctly abstained on **36/36 null targets**. These two behaviors are reliable in this task distribution; broader action choice remains incomplete.
- Changed-payload pair tests require both original and altered answers to be correct. Across the three model seeds, blocked scores **73/96 pairs**, uniform **26/96**, interleaved **34/96**. Interleaved seed 19 alone scores 32/32. Some models genuinely use changed evidence behaviorally; success is highly seed-dependent.
- Blocked memory cases score **44/48 model-case evaluations** and missing-evidence cases **48/48**. These are narrow benchmark outcomes, not broad or independently causal memory capabilities.

## Failures and first divergences

- **Composition is the clearest residual weakness:** blocked routing scores only **10/48**, despite memory 44/48. Known-value cases also remain imperfect at 32/48. Uniform and interleaved routing each score 5/48.
- **Answer learning remains unstable.** At epoch two, uniform seeds 7 and 41 answer every terminal value with I and N respectively. By epoch four their development non-null accuracy improves to 49/71 and 15/71. Collapse can diminish with training, but extra training is not a reliable cure.
- **Querying alone is insufficient.** All final models query when needed, yet final non-null correctness ranges from 8/76 to 58/76. Models also make 10–11 unnecessary queries per 112 confirmation cases. Supplying payloads directly does not broadly rescue weak models.
- Across **386 observed-query comparisons**, the subsequent prompt is identical to the supplied-payload prompt, and paired correctness never differs. These stateless conditions do not identify a distinct post-query neural-conditioning failure.
- Even the 28-world training monitor is incompletely fitted: final non-null correctness ranges from 2/14 to 10/14. This is not simply a demonstrated train-perfect/test-poor failure; the monitor is small and differs in difficulty from confirmation.
- Concrete blocked-seed-19 examples: a route case expects F but immediately answers G; a known-value case expects B but answers H; a missing-evidence case correctly performs QUERY then ANSWER I. Case IDs, prompts and decisions are preserved in `METRICS.json`. These are selected illustrations, not new aggregate evidence.

Engineered controls provide context: fixed/oracle/always-query score 112/112; no-query 84/112; recency 64/112. They are scripted competitors, not parameter-matched neural models. Some trained models remain below the engineered no-query score.

## Conclusion and next experiment

**CAUSE UNRESOLVED.** Training order materially changes the observed trajectories, but this study does not isolate the neural cause of collapse, prove catastrophic forgetting, or demonstrate forward transfer. Blocked development scores can decline slightly from epoch two to four; that observation alone is not a retention mechanism.

Scope is limited: three seeds, shared synthetic task authorship and a fixed 16-symbol alphabet. The payload-swap generator chooses the first different symbol, predominantly A/B; it does not balance alternate labels. Narrow pair success therefore needs broader, balanced counterfactual confirmation.

The next focused experiment should isolate routing: on frozen checkpoints from all three seeds, compare matched one-hop/two-hop worlds with a correct public intermediate-state supply versus an equal-budget sham. Use new independently generated held-out worlds; do not tune on this confirmation set or provide the final answer. This distinguishes edge/value binding from composition more directly. Follow with a conventional pointer/copy competitor if the evidence warrants training; do not scale model size merely to chase the best seed.

Recorded controller wall time is **61.37 minutes before final reporting/export**; summed worker reservations are **109.41 GPU-minutes**, including worker training/evaluation/recovery overhead, not pure kernel time. Final export overhead is excluded from the recorded controller wall measure.

`METRICS.json` preserves the measurements, trajectories and selected cases; `PROVENANCE.json` preserves essential source/archive hashes and verification status. Original owner-supplied receipts and ZIPs stay outside the current Git tree. Reproduction logic is maintained in `tools/analyze_elendril_v7_owner_results.py`, accepting externally retained input files.
