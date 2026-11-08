# Active hypothesis register

Current V7 priority is to diagnose the observed V6 failures before increasing model size. [Fixed design](../research/v7/PROTOCOL.md) · [Evidence](../research/v7/EVIDENCE.md)

| V7 hypothesis | Discriminating observation / falsifier | State |
| --- | --- | --- |
| Query choice limits otherwise usable evidence | Supplied public payload improves non-null outcomes and changed-payload pairs; no gain weakens this explanation | Implemented intervention; owner GPU execution pending |
| Value-answer collapse reflects poor acquisition rather than query choice alone | Fixed training monitor also fails while terminal answers concentrate; successful fitting with held-out failure instead points toward generalization | Training/development trajectories and value/EOS probes implemented; pending |
| First-epoch ordering causes an acquisition/retention tradeoff | Query/value trajectory rises then falls in blocked ordering more than uniform/interleaved across paired seeds; no reproducible difference falsifies this specific lead | Same multiset, four-epoch horizon, three seeds preregistered; pending |

The historical register follows; its pending statements are not new positive results.

| Priority | Hypothesis and unchanged control | Falsifier | Status |
| --- | --- | --- | --- |
| 1 | CE can learn costed ANSWER/QUERY/ABSTAIN in the existing episode loop; compare fixed, oracle, no-query, always-query, random, query-blind and recency | Formation floor or query-cost gate fails after positive overfit control | Dataset, objective, verifier and tiny token control qualified; full neural formation pending GPU. |
| 2 | Action-token weighting improves complete actions over CE at equal examples/updates | No prospective utility improvement or a state-kind regression | Implemented and denominator-tested; comparative result pending. |
| 3 | Uncertain-world first-epoch ordering improves formation over the identical row multiset in shuffled order | No utility gain, redundant queries, or accuracy regression | Implemented and exposure-tested; comparative result pending. |
| 4 | Native-GQA execution improves resource use at equivalent gradients | Parity fails or measured cost worsens | CPU parity passed, measured latency worsened; default unchanged. CUDA unresolved. |
| 5 | Versioned UTF8 representation enables broader inputs without corrupting legacy checkpoints | Round-trip, manifest, framing or migration fails | Engineering tests passed; language learning and representation efficiency untested. |
| Later | Retention and genuinely new-skill forward transfer can coexist at matched cumulative work | Failed new-skill acquisition, lost old skill or gain explained by extra exposure/spend | Blocked on independently qualified skills. |

See [frozen gates and budgets](ALGORITHM_COMPARISONS.md). Curriculum prioritizes missing/conflicted-world examples including their observed post-query responses, not hidden truth. Cloud formation uses fresh weights intentionally and cannot establish retention of the old route checkpoint.

V6 extends these questions using raw public observations rather than derived status codes. The source-pinned [registry](../anra_v5/elendril/v6_registry.py), [15-panel package](../research/v6/ELENDRIL_V6_EXPERIMENT_REGISTRY.json) and [prospective amendment](../research/v6/IMPLEMENTATION_AMENDMENT_2026-11-07.md) specify exact exposures, denominators, gates and differences. All comparative outcomes remain pending owner GPU execution.

| V6 question | Controlled test and falsifier |
| --- | --- |
| Does action formation survive real evidence revision? | Costed complete episodes with query-blind/payload-swap, time-order and row-order controls. Correct first actions without correct post-query revisions fail the stronger claim. |
| Does augmentation improve semantic invariance? | A4 preserves ordered latent worlds while changing templates/aliases; OOD alias/template/distractor/combined panels. No stable held-out gain or subgroup regression rejects benefit. Byte/work differences are charged. |
| Does generated intermediate reasoning improve depth? | A5 generates its own scratch; compare unchanged CE and direct controls on composition and three-hop cases. No improvement after accounting for added supervision/compute rejects the proposed advantage. |
| Does retrieval guide a frozen learner on new instances? | Memory off/prefetch/irrelevant/swap interventions and before/after weight identities. Success explained by the controller or irrelevant context rejects learned memory-use attribution. |
| Does representation change help at comparable input semantics? | R0 legacy-ASCII versus R1 UTF8-ASCII with identical worlds/rendering, explicit parameter/token/position differences. Broader raw UTF8 arms cannot isolate tokenizer benefit. |
| Does scale address the measured bottleneck? | S8 winner versus unchanged CE at matched epochs/seeds after S3 gates; S20 requires both-seed S8 signal and available budget. Scaling without formation or reproducible signal is skipped. |
| Can acquired skills support retention and genuinely new C learning? | P15 requires separately qualified A/B acquisitions and matched new-C work curves. No such chain is scheduled here; this panel remains NOT_QUALIFIED rather than treating repeated aliases as new skills. |

V3 surface-invariance, recurrent-composition, intermediate-memory and later adaptation hypotheses remain preserved at [their exact V4 archive](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/f516e0bdb8f9137532b35c0c23299f7f9dbdf185/docs/HYPOTHESIS_REGISTER.md). Historical negatives are unchanged.

The investigator-designed S16–S19 supplement probes decision-budget sensitivity, consistent global symbol remapping, stale-memory interference and query-price robustness. These reuse the frozen checkpoints and preserve the original P01–P15. Exact cohorts, falsifiers, limitations and resulting engineering decisions are specified in the [prospective supplement](../research/v6/INVESTIGATOR_SUPPLEMENT_2026-11-07.md).
