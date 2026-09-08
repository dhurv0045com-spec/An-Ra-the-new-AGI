# Discovery V6 — validated imported campaign

## Provenance

User-supplied Colab bundle: `ARKENSTONE_DISCOVERY_V6_RESULTS.zip`

- bundle SHA256: `1ec3224075de49b080e229111e4c4f897430c3eca888c487116aff07a65c8d15`
- bundle size: 162,312 bytes
- JSON receipts in bundle: 14
- canonical receipt hashes independently revalidated: **14/14 PASS**
- failure receipt present: **NO**
- GPU smoke receipt: **PASS**
- runner commit: `2238f5519c3225f44010e74ba6d1af9f31d8524f`
- master preregistration: `f46bae74c783821452947f3927acddbc14cc6dbf`
- device: CUDA
- torch: `2.11.0+cu128`
- program-reported runtime: **179.09 minutes**

Internal task-manifest hashes for ARK-013 and ARK-014 were independently recomputed and matched. ARK-011's CONTROL/SEALED assignment hash was independently recomputed and matched.

## Campaign verdicts

| experiment | validated outcome |
|---|---|
| ARK-011 | **SUPPORTED_ADAPTIVE_PROTECTION** — 6 sealed-qualified recovery forks across 3 fresh acquisitions; HIGH recurrent instability 3/6 vs SWITCH_LOW 0/6; paired risk difference -0.50 |
| ARK-012 | **TIME_NOT_STATE_SCREEN** — selected-event threshold screen; 0.85/0.90 had best mean sealed area, but thresholds aliased and did not order monotonically; no exact threshold law |
| ARK-013 | **INCONCLUSIVE_NEW_SKILL_NOT_ACQUIRED** — FIXED_HIGH never acquired sustained T3 G90; adaptive never switched. Secondary boundary: LOW did not preserve T2 under 12k no-replay cross-task T3 training |
| ARK-014 | **ORDER_ROBUSTNESS_REPAIRED** for acquisition; **ROBUST_BINDING_ACQUIRED_BUT_TRANSFER_INCONCLUSIVE** for LR transfer — order augmentation qualified, but HIGH and LOW both had 0/3 retention failures |

## Program interpretation

Discovery V6 strengthens the state-dependent LR story on the original Micro T2 domain: HIGH is useful for movement/recovery and LOW is useful after recovered capability is present. It simultaneously exposes two limits: exact switch-threshold identity is unresolved, and LOW LR alone is not a general solution to cross-task interference.

The non-arithmetic program bottleneck changed: robust binding acquisition is now available through order augmentation, but the LR-retention transfer effect is still **NOT DEMONSTRATED** because the qualified non-arithmetic subject generated no retention failures.

No result in this campaign authorizes a universal optimizer law, a Cymek production scheduler change, or an AGI claim.

See `experiments/ARK-011..014/{RESULT.json,ANALYSIS.md,REDTEAM.md,NOVELTY.md}` for experiment-level records.
