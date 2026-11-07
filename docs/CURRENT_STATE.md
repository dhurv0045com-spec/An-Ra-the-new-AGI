# Current state

V5 implements a versioned foundation beside the accepted V4 reference. The symbolic codec and checkpoint identities are preserved. A separate lossless UTF8 byte path adds Unicode framing, offsets and explicit vocabulary migration. A validated neural backend serves both the original frozen cognitive policy and new action-policy decoding. The new trainer supports variable lengths, exact weighted gradient accumulation, immutable recipes, complete-epoch curricula, FP32/optional FP16, explicit stage migration and decoded transactional recovery.

The bounded eight-update QUERY/EOS overfit control scored 4/4 on its four training prompts; loss fell 3.8082→0.7463 in 0.7044 measured seconds excluding startup. This establishes a trainable token path, not when-to-query competence. On ten new validation worlds the engineered fixed policy is 10/10 with four requests, always-query 10/10 with eight, no-query 6/10. Raw outcomes, cost traces, hashes and the failed earlier LR .01 canary are preserved. [Qualification](FOUNDATION_QUALIFICATION.md) · [Raw evidence](../artifacts/elendril/foundation-v5)

V4 remains **MECHANICS_QUALIFIED**. The frozen ELD-DEV-004 checkpoint scored route-memory 4/4, no-memory 0/4, irrelevant same-token context 1/4 and status-aware revision 0/4. Retrieval was fixed, not learned. It never queried. Its original weights and all four V4 receipts remain intact. [Frozen scorecard](COGNITIVE_SCORECARD.md)

Native-GQA execution passed packed forward/backward parity but was 1.169× slower in the small CPU probe, so repeated-KV SDPA stays default. CUDA performance is unknown. The new UTF8 vocabulary has 260 entries versus 46 legacy entries; byte tokenization is not a pretrained language model or an efficiency result.

Historical custody checks reverified 78 evidence hashes, 63 receipts, 519 scoring snapshots, 237 manifests and 1,659 checkpoint components. Older S5 custody gaps remain unresolved. V3 graph wrappers and duplicate plans are now archived by exact Git object. They were not relabeled as cognitive notebooks.

Next decision: owner-run multi-action formation canaries, durable recovery in a different VM session, then a gated CE/weighted-CE/curriculum screen. **Research training, live CUDA and cloud persistence remain BLOCKED_PENDING_GPU.** No local research campaign ran. [Next experiment](NEXT_COGNITIVE_EXPERIMENT.md)
