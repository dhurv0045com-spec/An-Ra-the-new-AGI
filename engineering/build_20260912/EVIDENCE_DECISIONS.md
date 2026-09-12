# Evidence reviewed and resulting build decisions

Reviewed 12 September 2026. All source identities and local byte-preserving excerpts are in [evidence/SOURCES.json](evidence/SOURCES.json). Arkenstone ref: `9c09d9d682ce3b2cd60f8e5573c84e04922ec8a8`. Cymek was refreshed from the remote to `b850861545f79e219b55d6403f81e74f92f1592e`; its local worktree at `fadb835` was stale and must not be treated as the newest evidence.

## Findings and limits

| Finding | Evidence and scope | Build decision |
|---|---|---|
| Arkenstone LOW LR preserves a previously acquired state | ARK-007R: 9/12 HIGH continuation collapses versus 0/12 LOW, but only three independent acquisitions; displacement approximately .379 versus .008 | Separate preservation from acquisition; log plasticity and displacement so near-freezing cannot masquerade as improvement |
| LOW LR is not a general recovery policy | ARK-010: recovery 8/9 HIGH versus 2/9 LOW from selected collapse states | Controller requires an explicit REACQUIRE state; never implement `collapse -> lower LR` as a universal rule |
| Adaptive hysteresis itself remains unproven here | ARK-011/011S/012/013 have runners/plans, not scientific result receipts in inspected source | Build policy interface and deterministic state-machine tests; keep controller off by default until later qualification |
| Production representation can memorize without formation | CYR-GPU-011: compact 56.47% held-out exact at 44.89% exposure; production 0% at full 1,152,000 rows, 0/48 sealed; physical parameter counts differ | Make representation, output-class count, free generation and answer/EOS objective inspectable; do not scale based on training loss |
| Smaller vocabulary is not monotonically better | R1: V19 12.94%, V4096 100%, V24576 0% in one controlled seed at 512k rows | Physical embedding/output cardinality is explicit configuration, not an undocumented tokenizer side effect |
| Intermediate benefit is seed-sensitive | R1B: V4096 .5059/.4941, V8192 .7176/0, V16384 .6471/.1294, V24576 .0118/0 at 128k rows | No universal `4096 is optimal` setting; preserve full-vocabulary baseline and mechanism telemetry |
| Training softmax competition is a live hypothesis | R1C holds physical V24576 fixed; masks/offsets are proposed. Latest change repairs clipping tolerance after a launch failure | Implement training-only treatment as opt-in and state-bound; no causal success claim, no masked primary inference |
| Numerical checks can reject a valid update | R1C launch norm 1.0000042915 exceeded a 1e-6 tolerance; tolerance revised to 1e-4 in that implementation | Compare clip certificates with documented dtype/operation tolerance; reject genuine large breaches/nonfinite values; test both sides |
| Readiness is not 500M training evidence | Cymek receipts cover bucket/cursor/mixture/resume plumbing; 6GB frozen microstep does not fit; TPU certification remains absent in inspected records | Port behavioral contracts, not a readiness badge; tiny build must use the real training path |
| BRAMASTRA depth-two teaching is inconclusive | D02: 61/92 vs61/92 and63/92 vs65/92; only23 development mechanisms | Preserve teacher control as a diagnostic. Integrate usable learner/runtime before another tournament |

## Quality of evidence

ARK-007R/010 raw compact result files were inspected. Cymek V11 has a compact structured receipt; R1/R1B were inspected as committed result reports carrying original ZIP hashes. Their raw returned ZIPs were not independently available in this review. Treat the quantitative claims at that provenance level. A copied report is not independent replication. The cached historical `origin/arkenstone-ark011s-screen` ref remains locally visible but its remote name no longer exists; do not describe it as a newly fetched result.

The Arkenstone V6 preflight reports four optimizer updates plus identity/plumbing checks. It is useful engineering evidence, not a new capability result. The commuted arithmetic probe in Cymek V11 changes operand-role OOD membership, so its 100% result is not clean commutation invariance.

## What we adopt now

Adopt exact parent/optimizer/sampler provenance, real complete-answer scoring, public-only model inputs, single-writer checkpoint publication, update-boundary gradient reduction, target-count normalization, and explicit formation/retention telemetry. These are engineering contracts with direct utility.

Build the new formation/preservation/reacquisition controller and representation-treatment modules behind configuration switches. They are original integration choices informed by the evidence, not accepted improvement recipes. Keep full-loss and fixed-schedule controls usable through the same model/trainer.

Do not import tournament scripts as the training system, copy all of `v5_*`, hard-code benchmark answer sets at inference, use sealed feedback for scheduling, or advertise a 10x gain. The 10x ambition becomes a later measurable retained-transfer-per-cost comparison; this build provides the machinery needed to test it.
