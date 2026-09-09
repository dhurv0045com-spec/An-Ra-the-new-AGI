# ARKENSTONE

**Mission:** discover mechanisms that produce transferable cognition per parameter,
per token, and per compute — knowledge no other branch contains. Not a recombination
of known ideas; a laboratory for what remains unknown.

- **Branch:** `Arkenstone`.
- **Base:** `origin/cymek` at `28bf57a0d299a2c13a99fe0046616c00a1b8530c`.
- **Central question:** what causes An-Ra to acquire, preserve, recover, and transfer exact symbolic computation rather than merely lowering token-prediction loss?
- **Transition evidence:** the T2 memorize -> delayed structural-OOD transition is replicated; memorization timing does not explain G90 timing.
- **ARK-004A-R correction:** the claimed column-selectivity precursor was directionally inverted and is retained only as a transition marker.
- **Same-task retention evidence:** ARK-007R replicated strong LOW-LR protection on three fresh Micro T2 checkpoints: HIGH `1e-3` collapse90 9/12 vs LOW `1e-5` 0/12.
- **Recovery evidence:** ARK-010 found HIGH reacquisition 8/9 vs immediate LOW 2/9 after instability.
- **Adaptive controller evidence:** ARK-011 directly supports HIGH-recover -> LOW-retain on Micro T2: 6 sealed-qualified recovery forks across 3 fresh acquisitions, HIGH recurrent instability 3/6 vs SWITCH_LOW 0/6, risk difference -0.50.
- **Threshold status:** ARK-012 does not identify an exact switch threshold; 0.85/0.90 performed best in a two-source selected screen but threshold timings alias at the 200-step evaluation cadence.
- **Plasticity boundary:** ARK-013 is inconclusive for new-skill acquisition because T3CARRY never reached G90 under HIGH. It does show LOW LR alone is not enough to preserve old T2 during 12k no-replay T3-only updates.
- **Non-arithmetic acquisition:** ARK-014 repairs ARK-009's order brittleness using deterministic order augmentation. The qualified augmented subject reached sealed ORDER_ONLY/QUERY_ORDER ~0.987 at fork. **LR-retention transfer remains NOT DEMONSTRATED** because HIGH and LOW both had 0/3 retention failures.
- **Core promotion:** NOT JUSTIFIED. Transfer, mechanism-credit, plasticity and scale gates remain open.

## Current execution target — Discovery V7

Discovery V7 is **PREREGISTERED + IMPLEMENTED + PREEXECUTION-AUDITED, NOT GPU EXECUTED**.

It contains two deliberately linked experiments:

- `ARK-015` — the user's proposed non-arithmetic transfer retry, sharpened into an order-invariance retention stress. Fresh order-augmented binding subjects are continued on canonical-only presentation at HIGH vs LOW LR, with an augmented-HIGH reference to test whether any failures are specifically caused by distribution narrowing.
- `ARK-016` — a mechanism red-team of the near-freezing explanation. From fresh recovered T2 states it compares LOW_REFERENCE, HIGH_UNCAPPED, HIGH with applied updates capped to the matched LOW per-step movement, and HIGH capped at 10x that movement.

The 300-minute Colab campaign is designed to end with `TRAINING_DESIGN_DECISION.json`, which maps the evidence to one of a small number of next-training directions:

- candidate capability-state LR controller;
- candidate applied-update trust-region controller;
- transferred retention effect with mechanism unresolved;
- arithmetic/stress-specific boundary;
- insufficient event evidence.

This is a research handoff only. It never auto-authorizes Cymek scheduler changes, PRE500M, or 500M training.

Execution artifacts:

- `experiments/ARK-015/PLAN.md`
- `experiments/ARK-015/run_ark015.py`
- `experiments/ARK-016/PLAN.md`
- `experiments/ARK-016/run_ark016.py`
- `experiments/COLAB/MASTER_DISCOVERY_V7_PLAN.md`
- `experiments/COLAB/DISCOVERY_V7_PREEXECUTION_AUDIT.md`
- `experiments/COLAB/run_discovery_v7.py`
- `experiments/COLAB/arkenstone_discovery_v7.ipynb`

The Colab launcher checks out exact executable commit `4a8ffe84223fa64a7409b7d9936f2ff80d3828df`, compiles the V7 runners and dependencies, verifies frozen task manifests, tests exact model/optimizer reload and deterministic next-update identity, and exercises the nonstandard delta-cap primitive on CUDA before the expensive campaign can begin.

## Latest validated campaign — Discovery V6

See:
- `experiments/COLAB/results/v6/DISCOVERY_V6_VALIDATED.md`
- `experiments/COLAB/results/v6/RECEIPT_AUDIT.json`
- `experiments/ARK-011/`
- `experiments/ARK-012/`
- `experiments/ARK-013/`
- `experiments/ARK-014/`

Imported Colab bundle SHA256:
`1ec3224075de49b080e229111e4c4f897430c3eca888c487116aff07a65c8d15`

The campaign reported 179.09 minutes on CUDA / torch `2.11.0+cu128`. GPU smoke passed, no failure receipt was present, and **14/14 uploaded JSON receipt hashes independently revalidated**. Internal ARK-013/014 task-manifest hashes and the ARK-011 CONTROL/SEALED assignment hash also matched.

## Cymek boundary

Arkenstone inspected live `cymek-500m-readiness` read-only while designing V7. Cymek independently identifies the same open gates: non-arithmetic transfer with useful event contrast and separation of LOW LR from near-freezing movement. Arkenstone may produce a challenger and instrumentation recommendation; Cymek remains the owner of production training and promotion.

## Current bottlenecks

1. Determine whether LOW/adaptive retention protection transfers to robust non-arithmetic binding under an informative but natural presentation-distribution shift — ARK-015.
2. Determine whether the protection is mediated mainly by applied update magnitude, or whether literal LR/optimizer trajectory matters — ARK-016.
3. Convert the result into measurement infrastructure and one bounded algorithm challenger for the next serious training run, rather than directly changing production training.
4. Scale only after transfer + mechanism-credit gates are met.

## Progress tracking

- `PROGRESS.md` — dated execution history
- `IMPROVEMENTS.md` — adopted improvements
- `FAILURES.md` — experiment failures and falsified claims
- `EXPERIMENT_LOG.md` — one line per experiment
- `MECHANISM_TOURNAMENT.md` — candidate mechanisms and verdicts
- `AGI_FEATURE_LEDGER.md` — evidence/cost/novelty/status
- `NEGATIVE_RESULTS.md` — failures preserved permanently
- `NOVELTY_REGISTER.md` — novelty classification
- `UNIFIED_EVIDENCE_MAP.md` — cross-branch evidence map
- `COGNITION_BOTTLENECK_GRAPH.md` — current dependency/bottleneck model

## Rules

Loss is a diagnostic, never proof of cognition. Execution artifacts beat prose.
Failures are preserved. Reproductions are labeled reproductions. Every claim gets
a novelty class. Historical preregistration/receipts are immutable. Branch isolation
is absolute.
