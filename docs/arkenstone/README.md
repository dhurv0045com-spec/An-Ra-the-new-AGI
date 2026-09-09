# ARKENSTONE

**Mission:** discover mechanisms that produce transferable cognition per parameter, per token and per compute, then turn only well-supported mechanisms into bounded training challengers.

- **Branch:** `Arkenstone`
- **Base:** `origin/cymek` at `28bf57a0d299a2c13a99fe0046616c00a1b8530c`
- **Central question:** what causes An-Ra to acquire, preserve, recover and transfer useful computation rather than merely lowering token-prediction loss?

## Current evidence state

- T2 memorize→generalize transition is replicated; memorization timing does not predict G90 timing.
- ARK-007R: same-task Micro T2 LOW-LR protection replicated, HIGH `1e-3` failure 9/12 vs LOW `1e-5` 0/12.
- ARK-010: after instability, HIGH reacquired sustained G90 8/9 vs immediate LOW 2/9.
- ARK-011: HIGH-recover→LOW-retain adaptive switch directly supported on Micro T2, HIGH recurrent instability 3/6 vs SWITCH_LOW 0/6.
- ARK-012: no exact universal switch threshold identified.
- ARK-013: LOW is not a general no-replay cross-task forgetting solution.
- ARK-014: deterministic order augmentation repaired robust non-arithmetic binding acquisition.
- **ARK-015: non-arithmetic retention transfer now DEMONSTRATED at Micro scale under controlled distribution narrowing.** Across 3 fresh acquisition parents and 8 matched pairs, NARROW_HIGH failed 8/8, NARROW_LOW 0/8, and AUGMENTED_HIGH_REFERENCE 0/8. Canonical exact remained 1.0 while order robustness eroded under narrowed HIGH training.
- **ARK-016: mechanism credit remains unresolved.** All 3 fresh T2 parents acquired, but only 1/12 continuation opportunities produced a qualifying collapse→recovery fork, so the update-cap comparison was inconclusive.

Validated Discovery V7 bundle SHA256:

`25c4ac0aa01478cb2067147a240a12b4e30516810dad465e619391ff8a6f7faf`

V7 reported 166.24 minutes on CUDA / torch `2.11.0+cu128`; GPU smoke passed; no failure receipt was present; **11/11 JSON receipt hashes independently revalidated**.

See:

- `experiments/COLAB/results/v7/DISCOVERY_V7_VALIDATED.md`
- `experiments/COLAB/results/v7/RECEIPT_AUDIT.json`
- `experiments/ARK-015/{PLAN.md,RESULT.json,ANALYSIS.md,REDTEAM.md,NOVELTY.md}`
- `experiments/ARK-016/{PLAN.md,RESULT.json,ANALYSIS.md,REDTEAM.md,NOVELTY.md}`

## Current causal picture

ARK-015 shows that continued specialization can narrow a capability without ordinary task accuracy exposing the damage:

`ROBUST INVARIANT CAPABILITY --(HIGH plasticity + narrowed presentation support)--> CANONICAL skill retained, INVARIANT eroded`

Two known protections both remain plausible:

`reduced plasticity / movement`

and

`continued support for the broader invariant`

Large movement alone is not a sufficient explanation because AUGMENTED_HIGH moved farther than NARROW_HIGH while preserving robustness.

## Current execution program — Discovery V8

Discovery V8 is a three-stage promotion funnel, preregistered before V8 implementation:

1. **ARK-017 — mechanism factorial.** Reuse the reliable ARK-015 failure regime and independently manipulate LOW-matched applied-update caps and 1/16 invariant-supporting replay. Goal: identify movement, data support, or their interaction.
2. **ARK-018 — 1GB real-data substrate bridge.** The user supplies ~1GB real text on Google Drive. The file is full-hash-bound before training; a ~20–25M decoder proxy is pretrained with exact exposure accounting, then the controlled invariant-capability retention test is repeated while held-out language NLL is measured.
3. **ARK-019 — Capability Guardian.** Starting from the real-data substrate, compare PLASTIC_HIGH, LOW_ALL, STATIC_PROTECT and a closed-loop controller that activates the prospectively selected protection only when CONTROL probes detect old-capability erosion while a new capability is learned.

Master plan:

`experiments/COLAB/MASTER_DISCOVERY_V8_PLAN.md`

Individual plans:

- `experiments/ARK-017/PLAN.md`
- `experiments/ARK-018/PLAN.md`
- `experiments/ARK-019/PLAN.md`

## Training-infrastructure implication already justified

Intervention promotion is still blocked, but measurement infrastructure is now justified. Future serious training proxies should expose:

- capability probe registry with explicit CONTROL vs SEALED roles;
- robustness/invariance probes in addition to ordinary held-out loss;
- raw and applied update norms;
- cumulative parameter path and milestone displacement;
- data-regime / replay state identity;
- acquire / stable / eroding / recovered / protected state events;
- exact resumable controller state;
- old-skill/new-skill metrics across distribution shifts.

## Cymek boundary

Arkenstone may inspect Cymek read-only and hand off a research challenger. **No Arkenstone result currently authorizes a Cymek production scheduler change, PRE500M, TPU promotion or 500M training.** Cymek remains the production substrate and must independently qualify any promoted challenger.

## Program rules

Loss is a diagnostic, never proof of cognition. Execution artifacts beat prose. Failures are preserved. Reproductions are labeled reproductions. Every claim gets a novelty class. Historical preregistration/receipts are immutable. Branch isolation is absolute.
