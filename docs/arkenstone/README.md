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
- **ARK-015: non-arithmetic retention transfer is DEMONSTRATED at Micro scale under controlled distribution narrowing.** Across 3 fresh acquisition parents and 8 matched pairs, NARROW_HIGH failed 8/8, NARROW_LOW 0/8, and AUGMENTED_HIGH_REFERENCE 0/8. Canonical exact remained 1.0 while order robustness eroded under narrowed HIGH training.
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

Two protection mechanisms remain plausible:

1. restrict applied movement / plasticity;
2. keep supplying evidence for the broader invariant.

Large movement alone is not a sufficient explanation because AUGMENTED_HIGH moved farther than NARROW_HIGH while preserving robustness.

## Current execution program — Discovery V8 V2

The second-pass design review is now frozen before any V8 GPU result. The program is deliberately sequential:

### 1. ARK-017 V2 — causal mechanism

Use the reliable ARK-015 event generator and independently manipulate movement and invariant-supporting data.

V2 fixes an execution defect found before running: sparse replay now guarantees exactly non-canonical alternative orders instead of allowing identity permutations. Core arms remain HIGH, LOW, HIGH_CAP1X, HIGH+1/16 replay, CAP+replay and augmented-HIGH. A preregistered secondary efficiency screen may test CAP4X/CAP16X or replay 1/32/1/64 after the primary causal verdict.

Status: **PREREGISTERED + IMPLEMENTED + STATICALLY AUDITED; READY FOR PINNED GPU SMOKE; NOT EXECUTED.**

Launcher:
`experiments/COLAB/arkenstone_ark017_v2.ipynb`

Pinned scientific runner commit:
`377c4743f8017e3455f576eafb75bb8ab9c50284`

### 2. ARK-018 V2 — ~1GB real-data continuation bridge

The user supplies a real-text file on Google Drive. The corpus is streamed, full-hash-bound and split deterministically before training. A conventional ~20–25M decoder is pretrained with exact byte/token exposure accounting.

After a controlled robust SKILL_A is acquired, the **primary retention stress is continued real-text learning**, not synthetic-only continuation. HIGH, LOW, capability-specific replay, distractor replay and at most one prospectively selected ARK-017 mechanism are compared while SEALED language NLL and capability robustness are measured together.

Status: **V2 PREREGISTERED; DATA NOT YET BOUND; NOT IMPLEMENTED/EXECUTED.**

### 3. ARK-019 V2 — real-mixture Capability Guardian

Only after ARK-017/018 evidence selects a mechanism, train with a fixed 7/8 real-text + 1/8 SKILL_B mixture while preserving SKILL_A. Compare PLASTIC_HIGH, LOW_ALL, STATIC_PROTECT, reactive Guardian and an anticipatory-margin Guardian. Controller state must resume exactly.

Status: **V2 PREREGISTERED; EVIDENCE-GATED; NOT IMPLEMENTED/EXECUTED.**

Program documents:
- `experiments/COLAB/MASTER_DISCOVERY_V8_PLAN.md`
- `experiments/COLAB/MASTER_DISCOVERY_V8_V2_ADDENDUM.md`
- `experiments/COLAB/DISCOVERY_V8_DESIGN_REVIEW_V2.md`
- `experiments/ARK-017/PLAN_V2_ADDENDUM.md`
- `experiments/ARK-018/PLAN_V2_ADDENDUM.md`
- `experiments/ARK-019/PLAN_V2_ADDENDUM.md`

## Training-infrastructure implication already justified

Intervention promotion is still blocked, but measurement infrastructure is justified. Future serious training proxies should expose:

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
