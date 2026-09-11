# ARK-020 V2 — MULTI-SKILL CONTINUAL COGNITION, SCIENTIFIC REPAIR

**Status: PREREGISTERED BEFORE IMPLEMENTATION EXECUTION. V1 (`experiments/ARK-020/`) is
preserved as an immutable historical record of the original design; this V2 is a new
experiment created because the V1 pre-execution audit confirmed five blocking defects.**

## Question (unchanged in substance)

Can one real-text-pretrained model **sequentially acquire several meaningfully different
capabilities** while retaining earlier capabilities, preserving real-text competence, and
using **adaptive capability-specific protection more efficiently than permanent
rehearsal**?

## V4 premise (verified evidence record)

`experiments/ARK-019/FINAL_RESULT_AUDIT_V4.md` records the externally audited V4 result
`GUARDIAN_CONTINUAL_PROXY_CANDIDATE` (provenance-marked: operator-returned, externally
audited; not re-audited byte-level here). Design consequences: the single-skill Guardian
question is answered at candidate strength; the open questions are multi-skill
generalization, prevention-vs-recovery, protection-cost scaling, and cross-family
protection.

## Confirmed V1 defects (each repaired in V2; evidence in PREEXECUTION_AUDIT_V2.md)

1. **V4 integration boundary** — V1 passed nested template dicts and raw factset splits
   where V4 expects flat templates and semantic task splits (`run_ark020.py:760,774`).
   Would have crashed at the first parent call on Colab. V2: flat/semantic objects +
   integration-contract tests exercising the real V4 call shapes on CPU.
2. **Skill C was not inferable** — V1's successor-cycle sealed keys were never trained
   and the prompt contained no mapping information. Information-theoretically impossible.
   V2: C replaced by **two-hop relational composition in-context** (TASK_VALIDITY_ANALYSIS).
3. **Phase order seeds unused** — V1 preregistered per-phase seeds but streamed every
   phase with the B seed (`run_ark020.py:792,796`). V2: per-phase seeds bound into arm
   and checkpoint identity; tests prove seed independence across phases.
4. **Global-vs-phase confirmation step** — V1 stored global confirmation steps; the
   1.5× ratio rule was distorted by the phase offset (slowdowns masked). V2:
   phase-relative confirmation is the primary metric; global recorded too.
5. **Exact-resume claim exceeded coverage** — V1's smoke checked model/optimizer hashes
   and telemetry rows only. V2: the smoke hashes every state class that matters
   (registry, controller, counters, phase identity, RNG, confirmations, telemetry) and a
   deliberate corruption test proves fail-closed resume.

## The four capabilities (tokens remain disjoint across skills — role documented in
DESIGN_REASONING; each has train/control/validation/sealed + robustness modes)

- **A** relational binding (V4-identical construction, splits seed 524218).
- **B** second binding family / alternative template (splits seed 524219).
- **C** **two-hop relational composition** — "Links: x gives m; … m makes y; … Trace: x
  to" → y, chaining through an intermediate that is never the answer (splits seed
  524220; full validity analysis in TASK_VALIDITY_ANALYSIS.md; chance 1/3).
- **D** inverse retrieval, value→key (splits seed 524221).

## Phases, seeds, arms (frozen)

- Parent stage per parent seed (31801, 31902): A acquired jointly with real text under
  the V4 joint gate (A qualified 3-streak + science CONTROL relative NLL ≤ 0.15 +
  validation-qualified). V4 Drive parents reused when identity-matched.
- Phase B: 2000 updates, task slots from B dose selection (V4 `DOSE_SELECTION.json`
  inherited when present; else V4-style prospective pilot 2 parents × {8,12,16}).
- Phase C: 1500 updates, 12 task slots. Phase D: 1500 updates, 12 task slots.
- Order seeds: B (429001, 429002), C (429003, 429004), D (429005, 429006) — **each phase
  streams with its own seed**, bound into checkpoint identity.
- Matched sets: (parent_seed × B-order-seed) = 4; every arm of a set shares parent
  snapshot, task streams per phase, real-text stream seeds, dose, CAP calibration.
- Arms (7): PLASTIC_HIGH, STATIC_REPLAY_1OF64, STATIC_REPLAY_1OF32, STATIC_CAP16X,
  GUARDIAN_REACTIVE, GUARDIAN_PREDICTIVE, GUARDIAN_HYBRID.

## Guardian (unchanged in structure; phase-relative metric fixed)

Per-capability levels PLASTIC → SPARSE64 → REPLAY32 → (HYBRID only) EMERGENCY_CAP16X →
de-escalation on 4-healthy streak, with a 200-update post-phase SPARSE64 floor.
REACTIVE triggers only on formal CONTROL failure. PREDICTIVE additionally triggers on
the preregistered warning: robust-min < 0.95 OR degradation ≤ −0.03 per 100 updates over
the last 3 CONTROL observations. Replay is risk-allocated (lowest robust-min first,
deterministic tiebreak), max 2 slots/update, replacing real-text capacity only. Static
arms rotate one replay slot round-robin over old capabilities regardless of health;
STATIC_CAP16X applies the per-set 16× LOW-shadow cap with no replay.

## Verdict logic (frozen; exact code in ark020_v2_core.decide)

- Formation gates first: plastic reference must qualify each new skill in ≥ 3/4 matched
  sets → else `INCONCLUSIVE_FORMATION_INSTABILITY_PHASE_{B,C,D}`.
- Interference gate: plastic sets with ≥1 old-capability CONTROL failure ≥ 2/4 OR static
  1/32 area advantage ≥ 0.15 → else `INCONCLUSIVE_LOW_INTERFERENCE`.
- Guardian success (all required): final sealed ALL capabilities qualified in ≥ 3/4 sets;
  mean all-capability sealed area within 0.10 of best static arm; **median
  phase-relative confirmation ≤ 1.5 × plastic median, per phase**; final science SEALED
  NLL within 5% of plastic per set; mean protection duty < 0.60; mean replay fraction
  < 0.05; total replay slots ≤ STATIC_REPLAY_1OF32 total.
- Verdicts: `GUARDIAN_MULTI_SKILL_CANDIDATE` / `GUARDIAN_SUPPORTED_NOT_EFFICIENT` /
  `GUARDIAN_NOT_SUPPORTED_MULTI_SKILL`; PREVENTION_SIGNAL / RECOVERY_SIGNAL /
  PREDICTIVE_ADDS_VALUE flags; cost slope reported for k = 1→2→3.

## Durability, output, launcher

Exact-resume checkpoints with full identity binding (defect #5 repair); single-writer
advisory lock; PARTIAL receipts per session; runtime calibration estimates sessions and
may not change the protocol. Drive root
`/content/drive/MyDrive/genisis-arkenstone/ARK020_V2_CONTINUAL/` (V1 artifacts untouched).
Launcher `experiments/COLAB/arkenstone_ark020_v2.ipynb` with a read-only Cell-0 resume
scan (completed arms, active arm/phase/step, checkpoint identity, SAFE ACTION).

## Claim ceiling

Controlled real-text-proxy evidence for a multi-capability continual-learning controller
candidate. No universal continual-learning law, no PRE500M/500M authorization, no AGI or
consciousness claims. Negative and mixed results are valid outcomes.
