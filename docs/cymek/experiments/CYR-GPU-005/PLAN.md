# CYR-GPU-005 — PLAN

Status: PREREGISTERED, PENDING_OPERATOR_EXECUTION (hash-bound in
`PREREGISTRATION.json`; this plan is frozen in the executable commit).

## Objective

Measure what happens to THE SAME acquired capability under four
registered continuation policies, with fork identity (same parent bytes,
same future stream) guaranteed mechanically. Decide whether a
state-dependent LR policy beats fixed policies on retention, without
mistaking near-freezing for consolidation.

## Prior evidence (audited this cycle; raw receipts over docs)

- Arkenstone @ `fc3e689`: ARK-007R (LOW protects: 0/12 vs 9/12 collapse,
  risk difference −0.75), ARK-010 (recovery: HIGH 8/9 vs LOW 2/9),
  ARK-009 (transfer gate NOT qualified; order-reversal confound),
  Discovery-V6 (14/14 receipt hashes valid).
- BRAMASTRA @ `4655733`: chief D02 horizon audit; discovery_dev_701
  parent/child comparisons.
- Local: closure receipts, 263-test green at the audit base; CYR-GPU-004
  SUPERSEDED_BEFORE_EXECUTION (see `../CYR-GPU-004/SUPERSEDED.md`).

## Hypotheses

- H-primary (state-dependent): a hysteretic HIGH↔LOW controller retains
  acquired capability at LOW-like levels while re-acquiring at HIGH-like
  levels; its RET90 area beats both fixed arms on qualified parents.
- H-fixed-time: switching HIGH→LOW at a fixed fraction of actual
  continuation tokens captures most of LOW's protection with most of
  HIGH's headroom; if it matches HYST, the "state" in state-dependent
  policy is mostly TIME, not capability state.
- H-null: all four arms are statistically indistinguishable on qualified
  parents — protects the 500M recipe from an unnecessary controller.
- Strongest alternative to the LR framing: LOW's protection is parameter
  near-freezing; adjudicated by the displacement/moment red-team ledger,
  not by another arm.

## Model, data, controls

- Model: REAL Cymek V5 via `ModelSpec` + `v5_model.core.initialize()`;
  scale chosen by the hardware-only resolver from the canonical registry
  (MIDI preferred; MICRO / RESEARCH_SMALL downshift before dose cuts;
  P35 only as a larger finalist stage if the resolver affords it).
- Task: T2 two-digit addition, first-operand tens bands 1–5 train vs
  6–7 eval (ordered-question holdout; canonical closure declared).
- Splits: train 500 · DEV_CONTROLLER 96 (controls G90 + hysteresis) ·
  DEV_MEASUREMENT 112 (passive trajectory) · SEALED_RESERVED 48 (never
  consulted during the run) · train probe 16 (M99). Manifest SHA over
  the exact rendered rows; leak audit fail-closed.
- Tokenizer: frozen 24,576 artifact; special IDs read from the artifact.
- Loss: content + EOS (EOS carries loss); PAD never; candidate-free
  generated scoring with the five preregistered rates.

## Seed and fork strategy

- 3 independent acquisition parents (707/808/909), exactly ONE
  acquisition each, actual-token targeted (2–6M).
- G90_CONFIRMED = candidate-free generated complete-exact ≥ 0.90 on
  DEV_CONTROLLER, sustained 3 consecutive evaluations. Unqualified
  parents fork NOTHING (`parent_status=NOT_QUALIFIED`).
- 4 forks per qualified parent from THE SAME checkpoint bytes:
  HIGH_CONTINUE (1e-3), LOW_CONTINUE (1e-5),
  FIXED_TIME_HIGH_TO_LOW (switch at τ=0.5 of ACTUAL continuation
  tokens), HYSTERETIC_HIGH_LOW (enter retention ≥ 0.90 sustained 3,
  re-enter plasticity < 0.50 sustained 3; observes BOTH states).
- Continuation 0.5–2M actual tokens per arm, identical across arms;
  arm order counterbalanced by parent seed (preregistered rotation);
  arm-local RNG seeds are fixed offsets (`ARM_SEED_OFFSETS`), the data
  stream is shared.

## Hardware resolver and wall budget

- CELL 0 calibrates the real proxy + tokenizer + batch builder +
  AdamW; the resolver sees ONLY GPU/VRAM/throughput/wall — never
  accuracy or loss — and writes `RESOLVED_PREREGISTRATION.json` before
  training.
- One absolute campaign deadline (`monotonic_start + wall budget`);
  every loop checks it; the deadline leaves packaging margin. Target
  120–150 min; hard max 175 min.

## Primary metric and decision

- Primary: RET90 area over DEV_MEASUREMENT trajectory per arm.
- A comparison exists only if every compared arm is COMPLETE, reached
  its actual-token target, shares the valid parent, shares the future
  tail, and passes the red team — otherwise INCONCLUSIVE (recorded,
  not buried).
- SEALED_RESERVED is scored once, after DECISION.json is written, and
  never selects winners.

## Red team / diagnostics

- Per evaluation: parameter displacement (L2 + relative), Adam
  moment norms, gradient exposure split (HIGH/LOW tokens), lr trace.
- Section-62 self-red-team is re-run at packaging; any uncertain answer
  blocks `ready=true`.

## What changes our mind

- New Arkenstone/BRAMASTRA results fetched at freeze time supersede the
  trajectory above only if their receipts verify; anything else is
  appended as evidence, not absorbed as belief.
- If the resolver cannot afford the dose floor, the campaign refuses to
  run rather than shrinking the science.
