# ARK-021 — RETENTION VS REACQUISITION (draft preregistration)

**Status: DEVELOPMENT_READY (core + tests implemented). Full campaign
BLOCKED_ON_ARK020V4_RESULT — probe states parameterize on V4's verified controller.**

## Question

When Guardian-protected capability A stays accurate under continuation, is it PRESERVED
(present before any corrective exposure) or RECONSTRUCTED (lost, then rapidly relearned
by replay)? V4's result cannot distinguish these; the distinction decides how every
continual-learning result in this program is read.

## Design (simplest that separates the hypotheses)

Matched arms from identical A-parents:

1. PLASTIC_DESTROY — no replay (loss curve reference);
2. DORMANT — replay withheld entirely (persistence test);
3. REPLAY_ALWAYS — static 1/32 (protection reference);
4. GUARDIAN (V4 controller) with **probe interruptions**: at frozen probe steps,
   replay is withheld for exactly 1 evaluation window and A is evaluated immediately —
   before any recovery is possible — then replay resumes.

Probe-time sealed A evaluation is the discriminator: PRESERVED (probe accuracy ~
replay-on accuracy), RECONSTRUCTED (probe accuracy ~0 with fast post-replay recovery),
MIXED (partial retention + reconstruction). Recovery latency and hidden-state
similarity to the parent on A-probes are secondary diagnostics.

## Preregistration draft (frozen numbers marked DRAFT until V4 result exists)

- probe steps: every 200 updates from 200..2000 (DRAFT);
- probe evaluation: SEALED A battery, no optimizer updates between probes and resumes;
- classification: PRESERVED if mean probe robust-min >= 0.85 while replay withheld;
  RECONSTRUCTED if < 0.30 and post-replay recovery <= 100 updates; MIXED otherwise;
- arms share parent, streams, task exposure; SEALED never controls probing.
