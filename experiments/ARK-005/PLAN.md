# ARK-005 — GENERALIZATION RETENTION / CONSOLIDATION (preregistered; plan-only commit)

Gate basis: ARK-004A-R verdict B+C cancelled ARK-004B. The strongest
demonstrated unexplained phenomenon is ACQUISITION != RETENTION: 3 of 4
ARK-004A seeds decayed after sustained G90 (RETENTION_OBSERVATIONS.json:
seed 101 STABILITY_GAP 0.597 with collapse 1000 steps after G90; seed 202
STABLE). This plan is committed before any ARK-005 training.

## Question
What determines whether a generalized solution consolidates or decays under
continued training on the T2 pool?

## Competing hypotheses (one variable at a time later; first observational)
- H-OPT: continued optimization moves out of the generalized basin.
- H-WD: weight decay erodes the generalized solution.
- H-DATA: continued replay of the same 500 rows destroys transferable structure.
- H-LR: constant 1e-3 is too high post-transition.
- H-STATE: generalization is metastable and needs consolidation machinery.
Observational evidence already in RETENTION_OBSERVATIONS.json: decay onset
~1000-6400 steps after G90; no seed shows recovery after collapse (seed 101
never re-crosses 0.50) — suggesting basin exit, not oscillation (weak
H-OPT/H-LR support over H-STATE; to be tested, not assumed).

## Design (trigger-controlled; trigger identical across arms)
Trigger: first sustained G90 (3 consecutive evals >= 0.90 at the 200-step
cadence) — the same sustained rule everywhere. At the trigger:
- A CONTROL: continue ordinary training (identical to ARK-004A continuation).
- B LR-DECAY: multiply LR by 0.1 at the trigger.
- C WD-REMOVAL: set weight decay to 0 at the trigger.
- D EMA-CONSOLIDATION: maintain an EMA of weights (decay 0.999) from the
  trigger; evaluation (and only evaluation) uses the EMA weights.
Arms run to the same total step budget; fresh seeds 505 and 606 (frozen here);
model/optimizer/data identical to ARK-004A (frozen manifest 0dd930569704...).

## Metrics (already implemented in retention_observations.py)
PEAK_G, RET90, RET50, T_COLLAPSE_90/50, GENERALIZATION_AREA, STABILITY_GAP,
plus full trajectories and per-position accuracy.

## Success criteria
An arm is a consolidation candidate if it moves RET90 from the control range
(0.2-0.95, seed-variable) to >= 0.95 on both fresh seeds without reducing
PEAK_G by more than 0.05. Because ARK-004A showed ~2x seed variance in
transition timing, single-seed claims are forbidden.

## Futility rule
If no arm moves RET90 by >= 0.2 over its own control-seed pairing across both
seeds, record the null and stop the retention program at micro scale.

## What this does not justify
No AGI claim; no core promotion; no architecture change. A retention fix at
micro scale must still transfer to a second task family before any broader
claim.
