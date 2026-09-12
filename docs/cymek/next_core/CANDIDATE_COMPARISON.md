# CANDIDATE COMPARISON

**Scoring (explicit, 1–5; 5 best unless noted):** evidence support (ES), expected capability impact (CI), simplicity (S), causal interpretability (CIx), training stability (TS), compute economy (CE — 5 = cheapest), memory economy (ME — 5 = cheapest), implementation risk (IR — 5 = lowest risk), evaluation risk (ER — 5 = lowest), reversibility (R), unresolved-assumption load (UA — 5 = fewest). WEIGHTED = Σ with weights (ES .20, CI .15, S .15, CIx .10, TS .10, CE .05, ME .05, IR .05, ER .05, R .05, UA .05). These are decision-model estimates, not measurements.

| Criterion (weight) | A: V5.1 conservative | B: representation-aware | C: alternative-explanation |
|---|---:|---:|---:|
| evidence support (.20) | **5** | 1 (needs R1C) | 1 (needs R1C-world-C) |
| capability impact (.15) | 3 | 5 *if* mechanism holds | 4 *if* geometry holds |
| simplicity (.15) | **5** | 4 | 3 |
| causal interpretability (.10) | **5** | 4 | 4 |
| training stability (.10) | **5** | 3 (train/inference mismatch) | 4 |
| compute economy (.05) | **5** | 4 | 5 |
| memory economy (.05) | **5** | 5 | 4 |
| implementation risk (.05) | **5** | 3 | 3 |
| evaluation risk (.05) | **5** | 2 (dual endpoints mandatory) | 3 |
| reversibility (.05) | **5** | 4 | 4 |
| unresolved assumptions (.05) | **5** (none added) | 3 | 3 |
| **WEIGHTED** | **4.90** | **3.35** | **3.20** |

## Sensitivity analysis

- Any weighting that makes "evidence support" < 15% of the score can promote B/C — i.e., the winner is stable **iff** evidence remains the dominant criterion, which is the founding law of this program (§4: architecture must pay rent; §34: no architecture theater).
- If R1C returns World A **and** CS-TRANSFER-001 confirms transfer, B's ES jumps 1→4 and CI becomes demonstrated: then B (as the R1C-justified output treatment on the unchanged V5 block) legitimately overtakes A. This is the designed handoff in ARCHITECTURE_DECISION_TREE.md, not a scoring trick.
- **There is NO STABLE ARCHITECTURE WINNER between B and C today** — both depend on R1C. There IS a stable winner overall: **A**, because it is the only candidate whose evidence support does not depend on experiments that have not run. The comparison therefore selects A and encodes B/C as conditional successors.

## Comparisons against the control (§33)

Control = CURRENT V5 + mandatory corrections (identical to A). B and C differ from the control **only in output-space fields**, each justified by exactly one pending experiment. Neither claims block-level superiority; neither can — no block-level bottleneck has evidence (BOTTLENECK_MODEL.md rank 8).

## Unresolved assumptions per candidate

- A: that representation/data/evaluation bottlenecks (ranks 1–3) can be closed without touching blocks — monitored, falsifiable by any future isolated block-level bottleneck.
- B: that R1C's verdict transfers beyond the micro task (CS-TRANSFER-001 gates it twice).
- C: that the class-space effect is mechanism-robust at all (R1B's seed sensitivity is the standing warning).
