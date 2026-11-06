# Correctness qualification

| Gate | Result | Scope |
| --- | --- | --- |
| G0 evidence and custody | Scoped reconciliation complete. Historic checkpoint recovery open. | SHA-bound compact records and negative results. Original S5 states unavailable in reviewed custody. |
| G1 model and training mechanics | Pass. | Tested reversible vocabulary, prompt/target alignment, exact one-token shift, terminal target, causal non-leakage, finite outputs and clipping, parameter movement and both Adam moments. The model's executable inventory has 42,512 parameters. |
| G2 exact restart | Pass on the pinned CPU runtime. | In-process comparison tested fail-closed identity, missing/corrupt bytes, stale writer and interrupted staging. Separate process restart at update 3 of 8 reproduced the six state components, all per-update telemetry and final free-generation predictions. Checkpoint chain manifests intentionally differ. |
| G3 development scoring | Pass for fixture and metric controls. | Oracles, deliberate failure, malformed terminal outputs, missing/duplicate IDs, split refusal, query-blind and recency controls, and independent receipt replay. Confirmation and sealed validation remain open. |
| G4 learnability | Copy pass. Binding fail. | See [pilot results](REPRODUCIBILITY_REPORT.md). The reference is qualified for the measured CPU mechanics and copy control. |
| G5–G7 causal transfer, promotion and scale | Pending. | A/B/C staged training requires independently learnable skills and a globally disjoint task surface. Longitudinal arithmetic is implemented and tested. |

Use the test command in the [handoff](SESSION_HANDOFF.md). The GitHub workflow targets the Elendril branch and has a CPU-only dependency. CI execution on the remote runner was outside the local verification.
