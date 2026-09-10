# ARK-019 V3 PRE-EXECUTION AUDIT

**Status:** STATIC AUDIT PASS / READY FOR OPERATOR CUDA PREFLIGHT / SCIENTIFIC RESULT NOT EXECUTED

## Frozen scientific identity

- R2-informed design addendum: commit `b3641d98a310981516aa5c93b6ad13abc032c781`.
- execution clarifications: commit `d44bf550d00e0c5dbe9587c717835234dbf01a7f`.
- prospective 1000-step horizon amendment: commit `28babe021a9b8fbc4c2e23c140e31313d50cec97`.
- frozen executable commit: `63dc47d9bdcd85e004c03aaf9a738c9682fcd8f0`.
- core blob: `bb815ebf1c6c32c45c211a744e0a4d19099dd542`.
- runner blob: `ef41096069714668b3b1e5c0e165904908ebc885`.
- pure policy test blob: `c52d6857aa2933c28291d4d115205af927c37c5f`.
- inherited ARK-018 common blob: `05b1c6a7832749740420b5b540c3214ef4c84492`.
- inherited ARK-018 binding-selection blob: `7492c697eccc5d87528094acf1b2e6164e47b1e1`.
- launcher blob: `1585da6d72270102ddd884527cef694a806bc16d`.

The Colab launcher checks out the frozen executable detached and verifies the three new V3 source/test blob identities before import or training.

## Static verification performed

The exact locally authored core/runner sources were Python-compiled successfully before being written to GitHub. The pure controller suite executed **7/7 PASS** before freeze, covering: margin-warning escalation, formal-failure escalation, persistent-failure emergency CAP16X transition, healthy de-escalation, three-confirmation SKILL_B rule, guaranteed non-identity replay permutation, and distinct query-order construction.

The launcher repeats `py_compile` and the 7-test policy suite in Colab against the frozen commit before touching scientific continuation arms.

## Runtime/data firewall

The runner refuses execution unless:

1. CUDA is available;
2. the exact ARK-018 prepared receipt exists in Drive;
3. ARK-018 science identity equals `b397427cd5964b7cc2a41264ca8789a0c020f96d4e403b314900798711a2ead5`;
4. prepared horizon is 8000;
5. tokenizer hash matches the prepared receipt;
6. train/control/sealed cache hashes match when bound hashes are present;
7. both `SCIENCE_ONLY` checkpoints exist for seeds 31801 and 31902, are at the final prepared horizon, and bind the same science/tokenizer identities;
8. 24 distinct binding tokens can be selected and A/B token namespaces are disjoint;
9. both fresh SKILL_A parents qualify prospectively on CONTROL and then SEALED;
10. an exact model/optimizer/scaler/RNG resume smoke produces identical next-10-update parameter hashes;
11. actual-GPU runtime calibration projects the entire fixed 4-set x 4-arm campaign inside the 175-minute wall with the 1.30 safety factor.

If the runtime gate fails, V3 stops before matched continuation arms rather than reducing seeds, arms, horizon or thresholds.

## Causal matching

Within each of 4 matched sets, all four arms start from the same frozen SKILL_A parent and consume the same deterministic real-text starts and SKILL_B semantic IDs. Replay replaces a real-text slot only; SKILL_B exposure is not reduced. `PLASTIC_HIGH` is the interference reference and `STATIC_REPLAY_1OF64` is the static-protection reference.

The dynamic arms are prospectively distinct:

- `GUARDIAN_REPLAY`: PLASTIC -> 1/64 replay on margin warning -> 1/32 replay on formal failure -> de-escalate after recovery.
- `GUARDIAN_HYBRID`: same policy, but persistent failure while already at 1/32 activates a temporary CAP16X emergency brake.

CAP16X is frozen per matched set from a 32-step LOW shadow before arm outcomes. The shadow state is discarded. The cap projects the applied parameter delta but intentionally does not rewrite AdamW moments; no trust-region claim is permitted.

## Evaluation firewall

CONTROL drives parent qualification and controller transitions. SEALED is measurement-only and never changes controller state. SKILL_A, SKILL_B and science CONTROL/SEALED metrics are recorded every 100 continuation updates. Full model checkpoints overwrite a durable per-arm checkpoint every 300 updates and at final, allowing bounded replay after a Colab disconnect rather than losing the entire arm.

## Known limitations preserved in the claim boundary

- R2's 1/64 and CAP16X evidence was a secondary Micro screen; V3 is the first prospective real-text proxy test of those actions.
- The 32-slot objective is a proxy continuation objective, not a production pretraining mixture.
- Per-step movement telemetry is projected except when a full CAP calculation is required; full-model displacement is measured at evaluation boundaries.
- The ~21M ARK-018 subject is still far below production scale.
- A positive V3 result is only a Guardian challenger for larger/Cymek replication; it does not authorize PRE500M, a production scheduler, or an AGI claim.

## Verdict

**STATIC AUDIT PASS. ARK-019 V3 is ready for the operator Colab CUDA preflight.**

The remaining unknowns are runtime facts that cannot be established statically: whether the current T4 passes exact deterministic resume, whether both parents qualify again under this exact V3 construction, whether the complete campaign fits the measured wall, and the scientific outcome itself. The notebook must fail closed on any of those conditions.