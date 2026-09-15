# FORMATION-BASELINE-GATE-001

Purpose: determine, before any new expensive mechanism campaign, whether the frozen S5 `M0_STANDARD` control can reliably leave the identity floor.

This is a **post-outcome diagnostic gate**, not an amendment to FORMATION-MUX-001. The completed S5 verdicts remain `CS-MECH-002 = NULL` and `REP-FORM-003A = NULL`.

## Colab budget

Designed for a single T4 and an adaptive ~2–3 hour wall budget.

- Stage A: full six-family S5 control, seed `73012`, 3000 updates.
- Stage A: full six-family S5 control, fresh seed `91001`, 3000 updates.
- If both are decisively floor-limited, Stage B uses fresh seed `91002` on identity-only training for 2000 updates to distinguish mixture interference from deeper substrate failure.
- Otherwise Stage B runs fresh seed `91002` on the full six-family mixture for 3000 updates to test reproducibility.
- If Stage A is already at ceiling, Stage B is skipped.

GO cannot be licensed by the historically selected seed `73012`; both fresh seeds must pass the frozen GO rule in `PREREGISTRATION.json`.

## Durability

The Colab launcher writes state to Google Drive and uses the existing exact-resume S5 checkpoint path. Re-running the notebook resumes compatible checkpoints. It produces a small results ZIP and a separate checkpoint ZIP with SHA-256 receipts.

No sealed score is used for any decision.
