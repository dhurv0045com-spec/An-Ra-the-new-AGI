# AMENDMENT S6 (PROSPECTIVE) — optimizer/schedule + instrument hardening

Status: **PROSPECTIVE PROPOSAL. NOT FROZEN. NOT EXECUTED.**
Frozen Science S5 (`c15ad8be`) is unchanged and remains the authority for all
S5 verdicts. Nothing below may run as S5; it needs its own science commit,
hash-bound preregistration and pre-execution audit before any launch.

## Why

1. Every S5 CS-MECH official arm recorded `clip_fraction = 1.0` (REP-FORM R1
   likewise; R0 ~0.88–0.98). The mechanism contrasts may be
   optimization-choked rather than mechanism-null.
2. `ArmOptimizerView.step` propagates the row-group LR to itself (no-op), so
   the tied-row optimizer LR is frozen even under a future WSD schedule.
3. S5 workers do not assert the 1-proc:1-GPU pin; wrong-hardware runs fail
   after hours instead of seconds.
4. `formation_diag` margin picks the wrong runner-up index when correct, and
   the rescue denominator defaults to 128 instead of the shared 4096.

## Exact frozen-file diffs proposed (apply AT the S6 commit, never before)

- `anra_v5/formation_mux_model_v2.py` `ArmOptimizerView.step`: read
  `main.param_groups[0]["lr"]`, write it to `_row_param_group["lr"]` AND
  `rows.set_lr`. (S5 ran constant LR so S5 outcomes are unaffected in
  retrospect; the fix only matters under a schedule.)
- `tools/formation_mux_001_worker_v5.py` entry: after torch import, fail
  closed unless `cuda.is_available()` and `device_count() == 1`, then
  `set_device(0)`.
- `anra_v5/formation_diag.py` `teacher_forced_diagnostics`: runner-up index
  `[1 if hit == 1 else 0]`; `full_vs_shared_rescue` default
  `shared_rows=4096`.
- New LR/schedule bracket arm(s) ONLY as specified in `PREREGISTRATION_S6.json`.

## Kill rules

- If the S6 bracket shows no clip relief + no formation change, close the
  optimizer line and return to representation/instrument work.
- If S6 changes S5 verdicts retroactively in any claim, the amendment fails
  closed (S5 NULLs stand as executed under S5).

## Provenance

- S5 bundle `859489d9...`; audit `RETURNED_BUNDLE_AUDIT.md`
  (INCONCLUSIVE_AT_ZERO_BASELINE); probe schema
  `anra.lr-clip-probe/v1` (advisory only).
