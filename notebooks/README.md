# Notebook launch queue

This folder contains current unrun launchers plus two notebooks retained for status review.
Completed and superseded launchers were removed from this active list; their
experiment records and preregistered hashes remain in `docs/cymek/experiments`,
and the deleted notebook contents remain recoverable from Git history.

## Current actions

| Notebook | Use | Status |
|---|---|---|
| `SIGNAC_100M_KAGGLE_TPU_PREFLIGHT.ipynb` | Eight-device Kaggle TPU synthetic engineering canary for the 100M Signac model | Implemented; Kaggle target run is still pending. This is the notebook for the current Signac TPU qualification objective. |
| `CYMEK_METRIC_RES_001_KAGGLE_T4X2.ipynb` | Checkpoint-only Cymek identity-resolution audit; no training | The Cymek roadmap's next research action. Requires the preserved S5 checkpoints attached as Kaggle input. |
| `CYMEK_METRIC_RES_001_COLAB_T4.ipynb` | Same METRIC-RES-001 audit on Colab T4 | Alternative to the Kaggle notebook above; run one platform, not both. |

## Planned or gated work

`CYMEK_FORMATION_BASELINE_GATE_001_COLAB_T4.ipynb`,
`CYMEK_FORMATION_DIAG_001_T4X2.ipynb`, and
`CYMEK_TIE_ROLE_PILOT_001_COLAB_T4.ipynb` are prospective diagnostics. They
are not the next launch; follow their preregistrations and wait for their entry
conditions.

Two retained files need status review before anyone launches them:

- `cymek_colab_gpu_r1c_preflight_debug.ipynb` has no clear experiment status in
  the current records.
- `cymek_colab_gpu_r1c_softmax_mechanism_v4.ipynb` conflicts with the R1C
  post-run record: `docs/cymek/next_core/R1C_POSTRUN_UPDATE.md` says complete,
  while `docs/cymek/experiments/CYR-GPU-014-R1C/RUN_READINESS_V4.json` says
  not executed. Do not launch it until those records are reconciled.

The Signac and Cymek actions are separate tracks: the Signac notebook checks
TPU engineering plumbing with synthetic data, while METRIC-RES-001 reads saved
Cymek checkpoints and uses no accelerator training.
