# Developer Notes

This branch is `iterate500`.

The public training profile is `frontier`, which maps to the 500M-class AN-RA
model defined in `training/v2_config.py`, `anra/architecture.py`, and
`config/anra_frontier.yaml`.

Current path conventions:

- `phase2/agent_loop_45k`
- `phase2/fine_tuning_45i`
- `phase3/identity_45n`
- `phase3/symbolic_bridge_45q`
- `phase3/sovereignty_45r`
- `phase2/master_system_45m`

Use the dedicated Colab notebooks for training:

- `notebooks/AN_RA_T4_TRAINING.ipynb`
- `notebooks/AN_RA_TPU_TRAINING.ipynb`
