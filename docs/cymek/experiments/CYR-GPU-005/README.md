# CYR-GPU-005 — shared-parent LR-retention fork campaign

Status: **PREREGISTERED, PENDING_OPERATOR_EXECUTION** (see
`RUN_READINESS.json` after the preregistration commit). No GPU science
has run locally; the machine boundary is honored.

One acquisition per seed → G90_CONFIRMED parent checkpoint → four forks
restore THE SAME bytes and consume THE SAME future examples:

```
                    +--> HIGH_CONTINUE   (1e-3)
                    +--> LOW_CONTINUE    (1e-5)
checkpoint P -------+--> FIXED_TIME_HIGH_TO_LOW (switch at τ=0.5 actual tokens)
                    +--> HYSTERETIC_HIGH_LOW    (state machine, both states)
```

## Files

- `PLAN.md` — preregistered design (objective, hypotheses, controls,
  budgets, decision rules).
- `DESIGN_REASONING.md` — why this design and not its alternatives.
- `PREEXECUTION_AUDIT.md` — independent audit that killed CYR-GPU-004
  and drove the 005 design, incl. the production XLA accumulation fix.
- `THREATS.md` — 20 threats, each with a mechanical mitigation.
- `PREREGISTRATION.json` — hash-bound preregistration (committed in the
  preregistration freeze commit B; binds the executable SHA, file
  hashes, tokenizer identity, data manifest, arms, thresholds, resolvers,
  verdict rules).
- `RUN_READINESS.json` — the section-59 gate; the Colab link is valid
  only when `ready=true`.

## Code map

- `v5_experiments/cyr_gpu005.py` — pure core: proxy registry, T2 worlds,
  manifest, leak audit, future stream, policies, verdict rules,
  readiness gate, freeze contract.
- `anra_v5/cyr_gpu005_run.py` — torch orchestrator (the ONE campaign
  runner): calibration, resolver, acquisition, forks, arms, transfer,
  red team, packaging. Smoke mode = TINY plumbing; full mode = Colab
  only, fails closed without CUDA.
- `v5_experiments/xla_accumulation_oracle.py` — distributed CPU oracle
  + negative regression for the accumulation-boundary fix.
- `notebooks/cymek_colab_gpu_research_v5.ipynb` — 3-cell thin operator
  wrapper (bootstrap/verify/calibrate → run/resume → package/download).
- Tests: `tests/test_v5_cyr_gpu005_core.py`,
  `tests/test_v5_cyr_gpu005_plumbing_e2e.py`,
  `tests/test_v5_cyr_gpu005_freeze_sim.py`,
  `tests/test_v5_cyr_gpu005_notebook.py`,
  `tests/test_v5_xla_accumulation_oracle.py`.
- Receipt: `artifacts/v5/cyr_gpu_005_test_receipt.json`.
