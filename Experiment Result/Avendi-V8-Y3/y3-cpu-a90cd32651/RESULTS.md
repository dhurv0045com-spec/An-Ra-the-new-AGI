# Avendi V8-Y3 — CPU Qualification & Source-Pinned Packaging Receipt

- **Branch:** `exp/avendi-v8-y3` (branched from `0c9342223afb421ea084c26bf8148f0247be3fd7`)
- **Pinned Implementation Commit (`IMPL_SHA`):** `a90cd32651378022dcc9127b2c7d4d083556ed87`
- **Source-Pinned Kaggle Notebook:** `notebooks/AVENDI_V8_Y3_T4X2.ipynb` (`--check-notebook`: `match=true`)
- **Scope Note:** This receipt covers bounded CPU engineering qualification, Palantiri plugin/CLI custody checks, and clean-archive notebook verification. Full 2×T4 GPU training (54 main jobs / 55,296 updates + 24 short forks / 5,376 updates), Y1 checkpoint mount (`ELENDRIL_V8_Y1_FULL.zip`), and confirmation outcomes remain pending owner GPU launch.

## Package & Line Budget (`1,715 / 1,900` lines)

| File | Physical Lines |
|---|---:|
| `experiments/avendi_y3/plan.py` | 157 |
| `experiments/avendi_y3/tasks.py` | 408 |
| `experiments/avendi_y3/models.py` | 437 |
| `experiments/avendi_y3/module.py` | 476 |
| `tests/test_avendi_y3.py` | 237 |
| **Total** | **1,715** |

## Verification Summary (CPU)

- `python -m anra_v5.palantiri doctor` -> `GREEN_SIGNAL` (exit `0`)
- `python -m anra_v5.palantiri calibrate` -> `6/6` detected, `SHORTCUTS_REJECTED_AND_ORACLE_QUALIFIED` (exit `0`)
- `python -m pytest tests/test_avendi_y3.py -q` -> `12 passed` (includes 3 campaign recovery tests)
- `python -m pytest -q` -> `560 passed, 1 skipped, 14 subtests passed`
- Fresh-subprocess CLI loop (`init -> validate --canary -> run --canary -> verify-bundle -> replay-bundle`):
  - `x1_s20`: `gates_passed=true`, `verified=true`, `prediction_agreement_status=MATCHED`, `result.zip` SHA-256 `1cb02de4a12d9168ece04d791e46ffdd91fe5612bac8911a34970955fc6e9949`
  - `avendi_y3`: `gates_passed=true`, `verified=true`, `prediction_agreement_status=MATCHED`, `result.zip` SHA-256 `b35022add8cb7b5d998280149ad11b72d5ce35320539b8d7d597f1518aff4d42`
- `python -m experiments.avendi_y3.module --mode CANARY` -> `COMPLETE`, `50/50` studies emitted under `studies/Uxx/...`, `44/44` platform uses (`PASS` / `UNAVAILABLE_MISSING_PARENT`), `rescore_verified=true`, `AVENDI_Y3_SUMMARY.zip` = `5,411` bytes (SHA-256 `4d8d4a922b807a4e300e232be556fc439651c4c4f479b768ad1dc4fd1065bf5a`).
- Clean `git archive a90cd32651378022dcc9127b2c7d4d083556ed87` extraction -> `plan` (`54` jobs), `--inventory-only` (`50` studies), and `palantiri init --backend avendi_y3` (`TEMPLATE_WRITTEN`) all exited `0`.
