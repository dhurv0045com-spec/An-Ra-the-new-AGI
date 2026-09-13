# OPERATOR RUNBOOK (V5.1 canary)

**Never** modify code, thresholds, or the preregistration during a run. The launcher pins an exact commit; any change invalidates identity.

## Local CPU qualification (already executed)

```bash
python tools/validate_next_core_spec.py
python -m pytest tests/test_next_core_compute_model.py tests/test_v51_canary.py -q
V51PY -m anra_v5.v51_canary_run --mode prepare          # dataset + screens + receipts
V51PY -m anra_v5.v51_canary_run --mode run --rung A --updates 120 --checkpoint-every 16
V51PY -m anra_v5.v51_canary_run --mode evaluate --rung A
V51PY -m anra_v5.v51_canary_run --mode finalize --rung A
```

## Colab T4 execution (Rung B)

1. Runtime → Change runtime type → **T4 GPU** (required; the notebook aborts otherwise).
2. Run every cell top to bottom. The notebook:
   - mounts Drive;
   - clones the repo and checks out the **frozen `CANARY_EXECUTABLE_COMMIT`** (recorded in the notebook header and in the execution receipt — never a moving branch);
   - verifies critical blob hashes (runner, generator, preregistration, step, schedule, optimizer, checkpoint modules);
   - runs `--mode scan` and displays **START / RESUME / COMPLETE / FAIL_CLOSED** — obey it;
   - `--mode run --rung B --cuda --bfloat16 --updates <total>` with per-16-update checkpoints;
   - if the session dies: rerun the notebook — scan returns RESUME and training continues from the durable checkpoint (the WSD schedule continues; never rewarm);
   - after the final update: `--mode evaluate --rung B --cuda` then `--mode finalize --rung B --cuda`;
   - packages `CYMEK_V51_CANARY_RESULTS.zip` (or `..._PARTIAL.zip` with its manifest) into the Drive root.

## Drive root

`/content/drive/MyDrive/CYMEK/V5_1_CANARY` — dedicated to this canary. **Never** write into `CYMEK/CYR-GPU-014-R1C` or any ARK-020 root.

## Safe actions

| Scan result | Meaning | Operator action |
|---|---|---|
| START | no state exists | run |
| RESUME | valid LATEST + identity match | rerun (continues exactly) |
| COMPLETE | finalization receipt for this commit exists | package/inspect only |
| FAIL_CLOSED | identity drift, corruption, partial publication, conflicting result | stop; resolve per FAILURE_MODES.md; never force |

## Recovery

- Stale `.staging-*` directory after a crash: delete it (documented recovery), rerun — last-known-good checkpoint is preserved.
- Corrupted checkpoint: `store.restore` fails closed; roll back to the previous LATEST or republish from the last good state.
- Any failure: classify per FAILURE_MODES.md, reproduce smallest, fix root cause, add a regression test. Never loosen a tolerance without a quantitative derivation proving genuine failures still fail.
