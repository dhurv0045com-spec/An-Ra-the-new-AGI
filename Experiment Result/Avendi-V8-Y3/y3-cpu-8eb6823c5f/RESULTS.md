# Avendi V8-Y3 — Upgraded Dual-T4 Resumable `FULL` Campaign & Notebook Receipt

- **Branch:** `exp/avendi-v8-y3`
- **Pinned Implementation Commit (`IMPL_SHA`):** `8eb6823c5fde5b37b26d2f2040e0e9d95fa54aa9`
- **Source-Pinned Kaggle Notebook:** `notebooks/AVENDI_V8_Y3_T4X2.ipynb` (`--check-notebook`: `match=true`)
- **What Changed vs `a90cd326`:** Replaced the compressed single-process `FULL` path with per-job CLI worker dispatch (`--job` across `cuda:0` and `cuda:1` via `CampaignOrchestrator.run_waves`), real `1,024`/`256`/`256` family denominators, Tier-0/Tier-1 dependency waves with `PENDING_QUEUE.json` deferral, full `ckpts/` + `jobs/` cross-session bundling & verified `RESUME_INPUT` restore, and sealed confirm evaluation (`_confirm_one`).

## Verified CPU Checks (`8eb6823c5f`)

- `python -m anra_v5.palantiri doctor` -> `GREEN_SIGNAL`
- `python -m anra_v5.palantiri calibrate` -> `6/6 SHORTCUTS_REJECTED_AND_ORACLE_QUALIFIED`
- `python -m pytest tests/test_avendi_y3.py tests/test_elendril_tree.py tests/test_planthir_v2_audit_upgrades.py -n auto -q` -> **`24 passed in 293.47s`** (includes mocked dual-GPU 2-session `PARTIAL -> COMPLETE` all-88-job deferral & `AVENDI_Y3_FULL.zip` resume test!)
- Fresh-subprocess `avendi_y3` CLI loop (`init -> validate --canary -> run --canary -> verify-bundle -> replay-bundle`): `verified=true`, `prediction_agreement_status=MATCHED`, `result.zip` SHA-256 `bfd6413d8c6e4a57f7199ab246259d9bf58b5658806f7e60d11669cac363e984`, `checkpoint_full.zip` SHA-256 `e6a07f78b24ea321540d5d16450b9fbc55c1fb17f986714a4f0e8ac8fe0e85d1`.
- Clean `git archive 8eb6823c5fde5b37b26d2f2040e0e9d95fa54aa9` extraction: `--mode FULL --inventory-only` (`54` main + `24` short forks + `9` P6 adapt jobs, `50` studies, `44` platform uses) & `init --backend avendi_y3` (`TEMPLATE_WRITTEN`) exited `0`.
