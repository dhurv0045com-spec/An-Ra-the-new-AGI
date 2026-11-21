# Avendi V8-Y3: Resumed 88/88 Campaign & Independent Evidence Audit

**Execution ledger vs. verified evidence:** The resumed Kaggle notebook archive `AVENDI_Y3_ANALYSIS (1).zip` (`3,946` bytes, SHA-256 `f558677b214991db74da784114df1f216dcaf1430ec4a815182924734cb652cc`, pinned notebook commit `57bce6bd5a17e96fb9856505c50f2ac2f9e9b123`) reports **88/88** job execution completion (`57` reused from Run 1 + `31` executed in two waves of `4` + `27`, `0` overlap, `failed_jobs = {}`, `pending_after_session = []`) and marks **50/50** studies `COMPLETE`. However, **resumed `AVENDI_Y3_EVIDENCE`, resumed `AVENDI_Y3_FULL`, `CONFIRM_LOCK.json`, and the Kaggle execution `.log` are not present locally**. Independent verification on all locally available saved archives therefore establishes **`57/88` jobs verified**, **`18/50` studies independently verified** (`U01..U06`, `U15..U25`, `U27`), **`8/50` blocked without Y1 parents** (`U07..U14`), and **`24/50` unverified** (`U26`, `U28..U35` duplicated Run-1 payloads / missing study axes, `U36..U46`, `U48..U50` from the missing resumed 31-job evidence, and `U47` missing original `CONFIRM_LOCK.json` custody).

## 1. Local Archive & Ledger Reconciliation

| Artifact | Scope | Size (bytes) | SHA-256 | Notes |
|---|---|---:|---|---|
| `AVENDI_Y3_ANALYSIS.zip` | Run 1 (`57/88`) | 4,462 | `2a0cb8b853377c658c120275876424b7dde6e1f449b312a63777f1a592752bb3` | `pending = 31`, `studies = 27/50 COMPLETE` |
| `AVENDI_Y3_EVIDENCE.zip` | Run 1 (`57/88`) | 201,240,025 | `2159ab36ea62ab60ccd95709099c79c686f122b8d5f3e7c22c4afca15fc257cc` | `305` members; no `CONFIRM_LOCK.json` |
| `AVENDI_Y3_FULL.zip` | Run 1 (`57/88`) | 2,849,446,679 | `cda53e1b875100ac664836786eacc97dd5be9e1095f8ff95ac287b4efda19535` | Extracted to `y3_resume_verification`; `57/57` job receipts pass |
| `AVENDI_Y3_ANALYSIS (1).zip` | Run 2 (`88/88`) | 3,946 | `f558677b214991db74da784114df1f216dcaf1430ec4a815182924734cb652cc` | `7` members; `already_verified = 57`, waves `4 + 27 = 31` |
| `AVENDI_V8_Y3_T4X2 (2).ipynb` | Run 2 notebook | 7,565 | `a01865571197fc64034f99e00c26899ae3bbbf7cf08099ae6b13811cabff72db` | Pinned `SOURCE_SHA = 57bce6bd5a17e96fb9856505c50f2ac2f9e9b123` |
| Resumed `EVIDENCE` / `FULL` / `.log` | Run 2 (`31` new) | — | *Unavailable locally* | Required to verify the `31` resumed jobs (`P1, P5, P6, P7-ark`) and wall exit |

- **Exact 57 + 31 Partition:** `already_verified` in `AVENDI_Y3_ANALYSIS (1).zip` equals the exact `57` job IDs in `y3_resume_verification`. The two waves (`4` Tier-0 `P1-frozen` + `P6-diagnostic_policy-s{7,19,41}`, then `27` Tier-0/Tier-1 `P5-*` + `P6-adapt-*` + `P7-ark_*`) equal the exact `31` pending IDs from Run 1, with `0` overlap and `pending_after_session = []`. Both Run 1 and Run 2 JSON report `source_commit = ba1c0e978f890021a10e2365b61d5403bc486ce5` (the plan baseline constant); no Kaggle `.log` is present locally to verify runtime wall clock or checkout HEAD.

## 2. Independent Verification Claims (Audited from Saved Evidence)

| Claim | Audited Status | Coverage (Saved) | Audit Basis & Root Reason |
|---|---|---|---|
| `byte_integrity` | `PARTIAL` | `57 / 88` | All 57 Run-1 receipts & `.npz`/SHA-256 archives pass; 31 Run-2 job receipts missing locally; `CONFIRM_LOCK.json` absent. |
| `oracle_fixture_correctness` | `PARTIAL` | `0 / 4` (or `4 / 4` on `EVIDENCE` studies) | Run-1 `EVIDENCE.zip` shows `U01, U02, U03, U05` verified; extracted `y3_resume_verification` only kept `U47`. |
| `recorded_action_replay` | `PARTIAL` | `39 / 40` | All 39 saved byte-model jobs (`P2, P3, P7-s20`) have consistent first-logit `.npz` arrays & token predictions (`0` corrupted tokens); `P0_REPLAY_RECEIPT.json` not exported by pre-fix run. |
| `checkpoint_restore` | `UNAVAILABLE` | `0 / 33` | 6 `P1` Y1 parents & 27 child jobs (`P5, P6-adapt, P7-ark`) were all in the missing 31-job Run-2 export. |
| `regenerated_per_case_agreement` | `UNAVAILABLE` | `0 / 6` | Run-2 ran prior to the `_y1_ref_parity` fix (at `57bce6bd`, which still labeled Y2 logit parity unimplemented) and its `P1-frozen` receipt is missing locally. |
| `training_reproduction` | `UNAVAILABLE` | `0 / 1` | Pre-fix runs did not export `TRAINING_REPLAY_RECEIPT.json`. |

## 3. Confirmation Custody Audit

- `AVENDI_Y3_FULL.zip` and `AVENDI_Y3_EVIDENCE.zip` include `studies/U47/U47-12jobs/confirm/receipt.json` (`LOCKED_AND_OPENED_ONCE`, `lock_sha256 = c51fe4ab684d91139f29680ddb94ac055cf4c9cd8e01693d9b8faa9a8ce7ace4`, `result_sha256 = da4f10eb314508255aaf02e8be10b9368924d73a4d0d47803503853d2398828a`, `33` confirm-eligible models), but omit `CONFIRM_LOCK.json`.
- Because Session 2 ran at `57bce6bd` without `CONFIRM_LOCK.json` preservation or prior-opening detection in `_unpack_resume` / `aggregate_full`, its reported `confirm = LOCKED_AND_OPENED_ONCE` re-evaluated the already-opened confirmation cohort rather than reusing a verified prior lock. Audited custody is therefore `CUSTODY_LIMITATION_MISSING_ORIGINAL_LOCK` (`custody_valid = false`).

## 4. Scientific Summary of Audited Saved Measurements

- **P2 CE vs. Margin (`U15` Dev, `U47` Confirm):** On dev (`2,048` cases), `P2-ce` fam4 = `0.1445, 0.1328, 0.1406` (mean `0.1393`, fam8 `0.0`); `P2-margin` fam4 = `0.2266, 0.1680, 0.1953` (mean `0.1966`, `+0.0573`, fam8 `0.0508` on `s7` only). On confirm, `P2-ce` fam4 = `0.1328, 0.1211, 0.1367` (`fam8 = 0.0`); `P2-margin` fam4 = `0.1836, 0.1484, 0.1641` (`fam8 = 0.0039` on `s41`). Does not meet P2 qualification (`fam4 >= 0.80`, `fam8 >= 0.80`, `margin_dev_gain >= 0.10`). Compact `METRICS.json` in Run 2 preserves identical `P2` values.
- **U19 Gradient Sign Reversal:** All 6 runs measure `model_grad_cosine ~ -1.0008`; `margin` reaches `0.9688..1.0000` train fit vs. `0.0039..0.0078` dev fam4.
- **P3 Interventions & P7 S20 Scale:** All P3 dev (`U27`) and P7 S20 confirm scores remain well below `0.80` (`P3-curriculum` dev `0.1055..0.1836`, confirm `0.1055..0.1406`; `P7-s20_ce` confirm `0.1016..0.1680`; `P7-s20_low_lr` confirm `0.1094..0.1445`; all `fam8 = 0.0`).
- **P4 Multi-Hop Causality (`P4-*`, 18 jobs verified):** `P4-categorical` achieves `1.0` dev & topology rate at trained depths `1..2`, `0.0` at unseen depths `3..4`, and strong causal interchange (`donor_patch_acc = 1.0`, `suffix_rewire_both_correct = 1.0`, `oracle_prototype_injection_acc = 1.0` vs. wrong-donor `0.0234..0.0469`), outperforming matched `continuous` (`donor_patch_acc = 0.1484..0.1562`) and `dense` (`0.0..0.0039`). In Run-1 archives, however, `U28..U35` wrote identical multi-study payloads and lacked distinct study-specific axes (`study_axis`).

## 5. Repairs Applied & Shortest Outstanding-Work List

- Implemented `experiments/avendi_y3/verify.py` and repaired `experiments/avendi_y3/module.py` so future runs verify all 88 receipts independently, run real `ELENDRIL_V8_Y1` parity in `_p1_job`, track `P5`/`P6`/`P7-ark` parent checkpoint SHAs, emit study-specific payloads across `U08..U14`, `U28..U46`, persist `P0_REPLAY_RECEIPT.json`, `TRAINING_REPLAY_RECEIPT.json`, `SESSION_TELEMETRY.json`, `VERIFIED_AUDIT.json`, and `CONFIRM_LOCK.json`, and protect prior sealed confirmation across sessions.
- **Shortest outstanding work to reach `88/88` independently verified:**
  1. Either supply the resumed Run-2 `AVENDI_Y3_FULL` / `EVIDENCE` archives + Kaggle `.log` to run `--verify-only` on the 31 resumed jobs (`P1-frozen`, `P5`, `P6`, `P7-ark`), **OR** run one targeted resume from `AVENDI_Y3_FULL.zip` on the new pinned notebook (with `Y1`/`Y2` inputs mounted so `P1-frozen` resolves & runs real logit parity, while preserving the 57 already-verified jobs and flagging the Run-1 missing-`CONFIRM_LOCK.json` custody limitation).
