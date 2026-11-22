# Avendi V8-Y3: Full 88-Job GPU Campaign Results & Independent Audit

**Execution & verification summary:** The clean single-session Kaggle 2×T4 run completed **88 / 88** planned jobs (`54/54` main training, `24/24` short forks, `9/9` P6 adaptation jobs, `1/1` P1 frozen intake; `pending = []`, `failed = []`) in **`4,132.65 s`** (**`68.88 min`**, `2026-11-22T05:22:56Z` to `06:31:49Z`, `exit_reason = ALL_JOBS_COMPLETE`), marking **50 / 50** studies `COMPLETE` and executing a freshly sealed **single-session confirmation opening** (`CONFIRM_LOCK.json`, `LOCKED_AND_OPENED_ONCE`, `custody_valid = true`). On independent evidence verification (`VERIFIED_AUDIT.json`), **`79 / 88` jobs** and **`36 / 50` studies** pass all checks while overall `verification_status` is **`PARTIAL_UNVERIFIED`** (`scientific_verified = false`) because (1) all six `P1-frozen` Y1 lineages were **`REJECTED`** (`OSError: [Errno 30] Read-only file system` on mounted read-only Y1 `analysis/jobs` without `ELENDRIL_V8_Y1_EVIDENCE.zip` logit parity), blocking `U07..U14`, and (2) all nine `P6-adapt-*` jobs recorded `completed_updates = 1024` from their inherited `P6-diagnostic_policy` trainer rather than `256` expected by `verify.py` (`UPDATE_COUNT_MISMATCH:1024!=256`), leaving `U41..U46` unverified.

## Executive Scientific Decision

| Stage / Studies | Execution Status | Evidence Verification | Scientific Qualification Verdict | Summary Finding |
|---|---|---|---|---|
| **P0 (`U01..U06`)** | `COMPLETE` | `VERIFIED` (`6/6`) | **Passed** | Oracle (`256/256`), split custody, zero-family shortcuts (`9` policies), balanced stream check (`224` vs `22` pairs), corruption detector, `P0_REPLAY_RECEIPT`, and `TRAINING_REPLAY_RECEIPT` all pass. |
| **P1 (`U07..U14`)** | `COMPLETE` (job executed) | `BLOCKED_MISSING_PARENT` (`0/8`) | **Blocked** | All 6 `Y1` lineages rejected on read-only `/kaggle/input/.../analysis/jobs` and `UNAVAILABLE_Y1_EVIDENCE_NOT_MOUNTED`; see [Y2 frozen checkpoint audit](../../Avendi-V8-Y2/avendi-frozen-63c10bbd2f/RESULTS.md) for prior Y1 weights (`0/256`). |
| **P2 (`U15..U20`)** | `COMPLETE` | `VERIFIED` (`6/6`) | **Not qualified** | `P2-margin` improves dev `fam4` over `P2-ce` on all 3 seeds (`0.1966` vs `0.1393`, `+5.73` pp) and confirm `fam4` (`0.1654` vs `0.1302`, `+3.52` pp), but misses preregistered gates (`fam4 >= 0.80`, `fam8 >= 0.80`, `margin_dev_gain >= 0.10`, `literal >= 0.95`). `U19` shows `-1.0008` gradient cosine reversal. |
| **P3 (`U21..U27`)** | `COMPLETE` | `VERIFIED` (`7/7`) | **Not qualified** | None of the 7 interventions (`address`, `address_sham`, `curriculum`, `low_lr`, `refresh`, `rename`, `rename_sham`) exceeds mean dev `fam4 = 0.1536` or confirm `fam4 = 0.1198` (`fam8 = 0.0` everywhere). |
| **P4 (`U28..U35`)** | `COMPLETE` | `VERIFIED` (`8/8`) | **Passed on trained depths `1..2`; Failed length `3..4`** | Supervised `P4-categorical` achieves `1.0000` at depths `1..2` (dev, topology, distractor) and `1.0000` causal donor patch, suffix rewire, and ridge probe across all 3 seeds, cleanly separating from `continuous` (`0.1484..0.1562` donor) and `dense` (`0.0000..0.0039`), but drops to `0.0000` at unseen depths `3..4`. |
| **P5 (`U36..U40`)** | `COMPLETE` | `VERIFIED` (`5/5`) | **Partial / Exploratory** | Forks with `info` (`direct_plus_info` mean `0.7857`, `replace_direct` `0.7714`) solve all 4 query/stale/missing-tool conditions (`1.00`) vs `P2-ce` parent (`0.1357`) and `direct_plus_neutral` (`0.1429`), but remain slightly below the `0.80` macro-optimal gate due to `sufficient` (`0.35..0.55`) and distractor filtering (`0.40..0.60`). |
| **P6 (`U41..U46`)** | `COMPLETE` | `UNVERIFIED` (`0/6`, update-count audit flag) | **Engineered baseline only; Not qualified** | `P6-diagnostic_policy` reaches `0.3320..0.3711` accuracy across feedback grids (`persistence ~ 0.35`), bounded weight updates stay `<= 3.95e-5 < 0.001`, and writer is explicitly `ENGINEERED_NOT_LEARNED`. `P6-adapt` receipts report `1024 != 256` updates. |
| **P7 (`U47..U50`)** | `COMPLETE` | `VERIFIED` (`4/4`) | **Not qualified (S20); Stability-plasticity trade-off** | Scaling `S20` (`s20_ce` confirm `0.1250`, `s20_low_lr` confirm `0.1328`, `fam8 = 0.0`) does not beat `S3`. `P7-ark` isolates a clean cap trade-off: `ark_capped` preserves parent `child_a` (`0.1406`) without learning P5 (`0.1405`), while `ark_uncapped`/`ark_expaug` reach `~0.79` P5 as `child_a` collapses (`0.0625` / `0.0000`). |

---

## 1. Archive Integrity, Provenance & Telemetry

Every member digest in both Kaggle session exports verified with **zero mismatches and zero missing files**:

| Artifact | Bytes | SHA-256 | Members / Verified `SHA256SUMS` | Pinned Source / Notes |
|---|---:|---|---:|---|
| `AVENDI_Y3_ANALYSIS (2).zip` (matches uploaded `media_1795336373753_a4fe2160.zip`) | 15,259 | `fad1f35312d591cd9381af6fc0a7269b88a909441df8634fda80fc116b153d1f` | `12` members / `10` checked, `0` mismatched | Full JSON summaries, `CONFIRM_LOCK.json`, `VERIFIED_AUDIT.json`, `SESSION_TELEMETRY.json` |
| `AVENDI_Y3_EVIDENCE (1).zip` | 201,307,755 | `a518838ffaa2b937d2639e405e055a087a18bb437ad62560e173da5dd5a3b49a` | `341` members / `339` checked, `0` mismatched | All `88` job receipts, `.npz` first-logit arrays, all `50` study receipts |
| `AVENDI_V8_Y3_T4X2 (4).ipynb` | 7,821 | `bc882cc84740eb9c145fa4a53985d30c8d5428921499164fa8eb183bf0245474` | `SOURCE_SHA = 60e8cb27f6e7fc21f9aaca5992a2d5ecf0c77d6b` | Hitched commit `69c663b7` (`progress.md` + notebook pin) |

- **Provenance note:** All job/plan JSONs record the static preregistration technology baseline `source_commit = ba1c0e978f890021a10e2365b61d5403bc486ce5`, while the Kaggle notebook checked out and ran runtime commit `60e8cb27f6e7fc21f9aaca5992a2d5ecf0c77d6b` cleanly without errors across dual T4 GPUs.
- **Session wall-clock telemetry:** Started `2026-11-22T05:22:56.591637+00:00`, finished `2026-11-22T06:31:49.242038+00:00`, total elapsed `4,132.65 s` (`68.88 min`), `reused_job_ids = []` (`0`), `executed_job_ids = 88`, `failed_job_ids = []`, `pending_job_ids = []`, `exit_reason = ALL_JOBS_COMPLETE`. Cumulative single-GPU worker seconds summed across all 88 jobs was `6,803.65 s` (~`1.65x` dual-T4 parallel speedup including sequential Tier-0/Tier-1 gates and confirmation evaluation).

---

## 2. Independent 6-Claim Verification Matrix & Audit Dispositions

| Claim | Audited Status | Coverage | Audit Basis |
|---|---|---|---|
| `byte_integrity` | **`PASS`** | `88 / 88` | All 88 job receipts, `job_archive.zip` SHA-256s, specs, updates, and `CONFIRM_LOCK.json` verified. |
| `oracle_fixture_correctness` | **`PASS`** | `4 / 4` | `U01`, `U02`, `U03`, `U05` fixture suites verified complete and non-leaking. |
| `recorded_action_replay` | **`PASS`** | `40 / 40` | `P0_REPLAY_RECEIPT` (`verified = true`) plus all 39 byte-model jobs (`P2, P3, P7-s20`) with matching `.npz` logits and decoded tokens (`0` corrupted tokens). |
| `checkpoint_restore` | **`BLOCKED`** | `18 / 33` (`0/6` Y1, `18/27` children) | All `9` `P5` + `9` `P7-ark` child jobs verified parent SHA & initial core hash; `0/6` `Y1` parents restored (`REJECTED`); `9` `P6-adapt` jobs matched parent SHA `3d95e971/21aa3a6e/2efa6e0e` but failed `UPDATE_COUNT_MISMATCH:1024!=256`. |
| `regenerated_per_case_agreement` | **`UNAVAILABLE`** | `0 / 6` | Blocked upstream by `P1-frozen` Y1 parent rejection (`matched_reference_cases = 0`, `y2_logit_parity = UNAVAILABLE_Y1_EVIDENCE_NOT_MOUNTED`). |
| `training_reproduction` | **`PASS`** | `1 / 1` | `TRAINING_REPLAY_RECEIPT` (`passed = true`, `losses_matched = true`, `weights_matched = true`, shared core SHA `e102cc16...e183db`). |

**Why 9 `P6-adapt-*` jobs and `U41..U46` were marked `UNVERIFIED`:** All 9 `P6-adapt-{context,memory,weight_update}-s{7,19,41}` jobs restore `P6-diagnostic_policy-s{seed}` at `global_update = 1024` and run `_p6_eval` (`256` dev blocks) with `restored_parent = true` and zero additional gradient steps, writing `completed_updates = 1024`. Because `plan.py` lists `updates = 256` on `p6_adaptation_jobs`, `verify.py` flagged `UPDATE_COUNT_MISMATCH:1024!=256`. Their evaluation measurements are nonetheless complete and share identical parent checkpoints (`3d95e971...` for `s7`, `21aa3a6e...` for `s19`, `2efa6e0e...` for `s41`).

---

## 3. P0 Instrument Qualification & P1 Frozen Y1 Status

- **P0 (`U01..U06`, `VERIFIED`):** `U01` confirms `oracle_ok = true` on `256` dev families (`2,048` cases). `U02` passes split custody (`0` defects, planted leak caught). `U03` confirms `oracle = 2,048/2,048` cases (`256/256` families), `opposing_truth_masked_views = 1,024`, and `0/256` complete families across all eight shortcut baselines (`abstain` `0/2,048` cases; `bridge_blind` `0/2,048`; `alphabet_majority`, `first_row`, `last_row`, `query_blind`, `terminal_copy`, `timestamp_rank` all `1,024/2,048` cases, `0/256` families). `U04` documents `224` balanced distinct query-gold pairs vs `22` historical (`top_gold_share`: `0.078125` vs `0.5`). `U05` detects both EOS flip and case swaps.
- **P1 (`U07..U14`, `BLOCKED_MISSING_PARENT`):** `P1-frozen` resolved `checkpoint_root = /kaggle/input/notebooks/drdoom1/elendril-v8-y1-kaggle-t4x2/elendril_y1_all/analysis/jobs` ahead of `ELENDRIL_V8_Y1_FULL.zip`; opening those read-only metadata directories triggered `OSError: [Errno 30] Read-only file system` when creating `objects/`, marking all 6 lineages `REJECTED` (`y2_logit_parity = UNAVAILABLE_Y1_EVIDENCE_NOT_MOUNTED`).

---

## 4. P2 Core Acquisition, Generalization & U19 Gradient Sign Reversal (`U15..U20`)

All metrics below are on `256` dev families (`2,048` cases) across `5` crossover/probe panels (`S3` byte model, `1,024` updates).

| Job | Wall (s) | Native fam4 | Native fam8 | Native raw | Native margin | Renamed fam4 | TsPerm fam4 | Irrel fam4 | Literal raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `P2-ce-s7` | 92.76 | 0.144531 | 0.000000 | 0.508301 | +0.009569 | 0.125000 | 0.156250 | 0.000000 | 0.312500 |
| `P2-ce-s19` | 87.93 | 0.132812 | 0.000000 | 0.503906 | +0.001058 | 0.218750 | 0.062500 | 0.000000 | 0.396973 |
| `P2-ce-s41` | 92.09 | 0.140625 | 0.000000 | 0.494629 | +0.000206 | 0.156250 | 0.171875 | 0.000000 | 0.293945 |
| `P2-margin-s7` | 98.03 | 0.226562 | 0.050781 | 0.542480 | +0.152720 | 0.187500 | 0.281250 | 0.000000 | 0.301758 |
| `P2-margin-s19` | 92.58 | 0.167969 | 0.000000 | 0.506836 | +0.006785 | 0.203125 | 0.140625 | 0.000000 | 0.389648 |
| `P2-margin-s41` | 101.79 | 0.195312 | 0.000000 | 0.504395 | +0.000704 | 0.187500 | 0.250000 | 0.000000 | 0.287598 |

- **Mean paired difference:** `P2-margin` improves native dev `fam4` from `0.139323` to `0.196615` (`+0.057292`, missing `margin_dev_gain_min = 0.10`), improves every paired seed (`+0.0820`, `+0.0352`, `+0.0547`), and solves non-zero `fam8` only on `P2-margin-s7` (`0.0508`). Both objectives score `0.0000` complete families on `irrelevant_rows` and `literal` panels (`irrelevant_rows` raw `<= 0.0117`).
- **U19 gradient-sign & memorization audit (`128` updates):** All 6 `U19` jobs reproduce `model_grad_cosine ~ -1.0008` (`s7: -1.000853`, `s19: -1.000843`, `s41: -1.000833`), confirming exact sign reversal of loss gradients under swapped query labels, while dev `fam4` collapses to `0.0000..0.0078` despite high train-fit `fam4` (`ce`: `0.5313, 0.7500, 0.4375`; `margin`: `1.0000, 0.9688, 0.9688`).

---

## 5. P3 Intervention Screening (`U21..U27`)

| Job | Wall (s) | Native fam4 | Native raw | Renamed fam4 | Mean Dev fam4 |
|---|---:|---:|---:|---:|---:|
| `P3-address` (`s7 / s19 / s41`) | 94.33 / 97.30 / 93.86 | 0.0977 / 0.1445 / 0.1680 | 0.5049 / 0.5015 / 0.4971 | 0.1406 / 0.1250 / 0.2031 | **0.1367** |
| `P3-address_sham` (`s7 / s19 / s41`) | 94.58 / 93.78 / 95.41 | 0.1172 / 0.1367 / 0.0898 | 0.5010 / 0.5063 / 0.5063 | 0.1094 / 0.1719 / 0.1094 | **0.1146** |
| `P3-curriculum` (`s7 / s19 / s41`) | 95.06 / 91.67 / 93.40 | 0.1719 / 0.1055 / 0.1836 | 0.5054 / 0.5015 / 0.5020 | 0.1875 / 0.1563 / 0.0938 | **0.1536** |
| `P3-low_lr` (`s7 / s19 / s41`) | 90.87 / 90.47 / 94.28 | 0.0664 / 0.0430 / 0.0273 | 0.5098 / 0.5059 / 0.5020 | 0.0469 / 0.0781 / 0.0469 | **0.0456** |
| `P3-refresh` (`s7 / s19 / s41`) | 89.04 / 94.43 / 94.33 | 0.0078 / 0.0078 / 0.0039 | 0.4966 / 0.4961 / 0.5073 | 0.0000 / 0.0156 / 0.0000 | **0.0065** |
| `P3-rename` (`s7 / s19 / s41`) | 182.65 / 187.70 / 183.77 | 0.0938 / 0.0859 / 0.0664 | 0.5029 / 0.4971 / 0.4971 | 0.0938 / 0.1094 / 0.0938 | **0.0820** |
| `P3-rename_sham` (`s7 / s19 / s41`) | 175.99 / 181.50 / 181.00 | 0.0547 / 0.0117 / 0.0039 | 0.5039 / 0.4985 / 0.4858 | 0.0313 / 0.0000 / 0.0156 | **0.0234** |

All 21 `P3` models score `fam8 = 0.0000` on dev and fall far short of the `0.80` qualification gate.

---

## 6. P4 Multi-Hop State Representation & Causal Interventions (`U28..U35`)

Width matching yields `categorical` (`width=128`, `params=37,648`), `continuous` (`width=66`, `params=37,636`, `param_rel_diff=0.000319`), and `dense` (`width=64`, `params=43,792`, `confounded=true`). Every causal row has `identity_preserved = 1.0` and `distinguishable_pairs = 256`.

| Job | d1 | d2 | d3 | d4 | Top-d2 | Dist-d2 | Unpatched | Donor | Rewire | Noise | Wrong/Actual | Wrong/Sham | Ridge | Oracle | Recon Err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `P4-categorical-s7` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2539 | 0.0430 | 0.2773 | 1.0000 | 1.0000 | 0.000043 |
| `P4-categorical-s19` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2031 | 0.0234 | 0.2539 | 1.0000 | 1.0000 | 0.000020 |
| `P4-categorical-s41` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3008 | 0.0469 | 0.3242 | 1.0000 | 1.0000 | 0.000003 |
| `P4-categorical_endonly-s7` | 1.0000 | 0.1094 | 0.0000 | 0.0000 | 0.1406 | 0.1250 | 0.3359 | 0.3594 | 0.3359 | 0.2578 | 0.0508 | 0.2188 | 0.5781 | 0.4063 | 0.740484 |
| `P4-categorical_endonly-s19` | 1.0000 | 0.0000 | 0.0156 | 0.0000 | 0.0000 | 0.0000 | 0.1563 | 0.1484 | 0.1133 | 0.1523 | 0.0391 | 0.1484 | 0.1172 | 0.1484 | 0.896905 |
| `P4-categorical_endonly-s41` | 1.0000 | 0.7031 | 0.0000 | 0.0000 | 0.7656 | 0.6875 | 0.8281 | 0.8047 | 0.8008 | 0.2656 | 0.0430 | 0.2305 | 0.9141 | 0.8438 | 0.308416 |
| `P4-continuous-s7` | 1.0000 | 0.3438 | 0.0313 | 0.0156 | 0.2188 | 0.2656 | 0.1406 | 0.1523 | 0.0781 | 0.1055 | 0.0430 | 0.1133 | 0.1094 | 0.1211 | 0.902171 |
| `P4-continuous-s19` | 1.0000 | 0.4375 | 0.0469 | 0.0156 | 0.2188 | 0.3281 | 0.1445 | 0.1563 | 0.0625 | 0.1133 | 0.0781 | 0.1445 | 0.0547 | 0.1172 | 0.885888 |
| `P4-continuous-s41` | 1.0000 | 0.4844 | 0.0469 | 0.0156 | 0.2813 | 0.1875 | 0.1172 | 0.1484 | 0.0977 | 0.1094 | 0.0508 | 0.1211 | 0.0547 | 0.1172 | 0.903770 |
| `P4-continuous_endonly-s7` | 0.9531 | 0.1406 | 0.0469 | 0.0000 | 0.0469 | 0.0625 | 0.1484 | 0.1406 | 0.0430 | 0.1563 | 0.0469 | 0.1484 | 0.0859 | 0.1328 | 0.938330 |
| `P4-continuous_endonly-s19` | 0.8750 | 0.1719 | 0.0469 | 0.0156 | 0.0938 | 0.0938 | 0.1641 | 0.1484 | 0.0508 | 0.1914 | 0.0742 | 0.1563 | 0.0547 | 0.1797 | 0.936733 |
| `P4-continuous_endonly-s41` | 1.0000 | 0.1250 | 0.0313 | 0.0156 | 0.1250 | 0.1719 | 0.1641 | 0.1563 | 0.0664 | 0.1641 | 0.0352 | 0.1484 | 0.0625 | 0.1445 | 0.945551 |
| `P4-dense-s7` | 1.0000 | 0.1719 | 0.0469 | 0.0313 | 0.2031 | 0.1094 | 0.1680 | 0.0039 | 0.0000 | 0.0898 | 0.0039 | 0.0781 | 0.1172 | 0.0781 | 0.936584 |
| `P4-dense-s19` | 1.0000 | 0.1875 | 0.0313 | 0.0313 | 0.0781 | 0.1406 | 0.1758 | 0.0039 | 0.0000 | 0.0664 | 0.0000 | 0.0859 | 0.0703 | 0.1133 | 0.947152 |
| `P4-dense-s41` | 1.0000 | 0.2656 | 0.0469 | 0.0000 | 0.1563 | 0.1719 | 0.2266 | 0.0000 | 0.0000 | 0.0938 | 0.0000 | 0.0859 | 0.1172 | 0.0898 | 0.948075 |
| `P4-dense_endonly-s7` | 0.7656 | 0.0625 | 0.0469 | 0.0000 | 0.0313 | 0.0625 | 0.1523 | 0.0234 | 0.0000 | 0.1172 | 0.0195 | 0.0820 | 0.1094 | 0.0898 | 0.939157 |
| `P4-dense_endonly-s19` | 0.8125 | 0.0469 | 0.0469 | 0.0000 | 0.0625 | 0.0469 | 0.1719 | 0.0156 | 0.0000 | 0.0469 | 0.0117 | 0.0781 | 0.0625 | 0.0859 | 0.934191 |
| `P4-dense_endonly-s41` | 0.8906 | 0.1250 | 0.0625 | 0.0156 | 0.0781 | 0.0469 | 0.2188 | 0.0000 | 0.0000 | 0.0781 | 0.0039 | 0.0703 | 0.0938 | 0.0625 | 0.942031 |

Categorical intermediate supervision isolates crisp prototype states (reconstruction error `3e-6..4.3e-5`, hard-argmax dev matches soft `1.0000` at depths `1..2`), but training only on `depths=(1, 2)` yields `0.0000` length extrapolation to depths `3..4`. Unlike `P4-categorical`, endpoint-only training varies heavily by seed (`0.0000..0.7031` on `d2`), and continuous/dense models do not form transferable causal step-1 residuals.

---

## 7. P5 Executive Policy Forks & Retention vs. Acquisition (`U36..U40`)

All 9 P5 forks inherit exact initial weights from `P2-ce-s{seed}` (`parent_core_sha256 == child_initial_core_sha256` on `9/9`). Reference DP policies on the 7 executive conditions score `dp_optimal = 1.0` (`max_regret = 0.0`), `prior_only = 0.7143`, `always_query = 0.5714`, `always_abstain = 0.2857` (`1.0` false abstention), and `tool_blind = 0.1786`. Every `P2-ce` parent measures `parent_p5 = 0.135714` and `false_abstention = 0.0`.

| Job | `parent_a` | `child_a` | `child_p5` | False Abstain | Sufficient | Newly Rel. | Newly Irrel. | Req. Obtainable | Req. Unobtain. | Stale Contra. | Irrel. Missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `P5-direct_plus_neutral-s7` | 0.1563 | 0.4063 | 0.1571 | 0.00 | 0.25 | 0.00 | 0.25 | 0.00 | 0.00 | 0.00 | 0.60 |
| `P5-direct_plus_neutral-s19` | 0.0781 | 0.1719 | 0.1286 | 0.00 | 0.10 | 0.00 | 0.25 | 0.00 | 0.00 | 0.00 | 0.55 |
| `P5-direct_plus_neutral-s41` | 0.2188 | 0.2188 | 0.1429 | 0.00 | 0.30 | 0.00 | 0.10 | 0.00 | 0.00 | 0.00 | 0.60 |
| `P5-direct_plus_info-s7` | 0.1563 | 0.2813 | 0.7929 | 0.00 | 0.55 | 1.00 | 0.45 | 1.00 | 1.00 | 1.00 | 0.55 |
| `P5-direct_plus_info-s19` | 0.0781 | 0.0938 | 0.7929 | 0.00 | 0.50 | 1.00 | 0.45 | 1.00 | 1.00 | 1.00 | 0.60 |
| `P5-direct_plus_info-s41` | 0.2188 | 0.1875 | 0.7714 | 0.00 | 0.40 | 1.00 | 0.40 | 1.00 | 1.00 | 1.00 | 0.60 |
| `P5-replace_direct-s7` | 0.1563 | 0.3438 | 0.7786 | 0.00 | 0.45 | 1.00 | 0.40 | 1.00 | 1.00 | 1.00 | 0.60 |
| `P5-replace_direct-s19` | 0.0781 | 0.2188 | 0.7643 | 0.00 | 0.35 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 |
| `P5-replace_direct-s41` | 0.2188 | 0.2188 | 0.7714 | 0.00 | 0.40 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 |

`info`-containing forks acquire `1.00` accuracy on `newly_relevant`, `req_missing_obtainable`, `req_missing_unobtainable`, and `stale_contradictory` without inducing false abstention (`0.00`) or regressing `child_a` below `parent_a` (`mean child_a = 0.1875 / 0.2604` vs `0.1510` parent mean), yet remain marginally below the `0.80` macro-optimal gate (`0.7857 / 0.7714` mean) because direct-answer sufficiency and distractor resistance stay `0.35..0.60`.

---

## 8. P6 Stateful Memory, Bounded Weight Adaptation & Diagnostic Utility (`U41..U46`)

- **Diagnostic policy training (`P6-diagnostic_policy`, `1,024` updates, `256` dev blocks):**
  - **`s7` (`wall = 38.42 s`, ckpt `3d95e971...`):** `persistence_after_context_clear = 0.3477`; grid acc `0.3320..0.3477`; memory controls (`erase / freeze / shuffle`) all `0.3359`; weight updates (`incorrect / irrelevant / relevant`) all `0.3359` acc with `max_rel_update = 3.74e-5 / 3.81e-5 / 3.52e-5 <= 0.001`; utility `learned = dp_optimal = -0.3547 < passive (-0.3281)`.
  - **`s19` (`wall = 39.59 s`, ckpt `21aa3a6e...`):** `persistence = 0.3594`; grid `0.3516..0.3711`; memory controls `0.3711`; weight updates `0.3711` (`max_rel_update <= 2.96e-5`); utility `learned = -0.3313 < passive (-0.2578)`.
  - **`s41` (`wall = 38.73 s`, ckpt `2efa6e0e...`):** `persistence = 0.3516`; grid `0.3359..0.3516`; memory controls `0.3398`; weight updates `0.3398` (`max_rel_update <= 3.95e-5`); utility `learned = -0.3469 < passive (-0.3203)`.
- **P6 adaptation forks (`P6-adapt-{context,memory,weight_update}-s{7,19,41}`):** All 9 jobs restored their exact `P6-diagnostic_policy-s{seed}` parent checkpoint SHA and evaluated identically per seed across the `256` dev blocks (`restored_parent = true`, `no_info_ceiling = 0.5`), failing only `verify.py`'s `completed_updates == 256` check (`1024 != 256`). Both policy accuracy (`0.33..0.37 < 0.75`) and writer provenance (`ENGINEERED_NOT_LEARNED`) fail learned mechanism qualification.

---

## 9. P7 Scale (`S20`), Arkenstone Forks, Donor Audits & Sealed Confirmation (`U47..U50`)

- **Sealed confirmation (`CONFIRM_LOCK.json`, `LOCKED_AND_OPENED_ONCE`, `custody_valid = true`):** `lock_sha256 = c51fe4ab684d91139f29680ddb94ac055cf4c9cd8e01693d9b8faa9a8ce7ace4`, `result_sha256 = da4f10eb314508255aaf02e8be10b9368924d73a4d0d47803503853d2398828a`, covering all **`33`** eligible models (exact match to Run-1 hashes).

| Arm (`s7 / s19 / s41`) | Dev `fam4` | Confirm `fam4` (mean) | Confirm `fam8` | Confirm `raw` | Confirm `literal_raw` |
|---|---|---:|---:|---:|---|
| `P2-ce` | 0.1445 / 0.1328 / 0.1406 | 0.1328 / 0.1211 / 0.1367 (**0.1302**) | 0.0000 / 0.0000 / 0.0000 | 0.5151 / 0.5015 / 0.5000 | 0.2803 / 0.3916 / 0.2998 |
| `P2-margin` | 0.2266 / 0.1680 / 0.1953 | 0.1836 / 0.1484 / 0.1641 (**0.1654**) | 0.0039 / 0.0000 / 0.0000 | 0.5405 / 0.5020 / 0.5005 | 0.2891 / 0.3906 / 0.2949 |
| `P3-address` | 0.0977 / 0.1445 / 0.1680 | 0.0781 / 0.1289 / 0.1484 (**0.1185**) | 0.0000 / 0.0000 / 0.0000 | 0.5146 / 0.5034 / 0.4985 | 0.3696 / 0.3687 / 0.2920 |
| `P3-address_sham` | 0.1172 / 0.1367 / 0.0898 | 0.0898 / 0.1055 / 0.0586 (**0.0846**) | 0.0000 / 0.0000 / 0.0000 | 0.5024 / 0.5010 / 0.5029 | 0.3716 / 0.3877 / 0.3218 |
| `P3-curriculum` | 0.1719 / 0.1055 / 0.1836 | 0.1133 / 0.1055 / 0.1406 (**0.1198**) | 0.0000 / 0.0000 / 0.0000 | 0.5000 / 0.5044 / 0.5015 | 0.2588 / 0.4063 / 0.3213 |
| `P3-low_lr` | 0.0664 / 0.0430 / 0.0273 | 0.0547 / 0.0430 / 0.0234 (**0.0404**) | 0.0000 / 0.0000 / 0.0000 | 0.5049 / 0.5059 / 0.4927 | 0.2964 / 0.2935 / 0.3569 |
| `P3-refresh` | 0.0078 / 0.0078 / 0.0039 | 0.0000 / 0.0078 / 0.0078 (**0.0052**) | 0.0000 / 0.0000 / 0.0000 | 0.5000 / 0.4961 / 0.4951 | 0.2344 / 0.3613 / 0.3159 |
| `P3-rename` | 0.0938 / 0.0859 / 0.0664 | 0.0586 / 0.1055 / 0.1055 (**0.0898**) | 0.0000 / 0.0000 / 0.0000 | 0.5010 / 0.5063 / 0.5000 | 0.2583 / 0.3799 / 0.2671 |
| `P3-rename_sham` | 0.0547 / 0.0117 / 0.0039 | 0.0391 / 0.0195 / 0.0039 (**0.0208**) | 0.0000 / 0.0000 / 0.0000 | 0.4932 / 0.5024 / 0.4985 | 0.2871 / 0.3003 / 0.2871 |
| `P7-s20_ce` | 0.1094 / 0.0703 / 0.1328 | 0.1680 / 0.1016 / 0.1055 (**0.1250**) | 0.0000 / 0.0000 / 0.0000 | 0.5063 / 0.5063 / 0.4956 | 0.2769 / 0.3877 / 0.2563 |
| `P7-s20_low_lr` | 0.1289 / 0.1445 / 0.1836 | 0.1445 / 0.1094 / 0.1445 (**0.1328**) | 0.0000 / 0.0000 / 0.0000 | 0.5122 / 0.5010 / 0.4941 | 0.1558 / 0.2837 / 0.2031 |

- **Arkenstone stability-plasticity forks (`U48`, all `9/9` verified `parent_core_sha256 == child_initial_core_sha256`):**

| Arm | `s7 / s19 / s41` `child_a` (mean) | `s7 / s19 / s41` `child_p5` (mean) | Functional Norm Cap / Stability Note |
|---|---|---|---|
| `P7-ark_capped` | 0.1250 / 0.0781 / 0.2188 (**0.1406**) | 0.1357 / 0.1429 / 0.1429 (**0.1405**) | Cap active (`scale = 0.0241..0.0253`, clamping `unclipped_delta = 2.67..2.79` around `parent_norm ~ 67.36`). Retains parent `child_a` while blocking P5 learning. |
| `P7-ark_uncapped` | 0.0938 / 0.0156 / 0.0781 (**0.0625**) | 0.7857 / 0.7571 / 0.8214 (**0.7881**) | No cap; acquires P5 near `0.79` (`1.00` on 4 conditions) at substantial parent degradation (`child_a` drops vs `0.1510` parent mean). |
| `P7-ark_expaug` | 0.0000 / 0.0000 / 0.0000 (**0.0000**) | 0.7857 / 0.8071 / 0.7786 (**0.7905**) | No cap + exp-aug; highest P5 mean (`0.7905`) with total loss of parent direct accuracy (`0.0000` on all 3 seeds). |

- **Donor audits (`U48/U49`):** `arkenstone` cap test verified (`uncapped_norm = 1.4289` vs `capped_norm = 0.0118`); `cymek` manifest check rejects same-shape mismatch; `elendril` X1 parity passes across `42` core jobs (`backbone_parameters = 19,871,616`, exact backbone SHA `f3b444...14a0c`, `equal_input_max_diff = 0.0`, head overhead `0.001006`).

---

## 10. Structured Failure Attribution & Operational Summary

1. **S3 / S20 Systematic Grounding & Literal Probing (`P2`, `P3`, `P7-s20`):**
   - **(a) Observed metric:** Best confirm `fam4 = 0.1836` (`P2-margin-s7`), best confirm `fam8 = 0.0039`, `literal_raw = 0.1558..0.4063`, `irrelevant_rows fam4 = 0.0000`, `U19` dev `fam4 <= 0.0078` with train-fit up to `1.0000`.
   - **(b) Tested falsifiers:** `P2-margin`, `P3-address/rename/curriculum/refresh/low_lr`, `P7-s20` scale (`S20`), `U19` gradient-reversal check (`cos = -1.0008`).
   - **(c) Plausible explanation:** Byte transformers overfit surface query-token associations (`U19` high train fit, `0.0` irrelevant robustness) rather than learning relational lookup invariance.
   - **(d) Unresolved alternatives:** Whether structural categorical recurrence (`P4-categorical`) or explicit symbolic bindings embedded into byte-level sequence models resolve multi-hop composition without intermediate state supervision.
   - **(e) Raw locations:** `METRICS.json` keys `p2_dev`, `u19_gradient_reversal`, `p3_dev`, `confirm_custody_and_results`; local archive `AVENDI_Y3_EVIDENCE (1).zip` (`jobs/*/receipt.json`, `*_first_logits.npz`).

2. **P4 Unseen Length Extrapolation (`d3/d4` = `0.0000`):**
   - **(a) Observed metric:** `P4-categorical` scores `1.0000` on `d1/d2` and causal interventions (`donor`, `rewire`, `oracle = 1.0`) but `0.0000` on `d3/d4`.
   - **(b) Tested falsifiers:** Hard argmax parity check (`dev_hard_argmax = 1.0000` on `d1/d2`, `0.0` on `d3/d4`), matched-param `continuous` and `dense` baselines, and `_endpoint_only` ablation.
   - **(c) Plausible explanation:** Training exclusively on `depths=(1, 2)` leaves deeper recurrence steps uncalibrated / unvisited during supervised state rollout.
   - **(d) Unresolved alternatives:** Training with mixed depth curriculum (`depths=(1, 2, 3)`) or step-invariant transition tying on `d4` held-out generalization remains untested in Y3.
   - **(e) Raw locations:** `METRICS.json` key `p4_multihop_and_causal`; `studies/U28..U35/*/receipt.json`.

3. **P1 Blocked Y1 Restore & P6 Update-Count Mismatch (`14` Unverified Studies):**
   - **(a) Observed:** `P1-frozen` rejected all `6` Y1 lineages (`OSError: [Errno 30] Read-only file system` on mounted `analysis/jobs` + missing `ELENDRIL_V8_Y1_EVIDENCE.zip` logit parity); `P6-adapt` receipts recorded `1024 != 256` updates.
   - **(b)–(c) Cause:** Read-only `/kaggle/input/.../analysis/jobs` matched `locate_checkpoint_store_root` before `ELENDRIL_V8_Y1_FULL.zip`, and `_p6_job` logged inherited `global_update = 1024` rather than zero delta adaptation updates.
   - **(e) Raw locations:** `METRICS.json` keys `p1_frozen`, `p6_adaptation`, `failed_job_checks`, `unverified_studies`.

- **Per-stage worker timing (`sum = 6,803.65 s`, wall = `4,132.65 s`):** `P1` `6.05 s`, `P2` mean `58.86 s` (including `U19` short runs; full `P2` `87.9..101.8 s`), `P3` mean `123.59 s` (`P3-rename/rename_sham` ~`182 s`), `P4` mean `16.65 s`, `P5` mean `103.02 s`, `P6` mean `16.50 s` (`P6-diagnostic` ~`38.9 s`, `P6-adapt` eval-only), `P7` mean `138.07 s` (`P7-s20` `272.1..298.6 s` slowest overall; `P7-ark` `22.3..24.3 s`).
