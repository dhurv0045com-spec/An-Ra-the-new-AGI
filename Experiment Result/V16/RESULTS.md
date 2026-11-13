# V16 — First owner T4×2 mechanism screening slice (H2B-NORM / seed 7)

**Filed:** 13 November 2026 · **Status of the slice: EXECUTED (pipeline complete; analysis-summary writer defect found, diagnosed and fixed in source)** · **Cause status: `CAUSE_UNRESOLVED` (preregistered — no causal claim, no promotion)**

## 1. Run identity and provenance

| Item | Value |
|---|---|
| Campaign run_id | `b89a011f1b88470891394942af00eb18` |
| Phase / slice | `mechanism`, predeclared owner screening slice: arm `H2B-NORM`, seed `7` (subset identity `63bba4fd…dafb`) |
| Execution & science commit | `682c40e1b525424d805e6fb44f0f29afee6c756c` (identical; source pin verified by clean GitHub checkout) |
| Preregistration SHA-256 | `3a098a597684eb294f9eeccb5c00b66c3356a9f204036d0913aadfd15ffaa624` (byte-stable, untouched) |
| Protocol SHA-256 | `26617d4512f68435a0955d978bd89db3068867f0dddbf6a7e22ef20a3e0abe0e` |
| Data split SHA-256 | `22e5a91f7465094eeeb2cd1ef565c0708042109b742a70ad621a9f48c56dc5b6` |
| Public surface SHA-256 | `26c5e89483c45715a12fc07898b77a9dc1a1fcfb7c6efe5539ce5e405adadf69` |
| Hardware | 2× Tesla T4 (`GPU-a5a5d7fc…`, `GPU-ab4c3be4…`), driver 580.178.04, CUDA 13.0, torch 2.11.0+cu130, Python 3.13.16, CC 7.5 |
| Free disk at start | 20,917,035,008 bytes (12 GiB gate passed) |

## 2. Custody

- **Parent archive PASS:** `ELENDRIL_V71_FULL.zip`, exactly **1,958,029,910 bytes**, SHA-256 `1826c85b5a858d2a600661f7e5916f0fa812c55a8d53c37013c695df28a15d19`, admitted `via: parent_archive` (the original V7.1 CORE run `20c80f66e76e4b589aed2053afbddffc`, source `6104ffd7134c…`).
- Custody gate `CUSTODY_STATUS: PASS`, `failures: []`; S8 analysis archive not required in the mechanism phase (`s8_analysis_verified: false` by design).
- `EXECUTION_BINDING: BOUND` (write-once); matched seed-7 initialization identity `08131f4b…`; seeds 19/41 correctly unbound (`null`, not run in this slice).
- This run is the first V16 Kaggle session in which the **manual parent attachment path worked**: the input was attached in the interactive editor, mounted under `/kaggle/input`, and passed the byte+SHA gate. (The earlier batch attempt was refused by Kaggle — notebook outputs cannot be attached at runtime in Save & Run All sessions; recorded in `research/v7.16/OWNER_KAGGLE_LAUNCH.md`.)

## 3. Pipeline stages — all passed

| Stage | Lane 0 (`GPU-a5a5d7fc…`) | Lane 1 (`GPU-ab4c3be4…`) |
|---|---|---|
| Positive learning/recovery canary (V7 worker) | PASS, 41.7 s, recovery PASS | PASS, recovery PASS |
| Production profile `S3-canonical-normalizer` | PASS, 76.6 s, **0.3138 s/update**, 0.02919 s/episode eval, recovery replay 2→4 + exact payload compare PASS | PASS, 79.3 s, same recovery PASS |

The profile→recovery subprocess seam — which killed both lanes on 13 November with `TypeError: recover() takes 1 positional argument but 2 were given` — **passed on real T4s in this run** (fix `682c40e1` verified in production).

Data preparation (write-once, audited): procedural-event-worlds generator, **9,472 examples / 7,335 episodes**, 2,200,185 text bytes, max sequence 268 tokens; kind mix: route 2,069, known 1,032, unavailable 1,049, null 1,048, memory 723, missing 697, conflict 717; generator seed 606. Independent replay audit: **PASS** (13 reports replayed from recorded actions, no neural inference, no causal promotion).

## 4. Training job `H2B-NORM-s7` — COMPLETE

- **2,368 / 2,368 optimizer updates**, epochs 4, batch 16 (S3), 8 microbatches/update; `interrupted: null`, `recovery_required: false`.
- **Exposure matches the registered lock exactly:** 75,776 supervised tokens, **8,914,404 unpadded input positions**, 2,368 updates. Padded: 9,410,470 positions, 2,351,210,574 attention pairs; replay swaps 0.
- Loss **5.6844 → 0.7574**; LR 3.0e-4 → 3.0e-5 (scheduled decay); wall **619.2 s** (campaign wall 828.0 s; GPU reservation 870.1 s; capacity forecast 2,286.7 s — ample headroom).
- Peak CUDA **177.5 MB allocated / 211.8 MB reserved**; parameters **3,084,800**; model: dense decoder, 4 layers, width 256, FFN 768, 4 query heads / 1 KV head, QK-norm (affine), RoPE 10,000, ctx 512, vocab 260, tied embeddings.
- Final checkpoint SHA-256 `f0199921a19e6d1d9e28524a852d96c0d3791e0bf6cf88c151da60b2ca59fdcd`.

## 5. Results — final free-decoding evaluation (epoch 4, update 2,368; the preregistered primary)

Learning curve on `base` (free decoding): **epoch 1: 42/112 (0.3750) → epoch 2: 52/112 (0.4643) → epoch 4: 60/112 (0.5357)**.

| Cohort (final free decoding) | Correct/Episodes | Accuracy | Invalid |
|---|---|---|---|
| base | 60/112 | 0.5357 | 2 |
| confirmation | 64/112 | 0.5714 | 2 |
| balanced-payload | 18/32 | 0.5625 | 0 |
| robustness-row-order (row reversal) | 112/224 | **0.5000 (exact)** | 4 |
| robustness-distractor | 25/112 | 0.2232 | 2 |
| intermediate-vs-sham | 4/32 | 0.1250 | 1 |
| memory-manipulations | 6/64 | 0.0938 | 34 |
| query-crossed | 22/96 | 0.2292 | 42 |
| query-irrelevant | 0/16 | 0.0000 | 0 |
| hop-1 / hop-2 / hop-2-clamped / hop-3 / hop-4 | 0/16, 0/16, 0/32, 0/16, 0/16 | 0.0000 | 0, 0, 0, 6, 16 |
| robustness-alias | 1/112 | 0.0089 | 0 |
| robustness-template | 2/112 | 0.0179 | 33 |

- **Row-reversal aggregate accuracy is exactly 0.5000 (112/224)** — the registered parity value for the H2B design. Per-world paired consistency is `UNAVAILABLE` in this export because all 112 shared worlds are ambiguous under the world-id pairing rule, so the parity claim rests on the aggregate alone. The row-structural signature (parity-level row-order insensitivity with near-zero alias/template) is present; it is a signature of what the policy binds to, not of capability.
- **Alias (0.0089) and template (0.0179) generalization is near zero** — the policy did not memorize surface forms; it binds to row structure.
- **Multi-hop composition is zero** on every hop cohort, with invalid-action load growing with hop depth (hop-4: 16/16 invalid).
- Invalid actions concentrate on out-of-base cohorts: query-crossed 42/96 (43.8%), memory-manipulations 34/64 (53.1%), robustness-template 33/112 (29.5%).
- Grammar-constrained decoding (diagnostic, not primary): base 61/112, confirmation 64/112, hop-3 1/16, **invalid = 0 in every cohort**. Grammar removes invalid actions but adds no accuracy — invalids are a decoding-structure problem, not a hidden-capability signal.
- Total invalid at final free decoding: **142 / 1,120 = 12.68%**, inside the V7.15-derived rate ceiling (143 for this record scope); grammar invalid total 0 ✓ (the preregistered invalid-action rule will be adjudicated at promotion, not here).

## 6. Engineered comparator context (same data, no learning)

| Policy | base | hop-1…hop-4 | confirmation |
|---|---|---|---|
| `fixed` (hand-written public algorithm, queries allowed) | **112/112** | **16/16 each** | 112/112 |
| `always-query` | 112/112 | 16/16 each | 112/112 |
| `no-query` (public algorithm, no queries) | 89/112 | 16/16 each | 84/112 |
| `recency` (hop-aware last-value heuristic) | 68/112 | 16/16 each | 64/112 |
| `random` (seeded uniform) | 5/112 | 0–2/16 | 6/112 |
| `query-blind` (constant "A") | 1/112 | 0–2/16 | 4/112 |

The entire registered evaluation surface is solvable at ~100% by the hand-written public algorithm, and every hop cohort is solved exactly by simple non-learned heuristics. The trained H2B-NORM policy therefore sits far below the achievable ceiling on `base` and loses every hop cohort to a five-line heuristic.

## 7. Interpretation (descriptive, preregistration-consistent)

1. **Baseline established.** H2B-NORM is the registered *control* arm (blocked-CE baseline exposure, canonical-normalizer mechanism). This slice fixes the control point the full 21-job campaign will measure mechanism arms against: base 0.5357, confirmation 0.5714, hops 0.0, row-reversal 0.5000.
2. **Mechanism signature present, capability absent.** Aggregate row-reversal accuracy sits exactly at the registered parity value (0.5000) and alias/template scores are near zero, showing the canonical-normalizer binding is row-structural; zero multi-hop composition shows the binding did not become a traversal procedure.
3. **Failure analysis (per filing rules).** (a) Observed: hop cohorts 0/16 with rising invalids; base 0.5357 < registered promotion per-seed threshold 79/112 (0.7054). (b) Tested falsifier: the cohort suite itself plus comparator controls; results reproduce the registered row definitions and the public-algorithm ceiling. (c) Plausible explanation: after 2,368 blocked-CE updates the policy fits single-row base structure but no route-traversal procedure; high final output entropy (3.23 bits on base) suggests unresolved distribution over answers rather than saturated fitting. (d) Unresolved alternatives: capacity/schedule/exposure-window effects, and whether any registered mechanism arm (H1/H4/H5/H6) changes the hop outcome — exactly what the remaining campaign must test. (e) Raw artifacts: `ELENDRIL_V716_ANALYSIS.zip` (hash below), all epoch/grammar/cohort reports inside it.
4. **No causal claim, no promotion** — `promotion: "NONE"`, `CAUSE_UNRESOLVED` as preregistered.

## 8. Defect found in the analysis-summary writer (fixed in source after this run)

- **Observed:** `summary.json` status `PARTIAL`; 2 errors on `jobs/H2B-NORM-s7/epoch-2.json` and `epoch-4.json`: `'>' not supported between instances of 'int' and 'tuple'`. Epoch-2/epoch-4 cohort rows were dropped, and the published arm aggregate (`42/112` base, `0/16` hop-3) was **epoch-1's evaluation mislabeled as final**.
- **Mechanism:** `tools/elendril_v716_report.py` compared the integer epoch against the `(epoch, data)` tuple used for final-epoch tracking; every epoch file after the first raised `TypeError` inside the per-file guard.
- **Impact on this export:** training, custody, checkpoints and every raw cohort report are intact (all 59 internal SHA256SUMS verify); only the derived summary aggregates were wrong. The corrected final values in §5 were recomputed directly from `epoch-4.json`.
- **Fix:** compare against `final_epoch[0]`; regression test `test_v16_report_tracks_final_epoch_across_multiple_epoch_files` reproduces the exact failure (errors empty, all three epoch rows present, aggregate = epoch-4). Full suite 443 passed, 1 environmental skip. Future runs will publish `AVAILABLE` with correct final aggregates; this run's pinned tree keeps the defective writer, so its `summary.json` stays as-is (never rewritten silently).

## 9. Artifacts and custody locations

- Owner export analyzed: `ELENDRIL_V716_ANALYSIS.zip` — **6,030,579 bytes**, SHA-256 `3451e93300bbbbf14b8db13470d1544c3681c8733db2abddcc0431c76fcdb7c4`, 61 entries, internal `SHA256SUMS` **59/59 verified**; scope: "complete stable output snapshot; includes failed/partial receipts; not a model promotion". The ZIP itself stays outside Git (filing rules); curated headline metrics are committed as `METRICS.json` beside this report.
- Kaggle output directory: `/kaggle/working/elendril-v716-682c40e1b525-mechanism/` (ANALYSIS + FULL exports; **download and retain the FULL ZIP** — later promotion-phase custody requires the mechanism output).
- Dataset provenance hashes (job receipt): cases `dabc2d7c…`, examples `d9f4a4ea…`, samples `781e9423…`, tokenizer `8343d8d2…`, plan `010f28a3…`.
- Repo records: `research/v7.16/HYPOTHESIS_DESIGN_AND_VERIFICATION.md`, `research/v7.16/OWNER_KAGGLE_LAUNCH.md`, `progress.md`.

## 10. Limitations

- One arm × one seed (owner screening budget); not the registered 21-job matrix; no promotion gate was run (`promotion: NONE`).
- Hop cohorts are exactly solvable by non-learned heuristics — they sharpen the failure contrast but cannot alone demonstrate composition capability.
- The published `summary.json` in this export carries the epoch-1 mislabel described in §8; corrected values here are recomputed from raw epoch reports, not from the defective summary.
- `origin/main` contains one merge (`b620f1c2`, PR #41) not in `opposing-star`; irrelevant to this run — execution fetched the pinned SHA directly.
