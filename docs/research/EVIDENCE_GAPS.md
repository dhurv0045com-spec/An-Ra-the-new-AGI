# EVIDENCE GAPS

**Date:** 2026-09-13. Everything this audit could not verify, with the reason and the cheapest closure. Nothing here is concealed elsewhere; the ledger marks affected entries.

> **Post-snapshot resolution (2026-09-13):** former gap G1 is now closed. The R1C launcher was rebound to reachable Amendment-2 commit `b850861545f79e219b55d6403f81e74f92f1592e` and the Colab pin was repaired on `cymek-500m-readiness` (`24ca7f3edf8e1f8affbe077ce7225b8dbb7a7d69`). `CYR-GPU-014-R1C` subsequently completed 24/24 arms with verdict `SOFTMAX_COMPETITION_NOT_SUFFICIENT`. Source bundle SHA-256: `2a9359e49f792f962774e42b4e5825b93fb7c91ddcad8f530c1c6f16c6f9bf9e`. See `R1C_FINAL_EVIDENCE_2026-09-13.md`. The row below is retained as a forensic record of the original audit finding, but it is no longer an active blocker.

## 1. Provenance / launch-blocking

| # | Gap | Consequence | Cheapest closure |
|---|---|---|---|
| G1 | **RESOLVED after this audit snapshot.** Originally: CYR-GPU-014-R1C launcher pinned unreachable `6653b4ce3a7e61b8fdffd3236c62c67622084dec`. | Historical launcher would fail a fresh clone; no longer current after the explicit Amendment-2 rebind and notebook repair. R1C is now executed/complete. | **CLOSED.** Reachable engineering fix `b850861…`; repaired launcher commit `24ca7f3…`; final bundle hash recorded above. |
| G2 | **Exact-head test receipts are STALE-BY-DESIGN**: `v5_training` constants were single-sourced after the receipts' tested commits (documented in EXPERIMENT_LOG, not hidden). | Receipt meta-checks do not certify the current head until the next full suite run. | Run the full suite once and refresh receipts. |
| G3 | ARK-019 **RUN_READINESS_V4.json still records `scientific_result_status: NOT_EXECUTED`** while `FINAL_RESULT_AUDIT_V4.md` transcribes an executed, externally audited result. Two repo documents disagree. | The Guardian question cannot be settled from the repository alone. | Commit + byte-audit the V4 raw bundle (or rerun) and reconcile both files. |

## 2. External / inaccessible evidence

| # | Gap | Consequence |
|---|---|---|
| G4 | **ARK-020 V4 external execution state**: output root `/content/drive/MyDrive/genisis-arkenstone/ARK020_V4_CONTINUAL` is operator-side Google Drive, inaccessible from this audit. No result artifact exists on any branch; launcher work continued through 2026-09-13 (deterministic-kernel fix `f4244e2`, A1.1–A1.3 amendments). | ARK-020 V4 classified strictly `IMPLEMENTED_NOT_EXECUTED` from repo evidence. If partial bundles exist on Drive, they were not auditable here. |
| G5 | **ARK-019 V4 raw bundle** (bundle for `GUARDIAN_CONTINUAL_PROXY_CANDIDATE`) not in repository. | ARK-019 V4 capped at SUPPORTED (transcribed external audit). |
| G6 | ARK-011 historical bundle hashes and ARK-015/016/017 raw ZIPs: the repo keeps distilled receipts + RESULT.json files (Tier 0/1 mix) — raw bundles are on operator storage. | Metrics cited are from committed RESULT/receipt files; per-row re-verification beyond committed hashes was not possible. |
| G7 | Citadel T1C raw session receipts live under `docs/citadel/tpu_receipts/t1c_session/` on citadel — audited via BRAMASTRA's transcription of them rather than re-parsed JSON in this pass. | T1C numbers quoted second-hand (though BRAMASTRA's audit included the raw-vs-prose denominator correction). |

## 3. Unreachable / at-risk evidence (git forensics)

| # | Gap | Contents at risk |
|---|---|---|
| G8 | **Stash `301f5f2` untracked parent `87ea5d6` holds the ONLY copy of the CYR-GPU-006 smoke campaign**: `campaign_receipt.json` (schema `anra-cyr-gpu006-campaign/v1`, supersedes CYR-GPU-005), `CYMEK_GPU_RESEARCH_V6_RESULTS.zip`, 36 arm checkpoint files. | One `git stash drop` or `git gc --prune` destroys it. Export to `artifacts/` (as an engineering-receipt archive, not scientific evidence) before any gc. |
| G9 | **SENORA P35-CMS-1 + CAD program exists only in unreachable commits** rooted at esoes tip (`30a8fa7` chain, 11 commits): P35_A/P35_B decision receipts, CAD preregistration, `cld_trajectory_metrics.json`, `cld_world2_phase_transition.json`, run receipts, triquetra-bridge causal records, cluster sbatch launchers. Zero senora paths on any live branch. | Recovery = `git fetch <local-clone> 30a8fa7` into a kept ref or bundle export before gc. No executed campaign is evidenced; the value is the frozen design/power contracts and dry-run receipts. |
| G10 | CYR-GPU-005 frozen COMMIT A executable `anra_v5/cyr_gpu005_run.py` (blob `41073505`) differs from the live version and survives only in unreachable `fcd9178`. | Frozen-executable provenance for the 005 lineage is incomplete on live refs. |
| G11 | R1C pre-rewrite notebook revisions (`8162c22` @ `e444249`, `8eb3b09` @ `8adedab`) are unreachable; live `f2c27a6` carries `bb9789b`. | Historical binder chain for the R1C launcher is not ref-anchored. This remains a provenance-history caveat, but no longer blocks the completed R1C result because the executed operator path is separately hash-bound and the final bundle is recorded. |
| G12 | `milestone/0001-honest-loop` (`20d8841`) is contained only by `origin/core-exp` + the tag. | Single-branch containment for a program milestone. |
| G13 | The two history shards share no merge-base; pre-2026-09-05 cymek history is only via local branch `cymek` (`4abeaeb`), which is **not pushed**. Local-only branch = single point of failure for old-shard evidence (incl. the founding-negative receipt chain `core-vnext@054619f`). | Push `cymek` (read-only) or export a bundle. |

## 4. Deleted-branch folklore

| # | Gap |
|---|---|
| G14 | ~15 remote branches were deleted shortly before 2026-09-13 (scratch-ignore-3/4/5, senora, stop-this, temp-do-not-use, temp-final, temp-ignore, temp-ignore2, temp-v7-results-staging, temp-v8-ignore, temporary-branch-z, tmp-never, v8-prep-scratch, why-branch). No reflogs survive locally (fetch --prune). Name→content mapping is confirmed only for `senora` (G9); the ARK-018 edition-1 and ARK-020 audit clusters are consistent with the temp-v8 family but unprovable locally. GitHub may still serve some deleted-tip SHAs if they were ever pushed — untested per-SHA. |

## 5. Metrics / compatibility caveats

| # | Gap |
|---|---|
| G15 | T1C raw core denominators are **0/1,000**, not the 0/500 widely carried in prose (BRAMASTRA audit correction adopted here). Older documents repeating 0/500 are wrong on the denominator. |
| G16 | ARK-015 robustness mean is quoted as "~0.468" in prose; the exact final mean lives in ARK-015/RESULT.json and was not re-parsed row-by-row in this pass (quoted from CURRENT_STATE + NEGATIVE_RESULTS). |
| G17 | BRAMASTRA "this-regime replay rescues retention: rejected" is carried only by the cross-branch negative-results carry list (cymek `NEGATIVE_RESULTS.md`); the original receipt was NOT re-located on BRAMASTRA (its RESULTS.md/EXPERIMENTS.md do not cover it). Tier-3 until found. |
| G18 | CORE-MC-v9/v10/v11 promotion receipts were not re-audited under current standards (pre-restart program on the V4 substrate that later failed readiness v2). Historical INCONCLUSIVE classification could change either way on a real audit. |
| G19 | Citadel 500M production-path audit is pinned to `28bf57a` (2026-09-06); cymek-500m-readiness has advanced substantially since. The MISSING/AMBIGUOUS classifications are stale in the direction of optimism OR pessimism — an updated audit is needed, not assumed. |
| G20 | ARK-017 secondary screens (CAP4X/CAP16X/replay-1/32/replay-1/64) are single-order per the ledger — treated as clues, not dose estimates; no further caveat needed beyond what RESULT_V2.md already states. |
| G21 | Seed counts: several executed experiments are single-seed by documented hardware calibration (CYR-012-R1; ARK-017 secondary screens). Where the JSON says R0/R1 this is deliberate, not an oversight. |
| G22 | ARK-021/022/028 portfolio claims (DEVELOPMENT_READY/DESIGN_READY/blocked-on) were taken from the portfolio document + directory listings; test-suite execution status for ARK-021 was not re-run in this audit. |
