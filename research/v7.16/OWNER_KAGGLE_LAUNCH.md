# V16 Kaggle owner launch — first screening slice

This notebook is explicitly **enabled** for the user to launch manually:
`notebooks/ELENDRIL_V716_OWNER_LAUNCH_KAGGLE_T4X2.ipynb`.

- The protected canonical notebook remains default-off: `ELENDRIL_V716_HYPOTHESIS_KAGGLE_T4X2.ipynb`.
- Owner-run notebook configuration: `RUN_STUDY=True`, `PHASE=mechanism`, `ARM_SUBSET=H2B-NORM`, `SEED_SUBSET=7`, `MINUTES=330`.
- Add the original **V7.1 CORE FULL** ZIP (NOT V7.15 scale-S8) under Kaggle **Add Input**. The run preflight streams SHA-256 over attached ZIP files until it finds the exact historical parent digest `1826c85b5a858d2a600661f7e5916f0fa812c55a8d53c37013c695df28a15d19`. Missing or mismatched archives produce a **NO-GO before code fetch and training**.
- In Kaggle settings choose GPU **T4×2** and enable Internet for immutable GitHub source checkout.
- The source is frozen to `3ef9dec7d09a40a4b960bce0a7eece6f32abbce2` and must not be substituted.
  This pin contains the recovery-dispatch repair (`682c40e1b525424d805e6fb44f0f29afee6c756c`,
  verified on the 2026-11-13 owner T4 run) plus the analysis-summary final-epoch fix; the earlier
  `6614864f09066e5626d5c7407cc6d7a80e830144`
  is the known-defective launch source recorded in the incident below and must never be rerun.
- A successful preflight is not a claim of training success. A verified run will produce V16 ANALYSIS and FULL exports; retain both and inspect receipts.
- No paid GPU runtime was launched by the author of this notebook.

The auto-discovery never accepts an arbitrary ZIP or substitutes analysis-only archive for parent custody. Do not bypass failed custody gates. If no authentic historical V7.1 CORE FULL archive is available, locate that original artifact first.

## Diagnosing an attached V7.15 notebook output

The input sidebar may show `ELENDRIL_V715_KAGGLE_T4X2` and individual
`scale-s8` files. These visible names do not establish a historical
parent FULL archive. The owner-launch notebook now prints dataset folders,
file counts, up to 15 sample paths, and hashes of any ZIP candidates before
any GPU training. It saves a JSON record at
`/kaggle/working/V16_INPUT_DIAGNOSTIC.json`, including exact observed hashes
and whether the required parent SHA-256 matched. A missing ZIP or mismatch is
an explicit custody NO-GO, not an experiment failure. Do not bypass the gate.

## Exact parent archive confirmed from V7.1 source custody

The input attached on 13 November was the wrong **V7.15 scale-S8** notebook output:
its FULL ZIP weighed 2,933,151,710 bytes with SHA-256
`a5a41267e034eb31ed0f453c94863087b291ec38571a38a126eba0e649347b5f`.
The original parent is from **V7.1 CORE**, run
`20c80f66e76e4b589aed2053afbddffc`, source
`6104ffd7134cb4cf4b3f340bd060d24a1e2dd200`,
file `ELENDRIL_V71_FULL.zip` of **1,958,029,910 bytes** and
SHA-256 `1826c85b5a858d2a600661f7e5916f0fa812c55a8d53c37013c695df28a15d19`.
The filename is identical between phases; only byte size and hash distinguish
them. This original artifact is not committed to GitHub and must be located in
the prior owner V7.1 CORE Kaggle output or external owner backup.

## 13 November 2026: real T4 profile-recovery dispatcher incident

The V16 owner session confirmed the original V7.1 CORE FULL ZIP SHA-256, both
physical T4s, source pin and dataset preflight. It stopped during the S3
production-shaped profile on both lanes with `TypeError: recover() takes 1
positional argument but 2 were given`. This was a deterministic V16 worker
interface defect, not missing/incorrect data or a scientific learning result.
The recovery worker now accepts the shared (request, deadline) signature used
by the CLI dispatcher, enforces its deadline during replay, and has CPU
regression checks for all three CLI modes (profile, recover, train). The
source-pinned owner notebook must be repinned to include this repair; prior
notebook source commit `6614864...` is known defective and must not be rerun.

## Automatic historical Kaggle-parent attachment

The V16 OWNER_LAUNCH notebook now tries existing mounted inputs first. If the
original V7.1 CORE FULL ZIP is not mounted, it calls Kaggle's supported
`kagglehub.notebook_output_download` for the exact historical file under
`ankitrajm7c/elendril-v71-kaggle-t4x2`. The retrieved file must match the
recorded 1,958,029,910 bytes AND SHA-256 or the run ends with NO-GO before
training. Automatic access may fail for private, removed or unavailable notebook
outputs and cannot be guaranteed. No backup training data is fabricated.
The code recovery bug is repaired in source `682c40e1b525424d805e6fb44f0f29afee6c756c`. Download the updated
owner notebook; earlier source `6614864f09066e5626d5c7407cc6d7a80e830144` is known broken and must not be reused.

## 13 November 2026: Kaggle refuses runtime attachment in batch sessions

The owner batch run with no attached input stopped at the input gate in ~10 s (zero GPU
seconds, zero optimizer updates). KaggleHub attempted the recorded automatic attachment and
Kaggle refused it:

`BackendError ... "New Notebooks cannot be attached in non-interactive sessions, found no
versions attached for Notebook [ankitrajm7c/elendril-v71-kaggle-t4x2]"` (error code 9).

This is a platform rule, not a custody or code defect: a Save & Run All session cannot attach
another notebook's output at run time. The automatic fetch remains a fallback that only works
in an interactive editor session. The supported path is to attach the parent before running:

1. Open `ELENDRIL_V716_OWNER_LAUNCH_KAGGLE_T4X2.ipynb` in the Kaggle editor.
2. Right panel **Add Input > Notebook Output**, select the output of
   `ankitrajm7c/elendril-v71-kaggle-t4x2` (the V7.1 CORE run
   `20c80f66e76e4b589aed2053afbddffc`) whose output contains `ELENDRIL_V71_FULL.zip`.
3. Keep **GPU T4x2** and Internet enabled, then **Save Version > Save & Run All**; the batch
   run mounts the attached input under `/kaggle/input` and the byte+SHA gate verifies it.
4. If the notebook output cannot be attached at all, download `ELENDRIL_V71_FULL.zip`
   (1,958,029,910 bytes) from that run's Output tab, upload it as a Kaggle dataset, and attach
   that dataset instead; the gates verify bytes and SHA-256 wherever the ZIP is mounted.
