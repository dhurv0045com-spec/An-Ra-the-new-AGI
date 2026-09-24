# Gandiva engineering status

## Owner Kaggle K8 results and repair — 24 September 2026

The owner supplied four integrity-valid ZIP exports from campaign
`k8-b796a20ecc04`, instance `b5e0e88d1ca3`. They contain one run, not four.
That run used source closure
`01eeada0ab124aa167b1dd7bb2e7ee0b16c0337d4a369ba1953311b44d9a8177` and data
identity `d64ff76188b612f5fa2744bd4739420e3f708a0d88c43be4e8fd1f0392f36681`.
Those identities differ from the pre-campaign full-build report below, so the
Kaggle results and that report must not be treated as evidence for each other.

E0, E1, E2 and E3 executed. E1 produced no held-out successes (0/32 per arm);
E2 policy, memory and workspace each had 0/256 successes with all episodes
truncated, while the symbolic comparator scored 208/256. E3 verified tool
execution receipts but had 0/4 retention successes per job. Both E4 workers
failed before updating because the architecture probe generated CPU indices
for a CUDA model. E5 did not run. This run is negative capability evidence for
its registered tasks, not AGI or RSI evidence. Full analysis and hashes are in
[`reports/K8_EXPERIMENT_20260923/RESULTS.md`](reports/K8_EXPERIMENT_20260923/RESULTS.md).

On this branch, E4 now samples probe tokens on the restored model's device.
The results-only packer also rewrites embedded restore manifests to reflect
checkpoint weights excluded from ZIPs, and normalizes archive member paths on
Windows. Focused checks passed 56 tests and 12 subtests. A temporary repack of
the supplied ZIP confirmed the eight omitted checkpoint payloads are now
reported accurately; the source ZIPs were untouched.

On 25 September, a fresh full bundle validated with identity
`20c3d876136ee9023a085cc11ae67c3929a0c2520c830c647081c988a209cfbd`. The first
no-update verifier attempt hit a Windows permission error while pytest
enumerated its default temp directory; the affected F04 group passed 67 tests
with an explicit writable temp root. The full verifier then reported
`VERIFIED`: F01–F24, all seven test groups, and all seven production-interface
exercises passed, with zero local optimizer updates. It marks
`ready_for_owner_experiment: true`; source closure is
`7f4a24d1b70639f5231cfec283ad0eb4a68feb977512ecdc58459998a70c0457`, source
commit `c6c64c89eb568c44cc1deed683a089580ff29494`, and verification duration
216.61 seconds. The report is
[`reports/FINAL_K8/gandiva-post-e4-c6c64c89-20260925-pytest-temp/build_verification.json`](reports/FINAL_K8/gandiva-post-e4-c6c64c89-20260925-pytest-temp/build_verification.json).
The initial environment-limited report is preserved at
[`reports/FINAL_K8/gandiva-post-e4-c6c64c89-20260925/build_verification.json`](reports/FINAL_K8/gandiva-post-e4-c6c64c89-20260925/build_verification.json).
G01–G04 remain runtime gates for Kaggle E0. This establishes code/build
readiness for the current closure, not a learned capability or AGI result.

## Pre-campaign full build verification — 24 September 2026

Before the owner campaign, the `Gandiva` checkout was synchronized with
`origin/Gandiva`; a deterministic full bundle was generated and validated with data identity
`7b354e0ac63dfd286e85df19ca7b5b076b8a119fd2184193239099acbf5d0bc0` and source
closure `adb0a9f79b8a80231de5e6534f7604ad838a0967337a9e75831e3aa595d1831a`.

A fresh `verify-build --no-updates` at that pre-campaign source passed all 24
requirements, seven check groups, and seven production-interface exercises
with zero local optimizer updates. Focused RSI/Kaggle/operational checks passed
68 tests and 12 subtests. The owner later ran the Kaggle campaign described
above. Runtime gates and scientific claims must be assessed against the exact
source/data identities used by each run.

## Three-family cognition rehearsal — 23 September 2026

Source revision `ec39a32811492f31047f504e4d066c8bbc74936e` expands the
zero-update production trace and integrated E2 fixture rehearsal to every
registered family: rule-inquiry, inventory, and program. The random-initialized
model made four real model-origin calls per family under the explicit cap; each
trace truncated with unknown success. The model stayed in evaluation mode,
retained zero gradients, and committed zero optimizer updates. This is honest
integration evidence, not evidence of useful cognition.

The new full build report
[`reports/FINAL_K8/gandiva-cognition-all-families-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-all-families-20260923/build_verification.json)
passes F01–F24 and all seven check groups in 81.985 seconds. Its data identity
is `79c9706d122050cc1e8f5e6a3363af68005fc1206db0451b40ea807121c8d7f8`; source
closure is `e603b5bbc6d3b7e4e7841482fe0b2327d9d8e253c015f5ef0f765956abb4f378`.
The report records `dirty=true` because the pre-existing user edit to
`tests/test_research_k8_real.py` was preserved. The fixture rehearsal now has
three matched groups across the E2 modes; those rows are explicitly fixture
evidence. Owner runtime gates G01–G04 still require the real Kaggle two-T4 E0
session. See
[`reports/GANDIVA_F6_FAMILY_REHEARSAL_20260923/HANDOFF.md`](reports/GANDIVA_F6_FAMILY_REHEARSAL_20260923/HANDOFF.md).

## Cognition privacy follow-up — 23 September 2026

Follow-up commit `719bcbcf5b84e80900e87f21a0cea2f49d550ff1` adds two
regressions for private-state isolation: changing unreceived private fields
cannot change the world-model prompt, and two environments with opposite hidden
answers produce the same prompt but opposite outcomes. It also fixes a false
positive in the older C04 diagnostic: the original check compared two
depth-one roots rather than observing a depth-two call. The corrected probe
exercises one real second-depth prediction. Its six recorded defect flags are
false on this revision.

Focused CPU-only checks passed: 50 cognition tests, including all 11 planning
tests, plus 7 RSI integration tests. No optimizer update, GPU run, or learned
capability was involved. This closes those regression gaps only; it is not a
complete acceptance of the broader cognition-foundation F1–F6 contract. See
[`reports/GANDIVA_COGNITION_FOLLOWUP_20260923/HANDOFF.md`](reports/GANDIVA_COGNITION_FOLLOWUP_20260923/HANDOFF.md)
and the [compact probe receipt](reports/GANDIVA_COGNITION_FOLLOWUP_20260923/probe_v2.json).

## Latest verified build — 23 September 2026

The latest code revision, `ab580776c4fab6398f23d503413eace41d50dbe7`, fixes the
cognition prefix contract: auxiliary channels use the canonical initial-state
prefix, while trajectory answer batches retain their full history. E3 now uses
the same canonical prefix. The regression suite confirms those contexts remain
distinct and correctly encoded.

The source-bound, no-update build report is
[`reports/FINAL_K8/gandiva-cognition-final-a6e99c9b-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-final-a6e99c9b-20260923/build_verification.json).
It reports all F01–F24 checks passing, all seven pytest groups passing, the
notebook interface exercise passing (24 cells; 12 Python cells; no hard-coded
paths or shell magics), and zero optimizer updates. The verifier completed in
113.375 seconds. Its source-closure SHA-256 is
`88e130fdc65c21afc97f92f61e62c610365643839a5ae4a127a4b1b7adccb73c`; the
verified full-bundle identity is
`79c9706d122050cc1e8f5e6a3363af68005fc1206db0451b40ea807121c8d7f8`.

Focused validation after the cognition fix passed **74 tests, 1 skipped, and 8
subtests**. The report marks the source tree dirty because the pre-existing
user edit in `tests/test_research_k8_real.py` was preserved. The implementation
did not stage or replace that edit.

## Owner experiment and limits

The normal self-sustaining Kaggle notebook is
[`../notebooks/bramastra_k8.ipynb`](../notebooks/bramastra_k8.ipynb). The
notebook contract was build-verified before the owner campaign. That campaign
qualified two CUDA workers and committed E0–E3 updates, but used an older
source/data closure and stopped on the E4 defect described above. It therefore
does not qualify the current checkout or establish a successful full E0–E6
run. The next owner campaign must use the updated branch, pass fresh
source-bound verification and E0, and use a new allocation/run ID. No AGI or
RSI capability is claimed by the available results.

The separate 100M TPU path remains a preflight and engineering design. It has
not been trained or qualified on Kaggle TPU and is not part of the normal K8
notebook campaign. See
[`TPU_100M_COGNITION_PROGRESS.md`](TPU_100M_COGNITION_PROGRESS.md).

## Continue from here

Use [`FINAL_K8_PROGRESS.md`](FINAL_K8_PROGRESS.md) for the detailed progress
cursor and [`FINAL_EXPERIMENT_EXECUTION.md`](FINAL_EXPERIMENT_EXECUTION.md) for
the active F01–F24 build contract. Keep generated failure reports as audit
evidence, do not overwrite earlier reports, and preserve the user's uncommitted
test edit.
