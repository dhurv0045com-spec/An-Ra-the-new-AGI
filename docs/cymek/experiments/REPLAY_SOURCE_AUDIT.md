# Replay source-selection repair and interpretation correction

Baseline: `d2abbdf87c4ef7c6c15341918ddd4bc3d0080fbe`.
Scope: local CPU engineering tests only; no training or sealed evaluation.
This record supplements, rather than rewrites, CYHEX_INTEGRITY_AUDIT.md and
its preserved initial-replay artifact.

## Reproduced defects and repair

`anra_v5/initial_replay.py:build_initial_flat` inserted the frozen root into
`sys.path` but did not call its existing module-cache purge. Python therefore
reused already-imported worktree builders. A regression replaces the cached
builder with one that raises `cached worktree builder used`: it failed before
the repair. The existing purge is now called; the test passes and checks that
the caller's cached module and search path are restored afterward.

A second regression selected an absent frozen directory. Previously the replay
silently fell back to the worktree and did not raise. It now requires the three
entry-source files before changing import state and raises FileNotFoundError.

These checks are not complete source attestation: transitive dependency origins,
source-tree hashes, concurrent imports, and partially populated frozen packages
need further coverage. Do not advertise the loader as fully isolated or fully
fail-closed merely because these two regressions pass.

## Exact validation

Working directory: the isolated target-branch worktree.
Prefix for all commands:
`env -u PYTHONPATH -u PYTHONHOME OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 py -3.14`

- `-m pytest tests/test_v5_initial_replay.py::test_replay_does_not_reuse_cached_worktree_builder -q`
  — before repair: 1 failed in 5.08s, cached builder assertion.
- `-m pytest tests/test_v5_initial_replay.py::test_missing_frozen_source_fails_without_worktree_fallback -q`
  — before repair: 1 failed in 7.04s, expected FileNotFoundError not raised.
- `-m pytest tests/test_v5_initial_replay.py tests/test_v5_cyr_repro_ledger.py tests/test_v5_cyr_gpu012_duplicate_members.py tests/test_v5_cyr_gpu012_evidence.py tests/test_v5_cyr_gpu012_closure.py -q`
  — after repairs: 23 passed in 7.38s.
- `git diff --check` — passed.

The test suite includes the existing bundle-backed replay integration test.
The preserved historical artifact was not overwritten or reissued. No full-suite,
remote-CI, new scientific experiment, or historical replication claim is made.

## Interpretation corrections (superseding earlier prose)

- Initial historical tensor identity remains UNKNOWN after the norm replay.
  A matching scalar norm and deterministic local reconstruction do not recover
  an absent historical tensor hash. Kernel selection is not the only unknown.
- Similar relative-displacement magnitudes do not prove similar trajectories:
  vectors with the same magnitude can point in different directions.
- Byte-identical source files do not prove identical runtime math or numerical
  behavior. The source comparison excludes some code changes, not all causes.
- A controlled GPU comparison is one possible test of a specified environment
  hypothesis, not the only direct test or a default full-campaign recommendation.
- FORMATION-MUX's historical returned-bundle audit lines 40–43 use 'exonerations'
  despite a zero baseline. Those claims are not licensed; the applicable current
  interpretation is INCONCLUSIVE_AT_ZERO_BASELINE. Frozen endpoints stand.

## Other inspected evidence and incomplete work

The local FORMATION_MUX_001_RESULTS.zip inventory contained 711 members,
checkpoint receipts, and no names ending in .pt or .bin. This was an inventory,
not a hash/schema/contents validation or exhaustive checkpoint-custody search.
No sealed rows were read. Required checkpoint identities and recovery locations
remain to be established before diagnostics; do not rerun the campaign.

A separate read-only HORM-001/002 review was dispatched, not yet accepted as
verified findings in this milestone. Those lines remain unaudited here.
Required startup reading and comprehensive source-difference review remain
incomplete: a bulk-reading operation was blocked before execution and was not
retried. Earlier commentary calling all startup reading complete was incorrect.

## Next action

Complete replay source attestation: reject partial-tree fallback and verify every
project module used by the builder against the selected frozen tree, including
exception-path state restoration. Acceptance: direct negative regressions plus
unchanged measured replay digest under the verified source tree. Local CPU work
requires no additional authorization; no pilot/frontier or production gate changes.
