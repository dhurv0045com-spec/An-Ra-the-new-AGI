# HORM-001 / HORM-002 checkout custody audit and recovery request

This is a new engineering diagnostic, not a revision of either preregistration,
RESULT.json, ANALYSIS.md, or historical verifier. Baseline: `93b1d3f`.
No training, checkpoint deserialization, sealed evaluation, external search,
or remote contact occurred. Historical files remain untouched.

## Measured custody status

Both lines are `ARTIFACT_INTEGRITY_BLOCKED` at their expected checkout paths.

| Line | Declared artifacts | Missing checkpoints | JSONs matching only LF-to-CRLF reconstruction |
|---|---:|---:|---:|
| HORM-001 | 9 | 4 | 5 |
| HORM-002 | 20 | 9 | 11 |

Machine-readable evidence and exact recovery identities:
`HORM_CUSTODY_AUDIT.json` (schema `anra-horm-custody/v1`).
SHA-256: `47ef1e6da72d8a961c15a31d6378e24503f2b78a90bf5f831dd071a3b209a5ca`.
It includes the diagnostic source SHA-256, raw RESULT manifest SHA-256/byte count,
every declared artifact SHA-256, observed byte count and raw hash when present,
explicitly diagnostic EOL reconstruction hashes, and per-file recovery requests.
All report rows were read back and compared with a fresh inventory: MATCH.

The observed RESULT manifests are:
- HORM-001: `458902c08373a5a2109757f8209f71f3035a8db216125ef7ab0e81198a228ad5`.
- HORM-002: `77923d08c080ac2d11a4a7c6f756577bf29d7d9a81911cc9fc78e2ad182fbe62`.

A matching reconstructed hash explains a byte representation discrepancy;
it is NOT raw-byte verification, independent source authentication, or proof
of how transport changed the file. These bytes were reconstructed in memory
only. No normalized replacement was written.

## Recovery request (owner action; not sent externally)

For each `lines.<experiment>.artifacts.<name>.recovery` entry in the JSON:
1. Locate original run output or backup. External storage location is UNKNOWN.
2. Recover the file corresponding to `<experiment>/results/<name>` into a new
   staging directory, without overwriting historical checkout evidence.
3. Verify its raw SHA-256 against `declared_sha256`. A filename, a newly trained
   checkpoint, or an EOL-normalized replacement is not sufficient.
4. Preserve custody/source information for the recovered copy and rerun the
   original verifier against a separate, restored evidence tree. Independently
   validate checkpoint loading only after raw custody checks pass.

All 13 checkpoint paths and full hashes are in the receipt. Two HORM-002 paths
share a declared checkpoint hash; path-level recovery requests are retained.
Missing at the expected path does not mean absent from all storage or
unrecoverable. No broad search, regeneration, or original campaign rerun is
licensed by this report.

## Engineering implementation and validation

`anra_v5/horm_custody_audit.py` only inventories declared bytes. Strict JSON
parsing rejects duplicate keys and nonfinite constants. It rejects empty/null
artifact maps, malformed SHA-256 strings, unsafe paths, case-aliased paths and
resolved paths outside results. Present JSON artifacts must parse strictly.
Existing output files are never overwritten (exclusive creation).

CLI exits 2 when declared custody is blocked, 0 only for declared-byte
consistency, and nonzero on malformed input. The latter consistency status
is explicitly NOT scientific validation. A caller must check process exit;
an old report is not refreshed if a later attempt fails.

Exact commands from the isolated worktree:

- `env -u PYTHONPATH -u PYTHONHOME py -3.14 -m pytest tests/test_v5_horm_custody.py -q`
  Initial RED: missing module (collection error). Initial GREEN: 6 passed.
  Added invalid-input coverage: 10 failed, 7 passed before repair, then 17 passed.
- `env -u PYTHONPATH -u PYTHONHOME py -3.14 -m pytest tests/test_v5_horm_custody.py::test_cli_never_overwrites_existing_evidence -q`
  RED: 1 failed; existing synthetic output was overwritten before repair.
- `env -u PYTHONPATH -u PYTHONHOME py -3.14 -m anra_v5.horm_custody_audit --out docs/cymek/experiments/HORM_CUSTODY_AUDIT.json`
  Actual report generated; exit 2, `blocked=True`, as required.
- `env -u PYTHONPATH -u PYTHONHOME OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 py -3.14 -m pytest tests/test_v5_horm_custody.py tests/test_v5_initial_replay.py tests/test_v5_cyr_repro_ledger.py tests/test_v5_cyr_gpu012_duplicate_members.py tests/test_v5_cyr_gpu012_evidence.py tests/test_v5_cyr_gpu012_closure.py -q`
  Final: **41 passed in 9.96s**, including 18 custody tests.
- `git diff --check`: passed.

## Limits and decision consequence

This is not a full HORM schema, source, provenance, seed, appraisal, recurrence,
checkpoint, result-state, or verdict validator. It inventories only declared
paths; no expected complete campaign schema or extra-file validation is claimed.
The observed manifest is an input, not independently authenticated authority.
Original verifiers remain unchanged and blocked on raw artifact mismatches.
No mechanism effect or scientific replication is inferred from the diagnostic.

The HALState history and wrapper-bound defects identified separately are NOT
repaired by this commit. FORMATION-MUX checkpoint custody is also still open.
The next highest-value local task is FORMATION-MUX checkpoint receipt inventory
and an equally precise recovery request; no sealed data, model loading or
campaign rerun. Acceptance: declared checkpoint identities and verified source
receipt association, or an explicit ambiguity blocker. No authorization needed
for that local metadata-only work; remote recovery/contact remains gated.
