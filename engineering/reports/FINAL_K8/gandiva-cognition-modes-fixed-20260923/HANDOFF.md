# Gandiva cognition mode matrix and build handoff

Date: 23 September 2026; branch: `Gandiva`.

Base Git HEAD when measured: `74379af39031a2c5355b08b7ca2e4d4bda20f8f6`

## What changed

The no-update live cognition diagnostic now covers all 12 registered E2
family/mode pairs: rule-inquiry, inventory, and program, each through
`b-policy`, `b-workspace`, `b-planner`, and `b-memory`. Calls are capped at two
per trace (four for bounded planning), and memory retrieval is built from the
training split only. The verifier checks the model-call provenance, complete
family/mode coverage, memory reads, eval mode, discarded gradients, and zero
optimizer updates.

The initial matrix run exposed a real integration regression: the expanded
episode helper needed the data path for its training-only memory index, but the
integrated rehearsal did not pass one. That failed report is preserved at
[`../gandiva-cognition-modes-20260923/build_verification.json`](../gandiva-cognition-modes-20260923/build_verification.json).
The fix passes the already-validated full bundle into the rehearsal's live
diagnostic while retaining its independently generated small fixture bundle.

## Verified evidence

The source-bound report beside this handoff
([`build_verification.json`](build_verification.json)) records:

- `VERIFIED`: F01–F24 all pass; all seven pytest check groups and all seven
  real-interface exercises pass.
- Total verifier duration: 223.015 seconds. The integrated rehearsal took
  55.578 seconds. Local optimizer updates: zero.
- Owner notebook interface: 24 cells, 12 code cells, zero shell magics, and
  zero hard-coded host paths.
- Data bundle identity:
  `79c9706d122050cc1e8f5e6a3363af68005fc1206db0451b40ea807121c8d7f8`.
- Implementation closure SHA-256:
  `9f41d8e6424e719eba5c286a8f9a81521d35335575e41ff2f9f178e7f998307c`.
- Report identity: `0f2455901507661219e7d1926a0da4c432796bdcec287db89026d3fbf20e81b9`.

The traces prove execution and data-scope wiring, not useful cognition. The
model was randomly initialized. All 12 traces had genuine model-origin calls;
each `b-memory` trace read two training records. The three planner episodes
encountered model parse failures, selected fallback actions, and ended
unsuccessfully. Policy, workspace, and memory traces exhausted their small
call caps before an outcome (`success=null`). There was no training, GPU
qualification, or learned capability result.

The report records `source_identity.dirty=true`. The existing user edit to
`tests/test_research_k8_real.py` was preserved and left out of the scoped
implementation commit. The source closure above identifies the measured
implementation bytes; the report's `git_head` is the base revision shown at
the top of this handoff.

## Reproduction

From the repository root in PowerShell, using the prepared local bundle:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:PYTEST_ADDOPTS = '-p no:cacheprovider --basetemp=.pytest-tmp-gandiva-modes-fixed-20260923'
python -B -m bramastra_lab.research.campaigns.k8 verify-build `
  --data 'C:\Users\ankit\AppData\Local\Temp\bramastra-k8-build-ab580776-20260923' `
  --report-dir 'engineering/reports/FINAL_K8/gandiva-cognition-modes-fixed-20260923' `
  --no-updates --notebook 'notebooks/bramastra_k8.ipynb'
```

That report directory already contains the recorded result. For a rerun, use
new unique names for both `--report-dir` and `--basetemp` so prior evidence is
not overwritten.

## Next owner action and limits

The normal operator notebook is
[`notebooks/bramastra_k8.ipynb`](../../../../notebooks/bramastra_k8.ipynb).
Run its E0 gate on the actual Kaggle two-T4 session. G01–G04 are still
pending: device mapping/precision/allocation, real full-profile update and
fresh-process continuation, measured throughput, and frozen source/data/
protocol identity. E1–E6 must wait for those runtime gates to pass. The local
report establishes implementation readiness only; it is not a training result
or an AGI claim.
