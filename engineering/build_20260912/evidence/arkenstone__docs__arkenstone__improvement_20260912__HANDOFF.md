# Agent handoff — ARK-HARDEN-20260912

## Assignment and status

- Primary role: integrator and engineering reviewer.
- Execution role: Sol produced initial runner changes; Luna owns the remaining runner corrections and focused tests at the owner's request.
- Work order: [WORK_ORDER.md](WORK_ORDER.md), packets A and B.
- Base commit: `933d4f32019af2f0d61092d092b1c3db67407700`, cached `origin/Arkenstone`.
- Branch/worktree: `codex/arkenstone-improvements`, `C:/Users/ankit/AppData/Local/Temp/arkenstone-improvements-20260912`.
- No accelerator run or paid compute was performed. Initial verification preceded Git publication; the owner subsequently authorized committing/pushing this work and a next-agent assignment. Total engineering elapsed effort was not measured; tool/test and preflight durations are recorded where available.

## What changed

Root owns shared receipt/budget/metric runtime, ARK-011 snapshot/sampler helpers, the new bounded preflight/explicit single-campaign CLI, runtime tests, root README and clarity report. Packet A owns ARK-012/013 runner corrections, campaign tests and its [handoff](RUNNER_HANDOFF.md).

The final behavior creates fresh output directories; writes receipts atomically with preserved revisions; bundles only this run's evidence and explicit source snapshots; binds shared code/plans/task manifests; avoids import-time result-directory writes; isolates CPU optimizer forks; preserves exact minibatch sequences while vectorizing RNG calls; detects recurrence only after recovery; rejects malformed accuracy trajectories; and enforces cooperative monotonic budgets.

The CLI records ordinary failures and interrupts, including unavailable CUDA initialization, before packaging in a finally path. These guarantees do not cover power loss, process kill, disk-full failure of all available storage, or process-level preemption. Checkpoint forks are in-memory; durable training restart is not implemented.

## Verification and reproduction

Runtime observed: Windows, Python 3.11.15, PyTorch 2.14.0+cpu, CPU processor `AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD`.

Exact verification commands and outcomes are recorded in the final validation receipt alongside captured test output. Reproduction entry points:

```powershell
python -m unittest discover -s tests -p 'test_discovery_v6*.py' -v
python -m unittest discover -s tests -p 'test_ark*.py' -v
python experiments/COLAB/run_discovery_v6.py --output-dir artifacts/arkenstone/discovery_v6/a-new-run-id
```

Every run requires a new directory. The preflight performs four small optimizer updates: one to initialize nonempty optimizer state, two identical forks and one different-LR control. All model weights start randomly initialized. CompactVocab, task generators and CONTROL/SEALED splitting are declared fixed priors. There is no retrieval, external teacher, pretrained checkpoint or network access.

The initial `cpu-preflight-01` passed. It is preserved and binds an earlier working-tree source revision. Final-source evidence uses a separate run ID; neither run overwrites the other. Development tests initially exposed a Windows source-path separator mismatch and an overly strict clock-resolution assertion; both were corrected before final validation.

## Scientific boundaries and next action

No ARK-012/013 full horizon was trained. No model capability, retention, transfer or AGI improvement is claimed. The measured sampler acceleration is a CPU preparation microbenchmark and cannot be multiplied into whole-training throughput.

Pending: full CUDA qualification; complete frozen multi-seed 8k/12k arms; durable disk restart; ARK-014 and combined V6 orchestration; prospective clarification of ambiguous verdict language; review of the inherited BOS/padding-supervision objective before any protocol change. Historical objectives, plans and receipts were preserved.

The chief accepts only the implementation behaviors supported by final local checks. Conservative summary interpretations are explicitly labeled in new code and the runner handoff; historical plan wording was not rewritten to make it appear more precise.

## Acceptance criteria and final chief review

Status: **bounded engineering packet complete; locally accepted**. Broader scientific and accelerator claims are not accepted.

| Criterion | Result | Evidence / limitation |
|---|---|---|
| Correct summary logic, matched horizons, evidence identities | PASS locally | Campaign tests cover missing/short arms, duplicate identities, partial verdicts, sourcewise threshold checks, no-recovery protection rejection, final/area agreement and reverse-direction gain |
| Preserve completed arms before later failure | PASS locally | Fake-runtime injected second-arm failures retain the first completed arm |
| Isolated atomic receipts and source provenance | PASS locally | Runtime fault-injection tests, receipt/source audit, immutable revisions and bundled code |
| No import-time result writes | PASS locally | Patched-directory-write import test |
| Bounded CPU exact-fork diagnostic | PASS | `cpu-preflight-02/PREFLIGHT.json`, immutable snapshot, optimizer and Torch RNG restore, identical next parameter hash, different-LR control |
| Preserve legacy sampler stream and measure setup cost | PASS locally | 18,000 x 64 indices checked exactly; initial speedup 3.38x, final speedup 2.85x; five timings per implementation per run |
| Focused regressions | PASS | 50/50 tests; [commands/logs](evidence/final-validation-01/VALIDATION.json) |
| Historical scientific plans/results preserved | PASS | Changed tracked files are root README and current runner/runtime code; historical plan/receipt files were not edited |
| CUDA qualification and full scientific experiment | NOT RUN | CPU-only local engineering acceptance |

Final preflight receipt SHA256: `df5d4245b5a100ebba65ed452b13259b57b0e7ed5d0383d31c25c2a576768c79`. Final next-update parameter SHA256: `98d196a4cc47df5c977313651d7c592b9481743e716eef03db910422d28aaa6d`. Final sampler order SHA256: `6addecadb6727f9bf232ae8249a550d0cec35e2a630a71384ac6d76b1dba7049`. Measured preflight campaign time: 7.50 seconds; host process elapsed time is separately recorded in validation JSON.

The chief added four integration test methods after Luna's six passed, including explicit campaign startup budgets, reverse-direction Pareto gain, duplicate/missing identities, and premature-negative rejection. Summary completion now requires valid frozen source identities and excludes all conflicting duplicate identities. A negative new-skill-acquisition verdict waits for all four frozen triplets; positive decisions retain the plan's two-triplet/both-seed requirement. Operational interpretations are labeled in summary payloads.

Existing legacy metric tests emitted non-failing ResourceWarnings for unclosed file reads in `experiments/lib/ark_metrics.py`; that unrelated helper was not modified. No full repository suite was run. Artifact audit verified 14 source files against working tree, source snapshots and ZIP entries, plus each final run receipt hash and archive CRC.

Publication note: evidence source snapshots retain their original measured bytes through a local `.gitattributes` override. Normal repository code follows the root LF policy; Git may normalize CRLF from the initial Windows working tree. Receipt source hashes refer to the archived measured bytes, not an assertion that every subsequent checkout has identical line endings.
