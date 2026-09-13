# Chief review of B2.2 at acfa249

2026-09-13. The branch was clean and aligned with origin when inspected. The new commit implements substantial corrections, including configured accumulation in the CLI, safe checkpoint loading, data-byte checks, explicit case pairing, writer/parent checks and persisted loop state. **Blanket closure of R1–R7 is not accepted yet.** Preserve this work and close the remaining cases below as the final M00 gate.

The chief independently ran four focused non-learning suites: `test_research_data.py`, `test_research_evaluation.py`, `test_research_pair_path.py`, `test_research_loop_traces.py`. Result: **60 passed in 4.58 seconds**. These tests include a mirrored loop harness, which is useful but does not prove every public orchestration path. No learned probe was rerun. The agent's uninterrupted/resumed learning result is reported evidence; its `/tmp/b22-acceptance` checkpoint outputs were not independently rerun or verified here.

## Remaining corrections

### F1 — Prepared identity and actual resumed data are still separable

`commands._read_prepared` hashes row bytes against the mutable `split_integrity` map but never recomputes the prepared manifest's own content identity. Changing a row and its row digest while retaining the old top-level identity is accepted by both `_read_prepared` and `_load_rows`. The [zero-update probe](probe.py) reproduces this.

Separately, `commands.resume` reads `data_source.json`, restores the original checkpoint/run identity, then loads training rows from the mutable path without comparing that prepared dataset's identity with `ctx.data_identity`. Checkpoint-versus-run equality does not bind the data actually consumed. This second route is source-verified, not exercised with a learned checkpoint by the chief.

Required correction: one strict prepared-manifest validator recomputes its identity and validates schema, inventory and row bytes; all actual train/resume/evaluate consumers call it. Resume additionally requires the chosen prepared identity to equal the run/checkpoint identity before model allocation or lease acquisition. A deliberate data change is a recorded child/migration, never an exact resume. Add actual public-path negative fixtures with a fake restore seam; changing A's data pointer to internally valid B must fail before `_training_loop`.

### F2 — Per-example trainability mutates the file-level rule

`data.manifest.load_dataset` assigns `trainable = trainable and example_trainable` inside the record loop. Once a row is false, all later rows in that file inherit false, even when explicitly true. The probe shows `excluded,eligible` makes both false, while reversing the rows makes the eligible row true.

Required correction: preserve immutable entry eligibility and calculate a fresh `effective_trainable` per record. A row can narrow file-level eligibility but cannot broaden a forbidden file. Verify both orderings and false-file/true-row behavior. Check admitted counts through preparation and sampling, not just the dataclass field.

### F3 — Promotion still trusts inconsistent or nonfinite nested evidence

`evaluation.scoring.EvidenceBundle` validates some top-level strings and top-level metric floats, but nested family/paired/uncertainty mappings remain unvalidated and unbound to raw receipts. `decide_promotion` consults the supplied `ci_includes_zero` flag rather than requiring a valid positive lower bound. The probe supplies a wholly negative interval and a contradictory positive aggregate gain: decision is `accept`. It also supplies a NaN protected-family score with a positive interval: decision is `accept` because the NaN comparison does not register regression.

Required correction: construct comparison evidence from validated immutable raw outcome receipts and protocol identities; strictly validate all numeric values/ranges, required family sets, cluster counts and interval ordering. Derive flags from bounds, require the protocol's actual lower-bound condition, and bind uncertainty to the same cases/aggregation/parent/child/data. Reject inconsistent aggregates rather than choosing the favorable one. An arbitrary caller-provided bundle is an untrusted input. Add these exact negative cases to the public promotion path. This is a prerequisite for any learned improvement transaction.

### F4 — Failed resume setup leaks its writer lease

`runtime.resume.restore_run` acquires a writer token before returning. `commands.resume` performs expected-parent checks, row loading, sampler/replay restoration and diagnostics setup before entering its release-protected `try/finally`. Any failure there leaves the lease active. The wrong `--expect-parent` route is an immediate example. Source-verified jointly by chief and Luna.

Required correction: guard every operation after acquisition with release-on-failure, including constructor/setup failures. Prefer identity and expected-parent checks before acquisition where possible, with a final fenced recheck at the publication boundary. Prove a failed setup leaves the next valid attempt able to acquire the lease without force recovery. Do not solve this by stealing a possibly live writer's fence.

### F5 — Public single-step API ignores a configured multi-microbatch window

The CLI now groups microbatches, but `Trainer.training_step` still unconditionally accumulates once and finalizes, even when config `grad_accum_steps > 1`. Its docstring says one-microbatch; no runtime guard enforces that restriction. Source-verified.

Required correction: either reject this convenience method for multi-microbatch configuration or expose an explicit window API that enforces the configured boundary. The public contract must not silently disagree with the CLI. A fake accumulator/finalizer or bounded gradient-only check can prove dispatch without spending an optimizer update.

### F6 — Resource incident and evidence reporting need precise closure

The ledger now records **206/200 CPU updates and 177.234/300 learned-smoke seconds**, zero GPU usage. The increment from 193 is **13**: one failed attempt after one update, then two six-update runs. The handoff's approximately 171 seconds omits the failed attempt's six seconds; use the ledger's actual total. No further learned checks are authorized in this phase, including attempts to repair the overshoot with another probe.

Required correction: retain all history; correct current status/handoff by appending a clarification, not rewriting old receipts. Test hard pre-execution accounting and failed-run accounting using injected callbacks. Verify subprocess tests cannot bypass the shared allowance merely because they are launched outside `train --smoke`. Record the missing raw acceptance-output location as an evidence limitation; recover existing outputs if available without rerunning them. Reserve counts from earlier phases do not create a fresh allocation.

## Agent action and evidence

Close F1–F6 under M00 using non-learning checks. Keep learned acceptance provisional until the already-produced evidence is recovered/verified or a future allocation permits a new uniquely identified check. Continue independent cognition/RSI contracts and implementation according to the current program; do not use partial foundation acceptance to publish learned candidates.

Exact chief commands:

```powershell
& 'C:/Users/ankit/Downloads/An-Ra-the-new-AGI-1/.venv/Scripts/python.exe' -m pytest tests/test_research_data.py tests/test_research_evaluation.py tests/test_research_pair_path.py tests/test_research_loop_traces.py -q -p no:cacheprovider -o addopts=
& 'C:/Users/ankit/Downloads/An-Ra-the-new-AGI-1/.venv/Scripts/python.exe' engineering/reports/B2_2_CHIEF_20260913/probe.py
```

The immutable [observations](observations-001.json) contain the four reproduced outputs and both trainability orderings. F1's resume route, F4 and F5 are source findings; this report does not present them as runtime reproductions. The cognition/RSI extension is an architecture assignment, not an assertion that this pushed branch already has those abilities.
