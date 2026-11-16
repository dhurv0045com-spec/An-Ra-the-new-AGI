# Planthir reliability review — 2026-11-16

Reviewed baseline: `de4bce0ce7516a9e7d190a86de749ce64263e745`.
Implementation: `764583debe71be8f43b683ec9e85d6199437b8e5`.
Reviewer: Codex, with bounded Terra reviews and implementations.

**Decision:** the supported V8 path is substantially hardened and passes bounded CPU engineering checks. It is not yet a generally qualified experiment engine or an owner-GPU-qualified release. No research training campaign was run locally. Historical experiment results remain historical; this review does not reproduce their scientific findings.

## Findings and repairs

| Confirmed engineering problem | Implemented repair | Evidence or limit |
|---|---|---|
| Runner could not resume a campaign reliably | Identity-bound explicit resume, writer lock, durable status/attempt records, contiguous update journal, checkpoint milestones and cooperative deadline | Injected interruption resumes to the identical final checkpoint SHA as an uninterrupted tiny control |
| Crash between checkpoint commit and milestone marking could lose a milestone during pruning | Restore missing milestone binding before further training/pruning | Dedicated regression passes |
| Declared experiment factors omitted meaningful recipe/model differences | Compare complete recipes and fixed model/data/order/policy fields; explicit factor and capacity checks | Undeclared precision, seed, capacity and invalid input regressions; parameter tolerance is not FLOP matching |
| B0 execution did not honor the registered recipe and used a nominal gradient check | Use the declared recipe and actual masked cross-entropy gradient checks | B0 end-to-end regression passes |
| Instrument checks could alter model/RNG state or overlook malformed observations | Restore gradients, buffers, modes and RNG; recursive truth checks; actual observation/action parity and action-budget enforcement | Instrument negative tests and six injected faults detected |
| Probe overlap, misaligned calibration and isotropy definition could distort diagnosis | Reject shared canonical graphs/lineage, align confidence/correctness, use directional second moment | Scientific regressions including antipodal rank-one representations pass |
| Oracle differences or surviving a falsifier could be described too strongly | Retain suspected/unresolved claims; require explicit cluster assumptions for intervals | No causal root cause is established by this review |
| Large checkpoint export could exceed the compact cap or exhaust memory | Stream checkpoint files; evidence-only compact mode with full sidecar; mark FP16 checkpoints lossy | Archive/reload tests pass; S20 target hardware not measured here |
| Larger diagnostics could exceed caps or lose detailed evidence | Explicit bounded summary/capsule references; full-precision arrays, diagnostics and traces retained in full sidecar | Original/selected rows and omissions disclosed; chunked raw array writing remains future work |
| ZIP parsing/restoration accepted ambiguous names or unsafe structures | Reject duplicates, unsafe paths and platform aliases; verify hashes and restore through staging | Tamper/path/corruption regressions pass |
| Completion could precede verified delivery | Verify final staged archives before publishing COMPLETE; retain PARTIAL outputs on verification failure | Injected export and final verification failures cannot yield a complete run receipt |
| Parity verification could retain two GPU models during successive loads | Release each loaded model before loading the next arm | Two-arm weak-reference lifetime regression passes on CPU; GPU peak not measured |
| Fixed 82% savings figure implied measured productivity | Replace authoring-savings claims with NOT_MEASURED; distinguish byte compression | A real paired productivity study remains required |
| CI did not trigger on planthir; legacy tree policy rejected newly authored agent.md | Enable planthir workflow; classify rewritten guide explicitly while retaining original archive blob | Local tree/evidence checks pass; remote CI tracked separately |

## Actual validation

- Full repository suite: **472 passed, 1 skipped, 14 subtests passed**, 315.54 seconds.
- Focused Planthir/V8/tree integration: **71 passed**, 115.30 seconds.
- Final runner subset, including the final model-lifetime correction: **14 passed**, 32.55 seconds. These overlap the broader suite and must not be added as independent tests.
- Calibration: **6/6 injected defects detected**, healthy V8 kernel passed. Faults cover consumed tools, truth leakage, prompt drift, disconnected gradients, aggregation shadowing and RNG corruption.
- `python -m tools.verify_elendril_evidence`: passed; preserved 237 checkpoint manifest records, 63 pilot receipts, 78 receipt/figure/resource hashes and 519 scoring snapshots.
- `python -m elendril audit --output <temporary path>`: completed successfully as a fixture audit, not a model research result.

An earlier full pass attempt found one tree-inventory failure (470 passed, 1 failed, 1 skipped). The baseline's new `agent.md` conflicted with an old removal classification. The classifier and generated inventory now explicitly mark the rewritten operator guide as reimplementation; the historical blob is unchanged. No assertion was disabled.

`METRICS.json` records environment, exact implementation commit and local log hashes. Logs remain outside Git. Full-suite timing is elapsed time on this local validation environment, not a GPU throughput estimate. Remote CI was pending when this record was first written; any later CI receipt is an additional observation.

## Remaining limitations and next action

The runner deliberately rejects unsupported non-V8 kernels/custom trainers. Real process-kill/power-loss durability, atomic generation-pointer publication, measured GPU/disk quotas, independent T4 workers, cross-environment replay and statistically qualified seed-level mechanism comparisons remain unqualified. Full raw arrays are preserved but not yet written as bounded streaming chunks. A passing certificate is engineering coverage, not evidence of cognition or causal identification.

Give the next agent [the executable Gemini handoff](../../../research/planthir/GEMINI_RELIABILITY_AND_UPGRADE_HANDOFF.md). It specifies backend contracts, recovery/publication, resources, design/statistical units, evidence custody, cloud qualification and a measured productivity study, with file-level implementation targets and acceptance criteria. Start by extending the backend contract and actual process-interruption qualification; keep the current V8 path passing throughout. Owner GPU canaries precede any claim of launch readiness for larger experiments.

## Agent usability follow-up

Added `agent-info` (derived input fields and current capabilities) and read-only `inspect-run --work-root PATH` with identity checks and localized metadata errors. Inspection explicitly does not verify archives. The obsolete GPU CLI now fails before execution with migration guidance; its historical study function is preserved. Ten combined agent-interface/execution tests passed in 2.23 seconds after this addition. The handoff specifies a future data-only experiment schema, validation, explicit launch/resume, actionable error codes and an independent fresh-agent usability trial. These new CLI operations are discovery/inspection tools, not a completed declarative campaign launcher.
