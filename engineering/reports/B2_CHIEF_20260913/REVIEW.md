# Chief review of the pushed B2/B2.1 build

Reviewed commit: `c3662b3ca097abf6ca5acaa73eb91cd9d074bc7a`, fetched from `origin/BRAMASTRA`, 13 September 2026. Review mode: source audit plus bounded non-training verification; zero model instantiations and zero optimizer updates in the chief's probes. A Luna reviewer independently audited checkpoint/trainer paths.

## Decision

The agent delivered substantial implementation: a shared decoder wrapper, codec, data pipeline, trainer, CLI, environment/collection adapters, controller/replay wiring and checkpoint infrastructure. Preserve that work. The integrated build is **not yet accepted as complete**: several critical data and state contracts work in isolated tests but are broken in the actual command path. The next phase is B2.2 integration correctness, not scaling or a new capability campaign.

The agent's fresh-process receipt establishes its particular default-path fixture. It does not certify intermediate checkpoints, replay/controller continuation, configured accumulation or modified prepared data. The report of 426 passing tests and ten historical failures is the implementation agent's record; the chief did not rerun that entire suite or independently reconfirm its baseline attribution.

Chief command: `python -m pytest tests/test_research_data.py tests/test_research_evaluation.py tests/test_research_plasticity.py -q -p no:cacheprovider --basetemp=.codex-test-tmp-chief-review-20260913`. Actual result: **63 passed in 0.23 seconds**. This suite performs no learned updates. The recorded 193/200 learned CPU updates remain unchanged.

Run [probe.py](probe.py) from the repository to reproduce the data/evaluation defects. It uses temporary fixtures, no model and no optimizer. [observations.json](observations.json) preserves the observed values at the reviewed commit. This is a defect demonstration, not a passing acceptance receipt.

## Findings to close

### R1 — Prepared data can change without changing the asserted training identity (P0)

`commands.py:101` records metadata and counts, without prepared-file content digests. `_read_prepared` at line175 checks only config identity. `_load_rows` consumes bytes without checking their hash. Resume loads those rows directly. The probe changed a prepared token and the original identity was still accepted. Source `HEAD-dirty` is also not an identity for the actual dirty code bytes (`commands.py:263`).

Required: content-bound preparation manifest, validation before every consumer, run/checkpoint data binding, source closure hashes, explicit migration for changed data/code rather than silent resume.

### R2 — Pair, semantic and trainability metadata are dropped or weakened (P0)

`commands.py:61-99` substitutes example IDs for semantic identities, fails to carry `Example.group_id` into SequenceRow, and renders every training-split example regardless of `trainable`. The probe's two grouped examples become `pair_group_id=null`, so the pair-loss path receives no pair even when enabled. Both examples marked nontrainable are emitted for training. Changing the manifest trainability flag leaves `DatasetHandle.identity` unchanged (`data/manifest.py:294`).

The current semantic digest at `data/manifest.py:148` is input/answer content, not a validated hidden-mechanism equivalence class. Do not imply it detects arbitrary renamed/rephrased task mechanisms. Carry content identity and qualified task-cluster identity separately; require the latter for cluster-transfer claims.

### R3 — Loop continuation and intermediate checkpoints omit live state (P0)

`commands.py:489` computes replay slots and controller cadence from invocation-local indexes. A split run can choose different replay/controller events than an uninterrupted run. `_checkpoint` inside the loop reads `ctx`, but sampler/controller state is written back only after the loop. `ctx.replay_cursor` starts null and is never updated by the command path. Resume recreates a replay engine without restoring that cursor. Empty/unrenderable replay batches increment `updates_done` despite no optimizer step. Intermediate controller checkpoints can also capture the old controller state.

Required: global committed-update scheduling, explicit fractional replay scheduling, one complete boundary snapshot of all live state, actual-step accounting and replay shortfall policy. Rebuild resume equivalence around a mid-loop interruption with optional paths enabled.

### R4 — Checkpoint validation and single-writer behavior are incomplete (P0)

`runtime/resume.py:85-103` does not pass or validate `RunContext.writer_token` when publishing; the fresh train context uses literal `"train"`. The save path reads the current parent implicitly and publishes LATEST without verifying the caller's expected parent. A second restore can acquire a fence, but publication does not enforce it. The finding concerns stale/concurrent writers for an existing lineage, not two successful `create_run` calls on the same directory.

`runtime/checkpoint.py:296-354` selects a directory from LATEST without binding pointer ID/update to the loaded manifest, and restore does not validate expected data/code identities. The loader also catches any restricted-load error and retries unrestricted `torch.load(weights_only=False)`. Remove that broad fallback; encode supported RNG state in safe primitive/tensor form and reject incompatible payloads with a clear migration requirement. Validate pointer path containment and manifest/marker/identity consistency before loading.

### R5 — The configured gradient accumulation count is ignored by CLI training (P0)

`config.py:134` exposes `grad_accum_steps`, while `Trainer.training_step` at line286 accumulates once and immediately finalizes. `_training_loop` always calls that method once per chosen batch. The low-level manual accumulation test does not prove that the configured command path accumulates. Implement its real semantics; do not silently change effective update/data budgets. Pair-loss aggregation must be defined across the complete accumulation group.

### R6 — Evaluation and promotion accept insufficiently bound evidence (P0 for promotion)

`clustered_bootstrap_delta` at `evaluation/scoring.py:253` matches only world-ID sets. The probe compares different cases/labels within the same world and receives a +1.0 delta. It must pair exact case identities, labels, family, budget and protocol before clustering. Stable world IDs alone are insufficient.

`decide_promotion` at line324 skips uncertainty and paired checks when their receipts are None. The probe obtains `accept` without either receipt. Required evidence must be a protocol decision, not inferred from the caller's omission. Raw scores, family retention and intervals must derive from one verified comparison, with finite values and matching cluster aggregation.

The CLI at `commands.py:810` stores even confirmation-split outputs in the measurement pool, supplies `role=None` for paired outcomes, and uses an eight-token generation cap regardless of task protocol. Once R2 preserves groups, this path will reject paired outcomes rather than score them. Correct pool routing and task-bound generation budgets before declaring evaluator integration complete.

### R7 — Operational resource and failure paths need closure (P1)

Train records smoke use only after success; exceptions can consume real work without updating the ledger. The loop has no live smoke deadline checks. Resume always applies the tiny-smoke ledger, including non-smoke lineages. Separate persistent run resource policy from the cumulative build smoke allowance, preserve partial consumption/failure status in finally paths, and do not reserve the whole production lifecycle against the remaining seven build updates.

## Next acceptance scope

Close R1–R7 with code, command-path regressions and a limited reproducible resume comparison. No new architectural mechanism is necessary. Do not reset the recorded smoke ledger or rerun the full learned test suite. The detailed assignment is [B2.2](../../phase_b22_20260913/README.md). Scientific qualification, data supply and target-device certification remain separate.
