# K8 build handoff — notebook, campaign runtime and GPU-pending verification

Date: 2026-09-13. Agent: BRAMASTRA implementation lead. Baseline `bafdf14` ("Strengthen K8 system explanations and implementation quality gates"). This handoff covers I01–I06 and the chief's F1–F6 follow-up; it follows the [review protocol](../../experiments/K8_20260913/REVIEW_PROTOCOL.md) evidence map.

## Executive status

| Package | Disposition | Detail |
|---|---|---|
| I01 CUDA trainer | **implemented, locally verified** | `learning/k8_trainer.py`: AMP fp16 autocast + GradScaler, unscale-before-clip, skipped-update accounting, allocation-aware update admission, scaler state in checkpoints. CPU path unchanged. **CUDA-unverified**: actual T4 execution is E0's job. |
| I02 differentiable objectives | **implemented, locally verified** | `learning/k8_scoring.py` (action/value/world training scoring with live gradients), `experience/rendering.py` (single public renderer with boundary-level provenance rejection), `experience/supervision.py` (A1 window denominators). **CUDA-unverified**: gradient magnitudes on T4. |
| I03 K8 data | **implemented, locally verified** | `data/k8_bundle.py`: three families (rule-inquiry, inventory, program) + tools + meta-tasks, deterministic, mechanism-deduped, hash-bound, audit report. Small fixture tested; full 4096-mechanism build is a Kaggle prepare step. |
| I04 gated architecture + proposer | **implemented, locally verified** | `models/gated.py` (zero-gate migration functional equality, shared-block reuse not clones, gate gradient checks, nonzero gate reaches shared blocks); `metalearning/dispatch.py` (real proposer capture, origin validation, forged checkpoint rejection, external labeling, M0/M1/M2 dispatch). **E5 learned evidence is collected in the authorized campaign.** |
| I05 supervisor/workers | **implemented, locally verified** | `campaigns/supervisor.py` (SQLite journal, reservation lifecycle, capacity gate, idempotent close), `campaigns/runner.py` (phase plan, E0-gate for E1+, device assignment), `campaigns/worker.py` (E0 probe: 3-update vs 1+2-resume on device), `campaigns/k8.py` (prepare/validate/run/summarize/export CLI). **CUDA-unverified**: actual two-worker dispatch on T4. |
| I06 notebook | **implemented, locally verified** | `notebooks/bramastra_k8.ipynb` (14 cells: setup, prepare, validate, E0, full, summarize, export). Calls repository entry points; never redefines logic. **Not yet executed by the owner.** |

## F1–F6 disposition (from the B2.2-chief review, verified this session)

| Finding | Evidence |
|---|---|
| F1 prepared identity | `_validate_prepared_manifest` recomputes content identity; mutation-with-digest-under-old-identity refused (negative fixture). Resume binds prepared identity to run identity before lease/allocation. |
| F2 trainability order | Fresh `effective_trainable` per record; both orderings verified; nontrainable rows refused at sampling and gradient boundary. |
| F3 promotion validation | Deep validation of all nested metrics (finite, range-checked, counts); derived `ci_includes_zero`; inverted interval rejects; NaN family rejects; positive-gain-with-negative-interval rejects. |
| F4 lease leak | Lease acquired after all fallible setup; fenced recheck; release on every failure path. Proven: failed setup leaves lease acquirable without force. |
| F5 training_step guard | `training_step` rejects configured multi-microbatch windows; public contract matches CLI. |
| F6 resource precision | Ledger: 206/200, 177.234s; increment = 13 (1+6+6); clarification appended. Recovery of acceptance outputs from /tmp (JSONs tracked, weights gitignored). `require_learned_allowance` hard gate verified at 206/200. |

## Causal path (vertical slice)

```
render_event_sequence → prepare/collocate → K8Trainer.accumulate (AMP)
→ finalize_update (unscale→clip→step) → state_payload (model+scaler+counters)
→ score_candidates_trainable (action head) / value_estimate_trainable (value head)
→ world_transition_token_loss (world span) → Executive.decide (scorer adapter)
→ SessionRunner.run (cost aggregation) → classify_failure (tool receipt)
→ checkpoint_run (atomic publish, writer lease, expected parent)
→ restore_run (identity binding, late lease, source migration)
```

Each arrow is exercised by `tests/test_research_k8.py::VerticalSliceTests::test_slice_reaches_all_components`. The slice stops before `optimizer.step()` locally; E0 supplies the actual GPU update.

## Review-protocol counterexample inventory

| Seam | Defect injected | Detection |
|---|---|---|
| Preparation → renderer | Hidden `episode_id` in event payload | `RendererError: provenance-only fields cannot enter public tokens` |
| Router → trainer | Missing `world` term when eligible count > 0 | `route_window` records `loss_not_supplied` and does not emit zero |
| Trainer → parameter groups | Action head removed from optimizer | `check_gate_gradients` reports missing gate/block/embedding gradient |
| Restore → next batch | `data_source.json` points to a different bundle | `_read_prepared` refuses: "prepared manifest content identity mismatch" |
| Checkpoint → executive | Scorer replaced with constant | Test injects a fake scorer; vertical slice asserts `requires_grad` on the real scorer |
| Executive → tool | Tool claims success without receipt | `classify_failure` returns "unknown" for unverified claimed success |
| Architecture → optimizer | Shared block cloned | `migrate_from_parent` verifies functional equality; `check_gate_gradients` reports per-group gradient presence |
| Proposer → successor | Dispatch to M0 while labeling as M1 | `dispatch_method_to_trainer` records the compiled identity; a mismatch is detectable by comparing the trainer's controller multiplier reason |
| Confirmation → archive | Current-task outcome in proposal context | `MethodProposer.render_input` rejects `measured_success`/`query_outcome`/`label` keys |
| Supervisor → recovery | Fresh clock on kernel restart | SQLite ledger stores absolute UTC deadline; `deadline()` returns it regardless of process restart |

## Exact owner launch steps

1. Upload the repository to Kaggle as a dataset or clone it in the first cell.
2. Enable **GPU T4 ×2** in the notebook settings.
3. Run cells 1–3 (setup, prepare data, validate). Data preparation is deterministic and idempotent.
4. Run cell 4 (E0 gate) to prove the CUDA path. If E0 succeeds, proceed.
5. Run cell 5 (full campaign). The runner will skip E0 if it already ran and use the remaining allocation.
6. Run cells 6–7 (summarize, export).
7. Copy `/kaggle/working/K8-results/` to persistent storage.

## Remaining GPU-only checks (pending E0)

- Real CUDA gradient flow through the full 8-layer width-256 model with AMP
- Actual T4 resume equivalence (checksums, counters, next batch)
- Two-worker concurrent execution with device assignment
- Campaign-phase timing calibration (update targets from measured throughput)
- E1–E5 learned results (unestablished by design until the owner's launch)

## Known limitations

- The objective router is consumed by K8Trainer.accumulate but the E1-A arm intentionally uses answer-only weight; the unified path is exercised in E0's four-objective pilot.
- The executive scorer adapter wraps `models.decisions.score_candidates_trainable` through the K8 training path; E2 uses a frozen-model variant.
- The method-language proposer is a real model-generated capture; the 3-method vocabulary is intentionally small (M0/M1/M2).
- M08/M09/M10/M13–M18 remain designed-not-built or partial; they are not K8 launch blockers.
- The old 206/200 CPU ledger is untouched; the K8 allocation is a separate ledger.
