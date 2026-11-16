# AN-RA V8-Y1 & Hitchhiker context

**Latest prospective protocol (2026-11-16, branch `hitchhiker`):** [V8-Y1 Registry](research/v8-y1/REGISTRY.json) (`3809785b27aa4feca1c0abfc8c81e9ed6dd69e48ff3c4bc0093481a647322cac`), [Checkpoint Intake](research/v8-y1/CHECKPOINT_INTAKE.json), `anra_v5/elendril/y1_{protocol,data,diagnostics,training,evaluation}.py`, `tools/elendril_y1_{campaign,worker,export}.py`, `tools/analyze_elendril_y1_results.py`, and [ELENDRIL_V8_Y1_KAGGLE_T4X2.ipynb](notebooks/ELENDRIL_V8_Y1_KAGGLE_T4X2.ipynb) (`PHASE = "auto"` -> `"all"` single-session `diagnose -> core -> followup` with milestone exports and 2-generation checkpoint pruning) are implemented on baseline `64ed23a433ebb1ce2f620e0db8dcc24f4d8c23be` (with historical Hitchhiker Stage 1 S3 weights from `6859904a02c8083c4a0ab320c9c4a511899f73e4`; historical S20 low-lr weights marked `NOT_AVAILABLE`).

**Prior owner results (2026-11-15):** [Stage 1 S3 results](Experiment%20Result/Hitchhiker/hh-stage1-4bfe3a806e3b/RESULTS.md) completed all 15 jobs; no intervention passed the screen. C seed 7’s apparent four-hop success follows a last-fact positional shortcut on the tested row-order cohort. Stage 0 and follow-up/scale training remain unexecuted; checkpoint restore and final dual-export completion are not certified from the analysis-only ZIP. Earlier future-tense planning below is historical and superseded where it conflicts with these results.

Updated: 2026-11-16. Branch: `hitchhiker`.

## Purpose and requested workflow

The user proposed mechanisms A–H for compact cognitive generalization and asked for a technical, evidence-producing experiment design before implementation. The first deliverable is complete: [conclusion and experiment design](research/hitchhiker/CONCLUSION_AND_EXPERIMENT_DESIGN.md). The [original supplied proposal](research/hitchhiker/USER_ARCHITECTURAL_PROPOSAL.md) is preserved byte-for-byte, with its hypotheses and reported numbers unchanged.

The documentation deliverable established the reviewed design ([conclusion and experiment design](research/hitchhiker/CONCLUSION_AND_EXPERIMENT_DESIGN.md)) alongside the [original supplied proposal](research/hitchhiker/USER_ARCHITECTURAL_PROPOSAL.md). The full Hitchhiker harness and Kaggle T4x2 campaign tooling — 7-factor counterfactual suite + structural holdouts, 4 health probes, Stage 0 raw/availability/grammar 260-way logit diagnostics & evidence sidecar packaging, deterministic static vs fresh/relabeled Stage 1–3 streams & trainer, episode-level fail-closed statistical gates, isolated 2-GPU worker/coordinator, dual verified bundle export/restore, and source-pinned notebook generator — are implemented and CPU-mechanically verified locally. Stage 1 S3 ran on the owner GPUs; Stage 0 and Stage 2–3 executions remain pending, and final dual-export completion is not certified.

## Source identities

| Item | Identity |
|---|---|
| Branch base, from `Elendril` | `2affffb806b7f73b00bc21aa90538019f37ce904` |
| Audited V7.17 ancestor (merged on `hitchhiker`) | `0c9cb284d2c8c949e17ac342959c6f2bf351bd30` |
| Merged pre-fix `hitchhiker` tip | `6a8d58caba692a1e79613aa22e0db597a603022e` |
| Recorded V7.17 optimization run source | `724af4cb508173e0eb3f4397d5dcb4a42a66712f` |
| Recorded V7.17 optimization run | `f443d3f4141b4163bbc130d46ebd7395` |

The audited V7.17 source (`0c9cb284d2c8c949e17ac342959c6f2bf351bd30`) is committed on `hitchhiker` and bound in `anra_v5/elendril/hitchhiker_protocol.py`. Existing V6/V7 documents retain their historical context.

The audited [V7.17 result report](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/0c9cb284d2c8c949e17ac342959c6f2bf351bd30/Experiment%20Result/V7.17/f443d3f4141b4163bbc130d46ebd7395/RESULTS.md) records earlier execution and replay. This document does not claim those experiments were run again for Hitchhiker.

## Findings that govern implementation

1. Lower LR improved mean confirmation from 62.80% to 84.52%, but novel one-hop and three-hop remained 0/48 pooled seed/case observations. These are three seeds on shared worlds, not 48 independent worlds.
2. UTF8 is a byte tokenizer. Novel aliases do not allocate new entity embedding rows. Formatting, sequence length, positions, and task wording are competing explanations.
3. Training randomizes entity associations across worlds, then repeats a fixed corpus. Freshness is testable, but static global entity-to-answer lookup is not an established generator defect.
4. Chain diagnostics jointly change wording, timestamps, distractors, and row order. Matched counterfactuals are needed to isolate causes.
5. Full grammar decoding already failed to rescue lower-LR three-hop results. An availability-only mask is a separately attributed diagnostic, not a demonstrated solution.
6. The six unrun representation jobs are baseline versus row reversal at 3e-4, not slot or binding training. Amendments need new protocol identities.
7. Registered model scales are S3, S8, S20. A new S9 requires an explicit architecture and parameter receipt.
8. External canonicalization, action masks, and algorithmic verification must remain distinguishable from learned representation formation.

## Next execution sequence

1. Integrate the reviewed implementation and establish source/protocol provenance.
2. Locate and verify original frozen checkpoints. The committed report says weights were absent from the supplied analysis ZIP; availability of the external FULL archive remains unestablished here.
3. Run matched frozen-checkpoint diagnostics with raw, availability-only, and existing grammar decoders. Log complete first-byte distributions and complete outcomes.
4. Register a five-arm S3 screen across seeds 7, 19, 41: static baseline, fresh worlds, entity relabeling, fresh-plus-relabeling, and targeted availability/task-coverage supervision. Total: 15 runs.
5. Add justified interaction and slot-control experiments; promote the selected recipe to matched S8/S20 comparisons. Keep recurrent reasoning and associative-memory changes as subsequent attributable experiments.
6. Evaluate fresh confirmation worlds, causal answer changes, renaming consistency, distractors, revisions, missing bridges, and independently acquired retention/transfer. Publish failures and inconclusive outcomes as well as successes.

The full design specifies proposed thresholds, compute matching, uncertainty, falsifiers, and an implementation prompt. Freeze final counts and criteria before new runs. Passing engineering tests does not establish cognition.

## Current handoff state (AN-RA V8-Y1)

- Complete (code, registry, orchestration, tests): `anra_v5/elendril/y1_{protocol,data,diagnostics,training,evaluation}.py`, `research/v8-y1/{REGISTRY.json,CHECKPOINT_INTAKE.json}`, `tools/elendril_y1_{campaign,worker,export}.py`, `tools/analyze_elendril_y1_results.py`, `tools/build_elendril_y1_notebook.py`, `notebooks/ELENDRIL_V8_Y1_KAGGLE_T4X2.ipynb`, and `tests/test_elendril_y1_{data,diagnostics,training,campaign,export}.py`.
- Verified locally (CPU only): `tests/test_elendril_y1_*.py` (`11 passed in 214.71s`, including real `ELENDRIL_HITCHHIKER_FULL.zip` verification and `S3-C-s7` checkpoint restoration, D0/D1 260-way `.npz` binary logit sidecars, optional D2 L2=1 probe/patching, Group-CE gradient equivalence, cumulative budgets, 5 MiB SUMMARY enforcement, and CPU canary E2E + offline replay) and focused existing checkpoint/export/cloud/foundation tests (`89 passed, 1 skipped in 13.67s`).
- Kaggle T4x2 staged execution (`notebooks/ELENDRIL_V8_Y1_KAGGLE_T4X2.ipynb`, default `PHASE="auto"` -> `diagnose` when verified `ELENDRIL_HITCHHIKER_FULL.zip` is mounted under `/kaggle/input/`, failing closed if absent):
  - `diagnose` (90m ceiling): `--phase diagnose --checkpoint-archive <ELENDRIL_HITCHHIKER_FULL.zip>` (emits compact `SUMMARY.zip` + `EVIDENCE.zip`).
  - `core` (360m cumulative ceiling): `--phase core --owner-enabled` (+ `--handoff-archive <diagnose-EVIDENCE/SUMMARY.zip>`, runs `T0` + seed-balanced 12-job `T1` + locked `T2` + `T3` bottleneck admission -> emits `SUMMARY.zip`, `EVIDENCE.zip`, `FULL.zip`, and external `FINAL_STATUS.json`).
  - `followup` (240m cumulative ceiling): `--phase followup --owner-enabled --handoff-archive <core-FULL.zip> --selection-receipt <selection_receipt.json>` (admits `targeted_group_ce` or `capacity_s20` or `DEFERRED`).
- Pending (not yet executed): physical Kaggle T4x2 GPU qualification, `diagnose` D0/D1 across all 15 restored historical S3 checkpoints, `core` T0/T1/T2 training/confirmation, and conditional `followup` T3. No capability promotion is claimed.
