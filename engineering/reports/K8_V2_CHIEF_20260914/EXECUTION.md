# Direct build order: finish the campaign before asking for hardware

Read [the acceptance review](REVIEW.md), root [experiment.md](../../../experiment.md), and [I01–I06](../../experiments/K8_20260913/READINESS.md). Existing treatments, numerical settings, data sizes, schedules and compute caps remain authoritative. The packages below make the missing implementation concrete. They do not add experimental arms or authorize learning locally.

## D1 — Supervisor and slot execution

Integrator owns campaigns/runner.py, process_supervision.py and supervisor.py. Add explicit ordered slots containing at most one active job per physical device. E1 has two slots as specified; E3/E4 reverse treatment order between seeds. Reserve and launch only a slot's jobs. Carry separate physical device identity and local device index. Persist slot progress and deadlines, and recover individual jobs without repeating accepted outcomes.

Own child processes explicitly. Enforce actual termination and preserve event/checkpoint evidence from interrupted work. Refuse live-lease takeover regardless of age and fence release by ownership token. Return nonzero for failed required work. Local acceptance runs the production launcher with finite CPU worker functions: two different devices overlap, same-device jobs never overlap, a timed-out child cannot write a delayed marker, and restart preserves allocation and completed slots. No GPU is needed to implement or prove these control properties.

## D2 — One learning-window and checkpoint interface

Integrator owns trainer interfaces; a bounded Luna task may implement nonoverlapping checkpoint validation helpers. Define one update-window input with objective sums/counts, batch/source identity, counters and declared allocation. Accumulation does not step; explicit finalization steps exactly once. Avoid double normalization and means of microbatch means. Connect token/world/action/value/pair weights to actual losses; disabled terms must not execute or contribute gradients.

Checkpoint publication carries config, architecture, model, optimizer, scaler, RNG, stream cursor and controller state plus allocation/job identity. Resume validates a live reservation instead of granting a new allowance. D2 local evidence uses backward equivalence and no-op optimizer doubles; it must not mutate learned parameters. E0 later supplies real updates. The subprocess E0 helper restores the correct configured device/profile/precision and performs the resumed continuation itself. Both branches call the same explicit accumulation/finalization function.

## D3 — Data compiler and real environment adapters

Data executor owns k8_bundle generation/compiler and isolated environment adapters, excluding trainer/supervisor edits. Produce actual executable episodes and all objective targets under the frozen protocol, with globally grouped identities and qualified primary/meta/protected splits. A manifest must bind real source bytes, not a claimed count. Fail preparation before GPU launch if required cases, references or context limits are invalid.

Implement and verify finite tool execution with request/receipt IDs and output checks. Store private evaluator material separately from public model inputs and target labels. Use the same renderer in compilation and inference. Deliver deterministic mini-fixtures for local integration plus instructions to prepare the full bundle offline. No pretrained core or external LLM teacher is introduced.

## D4 — Actual E1–E5 executors

Implement repository phase modules behind worker dispatch. These must exist before notebook delivery; GPU testing is a later execution environment, not an implementation dependency. Use one typed job input and typed result with phase/slot/arm/seed/parent, frozen source/data/protocol identity, successful/attempted updates, observations, costs and artifact paths. Expensive model operations may be test doubles only in tests.

| Executor | Required production work | Local evidence without updates |
|---|---|---|
| E1 | Random initialization for paired seeds; A/B objective configuration; identical per-seed starting state and stream; checkpoint fractions; free-generation evaluation | Trace both slots and verify parent/stream equality, treatment weights and checkpoint callbacks through the production executor |
| E2 | Load frozen E1 checkpoints; actual learned policy/workspace/planner adapters; existing contradictions, inquiry and goal controls | Step real mini environments with injected model responses; enforce action/call/node budgets and verify no optimizer path |
| E3 | Clone each E1-B parent separately for T0/T1; declared replay mixture and objective package; tool acquisition plus old-task evaluation | Verify parent isolation, exact sampling receipts, actual tool output verification and protected evaluator exclusion from training |
| E4 | Clone E1-B independently of E3; S0/S1 gate configuration; preserved model/head/segment contract; declared shared-block computation | Zero-gate and nonzero-path checks, device-correct migration and restoration, treatment slot ordering |
| E5 | Materialize trial archive, fixed adaptation anchors, P0 method learning, generated method application to P1, matched P_fixed, and fresh confirmation choices | Exercise all three blocks with deterministic trainer doubles; reject future outcomes, changed anchors and unapplied recipes; verify every lineage and cost event |

Each executor must return failure for missing evidence and report actual counts. E2 is frozen evaluation and must not be required to have positive optimizer updates; its success receipt instead binds evaluated cases and checkpoint identity. Likewise E6 is export, not training. Do not force a generic positive-update gate onto all phases.

## D5 — Packaging and independent delivery review

Integrator owns CLI, notebook and export. Preflight must validate the actual configured bundle and availability of all handlers before starting E0. Freeze the measured E0 update targets only after authorized calibration. Notebook cells call repository functions through checked arguments, retain the same allocation across modes, and never silently rerun accepted jobs.

E6 exports the specified complete bundle, with compact summaries in Git and full restorable payloads in owner storage. Verify hashes, required parent checkpoints, failed-run records and a real load of each required payload. An identity string or list of filenames is not restore evidence. Export failure must not become campaign success.

Return K8_DELIVERY_20260914/HANDOFF.md with D1–D5 and R01–R08 mappings, precise changed production functions, local commands/results, immutable evidence locations and only genuinely hardware-dependent checks pending. Delegate bounded Luna tasks with explicit owned/excluded files; the integrator owns shared APIs and final review. Stop once acceptance evidence is complete; do not pad time, code or token use.
