# Acceptance, efficient verification and execution-agent prompt

## Required evidence matrix

The handoff must contain one row for each U01–U10 package and J01–J14 finding: implementation status, precise changed consumer, regression evidence, production-path evidence, source identity, remaining limitation and next owner action. PASS means the stated criterion passed; NOT RUN means no supporting execution. A source review finding can be closed by a targeted test plus the actual corrected consumer, not by a comment claiming alignment.

Keep four independent levels: implemented, CPU-verified, GPU-qualified, experimentally supported. A code fix can be accepted without proving that training will improve capability; a positive learning claim needs the actual registered experiment. Neither passing tests nor model-origin text proves AGI or recursive improvement.

## Current reproducible baseline

From the BRAMASTRA root, with the project Python and PYTHONPATH set to the root:

```text
python engineering/integrated_readiness_20260914/audit.py --output <new-receipt-path>.json
python -m pytest tests/test_research_k8_foundation.py -q -p no:cacheprovider -k "not real_history_never_mutated and not production_trace_per_family_with_invalid_case"
```

The committed source audit is a historical baseline against the in-progress API. It returns per-criterion statuses and source hashes; exit zero means the diagnostic completed, not that the foundation passed. Do not treat changed private helper names as a passing repair. Port the same behavioral counterexamples to the corrected public interfaces in new tests, retaining the old receipts.

The second command produced 26 passes and two deselections; the excluded paths require the U04 termination repair before unbounded suite execution. Run the repaired episode tests in a child process with a deadline. Keep tests focused on input preservation, task solvability, actual consumer routing, gradients, counter monotonicity, state transitions, authority and evidence reductions. Tests that assert a report has a field without checking how it was produced do not close a semantic defect.

## Pre-GPU acceptance sequence

1. Freeze public task/action/observation schemas and consumer mappings; regenerate compatible small local fixtures in new directories.
2. Pass task solvability and teacher replay checks, compiler/consumer parity, codec round trips and lineage compatibility rejection.
3. Pass bounded full episode traces with real adapters and capture doubles, plus random-weight real-model no-update traces with honest failure accounting.
4. Pass backward-only objective connectivity and whole-window equivalence tests; discard gradients. No local optimizer update is authorized.
5. Pass simulated schedule/deadline/ledger tests and independent proposer capture ordering, including failed trials and zero-progress errors.
6. Export and reload the intended compact evidence bundle; reconcile every aggregate to source events and independently inspect the per-treatment consumer map.
7. Submit the complete acceptance matrix and chief review request through the branch handoff. Keep the launch block until code acceptance. List CUDA/full-model qualification as NOT RUN locally.

E0 on the owner notebook must qualify the actual two-T4 hardware, full model, precision, complete optimizer window, fresh-process resume and heaviest evaluation path before the frozen protocol permits learning phases. If that fails, preserve evidence and stop; do not spend the remaining allocation on a smaller unregistered experiment. All E0, retry, adaptation, proposer and export work stays inside the 480-minute campaign and existing per-phase schedule.

## Prompt to give the implementation agent

Work on BRAMASTRA and inspect current dirty work before editing. Read AGENTS.md, engineering/README.md and engineering/STATUS.md, then every document in engineering/integrated_readiness_20260914. This is the current consolidated chief dispatch, incorporating the in-progress F1–F6 work without accepting its readiness claim. Reproduce the bounded audit and preserve its historical receipts. Implement U01–U10 as one integrated delivery: solvable tasks and correct labels; shared DecisionExample/compiler contracts; trained action/world/value consumers; typed submissions; bounded planning; actual A/B comparisons; full-profile E0 and honest update accounting; valid trial authority/retention; trained independent proposer/successor decisions; complete export evidence. Resolve J01–J14 against the real consumers. Use Luna/Sol for bounded independent work with explicit file ownership if helpful; own final integration yourself. Preserve from-scratch weights, canonical split exclusion, existing user work and the fixed K8 budget. No local optimizer updates, readiness bypass or long accelerator run. Return a new INTEGRATED_READINESS handoff with exact code/data/schema identities, commands, per-criterion results and remaining GPU-only checks, then push only your scoped work. The creative evidence-repair extension follows foundation acceptance; do not add new arms to the present campaign. Finish the connected milestone rather than returning another list of intentions.
