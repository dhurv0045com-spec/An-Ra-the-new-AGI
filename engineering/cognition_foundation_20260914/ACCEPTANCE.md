# Implementation and acceptance contract

Implement in dependency order; keep fixes in reviewable commits. Estimate useful effort, then finish the criteria instead of padding work.

| Package | Required production change | Acceptance evidence |
| --- | --- | --- |
| F1: encoding and state | Bijective versioned rendering, typed public state, explicit context admission and artifact invalidation | x/y collision fails before repair; round trips after repair; private-world invariance; all live environment fields represented |
| F2: workspace | Subject/predicate/time-aware evidence and conflict relation | Different subjects coexist; normal transitions supersede; same-time contradiction remains unresolved; duplicates do not amplify support |
| F3: predictor | Complete evidence-conditioned prompt and matching train/inference target contract | Captured prompts change with relevant public history; action suffix preserved; backward-only target connectivity; invalid output recorded |
| F4: search | Imagined state transition, fair root coverage, explicit budget accounting | Delayed-reward fixture selects correct root; permutation check; no real-state mutation; failures consume calls; no cap overshoot |
| F5: measurement | Executed-action prediction joins and typed provenance | Non-first selected root scored correctly; wrong horizon refused; fixture rows excluded from learned claims; counts reconcile |
| F6: integration | E2 production consumers use F1–F5; compatible artifacts and honest handoff | Small CPU no-update production trace for every supported family/mode, plus deliberately invalid cases; remaining noncognition gates enumerated |

The [source probe](probe.py) is a baseline diagnostic, not an acceptance suite. Its six flags were true on ecc5953. Convert each counterexample into meaningful regression coverage against the repaired public interfaces; do not change a diagnostic to return false unconditionally. Preserve this baseline receipt and add a new receipt for the repair. Tests using scripted predictions must be explicitly labeled as mechanics checks.

## Required negative cases

1. Same value on different variables; same subject at different times; multi-valued observations; nested tool error with long payload; false versus zero and missing versus null.
2. Same goal with opposite received evidence. The actual predictor prompt must differ while changes only to unreceived private state leave it identical.
3. More legal roots than remaining node budget, two legal roots with a useful second step, all invalid predictions, and a best action placed last in the list.
4. Correct prediction for an unchosen action and incorrect prediction for the chosen action. The metric must score the chosen one and retain its ID.
5. Context exhaustion after a received observation, call exhaustion during planning, invalid model action, and unsupported output schema. Preserve failure receipts and denominator membership.
6. Real model boundary with random weights and zero optimizer commits. It may fail every task; its receipt must remain model-origin without claiming trained capability.

## Evidence bundle

Include a handoff, compact test output, baseline and repaired diagnostics, a schema migration inventory, public-input traces, prediction/action joins, source/config/tokenizer identities, and a per-criterion status table. Include exact commands and measured runtimes if reporting them. No large weights or data in Git. Preserve previous reports and failed outcomes.

Report four levels independently: implemented; CPU-verified; GPU-qualified; experimentally supported. This packet can establish the first two. Do not turn CPU fixture success into accelerator qualification. Do not mark the entire campaign ready without checking the remaining operational learner criteria and obtaining chief integration acceptance. No readiness override or local optimizer step is authorized.

## Prompt for the execution agent

Work on BRAMASTRA. Read AGENTS.md, engineering/README.md, engineering/STATUS.md, then every file linked by engineering/cognition_foundation_20260914/README.md. This packet is the current chief response to ecc5953 and specializes the earlier operational learner assignment. Reproduce the semantic counterexamples, implement F1–F6 in the real cognition/data/E2 consumers, and prove corrected behavior with targeted tests and CPU no-update production traces. Preserve the K8 allocation, from-scratch constraint, existing evidence and other agents' files. Do not bypass readiness or run local training. Keep the evidence-repair idea as a separately specified future hypothesis until the cognition foundation is accepted. Submit a new uniquely named handoff with exact identities, commands, outcomes, unresolved gates and committed source; push your scoped work. Report whether the foundation is code-ready, what remains for GPU qualification, and what is still unproven.
