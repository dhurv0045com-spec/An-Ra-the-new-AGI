# Chief design handoff: complete learner program

Date: 2026-09-13. Branch: BRAMASTRA. Inspected source: `baa655df7989ab86c447c853fc23703c8b894fe7`, including agent build `c3662b3`. This report concerns a design and execution assignment, not implementation acceptance for M00–M18.

## Delivered

The [master program](../../master_program_20260913/README.md) contains seven Markdown specifications, a machine-readable dispatch manifest, nineteen packages and 12,626 whitespace-delimited Markdown words at validation. The scope includes the complete next learner architecture, exact objective equations and normalization, public outcome prediction, action/value learning, memory, bounded inquiry/search, teacher/trajectory preparation, curriculum, replay/retention, independent evaluation, candidate transactions, from-scratch corpus support, richer tasks, runtime readiness and the operator workflow.

The chief authored the program. One existing Luna agent performed two bounded read-only audits: current learning connections, then design consistency. No production implementation was delegated or performed during this expansion. No new experiment, model allocation or optimizer update was run. The owner receives the [complete operator prompt](../../master_program_20260913/AGENT_PROMPT.md); an implementation agent can obtain the full assignment from repository files alone.

## Source findings that shaped the design

`models/wrapper.py` has action/value heads, but `learning/trainer.py` consumes token/pair supervision only. `planning/planner.py` scores roots and generates continuation text without recursive typed outcome valuation. `LearnedCollectionPolicy` is explicitly an untrained hook. `collection/runner.py` writes observed receipts without joining them to trajectory objectives. Episodic retrieval and working-memory input are absent from the canonical learner path. The new program addresses these specific gaps; it preserves existing data/runtime/evaluator work and the answer-only baseline.

The prior chief R1–R7 review remains authoritative. M00 incorporates all B2.2 repairs. The owner's expanded request changes the dispatch boundary: after M00's checks/report, agents may continue provisional build integration. It does not award chief acceptance or scientific promotion. Root AGENTS/README, engineering entry documents, historical specification headers and the paper's implementation-status note were updated so an executor will find one current assignment.

## Review corrections incorporated

1. Initial recursive expected-return search is limited to declared finite outcome supports. Open-text prediction is supported separately; sampled planning requires an explicit estimator/uncertainty contract and an approximate-method identity.
2. ResourceVector and one comparison predicate distinguish equal allowance, equal realized resources and cost/quality tradeoffs. Unknown required quantities block the claim instead of becoming zero.
3. M06 has an explicit seed-ledger prerequisite obtainable from the existing real fixed-policy collector before M08. At most six short episodes, zero model updates, with actual interaction accounting.
4. Candidate transactions have execution/evidence classes, change kinds and verified update counts. Fixture registries cannot publish learned parents. Real learned publication requires authentic receipts and hash-bound chief approval. Qualified synthetic task data remains valid; evaluator test doubles are not learned evidence.

Chief review also required candidate-isolated scoring to avoid causal enumeration bias, action-free value prefixes, separate objective denominators across accumulation, missing/truncated target eligibility, on-policy checkpoint binding, future-event exclusion, memory scope enforcement and preservation of the cumulative ledger.

## Verification and limits

Run from the BRAMASTRA worktree:

```powershell
& 'C:/Users/ankit/Downloads/An-Ra-the-new-AGI-1/.venv/Scripts/python.exe' engineering/reports/MASTER_DESIGN_20260913/validate.py
git diff --check
```

The design validator checked all nineteen package IDs, an acyclic dependency graph reaching every package from M18, agreement between prose and manifest dependencies, and 96 local Markdown links across the master and entry documents. It returned no errors. It computes canonical UTF-8/LF hashes for the eight primary packet files. The immutable [validation receipt](validation-001.json) records the output. These are structure/link checks, not algorithm implementation tests. Git whitespace validation is separate.

The historical ledger remains at 193 CPU optimizer updates, 111.234 learned-smoke seconds and zero GPU usage. No additional allocation is implied. Actual corpus is DATA_NOT_READY; TPU/backend qualification and learned efficacy remain unresolved. The requested 100 million implementation-agent tokens is recorded as an external-app preference, with no fabricated allocation or consumption.

## Next action

The external implementation agent should read the master prompt and execute M00–M18 with the defined dependencies, file ownership, budget and evidence rules. Its final output belongs in `engineering/reports/MASTER_20260913/HANDOFF.md`, distinct from this chief-design report. It should deliver the build and qualification protocols, not launch an unallocated campaign. AGI, 10x improvement and superiority to all historical branches remain unestablished claims requiring evidence beyond this design.
