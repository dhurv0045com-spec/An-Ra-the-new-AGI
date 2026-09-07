# W02 — Discovery and planning environments

**Status:** assignable against documented W01 interfaces; integration depends on W01. **Effort:** 3–5 hours. **Role:** environment/data engineer. **Compute:** bounded CPU enumeration and qualification.

Read SYSTEM_ARCHITECTURE.md, DATA_CONTRACTS.md and LEARNING_ALGORITHMS.md sections B–C. Own `research/environments/`, `tests/test_research_environments*`, generator fixtures/evidence under a unique run, and `engineering/reports/W02/`.

## Deliverable

Three small, executable environment families with a shared reset/step public API:

1. **Switch laboratory:** hidden acyclic Boolean causal mechanisms, interventions, observed consequences, forbidden direct target inquiry.
2. **Inventory world:** state transitions, prerequisites, overwrites, delayed effects and a bounded goal; enough partial observability to test memory.
3. **Program laboratory:** small hidden programs or faults, typed test inputs, execution feedback and an independently scored prediction/repair goal.

Use finite interpreters/simulators rather than executing arbitrary generated host code. Expose no rule IDs or hidden state to learner records. Keep examiner access explicit and separate.

## Required experiment/data design

Split graph/program semantics before rendering. Generate training/development/confirmation identities deterministically. Qualify correctness and identifiability: an oracle should solve a declared fraction under permitted information, and ambiguous cases need explicit scoring/exclusion rules.

Include the two-query parity case, irrelevant noisy observations, renamed/reordered surfaces, longer prerequisite chains, and mechanism compositions absent from training. New random seeds alone do not establish structural transfer.

## Acceptance evidence

- All three families run through one public observation/action contract with deterministic seed replay.
- Invalid/repeated/forbidden queries and budget exhaustion behave exactly as specified.
- Hidden data cannot be recovered through serialized public records or accidental object references.
- Independent brute-force or interpreter checks validate generated outcomes on small exhaustive spaces.
- Semantic duplicates across splits are zero under the declared canonicalizer.
- Qualification reports include simple random/coverage baselines, oracle performance, ambiguity rates and shortcut diagnostics.
- Each domain provides small training fixtures and held-out mechanism fixtures to W03/W04/W05/W08.

Do not silently relax restrictions to make the oracle succeed. Do not claim broad causal reasoning from calculator-like strings. If three full families do not fit the packet, finish the switch world and shared contracts first, and report exactly which second/third-family criteria remain open; no fake completed stubs.
