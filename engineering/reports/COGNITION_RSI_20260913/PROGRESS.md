# Cognition/RSI combined progress

Updated: 2026-09-13. Implementing agent: BRAMASTRA implementation lead. Assignment: `engineering/cognition_rsi_20260913/AGENT_PROMPT.md` (combined M00–M24). Baseline: `6d32bbc`.

## Resource discipline

Ledger preserved: **206/200 CPU optimizer updates, 177.234/300 s, zero GPU** — over cap; `SessionLedger.require_learned_allowance` now hard-gates every learned entry point (verified by `tests/test_research_accounting.py` subprocess test). This phase consumed **zero optimizer updates**; all verification is non-learning. No ledger reset.

## Completed this session

- M00 gate: F1–F6 closed (see HANDOFF table; chief probe now refuses F1's cases; F2 orderings correct; F3 negatives reject; F4 lease-leak proven; F5 guard; F6 precision/recovery/gate).
- M01–M06, M12 spine: trajectory/supervision records, candidate-isolated decisions, finite-support world prediction, lexical memory, objective router, episode compilation with teachers, candidate transactions.
- M07: bounded branching planner (Q recursion on declared supports, conservative unknown mass, budgets, root-score-independent selection).
- M19–M24: cognitive workspace/beliefs, executive/deliberation, derivations/abstractions, meta-episodes/curves/comparison/attribution, typed method language with origin validation, fixture generation chain.

## Verification

- Full non-learning suite: **524 passed / 10 pre-existing baseline failures** (historical e1/e2/v5 receipt suites, unchanged at baseline `02b94d3`) / 11 skipped (learned tests gated and NOT authorized at 206/200).
- New focused tests this session: `test_research_master_m01_m04.py` (17), `test_research_master_m05_m06_m12.py` (19), `test_research_cognition.py` (23), `test_research_meta_rsi.py` (18), `test_research_branching.py` (4), `test_research_accounting.py` (8).

## Remaining

See HANDOFF "Remaining blockers / next steps": M08/M09/M10/M13–M18 designed-not-built or partial; learned qualification requires a new owner allocation; DATA_NOT_READY stands.
