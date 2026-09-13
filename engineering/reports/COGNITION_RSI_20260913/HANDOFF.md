# Cognition/RSI combined handoff — M00–M24 status

Date: 2026-09-13. Implementing agent: BRAMASTRA implementation lead. Baseline: chief review commit `6d32bbc`; agent implementation base `acfa249`. Delivered on `BRAMASTRA` (commit hash recorded in PROGRESS.md); pushed normally, no force-push.

## Executive summary

The chief's F1–F6 findings are closed with non-learning evidence (zero optimizer updates consumed in this phase — the over-cap ledger is untouched). The combined M00–M24 program is then built as real, tested modules: the foundation spine (M01–M06, M12), the cognition extension (M19–M21) and the recursive-improvement extension (M22–M24), plus a bounded branching planner (M07). Every module has focused tests; the full non-learning suite passes at **524 passed / 10 pre-existing baseline failures** (historical e1/e2/v5 receipt suites, unchanged since baseline `02b94d3`). Learned qualification remains **separately unallocated**: the ledger is over cap and no learned check ran in this phase.

## F1–F6 dispositions (M00 gate)

| Finding | Disposition | Evidence |
|---|---|---|
| **F1** prepared identity / resumed data separable | CLOSED. New `_validate_prepared_manifest` recomputes the prepared manifest's own content identity (so mutating rows **and** the split-integrity digest under an old identity is refused), validates schema/inventory/bytes; all consumers route through `_read_prepared`. `commands.resume` binds `prepared["identity"] == run_manifest["data_identity"]` **before** lease acquisition or model allocation. | Chief probe: the changed-rows case now raises `CommandError` ("prepared manifest content identity mismatch"). Negative fixture: rows+digest tampered under old identity → refused. `tests/test_research_pair_path.py`, `test_research_accounting.py`. |
| **F2** per-example trainability corrupts file rule | CLOSED. Entry eligibility is immutable; each record computes a fresh `effective_trainable = entry_trainable and record_trainable` — a row can narrow but never broaden. | Probe now reports `false_then_true: {excluded: false, eligible: true}` and reversed ordering unchanged; admitted counts verified through `_sampler_units` (nontrainable never sampled) and `_collocate_records` (refused at the gradient boundary). |
| **F3** promotion trusts inconsistent/nonfinite nested evidence | CLOSED. `EvidenceBundle.__post_init__` deep-validates metric mappings (finite, [0,1]), family metrics (finite rates, nonnegative integer counts), paired receipts (range-checked rates, gap in [-1,1]) and clustered uncertainty (positive clusters, finite bounds, `ci_low <= ci_high`; `ci_includes_zero` is **derived from bounds**, never trusted). `decide_promotion` requires a positive lower bound for accept and rejects a positive aggregate gain against an entirely negative interval (`aggregate_interval_inconsistent`). NaN family rates reject at construction. | Chief's two negative cases now refuse: negative-interval-with-gain → `no_promotion` with `aggregate_interval_inconsistent`; NaN family → construction error. Tests: `test_research_evaluation.py` (35 tests). |
| **F4** failed resume setup leaks writer lease | CLOSED. `restore_run` acquires the lease **late** (`acquire_lease=False` for the CLI path); all fallible setup (expect-parent, row loading, sampler/replay/controller restore, diagnostics) runs before acquisition; after acquisition a fenced recheck verifies LATEST has not moved, and every failure path releases the lease. No age-based stealing exists. | Runtime proof: wrong `--expect-parent` fails, `writer.lock` absent afterwards, and the next `acquire_writer_fence` succeeds **without force**. Source-verified for constructor failures (lease acquired after construction). |
| **F5** public single-step API ignores configured window | CLOSED. `Trainer.training_step` raises `TrainerStateError` when `grad_accum_steps != 1`, naming the accumulate/finalize window boundary; docstring states the restriction. | `tests/test_research_accounting.py::TrainingStepGuardTests` (guard fires with the configured count in the message; single-window config unaffected). |
| **F6** resource incident and evidence reporting | CLOSED with the incident recorded. Ledger precision: **206/200 CPU updates, 177.234/300 s, zero GPU** — the increment from 193 is exactly 13 (1 failed attempt + two 6-update comparisons); the earlier "≈171 s" figure is superseded by an appended clarification (history preserved, nothing rewritten). Raw acceptance outputs were recovered from session work directories into `reports/B2_2/recovered_acceptance/` (JSONs tracked; `payload.pt` weights local-only via `.gitignore`) **without rerunning the probe**. New hard gate `SessionLedger.require_learned_allowance` refuses any learned entry point — including subprocess probes launched outside `train --smoke` — when the allowance is exhausted; the B2.2 probe now calls it and refuses at 206/200. | `tests/test_research_accounting.py` (8 tests: over-cap refusal, within-budget admission, failed-run consumption with injected callbacks, subprocess bypass prevention, additive precision). `tests/_b22_resume_probe.py --ledger` refusal verified at 206/200. |

## M00–M24 status

| Package | Status | Implementation | Tests | Learned status |
|---|---|---|---|---|
| M00 repairs | CLOSED | F1–F6 above | 8 accounting + chief suites 60/60 | n/a (non-learning) |
| M01 experience contract | implemented, locally checked | `experience/trajectory.py`, `experience/supervision.py` | `test_research_master_m01_m04.py` | n/a |
| M02 decision isolation | implemented, locally checked | `models/decisions.py` (independent rows, dedup, permutation equivariance, action-free value prefix) | same | heads untrained |
| M03 world prediction | implemented, locally checked | `models/world.py` (finite-support normalization, duplicate aggregation, action-conditioned) | same | unqualified |
| M04 memory | implemented, locally checked | `memory/store.py` (lexical overlap v1, filter-before-rank, stable ties, frozen identity, context caps) | same | retrieval is an engineered prior |
| M05 objective router | implemented, locally checked | `learning/router.py` + `experience/supervision.py` (A1 denominators, teacher CE, Huber value, on-policy contract, feature-off routing) | `test_research_master_m05_m06_m12.py` | gradient wiring checked; no optimizer steps |
| M06 episode compilation | implemented, locally checked | `experience/prepare_episodes.py` (MC returns, truncation exclusion, failed submissions never gold, teacher ties uniform) | same | teachers are fixtures |
| M07 branching planner | implemented, locally checked | `planning/branching.py` (Q recursion on declared supports, conservative unknown mass, budgets, root-score-independent selection) | `test_research_branching.py` (4) | fake model in tests; unqualified |
| M08 experience cycle | partially implemented | collection exists; `orchestration/experience_cycle.py` not yet built | — | — |
| M09 curriculum | designed, not implemented | A7 selector specified | — | — |
| M10 preservation wiring | partially implemented | controller/replay adapters exist from B2.1; objective-batch connection pending M05 consumers | — | — |
| M11 examiner extension | partially implemented | matched-case comparison, clustered bootstrap, retention, promotion gates exist (B2.x); full loop coverage pending M03/M07 consumers | `test_research_evaluation.py` | — |
| M12 candidate transactions | implemented, locally checked | `orchestration/candidates.py` (idempotent retries, crash-safe statuses, zero-update weight claims rejected by the API, disjoint namespaces) | same | fixtures only |
| M13 documents | designed, not implemented | manifest infrastructure exists; document compiler pending | — | — |
| M14 environments | partially implemented | three qualified families with oracles; new families pending | `test_research_environments.py` | — |
| M15 runtime readiness | partially implemented | inspect/preflight/leases/ledger exist; backend adapters pending | `test_research_checkpoint.py` | CPU only |
| M16 operator workflow | partially implemented | CLI commands exist; runbook pending | `test_research_cli.py` | — |
| M17 qualification protocols | designed, not implemented | EVALUATION.md is the design | — | no launch authorized |
| M18 integration audit | partially implemented | six fixture stories targeted next session | — | — |
| M19 cognitive workspace | implemented, locally checked | `cognition/workspace.py`, `cognition/beliefs.py` (statuses as provenance classes, reference finite-support update with MODEL_MISMATCH, transitive corroboration roots, acyclic subgoals, pool firewall, deterministic rendering/round-trip) | `test_research_cognition.py` (23) | no learned model |
| M20 executive/metacognition | implemented, locally checked | `cognition/executive.py` (typed operations, frozen fake-scorer origin reporting, cost aggregation once, no-progress termination, pool firewall for summaries, deliberation threshold rule) | same | decision maker is a fake model |
| M21 derivations/abstractions | implemented, locally checked | `cognition/derivations.py` (bounded rule-table checker, invalid steps get no gold, distinct validation cases, counterexample restrict/retract) | same | — |
| M22 meta-episodes | implemented, locally checked | `metalearning/episodes.py` (leakage rejection, conservative missing points, equal-grid AUC, start-state binding, R0–R4 attribution, deterministic dispatch) | `test_research_meta_rsi.py` (18) | no meta-training |
| M23 method language | implemented, locally checked | `metalearning/method_language.py` + `metalearning/generations.py` capture/parse/origin (typed AST, prohibited inputs, deterministic compile identity, forged checkpoints reject, external transcripts labeled assisted) | same | no learned proposer |
| M24 generation chain | implemented (fixtures), locally checked | `metalearning/generations.py` (GenerationReceipt, disjoint registries, reject/crash/retry/no-change paths, substituted-proposer visibility, learned publication gated on chief approval) | same | fixtures only, not learned improvements |

## Decision origin summary (who did the work)

- **Fixed rules / host code:** operation candidate construction, schema validation, the frozen deliberation threshold rule, the compiler, the reference belief updater, retrieval, curriculum designs.
- **Fake/simulated (labeled):** the executive's scorer in tests is a frozen fake model; fixture generation receipts are labeled `fixture=True` and cannot write the learned parent pointer.
- **External-agent assistance:** this implementation itself; `classify_external_submission` labels any unbound transcript `external_assisted`.
- **BRAMASTRA (learned):** nothing yet — no learned decision, proposal or improvement exists. Attribution levels above R0 remain unestablished.

## Resource accounting

- Ledger: **206/200 CPU optimizer updates, 177.234/300 s, zero GPU** — over cap, recorded with cause (see B2.2 HANDOFF clarification). This phase consumed **zero** optimizer updates; all verification is non-learning.
- No paid compute, downloads, accelerator runs or ledger reset.

## Remaining blockers / next steps

1. **F-release gates:** learned acceptance evidence stays provisional until either the recovered receipts are independently verified or a new owner allocation funds a uniquely identified rerun.
2. M08/M09/M10/M13–M18 remain designed-not-built or partial (statuses above); M18's six fixture stories are the next integration target.
3. The next allocated experiment (proposed): rerun the B2.2 acceptance probe under a fresh ledger allocation (6 updates) to produce a uniquely identified, independently verifiable resume-equivalence receipt; then the first bounded meta-episode comparison (R0/R2) per LEARNING_AND_EVALUATION §2.
4. No AGI, 10x, superiority or recursive-improvement claim is made; all capability levels beyond R0 are unestablished.
