# Agent brief: CYMEK research + readiness handoff

Branch: `cymek-500m-readiness`. Date: 2026-09-08. No training started.

## SOFTWARE
- 500M campaign engine: bucket-pure microsteps, accumulation accounting,
  trainer state machine, durable milestones + persistent mirror, mixture
  scheduler, XLA adapter (pending), eval hooks. 50/50 entry tests.
- Research stack (new): `v5_experiments/cyr_tournament.py` (pure: worlds,
  sampler, controller, metrics, redteam, resolver, packaging) +
  `anra_v5/cyr_execute.py` (torch arms, forks, stage driver, CLI guards).
  Import planes enforced by test.

## EVIDENCE
- Closure cycle: 213 pytest + 50 entry tests green, head-bound receipt
  (`artifacts/v5/cymek_500m_closure_test_receipt.json`).
- Audits: Arkenstone@4911b84 (ARK-011 UNEXECUTED; 007R/010 patterns hold),
  BRAMASTRA@90ee31a (terminal supervision replicated; query-blind proven;
  grouping intervention UNEXECUTED). Matrix:
  `docs/cymek/research/CROSS_BRANCH_EVIDENCE_AUDIT.md`.
- Literature check: plasticity framing supported; no external candidate
  admitted (fashion bar not met).

## CURRENT HYPOTHESES (CYR-GPU-001, preregistered, UNEXECUTED)
- H1 pair-preserving minibatches → query control (decisive either way).
- H2 hysteretic HIGH→LOW + fixed-time control (first execution anywhere).
- H3 displacement-matched intermediate LR (freezing vs consolidation).
- H4 mixture screen: DEFERRED with justification.

## GPU EXPERIMENT READY?
YES — notebook `notebooks/cymek_colab_gpu_research.ipynb`, prereg
`docs/cymek/experiments/CYR-GPU-001/PREREGISTRATION.json` (hash-bound),
thin cells, failure-proof packaging. Status: PENDING_OPERATOR_EXECUTION.
Local validation this cycle: unit suites + tiny smoke only (no training
runs on the operator machine). Full validation (unit suites + entry
suite minus the repo-hygiene self-check) runs as the CELL 0 gate on
Colab hardware; the committed closure test receipt refreshes from Colab
evidence next cycle.

## TPU STATUS
IMPLEMENTED_PENDING_PRE500M_TPU. Zero TPU evidence. GPU results never
count as TPU evidence. PRE500M only after CYR results + Citadel re-audit.

## DATA STATUS
DATA_NOT_READY (500M supply). Tournament uses rendered synthetic worlds
with hashed split manifests — production corpus untouched.

## NEXT OPERATOR ACTION
Open the Colab link, choose GPU runtime, run CELL 0 → CELL 1 → CELL 2,
return CYMEK_GPU_RESEARCH_RESULTS.zip. Do NOT run PRE500M yet.
