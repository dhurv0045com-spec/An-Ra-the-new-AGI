# Gandiva cognition and TPU preflight handoff

**Branch:** `Gandiva`
**Implementation revision:** `655c8660c4cde7034bbba087b95d695e5feb0b9b`
**Scope:** bounded model-based cognition repairs and a Kaggle TPU v3-8, zero-update backward preflight.

## What is implemented

The dedicated [`bramastra_tpu_100m_preflight.ipynb`](../../../notebooks/bramastra_tpu_100m_preflight.ipynb) and [`tpu_100m.py`](../../../bramastra_lab/research/campaigns/tpu_100m.py) form an isolated TPU path. The notebook clones the pushed Gandiva branch when needed, discovers or builds the prepared K8 data bundle, validates it, runs the preflight, then packages reports into a ZIP with a SHA-256 receipt. It does not modify or replace [`bramastra_k8.ipynb`](../../../notebooks/bramastra_k8.ipynb).

The preflight refuses to continue unless Kaggle exposes the expected eight-replica PJRT TPU. Before launching workers, it compiles a deterministic plan of eight distinct training rows and counterfactual partners from the validated bundle. On each rank, it constructs the registered `tpu_100m` configuration (100,334,720 parameters), broadcasts and verifies identical initial state, then runs one BF16 B-arm forward/backward window and checks gradients for finiteness. It does not call the optimizer update boundary. A pass proves only that this backward path ran; it does not prove optimizer-state fit, a committed training update, checkpoint/resume, sustained throughput, or campaign readiness.

The cognition updates make depth-two planning condition on a **local imagined successor** containing the first action and its predicted feedback. This synthetic event is not appended to actual episode history. Search obeys remaining node/call budgets, rotates a partially covered root set, and shares child rollouts round-robin across root actions. The model adapter validates response structure and records inference token counts; planner use is charged in trace metadata. E2 calibration now joins the executed root action only to its depth-one prediction, never to an imagined child. Failures and exhausted budgets use explicit fallbacks rather than silently treating missing predictions as evidence.

## Local verification

All checks below ran on the implementation revision before this report was added:

```text
python -m py_compile bramastra_lab/research/cognition/episode.py bramastra_lab/research/campaigns/phases/e2.py bramastra_lab/research/campaigns/tpu_100m.py tests/test_research_cognition_planning.py tests/test_research_tpu_100m.py
PASS

python -m unittest tests.test_research_cognition_planning tests.test_research_tpu_100m -v
19 tests passed

python -m pytest -p no:cacheprovider tests/test_research_cognition.py tests/test_research_cognition_runtime.py tests/test_research_gandiva_rsi_cognition.py tests/test_research_k8_operational.py -q --maxfail=1
84 passed, 16 subtests passed

python -c "from bramastra_lab.research.campaigns.tpu_100m import _build_sample_plan; p=_build_sample_plan('.codex-test-tmp-gandiva-20260923-data', seed=1701); print({'training_rows':p['training_rows'],'workers':len(p['workers']),'objective_counts':[w['objective_counts'] for w in p['workers']]})"
training_rows=49152; workers=8; every worker has pair denominator=2

git diff --check
PASS
```

The CPU check compiled real examples from the available prepared bundle. No 100M model was instantiated on this computer, no TPU device was available here, and no optimizer update was attempted. The Kaggle notebook has not yet been run; there is no TPU memory or execution result to report.

## Exact next action

Run the dedicated notebook on Kaggle with the TPU v3-8 accelerator selected and Internet enabled for the branch clone. Kaggle's [TPU documentation](https://www.kaggle.com/docs/tpu) and [Notebook documentation](https://www.kaggle.com/docs/notebooks) document the accelerator and currently state a nine-hour notebook-session limit; confirm live availability before starting and reserve time for the final ZIP and receipt download.

Review all eight `worker-*.json` files and the aggregate `preflight.json`. The only acceptable success state is `BACKWARD_PREFLIGHT_PASS`, eight ranks with matching initialization state, finite gradients, and exactly zero optimizer updates and attempts. Download and preserve the ZIP and SHA-256 JSON receipt. On failure, retain the failed run ZIP/logs and fix the earliest failing contract before a rerun; do not clear or replace evidence.

After a successful preflight, the next engineering gate is still to implement and separately qualify: a sharded cognition-bearing training stream with complete optimizer-window partitioning; optimizer-state memory measurement; exactly one committed update with finite state; and atomic checkpoint save/load/resume across eight ranks. Only then can a measured update budget or longer campaign be proposed. The full cognition F1–F6 acceptance contract and recursive self-improvement remain unproven. This branch does **not** claim AGI or demonstrated task competence.

## Agent prompt for the next phase

> Work only on BRAMASTRA branch `Gandiva`. Read `AGENTS.md`, `engineering/TPU_100M_COGNITION_PROGRESS.md`, and this handoff first. The pushed implementation is `655c8660c4cde7034bbba087b95d695e5feb0b9b`. Preserve the uncommitted owner edits in `notebooks/bramastra_k8.ipynb` and `tests/test_research_k8_real.py`; do not stage them. Run `notebooks/bramastra_tpu_100m_preflight.ipynb` on Kaggle TPU v3-8 with Internet enabled, capture the complete output, and download the ZIP plus SHA-256 receipt. Report the actual status and per-rank evidence. This notebook is a zero-update backward check: do not call it training, do not claim TPU qualification if it fails, and make no AGI/RSI claim from it. Preserve failed evidence. If it passes, design and implement the next bounded gate—correct global optimizer-window sharding, peak memory including optimizer state, one finite committed update, and eight-rank checkpoint/resume—with focused tests and unique evidence paths. Do not launch a multi-hour training campaign until those gates are implemented and reviewed. Update this handoff with measured results and exact source revision; commit and push only scoped files, without force-pushing.
