# MASTER GPU V5 — PRE-EXECUTION ADDENDUM

This addendum is committed before any V5 campaign result is produced. It preserves the scientific questions and frozen intervention values from `MASTER_GPU_PLAN.md` while hardening execution and transfer validity.

## Execution hardening
- New canonical launcher: `arkenstone_master_v5.ipynb`.
- Experiment logic lives in `run_master_v5.py`; the notebook only clones/checks the pinned repository commit and launches the runner.
- CUDA/T4 only. No TPU/XLA dependency and no `xla_sync()` calls.
- Before long training, the launcher performs `py_compile` on the runner and dependencies, then runs `--smoke-test`.
- Smoke test must verify: canonical T2 manifest hash, model forward/backward, optimizer step, snapshot/reload, frozen continuation order hash reproducibility, binding split invariants, and one-step binding training/evaluation.
- Any exception writes a failure receipt and packages all partial JSON receipts in a ZIP in a `finally` path.
- Partial receipts are saved after each completed acquisition/order, not only at campaign end.

## ARK-007R implementation clarification
Scientific design remains unchanged from the binding plan:
- acquisition seeds 909, 1010, 1111;
- continuation seeds 2701..2704;
- HIGH lr=1e-3, LOW lr=1e-5;
- 6000 treated steps;
- G90 onset and G90 confirmation are separate;
- treatment begins only after confirmation;
- paired arms consume byte-identical frozen continuation indices.

## ARK-009 transfer hardening
The intended task remains symbolic non-arithmetic variable binding/retrieval. To prevent hidden fact-set leakage:
- split by complete fact-set/assignment group before forming query examples;
- no fact-set may appear in both train and test, even under a different query;
- train uses 400 fact-sets × 3 queries = 1200 examples;
- test uses 100 fact-sets × 3 queries = 300 examples;
- fixed deterministic task seed 4242;
- query-swap is evaluated throughout acquisition AND retention;
- qualification remains sustained test exact >=0.90 and query-swap exact >=0.85 for 3 consecutive evals;
- transfer retention counts as preserved only when both ordinary test exact and query-swap remain qualified.

## ARK-010 recovery clarification
- recovery starts from the exact checkpoint at confirmation of 3 consecutive post-treatment evals below 0.90;
- both recovery forks consume the identical unused continuation tail;
- recovery HIGH=1e-3, RECOVERY LOW=1e-5;
- 4000 recovery steps;
- recovery90 remains 3 consecutive evals >=0.90;
- if fewer than 2 prospective collapse sources exist, verdict is `INCONCLUSIVE_LOW_EVENT_RATE`.

## Claim discipline
This addendum changes execution robustness and leakage control, not the frozen causal hypothesis. No universal optimizer, AGI, or V5-Core claim is authorized by these experiments alone.
