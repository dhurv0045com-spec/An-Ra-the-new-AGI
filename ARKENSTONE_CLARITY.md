# Arkenstone: clarity and improvement report

Date: 2026-09-12. Assignment: inspect Arkenstone, improve its engineering substantially, and explain the result clearly.

## What this branch is

Arkenstone is a small-model research laboratory for acquiring, retaining, recovering and transferring symbolic skills. Its current frontier is a state-dependent learning-rate controller: use a high rate to learn/recover and a low rate to retain an already acquired capability. This is a testable hypothesis, not an AGI recipe.

This work starts from `origin/Arkenstone` at `933d4f32019af2f0d61092d092b1c3db67407700`. Changes are isolated on `codex/arkenstone-improvements`. The existing BRAMASTRA checkout and its user changes were not edited. The initial measurements preceded publication; their receipts identify the checkout-base commit plus executed-source hashes. The owner subsequently authorized publishing these changes and the [next-agent work order](NEXT_AGENT_CAPABILITY_WORK_ORDER.md).

## What “10x better” can honestly mean

There is no defined metric under which every part of a research repository becomes ten times better. This pass removes concrete evidence-integrity defects and measures one performance improvement. It does not establish tenfold cognition, training throughput, sample efficiency or overall quality.

The initial five-trial CPU benchmark measured **3.38x faster continuation-stream generation**, comparing the original per-batch implementation to one vectorized draw for 18,000 batches of 64 indices from 500 examples. Every generated index was identical to the original. This is a preparation microbenchmark, not end-to-end training speed. The final-source rerun measured **2.85x** using the same five-trial method. Both runs are retained; timing varies with host load.

A future 10x cognition-efficiency claim should freeze a target held-out exact accuracy, full train/validation/test membership, token/FLOP accounting and multiple seeds before execution; compare total resources needed to reach the target while retaining protected skills. It should report unsuccessful runs and uncertainty. No such claim is made here.

## Concrete improvements

| Problem observed | Change implemented | Practical consequence |
|---|---|---|
| Shared global results folders could blend sessions | Fresh run directory, refusal to reuse an existing directory, explicit output selection | A new run cannot silently replace an old run |
| Partial receipts overwritten directly | Atomic writes plus retained immutable revisions | Completed evidence survives later failed writes |
| ZIP builder gathered unrelated legacy output | Archive limited to the selected run directory | A bundle cannot silently incorporate a stale sibling run |
| Receipts identified only a leaf runner | Hashes and bundled source snapshots include shared model/task/runtime code, the canonical task manifest and current plans | Changed dependencies are visible alongside the checkout-base commit |
| Imports created Colab output directories | Output creation deferred to execution | CPU imports and tests do not write to `/content` |
| CPU optimizer forks could alias snapshot tensors | Deep copy before optimizer-state restoration | Training a fork does not change the snapshot used by the next arm |
| Initial sub-threshold state counted as recurrent collapse | Detect recurrent drops only after confirmed sustained recovery | Recovery screens report actual post-recovery instability |
| Every minibatch used a separate PyTorch RNG call | One vectorized CPU draw, checked against original streams | Faster setup with preserved seed/order semantics |
| Wall budget checked mainly before long work blocks | Monotonic budget checked at the shared training-loss boundary | Expired budgets stop further updates; incomplete arms remain incomplete |
| Latest runners lacked a convenient bounded entry point | Default CPU preflight; explicit CUDA single-campaign mode | Local integrity can be checked without starting long training |
| Failure/interrupt path risked leaving no bundle | Failure receipt and ZIP in the entry point's finally path | Ordinary exceptions and Ctrl-C retain completed artifacts |
| Campaign verdicts and partial-arm retention needed stricter guards | See the [runner handoff](docs/arkenstone/improvement_20260912/RUNNER_HANDOFF.md) | Scientific comparisons use qualified, matched evidence |

A budget is cooperative: an in-flight update/evaluation and final receipt packaging can finish after the deadline. This is not process-level preemption. The runtime preserves metrics, not restartable full model/optimizer checkpoints on disk.

## Final validation

**50 focused tests passed**: 23 runtime/campaign checks and 27 existing Arkenstone checks. The final CPU preflight passed in 7.50 seconds of measured campaign time, and `git diff --check` passed. Receipt hashes, ZIP integrity and all 14 bundled source identities were independently checked against the final working tree.

- [Final preflight receipt](docs/arkenstone/improvement_20260912/evidence/cpu-preflight-02/PREFLIGHT.json)
- [Exact validation commands and logs](docs/arkenstone/improvement_20260912/evidence/final-validation-01/VALIDATION.json)
- [Receipt/source/archive audit](docs/arkenstone/improvement_20260912/evidence/final-validation-01/ARTIFACT_AUDIT.json)

Integration review also rejects duplicated or unknown source identities and requires all four frozen triplets before declaring new-skill acquisition failure. Positive adaptive evidence still requires at least two qualified triplets spanning both acquisition seeds.

## What is established, and what is not

| Claim | Evidence status |
|---|---|
| Canonical T2 and deterministic carry membership | Checked locally, including train/OOD commutation separation |
| CONTROL and SEALED membership separation | Membership separation checked locally; controller routing reviewed in code |
| Exact model/optimizer/RNG fork replay | Checked with the real randomly initialized Micro model on CPU |
| Same snapshot, minibatch and LR produce the same next update | Checked locally; source snapshot also checked for mutation |
| Different LR changes the update | Checked locally as an independent intervention control |
| Faster minibatch-order generation | Measured CPU microbenchmark with identical generated indices |
| Full ARK-012/013 scientific outcomes | Not run in this pass |
| CUDA/T4 runtime and performance | Not tested in this CPU environment |
| Non-arithmetic transfer / scaling / AGI | Not demonstrated by this work |
| Full ARK-014 and combined V6 orchestration | Still missing |

Historical evidence in [the branch research index](docs/arkenstone/README.md) supports Micro T2 low-LR retention and high-LR recovery patterns. Those results are not recomputed, reinterpreted as new evidence, or promoted by these changes.

## Reproduce the engineering checks

From the isolated checkout, using Python with CPU PyTorch:

```powershell
python -m unittest discover -s tests -p 'test_discovery_v6*.py' -v
python -m unittest discover -s tests -p 'test_ark*.py' -v
python experiments/COLAB/run_discovery_v6.py --output-dir artifacts/arkenstone/discovery_v6/my-new-preflight
```

Use a fresh output path each time. The entry point creates receipts, immutable revisions, source snapshots and `ARKENSTONE_DISCOVERY_V6_RESULTS.zip`. Receipt hashes bind the payload; per-file source hashes identify the working-tree implementation even when the checkout-base commit has not changed.

The [work order](docs/arkenstone/improvement_20260912/WORK_ORDER.md) defines ownership and acceptance criteria. The [integration handoff](docs/arkenstone/improvement_20260912/HANDOFF.md) records exact validation results, run identities and remaining limitations.

## Remaining research and implementation gaps

1. Complete and independently qualify ARK-014, then implement the combined V6 scheduler against frozen campaign order and budgets.
2. Validate the current source on CUDA before running full matched 8k/12k training horizons. Existing notebooks pin older commits and need a deliberate new launch pin.
3. Add durable disk checkpoint/restart with optimizer, RNG, sampler position and dataset identity; JSON partials alone cannot resume training.
4. Resolve ambiguous scientific verdict clauses prospectively. Runner implementation thresholds are not automatically preregistered scientific standards.
5. Audit the inherited answer loss before a new protocol revision: it supervises the answer BOS token and does not exclude padding on mixed answer lengths. This pass preserves that historical objective instead of silently changing comparisons. Mixed two/three-digit carry answers make the padding issue relevant.
6. Measure protected-skill retention and new-skill acquisition across all frozen seeds before drawing conclusions. Favorable selected seeds, lower loss and a passing preflight cannot settle the research question.

## Integration conclusion

The engineering pass strengthens reproducibility, evidence preservation, local verification, runtime boundaries and clarity. Acceptance is limited to the exact checks in the handoff. Scientific outcomes and accelerator behavior remain separate, unexecuted work.
