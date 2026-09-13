# V5.1 Canary-v2 — Extended-Exposure Formation Qualification

**Status:** implemented on `cymek-v51-canary-v2`; substantive result **NOT YET EXECUTED**.

Canary-v1 answered the engineering question: the V5.1 production spine can execute, update parameters, run token-indexed WSD, clip correctly, checkpoint durably, and resume bitwise-exactly. It failed its formation gate at 120 updates / 491,520 tokens: dev exact-with-valid-EOS 0.1331, identity 0.000, binding 0.203, while termination reached 0.313. Training loss fell to 0.612, so the failure was train-fit without adequate transfer rather than a demonstrated execution fault.

V2 asks one narrower question: **does the unchanged Rung-A V5.1 contract cross the formation gate when exposure is extended to a fixed 360 updates / 1,474,560 real tokens (~3 deterministic sampler epochs)?**

## What changes from V1

Only the exposure regime and diagnostics:

- fixed endpoint: 360 updates / 1,474,560 real tokens;
- deterministic epoch-aware sampler (`epoch=0,1,2...`) over one frozen training set;
- fresh generator seed and therefore fresh train/dev/sealed identities;
- checkpoints every 24 updates;
- explicit answer-vs-EOS diagnostics (`answer_exact_ignoring_eos`, answer-prefix exactness, EOS-stop rate, over-generation after a correct prefix);
- fail-closed one-shot sealed-test marker;
- full training trace is merged across resume sessions rather than overwritten.

Everything that could confound the question stays fixed: V5.1 block family, 10,227,456-parameter Rung A, 24,576 physical tied embedding/output matrix, canonical full softmax, tokenizer family, causal CE with EOS supervision, AdamW family/hyperparameters, peak LR 3e-4, production backend, checkpoint store, and V1 formation thresholds.

## R1C boundary

CYR-GPU-014-R1C is complete with `SOFTMAX_COMPETITION_NOT_SUFFICIENT`. Therefore V2 **does not** try MASK_4096, inactive offsets, or any other R1C treatment as a rescue. Those treatments remain non-canonical. R1C did not prove that physical V24576 is optimal; physical class-space transfer remains a later question.

## Scientific identity

Machine preregistration:

`experiments/V5_1_CANARY_V2/PREREGISTRATION.json`

Runner:

`anra_v5/v51_canary_v2_run.py`

Qualification tests:

`tests/test_v51_canary_v2.py`

The operator notebook is created only after the executable is frozen and pins that exact commit. Notebook edits therefore cannot silently mutate the scientific executable.

## Allowed claims

A positive result can establish only that the unchanged V5.1 Rung-A contract satisfies this development-scale formation gate under the extended-exposure regime. It cannot establish cognition, AGI, production-corpus readiness, tokenizer optimality, 250M/500M readiness, or architecture superiority.

If V2 passes, the next high-information experiment is `CS-TRANSFER-001`: a real **physical** V4096-vs-V24576 transfer comparison on clean attack-screened tasks / production-tokenizer rendering. If V2 fails, scaling is blocked and the failure must be diagnosed before larger models are authorized.
