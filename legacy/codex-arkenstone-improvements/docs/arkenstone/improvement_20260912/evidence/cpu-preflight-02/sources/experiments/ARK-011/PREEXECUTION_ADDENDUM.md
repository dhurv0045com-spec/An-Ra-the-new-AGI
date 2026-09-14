# ARK-011 PRE-EXECUTION ADDENDUM — EXECUTION HARDENING

**Committed before any ARK-011 training result exists.**

This addendum does not change the causal question, seeds, thresholds, LRs, phase budgets, or verdict rules in `PLAN.md`. It freezes operational safeguards for the Colab execution.

## Runtime

- target: Google Colab CUDA GPU; T4 is sufficient;
- default whole-session safety budget: `180 minutes`;
- partial evidence is valid and must be packaged if the session budget prevents later opportunities from starting;
- do not silently substitute CPU/TPU/XLA execution.

## Launcher

A thin Colab notebook will:

1. delete any stale local clone;
2. clone branch `Arkenstone`;
3. checkout the exact commit containing the audited ARK-011 runner;
4. `py_compile` the runner and imported historical ARK-001/task files;
5. run `--smoke-test`;
6. start full execution only after smoke-test PASS.

The notebook itself must not contain the scientific training loop.

## Required smoke tests

Before long training the runner must verify:

- CUDA availability;
- canonical T2 source split hash;
- deterministic CONTROL/SEALED partition, disjointness, and union equality;
- deterministic continuation-order hash reproducibility;
- one finite forward/backward/optimizer step;
- snapshot -> fresh-object reload equality;
- optimizer-state reload works on CUDA;
- two forks restored from the same snapshot and fed the same batch produce equal parameters when their LRs are equal;
- result/receipt writer works.

## Failure behavior

Any exception in full execution must write `FAILURE_RECEIPT.json` containing the traceback and provenance, then package all JSON receipts created so far.

The full result bundle must include source manifest/split identities, plan and addendum commit SHAs, runner source SHA256, checked-out git HEAD, device, torch version, and runtime.
