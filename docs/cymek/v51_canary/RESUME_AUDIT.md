# RESUME AUDIT (fresh-process exact resume)

**The mandatory experiment (§24–25), executed.** Two matched trajectories at Rung A on CPU:

- **REFERENCE:** one process, seed 20260913, 4 updates, uninterrupted, checkpoint at 4.
- **RESUME:** process 1: seed 20260913, 2 updates, durable checkpoint, **process terminated**; process 2: brand-new Python process, restores exclusively from the durable checkpoint bytes (`store.restore` → `restore_production`), runs 2 more updates, checkpoint at 4.

`tests/test_v51_canary.py::test_fresh_process_exact_resume_is_bitwise` drives both through real subprocesses and compares the committed artifacts:

| Artifact | Requirement | Result |
|---|---|---|
| model.bin | bitwise identical | PASS |
| optimizer.bin | bitwise identical | PASS |
| ledger.json | identical | PASS |
| cursor.json | identical | PASS |
| scheduler.json | identical | PASS |
| training state (minus parent pointer) | hash-identical | PASS |
| consumed example order | deterministic sampler (same pack + seed) | PASS |

`parent_checkpoint_sha256` is excluded from the state comparison **by documented design**: the resumed trajectory is fenced to its mid-run checkpoint (parent = that checkpoint's sha) while the uninterrupted reference fences to `None` at first publication. This is the parent-fence contract working, not divergence.

## The defect found and fixed (root cause, §50)

First execution FAILED bitwise: `rng_state_sha256` diverged. Investigation: no production component consumes torch RNG during a step (verified empirically); the culprit was the **process-start ambient generator** — PyTorch's default CPU generator state is non-deterministic across fresh processes, and receipts hash it. **Fix (root cause, not symptom):** the runner pins the ambient generator to the canary seed at backend construction (`torch.manual_seed(seed)` in `make_backend`). Regression coverage: the fresh-process test itself. This is exactly the class of defect §24 warned against calling "expected noise".

## Crash injection (§26)

`test_crash_injection_preserves_last_known_good`: publication with `inject_crash_at="after_stage"` raises `InjectedCrash`, `LATEST` remains the last-known-good checkpoint, the stale `.staging-*` directory requires the documented recovery (resolve staging, then republish), and the retry succeeds. `test_corrupted_checkpoint_rejected`: a flipped byte in `model.bin` makes `store.restore` fail closed (hash mismatch) — a partial/corrupt checkpoint can never look complete. `test_identity_mismatch_rejected_on_restore`: a different executable identity (data/config/schedule hashes) is refused by the runner's identity gate.
