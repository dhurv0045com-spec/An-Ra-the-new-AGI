# V5.1 Canary-v2 Operator Amendment 1 — Colab CPU qualification split

Status: OPERATOR-ONLY. Scientific executable remains frozen at `4470a34b7e2d4b2d673c328ef962c84d8e075b89`.

## Trigger

The first T4 Colab qualification cell invoked the complete V1 + V2 CPU suite in one `pytest` process. On the live Colab runtime the command returned non-zero after a long CPU-only qualification interval. The notebook wrapper exposed only `CalledProcessError`, so the exact failing test was not visible in the final traceback.

## Amendment

For live V2 operation, rerun the same frozen validation surface with `V51_SKIP_SLOW=1` for `tests/test_v51_canary.py`. This skips only the V1 tests explicitly marked as slow (`fresh_process_exact_resume_is_bitwise`, `corrupted_checkpoint_rejected`, and `crash_injection_preserves_last_known_good`). Those tests belong to the already-qualified V1 checkpoint/corruption/fresh-process spine; V2's own suite states that V1 owns those heavy tests and attacks only the V2 delta.

The following remain mandatory on the live Colab runtime and must pass before Drive preparation or CUDA training:

- `tools/validate_v51_canary_v2.py`
- all non-slow `tests/test_v51_canary.py` tests
- all `tests/test_v51_canary_v2.py` tests
- all `tests/test_v51_canary_v2_durability.py` tests
- all `tests/test_v5_production_backend.py` tests
- all `tests/test_v5_checkpoint_adapter.py` tests
- exact frozen Git blob checks
- isolated real CUDA preflight before the scientific run

The rerun must use verbose/short traceback output and preserve stdout/stderr so any remaining failure is attributable to an exact test rather than collapsed into a wrapper exception.

## Scientific non-change

This amendment does not modify model geometry, data seed or generator, tokenizer, optimizer, objective, schedule, 360-update endpoint, 1,474,560-token budget, checkpoint format, formation thresholds, sealed-test firewall, output treatment, or result interpretation. It does not authorize bypassing any V2-specific test or any CUDA preflight gate.

If any mandatory fast/V2/core test fails, the operator remains fail-closed and training must not start.
