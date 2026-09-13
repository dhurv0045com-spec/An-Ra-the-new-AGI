# CYMEK V5.1 Canary-v2 — completed result

Scientific executable: `4470a34b7e2d4b2d673c328ef962c84d8e075b89`

Observed bundle: `CYMEK_V51_CANARY_V2_RESULTS.zip`

## Verdict

`CANARY_V2_FAIL_FORMATION`

This is a narrow preregistered failure, not a broad failure of the V5.1 execution stack. The T4 run reached all 360 updates / 1,474,560 real tokens and every mandatory mechanical gate passed. Training wall time was 443.441 s on a Tesla T4, final loss 0.405396, and WSD, token accounting, checkpoint identity, full-softmax identity, split identity, EOS-contract execution, and durable trace gates all passed.

## Formation result

Development exact-with-valid-EOS: `0.519415`

Sealed exact-with-valid-EOS: `0.516641`

The dev-to-sealed gap is only about `0.00277`, so the large V1→V2 improvement is compatible with real held-out transfer on this controlled instrument rather than development-only overfitting.

| family | development | sealed |
|---|---:|---:|
| identity/copy | 0.1770 | 0.1891 |
| binding | 0.3646 | 0.3313 |
| state/order | 0.5113 | 0.5219 |
| composition | 0.9274 | 0.9349 |
| termination | 0.8563 | 0.8306 |
| missing-info | 0.2799 | 0.2920 |

Frozen formation gates: binding PASS, overall transfer PASS, any-family positive formation PASS, identity acquisition FAIL (`0.177 < 0.30`).

## What this changes

V1's weak result was substantially exposure-limited: extending from 120 to 360 updates raised development exact+EOS from about 0.133 to 0.519 while preserving a nearly identical sealed score. It is therefore no longer defensible to describe V5.1 Rung-A as generally unable to form capability.

The remaining failure is structured rather than broad. Identity/copy remains weak. Its development answer-exact ignoring EOS is only `0.2103`, so the identity failure is not explained only by termination. In contrast, state/order reaches `0.8321` answer-exact ignoring EOS but only `0.5113` exact+EOS, showing a substantial termination component for that family.

The reported `answer_prefix_exact` diagnostic is zero across all families even where exact decoded accuracy is high. Treat that diagnostic as suspect until its tokenization/metric contract is audited; it is not a basis for a scientific conclusion.

## Consequence after R1C

R1C already returned `SOFTMAX_COMPETITION_NOT_SUFFICIENT` at fixed physical vocabulary 24,576. Canary-v2 now shows that longer exposure rescues most formation but not identity/copy. The next decisive representation test must therefore change the **actual physical tied embedding/output class space**, not use masking as a proxy.

The highest-value next experiment is `CS-TRANSFER-001`: matched physical V4096 vs V24576 with identical token sequences for the primary causal task, followed by a production-tokenizer transfer layer as secondary external-validity evidence.

No V2 outcome authorizes PRE500M, 250M, 500M, production-tokenizer replacement, or AGI/cognition claims.
