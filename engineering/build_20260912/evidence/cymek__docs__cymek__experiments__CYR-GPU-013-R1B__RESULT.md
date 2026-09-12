# CYR-GPU-013 / R1B — FINAL RESULT

**Status:** EXECUTED / COMPLETE  
**Bundle SHA-256:** `7ffebfd49ad0bd8d81035e3cee56b23a5f31f34ba8af0f915408409e31b62792`  
**GPU:** Tesla T4  
**Campaign wall:** 5672.84 s = 94.55 min  
**Primary preregistered verdict:** `MIXED_OR_SEED_SENSITIVE_RESPONSE_CURVE`  
**Claim ceiling:** controlled developmental representation-mechanism evidence only.

## Integrity summary

The returned bundle passed ZIP CRC validation. `SESSION_MANIFEST.json` reports `COMPLETE`. Both mandatory matched curves completed: 12/12 arms reached the exact frozen endpoint of 2,000 optimizer updates / 128,000 semantic row presentations at batch 64. The frozen ARK-002B split identity remained `0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`. Within each prospective seed, all arms report the same shared-core + active-embedding initialization hash.

## Primary endpoint

| Vocabulary classes | seed 3611 / order 5901 | seed 3612 / order 5902 | mean |
|---:|---:|---:|---:|
| 19 | 0.0000 | 0.0000 | 0.0000 |
| 1,024 | 0.0000 | 0.0000 | 0.0000 |
| 4,096 | 0.5059 | 0.4941 | 0.5000 |
| 8,192 | 0.7176 | 0.0000 | 0.3588 |
| 16,384 | 0.6471 | 0.1294 | 0.3882 |
| 24,576 | 0.0118 | 0.0000 | 0.0059 |

Seed 3611 satisfied the preregistered intermediate-regime criterion: best intermediate V8192 = 0.7176 versus best extreme = 0.0118, gap = 0.7059. Seed 3612 retained a large intermediate-vs-extreme gap, but its best intermediate V4096 = 0.4941 did not cross the preregistered 0.60 minimum signal. Therefore the correct frozen verdict is `MIXED_OR_SEED_SENSITIVE_RESPONSE_CURVE`.

## Evidence update

**DEMONSTRATED:** the early capability-formation response to declared tied embedding/output class-space size is non-monotonic under the controlled active-token setup.

**STRONG MULTI-SEED DEVELOPMENTAL EVIDENCE:** the very small V19/V1024 and very large V24576 regimes are poor at the 128k-row endpoint, while the 4096–16384 region contains substantially stronger capability-forming states. V4096 was the most stable fresh intermediate level at this endpoint (~0.50 in both mandatory seeds).

**SUPPORTED / NOT YET DEMONSTRATED:** there may be an intermediate softmax/representation regime that improves structural formation. Exact optimum and mechanism remain seed-sensitive/unresolved.

**NOT DEMONSTRATED:** that 4096 is universally optimal; that this transfers to natural-language tokenizers or larger models; that inactive-softmax competition is the unique cause; PRE500M/500M readiness; broad reasoning or AGI.

## Why the next experiment changes question

R1 and R1B have now spent enough compute showing that class-space size matters. Another broad vocabulary sweep would have low information value. The next experiment should keep the physical V24576 tied matrix fixed and manipulate only the number/weight of inactive classes that participate in the training softmax denominator, while preserving full-vocabulary candidate-free evaluation. This directly tests whether **softmax competition / inactive partition mass is sufficient** to reproduce or rescue the formation effect independently of total parameter count.

Dense diagnostics should measure inactive probability mass, target and active margins, entropy, prediction-error class, gradient partition between active/inactive embedding rows and core parameters, and matched counterfactual-gradient alignment at fixed checkpoints. This turns the next step from another correlation map into a causal mechanism test.
