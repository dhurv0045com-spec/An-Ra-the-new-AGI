# CYR-GPU-012 / R1

**Status: EXECUTED / COMPLETE.**

R1 isolated the effect of tied embedding/output class-space size while preserving the same active arithmetic token IDs, semantic data stream, Cymek V5 block geometry, batch, optimizer family/scalars, model/order seed, shared non-embedding initialization, and first 19 embedding rows.

Final bundle SHA-256: `a22b538396a3d0957a60a27f39b0cf3dd3b20874585b4c15a03207224f613d29`.

At the fixed 8,000-update / 512,000-row endpoint on the executed seed:

- V19: 12.94% STANDARD exact;
- V4096: **100.00% STANDARD exact**;
- V24576: 0.00% STANDARD exact.

Official preregistered verdict: `MIXED_OR_INTERMEDIATE_REPRESENTATION_EFFECT`.

The strongest scientific interpretation is a large, apparently non-monotonic representation/optimization effect. This is one-seed developmental evidence, not proof that 4096 is universally optimal and not authorization for a production tokenizer change, PRE500M, 500M training, broad reasoning, or AGI claims.

Read `RESULT.md` for the complete evidence boundary. The next recommended experiment is R1B: a replicated vocabulary-response curve with direct active/inactive-class competition diagnostics.
