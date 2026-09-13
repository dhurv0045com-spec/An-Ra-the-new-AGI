# NEXT CORE SPEC (human summary)

**Machine authority:** [`NEXT_CORE_SPEC.json`](NEXT_CORE_SPEC.json) (schema `anra.next-core-spec/v1`) — sufficient for another agent to determine what is locked/provisional/blocked/rejected, why, with which evidence, and what remains unresolved.

## Selected architecture: **V5.1** (Candidate A)

Dense causal decoder-only Transformer, 26 layers × 896 width, GQA 14Q/7KV hd64, SwiGLU 2,368, pairwise RoPE base 10,000, affine QK norm, residual init 0.02/√2L, tied embeddings, vocab 24,576, context 4,096 — **250,216,960 parameters exactly** (receipt reproduced mechanically by `tools/next_core_compute_model.py`).

Training contract: causal CE (answer+EOS supervised, aux λ=0), AdamW(0.9, 0.95) wd 0.1 selective, BF16 compute + FP32 master/moments, global clip 1.0 (tol 1e-4), token-indexed WSD (must execute once end-to-end before any scale run), never stop at train saturation.

Interfaces: certified-corpus-only data consumption with attack screens; external treatment-exact replay with formation-first gates; candidate-free primary evaluation with orthogonal invariance axes, sealed firewall, worst-family reporting; V5 exact-resume checkpoint store.

## Field statuses

- **LOCKED:** family; attention block (mechanism priors D-024/25); precision layout; clipping; objective/EOS; exposure policy; evaluation architecture; checkpoint contract. Each carries evidence refs + falsifier in the JSON.
- **PROVISIONAL:** optimizer hyperparameters; WSD schedule (execution gap); parameter scale; context; tokenizer family; **canonical tied full-softmax output head** after R1C rejected masked-output promotion.
- **BLOCKED:** vocabulary size / physical output geometry (`CS-TRANSFER-001`); numeric/symbol representation (`CS-TRANSFER-001`); continual controller (`GRD-VALID-001`); data interface (`CORPUS-REGEN`).
- **REJECTED:** learned self-model heads; MoE/SSM/recurrence/learned-memory modules; internal Guardian heads.
- **EXPERIMENT_ONLY:** participating_mask / inactive_offset training treatments. R1C completed 24/24 and found inactive-softmax competition **not sufficient**, so these remain hash-visible experimental treatments and never become the default.

## R1C result and output-space consequence

`CYR-GPU-014-R1C` is now **COMPLETE**. The preregistered `MASK_4096 - FULL_24576` formation-AUC gaps were `[-0.139792, -0.242215, -0.019377, -0.040138]`, mean `-0.110381`; both functional and structural primary tests were unsupported / not sufficient. Bundle SHA-256: `2a9359e49f792f962774e42b4e5825b93fb7c91ddcad8f530c1c6f16c6f9bf9e`.

Observed world: **R1C-C / not sufficient**. This rejects a production move to masked-output Candidate B and keeps Candidate A's full-softmax path as the conservative canary default. It does **not** prove physical vocab 24,576 optimal because R1C held the physical matrix fixed. The live next representation gate is therefore an actual physical V4096↔V24576 transfer test (`CS-TRANSFER-001`). See [`R1C_POSTRUN_UPDATE.md`](R1C_POSTRUN_UPDATE.md).

The block family is unchanged; therefore the architecture remains **V5.1**, not V6.

## Why not the alternatives

- **V5 unchanged (the §33 control):** Candidate A *is* that control plus corrections; the spec exists to make the corrections, gates, and blocked fields explicit and machine-checkable.
- **Masked-output Candidate B:** not promoted — R1C's primary mechanism test failed and the paired AUC direction was negative in every seed.
- **Candidate C / physical-geometry dissection:** remains a research direction only if physical-class-space transfer survives; do not ship it before that evidence.
- **Larger V5:** scale is a variable, not a goal (`SCALING_PLAN.md`); no evidence that scale resolves the formation bottleneck; data gate open.
- **MoE/SSM/recurrence/memory:** REJECTED — no isolated bottleneck (`BOTTLENECK` rank 8).

## Claim ceiling (§44)

This is the strongest currently justified next-Core candidate; its implementation satisfies the specified mechanical contracts. R1C narrows one mechanism question but does not authorize a tokenizer change, PRE500M, 500M training, superiority, or AGI claim.
