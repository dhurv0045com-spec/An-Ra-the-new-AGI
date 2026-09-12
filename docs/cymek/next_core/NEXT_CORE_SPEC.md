# NEXT CORE SPEC (human summary)

**Machine authority:** [`NEXT_CORE_SPEC.json`](NEXT_CORE_SPEC.json) (schema `anra.next-core-spec/v1`) — sufficient for another agent to determine what is locked/provisional/blocked/rejected, why, with which evidence, and what remains unresolved.

## Selected architecture: **V5.1** (Candidate A)

Dense causal decoder-only Transformer, 26 layers × 896 width, GQA 14Q/7KV hd64, SwiGLU 2,368, pairwise RoPE base 10,000, affine QK norm, residual init 0.02/√2L, tied embeddings, vocab 24,576, context 4,096 — **250,216,960 parameters exactly** (receipt reproduced mechanically by `tools/next_core_compute_model.py`).

Training contract: causal CE (answer+EOS supervised, aux λ=0), AdamW(0.9, 0.95) wd 0.1 selective, BF16 compute + FP32 master/moments, global clip 1.0 (tol 1e-4), token-indexed WSD (must execute once end-to-end before any scale run), never stop at train saturation.

Interfaces: certified-corpus-only data consumption with attack screens; external treatment-exact replay with formation-first gates; candidate-free primary evaluation with orthogonal invariance axes, sealed firewall, worst-family reporting; V5 exact-resume checkpoint store.

## Field statuses

- **LOCKED:** family; attention block (mechanism priors D-024/25); precision layout; clipping; objective/EOS; exposure policy; evaluation architecture; checkpoint contract. Each carries evidence refs + falsifier in the JSON.
- **PROVISIONAL:** optimizer hyperparameters; WSD schedule (execution gap); parameter scale; context; tokenizer family.
- **BLOCKED:** vocabulary size (EXEC-R1C + CS-TRANSFER-001); output head (EXEC-R1C); numeric/symbol representation (CS-TRANSFER-001); continual controller (GRD-VALID-001); data interface (CORPUS-REGEN).
- **REJECTED:** learned self-model heads; MoE/SSM/recurrence/learned-memory modules; internal Guardian heads.
- **EXPERIMENT_ONLY:** participating_mask / inactive_offset training treatments (R1C family; independently implemented by BRAMASTRA B04; hash-visible via `v5_next.NextCoreContract.identity_sha256()`; never default).

## Output-space worlds (R1C-A/B/C) and the naming rule

The three R1C worlds each carry architecture/tokenizer/output-head/training consequences and a next required experiment (see JSON `output_space_worlds`). Under every world the block family is unchanged; therefore the architecture is **V5.1**, not V6 — only a World-A/B/C handoff that eventually changes blocks could earn a major version.

## Why not the alternatives

- **V5 unchanged (the §33 control):** Candidate A *is* that control plus corrections; the spec exists to make the corrections, gates, and blocked fields explicit and machine-checkable.
- **Larger V5:** scale is a variable, not a goal (SCALING_PLAN.md); no evidence that scale resolves the formation bottleneck; data gate open.
- **MoE/SSM/recurrence/memory:** REJECTED — no isolated bottleneck (BOTTLENECK rank 8).
- **Candidate B/C now:** both are evidence-gated on R1C; building them today would ship untested science as engineering.

## Claim ceiling (§44)

This is the strongest currently justified next-Core candidate; its implementation satisfies the specified mechanical contracts. Nothing more.
