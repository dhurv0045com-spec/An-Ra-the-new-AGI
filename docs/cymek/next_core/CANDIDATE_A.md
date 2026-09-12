# CANDIDATE A — CONSERVATIVE EVIDENCE CORE ("V5.1")

**Claim:** the smallest architecture consistent with demonstrated evidence is V5 itself, plus only contract corrections and governance fixes. No new blocks, no new mechanisms.

## Specification (all fields exact)

| Field | Value | Status |
|---|---|---|
| family / block | dense decoder, pre-RMSNorm, GQA 14Q/7KV hd64, SwiGLU 2,368, pairwise RoPE 10k, affine QK norm, residual init 0.02/√2L | EVIDENCE_LOCKED (mechanism priors D-024/25; no counter-evidence) |
| params / layers / d_model | 250,216,960 / 26 / 896 | PROVISIONAL (scale is a variable, not a goal) |
| context | 4,096 | PROVISIONAL |
| vocabulary / tokenizer | 24,576 byte-BPE frozen artifact | EXPERIMENT_GATED (R1C + transfer) — carried at V5 default until resolved |
| output head | tied full softmax | EXPERIMENT_GATED (R1C) |
| precision / optimizer / clip | BF16+FP32 master/moments; AdamW(0.9,0.95) wd 0.1; clip 1.0 tol 1e-4 | EVIDENCE_LOCKED |
| objective | causal CE, answer+EOS supervised, aux lambdas 0 | EVIDENCE_LOCKED |
| schedule | token-indexed WSD — **must be executed once end-to-end before any scale run** (execution gap, not design change) | PROVISIONAL |
| data interface | provenance-bound manifests, dedup, contamination screens, token accounting, attack-screened generator surfaces | EVIDENCE_LOCKED (contract), BLOCKED (corpus itself) |
| replay interface | external, treatment-exact sparse replay store (ARK-017 pattern), formation-first gates | PROVISIONAL, external |
| checkpoint contract | V5 exact-resume store unchanged | EVIDENCE_LOCKED |
| evaluation interface | candidate-free primary, orthogonal invariance axes, sealed firewall, attack screens, worst-family reporting | EVIDENCE_LOCKED |

## Why this is the control and likely the winner

1. Every V5 block survives the adversarial questions of §26: it solves no unmeasured bottleneck because it adds nothing; data/evaluation cannot be "fixed" by blocks; nothing moves parameters around; train/inference semantics identical; exact resume untouched; one-seed advantages absent (it adds none).
2. All formation/retention evidence was generated ON this architecture family — changing blocks now would confound every future experiment.
3. §33 control: CURRENT V5 + mandatory corrections IS this candidate; any competing candidate must state why it beats it, and none can today.

## What it does NOT include

No output-space treatment (BLOCKED on R1C), no vocabulary decision (BLOCKED), no controller/replay internalization (external), no MoE/SSM/recurrence/memory/self-model heads (REJECTED/UNJUSTIFIED), no context extension, no depth/width change.
