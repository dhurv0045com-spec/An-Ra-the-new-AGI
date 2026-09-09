# ARK-018 — 1GB REAL-DATA SUBSTRATE BRIDGE

## Status

**PREREGISTERED BEFORE IMPLEMENTATION / DATA BINDING / EXECUTION.**

## Purpose

ARK-015 demonstrated capability narrowing on a controlled non-arithmetic binding task: ordinary canonical accuracy stayed perfect while an acquired presentation-order invariant eroded under HIGH-LR distribution narrowing. ARK-017 will separate update magnitude from invariant-supporting replay on that reliable event generator.

ARK-018 is the next realism gate proposed by the user:

> Does the same broad-capability-retention phenomenon survive when the model's representation has first been shaped by a large real-text corpus, and can we measure the cost/benefit of protection on ordinary language-model quality?

This is **not** a claim that 1GB of text creates general reasoning. It is a bridge from toy-from-scratch models to a real-data-pretrained substrate.

## External dataset contract

The user will place one real-text corpus file in Google Drive. The dataset is external evidence and is not committed to GitHub.

Accepted input formats for the first implementation:

- UTF-8 `.txt` (blank-line document boundaries), or
- UTF-8 `.jsonl` / `.jsonl.gz` with one text field selected by a runtime argument.

Before any tokenizer or model training, the notebook must:

1. mount Google Drive;
2. resolve the exact user-supplied path;
3. compute full-file SHA256 and byte size by streaming the entire file;
4. reject files < 0.90 GiB or > 1.20 GiB unless the user explicitly supplies a different planned size before execution;
5. count/validate documents and UTF-8 decode failures;
6. create `ARK-018_DATA_BINDING_RECEIPT.json` containing path basename, file SHA256, size, parser mode, text-field name if any, document count, and deterministic split algorithm;
7. write that receipt before any training step.

The plan freezes the **binding procedure**, not a file hash that does not yet exist. The bound dataset hash becomes part of every later receipt.

## Data split

Documents are assigned by `sha256(normalized_document_bytes)` modulo 10,000:

- TRAIN: buckets 0..8999 (90%)
- CONTROL: 9000..9499 (5%)
- SEALED: 9500..9999 (5%)

Normalization is limited to canonical UTF-8 decoding plus CRLF→LF. No deduplication or semantic filtering may be added after seeing model outcomes.

CONTROL may guide acquisition qualification and controller state. SEALED is evaluation-only.

Report exact bytes/documents in each split and duplicate-hash rates. If CONTROL or SEALED contains < 10 MiB usable text, execution is blocked.

## Tokenizer

Train an 8,192-token byte-level BPE tokenizer using TRAIN only.

- tokenizer training sample: deterministic first 32 MiB of TRAIN documents ordered by document SHA;
- byte fallback enabled;
- no CONTROL/SEALED text may enter tokenizer training;
- tokenizer JSON/model SHA256 recorded;
- tokenizer package/version recorded and pinned by the implementation.

If tokenizer construction cannot be made deterministic under the pinned library/version, fail closed rather than silently continuing.

## Real-data proxy model

Dense decoder-only Transformer designed to fit a Colab T4 while being materially larger than the Micro subjects:

- vocab: 8,192;
- context: 256 tokens;
- d_model: 384;
- layers: 10;
- heads: 6;
- FFN: 1,536;
- pre-norm;
- tied token embedding / LM head;
- causal attention;
- approximately 20–25M parameters; exact count recorded;
- AdamW, betas `(0.9, 0.95)`, eps `1e-8`, weight decay `0.1`;
- bf16 when supported, fp16 fallback with scaler;
- gradient clip 1.0.

Architecture is intentionally conventional. ARK-018 studies training dynamics, not a new architecture.

## Pretraining budget

The entire 1GB file is scanned and hash-bound, but compute exposure is separately accounted. Do not claim "trained on 1GB" unless the actual token/byte exposure reaches that amount.

A 5-minute CUDA throughput calibration chooses one of these frozen TRAIN token budgets:

- 50M tokens;
- 100M tokens;
- 200M tokens.

Choose the largest budget whose projected pretraining time fits within **180 minutes**, using a conservative 1.25× safety factor. Minimum required budget: 50M tokens. If 50M cannot fit, status `INSUFFICIENT_GPU_THROUGHPUT` and stop before expensive training.

Training examples are deterministic packed streams derived from TRAIN documents ordered by a seeded shuffle; no CONTROL/SEALED tokens are used for gradient updates.

Persist the pretrained checkpoint to Google Drive with model/optimizer/RNG/tokenizer/data hashes so a disconnected Colab runtime does not destroy the substrate.

## Real-language evaluation

At fixed milestones report on CONTROL and SEALED:

- token NLL;
- perplexity;
- next-token accuracy;
- NLL stratified by document-length quartile;
- exact tokens seen;
- gradient norm;
- raw/applied update norm;
- cumulative parameter path;
- relative displacement from the final pretraining checkpoint.

Loss/perplexity are diagnostics only, not proof of cognition.

## Controlled invariant capability on the real-data substrate

After pretraining, teach a language-like temporary-binding task constructed from frequent **single-token words in the trained tokenizer**, rather than digits.

Example form (illustrative only):

`orchid means copper; harbor means silver; cedar means amber. Query: harbor means`

Answer is the bound single token.

Task construction is frozen algorithmically:

- select 12 eligible frequent content-like single-token words using tokenizer frequency rank and deterministic hash filters;
- 6 key words, 6 value words;
- 3 bindings per example;
- complete fact-set split before query expansion;
- 400 TRAIN fact-sets, 50 CONTROL, 50 SEALED;
- all queries represented;
- exact zero fact-set overlap;
- deterministic order augmentation.

This task is controlled on purpose: it gives us a known capability and robustness measure inside a model whose base representations came from real data.

Acquisition uses order augmentation and a HIGH fine-tuning LR chosen as `min(3e-4, final_pretraining_lr)`; LOW is exactly HIGH/100. Qualification is the same three-eval robust gate as ARK-015.

## Retention arms

From the same robust capability checkpoint, on identical canonical-only semantic streams:

1. `REAL_NARROW_HIGH` — HIGH LR, no replay.
2. `REAL_NARROW_LOW` — LOW = HIGH/100.
3. `REAL_NARROW_REPLAY_1OF16` — HIGH LR, exactly 1/16 examples keep alternative fact-order support.
4. `REAL_AUGMENTED_HIGH_REFERENCE` — HIGH LR, full order augmentation.

If ARK-017 has already produced a preregistered mechanism verdict before ARK-018 execution starts, ARK-018 may add **one** optional candidate arm selected by this frozen mapping:

- ARK-017 `UPDATE_MAGNITUDE_SUFFICIENT` -> add HIGH applied-delta CAP1X;
- `DIVERSITY_SUPPORT_SUFFICIENT` -> no extra arm, replay arm is already the candidate;
- `JOINT_CONTROL_REQUIRED` -> add HIGH CAP1X + replay;
- `BOTH_LEVERS_SUFFICIENT` -> add HIGH CAP1X only, keeping replay as the independent comparator;
- any unresolved verdict -> no optional arm.

The optional-arm selection and source ARK-017 receipt hash must be recorded before ARK-018 model training begins.

## Primary endpoints

1. SEALED invariant-capability failure risk/retention by arm.
2. SEALED real-text NLL delta from the pre-capability checkpoint after equal post-capability token exposure.
3. Capability–language Pareto score: preserve robust capability without degrading SEALED real-text NLL by > 5% relative to REAL_NARROW_HIGH.

## Preregistered interpretation

`REAL_SUBSTRATE_RETENTION_TRANSFER` requires:

- robust capability acquired on at least 2 independent fine-tuning seeds;
- at least 4 matched post-acquisition comparisons;
- REAL_NARROW_HIGH produces at least 2 SEALED robustness failures;
- at least one protection arm reduces failure risk by >= .40 without >5% relative SEALED real-text NLL degradation.

`REAL_SUBSTRATE_NO_EVENT` if HIGH produces insufficient failures.

`REAL_SUBSTRATE_TRANSFER_NOT_SUPPORTED` if event rate is sufficient but no protection arm meets the criterion.

## What ARK-018 can and cannot establish

A positive result says the controlled retention phenomenon survives on a model genuinely pretrained from a 1GB real-text source and that a candidate intervention does not obviously destroy held-out LM quality.

It does **not** prove broad reasoning preservation, production-scale benefit, or that the 1GB corpus itself trained AGI. Exact data exposure, parameter count, and compute are reported so the scale of evidence cannot be overstated.
