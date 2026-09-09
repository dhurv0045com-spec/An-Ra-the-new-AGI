# ARK-018 V3 — FINAL EXECUTION CONTRACT

## Status

**FROZEN BEFORE THE V3 RUNNER, NOTEBOOK, DATA BINDING, TOKENIZATION, OR GPU EXECUTION.**

This addendum is the final implementation contract for the science + Birth Book experiment. It preserves the scientific questions and arm definitions in `SCIENCE_BIRTH_PERIODIC_MIXTURE_ADDENDUM.md`, and sharpens execution, durability, diagnostics, and claim boundaries. Later code may implement this document but may not silently change the primary comparison after outcomes are visible.

## Exact external paths

The operator-provided Google Drive screenshot fixes the input location to:

`/content/drive/MyDrive/genisis-arkenstone/data_15.parquet`

The spelling `genisis-arkenstone` is intentional and must not be autocorrected.

Durable ARK-018 outputs live under:

`/content/drive/MyDrive/genisis-arkenstone/ARK018_SCIENCE_BIRTH_V1/`

Subdirectories:

- `prepared/` — tokenizer, compact token caches, immutable preparation receipts;
- `checkpoints/` — resumable per-seed/per-arm checkpoints;
- `results/` — compact JSON receipts and final ZIP;
- `logs/` — append-only operator logs when practical.

The code may stage large sequential reads to `/content/ark018_work/` for speed, but Drive is the durable source of truth for checkpoints and receipts.

## Exact source identities

### Scientific corpus

Expected object: Common Pile consolidated Parquet `peS2o/data_15.parquet`.

Expected SHA256:

`b397427cd5964b7cc2a41264ca8789a0c020f96d4e403b314900798711a2ead5`

Before tokenizer training or any model update, the runner must stream the entire Drive file and require that exact hash. A mismatch produces `SCIENCE_FILE_HASH_MISMATCH` and no training is allowed.

Required Parquet column: `text`. The runner records file bytes, row count, all column names/types, null/malformed text count, package versions and hash in the binding receipt.

### Birth Book

Repository artifact:

`experiments/ARK-018/ARK018_BIRTH_BOOK.md`

Required SHA256:

`8f17d092897f41df45100d227ecf7a2391ed5a5cfd3d3bcf373ad86c94b0c4f4`

Required byte size: `17,906,590`.

The runner recomputes the local checked-out file identity before preparation and fails closed on drift.

## Dataset firewall

Scientific rows are normalized only by UTF-8 representation and CRLF -> LF, hashed by content, then bucketed:

- TRAIN: 0..8999;
- CONTROL: 9000..9499;
- SEALED: 9500..9999.

Exact duplicate normalized text hashes must stay in one split. Report duplicate multiplicity and duplicate byte fraction. Birth Book tokens are TRAIN-only and cannot enter scientific CONTROL or SEALED caches.

No outcome-dependent filtering, deduplication, source selection, or split repair is permitted.

## One tokenizer, no treatment leakage

One 8,192-token byte-level BPE tokenizer is trained from scientific TRAIN only. The Birth Book cannot influence vocabulary learning.

Tokenizer training sample: first 32 MiB of scientific TRAIN documents after deterministic ordering by document SHA256.

The implementation pins tokenizer package/version, fixes single-threaded preparation where supported, builds the tokenizer twice from the exact same sample, and requires byte-identical serialized tokenizer SHA256. Failure to reproduce blocks training.

The runner then records:

- tokenizer SHA256;
- exact scientific TRAIN/CONTROL/SEALED token counts;
- Birth Book token count + token-stream SHA256;
- replay-control token count;
- every cache SHA256 and dtype.

## Model

Conventional decoder-only proxy; this experiment is about training dynamics/data, not architecture novelty:

- vocab 8,192;
- context 256;
- d_model 384;
- 10 transformer blocks;
- 6 attention heads;
- FFN 1,536;
- pre-norm;
- learned positional embeddings;
- tied token embedding / LM head;
- causal scaled-dot-product attention;
- approximately 20–25M parameters; exact count receipt-bound.

Optimizer:

- AdamW betas `(0.9, 0.95)`, eps `1e-8`, weight decay `0.1`;
- peak LR `3e-4`;
- 2% linear warmup;
- cosine decay to `3e-5` over the frozen arm horizon;
- gradient clipping at 1.0.

Use bf16 on hardware that supports it reliably; otherwise fp16 autocast + GradScaler.

## Equal-token update contract

Each optimizer update contains exactly 32 sequences x 256 prediction targets = **8,192 target tokens**.

GPU calibration selects the largest fitting microbatch from `{32,16,8,4}` and uses gradient accumulation so the effective batch remains exactly 32 sequences. Microbatch size can change runtime efficiency but not the scientific batch definition.

All matched arms within a seed use the same optimizer-update count and target-token count.

## Primary four-arm mixture

The 100-step macrocycle in the earlier addendum is unchanged:

1. `SCIENCE_ONLY`: 100/100 science.
2. `BIRTH_NATURAL_2PCT`: Birth at relative steps 25 and 75; 98 science steps.
3. `BIRTH_REHEARSAL_10PCT`: Birth at 5,15,...,95; 90 science steps.
4. `SCIENCE_REPLAY_10PCT_CONTROL`: same 10 replay slots as arm 3 but a deterministic scientific replay corpus token-matched to the Birth Book within 0.1%.

Birth/replay cursors traverse sequentially and wrap only at EOF. Normal scientific batches are deterministic from `(model_seed, absolute_step)` so shared science steps are exactly matched across arms.

Model initialization seeds remain:

- `31801`
- `31902`

All four arms for a seed load an exact identical initial model state. A strong causal conclusion requires both seeds; one seed is explicitly a screen.

## Horizon

Compute after tokenizer freeze:

`horizon = max(6200, ceil_to_100(ceil(birth_tokens / (8192 * 0.10))))`

then clamp to 8,000 updates.

Every arm uses the same horizon. If the 10% Birth arm cannot complete one Birth pass by 8,000 updates, report actual coverage and do not claim a complete pass.

No arm may terminate early because its result looks good or bad.

## Durability and exact resume

The campaign is allowed to span multiple Colab sessions. Scientific validity is more important than pretending a full two-seed/four-arm study fits one runtime.

For each `(seed, arm)`, checkpoint at least every 1,000 updates and at final. A checkpoint contains:

- model state;
- optimizer state;
- mixed-precision scaler state when used;
- absolute step;
- CPU and CUDA RNG states;
- source/birth/replay cursor state or enough frozen deterministic state to reconstruct it exactly;
- tokenizer/data/Birth/runner identities;
- telemetry aggregates;
- initial model-state hash.

Compact partial JSON is copied to Drive at every 500-step evaluation. Re-running the notebook must resume instead of silently restarting a partially completed arm.

## Maximum-information evaluation battery

Primary comparisons stay frozen. The following diagnostics increase information without changing treatment assignment.

### A. Scientific competence

At step 0, every 500 updates and final, on fixed CONTROL and SEALED science caches:

- token NLL;
- perplexity;
- next-token accuracy;
- fixed evaluation token count;
- NLL by available length category when feasible.

### B. Birth Book internalization

Freeze `BIRTH_BOOK_PROBES.json` before the runner executes. Score objective multiple-choice / short-completion likelihood families:

- project-record facts;
- architecture/training concepts;
- science/ML concepts covered in the hand-written book;
- paraphrased relational/compositional questions;
- matched distractors.

Also report Birth Book held-out-slice NLL as a memorization/internalization diagnostic. Neither metric alone is called reasoning.

### C. Narrow external transfer / reasoning diagnostics

The runner freezes deterministic algorithmic probes outside major generated Birth Book table ranges, including examples such as:

- binary/hex conversion above the book's 16,384 table boundary;
- number-name conversion above 20,000;
- arithmetic compositions outside the book's tabulated ranges.

These are narrow transfer diagnostics, not broad reasoning claims.

If internet access is available, the notebook may also score a fixed deterministic subset of `allenai/sciq` as a **secondary external diagnostic only**. Dataset fingerprint and download status are recorded. SciQ failure/unavailability cannot fail the primary experiment, and contamination is not assumed absent unless explicitly checked.

### D. Capability acquisition and retention

After pretraining, each final checkpoint receives the same frozen ARK-018 word-token temporary-binding adaptation protocol:

- deterministic token namespace selected from scientific TRAIN tokenizer frequencies;
- order-augmented acquisition;
- CONTROL qualification only;
- SEALED measurement;
- acquisition steps to robust qualification;
- if qualified, matched HIGH vs LOW canonical-only narrowing continuation for a bounded horizon.

This asks whether pretraining mixture changes *learning dynamics and robustness*, not just whether it remembers Birth Book facts.

### E. Optimization/mechanism telemetry

Record:

- training loss and LR;
- exact pre-clip gradient norm each update;
- source identity each update;
- exact cumulative source token counts;
- exact full-model displacement from initialization at evaluation milestones;
- Birth/replay cursor and epoch counts.

To avoid a misleading computationally expensive claim, per-step update-path and gradient-alignment diagnostics may use a **frozen deterministic parameter projection**. If so, receipts must name every projected tensor and explicitly label all such values `PROJECTED_*`, never full-model exact quantities.

At every 1,000 updates, compute projected science-vs-Birth gradient cosine/norm ratio on fixed CONTROL mini-batches. This tests whether the two data sources become aligned, orthogonal or interfering.

## Primary verdict remains prospectively fixed

Primary content-specific comparison:

`BIRTH_REHEARSAL_10PCT` vs `SCIENCE_REPLAY_10PCT_CONTROL`.

Across both seeds, Birth-specific internalization without major science cost requires:

- BirthContentScore improvement >= 0.10 absolute;
- same direction in both seeds;
- <=5% relative SEALED scientific NLL worsening.

Secondary practical comparison:

`BIRTH_NATURAL_2PCT` vs `SCIENCE_ONLY`.

A cognition/learning claim requires a separately preregistered downstream capability metric to move consistently across both seeds. BirthContentScore alone is content internalization.

## GPU smoke gate

Full execution is blocked until a T4/CUDA smoke test passes:

- exact science path exists;
- full science file SHA256 matches expected upstream hash;
- Parquet schema contains `text`;
- Birth Book hash/size match;
- split code and source schedule invariants pass;
- model parameter count is within planned range;
- forward/backward finite;
- 8,192 target tokens/update verified;
- model+optimizer snapshot reload exact enough for deterministic next-update reproduction;
- checkpoint write/read succeeds;
- no SEALED data enters any training cache.

## Claim boundary

Even a strong positive ARK-018 result can show only that structured Birth Book exposure causally changes internalization and possibly measurable downstream learning/retention on this ~20–25M real-science proxy. It does not demonstrate consciousness, selfhood, AGI, general reasoning, or production-scale benefit.
