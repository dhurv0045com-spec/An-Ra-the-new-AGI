# ARK-018 — SCIENCE + BIRTH BOOK PERIODIC MIXTURE ADDENDUM

## Status

**PREREGISTERED BEFORE IMPLEMENTATION, DATA BINDING, TOKENIZATION, OR GPU EXECUTION.**

This addendum sharpens ARK-018 into a direct causal test of the user's current proposal: train one conventional real-text proxy on the ~959 MB scientific `data_15.parquet` shard while periodically exposing it to the 17.9 MB `ARK018_BIRTH_BOOK.md`, and compare against controls that separate Birth Book content from the generic effect of repeatedly replaying a small corpus.

It supersedes the original ARK-018 `PLAN.md` only where this document explicitly changes input format, source scheduling, experiment arms, and primary pretraining comparisons. The original model/evaluation/claim boundaries remain in force unless explicitly replaced here.

## Scientific question

> Does bounded, repeated exposure to the Birth Book produce measurable and transferable changes in a model trained primarily on scientific text, beyond what is caused by repeatedly replaying any small corpus, while preserving scientific language-model quality?

A positive result is **not** evidence of consciousness, identity, AGI, or broad reasoning. The Birth Book is treated scientifically as a structured training corpus. We test content internalization, downstream capability acquisition/retention, and costs to scientific modeling.

## Frozen source A — Birth Book

Live Arkenstone artifact at preregistration:

- path: `experiments/ARK-018/ARK018_BIRTH_BOOK.md`
- edition: 3
- bytes: `17,906,590`
- words: `3,093,369`
- SHA256: `8f17d092897f41df45100d227ecf7a2391ed5a5cfd3d3bcf373ad86c94b0c4f4`

The implementation must recompute the GitHub checkout file SHA256 before tokenization and fail closed on mismatch.

The Birth Book is TRAIN-only. No verbatim Birth Book span is permitted in scientific CONTROL/SEALED evaluation.

## Frozen source B — scientific corpus

User-selected Google Drive object:

- expected folder: `My Drive/genisis-arkenstone/`
- expected basename: `data_15.parquet`
- Drive UI currently reports approximately `959 MB`.
- intended upstream source: Common Pile openly licensed peS2o scientific-paper shard `data_15.parquet`.
- expected upstream file SHA256, to be verified against the Drive copy before training: `b397427cd5964b7cc2a41264ca8789a0c020f96d4e403b314900798711a2ead5`.

The implementation must stream-hash the entire Drive file before any tokenizer or model update. If the hash differs, execution stops with `SCIENCE_FILE_HASH_MISMATCH`; the file may not be silently rebound after outcomes are visible.

The Parquet parser is now explicitly supported. Expected scientific text column: `text`. Schema, row count, byte size, column names, file hash, parser/library versions, and any malformed/null-text counts are written to `ARK-018_DATA_BINDING_RECEIPT.json` before training.

## Scientific split

For every peS2o row with usable `text`:

1. normalize only CRLF -> LF and UTF-8 representation;
2. compute `doc_hash = sha256(normalized_text_bytes)`;
3. assign `int(doc_hash[:16],16) mod 10000`:
   - TRAIN: 0..8999
   - CONTROL: 9000..9499
   - SEALED: 9500..9999.

Exact duplicate normalized texts therefore land in the same split. Report duplicate-hash counts. No semantic filtering/dedup policy may be changed after model outcomes are observed.

The Birth Book never enters CONTROL or SEALED.

## Tokenizer — one tokenizer for every arm

Train one deterministic 8,192-token byte-level BPE tokenizer from **scientific TRAIN only**. The Birth Book must not influence tokenizer construction; otherwise tokenizer differences would confound the treatment.

Use the same tokenizer artifact and SHA256 for all model seeds and all arms. Byte fallback is required.

After tokenizer freeze, tokenize the entire Birth Book and record:

- exact Birth Book token count;
- tokenized SHA256;
- number of 256-token packed sequences;
- expected full-pass count under each treatment arm.

## Model / optimizer

Retain the original ARK-018 conventional proxy:

- dense decoder-only Transformer;
- vocab 8,192;
- context 256;
- d_model 384;
- 10 layers;
- 6 heads;
- FFN 1,536;
- pre-norm;
- tied embedding / LM head;
- AdamW `(beta1=.9, beta2=.95, eps=1e-8, weight_decay=.1)`;
- gradient clipping 1.0;
- bf16 when supported, fp16+scaler otherwise.

Exact parameter count is receipt-bound.

### Fixed effective update size

Target **32 full 256-token sequences per optimizer update = 8,192 tokens/update**. The CUDA smoke/calibration may select a microbatch from `{4,8,16,32}` sequences and use gradient accumulation to reach exactly 32 sequences. If a microbatch of 4 cannot fit, fail before training.

Every arm has the same number of optimizer updates and the same 8,192-token update budget. Source choice changes content, never update size.

## Why the 17.9 MB book is not replayed in full every 50 steps

A full 17.9 MB book is millions of tokens. Replaying the *entire* book every 50 or 100 optimizer steps would massively oversample it and turn a ~1.8% corpus-size component into the dominant training signal.

Instead, this design implements the user's intent precisely but fairly: **every 50-step window in the natural-ratio arm contains one source-pure Birth Book update**, and Birth Book batches advance through the book sequentially with no random skipping. Thus the model repeatedly returns to the book while scientific text remains the dominant substrate.

## Deterministic 100-step macrocycle

For absolute optimizer step `s`, define `r = ((s-1) mod 100)+1`.

Scientific batches are generated from a deterministic seed-bound scientific stream indexed by absolute step. On all steps shared by two arms, the scientific batch IDs are byte-identical.

Birth Book batches use a deterministic sequential cursor through the frozen tokenized book. A Birth step consumes the next 8,192 Birth tokens (packed as 32x256); at EOF the cursor wraps to token 0 and increments `birth_epoch_count`. No section is selectively sampled after outcomes are known.

### Arm A — `SCIENCE_ONLY`

- all `r=1..100`: normal scientific TRAIN batch.
- purpose: ordinary scientific pretraining baseline.

### Arm B — `BIRTH_NATURAL_2PCT`

- Birth Book at `r in {25,75}`;
- normal scientific TRAIN on the other 98 steps.

This is exactly **2 Birth updates per 100**, close to the observed file-size ratio (~17.9 MB vs ~959 MB) and guarantees one Birth exposure in each 50-step half-cycle.

### Arm C — `BIRTH_REHEARSAL_10PCT`

- Birth Book at `r in {5,15,25,35,45,55,65,75,85,95}`;
- normal scientific TRAIN on the other 90 steps.

This intentionally oversamples the book to test whether repeated rehearsal produces a stronger, measurable effect. It is not called a natural corpus mixture.

### Arm D — `SCIENCE_REPLAY_10PCT_CONTROL`

- same special slots as Arm C;
- those 10 slots draw from a **small repeated scientific control corpus** rather than Birth Book;
- remaining 90 steps use the same normal scientific TRAIN stream as Arm C.

The small scientific replay corpus is created *before training* by sorting scientific TRAIN documents by deterministic hash and adding documents until its **token count matches the frozen Birth Book token count within 0.1%**. It is then traversed sequentially and wrapped exactly like the Birth Book.

This is the critical content control: C and D have the same optimizer count, source-repetition cadence, small-corpus size, wrap behavior, and 90% ordinary-science exposure. The systematic difference is Birth Book content versus similarly repeated scientific content.

## Seeds / matching

Target two independent model initialization seeds:

- seed A: `31801`
- seed B: `31902`

For a given seed all four arms begin from an **exact identical randomly initialized model snapshot** and the same optimizer state/RNG contract. Seed B may run in a separate Colab session, but its plan/seeds/schedules remain frozen.

A strong claim requires both seeds. One completed seed is a SCREEN only.

## Training horizon

Target per arm is the larger of:

1. **6,200 optimizer updates** (~50.79M packed tokens at 8,192 tokens/update), or
2. the smallest multiple of 100 updates that lets `BIRTH_REHEARSAL_10PCT` traverse at least **one complete tokenized Birth Book pass**.

Hard cap: **8,000 updates per arm**. If one full Birth Book pass would require more than 8,000 updates under the frozen tokenizer/batch contract, keep the 8,000-step cap, record the actual coverage fraction, and do not claim full-book exposure.

All four arms within a seed use the exact same horizon. No arm stops early because it looks better.

## Evaluation schedule

Checkpoint/evaluate at:

- step 0;
- every 500 optimizer updates;
- final step.

SEALED metrics never change training, source schedule, horizon, LR, or arm selection.

### 1. Scientific language quality

On fixed peS2o CONTROL and SEALED packed subsets:

- token NLL;
- perplexity;
- next-token accuracy;
- NLL by document-length quartile;
- NLL by available field-of-study metadata when sample size is sufficient.

### 2. Birth Book content internalization

Before GPU execution, freeze a `BIRTH_BOOK_PROBES.json` with CONTROL and SEALED partitions. It must contain no long verbatim passages from the book. Primary scoring is log-likelihood / multiple-choice or short cloze, not subjective free-generation grading.

Probe families:

- project-record facts and experiment results recorded in the book;
- architecture/training concepts explained in the book;
- science/ML concepts from the hand-written volumes;
- paraphrased relational questions that require combining two nearby book facts;
- matched false/distractor alternatives.

Report a single predeclared `BirthContentScore` plus each family separately. This measures **internalization of training content**, not consciousness or general intelligence.

### 3. External cognition/transfer battery

After the pretraining horizon, from each arm's final checkpoint run the same fixed low-cost downstream battery:

- the ARK-018 order-invariant temporary-binding acquisition task;
- acquisition steps to robust qualification;
- SEALED order robustness;
- matched narrowing-retention test where affordable;
- a frozen small scientific multiple-choice probe set (e.g. SciQ-style external items) scored by answer log-likelihood, with contamination checks against the peS2o shard where practical.

No downstream dataset may be selected after seeing which pretraining arm wins.

### 4. Optimization / mechanism telemetry

Record per arm:

- gradient norm;
- raw/applied update norm;
- cumulative parameter path;
- displacement from initialization;
- source identity at every step;
- Birth/small-science cursor and epoch count;
- exact scientific/Birth/replay tokens consumed.

Every 1,000 updates, on fixed CONTROL probe minibatches, compute separate science-gradient and Birth-gradient vectors (or a preregistered deterministic parameter projection if full vectors are too expensive) and report:

- gradient cosine similarity;
- norm ratio;
- whether alignment changes over training.

This helps distinguish useful rehearsal from source interference.

## Primary comparisons

### Primary causal comparison — content versus generic small-corpus repetition

`BIRTH_REHEARSAL_10PCT` vs `SCIENCE_REPLAY_10PCT_CONTROL`.

A Birth Book-specific effect requires, across both seeds:

1. `BirthContentScore` improvement >= **0.10 absolute** over the matched replay control, and
2. no > **5% relative worsening** of SEALED scientific NLL, and
3. same-direction BirthContentScore effect in both seeds.

Verdict: `BIRTH_CONTENT_INTERNALIZATION_WITHOUT_MAJOR_SCIENCE_COST`.

If BirthContentScore moves but SEALED science NLL worsens >5%: `BIRTH_CONTENT_INTERNALIZED_WITH_SCIENCE_TRADEOFF`.

If the Birth arm and token-matched science-replay control move similarly: `GENERIC_SMALL_CORPUS_REPLAY_EFFECT` rather than Birth-specific.

### Practical low-dose comparison

`BIRTH_NATURAL_2PCT` vs `SCIENCE_ONLY`.

This estimates whether the approximately size-proportional, one-exposure-per-50-steps schedule is enough to create measurable Birth Book internalization with negligible science cost. It is secondary because the control does not match small-corpus repetition as tightly as C vs D.

### Cognition effect

Any claim that Birth Book exposure changes cognition requires a preregistered downstream metric (binding acquisition/SEALED robustness or external scientific probe) to improve in the same direction across both seeds. Better BirthContentScore alone is **not** called better reasoning.

## Additional red-team requirements

Before execution the implementation audit must verify:

- exact Drive science hash and Parquet schema;
- exact Birth Book hash;
- tokenizer trained only on scientific TRAIN;
- zero Birth Book bytes in scientific CONTROL/SEALED;
- deterministic 100-step source schedule;
- exactly 2/100 Birth slots in Arm B and 10/100 in Arm C;
- matched token count of small-science replay corpus to Birth Book within 0.1%;
- exact equal optimizer updates/tokens across arms;
- exact initialization/optimizer equality at fork;
- exact resume of model, optimizer, source cursors, scientific stream position, and RNG;
- partial receipts copied to Drive periodically so Colab disconnect cannot erase the experiment.

## Required receipts

At minimum:

- `ARK-018_DATA_BINDING_RECEIPT.json`
- `ARK-018_TOKENIZER_RECEIPT.json`
- `ARK-018_MIXTURE_MANIFEST.json`
- `ARK-018_SEED_31801_RESULT.json`
- `ARK-018_SEED_31902_RESULT.json`
- `ARK-018_RESULT.json`
- `ARK-018_REDTEAM.json`
- `ARK-018_FAILURE_RECEIPT.json` only on failure
- final ZIP with per-file hashes.

## Interpretation boundary

This experiment can establish that a small, structured first-party corpus has a measurable causal effect on what a real-science-pretrained proxy internalizes, and whether that effect changes downstream capability measurements or scientific LM quality.

It cannot establish that the Birth Book creates a self, consciousness, motivation, AGI, or human-like identity. Those claims are outside the measurements here.
