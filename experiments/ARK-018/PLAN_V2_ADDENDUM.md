# ARK-018 V2 ADDENDUM — REAL-CONTINUATION RETENTION BRIDGE

## Status

**PREREGISTERED BEFORE ARK-018 IMPLEMENTATION / DATA BINDING / EXECUTION.**

This addendum supersedes only the post-pretraining intervention design in base `PLAN.md` (commit `22d299de1cb68fca335d0c3cfd65263f0cc5b8dc`). The external-data hash binding, TRAIN/CONTROL/SEALED firewall, conventional decoder architecture, exact exposure accounting, Drive durability and claim boundary remain mandatory.

## Why V2 is stronger

The base plan pretrained on real text but then tested retention mainly during synthetic canonical-only continuation. That establishes a real-data-pretrained substrate, but it does not directly answer the more important engineering question:

> Can an acquired capability survive while the model continues learning from **real text**, and can a protection mechanism preserve it without sacrificing language-model progress?

V2 makes real-text continuation the primary interference regime. The old synthetic narrowing stress remains only a positive-control calibration.

## Data binding refinements

For `.gz` input record both:
- physical file bytes + SHA256;
- decoded logical text bytes.

The ~1GB acceptance gate applies to physical file size unless the runtime launch receipt explicitly declares a different planned physical size before hashing begins.

Exact duplicate document hashes must remain in one split by the base hash-bucket rule. Report unique-document count, duplicate multiplicity distribution and duplicate byte fraction; do not deduplicate after seeing outcomes.

## Tokenizer determinism refinement

Eligible tokenizer implementation must be pinned by package name/version and single-threaded deterministic training. The data-binding smoke must rebuild the tokenizer twice on the same deterministic sample and require identical serialized tokenizer SHA256. If this cannot be achieved, fail closed.

## Frozen pretraining schedule

Use the base ~20–25M dense decoder.

Optimizer:
- AdamW betas `(0.9,0.95)`, eps `1e-8`, wd `0.1`;
- peak LR `3e-4`;
- linear warmup over first 2% of selected token budget;
- cosine decay from peak to `3e-5` over the remaining budget;
- gradient clip 1.0.

A 5-minute throughput calibration chooses the largest of:
- one full TRAIN-token pass up to 250M tokens;
- 200M tokens;
- 100M tokens;
- 50M tokens;
that fits the 240-minute pretraining allocation with a 1.25x safety factor. If a full TRAIN pass fits under 250M tokens, it has priority. Exact raw bytes and tokens consumed are recorded.

No statement may say “trained on 1GB” unless receipts show the model actually consumed the complete bound TRAIN text at least once. Otherwise wording is “pretrained from a 1GB-bound corpus with X tokens / Y bytes consumed.”

## Deterministic controlled capability

Keep the base SKILL_A temporary-binding task, but make token selection fully mechanical:

Eligible word token must:
- decode to lowercase ASCII letters only;
- length 4..10 characters;
- contain no whitespace or punctuation;
- encode by itself to exactly one tokenizer token;
- occur at least 256 times in TRAIN tokenization.

Sort eligible tokens by decreasing TRAIN frequency, then token id. Select the first 24 passing `sha256(decoded_word) mod 4 == 0`; first 12 form SKILL_A namespace, remaining 12 are reserved for SKILL_B. If fewer than 24 exist, relax only the frequency threshold in frozen order `256 -> 128 -> 64`; if still insufficient, block execution.

Fine-tuning HIGH LR = `3e-4`; LOW = `3e-6`. Do not derive HIGH from the final decayed pretraining LR.

## Stage B1 — positive-control capability narrowing

Before the real-continuation primary test, one qualified SKILL_A parent runs a short 4k-step calibration:
- synthetic canonical-only HIGH;
- synthetic LOW;
- full order-augmented HIGH.

This stage only checks whether the larger real-pretrained subject reproduces the known controlled narrowing contrast. It does not count toward the real-continuation primary endpoint.

If HIGH does not erode and all arms remain stable, continue to the real-text primary test anyway; mark `SYNTHETIC_POSITIVE_CONTROL_NO_EVENT`.

## Stage B2 — primary: continued real-text learning after capability acquisition

From each qualified SKILL_A parent, all arms start from the exact same model/optimizer/RNG checkpoint and consume the **same ordered real TRAIN token stream** for an equal token budget.

Primary continuation budget: 25M real-text tokens, extendable to 50M only by the frozen rule below.

Arms:

1. `REAL_CONTINUE_HIGH`
   - HIGH LR `3e-4`;
   - 100% real-text continuation;
   - no SKILL_A replay.

2. `REAL_CONTINUE_LOW`
   - LOW LR `3e-6`;
   - same real-text tokens;
   - slow-learning / near-freezing reference.

3. `REAL_CONTINUE_REPLAY`
   - HIGH LR;
   - same total token budget;
   - 1/64 packed training sequences are replaced by SKILL_A order-augmented sequences;
   - replacement positions deterministic; real-text token loss is counted explicitly.

4. `REAL_CONTINUE_STATIC_CANDIDATE`
   - included only if ARK-017 has a prospectively qualifying mechanism before ARK-018 starts;
   - frozen mapping: movement -> selected cap dose; replay -> replay arm already serves as candidate; joint -> cap + replay; unresolved -> arm omitted.

5. `REAL_CONTINUE_HIGH_REPLAY_CONTROL`
   - HIGH LR;
   - 1/64 sequences replaced by a matched-size **new, disjoint SKILL_B-like distractor binding stream** that does not rehearse SKILL_A;
   - distinguishes generic structured-data regularization from capability-specific support.

All replacement streams preserve total sequence count and context-token budget as closely as packing permits; exact real/synthetic token counts are reported.

## Frozen extension rule

At 25M continuation tokens inspect **CONTROL SKILL_A only**:
- if `REAL_CONTINUE_HIGH` has not shown either robust-qualification failure or >=0.10 robustness-area deficit versus the best protection reference, and throughput budget permits, extend **all already-started arms symmetrically** to 50M tokens;
- SEALED never decides extension.

If there is still no contrast: `REAL_CONTINUATION_LOW_INTERFERENCE`.

## Primary endpoints

Primary scientific question is no longer synthetic narrowing. It is:

- SKILL_A SEALED robust retention during real-text continuation;
- real-text SEALED NLL improvement/degradation over the same continuation tokens;
- capability–language Pareto efficiency;
- movement/path telemetry;
- exact amount of SKILL_A replay and displaced real-text exposure.

`REAL_SUBSTRATE_RETENTION_TRANSFER` requires >=2 independent SKILL_A parents and >=4 matched sets, plus either:
- HIGH has >=2 SEALED failures and a protection arm reduces risk by >=0.40; or
- HIGH mean SEALED robustness area is >=0.15 below a protection arm across matched sets,
while the winning arm's SEALED real-text NLL is not >5% worse than HIGH.

## Interpretation boundary

A positive V2 result means a controlled acquired capability can be protected during **continued real-data learning** on a 1GB-bound real-text substrate. It still does not prove general reasoning preservation or production-scale benefit.