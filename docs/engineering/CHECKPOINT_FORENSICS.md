# Checkpoint Forensics

Updated: 2026-07-23  
Purpose: determine what a checkpoint really contains, whether it can resume,
why it behaves as it does, and what claims its evidence supports.

## The historical lesson

The earlier `anra_frontier_500m.pt` artifact was approximately 2.00 GB and
reported low loss, yet diagnostic generation was incoherent. Prior inspection
recorded an exact-loadable tensor set but also zero coherent acceptance and
systematic EOS failure. That combination demonstrated that:

- low loss did not prove language ability;
- architecture and training lineage mattered;
- output collapse could hide behind aggregate metrics;
- residual instability and dormant routing needed direct measurement;
- model-only size and full-resume size were being confused.

Those historical facts remain evidence about that artifact, not a description
of the new V4 model.

## First question: what artifact class is this?

### `full_resume`

May continue training only when it contains and validates:

- schema version 9;
- model, optimizer, scheduler, scaler, and RNG states;
- completed optimizer boundary;
- zero unsafe partial accumulation;
- sampler cursor and signed token window;
- global step and tokens processed;
- architecture, tokenizer, data, recipe, commit, and seed lineage.

### `fp16_inference`

Contains model tensors and enough identity metadata for inference. It omits
optimizer and continuation state. It is normally much smaller and must never
resume training.

### `growth_initialization`

Contains a mapped child model plus growth provenance. It explicitly requires a
fresh optimizer and is not presented as an exact parent optimizer resume.

If the artifact class is missing or inconsistent, stop. Do not infer
resumability from the filename.

## Safe inspection order

1. Copy the artifact or work from an immutable source.
2. Record absolute path, byte size, modification time, and SHA-256.
3. Load metadata on CPU with safe/weights-only loading where supported.
4. Determine artifact class and schema.
5. Compare model configuration with the registered profile.
6. Account for every tensor: expected, missing, unexpected, and mismatched.
7. Verify tokenizer artifact and metadata hash.
8. Verify data and training recipe lineage.
9. Inspect numerical health.
10. Run deterministic generation only after compatibility passes.

Never modify the only copy during inspection.

## Structural checks

For `anra-v4-180m`, verify:

- 181,132,071 exact parameters;
- vocabulary 32,768;
- 18 layers, width 896;
- 14 query heads, 2 KV heads, head dimension 64;
- FFN 2,432;
- tied embeddings;
- declared hybrid attention modes.

For `anra-v4-500m-growth`, verify:

- 499,880,031 exact parameters;
- 27 layers, width 1,280;
- 20 query heads, 2 KV heads;
- FFN 3,456;
- parent and growth manifest hashes;
- real logits-parity result;
- optimizer restart declaration.

A tensor count match alone does not prove that tensors correspond to the
correct semantics.

## Numerical pathology checks

Inspect by layer and component:

- parameter norms and non-finite values;
- activation means, variances, maxima, and residual growth;
- attention logits and entropy;
- gradient norms and clipping frequency;
- embedding/output-logit scale;
- router entropy, expert balance, and actual route usage;
- repeated-token and EOS probabilities;
- output vocabulary utilization.

Warning patterns:

| Pattern | Possible meaning |
| --- | --- |
| Low loss, incoherent output | favorable source mix, leakage, local-statistic learning, or collapse |
| Nearly one token dominates | collapsed output distribution |
| Residual norm grows layer by layer | initialization or normalization instability |
| Router always chooses one route | dormant or collapsed routing |
| Training resumes with different loss spike | optimizer/RNG/sampler discontinuity |
| Good aggregate validation, poor source slice | source imbalance hidden by averaging |
| Tiny checkpoint for a large AdamW run | probably model-only, not full resume |

## Behavioral checks

Use deterministic prompts first:

- short sentence completion;
- instruction formatting;
- two-turn context retention;
- copying versus transformation;
- factual uncertainty;
- arithmetic and short reasoning;
- code syntax;
- retrieval-grounded answer;
- repetition and EOS behavior.

Record prompt, tokenizer IDs, generation settings, seed, cache mode, model hash,
output, and evaluator result. Compare cached and uncached generation where
appropriate.

Do not ask only open-ended questions. A broken model can occasionally produce
a plausible phrase, while a useful small model may fail knowledge-heavy trivia.

## Resume forensics

For an exact-resume drill:

1. Start from a completed optimizer boundary.
2. Record step, tokens, token-window ID, sampler cursor, RNG fingerprints, and
   optimizer checksum.
3. Perform a bounded next update and save.
4. Restore the earlier full resume into a new destination.
5. Repeat the same update.
6. Compare boundary, data IDs, losses, and state continuity.

For cloud handoff, repeat across two authorized workers. The second worker must
take a new signed lease and the exact next token position. The old worker must
be unable to commit after lease expiry.

## Durability forensics

A protected checkpoint should have:

- immutable local snapshot manifest;
- 128 MiB content-addressed chunk records;
- verified chunk sizes and SHA-256;
- canonical remote receipt;
- protected-state receipt count;
- typed canonical pointer;
- retained previous full-resume generation.

An upload existing in Drive is insufficient if the manifest or hash receipt is
missing.

## What to preserve in a forensic report

- artifact path, class, size, and SHA-256;
- inspection code commit and dirty state;
- architecture/tokenizer/data/recipe identities;
- tensor-accounting table;
- numerical-health summaries;
- deterministic behavioral outputs;
- resume-continuity results;
- conclusions separated into fact, inference, and unknown;
- safe next action.

## Live truth sources

- Artifact validation: `training/v2_runtime.py`
- Resume lineage: `training/checkpoint_durability.py`
- Architecture profiles: `training/v2_config.py`
- Growth validation: `training/csii.py`, `training/growth_runtime.py`
- Evaluation: `training/eval_v2.py`
- Historical evidence: `docs/engineering/ENGINEERING_LOG.md`
- Current checkpoint events: `runtime/evidence_stream.py`

The canonical pointer and its referenced immutable manifest are the live truth
for “latest protected checkpoint.” A local filename is not.
