# An-Ra Improvement Map

This is the evidence-driven path from the current research checkpoint to a model
that can be promoted with credible claims. It records problems, their likely
causes, the implemented controls, and the next measurable action.

## First Principle

An-Ra is improved in place. No external pretrained weights replace its model,
tokenizer, architecture, mathematics, data lineage, or identity. Compatible
tensors and token IDs are preserved; incompatible growth requires a named,
versioned migration.

## What Has Been Repaired in Code

| Area | Earlier failure mode | Current control |
| --- | --- | --- |
| Checkpoint load | Partial load could look successful | Exact tensor accounting and blocked core-weight gaps |
| Tokenizer | Wrong IDs could silently decode a checkpoint | Hashes, special-token map, and 500 probes |
| Prompt path | 512-token/character-based truncation | Tokenizer-aware 1,024-token assembly and trace |
| Memory | Duplicate or oversized injection | Insert once, score, budget, and truncate explicitly |
| Repetition | Penalty affected prompt and signed logits incorrectly | Answer-only, sign-correct penalty |
| KV cache | Incremental RoPE positions could be wrong | Absolute offsets and parity gate |
| Session state | HAL/ESV/ghost state could leak globally | Request/session scope and accepted-output commit |
| MoD | Sequence-wide softmax diluted updates | Per-token sigmoid, straight-through top-k routing |
| ESV/RIM | Batch averaging could leak identity state | Per-sample channels and bounded projection |
| DSTP | Fixed temperatures could not learn | Bounded trainable temperatures with regularization |
| HAL | Confidence could reward bad output | Verifier/coherence/task/CIV evidence after generation |
| Data objective | Foundation text could become fake dialogue | Separate raw-causal and conversation packers |
| Corpus | Small repeated mixture hid token reuse | Immutable shards and unique-token accounting |
| Evaluation | Keyword checks overstated quality | Private tasks, execution/parsers, seeds, ablations |
| Release | UI/loss could be mistaken for readiness | Signed evidence bundle and rollback drill |

These controls improve correctness and observability. They do not by themselves
prove that the existing weights became more capable.

## Current Blocking Problems

### 1. Real checkpoint capability is not yet established

The current checkpoint has shown malformed language in interactive screenshots.
Possible causes include undertraining, data/objective mismatch, tokenizer lineage
error, malformed checkpoint restoration, or harmful subsystem interaction.

**Required evidence:** exact checkpoint/tokenizer proof, diagnostic greedy outputs,
and the 200-prompt recovery gate.

**Decision:** if exact loading passes but native coherence is below 80%, classify
the primary problem as undertraining and proceed to continuation training.

### 2. Low training loss is not enough

Loss can be low on repeated, narrow, incorrectly packed, or prompt-dominated data.
It does not prove coherent chat, instruction following, memory, reasoning, or tool
use.

**Required evidence:** held-out source-hash splits, unique tokens consumed,
per-source validation loss, coherence, format compliance, executable code/math,
and human review.

### 3. The native systems need causal evidence

Execution traces prove that a subsystem ran, not that it helped. A subsystem may
add latency while reducing capability.

**Required evidence:** same-checkpoint, three-seed full-system ablations for MoD,
RIM, DSTP, ESV, and HAL. Every contribution must be positive with bounded latency
and less than 2% protected validation regression.

### 4. A 499M model has a finite capacity budget

Architecture can make training and inference better behaved, but it cannot create
knowledge absent from data or optimization. The model needs enough diverse,
quality-controlled tokens and instruction examples for its size.

**Required evidence:** campaign token counts, scaling curves, source-stratified
validation, and checkpoint comparisons at fixed evaluation settings.

### 5. Colab is an experimental environment

T4 sessions are preemptible, slow for exhaustive evaluation, and easy to corrupt
with concurrent checkpoint writers.

**Control:** immutable job manifests, unique artifact paths, optimizer-boundary
checkpoints, Drive hash verification, and a single promoted continuation lineage.

## Gated Recovery Sequence

### Gate 0: freeze the baseline

- Record checkpoint SHA-256, tokenizer SHA-256, config, corpus manifest, source
  commit, and current malformed outputs.
- Never overwrite this evidence.

### Gate 1: prove artifact identity

- Require 100% core tensor accounting.
- Require tokenizer ID compatibility and all 500 probes.
- Reject unknown migrations or incomplete corpus metadata.

### Gate 2: prove deterministic inference

- Greedy strategy, seed `0`, 128 output tokens, cache off.
- Require finite logits/probabilities, explicit stop reasons, and deterministic
  output-token replay.
- Compare diagnostic and native modes on exactly 200 prompts.

### Gate 3: choose repair or continuation

- Coherence at or above 80%: isolate remaining prompt/generation/subsystem defects.
- Coherence below 80% with exact loading: begin the continuation curriculum.
- Artifact proof failure: repair lineage first; training an unknown load compounds
  the problem.

### Gate 4: continue training by objective

| Phase | Target | Promotion signal |
| --- | ---: | --- |
| A | 1B raw foundation tokens | perplexity below 12, numerical and tokenizer stability |
| B | 1B raw tokens with staged native unfreezing | complete traces and no protected regression |
| C | 200M code/math/science/verified-DFC tokens | integration regression at most 2% |
| D | 100M conversation/instruction tokens | coherence at least 90%, format at least 85% |
| E | 10M verifier replay/tool tokens | reasoning at least 70%, verification at least 90% |

Training checkpoints occur on optimizer boundaries. Evaluate every 250 optimizer
steps and checkpoint every 500. Select by held-out capability and validation, not
training EMA loss.

### Gate 5: prove full-system value

- Run the integration probe.
- Run 500+ private tasks across three modes and three seeds.
- Run each native subsystem ablation across all seeds.
- Complete blinded review for open-ended coherence.

### Gate 6: promote or reject

Promotion requires:

- coherent response rate at least 90%;
- instruction format compliance at least 85%;
- repetition/EOS failure below 1% over 1,000+ generations;
- positive native subsystem contributions;
- zero session-state leakage and cache parity;
- verified corpus/config/checkpoint/tokenizer identities;
- signed release bundle and successful rollback drill.

Any failed gate blocks the candidate and produces the next experiment target.

## Data Improvement Program

The default 30 GB campaign targets this token mix:

| Source class | Share |
| --- | ---: |
| FineWeb-Edu foundation | 55% |
| Permissively licensed code | 15% |
| FineMath-4+ | 12% |
| Science and technical text | 8% |
| Verified general instructions | 5% |
| Verified DFC | 3% |
| An-Ra identity and replay | 2% |

Every source must pass license allowlisting, language detection, repetition and
boilerplate filtering, PII removal, exact and MinHash deduplication, source-hash
splitting, and domain-specific checks. Code should parse where applicable; math
and DFC confidence must come from verifiers, not labels invented by the model.

Monitor these data metrics per session:

- unique tokens consumed;
- repeated-token percentage;
- realized source mix;
- train/validation source overlap;
- source-stratified validation loss;
- answer-token share during instruction training;
- rejected source records and reasons.

## Experiment Design

Change one causal factor per candidate. Every job manifest should contain:

```text
base checkpoint hash
tokenizer hash
corpus manifest hashes
source commit
model and training config
continuation phase
seed
maximum tokens / optimizer steps
output artifact path
expected evaluation suite
```

For multiple Colabs, use them as an experiment farm: shard validation, tokenizer
fertility, subsystem ablation, continuation candidate, evaluation, and
reproducibility jobs. Never average unrelated optimizer states and never let two
workers write the same checkpoint.

## How to Claim Improvement

Do not claim “50% better” from code volume, loss, or subjective samples. Define a
fixed metric bundle before the run:

```text
capability_delta = candidate_private_score - baseline_private_score
coherence_delta = candidate_coherence - baseline_coherence
format_delta = candidate_format - baseline_format
failure_delta = baseline_generation_failures - candidate_generation_failures
latency_delta = candidate_latency - baseline_latency
```

Report absolute values, relative deltas, confidence across seeds, protected
regressions, and the exact artifact hashes. A large gain on a broken baseline is
useful recovery evidence, but it is not evidence of world-leading intelligence.

## Near-Term Definition of Done

The next integrated release is done only when:

1. Cell 10 restores the intended checkpoint and opens the UI reproducibly.
2. Checkpoint and tokenizer reports are exact.
3. The 200-prompt gate passes.
4. Full-system integration passes without evaluation-state persistence.
5. The private promotion suite and human review pass.
6. Every native subsystem has positive three-seed evidence.
7. The signed release bundle re-verifies and rollback succeeds.

Until then, the repository is a serious native AGI research platform under
recovery and continuation, not a finished AGI claim.
