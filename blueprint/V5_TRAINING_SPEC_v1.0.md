# An-Ra V5-A Core and Training Specification v1.0

Status: **implementation-frozen candidate; main run not authorized**

Executable receipt: `artifacts/v5/training_spec_v1.json`
Authoritative constants: `v5_contracts.training_spec`

This is the single code-facing specification for V5-A. It removes undefined
defaults and contradictory alternatives. “Frozen” means implementation must
match this receipt or create a new version. It does not mean every choice is
scientifically proven. The major 250M run remains blocked by the identities and
experiments in section 12.

## 1. Design decision

Build a conventional dense decoder-only Transformer. Do not add recurrence,
MoE, SSM blocks, latent-thought heads, routers, tools, memory, or a separate
cognition head. Put cognition pressure into mechanically verified training
data, nuisance-resistant evaluation, and checkpoint selection. Keep tools,
search, durable memory, permissions, intervention routing, and external
verification in Connector/runtime.

Classification: **STRONG INFERENCE**, because it maximizes causal attribution
and minimizes the probability that a speculative module consumes the run.

## 2. Exact Core

| Constant | Value | Classification |
|---|---:|---|
| family | dense causal decoder Transformer | EVIDENCE-BACKED baseline |
| parameters | 250,216,960 | arithmetic EVIDENCE-BACKED; scale EXPERIMENT REQUIRED |
| layers / width | 26 / 896 | PROVISIONAL FREEZE; E2/M102 REQUIRED |
| query / KV heads | 14 / 7 | STRONG INFERENCE; 2:1 scale invariant |
| head dimension | 64 | STRONG INFERENCE |
| FFN | 2,368, SwiGLU `SiLU(gate) * up` | STRONG INFERENCE |
| attention | full causal in all layers | STRONG INFERENCE; target kernel gated |
| context | 4,096 native | EXPERIMENT REQUIRED |
| normalization | pre-RMSNorm, final RMSNorm, epsilon `1e-5` | EVIDENCE-BACKED family |
| QK norm | affine, per head over 64 dimensions, epsilon `1e-6` | LOCAL EVIDENCE; learning/TPU REQUIRED |
| position | full-head pairwise RoPE, base 10,000, positions 0–4095 | LOCAL CONFORMANCE |
| RoPE table | float32 phase table; no scaling or extrapolation | EVIDENCE-BACKED implementation boundary |
| embeddings/output | tied storage identity; scale 1; no logit cap/bias | STRONG INFERENCE |
| bias / dropout | no linear bias; dropout 0 | STRONG INFERENCE |

Initialization is exact: embedding, Q/K/V, gate, and up tensors use
`Normal(0, 0.02)`. Attention-output and FFN-down use
`Normal(0, 0.02/sqrt(52))`, whose standard deviation is
`0.002773500981126146`. RMSNorm and affine QK scales start at 1. There are no
bias tensors. **EVIDENCE-BACKED locally; target/learned evidence required.**

Precision is one FP32 persistent parameter set with BF16 autocast compute,
FP32 logits/loss/global-gradient-norm reductions, and FP32 Adam moments. There
is no persistent BF16 shadow and no loss scaler. This makes the tensor-only
full-resume planning floor `12 bytes × parameters`, before metadata. **LOCAL
EVIDENCE; target long-run test required.**

The consistent scale-transfer family is:

| Stage | Layers×width | Q/KV | FFN | Parameters |
|---|---:|---:|---:|---:|
| P35 recipe | 16×384 | 6/3 | 1,024 | 35,414,400 |
| M102 recipe | 20×640 | 10/5 | 1,600 | 101,790,080 |
| V5-A | 26×896 | 14/7 | 2,368 | 250,216,960 |

Earlier P35 shape/MHA and 3:1 GQA canaries remain named mechanism evidence;
they are not the P35 recipe and cannot justify V5's 2:1 topology.

## 3. Tokenizer

Use byte-level BPE with byte fallback, exactly 24,576 entries, no Unicode
normalization, case folding, whitespace rewrite, prefix-space insertion, or
tokenizer dropout. Reserved IDs are PAD 0, UNK 1, BOS 2, EOS 3. Unexpected UNK
count must be zero. No task, family, difficulty, answer-index, or hidden-state
tokens are permitted.

Classification: **PROVISIONAL FREEZE / E1 REQUIRED**. The current local 24k
artifact supports this center but was trained on an inadequate local corpus;
its merge artifact is deliberately not frozen. The E1 winner's artifact and
training-corpus SHA-256 must fill the null identities in the executable receipt
before packs or weights are produced.

## 4. Data and packing

The budget is exactly 5,000,000,000 real non-padding input tokens:

| Slice | Fraction | Tokens |
|---|---:|---:|
| high-quality natural | 65% | 3,250,000,000 |
| code, mathematics, formal/structured natural | 20% | 1,000,000,000 |
| mechanically verified cognition | 15% | 750,000,000 |

This is a **PROVISIONAL FREEZE / E3 REQUIRED**. Source origin, license class,
acquisition, filtering, exact/near deduplication cluster, contamination result,
split, tokenizer, and token count are mandatory. Splits are assigned at the
source-disjoint deduplication-cluster level. LLM paraphrase target is zero and
the hard ceiling is 5% of total tokens; every such token remains provenance
tagged.

Packing uses BOS-content-EOS segments, block-diagonal causal attention between
segments, and position IDs reset per segment. BOS and PAD are excluded from
loss; content and EOS are included. Documents longer than 4,096 are split into
deterministic non-overlapping chunks with their own boundaries. Padding never
enters the token ledger.

The exact token bucket mix is 25% at 512, 25% at 1,024, 30% at 2,048, and 20%
at 4,096. The deterministic 20-microstep supercycle is stored in the executable
receipt and checkpoint cursor; it contains bucket counts `5/5/6/4`.

## 5. Native cognition curriculum

Every family uses direct examples plus mechanically checked sensitivity and
invariance groups. Natural/semi-natural surface is crossed within every family,
not treated as a separate family. At least 25% of each family uses natural or
semi-natural surface/provenance. No family disappears during training.

| Cognition family | Share of cognition | Tokens at 15% |
|---|---:|---:|
| identity and exact copy | 8% | 60,000,000 |
| query-conditioned binding | 16% | 120,000,000 |
| semantic-time state/precedence | 16% | 120,000,000 |
| interference-resistant retrieval | 10% | 75,000,000 |
| relational composition | 20% | 150,000,000 |
| counterfactual sensitivity | 10% | 75,000,000 |
| held-out-structure rule induction | 10% | 75,000,000 |
| missing-information recognition | 5% | 37,500,000 |
| faithful realization/format | 5% | 37,500,000 |

Difficulty is uniformly interleaved across the run with token shares 34% easy,
35.5% medium, and 30.5% hard. A staged curriculum is an E4 challenger, not a
launch default; if compared, it must reorder the identical example multiset so
total family and difficulty exposure cannot confound the result.

Frozen grids are binding cardinality `2/4/8/16`; distractors
`0/2/4/8/16/32`; state variables `1/2/4` crossed with updates `2/4/8` and
balanced latest/intermediate/rollback/precedence queries; composition hops
`1/2/3` with matched direct-retrieval controls; rule demonstrations `2/4/8`
with held-out structures; and all four context-position quartiles. Answer
length, suffix-token count, first token, format, and candidate position must be
crossed rather than correlated with truth.

Classification: family definitions are **EVIDENCE-BACKED measurement needs**;
exact weights are **STRONG INFERENCE / E3 REQUIRED**.

## 6. Objective

The launch objective is causal cross-entropy only. It is the replica-global
mean over eligible target tokens; label smoothing and z-loss are zero.
Query-swap lambda and trace-loss lambda are both exactly zero.

This is deliberate. Sum, token mean, byte mean, DC-PMI, and contextual
calibration all failed variable-length candidate nuisance tests; the powered
DC-PMI/contextual run selected the fewest-token candidate 100% across all three
tokenizers. An unproven auxiliary must not enter the expensive run.

The only admissible E3 query-swap challenger freezes `lambda={0,.05,.15}`,
margin 0, one gold plus three plausible negatives, two fact-fixed
counterfactual queries, identical suffix-token counts under the frozen
tokenizer, all candidate rotations, summed suffix log-probability, and measured
FLOP matching. It requires fresh candidate-free Core and natural-transfer
gains before a new training-spec version may set lambda above zero.

## 7. Optimization and TPU topology

| Constant | Value |
|---|---:|
| optimizer | AdamW |
| beta1 / beta2 / epsilon | 0.9 / 0.95 / `1e-8` |
| weight decay | 0.1 on all `ndim>=2` tensors, including tied embedding |
| no-decay | every `ndim<2` tensor, including RMSNorm and QK scales |
| global gradient clip | replica-global L2 norm 1.0 |
| global real tokens/update | 131,072 |
| full/final updates | 38,146 full + one 127,488-token partial = 38,147 |
| replicas | 8 data-parallel TPU/XLA replicas |
| tokens/replica/microstep | 4,096 |
| global tokens/microstep | 32,768 |
| accumulation | 4 microsteps; synchronize at boundary only |
| sequences/replica | 8×512, 4×1024, 2×2048, or 1×4096 |
| activation checkpointing | every Transformer block |

The final partial update uses global microsteps
`32768+32768+32768+29184`; the last has 3,648 real tokens and 448 ignored PAD
positions per replica in a 2,048 bucket. This plan and supercycle index are
checkpointed.

The token-indexed WSD schedule is exact:

- `[0, 50M)`: linear `0 → 3e-4`;
- `[50M, 4.5B)`: constant `3e-4`;
- `[4.5B, 5B]`: linear `3e-4 → 3e-5`.

LR is evaluated from pre-update cumulative real tokens. Resume, notebook,
worker, or pack changes never rewarm it. A nonfinite loss/gradient aborts the
update and run; parameter, optimizer, schedule, cursor, and token counters do
not advance.

Classification: optimizer family and local wiring are **EVIDENCE-BACKED**;
exact LR/batch/schedule/topology are **EXPERIMENT REQUIRED**.

## 8. Checkpoint and durability

Recovery thresholds are every 10M real tokens with exactly two rotating durable
generations. Immutable milestones are every 100M, every 50M from 4.5B through
5B, and at any future curriculum boundary. A threshold fires at the first
completed optimizer update reaching or crossing it; both scheduled and actual
token counts are recorded. There are no mid-update snapshots.

Every full-resume checkpoint contains FP32 parameters, FP32 Adam moments and
step, schedule, exact token/source/family ledger, sampler and supercycle cursor,
all rank RNG states, topology, and every model/tokenizer/data/pack/source/code
identity. A pointer advances only after immutable upload, clean redownload hash
equality, and clean restore. Milestones never mutate.

## 9. Evaluation and promotion

Tier 0 runs every 25M tokens with 32 cases/family. Tier 1 runs every 100M with
512/family. Development chooses exactly one checkpoint. The sealed suite has
1,024/family and is consumed once for that checkpoint; a newly generated/source-
disjoint Tier 3 replication must confirm it.

Native selection is candidate-free generation. Candidate-set ranking is always
labeled assisted/diagnostic. Decoding is greedy, temperature 0, top-p 1, no
top-k, maximum 64 new tokens, stopping at EOS or the cap. Representation,
selection, addressing, and realization remain separate. Conditional
realization is conditioned on correct unassisted selection.

Promotion is conjunctive and worst-family: fresh selection Wilson LCB at least
chance+0.10; sensitivity flip LCB at least 0.80; invariance stable-and-correct
LCB at least 0.90; state OOD at least 0.70; two-hop at least 0.60 and above
matched retrieval; three-hop LCB chance+0.10 and no more than 20 points below
two-hop; missing-information balanced accuracy at least 0.80 with false
assertion at most 0.10; conditional realization at least 0.80; natural loss
regression at most 3%; code/math at most 5%; no family regression above 5%; and
M102 replication over two seeds plus fresh natural transfer paired LCB above
zero.

These thresholds are **STRONG INFERENCE / EXPERIMENT REQUIRED**. Production
candidate-scoring mode remains null after the scorer failure.

## 10. Abort rules

Abort immediately without advancing state on any nonfinite value; token,
cursor, optimizer, schedule, or parameter mismatch; hash/custody mismatch;
resume inequivalence; benchmark/candidate leakage; or train/evaluation source
collision. Pause and preserve when two consecutive Tier 1 checks show more
than five points of worst-family decline while LM loss improves. Deny V5-A if
fresh candidate-free/natural gains are absent, M102 does not reproduce, or TPU
and remote restore canaries fail.

## 11. Change control

Code may not invent a missing value. Any change to a constant above requires a
new schema/version, regenerated receipt, decision-log entry, tests, and a stated
evidence class. Filling a required SHA-256 after its owning gate is not a silent
default; it creates the final signed launch manifest.

## 12. Remaining launch blockers

The implementation candidate is complete, but `main_training_authorized=false`
until all of these close:

1. E1 freezes the real tokenizer artifact and corpus identity.
2. E2 tests the consistent 2:1 P35 architecture and context on target-relevant hardware.
3. E3 confirms the 15% CE cognition mixture on fresh candidate-free and natural transfer.
4. E4 confirms LR/batch/schedule or emits a new spec version.
5. M102 reproduces the effect across two seeds and fresh data.
6. TPU/XLA model/optimizer/collective/throughput/resume canaries pass.
7. Source/data/pack manifests and remote clean-redownload restore are signed.
8. The power-sized family/Wilson evaluator and sealed-custody path pass; conditional realization already uses correct unassisted-selection eligibility.

This is the strongest honest freeze today: exact enough to implement without
guessing, and strict enough not to turn provisional constants into false
scientific claims.
