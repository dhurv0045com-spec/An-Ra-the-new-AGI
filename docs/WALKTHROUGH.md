# An-Ra: From Raw Text to a Trusted Answer

Updated: 2026-07-23  
Purpose: tell the repository as one connected story so a reader can picture
where every major system enters and why it exists.

## Scene 1: before there is a model

At the beginning there are no intelligent weights. There are text sources,
licenses, revisions, and the question of whether the material is suitable for
training.

The acquisition pipeline downloads only declared sources. It records where
each source came from, under what license, at which immutable revision, and
with which hash. Cleaning removes unusable records. Deduplication reduces
repeated learning. Contamination checks prevent evaluation material from
quietly entering training. Splitting happens by document identity so fragments
of one document do not appear on both sides of validation.

The result is not simply “30 GB of text.” It is a reproducible corpus lineage:
bytes plus the evidence needed to rebuild and audit them.

## Scene 2: text becomes the model’s alphabet

The V4 tokenizer turns text into integer token IDs. It has 32,768 entries and
is the only tokenizer allowed in a new operational run. Its hash is bound into
every data pack and launch.

This is important because model embeddings are indexed by token ID. Changing
the tokenizer after training would change the meaning of those rows. V4
therefore remains fixed through the 181M foundation and its 500M child.

The tokenizer produces sequences up to the model’s 2,048-token context. Data
builders arrange those tokens into deterministic windows. A 170M-token pack is
one session-sized window of a much larger corpus, not the model’s total
education.

## Scene 3: the dense learner

The first learner is `anra-v4-180m`: 181,132,071 trainable parameters arranged
as 18 transformer layers of width 896.

Each input token selects an embedding. Attention lets token representations
read relevant earlier positions. Grouped-query attention uses 14 query heads
with 2 key/value heads to reduce cache and compute. QK normalization controls
attention scale. RoPE supplies position. Feed-forward layers transform each
token representation. Residual paths allow information and gradients to move
through depth. Tied embeddings reuse one learned matrix for input and output.

At the output, the model predicts a probability distribution over the next
V4 token. Training compares that distribution with the actual next token and
updates weights through AdamW.

Seed 1301 fixes random initialization and data-order randomness for
reproducibility. It is called a seed because it initializes the random number
generators from which the rest of the run grows.

## Scene 4: training is a lineage, not one command

Before a worker starts, the owner signs a launch manifest. It says:

- which exact Git commit must run;
- which architecture and tokenizer must load;
- which data manifests and token interval are assigned;
- whether the run starts from scratch, a full resume, or a growth artifact;
- where artifacts must be written;
- how often checkpoints must become durable;
- how much time and storage the worker may use.

Only one canonical trainer may change the model. It consumes its assigned
window and advances optimizer steps. Other workers can prepare the next pack,
evaluate an immutable checkpoint, test one architecture pilot, or archive
artifacts.

Every 200 optimizer steps or 60 minutes, whichever comes first, the trainer
saves a completed-boundary checkpoint. The durability pipeline breaks it into
hash-addressed chunks and uploads while training continues. It publishes a
canonical pointer only when the checkpoint is verified remotely.

If the machine disappears, the next authorized trainer downloads the latest
verified full resume. It restores weights, AdamW moments, scheduler, scaler,
random states, sampler cursor, optimizer step, and global token position. It
then takes a new fenced lease. A stale worker cannot later overwrite this
lineage.

## Scene 5: why low loss once lied

The historical checkpoint showed that low training loss can coexist with
broken language. A model can become good at local token statistics or an
over-represented source while its output distribution collapses. Residual
scale can become unstable. Routing modules can exist without contributing.
The data and architecture that produced the checkpoint may no longer be
reconstructable.

That is why the current system treats loss as telemetry, not a verdict. At
milestones it checks:

- source-stratified validation;
- coherent and diverse generation;
- EOS and repetition behavior;
- copying and memorization;
- uncertainty and abstention;
- short reasoning, math, code, retrieval, and context use;
- activation, gradient, and routing health.

The checkpoint becomes trusted through evidence, not through its filename.

## Scene 6: a pretrained model is not yet an assistant

Pretraining teaches continuation. It does not automatically teach reliable
dialogue, instructions, correction, or tool contracts.

After the dense language foundation is coherent, supervised fine-tuning uses
an audited mixture of instructions, dialogue, code, mathematics,
decomposition, uncertainty, and correction. That creates a separate signed
lineage.

Verifiable domains can then use RLVR or STaR-style outcome supervision. DPO is
allowed only when preference pairs have provenance and audit records. These
stages are currently contracted and gated; they have not yet produced an
accepted V4 post-trained checkpoint.

## Scene 7: external intelligence surrounds the weights

When a person sends a message, the API identifies the session and constructs
context. Retrieval can search attributable external knowledge. Memory can add
selected past facts. The V4 tokenizer converts the assembled context to IDs,
and the transformer generates candidate tokens.

The self-correction loop may then:

1. understand the request;
2. retrieve relevant evidence;
3. make a bounded plan;
4. generate a candidate;
5. call an appropriate verifier;
6. revise or abstain;
7. persist the outcome and evidence.

This is not a second model. It is an orchestration layer around the learned
model. External retrieval and reversible adapters let knowledge or skills
change without retraining every base parameter.

Tools and agents enter later. They require typed inputs, explicit permissions,
execution limits, result verification, provenance, and rollback. The ability
to call a tool is not proof that the model knows when it should call it.

## Scene 8: experimental systems wait outside the critical path

MTP, MoD, RIM, ESV, DSTP, HAL, MoE, hybrid attention, SSMs, latent reasoning,
multimodality, robotics, and world models all represent possible improvements.
They do not all run in the first baseline.

Each pilot starts from a frozen parent and changes one important variable.
MTP asks whether predicting multiple future tokens improves representations.
MoD asks whether compute can be allocated adaptively. MoE asks whether sparse
capacity can increase without unaffordable memory. HAL asks whether bounded
state policy improves behavior. Moonshots provide a laboratory for ideas whose
value is not yet known.

If a pilot wins on capability, stability, and useful-compute efficiency, it
may be promoted. If not, it remains off or is retired. This protects the
meaning of every result.

## Scene 9: the model grows

After the 181M model is useful, it can become the parent of the
499,880,031-parameter child. The growth process expands width and depth while
mapping parent tensors, preserving attention modes, and inserting residual
blocks that initially behave like identities.

Before training the child, the system compares parent and child logits.
Parent checkpoint hash, architecture mappings, and parity results enter a
growth manifest. The optimizer restarts because its old tensors do not match
the new geometry. A low-learning-rate alignment period and progressive
unfreezing reduce behavioral damage.

Only then does the larger model consume new tokens.

## Scene 10: an answer becomes evidence

At runtime, the response is stored with its trace, retrieval sources,
verification decisions, model and adapter identities, and relevant health
signals.

ThirdEye lets an engineer investigate one run. Matrix shows aggregated
operational state. Promotion uses evaluation and rollback evidence. A release
pointer changes only after gates pass. If a regression appears, the system
returns to a previously verified artifact.

The full loop is now visible:

```text
licensed source
  → immutable corpus
  → V4 tokens
  → dense training
  → durable checkpoint
  → behavioral evaluation
  → post-training
  → retrieval/memory/correction
  → response and trace
  → evidence
  → promotion, revision, or rollback
```

## Where the journey actually stands

The contracts, model geometry, exact resume, checkpoint durability, cluster
baton, growth mechanism, subsystem gates, and post-training evidence
interfaces are implemented. A new useful V4 language checkpoint, live
cross-worker handoff, promoted optional subsystem, accepted post-trained
checkpoint, and trained 500M child do not yet exist.

The next real scene is a 10–15 minute T4 canary, one forced handoff, and then
the 200M-token milestone.

## Live truth sources

- Present architecture: `docs/ARCHITECTURE.md`,
  `runtime/subsystem_catalog.py`
- Model and training constants: `training/v2_config.py`
- Data lineage: corpus and pack manifests under `output/v2/data_manifests/`
- Run identity: signed `anra-training-contract/v4` launch manifest
- Checkpoint identity: durability snapshot manifest and canonical pointer
- Runtime state: `/system-map`, `/phase-health`, `/evidence/status`
- Next scene: `TODO.md`

Whenever this story says something is “implemented,” use the linked code and
evidence to determine whether it is also locally verified, cloud verified, or
capability proven.
