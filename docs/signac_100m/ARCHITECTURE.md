# Signac architecture

## Research aim

Test the existing An-Ra research program at the next useful scale rung. The desired evidence is candidate-free acquisition, query control, structural transfer, counterfactual sensitivity, faithful termination, and retention/recovery—not language-model loss alone. The core remains simple enough to attribute failures. Data generation, curriculum, replay policy, tool use, memory, evaluation, and promotion are separate system layers; they do not become neural components by naming them “AGI.”

## Model contract

The canonical geometry comes from the M102 recipe in `signac_100m/spec.py`, using the shared `ModelSpec` contract in `v5_contracts/model_spec.py`; no dimensions were invented from a generic scaling rule. This recipe is an engineering candidate, not evidence that depth or width is superior.

| Field | Signac candidate |
|---|---:|
| Family | Dense causal decoder Transformer |
| Vocabulary | 24,576 (provisional working choice) |
| Width × layers | 640 × 20 |
| Query / KV heads | 10 / 5 |
| Head dimension | 64 |
| SwiGLU inner width | 1,600 |
| Native context | 4,096 (not learning-validated) |
| Attention | Full causal GQA; pairwise RoPE, base 10,000 |
| Normalization | Pre-RMSNorm; affine per-head QK norm; final RMSNorm |
| Embedding/output | One tied table; full 24,576-class output |
| Bias / dropout | None / 0 |
| Exact parameter count | **101,790,080** |

The core implementation is `v5_model/`. Parameter accounting comes from the same `ModelSpec.parameter_receipt()` used by V5 contracts. Initialization retains the V5 residual scaling. The training objective remains causal cross-entropy over eligible tokens with the answer EOS supervised. Auxiliary losses stay at zero absent transfer evidence. The development source builder now instantiates all nine frozen cognition families and keeps their subfamily identities attached through family-segregated packing; it remains synthetic development material and does not qualify the production corpus or demonstrate learned cognition.

Each model forward now builds FP32 RoPE angles once from the packed positions and shares them across query/key projections and all transformer layers. This removes repeated identical phase, sine, and cosine construction from the Python/XLA graph while preserving the original dtype cast before rotation. Tests compare outputs and gradients bit-for-bit in FP32 and BF16, and compare full-model gradients with activation checkpointing enabled. The change reduces redundant operations by construction; target throughput still needs Kaggle measurement.

The training path now exposes pre-projection hidden states and computes the tied full-vocabulary causal loss in 512-position chunks. Each chunk's tied linear projection and FP32 cross-entropy are checkpointed for recomputation during backward, avoiding retention of one full `[batch, sequence, vocabulary]` logits tensor. CPU and CUDA use PyTorch's non-reentrant checkpoint API; XLA dispatches to `torch_xla.utils.checkpoint`, because the current PyTorch/XLA implementation rejects `use_reentrant=False` ([implementation](https://github.com/pytorch/xla/blob/master/torch_xla/utils/checkpoint.py), [XLA checkpointing guidance](https://docs.pytorch.org/xla/master/perf/fsdp_collectives.html)). The target mask, eligible-token count, numerator, and tied embedding gradients match the full-logit reference in CPU FP32 tests; CPU BF16 autocast and empty-rank zero gradients are covered too. This reduces the intended live logits activation by construction, but the Kaggle wheel's checkpoint behavior, target peak memory, compilation cost, and throughput still need qualification.

Signac keeps the previously declared **M102** recipe as the primary research candidate. It compares two unpromoted TPU challengers:

| Candidate | Geometry | Exact parameters | Design question |
|---|---|---:|---|
| M102 primary | 640 × 20, FFN 1,600, 10/5 heads | 101,790,080 | Existing scale recipe and research continuity |
| Depth-preserving challenger | 640 × 20, FFN 1,536, 10/5 heads | 99,332,480 | Does a 128-aligned FFN improve TPU efficiency while preserving depth? |
| Fully tiled challenger | 768 × 12, FFN 2,176, 12/6 heads | 100,303,104 | Do fully aligned major matrix dimensions offset reduced depth and wider attention? |

The depth-preserving candidate changes only FFN width relative to M102. The fully tiled candidate also changes width, depth, heads, and attention cost, so it is a broader systems challenger. At batch 1 and context 4,096, a naively materialized BF16 attention-score tensor is 335.5 MB per active layer for both 640-wide candidates and 402.7 MB for the 768-wide candidate (one sequence per replica). This is a shape-based upper component, not a claim about the selected attention kernel's actual allocation. The Kaggle notebook records exact receipts and compile-inclusive first-update latency and attempts one full-context update per candidate. It does not supply reliable peak-memory or steady-state throughput measurements. No source-level estimate can choose the faster or more capable geometry; promote only after matched target performance, exact resume, and multiple-seed capability measurements.

## Cost envelope

These are analytic estimates, not measured TPU results:

| Quantity | Estimate | Interpretation |
|---|---:|---|
| Parameters | 101,790,080 | Exact contract receipt |
| Model + FP32 Adam moments checkpoint | ~1.22 GB | Excludes metadata, temporary copies, and filesystem overhead |
| FP32 model parameters + FP32 gradients + two FP32 Adam moments | ~1.63 GB (1.52 GiB) | 16 bytes/parameter in the current executable path; component subtotal, not peak; excludes activations, optimizer-step metadata, temporary copies, XLA buffers, compiler padding, fragmentation, and input staging |
| Tokens at 20× parameters | 2,035,801,600 | Generic planning prior only; not justified by An-Ra capability evidence |

Every candidate has separate exact estimates emitted by `candidate_receipts()`. Keep these values tied to code rather than copying them into a training launcher by hand.

The target runtime must measure peak allocated and reserved memory at the chosen batch, bucket, accumulation, and precision. A TPU preflight does not certify model fit or production training.

## System boundaries

The experiment identity must bind the model spec, source commit plus content-addressed source bundle, tokenizer bytes, corpus/source/split manifests, packing and mixture receipts, evaluation version, optimizer/schedule, seeds, and hardware topology. The source bundle inventory includes the Kaggle launch notebook so changing workload settings changes the run identity. Exact-resume state must include model and optimizer, token/update counters, data cursor, RNG, topology, and all identity hashes. A checkpoint that restores weights but changes the next batch is not an exact resume.

The existing local v1 checkpoint inventory remains unchanged. Distributed replicated training uses the separate v2 `CheckpointStore.publish_distributed/restore_distributed` contract: one shared model and Adam payload, a world-sized rank-state bundle containing each rank's RNG and sampler cursor bytes, per-rank model/optimizer receipts that must match the shared payload hashes, an ordered aggregate RNG identity in `TrainingState`, and a collective receipt bound to update, topology, world size, and per-rank receipts. A lineage can migrate once from v1 to v2 through the explicit distributed publish API; it cannot downgrade to v1. The shared `cursor.json` is the scalar `TrainingState` projection; rank-local continuation comes from `rank_states.bin`. The v2 store verifies the complete rank set and payload hashes. The production loop and resume verifier call the same `ProductionSampler` to rebuild deterministic bucket/family windows, bounded-replay transitions, target denominators, and rank-local microsteps. `TrainingState` stores the sampler SHA explicitly as well as binding it through the run identity; rank cursor receipts repeat that SHA beside the next-microstep fingerprint, and restore checks both against the supplied sampler before applying rank RNG state. Host tests cover two-rank trainer publication and fresh-backend restore, plus a three-session single-rank campaign continued through a terminal checkpoint. The campaign also has an opt-in CPU v2 mode. Before production checkpoint serialization, shared model and optimizer state trees are detached and copied to CPU, with repeated tensor references and state-dict metadata preserved; host tests verify this boundary, while actual XLA transfer and serialization still need Kaggle evidence. The XLA adapter serializes CPU and rank-local XLA RNG state and exposes an explicit restore hook. A host-only fake single-rank test traverses the bounded `execution="xla-development"` campaign through update, v2 publish, and fresh-session restore; it verifies orchestration only. The Kaggle notebook now offers an opt-in synthetic two-update integration path through the real M102 backend and sampler. Its final resume check runs through the fresh second worker group and avoids a duplicate M102 model/Adam allocation on the first group. It is disabled by default. Production `run_campaign(execution="xla")` still fails closed. No live XLA campaign test has run. Mocked host tests do not establish target-side collectives, memory fit, throughput, exact TPU continuation, or durable Kaggle output.

Training and evaluation stay separate. Development curves tune capability dose; sealed examples remain inaccessible until a preregistered endpoint. Report per-family and per-axis outcomes (canonical, query swap, order swap, combined changes, relevant/irrelevant intervention, structural OOD, and synthetic naturalized analogue). A narrow task score or lower loss cannot promote a general capability claim.

## Explicit non-claims

Signac has no proven optimal tokenizer, 100M transfer law, production corpus, TPU parity, TPU fit, cognition promotion, continual-learning controller, or AGI result. Every one of those remains a gate or hypothesis in [`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md) and [`TRAINING_PLAN.md`](TRAINING_PLAN.md).
