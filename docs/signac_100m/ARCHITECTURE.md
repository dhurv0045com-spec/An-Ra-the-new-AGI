# Signac architecture

## Research aim

Test the existing An-Ra research program at the next useful scale rung. The desired evidence is candidate-free acquisition, query control, structural transfer, counterfactual sensitivity, faithful termination, and retention/recovery—not language-model loss alone. The core remains simple enough to attribute failures. Data generation, curriculum, replay policy, tool use, memory, evaluation, and promotion are separate system layers; they do not become neural components by naming them “AGI.”

## Model contract

The canonical geometry comes from the M102 recipe already present in `v5_contracts/training_spec.py`; no dimensions were invented from a generic scaling rule. This recipe is an engineering candidate, not evidence that depth or width is superior.

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

The core implementation is `v5_model/`. Parameter accounting comes from the same `ModelSpec.parameter_receipt()` used by V5 contracts. Initialization retains the V5 residual scaling. The training objective remains causal cross-entropy over eligible tokens with the answer EOS supervised. Auxiliary losses stay at zero absent transfer evidence.

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
| BF16 weights + FP32 master + two FP32 Adam moments + FP32 gradients | ~1.83 GB | Component estimate; excludes activations, XLA buffers, compiler padding, fragmentation, and input staging; do not treat as fit proof |
| Tokens at 20× parameters | 2,035,801,600 | Generic planning prior only; not justified by An-Ra capability evidence |

Every candidate has separate exact estimates emitted by `candidate_receipts()`. Keep these values tied to code rather than copying them into a training launcher by hand.

The target runtime must measure peak allocated and reserved memory at the chosen batch, bucket, accumulation, and precision. A TPU preflight does not certify model fit or production training.

## System boundaries

The experiment identity must bind the model spec, code revision, tokenizer bytes, corpus/source/split manifests, packing and mixture receipts, evaluation version, optimizer/schedule, seeds, and hardware topology. Exact-resume state must include model and optimizer, token/update counters, data cursor, RNG, topology, and all identity hashes. A checkpoint that restores weights but changes the next batch is not an exact resume.

Training and evaluation stay separate. Development curves tune capability dose; sealed examples remain inaccessible until a preregistered endpoint. Report per-family and per-axis outcomes (canonical, query swap, order swap, combined changes, relevant/irrelevant intervention, structural OOD, and natural analogue). A narrow task score or lower loss cannot promote a general capability claim.

## Explicit non-claims

Signac has no proven optimal tokenizer, 100M transfer law, production corpus, TPU parity, TPU fit, cognition promotion, continual-learning controller, or AGI result. Every one of those remains a gate or hypothesis in [`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md) and [`TRAINING_PLAN.md`](TRAINING_PLAN.md).
