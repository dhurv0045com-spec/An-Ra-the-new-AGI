# Baseline selection and executable audit

Use a fresh 48-wide, two-layer V5 dense decoder with three query heads, one KV head, 16-dimensional heads, FFN width 96, context 32, affine QK normalization, RMSNorm, RoPE, SwiGLU and a single tied embedding/projection. The code derives this spec from the V5A shape contract and checks parameter inventory at construction. The tokenizer is a reversible character serializer over the declared symbolic alphabet with distinct BOS, EOS and PAD IDs. It is specific to this reference surface.

| Area | Inherited path and Elendril decision |
| --- | --- |
| Model and initialization | Keep `v5_model.core.initialize`, FP32 scaled residual initialization, tied linear projection and packed causal layout. Use one segment per example. Validated padding has a self-attention diagonal and no target. |
| Attention and position | Keep per-segment RoPE and explicit Boolean visibility masks. Future-token perturbation and existing packed-layout tests exercise this path. |
| Objective | Keep `v5_objectives.causal_lm_loss` and pass BOS/PAD IDs explicitly. The eligibility mask belongs to target indices. The inherited objective shifts logits and targets once. Include EOS in CE. |
| Optimizer and schedule | Keep Cymek AdamW parameter grouping and moments. Use constant LR 0.003 for the CPU instrument and clip at 1.0. The production 5B-token WSD schedule, accumulation, partitioning and distributed denominators serve a different workload. |
| Data and sampler | Regenerate immutable fixtures from code. Bind ordered data content and training recipe to checkpoints. An update-local RNG samples row indices from seed and update index. Training consumes one fixed-width family per run. Padding/shift tests exercise mixed-length tensor serialization. |
| Checkpoint | Keep Cymek `TrainingState` and `CheckpointStore` manifests and transaction formats. Elendril uses a process-scoped writer lock around the parent check and quarantines same-generation interrupted staging. Refuse stale parent, incompatible source, seed, recipe or data, missing files and byte corruption. Restore strictly with CPU `weights_only=True`. |
| Evaluation | Use the existing VisibleTask/EvaluatorTruth types. The evaluator scores complete emitted token streams and can rescore a saved receipt without importing Torch or the model. Process isolation and hidden future labels belong to the later verifier design. |

Formation-Mux experiment-specific treatments modify tied rows and softmax participation. The old S5 checkpoints are absent from inspected local artifact custody. Signac M102 and large V5 training paths are valuable rollback/audit references with much higher memory and data requirements. BRAMASTRA's independent implementation supplied validation ideas. The existing dense V5 model is the lower-risk integration point because it is inherited directly, has mechanical tests and avoids a second decoder framework.

The rollback reference is the charter commit over the exact Cymek ancestor. Elendril adds a research path inside the packaged `anra_v5` namespace. The production trainer, tokenizer assets, accelerator notebooks and historical artifacts retain their own established contracts.
