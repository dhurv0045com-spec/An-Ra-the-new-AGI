# An-Ra Core

This branch contains only the part of An-Ra that turns V4 tokens into next-token
predictions and text. It is deliberately not a copy of the full repository.

## What is here

- The exact 32,768-token V4 tokenizer.
- The dense 18-layer, width-896 decoder transformer.
- Grouped-query attention, adjacent-pair RoPE, QK normalization, hybrid
  full/sliding causal attention, RMSNorm, SwiGLU, and tied embeddings.
- A strict checkpoint loader and a small command-line generator.

The executable dense model has **180,093,312 parameters**. The historical
181,132,071 total included 1,038,759 parameters owned by dormant experimental
subsystems. Those parameters are intentionally absent here. This is the core
language model, not the full experimental ABI.

## What is not here

No training pipeline, optimizer, Drive integration, API server, UI, agents,
tools, retrieval, memory, telemetry, HAL, ESV, RIM, MoD, DSTP, MoE, MTP,
moonshots, robotics, or self-modification. Git history still preserves the full
project; the tip of this branch stays inference-only.

## Run

The learned checkpoint is not committed because it is approximately 2 GB.
Use your downloaded V4 checkpoint:

```powershell
python -m pip install -e .
python -m anra_core --checkpoint "C:\path\anra-v4-current-full-resume.pt" --prompt "Hello"
```

CUDA is selected automatically when available. Add `--device cpu` to force CPU,
or `--temperature 0.7 --top-p 0.92` for sampling. Greedy generation is the
default and is deterministic.

The loader reads the checkpoint on CPU first, validates every required dense
tensor and its shape, permits only the known removed subsystem tensors, then
moves the model to the requested device. It refuses partial or architecturally
incompatible checkpoints rather than silently running them.
