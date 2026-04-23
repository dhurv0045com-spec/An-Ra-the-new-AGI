# An-Ra Runtime/Training Architecture Decision

Date: 2026-04-22

## Decision
This repository now standardizes on **`anra_brain.py::CausalTransformer`** for both training and runtime model loading.

## Why Option A
The previous runtime path loaded `core/model.py::LanguageModel` (vocab 4096, d_model 128) while training produced checkpoints for `CausalTransformer` (char vocab ~93, n_embd 256). That mismatch prevented shared checkpoints across the full system. The `LLMBridge` now loads `CausalTransformer` checkpoints directly, so MasterSystem and downstream modules consume the same model that training updates.

## Runtime Checkpoint Priority
`anra_brain_identity.pt` → `anra_brain.pt`.

## Compatibility notes
- `CausalTransformer` now exposes `d_model` for Phase 3 wrappers.
- Weight tying is enabled (`lm_head.weight` tied to token embeddings).
