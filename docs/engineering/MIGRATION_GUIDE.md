# An-Ra Core vNext — Migration Guide

**Target Audience:** Engineers upgrading callers from frozen `anra_core` (`f72f193`) to `anra_core` vNext (`0.4.0-vnext`).

---

## 1. Summary of Changes

| Feature / Area | Frozen Reference (`f72f193`) | Core vNext (`0.4.0-vnext`) |
| :--- | :--- | :--- |
| **Execution Complexity** | $O(N^2)$ full prefix recomputation | $O(1)$ token decode via `CoreState` |
| **Model Runtime** | Monolithic `Brain` class | Decoupled `CoreExecutor` + `CoreState` |
| **Deliberation / Sampling** | Embedded inside `Brain._score` | Moved to Connector; `Brain` preserved as compatibility wrapper |
| **Error Handling** | Generic `ValueError` / `RuntimeError` | Structured `anra_core.errors.CoreError` hierarchy |
| **Model Introspection** | Static dictionary | Versioned `ArchitectureIdentity`, `CheckpointIdentity`, `CapabilitySet` |

---

## 2. Upgrading Existing Code

### 2.1 Using `CoreExecutor` Directly (Recommended)

```python
from anra_core import CoreExecutor, load_core_checkpoint, V4Tokenizer

# 1. Load executor
executor = CoreExecutor.from_checkpoint(
    "path/to/checkpoint.pt",
    tokenizer_path="path/to/tokenizer_v4_32k.json",
    device="cpu",
)

# 2. Stateful prefill & incremental decode
state = executor.create_state()
prompt_tokens = executor.tokenizer.encode("Hello An-Ra")
pred = executor.prefill(torch.tensor([prompt_tokens]), state=state)

next_token_id = int(pred.logits[:, -1, :].argmax(dim=-1).item())
step_pred = executor.forward_step(next_token_id, state=state)

# 3. Clean up state
executor.release_state(state)
```

### 2.2 Using `Brain` (Backwards-Compatible)

The existing `Brain` API remains fully functional and now automatically benefits from accelerated stateful decode:

```python
from anra_core import Brain, ThoughtPolicy

brain = Brain.from_checkpoint("path/to/checkpoint.pt", "path/to/tokenizer.json")
thought = brain.think("Hello", ThoughtPolicy(mode="direct", max_new_tokens=64))
print(thought.text)
```
