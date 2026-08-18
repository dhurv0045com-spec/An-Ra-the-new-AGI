# An-Ra Core

An-Ra Core is the small, standalone foundation that executes the learned V4
neural function. It is deliberately not a chat product, agent, memory system,
tool framework, or training campaign manager.

## What is here

- **Neural Program:** the dense V4 decoder: token IDs to next-token logits.
- **Executor:** validated device/dtype execution and isolated incremental state.
- **Checkpoint boundary:** CPU-first, `weights_only=True` loading; exact dense
  tensor checks; a strict V4 tokenizer contract; separate file and parameter
  identities.
- **Training port:** ordinary differentiable PyTorch forward and parameters.
  Objectives, optimizers, data policy, distributed execution, durability, and
  promotion remain outside Core.

The canonical dense program has 180,093,312 trainable parameters. Historical
181,132,071-parameter artifacts also carry 1,038,759 dormant pilot tensors.
Those tensors are inventory-validated for compatibility but are not executed by
the dense Core.

## Honest capability boundary

Core returns raw V4 next-token logits. It knows nothing about users, chat UI,
tools, retrieval, memory policy, agents, browsers, or storage. A Connector
turns logits into sampled tokens and decides what to put in context. An Outer
system owns streaming and actions. Training/Evaluation is the only authority
that changes weights or promotes new versions.

`CoreState` is an executor-owned, in-memory cache for a homogeneous batch. It
is bound to its executor, architecture, weights, representation, and execution
profile. It can be reset, forked, rolled back, and released. Portable state
serialization is intentionally **not** advertised until it has a versioned,
validated portability contract.

## Use

```powershell
python -m pip install -e .
python -m anra_core --checkpoint "C:\path\anra-v4-current-full-resume.pt" --prompt "Hello"
```

```python
import torch
from anra_core import CoreExecutor

executor = CoreExecutor.from_checkpoint("anra-v4-current-full-resume.pt")
state = executor.create_state()
result = executor.prefill(torch.tensor([[2, 100, 200]]), state=state)
next_id = int(result.logits[:, -1].argmax(dim=-1).item())
executor.release_state(state)
```

Use the exact V4 tokenizer bound to the checkpoint; arbitrary tokenization
breaks learned embedding semantics. Read the engineering documents before
treating an execution change as a model-capability improvement.
