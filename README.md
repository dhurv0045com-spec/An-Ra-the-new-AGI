# An-Ra Core vNext

An-Ra Core vNext is the standalone, high-performance neural foundation for the An-Ra system. It packages the exact 180,093,312-parameter dense V4 decoder transformer and introduces an explicit, hardware-accelerated `CoreExecutor` with isolated incremental `CoreState` management.

## Key Features

- **Exact V4 Dense Neural Architecture:** 18 layers, $d_{\text{model}}=896$, SwiGLU $d_{\text{ff}}=2432$, 14/2-head Grouped Query Attention ($d_{\text{head}}=64$), QK RMSNorm, adjacent-pair RoPE, and hybrid full/sliding causal attention.
- **Hardware-Accelerated Incremental State:** Eliminates repeated prefix projection during generation, achieving **2.84x faster token decode** on CPU.
- **Strict Layer Isolation:** Neural Model and Core Executor are decoupled from session management, cognitive deliberation, tool calling, and user interfaces.
- **Typed Error Taxonomy:** Structured machine-readable error hierarchy (`anra_core.errors.CoreError`).
- **Complete Introspection:** Versioned `ArchitectureIdentity`, `CheckpointIdentity`, `RepresentationIdentity`, and `CapabilitySet`.
- **Differentiable Autograd:** Cleanly exposed for external pretraining, SFT, and optimizer integration.

## Installation

```powershell
python -m pip install -e .
```

Or install from wheel:
```powershell
pip install dist/anra_core-1.0.0-py3-none-any.whl
```

## Quickstart

### 1. Using `CoreExecutor`

```python
import torch
from anra_core import CoreExecutor, AnRaCore, CANONICAL_CONFIG, V4Tokenizer

# Initialize or load model
model = AnRaCore(CANONICAL_CONFIG).eval()
tokenizer = V4Tokenizer.load("anra_core/assets/tokenizer_v4_32k.json")
executor = CoreExecutor(model, tokenizer=tokenizer)

# Create isolated execution state
state = executor.create_state(capacity=2048)

# Prefill prompt
prompt_ids = torch.tensor([[tokenizer.bos_token_id, *tokenizer.encode("Hello world")]])
res = executor.prefill(prompt_ids, state=state)

# Step next token
next_id = int(res.logits[:, -1, :].argmax(dim=-1).item())
step_res = executor.forward_step(next_id, state=state)

# Clean up
executor.release_state(state)
```

### 2. Command Line Interface

```powershell
python -m anra_core --checkpoint "C:\path\anra-v4-current-full-resume.pt" --prompt "Hello"
```

## Running Tests

```powershell
pytest -v
```
