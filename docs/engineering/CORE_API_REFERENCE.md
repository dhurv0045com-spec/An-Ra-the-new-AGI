# An-Ra Core vNext — API and Contract Reference

**API Version:** `1.0.0`  
**Runtime Version:** `0.4.0-vnext`  
**Package:** `anra-core`

---

## 1. Core Architecture Overview

An-Ra Core vNext provides the standalone, inference-accelerated, differentiable neural foundation for the An-Ra system. It strictly enforces the 4-tier system boundary:
- **Neural Model (`AnRaCore`)**: Pure mathematical definition of the 180,093,312-parameter dense transformer.
- **Core Executor (`CoreExecutor`)**: Runtime engine managing device placement, precision, execution profiles, and incremental KV state lifecycle.
- **Connector**: Outside Core. Owns context assembly, memory routing, tool calls, temperature/top-p sampling, and cognitive deliberation.
- **Outer**: Outside Core. Owns user interfaces, persistence, network streaming, and actions.

---

## 2. Primary Public Interfaces

### 2.1 `CoreExecutor`
The primary operational entrypoint for model loading and execution.

```python
class CoreExecutor:
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        tokenizer_path: str | Path | None = None,
        config: CoreConfig = CANONICAL_CONFIG,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> CoreExecutor: ...

    def architecture_identity(self) -> ArchitectureIdentity: ...
    def representation_identity(self) -> RepresentationIdentity | None: ...
    
    def create_state(self, *, capacity: int | None = None) -> CoreState: ...
    def prefill(self, token_ids: torch.Tensor, state: CoreState) -> PredictionResult: ...
    def forward_step(self, token_id: int | torch.Tensor, state: CoreState) -> PredictionResult: ...
    def forward(self, token_ids: torch.Tensor, state: CoreState | None = None) -> PredictionResult: ...
    
    def reset_state(self, state: CoreState) -> None: ...
    def fork_state(self, state: CoreState) -> CoreState: ...
    def release_state(self, state: CoreState) -> None: ...
    
    def describe(self) -> dict[str, Any]: ...
```

### 2.2 `CoreState`
An opaque handle managing intermediate acceleration state (e.g. KV cache) without exposing Transformer ontology or session data.

```python
class CoreState:
    architecture_version: str
    checkpoint_id: str
    execution_profile_id: str
    capacity: int
    current_length: int
    state_id: str
    is_released: bool

    def reset(self) -> None: ...
    def fork(self) -> CoreState: ...
    def release(self) -> None: ...
    def descriptor(self) -> dict[str, Any]: ...
```

### 2.3 `PredictionResult`
The raw mathematical prediction envelope produced by Core.

```python
@dataclass(frozen=True, slots=True)
class PredictionResult:
    logits: torch.Tensor          # Raw next-token logits [batch, seq_len, vocab_size]
    sequence_length: int          # Current cumulative token length
    execution_profile_id: str     # Profile identifier used for this step
    metadata: dict[str, Any]      # Extensible diagnostic telemetry
```

---

## 3. Error Taxonomy (`anra_core.errors`)

All Core exceptions inherit from `CoreError(ValueError)`:

| Exception Class | Error Code | Trigger Condition |
| :--- | :--- | :--- |
| `CheckpointIncompatibleError` | `ERR_CHECKPOINT_INCOMPATIBLE` | Missing/corrupt tensors, unallowlisted keys, or shape mismatch. |
| `RepresentationIncompatibleError` | `ERR_REPRESENTATION_INCOMPATIBLE` | Tokenizer vocabulary or cryptographic probe mismatch. |
| `UnsupportedProfileError` | `ERR_UNSUPPORTED_PROFILE` | Unsupported device or precision format requested. |
| `UnsupportedCapabilityError` | `ERR_UNSUPPORTED_CAPABILITY` | Optional capability (e.g. quantization) requested when unavailable. |
| `ContextOverflowError` | `ERR_CONTEXT_OVERFLOW` | Sequence length exceeds configured block size or capacity. |
| `StateIncompatibleError` | `ERR_STATE_INCOMPATIBLE` | State handle architecture does not match active model. |
| `StateReleasedError` | `ERR_STATE_RELEASED` | Operation attempted on an already released state handle. |
| `ResourceExhaustionError` | `ERR_RESOURCE_EXHAUSTION` | Out-of-memory on CPU or GPU backend. |
| `UnexpectedExecutionFault` | `ERR_UNEXPECTED_EXECUTION_FAULT` | Unrecoverable internal execution fault. |

---

## 4. Differentiable Training Contract

`AnRaCore` supports standard PyTorch autograd for external pretraining and SFT:

```python
from anra_core import AnRaCore, CANONICAL_CONFIG
import torch.nn.functional as F

model = AnRaCore(CANONICAL_CONFIG).train()
input_ids = torch.tensor([[2, 45, 128, 999]])
target_ids = torch.tensor([[45, 128, 999, 3]])

# 1. Forward
logits = model(input_ids)

# 2. Objective formulation outside Core
loss = F.cross_entropy(logits.view(-1, 32768), target_ids.view(-1))

# 3. Backward
loss.backward()
```
