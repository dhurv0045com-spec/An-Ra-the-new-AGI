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

## X-FACTOR

Connector **cognitive credit assignment**: on failure, generate one-variable
interventions from *observed* information only, rerun the Core, and diagnose
the cause from which intervention flipped the verifier. The hidden ground
truth is structurally inaccessible to the diagnostician
(`ObservedCase` vs `HiddenGroundTruth`).

**Code:** `connector/experiments/cognitive_credit/`
**Status:** methodology validated with runner-graded oracle physics
(20/20 families, 20/20 repairs; completers cannot manufacture success).
Measured on real checkpoints with nonce-fact probes (P1-P6,
`capability_probe.py` v2): in-context information use, plan following, and
tool-result use are 0/5 in *both* natural-language and tag protocols at every
training step. Verbatim copying works only in natural language and only at
mid-training (3/5 at step 5k, 4/5 at step 20k, 0/5 at step 30.4k) — the tag
protocol suppresses it, and continued pretraining past ~20k destroyed it.
The bottleneck is substrate-level context-to-answer binding, not protocol
unfamiliarity alone.

> Note: an earlier prototype (`anra_core/ablation.py`) was removed — its
> intervention generator read the planted failure label, invalidating its
> results as evidence of diagnosis. A second wiring flaw was fixed after
> that: completers used to return hard-coded success labels; success is now
> decided exclusively by the runner's verifier.

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

## Reference runtime (the one executable path)

```python
import anra  # façade over connector/runtime.py

result = anra.run(
    "What is the capital of Portugal?",
    checkpoint=r"C:\path\anra-v4-tpu-latest.pt",
    expected="Lisbon",
    knowledge=("The capital of Portugal is Lisbon.",),
)
print(result.status)   # success | repaired | failed | error
print(result.answer, result.diagnosis, result.changed_variable)
print(result.to_json())  # full structured execution record
```

Loop: task → attempt → Core → verify → on failure, one-variable interventions
(knowledge / plan / decode / tool) → diagnosis from measured flips → repair
retry → verified learning candidate (evidence for future protocol SFT).
Verification is the only source of success; completers return raw outputs.

```powershell
python -m connector.runtime --checkpoint <pt> --task "..." --expected "..."
python -m connector.experiments.cognitive_credit.capability_probe --checkpoint <pt>
python -m connector.experiments.cognitive_credit.run_real --checkpoint <pt>
```

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

## Canonical TPU continuation

The supported post-20k path is `core_vnext_tpu_training.ipynb`, which invokes
`python -m training.train_tpu`. It accepts one verified token pack and a
populated `full_resume` checkpoint. A schema-v1 step-20k checkpoint is an
explicit one-time migration boundary; every checkpoint it writes is schema v3
and includes AdamW moments, pack-local WSD position, verified manifest identity,
sampler cursor, topology, and execution policy for strict continuation.

`kaggle_anra_v4_tpu_training.ipynb` and `training.train_xla` are retained only
as the legacy text-stream/cosine path. Do not mix their checkpoints into an
active pack-aware lineage.
