# QUARANTINED — Invalid Stateless-Decode Evaluation

**Status: INVALID as capability evidence. Do not cite as model behavior.**

## What these scripts did

`eval_checkpoint_gpu.py` and `logit_diagnostics.py` (as first written) fed
single tokens back through a **stateless full forward**:

```python
current = model(ids)[:, -1, :]          # full-sequence forward
nxt = torch.tensor([[next_id]])
current = model(nxt)[:, -1, :]          # next token with NO prefix context
```

## Why that is invalid

A decoder conditions on the *entire prefix*. Calling `model(next_token)`
without the accumulated prefix means every token after the first is generated
from a one-token context. Any multi-token capability measurement beyond
token one is therefore measuring a broken protocol, not the model:

- "cannot complete a sentence" conclusions are unsupported — the model was
  never shown the sentence;
- repetition/degeneration measurements are confounded: a stateless loop
  cannot distinguish model attractors from missing context;
- exact-match scores after token one are meaningless.

## What remains valid here

- The **logit diagnostics at position one** (first-token margin/entropy) are
  valid — position one genuinely has only the prompt as context.
- The **checkpoint loading and identity handling** are reusable.
- The **prompt battery** (cases and gold answers) is reusable.

## Superseded by

All capability conclusions must come from the executor-based path
(`anra_core.generate` via `CoreExecutor`, which maintains incremental KV
state), e.g. `scripts/eval_soup_cpu.py` and
`connector/experiments/cognitive_credit/capability_probe.py`.

Evidence identity convention (mandatory for promotion-grade results):
`experiment_schema`, `source_commit`, `checkpoint_file_sha256`,
`checkpoint_parameter_sha256`, `global_step`, tokenizer identity,
architecture identity, execution profile, exact decode policy
(raw vs assisted), evaluator version, timestamp, and
`supersedes`/`invalidates` links where applicable.
