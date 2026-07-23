# An-Ra Developer Guide

Updated: 2026-07-23  
Purpose: help an engineer safely run, inspect, change, and verify the current
V4 repository without accidentally reviving an old training path.

## Start with the contract

The operational model is `anra-v4-180m`, V4 vocabulary 32,768, context 2,048,
AdamW, and routine seed 1301. The 500M profile is a growth child, not a second
scratch model. V3 tokenizer launches and arbitrary model sizes are rejected.

Read in this order:

1. `docs/ARCHITECTURE.md`
2. `docs/engineering/V4_ARCHITECTURE_GATE.md`
3. `TODO.md`
4. The code path you intend to change

## Local setup

From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This workspace also has `.venv-cuda` for the installed CUDA-compatible stack.
Do not assume a CUDA environment is correct merely because `torch` imports:

```powershell
.\.venv-cuda\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

Never commit `.env`, provider credentials, Drive OAuth files, checkpoints,
corpus shards, or generated evidence containing private data.

## Run the application

```powershell
.\.venv-cuda\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/developer`.

Useful read-only endpoints:

- `GET /health`
- `GET /system-map`
- `GET /phase-health`
- `GET /evidence/status`
- `GET /training/preflight`

Chat and generation endpoints require a compatible checkpoint and runtime:

- `POST /chat`
- `POST /generate`
- `GET /traces/{trace_id}`

## Understand a training launch

There are three different concepts:

1. **Model profile**: architecture and exact parameter contract.
2. **Token window**: the deterministic corpus interval leased to a worker.
3. **Seed**: a reproducibility input. Routine training uses seed 1301.

A cloud launch should be created only from a clean, pushed commit and an
immutable pack:

```powershell
$env:ANRA_MANIFEST_SIGNING_KEY = "<owner-held secret>"
.\.venv-cuda\Scripts\python.exe scripts\create_cloud_launch.py `
  --pack-root "C:\path\to\pack" `
  --output "output\v2\cluster_launch.json" `
  --artifact-path "output/v2/checkpoints/anra-v4-180m.pt" `
  --checkpoint-source scratch `
  --worker-id trainer-1 `
  --runtime-estimate-hours 3
```

The signed manifest is then bound to a cluster job. Do not hand-edit it after
signing. A worker verifies its signature, commit, window, source checkpoint,
destination, and resource limits before launching:

```powershell
.\.venv-cuda\Scripts\python.exe -m training.train_unified `
  --mode session `
  --launch-manifest output\v2\cluster_launch.json
```

Continuation uses the previous verified `full_resume` artifact as
`--checkpoint-source` and a new non-overlapping pack. Scratch and continuation
must never write over their source.

## Local bounded execution

Use local GPU runs only for a concrete engineering question: one optimizer
step, save/load, numerical parity, or a short canary. They are not a substitute
for the signed campaign.

```powershell
.\.venv-cuda\Scripts\python.exe -m training.train_unified `
  --mode preflight `
  --model-size anra-v4-180m
```

The trainer refuses CPU for canonical training unless an explicit pilot-only
override is set. Do not use that override for performance estimates.

## Checkpoint rules

`full_resume` is the only training source. It includes optimizer, scheduler,
scaler, RNG, sampler, step, tokens, and lineage. `fp16_inference` is smaller
because it contains model tensors only.

Never:

- rename a compact artifact and treat it as resumable;
- overwrite the checkpoint named as a signed parent;
- delete the previous resume generation before the new one is protected;
- accept a low loss as proof of coherent language;
- close paid compute before the checkpoint hash is independently verified.

Checkpoint protocol: `training/checkpoint_durability.py`.

## Changing the architecture

The canonical geometry lives in `training/v2_config.py`. Architecture changes
must create an explicit experimental profile; they must not silently mutate
`anra-v4-180m`.

For a subsystem:

1. Register its lifecycle and cost in `runtime/subsystem_catalog.py`.
2. Default it off unless it is already active.
3. Define the frozen parent, promotion question, metric, token budget, and
   rollback.
4. Compare it using the same parent checkpoint, data order, optimizer, and
   seed.
5. Promote only after capability, stability, and useful-compute gains.

The model growth implementation is in `training/csii.py`,
`training/grow_model.py`, and `training/growth_runtime.py`.

## Adding post-training capability

Use separate signed lineages:

- SFT teaches instruction and dialogue behavior.
- RLVR/STaR uses verifiable outcomes.
- DPO requires audited preference pairs.
- LoRA/DoRA adapters remain bound to one base checkpoint hash.

Do not mix these records invisibly into pretraining. Contracts live in
`training/posttraining_contract.py`; adapter promotion and rollback live in
`inference/adapters.py`.

## Focused verification

Run the smallest suite that covers the changed contract:

```powershell
.\.venv-cuda\Scripts\python.exe -m pytest -q `
  tests/test_training_contract_v4.py `
  tests/test_checkpoint_durability.py `
  tests/test_cloud_launch_contract.py `
  tests/test_model_growth_contract.py

.\.venv-cuda\Scripts\python.exe -m ruff check `
  training/launch_manifest.py `
  training/checkpoint_durability.py `
  training/train_unified.py
```

Run broad CPU suites before a release or after cross-cutting changes, not after
every prose or isolated contract edit. GPU tests need an explicit numerical or
performance question.

## Repository map

| Path | Responsibility |
| --- | --- |
| `anra/`, `anra_brain.py` | Model and architecture contracts |
| `training/` | Data, optimizer, training, checkpoints, growth, post-training |
| `inference/`, `generate.py` | Context, generation, cache, adapters |
| `retrieval/`, `memory/` | External knowledge and persistence |
| `cognition/` | Planning, verification, correction |
| `agents/`, `execution/` | Permissioned tools and action |
| `evaluation/`, `engine/` | Measurement, telemetry, promotion |
| `runtime/` | Registries and shared evidence |
| `scripts/` | Supported operational entrypoints |
| `tests/` | Contracts and evidence gates |

## Pull-request discipline

- Preserve unrelated user changes.
- Never include checkpoint or corpus binaries.
- State which claim the test actually supports.
- Distinguish code-complete, locally verified, live-cloud verified, and
  capability-proven.
- Update the subsystem catalog when lifecycle changes.
- Regenerate `docs/system_graph.json` when architecture truth changes.

## Live truth sources

- Current commit: `git rev-parse HEAD`
- Dirty state: `git status --short`
- Runtime CLI: `python -m training.train_unified --help`
- Model profiles: `training/v2_config.py`
- Launch validation: `training/launch_manifest.py`
- System claims: `runtime/system_registry.py`,
  `runtime/subsystem_catalog.py`
- Generated state: `docs/system_graph.json`
- Focused evidence: test outputs and `runtime/evidence_stream.py`

If a command in this guide disagrees with `--help`, the executable help wins
and this document must be corrected.
