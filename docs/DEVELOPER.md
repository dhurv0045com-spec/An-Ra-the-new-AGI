# An-Ra Developer Guide

This guide is for contributors changing An-Ra code, data contracts, training,
inference, evaluation, or the operator interface on `iterate500`.

## Ground Rules

1. Preserve the native An-Ra weight lineage. Do not introduce an external model
   backend as a hidden replacement.
2. Preserve canonical token IDs. Vocabulary growth is append-only.
3. Never report a partial checkpoint load as success.
4. Keep evaluation state out of real sessions.
5. Require evidence for capability claims and ablations for subsystem claims.
6. Do not commit checkpoints, private corpora, secrets, or generated run output.
7. Keep changes scoped; architecture, training objective, and release gates must
   not drift together in an untraceable patch.

## Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional groups:

```powershell
python -m pip install -e ".[dev,ml]"
python -m pip install -e ".[dev,evidence]"
python -m pip install -e ".[dev,observability]"
```

The package requires Python 3.10+. Project metadata and tool configuration live
in `pyproject.toml`; `requirements.txt` is intentionally empty.

## Canonical Entry Points

| Operation | Entry point |
| --- | --- |
| Package CLI | `python -m anra` or installed `anra` command |
| API and developer UI | `python app.py --host 127.0.0.1 --port 8000` |
| Frontier checkpoint proof | `python scripts/check_frontier_checkpoint.py` |
| Frontier chat/smoke suite | `python scripts/chat_frontier.py` |
| T4 trainer | `python scripts/build_brain.py` |
| TPU trainer | `python scripts/build_brain_tpu.py` |
| Unified training dispatcher | `python -m training.train_unified` |
| Corpus download | `python scripts/download_training_data.py` |
| Colab corpus cache | `python scripts/colab_prepare_data.py` |
| ThirdEye summary | `python scripts/show_thirdeye_summary.py` |
| Structure check | `python -m scripts.verify_structure` |

The public model profile is `frontier`. Historical `1B` aliases resolve to this
499M profile and must not be documented as a separate model.

## Source Ownership

| Directory | Responsibility |
| --- | --- |
| `anra/` | typed public package, architecture, core protocols, serving adapters |
| `anra_brain.py` | frontier transformer and native subsystem implementation |
| `generate.py` | checkpoint-backed generation, session state, guards, traces |
| `app.py` | API, developer UI, Matrix, evaluation and operator endpoints |
| `training/` | configs, runtime loading, trainer, stages, corpus and evaluation logic |
| `inference/` | prompt optimization, cache, sampling, model IO, system connector |
| `identity/` | ESV, RIM support, HAL, CIV and identity constraints |
| `memory/` | memory stores, retrieval, ghost context and routing |
| `cognition/` | planning, debate, consolidation, epistemic and causal services |
| `agents/` / `execution/` | orchestration and authorized tool execution |
| `runtime/` | registries, capability graph inputs, feedback and operator state |
| `evaluation/` | promotion, rollback, benchmarks and evidence adapters |
| `phase4/web/` | optional React/Vite client; Colab uses backend-served `/developer` |

## Development Loop

Before editing:

```powershell
git status --short --branch
python -m scripts.verify_structure
```

After a focused change:

```powershell
python -m ruff check <changed-paths>
python -m pytest <focused-tests> -q --tb=short
```

Before push:

```powershell
python -m ruff check training inference anra cognition intelligence evaluation data engine robotics multimodal runtime
python -m pytest tests -m "not gpu" --ignore=tests/test_drive_session_manager_integration.py --ignore=tests/test_v2_drive_artifacts.py -q --tb=short
python -m mypy anra --strict --ignore-missing-imports
```

Drive and GPU behavior requires the matching environment. State explicitly when
those tests were not run.

## Change Contracts

### Checkpoint or architecture changes

- Increment schema only when the serialized contract changes.
- Add a named migration and test its exact initialized tensors.
- Preserve compatible tensors byte-for-byte where possible.
- Block missing embedding, attention, MLP, norm, and output-head tensors.
- Update parameter accounting in `training/v2_config.py` and
  `anra/architecture.py` together.
- Add migration, exact-load, and malformed-load tests.

### Tokenizer changes

- Never modify IDs `0-8208`.
- Update metadata, vocabulary and special-token hashes, and fixed probes.
- For append-only growth, deterministically initialize embedding/head rows from
  constituent old-token embeddings plus bounded noise.
- Re-tokenize shards under a new manifest; never silently reuse old token shards.

### Prompt changes

- Count with the active tokenizer.
- Reserve output space before allocating context.
- Preserve a fitting current message.
- Truncate oldest history before relevant memory/current input.
- Record every decision in `PromptAssemblyTrace`.
- Test exact formatted prompt and boundary token counts.

### Generation changes

- Keep greedy, fixed-seed, cache-off behavior reproducible.
- Apply repetition controls only to generated answer IDs.
- Preserve finite checks, control-token blocks, EOS and explicit stop reasons.
- Keep state request-scoped and commit only after accepted output.
- Prove cached/uncached token parity before enabling cache by default.

### Native subsystem changes

- Add execution telemetry and a neutralized ablation path.
- Avoid batch-crossing state.
- Bound adaptive parameters and regularize them toward a known initialization.
- Compare the same checkpoint across at least three seeds.
- Reject a change that helps one metric while silently breaking protected
  language, identity, or safety dimensions.

### Data changes

- Store source revision, license, record and token count, quality distribution,
  tokenizer hash, and SHA-256.
- Split by source-document hash before token chunking.
- Run exact and MinHash deduplication.
- Keep raw causal and conversation objectives separate.
- Verify every required source; incomplete 15/30 GB profiles must fail closed.

### API changes

- Use typed Pydantic request models and bounded fields.
- Keep `/chat` backward-compatible unless a versioned migration is provided.
- Return trace identity and explicit quality/persistence status.
- Avoid exposing secrets, private suite contents, or raw sensitive memory.
- Add endpoint tests for success, validation failure, and state isolation.

## API Surface

Core runtime:

```text
POST /generate
POST /chat
GET  /traces/{trace_id}
GET  /stream
GET  /health
GET  /status
GET  /strategies
```

Evidence and diagnostics:

```text
GET  /evaluations/current
POST /evaluations/private-promotion
GET  /evaluations/private-promotion/status
GET  /evaluations/private-review-queue
POST /evaluations/private-review
POST /diagnostics/recovery-gate
POST /diagnostics/full-system-integration
POST /diagnostics/cache-parity
GET  /diagnostics/session-isolation
POST /diagnostics/rollback-drill
GET  /diagnostics/release-evidence
```

Systems and operations include cognition, owner model, experiments, training
preflight/manifests, HAL, memory, goals, plans, robotics workflows, capability
map, jobs, and sovereignty audit endpoints. Read `app.py` route declarations for
the canonical current list.

## Trace Contracts

- `CheckpointLoadReport` answers exactly what artifact material entered memory.
- `PromptAssemblyTrace` answers exactly what text and token allocation reached the
  model.
- `SubsystemTrace` answers which native systems ran and what they measured.
- `GenerationTrace` answers how tokens were selected, why generation stopped, and
  whether output was accepted.

When adding trace fields, use JSON-serializable stable names. Do not overload one
field with multiple units or meanings. Matrix summaries may be compact, but the
trace endpoint must retain the underlying evidence.

## Training Discipline

The five continuation phases have different objectives. `scripts/build_brain.py`
accepts `--continuation-phase A|B|C|D|E` and either
`raw_causal_shards_v1` or the conversation packing layout.

Use separate optimizer groups for existing transformer weights, corrected native
parameters, and appended tokenizer rows. Never record a fractional accumulation
as a completed optimizer step. Candidate selection uses validation and capability
scores, not train loss alone.

Checkpoint writes must be atomic and assigned to one worker. Parallel Colabs may
run independent jobs against immutable inputs, but cannot share a writable target.

## Evaluation Discipline

Tests establish implementation correctness. Evaluation establishes checkpoint
behavior. Keep the two claims separate.

The recovery suite uses exactly 200 prompts. The private promotion suite uses at
least 500 tasks, three seeds, three runtime modes, and native subsystem ablations.
Code and math use execution/verifiers, formatting uses parsers, identity uses a
semantic contract, and open-ended coherence requires blinded review.

Never print or commit the private suite. Store only hashes, aggregate results,
bounded failure samples, and review artifacts in controlled run output.

## Frontend Development

The Colab path serves the compact developer interface directly from `app.py` to
avoid a second proxy-sensitive Vite process. The React client in `phase4/web/` is
still available for local development:

```powershell
cd phase4/web
npm install
npm run dev
```

Its API routing must remain compatible with the backend. Validate both desktop
and mobile layouts and ensure long prompts, hashes, and trace payloads wrap rather
than overlap.

## Documentation and Engineering Log

Update documentation in the same change when behavior, commands, paths, schemas,
or gates change. Append significant changes with:

```powershell
python scripts/log_engineering_change.py `
  --component inference `
  --type FIX `
  --title "Describe the change" `
  --summary "What behavior changed and why" `
  --files "generate.py, tests/test_frontier_runtime.py" `
  --verify "python -m pytest tests/test_frontier_runtime.py -q" `
  --risk medium
```

Do not rewrite historical engineering-log entries.

## Commit Checklist

- [ ] Scope is one coherent change.
- [ ] No checkpoint, secret, private data, or generated output is staged.
- [ ] Architecture and schema accounting agree.
- [ ] Focused tests cover success and failure behavior.
- [ ] Ruff passes for changed modules.
- [ ] Full non-GPU suite passes, or unrun coverage is reported.
- [ ] Real GPU/Drive/checkpoint limitations are reported honestly.
- [ ] Documentation matches the actual command and runtime behavior.
- [ ] Capability claims cite evaluation evidence, not implementation effort.
