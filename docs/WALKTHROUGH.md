# An-Ra Operator Walkthrough

This guide covers the shortest path from a trained checkpoint to a traceable
conversation, then the separate path for continuing training.

## Choose Your Path

| Situation | Action |
| --- | --- |
| The checkpoint is already in Drive and you want to chat | Run T4 notebook **Cell 10 only** |
| You want a command-line smoke test in Colab | Run Cell 8 after the repository and Drive are available |
| You want the older UI launcher in an already prepared runtime | Run Cell 9 |
| You need to download data or train | Start at Cell 2 and follow the training path below |
| You are working locally | Use the local runtime section |

## One-Cell Chat and Matrix

Open `notebooks/AN_RA_T4_TRAINING.ipynb` in a Colab T4 runtime. Confirm that
Google Drive contains:

```text
MyDrive/AnRa/v2/checkpoints/anra_frontier_500m.pt
```

Then run **Cell 10 only**. Do not run Cells 2-9 first unless you intend to train
or inspect their individual steps.

Cell 10 will:

1. Mount Google Drive.
2. Clone `iterate500`, or fetch and fast-forward an existing clone.
3. Run the Colab bootstrap.
4. Restore the frontier checkpoint from known Drive locations.
5. Stop with an explicit error if no checkpoint is found.
6. Run checkpoint proof and tokenizer compatibility checks.
7. Start `app.py` on port `5173`.
8. Wait for `/status` to report readiness.
9. Open `/developer` through the official Colab proxy.

It does not invoke `scripts/build_brain.py` and cannot begin training.

### What success looks like

The cell prints a checkpoint path and size, then a backend status containing
fields such as:

```text
service_status
bundle_status
quality_status
device
param_count
```

It displays an **Open An-Ra Developer UI** link and an embedded interface. Use
the external link if the iframe is constrained by the notebook viewport.

## Dashboard: Talk to the Checkpoint

The Dashboard sends a validated `/chat` request. Start with deterministic,
checkable prompts rather than open-ended conversation:

```text
Return exactly three numbered steps for debugging a failed Python import.
```

```text
Differentiate x^2 + 3*x. Return only the expression.
```

```text
Repeat this key exactly once: cobalt-19.
```

Use the runtime modes deliberately:

- **Diagnostic** establishes the model-only recovery baseline and avoids
  committing adaptive state.
- **Native** enables MoD, RIM, DSTP, ESV, and HAL execution.
- **Full system** also enables memory, ghost context, cognition, explicit agent
  goals, and safe tool dispatch.

Compare the same prompt in diagnostic and native modes before changing sampling.
Greedy, seed `0`, 128 output tokens, and cache off is the reference configuration.

## Matrix: Read What Happened

After a response, open Matrix and inspect these fields in order:

1. **Checkpoint proof** - exact path, SHA-256, schema, loaded tensors, migrations,
   missing tensors, mismatches, and initialization.
2. **Tokenizer proof** - vocabulary and special-token hashes, schema, and probe
   compatibility.
3. **Formatted prompt** - the exact string sent to the tokenizer.
4. **Token allocation** - identity, current message, history, memory, reserved
   generation space, and truncation decisions.
5. **Generation** - mode, prompt/output token counts, duration, stop reason,
   repetition and fragment flags, and quality state.
6. **Subsystems** - whether MoD, RIM, DSTP, ESV, and HAL executed, plus their
   telemetry where available.
7. **Persistence** - whether session history, ESV, HAL, ghost state, and memory
   were committed. Rejected output must not update persistent state.

The API response contains `trace_id`; the complete payload is also available at:

```text
GET /traces/{trace_id}
```

## Run Evidence Gates

The Matrix buttons are ordered from cheap structural proof to expensive behavior
evaluation.

### 1. Rollback drill

Run this first. It verifies signed rollback material and artifact restoration.
A failed drill blocks release but does not prevent ordinary diagnostic chat.

### 2. 200-prompt gate

This runs exactly 200 prompts in diagnostic and native modes, then replays native
generation deterministically. It checks:

- finite probability telemetry;
- deterministic output-token replay;
- at least 80% candidate coherence;
- repetition and EOS behavior.

If coherence is below 80%, the report labels the primary failure as
`undertraining`. Do not try to hide that result with more sampling randomness.

### 3. Integration probe

This verifies execution and connectivity across:

- checkpoint-backed model generation;
- all native model subsystems;
- ghost state without evaluation-time persistence;
- memory write/search;
- verifier execution;
- cognition health;
- agent and safe tool paths;
- capability graph construction.

### 4. Full promotion evaluation

This is not a smoke test. It uses at least 500 private tasks across diagnostic,
native, and full-system modes, three seeds, and five subsystem ablations. It can
take hours on a T4. Keep the backend cell alive and use **Review outputs** when it
finishes to complete blinded human review items.

## Read the Result Correctly

| Observation | Meaning | Next action |
| --- | --- | --- |
| UI starts, checkpoint proof fails | Artifact incompatibility | Stop; repair or use a compatible checkpoint |
| Proof passes, output is fragmented | Behavioral failure, commonly undertraining | Run recovery gate, then continuation training |
| Diagnostic works, native regresses | Native subsystem defect or insufficient staging | Inspect traces and ablations |
| Native works, full system regresses | Memory/agent/cognition integration issue | Run integration probe and isolate the failing path |
| Training loss is low, chat is incoherent | Loss is not measuring usable response quality | Examine data objective, validation, and private tasks |
| Cache parity fails | Incremental positions or cache state differ | Keep cache disabled |
| Response is rejected | Quality guard fired | Inspect stop reason; state should not persist |

## Training Path

Only use this path when you intend to continue training.

1. **Cell 2** mounts Drive.
2. **Cell 3** clones or updates `iterate500` and bootstraps the runtime.
3. **Cell 4** restores and verifies the checkpoint lineage.
4. **Cell 5** prepares the selected immutable corpus profile. Default: `30gb`.
5. **Cell 6** runs a 180-minute Phase A raw-causal continuation session.
6. **Cell 7** shows ThirdEye evidence separately from loss.
7. **Cell 8** proves and smoke-chats the resulting checkpoint.

Before Cell 6, inspect:

```python
SESSION_MINUTES = 180
CONTINUATION_PHASE = 'A'
```

Do not change continuation phase casually. Phase behavior is encoded in the
trainer and documented in [IMPROVEMENT.md](IMPROVEMENT.md).

Never run two workers against the same writable checkpoint. For multiple Colabs,
give each job a unique checkpoint/output path and promote one candidate only
after evaluation.

## Local Runtime

Install and validate:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
$env:ANRA_MODEL_PROFILE = "frontier"
$env:ANRA_CHECKPOINT_PATH = "C:\path\to\anra_frontier_500m.pt"
python scripts/check_frontier_checkpoint.py --checkpoint $env:ANRA_CHECKPOINT_PATH
python app.py --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/developer`.

Terminal-only smoke test:

```powershell
python scripts/chat_frontier.py `
  --checkpoint $env:ANRA_CHECKPOINT_PATH `
  --suite smoke `
  --output output/v2/frontier_chat_traces.jsonl
```

## Troubleshooting

### Checkpoint not found

Confirm the exact Drive path and filename. Cell 10 will not train a replacement.
If Drive is mounted but the checkpoint lives elsewhere, copy it into the expected
checkpoint directory or update the path in your private notebook session.

### Blank Colab page

Use the link printed by Cell 10 rather than manually constructing a `colab.dev`
URL. Confirm the cell still runs and inspect:

```text
/content/anra_developer_ui.log
```

Re-running Cell 10 terminates the existing port-5173 process and starts a clean
backend.

### Backend exits during startup

The cell includes the last backend log lines in the raised error. Typical causes
are a malformed checkpoint, tokenizer mismatch, missing dependency, or GPU memory
failure. Fix the reported cause; do not bypass checkpoint proof.

### Output is gibberish

Use diagnostic greedy mode and run the 200-prompt gate. Check the exact checkpoint
hash, tokenizer hash, prompt, stop reason, and fragment/repetition flags. If the
load is exact and coherence remains below 80%, treat the model as undertrained and
continue the gated curriculum.

### UI works but Matrix is empty

Send one Dashboard message first. Matrix detail is keyed by the returned trace.
For release evidence, run the corresponding Matrix action and refresh the view.

## Artifact Locations

```text
output/v2/frontier_checkpoint_proof.json
output/v2/frontier_chat_traces.jsonl
output/v2/recovery_gate.json
output/v2/full_system_integration.json
output/v2/private_promotion_eval.json
output/v2/private_promotion_progress.json
output/v2/rollback_drill.json
output/v2/release_bundle.json
```

These are run artifacts, not source documentation. Keep checkpoint, tokenizer,
corpus, code commit, and evaluation hashes together when comparing experiments.
