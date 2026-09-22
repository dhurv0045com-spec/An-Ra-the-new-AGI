# Gandiva cognition and RSI implementation

This report describes the implementation included in source revision
`bd6bfb091b68679fcfdd9f544384b3b8cbd9c0f1`. It records an executable research
system and test evidence, not an AGI result.

## Cognition path added to E2

E2 now constructs a frozen exemplar index from the exact `training` pool. It
keeps one public example/answer per mechanism, marks every record as
`training`, and hashes the complete index and retrieval rule. Evaluation and
confirmation pools are not loaded into this index.

At each policy decision, the episode kernel queries the index using the
public goal, observed history, and workspace. Scope filtering happens before
ranking; only `training` and `controller` are eligible, and the current
mechanism ID is excluded. Complete memory records enter the same public-state
renderer as goal, history, workspace, and remaining budgets. The prompt gets
stable episode-local hashed aliases instead of record IDs or provenance.
Retrieval cost uses the repository's actual byte-token codec. The renderer
reserves context and refuses silent record slicing; if a full memory item
cannot fit, it is dropped whole and the trace records the retrieval decision.

The `b-memory` treatment uses the same frozen B model, generated mechanism,
environment, and seed as `b-policy`. The memory index is the only treatment
difference. The trace records index/rule identities, per-decision hashed
record aliases, retrieval counts, input-token counts, and actual memory token
cost. Focused tests prove that an eligible training exemplar reaches the
prompt, sealed text does not, and the matched E2 arm reads nonzero training
records.

This is a deterministic lexical retrieval baseline with one exemplar per
training mechanism. It does not learn an embedding, write back new memories,
or establish that retrieval improves generalization. The owner experiment
must compare its held-out success, cost, and truncation against `b-policy`.

## RSI and architecture evidence path

E5 method programs now match the support labels actually present. Current
support data carries answer-token supervision only, so proposals that assign
positive weight to unsupported world, action, value, or pair terms are
rejected instead of receiving credit for objectives that never execute.
The declared methods are:

- **M0:** token objective weight 1.0, baseline learning rate and clipping.
- **M1:** token objective weight 0.8 and half learning rate. This is a bundled
  intervention; its result cannot identify the isolated causal effect of
  either change.
- **M2:** token objective weight 1.0, clip norm 0.5, and enabled gated
  architecture.

Dispatch verifies each compiled recipe and applies the actual trainer state.
The proposer and successor captures are decoded independently and checked
against exact checkpoint payload identities and the restorable checkpoint
registry. P0 is anchored to its parent; P1 and P_fixed are both children of
P0. E5 archives the P0-to-successor chain as generation receipts with
predecessor linkage. Fixture receipts remain fixture-only and cannot qualify
as learned campaign or promotion evidence.

Gated architecture qualification probes are isolated from the training RNG
stream. Migration checks preserve device state, model train/eval modes, and
pre-existing gradients. This reduces the risk that architecture checks alter
later training or sampler behavior.

## Verification and scientific limits

Fresh command against the canonical generated data bundle:

```powershell
python -m bramastra_lab.research.campaigns.k8 verify-build --data .codex-test-tmp-gandiva-20260923-data --report-dir engineering/reports/FINAL_K8/gandiva-cognition-final-20260923 --no-updates
```

Result: **VERIFIED**, F01–F24 all pass, integrated rehearsal passes, zero
local optimizer updates, 130.563 seconds measured. Source closure:
`cb005c2cbb3a02da888a5cf271805b6751954e403cbee1de6794ba82482963b7`.
Data identity: `6fb94b7018406632b0e62dcd23ca777046ae5d881363bb8e8c78d74139785fd6`.
The evidence-bound readiness package is in `BUILD_READINESS.json` and
`HANDOFF.md`; the detailed receipts are in `build_verification.json`.

Focused checks also passed:

- `python -m pytest tests/test_research_gandiva_rsi_cognition.py -q --maxfail=1` — 6 passed.
- `python -m pytest tests/test_research_k8.py -k GatedArchitecture -q` — 3 passed.
- `python -m pytest tests/test_research_k8_operational.py -k 'E5 or O09' -q` — 3 passed.
- `python -m pytest tests/test_research_master_m01_m04.py -k Memory -q` — 3 passed.
- `python -m pytest tests/test_research_meta_rsi.py -k 'method or origin' -q` — 9 passed.
- `python -m pytest tests/test_research_k8_launch_gate.py tests/test_research_k8_readiness.py tests/test_research_k8_operational.py -q --maxfail=1` — 48 passed, 12 subtests passed.

The seven build groups reported 75+3 skipped (foundation), 34 (data and
splits), 65 (experience/codec), 39+1 skipped (checkpoint/ledger), 10
(gate/readiness), 91+12 subtests (phase contracts), and 53
(statistics/evaluation). G01–G04 remain live E0 checks for two actual T4s,
full-profile update/resume, timing sufficiency, and the current allocation.
No GPU training, optimizer update, learned result, or evidence of AGI/RSI
success is claimed. The next scientific step is the owner-run normal notebook
`notebooks/bramastra_k8.ipynb`; it should stop if any E0 gate fails and use the
matched E2 arms and E5 confirmations to decide whether the mechanisms help.
