# CYMEK_SYNC.md

Synchronization record for the Cymek-first research phase. Audited: 2026-09-03 (~23:45 local).
All Cymek-derived claims in Citadel documents are tied to the SHA below from this point forward.

## Recorded SHAs (after `git fetch --all --prune`)

| Ref | SHA | Note |
|---|---|---|
| `origin/cymek` | **`26a61f6242e9e5c1d1b028b4f8c3c7d26ac0fdc6`** | authoritative audit target |
| `citadel` | `79832db881c492b5aa5e359d76e21a37d73f7a93` | pushed to `origin/citadel` |
| `origin/esoes` | `85f44b7b449f2ee39a0e80203a2d7df04614983b` | unchanged since bootstrap |
| `origin/triquetra` | `fa44ea3b6b279d7a7072122765b93facc0c755ff` | unchanged |
| merge-base(citadel, origin/cymek) | `85f44b7b449f2ee39a0e80203a2d7df04614983b` | the esoes tip |

**FACT:** the previous documentation SHA (`92dcd56`, audited at Citadel bootstrap) is stale.
`origin/cymek` moved +8 commits (~4,760 lines changed, 69 files):

```
9775dc9  cymek: audit production V5 execution gaps and freeze launch criteria
dfb64c7  cymek: implement canonical production training backend with real mutation certification
ef750c3  cymek: implement true multi-segment packing, cursor microbatches, and data manifests
12b8548  cymek: repair promotion fail-closed gates and add checkpoint-backed evaluation with truth firewall
351c7e6  cymek: end-to-end miniature through the real production path with receipt
32ee826  cymek: P35 and bounded V5-A canaries through the certified production path on local CUDA
6a2b64b  cymek: add evidence ledger, cymek CI, exact-HEAD test receipt, and CLI entries
26a61f6  cymek: bind test receipt to audited head
```

## Delta audit summary (full report reflected in CYMEK_EXECUTION_GRAPH.md)

- **Real training chain now exists and executed:** `ProductionTrainingBackend` (mechanical
  before/after mutation certification, adversarial tests), true multi-segment packing, manifest
  builder, checkpoint-backed evaluation adapter, answer-blind firewall, repaired fail-closed
  promotion gates. Executed receipts: end-to-end miniature (4 real updates, 16,384 tokens,
  1.65M-param model), P35 CUDA canary (3 updates, 12,288 tokens, declining loss
  10.166→9.536, RTX 4050, torch 2.14.0+cu126), V5-A bounded canary (2 updates, 1,024 tokens,
  exact 250,216,960 params).
- **The executed data source is Cymek's own repository** (48 tracked text files, ~74K tokens).
  **Zero `verified_cognition` tokens have ever flowed through the production path.**
- **Bypassed in every executed run:** `v5_data.batch.microbatch`, `v5_data.pack.sampler_order`,
  mixture enforcement (tests-only code). Bounded-warmup constant LR used, not the frozen WSD.
- **Still missing:** corpus loader, external corpus, cognition-generator wiring, real manifests,
  distributed collectives, remote execution, sealed fixtures, promotion dossier evidence.
- **Receipt hygiene holes found:** miniature receipt lacks device/wall-time/timestamp;
  `source_commit` names a tree that does not contain the committed receipt; Cymek's own
  `evidence_ledger.json` quotes stale numbers from a superseded run (47 vs 48 documents,
  94.5% vs 96.7% pack efficiency, 68.7/31.3 vs 49.7/50.3 mixture split) and claims
  cursor-microbatch execution that the code contradicts (tests-only).
  This is the same stale-green pattern Citadel documented on triquetra (N10/N17): **receipts
  within a branch can contradict each other; the committed execution artifact wins.**

## Ancestry rule (unchanged)

Citadel still descends only from `origin/esoes`. No merge, rebase, or reset toward Cymek.
Cymek code is used at runtime from a detached read-only worktree pinned to the SHA above, with
module provenance recorded per receipt — no invisible copying.

## Runtime audit environment

- Cymek worktree: `C:\Users\ankit\.zcode\tmp\audit-cymek2` (detached at `26a61f6`)
- Citadel venv: `.venv` (Python 3.11.15, torch CPU wheel, tokenizers, numpy) — CUDA torch
  environment used by Cymek's own canaries (torch 2.14.0+cu126, RTX 4050) no longer exists on
  this machine and must be reinstalled before any GPU experiment.

## Sync record 2026-09-04 — Kaggle-TPU mission (§1)

**FACT** (after `git fetch --all --prune`, from the `An-Ra-the-new-AGI-1` clone):

| Ref | SHA | Note |
|---|---|---|
| `citadel` (local worktree) | `203ff60` | TPU-first audit + T0/T1 preregistration (unpushed) |
| `origin/citadel` | `f3d3ba8` | unchanged since prior mission |
| `origin/cymek` | **`4abeaeb`** | unchanged since `git pull` of 2026-09-04 |
| `origin/esoes` | `85f44b7` | unchanged foundation |
| merge-base(citadel, origin/cymek) | `85f44b7` | siblings under esoes; no merge performed |

**FACT:** since the pinned audit SHA `26a61f6` recorded above, `origin/cymek`
moved +3 commits (`95dbc97` causal-eval/RNG/sampler repair, `674c1ca`
CoreSubjectManifest/registry/eval-protocol, `4abeaeb` Triquetra handshake bind).
Nothing in those 3 commits invalidates the static XLA audit surface
(`v5_model/`, `v5_objectives/`, `v5_training/optimizer.py`,
`v5_data/pack.py` semantics unchanged for the miniature path); the delta is
evaluation/registry/handshake scaffolding. Full re-audit is deferred until the
first TPU device receipt exists.

**FACT:** `MINI_SPEC` (`anra_v5/miniature_run.py`: 2L×64, 4Q/2KV, d=16,
FFN 128, vocab 24,576) confirmed present at `4abeaeb` — first-TPU model per
mission §3. P35 is NOT the TPU bring-up model. No calculator prior art exists
on any branch (case-insensitive grep for "calculator" empty) — canary is
greenfield (`citadel_tpu/calculator_data.py`, `calculator-canary/1.0`).

**Execution-target reclassification (§2, binding):** CPU = debugging/reference;
CUDA (`cymek_receipts/ONE_UPDATE.json` etc.) = previous certification/reference,
preserved, certifies nothing about XLA; TPU/XLA = current target.
`TRAINING_PATH_READY` is false until demonstrated on the target TPU.
