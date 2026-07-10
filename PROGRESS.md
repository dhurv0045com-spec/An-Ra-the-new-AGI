# AN-RA Progress Log

Cross-session resume anchor. Read `docs/planning/MASTER_UPGRADE.md` (**v3,
FINAL — the Unified Intelligence Program**: five parallel workstreams, the
Experience Ledger spine, moonshot registry, 12-week campaign) for the
blueprint now being implemented, `docs/planning/MASTER_PLAN.md` for the
underlying stage plan, and `docs/IMPROVEMENT.md` for the gated recovery
sequence. The plan is frozen; sessions now execute workstream slices.
Training execution runs through the GPU-cluster control plane (companion doc:
`docs/planning/CLUSTER_CONTROL_PLANE.md`, adopted as MASTER_UPGRADE Layer
12-B after two-repo code inspection 2026-07-06; cluster P0 security fixes are
the first infrastructure slice). On a fresh
session, read this file, then continue from "Next Action".

---

## Current State (2026-07-10)

**Where we are in the plan:** Foundation hardening continues to move. MASTER_UPGRADE v2
Week 1 tokenizer slice is executed: per-source fertility measured with evidence
(`output/v2/fertility_week1.json`), V3 tax confirmed (English 2.518 tok/word vs
1.35 gate), append-only V4 draft built and validated held-out
(`tokenizer/tokenizer_v4_draft.json`, DFC −30.5%, English −0.2% — local corpus
cannot fix prose fertility; canonical V4 needs the ≥50 MB campaign corpus).
The runtime substrate now has ledgered generation/tool/gate/verifier traces,
shared verifier dispatch, hardened code verification, shared dense+BM25
retrieval, memory lifecycle trace binding, sealed ledger shard manifests,
ledger stress/overhead CI, and the sibling `gpu cluster` P0/P1 control-plane
security/storage pass. Stage 1 (Make It Speak) remains gated on the real
checkpoint + training compute.

P1's local runtime foundations are now complete: shared-retriever recall CI,
hash-only proof-carrying answer contracts, retrieval-context injection
filtering, and deterministic continuous-batching/paged-KV primitives. P2
now has a content-addressed adapter hot-load registry, a fail-closed
plan-act-verify runner with a 50-case harness, and API-exposed ledger-only
trust projections. This does not claim campaign, latency, soak, usability, or
real-goal gates have run; the live TODO distinguishes its code/test completion
from the still-required external evidence.

The accumulated implementation was re-audited on 2026-07-10. Semantic answer
contracts now reject internally contradictory re-signed payloads; verified
latency includes decode overhead and requires a comparator cohort;
post-training evidence must beat its ablation; and irreversible plan actions
must pass a named authorization verifier before the action is called. Large
adapter hashes are streamed, ghost memory is offline-deterministic by default,
and full-system repository discovery excludes runtime data, virtual
environments, and caches. These corrections close local fail-open and
unbounded-scan paths; they do not substitute for external campaign evidence.

P2's remaining local mechanisms are exercised: accelerator promotion requires
speculative benefit, token/distribution parity, <=1% QAT drift, and latency
evidence; DPO has a reference-policy objective; GEPA has a ten-cycle
proposal-only evidence runner requiring a rejection; the developer UI renders
ledger-only trust evidence; and canary/adversarial/rollback/bundle drill gates
fail closed. These are deterministic local tests, not a real campaign,
production canary, or human study.

The linked `gpu cluster` control plane now also has its P2/P3/P4 local mechanism
layer: append-only heartbeat samples derive throughput from observed counter
deltas; an operator-only dashboard exposes worker telemetry, incidents, quota,
and verified artifacts; pause/drain/halt/resume and replacement-worker actions
are audit-recorded and preserve lease fencing. G-C1 chaos, G-C2 soak, and G-C3
preemption evaluators fail closed unless every required timestamped, **live**
evidence record is present; a local simulation is never a campaign pass. The
P4 boundary now rejects operator-supplied boolean gates and requires An-Ra's
signed checkpoint-bound Gate-6, bit-exact reproducibility, adversarial-audit,
and rollback-drill evidence. Rollback atomically restores and validates the
previous full checkpoint before changing the active release. The cluster
changes pass the full 27-test suite and ruff, but there is no claim of a
Drive-backed chaos run, 24-hour soak, or five real Colab preemptions.

The Stream-B 30 GB first-tranche acquisition was started as one managed
background process on 2026-07-10 after a 313 GB-free-disk preflight. Its
standard streaming client had to be installed and the downloader was corrected
to remove the obsolete/unsafe `trust_remote_code` parameter required by
`datasets` 5; the focused compatibility test passes. The transfer is not
credited until its own completion status and immutable shard/manifest checks
pass; 30 GB is also only the first tranche toward the master plan's 120 GB
target.

The moonshot executor now runs every local-safe M1-M7 path and keeps smoke
evidence separate from acceptance evidence. The 2026-07-10 execution wrote
`output/v2/moonshot_local_execution.json` and
`output/v2/moonshot_pilot_status.json`: all seven local paths passed their
smoke checks. M6 also passed its real bounded-domain pilot, classifying all
100 fixed proof cases correctly (50 valid chains and 50 adversarial injected
conclusions), so M6 is checked. M1-M5 and M7 remain blocked on their exact
training, benchmark, data, or human-sovereignty evidence.

The remaining TODO executor was run against local artifacts on 2026-07-10.
Checkpoint forensics again reports `blocked` because the real 500M checkpoint
is absent; its 500 tokenizer probes still pass. The campaign-slice builder
successfully proves held-out disjointness but produces only 3.40 MB from the
available source, below the mandatory 50 MB gate. Local recovery, post-training
ablation, and all-seven moonshot gates are now code-complete and fail closed;
they await their specific compute/data evidence.

**Test baseline:** 581 non-GPU tests passing, 1 skipped. `ruff` clean on all
changed files. Full suite command:
```
py -3.14 -m pytest tests/ -m "not gpu" \
  --ignore=tests/test_drive_session_manager_integration.py \
  --ignore=tests/test_v2_drive_artifacts.py -q
```

## Done (this session — all tested, all in engineering log)

Evidence integrity + subsystem revival + DFC + KV gate. See
`docs/engineering/ENGINEERING_LOG.md` entries dated 2026-07-03/04 for the full
list. Headline changes:
- Deterministic evaluation; honest metrics/health/capability/telemetry surfaces.
- Ghost memory, identity injector, CIV, cognition, HAL all revived into the
  live `/chat` path with measured evidence (was: dead imports / constants).
- Symbolic falsification pass (DFC) live on math/logic; feeds HAL/CIV.
- KV parity gate strengthened to distribution level with fault-injection tests.

## Done (2026-07-06 session)

- MASTER_UPGRADE.md rebuilt to v2 (the 1000× program: pilot-science ladder,
  MoE sparse upcycling, serving stack, eval science, 12-week calendar, risk
  register). PROGRESS/plan pointers updated.
- Week 1 fertility slice: `scripts/measure_tokenizer_fertility.py` (+4 tests)
  measures per-source fertility against the V4 gates and runs the 1M-unit
  append audit. Evidence: English 2.518 tok/word (gate 1.35 — V3 tax
  confirmed), audit projected reduction 21.2% (eligible). V4 draft built
  (`tokenizer_v4_draft.json`): byte-safe, fixes V3's round-trip defect,
  DFC −30.5% held-out, but English −0.2% — proving canonical V4 candidates
  must come from the full ≥50 MB campaign corpus. Engineering log updated.

## Implementation Start (2026-07-06)

- S1 Experience Ledger is live: schema-versioned hashed JSONL events,
  validated replay, fail-open capture, chat/generate/tool/verifier/gate
  integration, and verifier-gated PII-filtered training compaction with atomic
  hash manifests.
- S2 verifier registry is live and routed through DFC, synthetic validation,
  agent critique, GEPA, and compatibility reward calls.
- S3 retrieval substrate is live with hybrid dense+BM25 fusion, provenance, and
  integrations across memory, citation grounding, agent skills, and corpus dedup.
- Memory lifecycle operations now emit replayable Experience Ledger events for
  writes, recalls, edits, and forgetting under the originating trace ID.
- Stream C is complete at code + focused-test level: Experience Ledger shards
  rotate, seal, verify, and prune by retention; CI has a p50/p99 write benchmark
  plus crash/flush stress probe; the sibling `gpu cluster` repo has P0 auth/
  worker hardening and P1 three-tier quota-ledger accounting.
- The executable campaign board is `docs/planning/IMPLEMENTATION_TODOS.md`.

## Done (2026-07-07 session — Stream A executable half)

- Forecast ledger live (`training/forecast_ledger.py`): hash-chained
  append-only predictions, outcomes, calibration report, and the Gate-5
  `audit_pre_launch` timestamp audit (post-hoc forecasts void the launch).
- Pilot factorial pre-registered (`training/pilot_factorial.py`): 23 cells
  (Muon/MoE/MTP/QK-Norm/SWA/V4 + interactions, 50M ladder, curriculum order,
  moonshots M1/M3/M5) with honest prediction ranges; one signed manifest per
  cell, three seeds each, forecast registered strictly before the manifest.
- Baseline freeze executed (`scripts/freeze_baseline_hashes.py` →
  `output/v2/baseline_freeze.json`): tokenizer identity + live 500-probe
  fingerprint (matches frozen manifest), config contract hash, corpus
  manifest hashes; checkpoint slot honestly `blocked_on_artifact`.
- Forensics driver ready (`scripts/run_checkpoint_forensics.py`): locator,
  exact tensor accounting, probes, deterministic 200-prompt recovery gate,
  undertraining decision rule; executed → blocked on the artifact (exit 3).
- 18 new focused tests; suite 518 passed / 1 skipped; ruff clean.

## Done (2026-07-07 session — Stream B executable half)

- Canonical 32k V4 append migration generalized and proven, Law-1-clean and
  non-destructive: `CANONICAL_V4_VOCAB_SIZE = 32_768` with a pinned
  530,602,567-param contract; `build_append_only_v4`/`audit_token_fertility`
  take a `target_vocab_size` ceiling; the frozen 8,209-token V3 prefix is
  asserted before every V4 write; 16,384 retained as the proven fallback.
- All vocab gates (runtime, checkpoint validator, ssg) accept {8209, 16384,
  32768} via `is_v4_vocab_size`.
- Pinned, license-checked upstream corpus manifests: `training/corpus_manifest.py`
  → `output/v2/data_manifests/upstream_corpus_manifest.json` (7 sources,
  allowlisted licenses, immutable revisions, weights sum 1.0, content-hashed).
- Campaign-slice builder (`scripts/build_campaign_slice.py`, deterministic
  held-out source splits, >=50MB gate) and 32k V4 build CLI
  (`scripts/build_v4_tokenizer.py`, self-proving) shipped and executed on the
  local corpus (honestly ineligible: needs the >=50MB campaign corpus).
- 24 new focused tests; suite 542 passed / 1 skipped; ruff clean.

## Next Action (start here)

**P1/P2 campaign and cluster proof gates.** Run measured throughput and actual
kill-9 recovery once the real checkpoint is restored; execute the completed
control-plane telemetry, chaos, live 24-hour soak, and five-preemption gates
against Drive and authorized workers; then use a real 50-goal suite, latency
budget data, 20-scenario UI study, and adversarial/canary release evidence
before checking the remaining P2 boxes.

**Owner actions that unblock the rest of Streams A and B:**
1. Restore the real checkpoint (or set `ANRA_CHECKPOINT_PATH`), then run
   `py -3.14 scripts/freeze_baseline_hashes.py` and
   `py -3.14 scripts/run_checkpoint_forensics.py --run-generation` — this
   completes tensor accounting, the recovery gate, and the Part 0
   undertraining decision in one pass.
2. Set `ANRA_MANIFEST_SIGNING_KEY`, then emit the signed pilot manifests:
   `py -3.14 -m training.pilot_factorial --owner-authorized`.
3. Acquire the corpus, then build the canonical V4:
   `py -3.14 scripts/download_training_data.py --profile 30gb` →
   `py -3.14 scripts/build_campaign_slice.py` (>=50MB slice) →
   `py -3.14 scripts/build_v4_tokenizer.py` (canonical 32k V4) → run the
   pre-registered `p150-v4tok` 150M three-seed pilot on GPU.

## Blocking Dependencies (not solvable in the code environment)

- **Training compute.** Recommended: rent one A100 for ~$60–120 to train 10B
  tokens (the single highest-leverage action for this project). Fallback:
  Colab TPU v2-8 fleet (free, ~3–5 sessions for Phase A).
- **The real checkpoint file** (kept outside git per repo policy).

## Standing Rules

- Two laws (MASTER_PLAN Part 0): lineage/tokenizer/IDs never silently touched;
  nothing "improved" without ablation evidence, nothing acts without the
  sovereignty gate.
- Every slice: tests + full suite green + engineering-log entry + update this
  file's "Next Action".
