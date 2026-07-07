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

## Current State (2026-07-07)

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

**Test baseline:** 518 non-GPU tests passing, 1 skipped. `ruff` clean on all
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

## Next Action (start here)

**Stream E - ledger-derived transparency projections.** Define the UI-facing
summary contract for verification, memory, and gate visibility from replay-safe
Experience Ledger events. Keep raw prompt/memory content out of the projection;
surface trace IDs, event kinds, gate/verifier outcomes, memory record IDs,
provenance, and hashes. (The forecast-ledger schema + pre-launch timestamp
audit box on Stream E is already delivered.)

**Owner actions that unblock the rest of Stream A:**
1. Restore the real checkpoint (or set `ANRA_CHECKPOINT_PATH`), then run
   `py -3.14 scripts/freeze_baseline_hashes.py` and
   `py -3.14 scripts/run_checkpoint_forensics.py --run-generation` — this
   completes tensor accounting, the recovery gate, and the Part 0
   undertraining decision in one pass.
2. Set `ANRA_MANIFEST_SIGNING_KEY`, then emit the signed pilot manifests:
   `py -3.14 -m training.pilot_factorial --owner-authorized`.

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
