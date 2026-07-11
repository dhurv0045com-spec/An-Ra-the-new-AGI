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

## Current State (2026-07-11)

The deeper recovery pass now includes real CUDA activation evidence, not only
generation samples and static tensors. The legacy checkpoint is finite but its
output distribution is collapsed (99.974-99.9997% mean top-1), residual RMS
amplifies 19-30x through depth, router context is dormant, and two routed layers
are near-closed. Fresh training now uses depth-scaled residual initialization,
masked logit z-loss, a truly dense Phase A, isolated Phase B, explicit later
subsystem recipes, and synchronization-free hot-path router telemetry. Exact
57.4M/159.1M pilot profiles are runnable; signed pilots bind immutable
train/validation shards and unimplemented axes remain blocked.

The planned attention foundation is no longer merely asserted: parameter-free
per-head QK-Norm and a 3:1 sliding-window/full-attention pattern are part of the
checkpoint-recorded model contract. Legacy artifacts restore their old
no-QK/full-attention semantics, while scratch pilots default to the new path.
The QK-Norm-off and full-attention-only factorial cells are now executable.
A bounded RTX 4050 forward/backward on the 57,374,343-parameter pilot was
finite at 498.7 MiB peak VRAM. The generation cache gate now also compares a
compact full-distribution fingerprint, catching stale-cache faults that leave
greedy tokens, entropy, and maximum probability unchanged.

The MTP and MoE factorial labels now map to real training behavior. MTP uses
two future-token heads (+2/+3) with a recorded weighted auxiliary loss; its
57M anchor becomes 58,194,823 parameters and passed a finite RTX 4050
forward/backward. The 8-routed + 1-shared top-2 MoE starts with exact dense
function parity, performs sparse expert computation, and balances load through
optimizer-bound routing bias rather than an auxiliary loss. Total expert-bank
sizes are explicit (375,940,743 at the 50M anchor; 1,100,615,335 at 150M), so
these runs require cluster-class optimizer memory even though active FLOPs are
sparse. Checkpoints cannot silently cross dense/MTP/MoE architecture boundaries.

The three curriculum cells are also executable. A deterministic source-range
sampler applies code-first, math-ramp, or identity-late multipliers over the
exact signed token budget while preserving immutable shards and held-out data.
The curriculum is checkpoint-bound. Campaign status now reports 17/23 cells
trainer-mapped; the remaining six are exactly the three V4-dependent cells and
three evidence-gated moonshots, not hidden baseline fallbacks.

Pilot manifests now bind a tokenizer path as well as its hash. V3 and V4 cells
select distinct tokenizer artifacts and train/validation shard families, and
the signed path is inherited by the trainer process. This closes the prior
risk that generating a mixed factorial under one global tokenizer could label
a V4 cell with V3 tokens (or vice versa).

Final verification for this implementation slice: **625 passed, 1 skipped**
in 229.62 seconds; changed-file Ruff and `git diff --check` are clean.

The resume LSH representation was then compacted after the old worker spent
about 40 minutes rebuilding 500k signatures at >5 GiB working memory without
touching the corpus. The replacement stores signatures in flat uint64 memory
and singleton bands as integers; a 100k benchmark reached ~154k inserts/sec at
62.08 MiB peak. The old worker was safely replaced only after verifying the
17,500,932,024-byte corpus still matched the audit. The optimized append-only
resume is active with unbuffered logs.

The interrupted 17.50GB corpus completed its full streaming audit: 3,347,036
valid records, zero structural/hash/license/duplicate failures, but every row
is FineWeb-Edu. Safe append-only acquisition of the missing code/math/science
portion of the 30GB tranche is active. `scripts/campaign_status.py` is the
plain-language fail-closed campaign preflight.

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

The real 500M checkpoint was supplied on 2026-07-10 and frozen outside git at
`C:\Users\ankit\Downloads\anra_frontier_500m.pt` (2,000,680,247 bytes;
SHA-256 `648354a4...d393`). Its recorded source commit is an ancestor of this
branch. Schema-4-to-7 loading accounts for all target tensors with zero
missing, unexpected, or mismatched tensors; the 500 tokenizer probes match the
frozen manifest. A CUDA smoke ran on the RTX 4050 at 7.44 tokens/second and was
quality-rejected. The completed 600-generation deterministic recovery audit
(200 diagnostic, 200 native, 200 replay) found 0.0% coherence against the 80%
gate. Exact loading, finite activations, and deterministic replay passed, but
the checkpoint is conclusively undertrained and is not a recoverable serving
candidate. Artifact-specific defects and the recovery program are recorded in
`docs/engineering/CHECKPOINT_FORENSICS.md`. The campaign-slice builder
successfully proves held-out disjointness but produces only 3.40 MB from the
available source, below the mandatory 50 MB gate. Local recovery, post-training
ablation, and all-seven moonshot gates are now code-complete and fail closed;
they await their specific compute/data evidence.

The remaining F-10 architecture defect is repaired in schema 7. The per-layer
temperature multiplier is now a bounded positive log-parameter rather than an
advertised-but-inert buffer; legacy positive scales migrate by `log(scale)` and
invalid scales fail closed. Native regularization, telemetry, optimizer-group
coverage, gradient proof, and exact parameter contracts are tested. The current
candidate has 499,167,075 parameters (28 more trainable scalars than the legacy
artifact); it still requires the registered three-seed ablation before campaign
selection. Forensic publication now refuses to overwrite an executed recovery
gate with a skipped structural report. The schema-7 CUDA replay completed and
refreshed the canonical artifact: exact structure and deterministic replay pass,
but coherence/acceptance remain 0.0% and EOS failure is 100%. The current
contract, tokenizer probes, corpus manifests, and unchanged checkpoint hash are
frozen in `output/v2/baseline_freeze.json` (config contract SHA-256
`15321a16...c14215`).

The training/evaluation loss contract is also repaired. Conversational data
now carries an explicit answer-token mask instead of inferring answers from a
numeric weight threshold. GPU, TPU, draft, validation, checkpoint, and runtime
metadata paths distinguish total, weighted, answer-only, and scaffold-only CE
with exact denominators. This closes the remaining path by which easy prompt
scaffolding could make validation look healthy while answers remained poor.

Conversational validation itself is now held out. The former D/E `eval_ds = ds`
path is replaced by deterministic whole-content-group assignment before
tokenization, distinct dataset objects, and a hashed zero-overlap split
manifest shared by GPU/TPU paths. Validation is stratified by raw source class
or conversational bucket. Stage promotion computes same-identity, newer
baseline/candidate regressions per protected domain (including answer loss in
conversational stages) and fails closed on missing/reused/operator-asserted
evidence.

**Test baseline:** 597 non-GPU tests passing, 1 skipped. `ruff` clean on all
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

**Do not continue the failed legacy checkpoint; proceed only through the named
fresh-training evidence path.** Acquire the required campaign corpus, then run
the pre-registered three-seed pilot/scratch control, measured throughput, and
actual kill-9 recovery; execute the completed
control-plane telemetry, chaos, live 24-hour soak, and five-preemption gates
against Drive and authorized workers; then use a real 50-goal suite, latency
budget data, 20-scenario UI study, and adversarial/canary release evidence
before checking the remaining P2 boxes.

**Owner actions that unblock the rest of Streams A and B:**
1. Set `ANRA_MANIFEST_SIGNING_KEY`, then emit the signed pilot manifests:
   `py -3.14 -m training.pilot_factorial --owner-authorized`.
2. Acquire the corpus, then build the canonical V4:
   `py -3.14 scripts/download_training_data.py --profile 30gb` →
   `py -3.14 scripts/build_campaign_slice.py` (>=50MB slice) →
   `py -3.14 scripts/build_v4_tokenizer.py` (canonical 32k V4) → run the
   pre-registered `p150-v4tok` 150M three-seed pilot on GPU.

## Blocking Dependencies (not solvable in the code environment)

- **Training compute.** Recommended: rent one A100 for ~$60–120 to train 10B
  tokens (the single highest-leverage action for this project). Fallback:
  Colab TPU v2-8 fleet (free, ~3–5 sessions for Phase A).
- **Owner signing authority and live external evidence.** The checkpoint is
  now present outside git; pilot signing, GPU campaigns, live soak/preemption,
  production canary, usability, and independent evaluation evidence remain.

## Standing Rules

- Two laws (MASTER_PLAN Part 0): lineage/tokenizer/IDs never silently touched;
  nothing "improved" without ablation evidence, nothing acts without the
  sovereignty gate.
- Every slice: tests + full suite green + engineering-log entry + update this
  file's "Next Action".
