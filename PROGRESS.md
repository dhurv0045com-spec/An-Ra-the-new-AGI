# AN-RA Progress Log

Cross-session resume anchor. Read `TODO.md` for the short unfinished
work list and `docs/engineering/V4_ARCHITECTURE_GATE.md` for the governing
architecture and evidence contract. Superseded long-form plans were removed.
Training execution runs through the GPU-cluster control plane (companion doc:
`docs/planning/CLUSTER_CONTROL_PLANE.md`, adopted as MASTER_UPGRADE Layer
12-B after two-repo code inspection 2026-07-06; cluster P0 security fixes are
the first infrastructure slice). On a fresh
session, read this file, then continue from "Next Action".

---

## Current State (2026-07-20)

**2026-07-20 V4 data-publication update.** The interrupted append journal was
reconciled without rescanning the verified 28.99 GB prefix. A Windows text-mode
defect had written CRLF while recording LF byte lengths, producing exactly one
missing index byte for each of 411,000 appended records; recovery now migrates
only that provable shape and all future appends write encoded bytes directly.
The resulting 31,160,180,241-byte corpus contains 5,685,479 records, all four
native source classes, and zero published audit failures. The V4 tokenizer's
quadratic longest-piece search was replaced by an exact prefix trie (250-sample
parity; 32 MiB probe: 2.02M tok/s), and immutable shard publication no longer
rebuilds an unbounded BM25 dedup index after the corpus MinHash audit.

Publication produced 11,423,800,574 train tokens, 115,134,145 validation
tokens, and 117,260,614 test tokens. Its first manifest correctly failed closed
because verified DFC and identity quality metadata scored 0.630/0.635 below the
0.650 band-pass threshold. Their truthful difficulty percentiles were repaired;
2,200 train DFC records plus identity replay were added as source-pure shards,
with 24/25 DFC records added to validation/test. The seven-source deterministic
sampling contract now passes. Full manifest construction with SHA-256 checking
loaded 1,148 train, 16 validation, and 16 test shards at context 2,048 in 30.4
seconds, exposing 5,577,097 train windows and all seven train source classes.
Stream B is complete. Focused verification: 55 passed; changed-file Ruff and
diff checks are clean. No model training or cloud spending occurred. The only
remaining pre-training gate is a full-context GPU rehearsal with exact
kill/restart recovery; local CUDA visibility is currently blocked by Windows
GPU permissions and must be restored or moved to the bounded cloud rehearsal.

**2026-07-16 intelligence-foundation update.** V4 now has one canonical,
reversible extension boundary rather than disconnected adapter mechanisms.
Verified DFC process spans receive a bounded 1.25x training weight only when
their tags are complete; ordinary rows, validation, and malformed/truncated
spans are unchanged, and the objective is bound into exact-resume metadata.
LoRA/DoRA capability artifacts freeze the base and cryptographically bind their
adapter-only tensors to the exact checkpoint, V4 tokenizer, model profile,
target modules, shapes, and source commit. Serving activation fails closed and
rolls back state. A transparent reasoning-budget policy now returns bounded
direct/verify/retrieve/search plans through `/reasoning/plan` without executing
tools or claiming capability. On the RTX 4050, the full 181,132,071-parameter
base plus 54 DoRA targets completed one BF16 AdamW step in 5.80 seconds with
919,296 trainable parameters and 1,300 MiB peak allocation. The canary caught
and verified a real GPU device/dtype attachment fix. No checkpoint, dataset
training, or intelligence-quality claim was produced.

**2026-07-15 foundation hardening update.** The canonical V4 run is now one
explicit training algorithm rather than a collection of overlapping controls:
AdamW with matrix-only weight decay, gradient clipping, 2% warmup, and cosine
decay to 1e-5. The dynamic-regret LR overlay was removed from the foundation.
Seed 1301 is enforced as a reproducibility address, not treated as a quality
setting. Schema 9 checkpoints bind and exactly restore Python, NumPy, Torch,
CUDA, DataLoader, optimizer, scheduler, scaler, recipe, and counter-based raw
sampler state. A resume with missing state, reset Adafactor moments, another
seed/optimizer/recipe, or a mismatched sampler cursor now fails closed. AMP
overflow cannot advance progress evidence. Focused CPU verification passes;
the CUDA path was then rechecked successfully on the local RTX 4050. One
bounded sequence-64 dense update exercised all 181,132,071 parameters in 5.48
seconds at 3,499.82 MiB peak allocation with finite loss and gradients. A
same-seed rebuild reproduced its fingerprint/logits; another seed changed
them, confirming that seed is a replay address rather than a quality setting.
The 182,739,495-parameter MTP candidate also completed one bounded update in
4.08 seconds at 3,527.00 MiB. No checkpoint, long training run, full-context
memory claim, or model-quality claim was produced.

**2026-07-15 execution update.** The interrupted 30GB append was recovered by a
new full audit of the final 28,993,027,495-byte corpus: 5,274,479 valid records,
all four native source classes, and zero structural, hash, license, duplicate,
or quality failures. The seven-source campaign builder then produced a 64.024
MiB train slice with deterministic disjoint held-out sets and a verified
55/15/12/8/5/3/2 mix. The bound canonical 32,768-ID V4 passed frozen-prefix,
byte-round-trip, parameter-contract, and fertility gates; projected token
reduction is 38.9932%. Stream B is complete through V4 (`--skip-shards`). A
120GB native extension was started from the audited boundary, then stopped at
the owner's request at roughly 29GB. No downloader is active. The appended
tail is journaled but requires recovery and a refreshed audit before training.

Execution also exposed and repaired two restart defects before the 120GB run:
the downloader now commits the exact fsynced corpus-byte boundary in the same
SQLite transaction as appended document rows and can discard only an
uncommitted tail after a hard termination; partial reasoning/science runs now
write scope-specific status files and cannot overwrite base-corpus completion.
Stream B requires explicit completed base-bucket evidence. Tokenizer fallback
encoding preserves exact IDs while using a single compiled special-token scan
and bounded piece caching (1,003-case parity probe; ~1.95x sampled speedup).

CUDA is available again on the local RTX 4050 (6 GiB, BF16). Measured,
non-promoting training probes: the 57.4M profile at 256 tokens reached 840.6
tok/s and 1.11 GiB peak allocation; the 159.1M profile reached 552.8 tok/s and
3.04 GiB at 256 tokens. At its configured 2,048-token context and microbatch 1,
the 159.1M profile reached 255.4 tok/s while allocating 8.0 GiB, exceeding
physical VRAM and relying on Windows spill. At that measured rate, 1.5B tokens
would take about 68 days per seed locally; the required three-seed pilot remains
a cluster/cloud execution, not a credible single-4050 campaign.

The stopped 30GB resume was diagnosed rather than credited as complete. It
reached 21,582,998,123 bytes: FineWeb-Edu met its quota and FineMath added
3.865GB, but Stack v2 required authentication and the installed dataset client
rejected Dolma's legacy loading script. Both are now replaced by exact-commit,
streamable Common Pile sources: openly licensed Stack v2 code and openly
licensed ArXiv science/technical text. A live integration probe loaded one
valid row from all four native sources with zero errors. License enforcement is
now per row and requires every declared license to be allowlisted.

The missing non-foundation inputs are no longer placeholders. The pinned
SmolTalk instruction tranche completed at 120,000 examples / 230,753,834
bytes. `scripts/build_verified_dfc_corpus.py` produced 2,249 unique records /
4,195,921 bytes, split across deterministic formal-proof and constraint
verifiers, with every row verified. Historical inferred synthetic DFC is
explicitly barred from the verified campaign bucket. The tokenizer-slice
builder records identity replay bytes instead of pretending the 1.5KB identity
source contains a megabyte of unique data.

Acquisition now publishes live byte/rate/source progress, fsyncs corpus bytes
before committing the matching SQLite boundary, and can advance a trusted
audit through a hash-chained online-validated append. The `120gb` native corpus
profile is executable on the pinned larger FineWeb tranche. The full audit of
the 21,582,998,123-byte file completed with 4,113,170 valid records and zero
structural, hash, license, duplicate, or quality failures. The repaired 30GB
resume passed that fail-closed boundary and is actively appending from the
resume-safe index toward the 27GiB native target. Canonical
V4 execution now also requires a ready seven-source slice manifest and exact
train-slice hash. Immutable token shards now stop at source-class boundaries;
the raw trainer deterministically samples the declared 55/15/12/8/5/3/2 mix,
and the tiny identity source is explicitly replay-materialized to 4,097 tokens
so it produces real 2,048-token windows. V3/V4 shards now publish into distinct
tokenizer-bound families, and signed pilots derive token capacity from their
own signed train manifest instead of a global V3 inventory. Consolidated
data/tokenizer/sampler/orchestration/pilot-contract verification: **75 passed**
in 26.63 seconds; changed-file Ruff and diff checks are clean.

The signed-launch path is now genuinely three-seed executable. Each of the 23
factorial cells expands to three independently signed schema-3 run manifests
(69 total), each with one exact seed, a unique checkpoint artifact, immutable
train/validation manifest hashes, and explicit data roles. The signed seed is
wired into Python, Torch, CUDA, model initialization, and raw-shard sampling.
The old representation placed three seed values in one manifest while starting
only one unseeded trainer, so it could not have produced three replicas. V4
checkpoint contracts and campaign gates now derive schema, vocabulary hash,
special-token IDs, and probe fingerprint from the live signed tokenizer rather
than the global V3 manifest. Invalid sampler weights/modifiers fail closed, and
V4 cells become launchable only after the canonical Stream-B artifacts resolve
their blocker. Scratch manifests no longer become a bogus `--resume_from
scratch` path, while real continuation sources are byte-hash-bound. Additional
focused verification: **64 passed, 1 skipped**;
changed-file Ruff and diff checks remain clean.

The ablation farm now has a resumable local dispatcher rather than a folder of
manifests with no execution path. `scripts/run_pilot_queue.py` verifies the
signature, immutable tokenizer/data/checkpoint hashes, forecast lead time,
blocker state, seed/run identity, and prior completion evidence before planning
each job. It writes an atomic queue plan plus per-seed status/log files, skips
completed replicas, excludes moonshots by default, runs base-only sessions,
and refuses non-CUDA execution unless explicitly overridden. Signed-worker run
reports are stored beside their unique checkpoint instead of racing on the
single global report path. Metrics, evaluation summaries, mix control, CDR,
data-route evidence, and progress journals are likewise isolated into a
per-seed report directory inherited by the trainer child process.
Forecast outcomes now reject non-finite values, duplicate resolution, blank
verdicts, and missing evidence; every accepted outcome records the resolved
evidence path and its SHA-256 in the append-only forecast chain.

Design suggestion for later review (non-binding): ThirdEye could become the
canonical evidence/reporting plane while An-Ra retains the signed launch and
model-specific enforcement contract. That would allow one normalized run,
artifact, metric, and evidence record without requiring the repositories to be
merged. Duplicate An-Ra reporting could be reduced only after parity tests;
local fallback behavior could remain available if ThirdEye is unavailable.

`scripts.execute_stream_b` now validates the completed native audit and
seven-source slice, proves the fixed canonical V4 identity, and publishes only
V4 train/validation/test shards. It cannot rebuild or fall back to V3. Shard
progress is written continuously to
`output/v2/data_manifests/token_shard_progress.json`. No acquisition or
training worker is running automatically.

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
candidate. Artifact-specific defects and the recovery decision remain recorded
in `docs/engineering/ENGINEERING_LOG.md`. The campaign-slice builder
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

The dated entries below are historical evidence, not runnable instructions.
V3 builders, draft artifacts, and the old factorial named in those entries
were deleted in the 2026-07-15 reclamation; use the V4-only action section.

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
- The authoritative unfinished-work list is `TODO.md`.

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
  non-destructive: `CANONICAL_V4_VOCAB_SIZE = 32_768` with the current schema-7
  530,602,595-param contract; `build_append_only_v4`/`audit_token_fertility`
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

## Next Action (start here) — V4 canonical path

The active repository is now V4-only. New training and serving use the
181,132,071-parameter `anra-v4-180m` profile, the 32,768-token canonical V4
tokenizer, and one checkpoint lineage named `anra_v4_180m.pt`. V3 has been
deleted and cannot be selected through supported entry points. Separate
V2 identity/ouroboros fine-tune checkpoints are no longer part of the unified
training sequence; those capabilities belong in the one model's curriculum.

The next owner action is to open `notebooks/AN_RA_T4_TRAINING.ipynb` in one T4
Colab and train the canonical seed 1301 against the published V4 shards. The
normal workflow does not require two additional seed runs. No GPU training was
started during this migration.

The foundation code is ready for that run: exact resume, verified process
weighting, and bounded evidence counters are part of the canonical recipe.
MTP, MoE, SSM, latent reasoning, HAL promotion, and moonshots remain separate
experiments. Do not combine them into the first baseline. Reversible
capability adapters become useful only after the immutable base checkpoint
exists; their local one-step canary is execution evidence, not a reason to
promote an unmeasured capability.

After the primary run, inspect generation quality and checkpoint compatibility
before spending compute on replication, ablations, moonshots, soak tests, or
promotion. The historical 69-run factorial remains research evidence tooling,
not the default training plan.

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
