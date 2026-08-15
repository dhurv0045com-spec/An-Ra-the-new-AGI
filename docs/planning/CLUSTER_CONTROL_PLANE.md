# The GPU-Cluster Control Plane — Training Infrastructure Proposal

*Written 2026-07-06 after a code-level inspection of both repositories. This is
the companion document to MASTER_UPGRADE v3 Layer 12-B. Every claim below is
classified: **WORKS** (verified in implementation and tests), **NEEDS-FIX**
(implemented with a specific named defect), **DOCUMENTED-ONLY** (contract or
doc exists, no code), **PROPOSED** (new), or **MUST-PROVE** (required evidence
before real training starts).*

Repositories, ownership preserved:
- **An-Ra** (`An-Ra-the-new-AGI-1`, branch `iterate500`) — owns the model,
  tokenizer lineage, training scripts, evaluation, and all Laws.
- **Cluster** (`gpu cluster`, branch `main`) — owns coordination: FastAPI
  coordinator + SQLite WAL on the laptop, free-ngrok ingress, Drive OAuth and
  artifact exchange, leased Colab workers.
- The boundary is already correct and must stay: **An-Ra never imports cluster
  code; the cluster invokes An-Ra only through allowlisted CLI scripts and
  JSON manifests** (`worker/campaign_worker.py:20-30` allowlist;
  `scripts/verify_anra_integration.py` enforces contract drift fail-closed).

---

## 1. What the master plan was missing

MASTER_UPGRADE v3 Layer 12 specified an experiment registry, artifact store,
job queue, and training CI — as requirements. The cluster repo is the
*implementation* of that layer, already substantially built, and the plan
never named it. Concretely missing from the plan until now: the coordinator/
worker execution model, the Drive storage economics (15GB against a 30GB
corpus), the campaign state machines that make a multi-week run survivable on
preemptible free compute, and the security posture of a laptop exposed
through a public tunnel.

## 2. The cluster's role in An-Ra's architecture

The cluster is the **execution half of Layer 12**: MASTER_UPGRADE defines
*what* must be true of any training run (manifests, lineage pinning,
reproducibility, gates); the cluster makes those properties hold on real,
unreliable, free hardware. It is also the natural transport for Stream A/B
of the parallel campaign: pilot cells, tokenizer jobs, data preparation,
draft proof, frontier rescue, and continuation training are all already
expressible as its job kinds (`backend/contracts.py:10-19`).

## 3. Findings (verified from implementation)

### WORKS — verified in code and covered by tests
- **Fenced leases.** `attempt_count` is a monotonic fencing token; commits
  re-check worker+lease+attempt+expiry (`backend/campaign.py:296-306`). A
  zombie with an expired or superseded lease cannot commit
  (test: `test_expired_attempt_is_requeued_and_cannot_commit`).
- **Idempotent commits** by `idempotency_key = job:attempt:artifact-hash`
  (`campaign.py:290-295`, `campaign_worker.py:330`).
- **Double-training prevention.** `AcceptedWindow` uniqueness rejects any
  window ever accepted before (`campaign.py:309-313`;
  test: `test_duplicate_window_is_rejected_across_jobs`).
- **Single canonical writer** — one leased/running canonical job at a time
  (`campaign.py:80-89,153-162`); no shared writable checkpoint.
- **Storage preflight fails closed** at campaign creation
  (`campaign.py:37-42`; test: `test_campaign_storage_gate_fails_closed`).
- **Lease renewal via heartbeat** every 30s against a 180s lease
  (`campaign_routes.py:221-245` calls `renew_lease`) — long jobs do not
  silently expire.
- **Atomic publication.** Release pointers: temp + fsync + rename
  (`artifacts.py:63-74`). Worker artifacts: `.partial` copy + hash check +
  rename into the Drive folder (`campaign_worker.py:128-137`), re-verified by
  SHA-256 + size at commit (`campaign.py:314`).
- **Promotion requires verification.** Only `verified` checkpoint artifacts
  with all gates true can be promoted; previous checkpoint recorded as the
  rollback target (`recovery.py:64-91`).
- **Secrets discipline.** `.env`/`client_secret.json` gitignored and
  untracked (verified); OAuth credentials Fernet-encrypted at rest with
  refresh + single-use CSRF state (`auth.py:23-123`); campaign key and worker
  secrets scrypt-hashed at rest; operator auth fails closed when unset
  (`security.py:109-113`).
- **Sparse aggregation is genuinely off** — double-gated (env flag returns
  410, manifest flag raises at creation; `main.py:49,331`,
  `campaign.py:43-44`). It is correctly *not* claimed to be exact distributed
  training.
- **Worker discipline.** Script allowlist, source-commit verification,
  SHA-256 verification of checkpoint/tokenizer/data-manifest inputs,
  `ANRA_`-only environment allowlist, tokenizer bundle round-trip with
  unsafe-path rejection, optimizer-boundary validation of checkpoint
  artifacts (`campaign_worker.py:98-227`).
- **Cross-repo contract verifier.** Job manifests checked against An-Ra
  script existence and `--help` surfaces; canonical-trainer contract strings
  (`--max-phase-tokens`, `training_progress_journal.json`,
  `ANRA_DURABLE_CHECKPOINT_STEPS`, `completed_optimizer_boundary` — all
  present in `scripts/build_brain.py`, verified); draft/V4 cache separation
  and draft-proof-before-V4 ordering enforced
  (`scripts/verify_anra_integration.py`).
- **WAL + atomic DB backup** with Windows-safe close-before-replace
  (`database.py:13-19`, `recovery.py:21-42`).

### NEEDS-FIX — implemented, specific defect, ordered by severity
1. **Unauthenticated v1 heartbeat.** `POST /api/workers/{id}/heartbeat`
   (`main.py:322`) lets anyone on the public URL keep dead workers "alive"
   and write fake loss/step telemetry. Remove the v1 route (v2 exists) or
   put it behind worker HMAC.
2. **Worker secret sent in cleartext on every request** (`x-worker-secret`
   header, `security.py:96`, `campaign_worker.py:62`) — the HMAC key rides
   alongside the HMAC, so signing adds replay protection but not credential
   confidentiality. Fix: store worker secrets Fernet-encrypted (like Drive
   credentials) instead of scrypt-hashed, verify HMAC server-side from the
   decrypted secret, and **drop the secret header entirely**.
3. **Orphaned training subprocess.** In `execute_lease`, a single failed
   heartbeat request raises out of the loop while `subprocess.Popen` keeps
   training (`campaign_worker.py:289-296`); the worker then claims new work.
   Fix: retry heartbeats with backoff; on lease loss (409) or fatal error,
   `process.kill()` before leaving.
4. **Dead progress telemetry.** Heartbeats hardcode
   `optimizer_step=0, tokens_processed=0, loss=None`
   (`campaign_worker.py:190-208`), and the `TrainingProgressReport` contract
   is never used. Fix: tail `training_progress_journal.json` (already
   emitted by `build_brain.py`) and report real step/tokens/loss.
5. **`renew_lease` missing expiry check** (`campaign.py:261-282`) — a lease
   already past expiry can be resurrected until the reaper runs (≤30s
   window). Commit-side fencing makes this safe but the resurrection is
   still wrong; add `lease_expires_at >= now` to the renewal predicate.
6. **`torch.load(..., weights_only=False)`** on worker-produced files
   (`main.py:409`, `artifacts.py:35`, `campaign_worker.py:214`) — pickle
   deserialization is arbitrary code execution. Use `weights_only=True`
   everywhere the payload is state-dict-shaped.
7. **No poison-job cap.** A job that always fails requeues forever
   (`campaign.py:163` increments, nothing bounds it). Add
   `max_attempts` (default 5) → `status="failed"` + incident.
8. **Unauthenticated read endpoints** leak worker `account_email`
   (`/api/workers`), Drive file listings (`/api/drive/files`), and live logs
   (`/api/logs/stream`) to anyone with the ngrok URL. Operator-token all
   three.
9. **No rate limiting** on a public tunnel. Add a simple per-IP limiter to
   auth and worker routes.
10. **Local-path trust in artifact verification.**
    `_artifact_verifier` hashes a worker-supplied path if it exists on the
    coordinator host before falling back to Drive
    (`campaign_routes.py:248-256`). Restrict to basenames under the managed
    artifact directory.
11. **Drive query interpolation** (`drive_sync.py:66`) — escape quotes in
    filenames.
12. **NonceRecord never pruned** (`security.py:46-51`) — unbounded growth;
    delete rows older than the skew window.
13. **Single-slot DB backup** (`cluster-latest.db` overwrite) — keep 3
    generations.
14. **Storage preflight formula is unsatisfiable on free Drive.**
    `(corpus + 12×checkpoint_bytes)×1.2` (`campaign.py:37`) demands ~86GB
    for a 6GB full training state — with 15GB available, every honest
    manifest fails. Replace with the tiered budget of §6 (this is a *model*
    fix, not a gate weakening: the gate stays fail-closed against the new
    budget).

### DOCUMENTED-ONLY
- **Hot-spare promotion.** `standby` job kind exists (`contracts.py:18`);
  no promotion logic exists. (§8 recommends the simple resolution.)
- **Rollback execution.** `RollbackReport` contract exists
  (`contracts.py:169-177`); promotion records the rollback target — but no
  code restores it. Rollback is a pointer, not an action, until P4.
- **"Requeued from last verified checkpoint"** (incident label,
  `campaign.py:408`) — true only when a partial commit set
  `resume_artifact_id`; a worker that dies before any commit requeues from
  the job's original checkpoint. Label should say which happened.
- `backend/scheduler.py` is dead code (never imported) — delete or mark.

### What the tests prove today
Storage gate fail-closed, lease fencing + idempotency, expiry/requeue +
zombie-commit rejection, window dedup across jobs, intermediate-checkpoint
requeue of canonical jobs, suspect→expired transitions, control/aggregation
DB–Drive sync, credential-gated cluster init. **Untested:** interrupted
uploads, nonce replay, heartbeat renewal, coordinator crash mid-commit,
tunnel outage, end-to-end worker loop, rollback. §10 closes these.

---

## 4. Recommended cluster-repo upgrades

Everything in NEEDS-FIX above (P0/P1 of §11), plus:

- **Coordinator restart grace.** On startup, extend all active leases by one
  full period. Today a laptop reboot or tunnel outage >180s expires every
  lease at once and triggers a requeue storm plus orphaned work on healthy
  workers. (Laptop is an accepted SPOF; this makes its restarts cheap.)
- **Campaign pause/drain/resume verbs.** `pause` (stop issuing leases,
  let running jobs finish = drain) vs `halt` (also refuse commits) — today
  only the v1 sparse-path pause exists. Add `POST /api/v2/campaigns/pause|
  drain|resume` (operator).
- **Quota ledger.** A `StorageSlot` table tracking every Drive object the
  cluster owns (kind, bytes, pinned/evictable, artifact link) + permanent
  deletion (Drive trash still consumes quota). Preflight per-*job* (will
  the artifact fit?) not just per-campaign.
- **Incident → An-Ra Experience Ledger export.** One-way JSONL export of
  incidents/attempts/promotions in S1 schema, so campaign history becomes
  ledger evidence. An-Ra imports files; no code coupling.
- **Worker config bootstrap via Drive.** Free-ngrok URLs rotate on restart;
  workers should read the current coordinator URL from a signed
  `cluster_bootstrap.json` on Drive instead of hardcoding it, so URL churn
  doesn't strand the fleet.

## 5. Required An-Ra integration changes (all additive)

- **Slim resume state.** `training/checkpoint.py` gains an option to save
  optimizer state in bf16 (or paged 8-bit) — full training state drops
  ~6GB → ~3–4GB, which is what makes the §6 Drive budget close. **Law 2
  applies:** ship only with a resume-parity ablation (bit-exact eval metrics
  after resume vs fp32 baseline, 2 seeds). Until proven, fp32 state stays
  and the budget uses the laptop-archive path.
- **Journal completeness.** Ensure `training_progress_journal.json` updates
  every optimizer boundary with step/tokens/loss (worker tails it for
  heartbeats). Already emitted; verify cadence with the draft pipeline.
- **Checkpoint proof pre-req.** `scripts/check_frontier_checkpoint.py`
  output becomes a required artifact before any rescue campaign (step.md
  already demands this; make the campaign bootstrap script check it).
- **Nothing else.** The allowlisted-CLI + JSON-manifest boundary is already
  the right integration surface; do not widen it.

## 6. Storage and data-delivery strategy (the 15GB answer)

Assessed options:

| Option | Verdict |
| --- | --- |
| (a) Whole prepared corpus on Drive (current `30gb` profile → `MyDrive/AnRa/data/...`) | **Rejected** — cannot fit 15GB alongside checkpoints; also makes Drive the bandwidth bottleneck |
| (b) Rotating active shard window through Drive | **Adopted for curated data** (Phases C–E, validation, identity) — small, high-value, coordinator-controlled |
| (c) Direct worker retrieval from licensed upstream sources, pinned by revision + file hash, tokenized deterministically on-worker | **Adopted for bulk** (Phase A/B raw text) — Colab bandwidth is free and fast; Drive never holds bulk |
| (d) Content-addressed Drive cache with pinned/evictable tiers | **Adopted as the accounting model** for everything on Drive |

**The three-tier design:**
- **Tier 0 — Drive (hot exchange, ≤ ~12GB budget, quota-ledger enforced):**
  `active_release.json` + candidate pointers; baseline model-only (~1GB
  fp16); latest full training state (slim, ~3–4GB); previous model-only
  snapshot (~1GB); tokenizer bundles + data manifests + contamination
  filter (MBs); validation shards (~200MB); 3 DB backup generations;
  one in-flight artifact slot. Everything else is evictable and tracked.
- **Tier 1 — Laptop (deep archive, authoritative lineage):** every published
  checkpoint, manifest, and evaluation bundle, pulled down by an archiver
  job after each promotion; Drive slots are then freed. The laptop already
  holds the corpus-preparation disk; it becomes the permanent record. The
  cluster DB already records every artifact hash, so archive integrity is
  checkable.
- **Tier 2 — Worker-local (ephemeral bulk):** Phase A/B token shards derived
  on-worker from upstream sources pinned by dataset revision + per-file
  SHA-256 in the `DataShardManifest`; `window_ids` derive deterministically
  from (source revision, file, row range, tokenizer hash), so the
  AcceptedWindow ledger keeps exactly-once semantics even though the bytes
  never touch Drive. Re-derivation after preemption costs minutes of CPU,
  not Drive quota.

**Corpus flow:** laptop (or a data-prepare job) computes the manifest chain
once — upstream file list, hashes, window partition, decontamination filter —
and publishes only manifests (KB–MB) through Drive. Workers fetch bulk bytes
from the source, verify hashes, tokenize, train. Interrupted uploads stay
solved by `.partial`+rename+commit-verification; interrupted *downloads* are
solved by per-file hash retry.

**Falsifiable sizing prediction (MUST-PROVE at smoke scale):** a full
smoke-profile cycle (prepare → draft-train → evaluate → promote) leaves Drive
occupancy within budget with ≥1 free checkpoint slot at all times, measured
by the quota ledger. If slim-optimizer parity fails, fallback: fp32 state
lives only in Tier 1 via upload-then-archive-then-free rotation (slower
promotion cadence, same safety).

## 7. End-to-end campaign lifecycle

`smoke proof → tokenizer proof (draft BPE + fertility gates) → draft train
(32M tokens, pipeline validator) → canonical V4 (campaign-corpus candidates,
per MASTER_UPGRADE Layer 3) → frontier rescue (continuation from verified
baseline) → phase continuation (acceptance.minimum_phase_tokens per phase) →
evaluation jobs (An-Ra deterministic suite) → promotion (PromotionDecision
with Gate-6 evidence attached) → next phase`, with the rollback drill
executed before the first promotion and after any incident that touches the
canonical lineage. The existing job graph (`campaign_jobs.example.json`)
already encodes the first five stages with correct dependency ordering,
including draft-proof-blocks-V4 — keep it as the template.

State machines (implemented today): job `queued→leased→running→completed`
(+requeue), attempt `started→running→checkpointed|completed|expired`, worker
`registered→active→suspect→expired`. Added by this proposal: campaign
`created→running→paused|draining→completed|halted`, promotion
`candidate→audited→promoted|rejected→rolled-back` — the `audited` state is
MASTER_UPGRADE's adversarial gate audit applied to promotions.

## 8. Failure recovery and security model

Failure matrix (cause → mechanism → status):
- Colab preemption / quota exhaustion → lease expiry → requeue; resume from
  `resume_artifact_id` when a boundary was committed — WORKS; add real
  progress telemetry so the operator sees where it died (P2).
- Interrupted upload → `.partial` + commit-side hash/size verification —
  WORKS; add chaos test (P3).
- Stale/zombie worker → suspect/expired tiers + commit fencing — WORKS;
  close the renew-race (P0).
- Laptop/tunnel outage → today: mass lease expiry; after P0: startup grace +
  Drive-bootstrap URL rediscovery; workers idle-poll harmlessly meanwhile.
- Coordinator crash mid-commit → SQLite WAL + single transaction per commit —
  WORKS in design; chaos-test kill-mid-commit (P3).
- Poison job → attempt cap + failed state (P0).
- Hot spare → **descope the promotion machinery**: with claim-polling every
  30s, a sixth registered worker *is* a hot spare — any requeued canonical
  job is claimed within one poll. Formalize only a priority rule (spare
  prefers canonical work). The `standby` job kind stays for keeping the
  spare warm (cache-hydration jobs).

Security model after P0: every mutating endpoint authenticated (operator
bearer / worker HMAC + nonce + timestamp); no credential material on the
wire after registration; read endpoints operator-gated; per-IP rate limit;
`weights_only=True` deserialization; secrets never in either repo (verified
today, kept by CI secret-scan in both repos); Colab workers are the user's
own authorized accounts — no quota evasion, sequential sessions only, which
this design respects by running 5+1 workers as distinct authorized users.

## 9. Observability and operator experience

One `/api/v2/campaigns/current` already reports campaign state. Add (P2):
- **Dashboard truth**: real per-worker optimizer_step/tokens/s/loss (from
  the journal relay), campaign token progress vs phase target, Drive quota
  ledger occupancy, worker states, incident feed, promotion history. The
  frontend components exist (`WorkerCard`, `LossCurve`, `ThroughputBadge`);
  they must stop rendering the dead zeros of finding #4.
- **Operator verbs**: pause / drain / resume / replace-worker (revoke secret
  + re-issue) / promote / reject / roll back — each a single authenticated
  call with an audit row; promote/reject/rollback already have DB scaffolding.
- **Weekly digest** exported to the An-Ra engineering log: tokens trained,
  incidents by category, requeue count, quota headroom, gate outcomes.

## 10. Tests, simulations, and acceptance gates (MUST-PROVE)

- **G-C1 Chaos suite** (new, cluster repo): kill coordinator mid-commit;
  kill worker mid-train; sever tunnel 10 min; truncate upload at 50%; replay
  a captured nonce; zombie commit after requeue; duplicate window replay.
  Pass = DB consistent, zero duplicate windows, campaign resumable, no
  unverified artifact accepted. (Unit tests already cover 4 of these
  invariants statically; the harness proves them under real crash timing.)
- **G-C2 Soak**: 24h, 2 workers, smoke profile: ≥95% heartbeat/renewal
  success, 0 orphaned subprocesses, 0 unverified artifacts.
- **G-C3 Preemption drill**: kill a Colab worker at a random minute, ×5:
  resume from last boundary every time; token accounting monotonic;
  AcceptedWindow query shows zero repeats.
- **G-C4 Security**: all mutating endpoints reject unauthenticated calls;
  replay suite blocked; rate limiter engages; secret-scan clean both repos.
- **G-C5 Efficiency**: coordination overhead <2% of worker wall-clock;
  Drive always ≥1 free checkpoint slot; dashboard shows live journal values.
- **G-C6 Reproducibility**: same job manifest + seed → bit-exact eval
  metrics (ties into MASTER_UPGRADE Layer 12's weekly re-run spot check).
- **G-C7 Rollback drill**: execute a real restore to the previous verified
  checkpoint; `RollbackReport` all-true; required before first promotion.

## 11. Ordered implementation phases

| Phase | Content | Exit |
| --- | --- | --- |
| P0 — Safety (first, blocking) | Findings 1–3, 5–12 of §3 NEEDS-FIX; coordinator restart grace | G-C4 partial (auth+replay) |
| P1 — Storage | Tier model + quota ledger + laptop archiver + preflight formula; An-Ra slim-optimizer option (ablation-gated) | Smoke-cycle sizing prediction verified |
| P2 — Truth | Journal→heartbeat telemetry; dashboard; operator verbs; DB backup generations | G-C5 |
| P3 — Chaos | Chaos harness + soak + preemption drills | G-C1, G-C2, G-C3 |
| P4 — Promotion | Evaluation-gated promotion wired to An-Ra suite; rollback executor + drill; adversarial promotion audit | G-C6, G-C7 |
| P5 — Ladder | smoke → tokenizer proof → draft → canonical V4 → frontier rescue on the real 5+1 fleet | first rescue checkpoint promoted with evidence |

P0–P2 are laptop-only work (agent-executable now). P3 needs one Colab
account; P5 needs the fleet and the restored baseline checkpoint
(step.md preconditions stand).

### Implementation status - 2026-07-10

- **P0-P2 local mechanisms implemented.** Security fencing, three-tier quota
  accounting, real journal telemetry, operator controls, and three-generation
  backups are exercised in the cluster suite.
- **P3 evidence evaluators implemented, live exits still open.** G-C1/G-C2/
  G-C3 reject missing, simulated, short-duration, or incomplete evidence. No
  real Drive chaos run, 24-hour soak, or five-preemption campaign is claimed.
- **P4 local mechanisms implemented, live exits still open.** An-Ra now emits
  a signed, checkpoint-bound promotion envelope containing Gate-6 evaluation,
  bit-exact same-manifest/same-seed reproducibility, adversarial-audit, and
  signed rollback-drill evidence. The cluster rejects arbitrary boolean gates
  and validates the fixed envelope without importing An-Ra code. Promotion
  records the previously active verified release as its rollback target.
  Rollback now copies the previous full checkpoint atomically, validates model,
  optimizer, scheduler, tokenizer, optimizer-boundary, and data-position state,
  writes a signed `RollbackReport`, and only then repoints `active_release.json`.
  Corrupt bytes leave the active release unchanged. The real G-C6/G-C7 exits
  remain open until owner-signed evaluation artifacts and a Drive-backed
  checkpoint drill are supplied.
- **Cross-repository drift check passes.** The cluster integration verifier
  executes An-Ra CLI help surfaces and checks the four P4 gate names; current
  result: 8 job contracts, zero errors. Cluster suite: 27 passed. An-Ra
  non-GPU suite: 581 passed, 1 skipped.

## 12. Risks, assumptions, unresolved decisions

- **Assumed:** Colab availability at ~3h/session/account; Drive API rate
  limits tolerate manifest+checkpoint churn (low volume after §6); upstream
  dataset hosts remain available (mitigation: pinned revisions + laptop
  holds a raw copy of anything irreplaceable).
- **Risks:** laptop SPOF (accepted; mitigated by restart grace + DB backups
  + Drive-held state); ngrok URL churn (mitigated by Drive bootstrap);
  free-tier ToS changes (respond by pausing, never by evasion);
  slim-optimizer parity failure (fallback path defined in §6).
- **Unresolved, needs an owner decision:**
  1. Optimizer-state precision (bf16 vs 8-bit vs fp32-archive-only) —
     decide after the parity ablation.
  2. Job completion currently trusts worker-reported
     `continuation_token_counts` (`campaign.py:345-347`); §10's G-C6 plus
     server-side journal cross-check would close it — decide whether phase
     completion should additionally require a validation-loss report.
  3. Whether cluster incidents stream into the Experience Ledger live
     (during campaign) or as post-campaign import — start with import.
