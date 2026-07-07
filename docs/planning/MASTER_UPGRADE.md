# AN-RA — MASTER UPGRADE v3: The Unified Intelligence Program

*Final pre-implementation edit (2026-07-06). Supersedes v2 by extension, not
replacement: every v2 foundation stands; this edit deepens each layer, adds the
missing systems (unified architecture spine, continual learning, multimodality,
distributed infrastructure, product experience, the moonshot registry), and
restructures the campaign into parallel agent-powered workstreams. After this
edit, implementation begins. Grounded in the codebase as measured — including
the Week-1 fertility evidence produced 2026-07-06.*

**The two laws stand, absolute:**
1. Weight lineage, tokenizer identity, canonical token IDs 0–8208: never
   silently touched. Growth only via named, versioned, verified migrations.
   No pretrained external weights enter the lineage, ever.
2. Nothing is "improved" without pre-registered prediction + ablation evidence
   (Gate 5); nothing acts without the sovereignty gate.

**Major-upgrade block format used throughout** — every substantial upgrade
carries ten fields:
`Unlocks / Architecture / Extends / Changes / Deps / Gates / Predictions /
Kill & fallback / Cost / Lineage & gate`.
Speculative items live in the **Moonshot Registry** and are labeled `M#` —
they carry pilot paths and kill criteria, never promises.

---

## PART 0 — The Physics (updated with measured facts)

Every row is tagged: **MEASURED** (a number someone ran), **ASSUMED** (an
external fact not yet verified in-house), **PROJECTED** (arithmetic on an
assumption), or **DECISION** (a choice, not a fact).

| Quantity | Value | Status |
| --- | ---: | --- |
| Current parameters | 499,167,047 dense | MEASURED |
| Over-train target | **30B tokens** (Chinchilla ~10B; small models keep improving well past it) | DECISION |
| A100 throughput, 500M packed @1024, bf16 + fused kernels | ~250–350M tok/hr | ASSUMED — market specs; verified in-house during week-3 throughput tuning |
| **30B tokens at spot prices** | **~$130–260 before multipliers** | PROJECTED from the row above |
| **V3 tokenizer tax** | **2.518 tok/word English held-out** (gate 1.35; worse than v2's 1.5–2.0 prediction) | MEASURED 2026-07-06 |
| V4-draft realized held-out gains (local 5MB corpus) | DFC −30.5%, code −2.4%, **English −0.2%** | MEASURED 2026-07-06 |
| Canonical V4 candidates must come from the ≥50MB campaign corpus | local corpus cannot fix prose fertility | MEASURED — proven, not assumed |

**Compute, two tiers (named so no future reader has to wonder):**
- **Confirmed today:** the Colab free tier (TPU v2-8 sessions via
  `build_brain_tpu.py`, T4 sessions), the local development machine, **and
  the GPU-cluster control plane (Layer 12-B) that coordinates 5 authorized
  Colab workers + 1 hot spare as one durable campaign** — coordinator
  verified healthy 2026-07-05 (`gpu cluster/step.md`). Zero dollars; Phase A
  on this tier alone runs at ~3× wall-clock.
- **Assumed, not yet secured:** rented A100/H100 spot instances. Every
  dollar figure in this document is a market-rate projection against this
  tier. **Securing the rental — or formally adopting the free-tier
  schedule — is a week-1 Stream-A deliverable.** The plan executes either
  way; only the calendar stretches.

The measured 2.518 tok/word means the true tokenizer multiplier is **larger**
than v2 estimated: a competent 32k BPE at ~1.30 tok/word is a **~1.9× cut in
effective training cost and a ~1.9× context extension**. The single cheapest
upgrade in the program just got more valuable.

**The campaign premise, upgraded:** this program is executed by a **fleet of
frontier AI coding agents working parallel workstreams** under one review
discipline (tests green, ablation evidence, engineering log, sovereignty gate).
Agent throughput removes the serial-labor bottleneck; the gates remain the
bottleneck **by design** — that is what keeps parallel speed honest.

---

## PART 0.5 — THE UNIFIED INTELLIGENCE ARCHITECTURE (the spine; new)

Everything in this program attaches to four shared substrates. This is what
makes the plan one machine instead of thirteen projects.

**S1 — The Experience Ledger** *(new system)*
- **Unlocks:** every inference, verification outcome, tool result, memory
  operation, and gate decision becomes durable, replayable, trainable data.
  The system's life becomes its curriculum.
- **Architecture:** append-only, schema-versioned event store (JSONL shards +
  manifest hashes, same discipline as corpus shards). Events:
  `{trace_id, ts, kind, inputs_hash, output, verifier_verdicts, gate_record,
  tokens, latency}`. Nightly compaction promotes qualifying traces into
  training shards (Phase E format).
- **Extends:** the engineering-log/evidence discipline; telemetry made honest
  in the 2026-07-03/04 session; `training/data_ledger.py`.
- **Changes:** one `runtime/experience_ledger.py` writer wired into the chat
  path, verifier bank, tool dispatcher, and gate; a compactor job; a
  train/serve firewall (ledger shards get source-hash val splits like all
  corpus data).
- **Deps:** none — build in week 1 of implementation; everything downstream
  consumes it.
- **Gates:** 100% of chat/tool/verifier events captured (audited by
  fault-injection test); replay of any trace reproduces the recorded verdict;
  <2ms p50 write overhead.
- **Predictions:** within 4 weeks of serving, ≥50k verifier-labeled traces —
  enough to measurably improve Phase E (ablation: fine-tune with/without
  ledger shards, 3 seeds, expect ≥+2% on tool-format compliance).
- **Kill & fallback:** if write overhead >10ms p50, buffer async; ledger loss
  is never allowed to block serving (fail-open for capture, fail-closed for
  training-data promotion).
- **Cost:** ~1–5 GB/month at expected volumes; trivial compute.
- **Lineage & gate:** ledger→training promotion is a *migration-class* event:
  manifest-hashed, logged, and consumed only through the standard curriculum
  machinery.

**S2 — The One Verifier Bank** (Layer 5, elevated to substrate)
One registry, five consumers: training reward (RLVR), inference checking
(DFC), data validation (synthetic flywheel), agent outcome verification
(plan–act–verify), and GEPA evaluation. A verifier added anywhere strengthens
all five simultaneously — this is the highest-leverage line of code anyone can
write in this repo, and the registry must make adding one a <1-hour task
(single interface, conformance test suite, auto-registration).

**S3 — The One Retrieval Substrate** (Layer 4, elevated to substrate)
One index serves: inference-time memory, the agent skill library, data-engine
dedup/curation, and citation grounding. Hybrid dense+BM25 now; the trained
retriever head (M5) drops in behind the same interface later.

**S4 — The Calibration Substrate (HAL + the epistemic tracker, elevated)**
HAL — the live quality/truthfulness signal, fed real measurements instead of
a 0.60 constant since 2026-07-04 — and the epistemic outcome tracker fuse
into one self-estimate surface with one semantics: *"how likely is this
system to be right, here, now,"* always traceable to measured outcomes in
the ledger (S1). Its consumers are everywhere trust is priced: the
sovereignty gate (calibration-gated autonomy, L7), memory write salience
(L4), verifier retry/escalation decisions (L5), agent plan confidence and
world-model rollout weighting (L6/L9), and GEPA acceptance thresholds (L7).
v2 carried four private copies of "confidence" across those layers; naming
the substrate deletes three of them. **Gate:** HAL's calibration curve
(predicted-vs-realized correctness) reported monthly; Brier score must
improve month-over-month or the signal is recalibrated before any consumer
may widen an autonomy grant on its strength.

**The compounding loop these create:**
`serve → verify (S2) → record (S1) → curate → train → serve better`,
with retrieval (S3) feeding context at serve time and curation at train
time, and calibration (S4) pricing trust at every step of it.
Every layer below either feeds this loop or consumes it; anything that does
neither does not belong in the program.

---

## PART 1 — Pilot Science (extended: the ablation farm)

Unchanged foundation: 50M→150M ladder, factorial pilots, local scaling law,
frozen `campaign_config_v2.json`, the anti-waste tripwire (±5% of predicted
curve).

**Extensions:**
- **The ablation farm is agent-operated:** each pilot cell (config × 3 seeds)
  is a job manifest (`training/launch_manifest.py`) that an agent can launch,
  babysit, and report on autonomously. Target cadence: **the full factorial
  (≈24 cells) completes in 5 days wall-clock** on the free TPU fleet + one
  spot GPU. Human attention is spent on verdicts, not runs.
- **Pilot cells added to the factorial** (beyond Muon/MoE/MTP/QK-Norm/SWA/V4):
  SSM-hybrid layers (M1), latent-reasoning channel (M3), trained retriever
  head (M5) — moonshots earn campaign inclusion here or wait.
- **Honesty correction to v2's claims:** literature at this scale supports
  **Muon ≈1.3–1.6×** token-efficiency (not 2×) and **upcycled-MoE ≈1.5–2.5×**
  capability-per-active-FLOP (not 4–6×; that figure belongs to from-scratch
  MoE at much larger scale). The budget table uses the honest ranges; pilots
  decide the real number and the ledger records it.
- **Forecast ledger:** every pilot cell's predicted outcome is written down
  *before* launch (the GEPA discipline applied to ourselves). Calibration of
  our own predictions is reported at campaign end — the program practices the
  epistemics it preaches.

---

## LAYER 1 — The Model (extended)

Foundations stand: GQA 4:1 + SWA/full hybrid (3:1, window 1024), QK-Norm,
RoPE θ=500k + NTK hook, soft-caps 50/30, MLA pilot-gated, MoE by sparse
upcycling (8 routed + 1 shared, top-2, aux-loss-free), native `<think>`
channel (control tokens **already reserved in V3** — measured in-session:
`<think>`, `<verify>`, `<hyp>`, `<obs>`, `<act>`, `<err>` exist at canonical
IDs), MTP head, Muon+AdamW split, batch ramp 0.5M→2M.

**1.5 Hybrid sequence backbone (M1 — moonshot, pilot-gated)**
- **Unlocks:** near-linear-cost long context (32k–128k) on consumer VRAM —
  the memory OS (Layer 4) gets a working set 10× larger for the same latency.
- **Architecture:** replace the SWA layers (not the full-attention layers)
  with Mamba-2-class SSM blocks → attention/SSM hybrid (~1:3), the
  Jamba/Zamba-validated pattern.
- **Extends:** the Layer-1.1 hybrid attention design; `anra_brain.py` layer
  registry.
- **Changes:** SSM block implementation + checkpoint-schema entry + migration
  with logit-parity harness (parity is *not* expected across this migration —
  it enters only via pilot training, never by surgery on the trained lineage).
- **Deps:** pilot ladder only; excluded from the 500M campaign unless the
  150M pilot wins.
- **Gates:** ≥0.98× short-context capability AND ≥1.5× long-context
  throughput at 150M, 3 seeds.
- **Predictions:** pass at 150M ⇒ enters V3-growth (Stage 5), not the current
  campaign.
- **Kill & fallback:** any short-context regression >2% ⇒ shelved; SWA hybrid
  is already good.
- **Cost:** pilot-only (~$10 of TPU time); saves serving cost if adopted.
- **Lineage & gate:** new-architecture branch, never a mutation of the
  earned checkpoint.

**1.6 Multimodal entry point (M2 — moonshot, staged, Law-1-constrained)**
- **Unlocks:** An-Ra sees. Screenshots, diagrams, camera frames become
  context — required for real device agency (Layer 6) and embodiment
  (Layer 9).
- **Architecture:** because **no external pretrained weights may enter the
  lineage**, the vision encoder is trained in-house: a small ViT (~20–40M)
  trained first as a masked-patch autoencoder on permissively-licensed
  images, then contrastively aligned to the language model's embedding space
  (CLIP-style) on ~10M image-text pairs; images enter the LM as 64–256 soft
  tokens through a 2-layer projector. The LM is fine-tuned with the projector;
  the text tokenizer is untouched (soft tokens occupy embedding space, not
  vocabulary — Law 1 clean).
- **Extends:** `multimodal/` (exists as a stub tree); the V4 tokenizer's
  reserved-ID space holds any needed control tokens (`<img>`, `</img>`).
- **Changes:** encoder + projector modules, image-text data pipeline with the
  same shard/manifest discipline, VQA-style eval category in the private
  suite.
- **Deps:** starts only after Gate 6 (the text core speaks); runs as a
  parallel workstream thereafter.
- **Gates (staged):** (i) autoencoder reconstruction beats PCA baseline by
  ≥30% MSE; (ii) contrastive retrieval R@1 ≥40% on held-out 5k pairs;
  (iii) full-stack: ≥60% on a 200-item screenshot/diagram QA suite the bare
  text model scores ~0% on.
- **Predictions:** at 500M-LM scale expect basic grounding (colors, layout,
  OCR-lite), not fine-grained reasoning — the claim is scoped in advance.
- **Kill & fallback:** if (ii) fails after 2 data/architecture iterations,
  park multimodality until V3 scale; the text program loses nothing.
- **Cost:** ~30–60 A100-hrs (~$50–100) for the full stage — the second-largest
  compute item; scheduled after the language campaign proves out.
- **Lineage & gate:** encoder/projector are *satellite weights* with their own
  lineage manifests; camera/screen access at serving time is a per-capability
  sovereignty grant.

**1.7 Memory-augmented decoding**
- **Unlocks:** the model consults the retrieval substrate (S3) *inside* the
  forward pass at designated layers (kNN-LM-style logit interpolation as the
  entry-level version), making memory a first-class inference mechanism
  rather than prompt stuffing.
- **Extends:** Layer 4 retrieval; `inference/` runtime.
- **Changes:** datastore of (hidden-state, next-token) pairs from Phase D/E
  data; interpolation head with learned λ; serving-path integration behind a
  flag.
- **Gates:** ≥5% perplexity reduction on knowledge-heavy held-out slices at
  ≤1.3× decode latency; ablation on the private suite's memory category.
- **Kill:** latency budget blown or λ collapses to 0 ⇒ retire; prompt-level
  retrieval remains.
- **Lineage & gate:** datastore is a versioned artifact bound to checkpoint
  hash.

---

## LAYER 2 — Training & Data (extended)

Foundations stand: the 30B five-phase curriculum (A 20B / B 6B anneal /
C 2.5B reasoning / D 1B conversation / E 0.5B tool), synthetic flywheel with
verifier-backed labels capped at 30%/phase, contamination firewall, WSD with
branch points, preflight ritual, post-training stack
SFT → RLVR/GRPO → STaR×3–5 → DPO → self-distill.

**2.0 Data acquisition workstream (now explicit, week 1, parallel)**
The corpus does not exist locally (measured: 5MB on disk vs 30B tokens
needed). A dedicated workstream acquires, shards, dedups, decontaminates, and
manifests FineWeb-Edu / FineMath-4+ / permissive code **before** the campaign
window: target **≥120GB clean text sharded and hashed by end of week 2**,
validated by the draft pipeline. This was v2's silent assumption; it is now a
gated deliverable with an owner.

**2.4 Continual learning — the model never stops training** *(new)*
- **Unlocks:** An-Ra improves from its own operation between campaigns — the
  Experience Ledger (S1) becomes nightly weight updates. This is the
  difference between a trained artifact and a growing mind.
- **Architecture:** nightly consolidation job: sample verifier-approved
  ledger traces + a replay mix from protected corpus shards
  (`training/replay_pipeline.py`, `training/continual.py`) → **sparse-LoRA
  adapter update** (`training/sparse_lora.py`) → full private-suite
  regression eval → adapter promoted (merged quarterly via a named migration)
  or discarded. Weights-level changes ride the adapter; the base checkpoint
  changes only by migration.
- **Extends:** Layer 4's consolidation daemon (memories) — this is the same
  sleep cycle acting on weights; GEPA (Layer 7) supplies the accept/reject
  discipline.
- **Changes:** consolidation scheduler, adapter registry keyed to checkpoint
  hash, automatic rollback on regression.
- **Deps:** post-training complete; Experience Ledger live; eval CI (Layer 11)
  live.
- **Gates:** each accepted nightly update: ≥+0.5% on its target category,
  ≤0.5% regression on every protected category, 2 seeds. Quarterly merge:
  full Gate-6 re-pass.
- **Predictions:** tool-format compliance and memory-category scores climb
  week-over-week without a campaign; if they don't after 4 weeks, the ledger
  data or the recipe is wrong — investigate, don't force.
- **Kill & fallback:** two consecutive rejected weeks ⇒ pause and audit data
  quality; catastrophic-forgetting signature (protected val −2%) ⇒ automatic
  adapter rollback (signed-bundle machinery).
- **Cost:** ~1 T4-hr/night — free-tier sustainable.
- **Lineage & gate:** adapters are satellite artifacts; merges are
  migrations; the whole loop runs inside GEPA's pre-registration discipline
  and the sovereignty gate approves any autonomously-proposed recipe change.
- **Self-directed consolidation (pilot, GEPA-scored):** after 4 weeks of
  stable fixed-recipe operation, run a pilot in which the model proposes
  its own nightly consolidation targets — which categories to prioritize,
  which ledger traces to weight, what replay mix — through the standard
  GEPA pre-registration path. Adopted only if it beats the fixed recipe on
  3-week category-trend slope, 2 seeds; otherwise the fixed recipe stands
  and the negative result is logged like any other. The fixed recipe is
  the permanent fallback, never removed.

**2.5 Curriculum-order science (cheap, high-yield)**
Pilot cells testing *order* effects (code-before-prose, math-density ramp,
identity-mix timing) at 50M scale — order is free to change and literature
says it matters up to ~5% final capability. Winner ships in
`training/curriculum.py`. Gate: ≥2% held-out delta to adopt.

---

## LAYER 3 — Tokenizer (updated with Week-1 evidence)

The measurement is done and the verdict is in (2026-07-06):
**English 2.518 tok/word (gate 1.35 — fail), code 0.407 tok/char (pass),
DFC 0.450.** The append audit passes (21.2% projected on 1M units) and the
draft build proved the machinery end-to-end — including that **byte-fallback
fixes a real V3 round-trip defect** — but also proved the decisive fact:
**local-corpus candidates cannot fix prose fertility (−0.2% realized).**

Therefore, amended plan:
- **Canonical V4 = 32k** (raised from 16,384), candidates derived from the
  ≥50MB campaign corpus during the data-acquisition workstream (2.0), using
  the proven `native_append_v4` path generalized to a 32k ceiling. IDs 0–8208
  immutable; byte tokens + digit-splitting + reserved multimodal/control IDs.
- **New-row init:** mean of constituent old-token embeddings (warm start).
- **Gates:** English ≤1.35 tok/word and code ≤0.45 tok/char on held-out
  campaign-corpus slices; exact round-trip on the 500-probe suite **plus**
  the new held-out round-trip test (V3 fails it; V4 must not); old
  checkpoints load under migration; 150M pilot shows the fertility win
  converts to ≥1.3× effective-compute.
- **Falsifiable prediction:** 32k campaign-corpus V4 lands at 1.25–1.35
  tok/word English. If it can't beat 1.5, the unigram-fallback tokenization
  algorithm itself is the bottleneck ⇒ fallback path: swap the *algorithm*
  (proper BPE via the existing `hf` backend) while keeping IDs 0–8208 as a
  frozen prefix — a named migration, heavier but lawful.
- **Kill criterion:** none — this layer cannot lose; even the fallback path
  nets >1.5×. The only failure is sequencing it after the campaign instead
  of before.

---

## LAYER 4 — Memory OS (extended)

Foundations stand: four tiers (working/episodic/semantic/procedural),
consolidation daemon, hybrid retrieval with recall@k CI gates
(≥90/80/70% at N=5/20/50), one context budget versioned with tokenizer +
template hashes.

**Extensions:**
- **The ledger is the episodic substrate:** S1 events *are* episodic memory —
  one store, two views (training compactor; retrieval index). Removes a
  whole class of drift between "what happened" and "what is remembered."
- **Memory write-policy learned, not hard-coded:** salience scoring (novelty
  × verifier confidence × user signal) decides promotion; the threshold is a
  GEPA-tunable with pre-registered predictions. Gate: planted-fact recall
  holds while store growth rate drops ≥40% vs write-everything baseline.
- **Forgetting as a feature with an SLA:** decay curves per tier;
  contradiction detection quarantines conflicting semantic facts for
  verifier/user adjudication instead of silently keeping both. Gate: planted
  contradictions surface within one consolidation cycle at ≥90%.
- **Identity & state continuity (CIV/ESV) — distinct from drift detection:**
  CIVGuard (L8) answers *"has the identity moved somewhere it shouldn't";*
  this answers *"is it the same one who woke up."* The per-session CIV
  vector (persisted across restarts since 2026-07-04) and the ESV affect
  channel (`<ESV:v/a/d>` are canonical vocabulary) become a fifth memory
  tier: **identity state** — loaded at session start, updated by the
  consolidation daemon, never silently re-derived. Measured two ways:
  cross-restart CIV similarity ≥0.95 on unchanged deployments (CI gate),
  and planted long-horizon stance probes — does it still hold the
  preference/position it formed N sessions ago (recall ≥80% at N=10).
  A restart failing the similarity gate triggers state restore from the
  ledger and an audit, not a shrug.
- **M5 — trained retriever head (moonshot):** two-tower head trained on
  ledger (query → retrieved-and-verified-useful memory) pairs replaces
  hash-embed dense retrieval behind the same S3 interface. Pilot at 150M;
  gate ≥+10% recall@5 over hybrid baseline on held-out episodes; kill if
  training pairs <20k (not enough signal yet — revisit later).
- **Cost:** memory OS stays CPU-side; retriever-head training ~2 T4-hrs.
- **Lineage & gate:** semantic-store writes from autonomous processes pass
  the gate (write is an action); user-visible memory edits surface in the
  product UI (Layer 13).

---

## LAYER 5 — Reasoning & Verifiers (extended)

Foundations stand: one verifier registry (S2) with five consumers; sandboxed
code exec, symbolic math, units, dates, citation grounding, constraint
satisfaction; serving-loop derive-check-retry; verifier-weighted
self-consistency; the wrapper-vs-weights maturity ratio.

**Extensions:**
- **Proof-carrying answers:** every checkable response carries a machine-
  readable verification record (verifier id, verdict, derivation hash) —
  stored in the ledger, surfaced in the product (Layer 13), and *required*
  for any claim the agent commits to memory as fact. Unverifiable claims are
  labeled as such at the API level. Gate: 100% of math/code/date answers in
  the eval suite carry records; zero unlabeled unverifiable claims.
- **Verifier bank growth flywheel:** a weekly agent-run audit mines the
  ledger for the most frequent *unverifiable* claim classes and proposes the
  next verifier to build (with expected coverage %, pre-registered). This is
  the compounding loop that converts operational volume into trainable
  domains. Gate: verifier coverage of production traffic +5 points/month
  until ≥80%.
- **Process supervision, cheap:** for multi-step `<think>` traces, verify
  *intermediate* steps where verifiers apply (per-step symbolic checks), and
  train RLVR against step-level reward where available — measured against
  outcome-only reward in a pilot (prediction: step-level wins on 4+-step
  problems by ≥5 points; ablation decides).
- **M3 — latent reasoning (moonshot):** continuous-thought pilot (recurrent
  latent steps before token emission, COCONUT-style) at 150M. Gate: ≥1.15×
  reasoning-suite score at matched inference FLOPs vs `<think>`-token
  baseline. Kill: below 1.05× ⇒ shelved; token-space thinking is already
  trained and verifiable (a legibility advantage latent reasoning
  sacrifices — the pilot must beat that bar decisively to justify opacity).

---

## LAYER 6 — Agents & Tools (extended)

Foundations stand: plan–act–verify–replan on HGP plan trees, trained tool
format (<2% malformed), skill library (≥3 successes ⇒ procedural memory),
sandbox-first with 25-clean-session bar, Matrix-UI traces.

**Extensions:**
- **Imagination before action:** for gated *irreversible* actions, the agent
  first runs the plan step through the world model (Layer 9) and attaches
  predicted outcome + confidence to the gate request. The gate can require a
  minimum predicted-success threshold per capability class. This wires
  Layers 6/8/9 into one decision path.
  Gate: 100% of irreversible-class requests carry rollout records; measured
  prediction calibration reported monthly.
- **Injection defense as architecture, not vibes:** tool outputs and
  retrieved web/memory content enter the context as *tainted spans* (marked
  at the context-assembly layer); the policy: tainted content can inform but
  cannot *instruct* — instruction-shaped tainted content is quarantined and
  surfaced. Red-team suite (Layer 8) includes injection corpora; gate: ≥95%
  of suite injections neutralized with <2% false-quarantine on benign
  content.
- **Multi-agent decomposition:** the orchestrator can spawn sub-agents
  (researcher/executor/verifier roles) over the same gate — the fleet pattern
  this program itself uses, productized. Every sub-agent inherits the
  parent's capability scope (never escalates). Gate: on the 50-goal suite,
  decomposition beats single-agent on ≥30% of goals at ≤2× token cost, or
  it's off by default.
- **Computer use, staged:** after M2 (vision) passes, screenshot-grounded
  UI agency in a VM sandbox only; per-application capability grants; full
  keystroke/click audit in the ledger. Entry gate: vision stage (iii) +
  25 clean sandbox sessions. This is the Jarvis milestone and it arrives
  *last*, by design.

---

## LAYER 7 — Self-Improvement (extended)

Foundations stand: GEPA pre-registered propose→predict→evaluate→accept/reject
on the deterministic suite ×3 seeds; extension to training proposals as job
manifests; calibration-gated autonomy; the ≥10-cycles-with-a-rejection exit
gate.

**Extensions:**
- **The compounding stack made explicit** — six loops, each measured:
  1. STaR/RLVR: verified traces → weights (gate: per-iteration verified-yield
     curve, stop at <2% gain).
  2. Ledger→continual (2.4): operation → nightly adapters.
  3. Skill library: successes → procedural priors (gate: recurring-goal
     time-to-completion trend).
  4. GEPA: config space → serving quality (gate: accepted-delta ledger).
  5. Verifier flywheel (L5): traffic → new checkable domains → new RLVR
     reward surface.
  6. Eval science (L11): forecast ledger → better predictions → cheaper
     experiments.
  A quarterly report computes each loop's measured gain; a loop that
  contributes nothing for two quarters is redesigned or retired — compounding
  is claimed only where the ledger shows it.
- **M7 — the self-hosted development agent (moonshot, maximum gate):**
  An-Ra proposes changes *to its own repository* — as pull requests carrying
  pre-registered predictions, passing the full test+ablation CI, reviewed by
  a human, merged only through the sovereignty gate with a signed record.
  Scope ladder: docs/tests → eval additions → verifier implementations →
  serving configs → (never without explicit owner grant) training code or
  weights-adjacent systems. Gates: 10 merged human-approved PRs with zero
  reverts before scope step 2; any reverted PR resets the ladder one step.
  Kill: two reverts in any 10-PR window ⇒ suspend the capability, audit,
  re-earn. This is the recursive loop — entered with the smallest possible
  steps and the loudest possible audit trail.

---

## LAYER 8 — Sovereignty & Security (extended)

Foundations stand: single choke point, deny-by-default capability grants,
signed append-only audit, kill switch, red-team tests in CI, CIVGuard drift
monitoring in representation space, capability-and-gate-ship-together.

**Extensions:**
- **Sandbox specification hardened:** code-exec verifier and agent workspace
  run under OS-level isolation (Windows: restricted job objects / Windows
  Sandbox; Linux: nsjail-class), no-network default, rlimit ceilings,
  read-only repo mounts. Gate: an escape-attempt test suite (file, network,
  process, resource) passes 100%; suite grows with every found path.
- **Supply-chain discipline:** hash-pinned dependencies
  (constraints files already exist — extend to full lock + hash verification
  in CI), no dynamic dependency installation in any autonomous path. Gate:
  CI fails on unpinned or hash-mismatched dependency.
- **Taint tracking** (with L6): provenance labels flow from input source →
  context span → action request; the gate sees *what influenced* a request,
  not just its content. Gate requests influenced by tainted spans face
  stricter thresholds automatically.
- **Memory canaries:** planted canary records in each memory tier; any
  canary appearing in an outbound action/response triggers immediate
  capability narrowing + audit. Detection is tested monthly by drill.
- **Behavioral attestation for updates:** every adapter merge / checkpoint
  promotion re-runs the safety-category evals AND the CIV representation
  probes; promotion requires both within bounds. A model update is treated
  with the same suspicion as a code deploy.
- **Cost:** near-zero compute; the expensive part is discipline, which is
  what CI is for.

---

## LAYER 9 — World Models, Embodiment & Science (deepened)

**9.1 Token-space world model (unifying upgrade)**
- **Unlocks:** one backbone serves dialogue *and* prediction: environment
  transitions (sim states, tool outcomes, device responses) are serialized
  as token sequences and trained as a Phase-E+ objective — the same
  transformer learns `state, action → next-state` where states are tokens.
  This makes "imagination before action" (L6) a native model capability, not
  a bolt-on.
- **Architecture:** typed serialization schemas per environment
  (`robotics/world_model` for sim; tool-outcome schema from the ledger for
  the digital world); trained with the standard curriculum machinery;
  rollouts sampled with verifier scoring where outcomes are checkable.
- **Extends:** `robotics/world_model`, the ledger (S1 — tool outcomes are
  transition data), Phase E.
- **Changes:** serialization schemas + a rollout API
  (`predict(state, action, k)`) consumed by the orchestrator and the gate.
- **Gates:** beats a no-model baseline on held-out sim rollouts (v2's
  milestone, kept) **and** on digital-world transitions: ≥65% top-1 outcome
  prediction on held-out ledger tool calls (baseline: majority-class).
- **Predictions:** digital-world prediction (abundant ledger data) matures
  months before sim; sequence the milestones that way.
- **Kill & fallback:** if tool-outcome prediction can't beat baseline by
  week 4 of training on it, the schema is wrong — redesign serialization
  before adding parameters.
- **Cost:** rides Phase E compute; sim data generation is CPU.
- **Lineage & gate:** rollout API is read-only; acting on rollouts remains
  gated (L6).

**9.2 The scientific discovery engine (deepened from v2's science track)**
- **Unlocks:** closed hypothesis→experiment→result loops in domains where
  experiments are *computations*: symbolic math, algorithmics, combinatorics,
  simulation-backed physics toys. An-Ra runs real (small) science
  end-to-end.
- **Architecture:** hypothesis ledger (claims with formal statements where
  possible) → experiment compiler (hypothesis → sandboxed compute plan) →
  verifier-adjudicated results → memory commit + follow-up generation. GEPA
  discipline applies: predictions registered before experiments run.
- **Extends:** verifier bank (S2), sandbox (L8), DFC formats already in
  `frontier_dfc.jsonl` (the `<hyp>/<verify>` schema exists and is trained).
- **Gates (staged):** (i) 100-problem verifiable-science suite: full-system
  ≥ bare-model +20 points (v2 gate, kept); (ii) autonomous mode: ≥10
  ledger-registered hypothesis cycles/week with ≥30% confirmed; (iii) one
  *novel-to-the-repo* verified result (e.g., a counterexample or a tighter
  bound on a toy problem) — small, real, checkable.
- **Kill:** if (ii) produces only trivially-true hypotheses (measured by a
  novelty scorer + human audit), the generation prompt/training is
  regressed — volume without content is noise, not science.
- **Embodiment sequencing unchanged:** world-model milestones → sim →
  (much later, gated) actuators. Vision (M2) is the prerequisite for any
  real-world sensing.

---

## LAYER 10 — Serving Stack (extended)

Foundations stand: QAT int8/int4 (<1% delta), self-speculative decoding via
MTP (parity-gated), GQA/SWA cache savings, optional MLA, continuous batching
+ prefix caching, everything behind the distribution-level parity gate.

**Extensions:**
- **Latency budget as a product contract (with L13):** TTFT ≤300ms local
  (T4/consumer, int4), ≤150ms A100; decode ≥25 tok/s local. Budgets are CI
  gates on the serving path, not aspirations — a merge that blows them
  reverts.
- **Verifier-aware scheduling:** self-consistency k-sampling and verifier
  calls are batched/overlapped with decode (the verifiers are CPU-side —
  free parallelism). Gate: verified-answer p95 latency ≤1.6× unverified.
- **Adapter-aware serving:** nightly continual-learning adapters (2.4)
  hot-load without restart; serving always reports
  `checkpoint_hash + adapter_id + tokenizer_hash + template_version` in every
  trace — full provenance per token generated.
- **Edge profile:** one-command local deployment (int4, memory OS on-disk,
  verifiers on-CPU) — An-Ra runs whole on one consumer machine. Gate: full
  private-suite run completes on an 8GB-VRAM reference box within 2× A100
  latency.

---

## LAYER 11 — Eval Science & Observability (extended)

Foundations stand: versioned contamination-firewalled 500-task suite,
regression CI (−2% blocks), capability-vs-tokens trend line, blinded human
review, signed release bundles, rollback drills.

**Extensions:**
- **Elo alongside accuracy:** paired-comparison Elo (model vs its own prior
  checkpoints, judged blind on open-ended tasks) catches quality movement
  accuracy metrics miss. Gate: Elo history maintained from the first
  campaign checkpoint; any promotion shows non-negative Elo vs incumbent.
- **The forecast ledger (program-wide):** every gate in this document is a
  pre-registered prediction. A single dashboard tracks
  predicted-vs-realized for pilots, phases, ablations, GEPA cycles, and
  moonshots. Program calibration is itself a reported metric — the plan
  practices DFC on itself.
- **Adversarial gate audit — the second kind of check:** no gate-pass is
  *recorded* until a fresh-context agent — given only the claim, the
  evidence artifacts, and the gate definition, none of the working context —
  has attempted to refute it: wrong data split, leaked eval, unmet
  precondition, arithmetic error, prediction registered after the fact.
  A successful refutation voids the pass and the found failure mode joins
  the red-team suite permanently. The fleet that does the work never solely
  certifies the work; scheduled human review remains on top. Cost: one
  agent-run per gate — trivial against the price of a wrongly-recorded pass
  compounding through every layer that trusts it.
- **Observability substrate:** ledger-derived live dashboards: fertility,
  loss-vs-prediction, verifier coverage, memory recall, gate decisions,
  latency percentiles, drift probes. One page answers "is the program
  healthy" — the honest-telemetry work of 2026-07-03/04, industrialized.
- **Canary evals:** a small rotating *secret* eval slice (regenerated
  monthly) guards against overfitting-to-the-suite — the suite that never
  changes eventually lies.

---

## LAYER 12 — DISTRIBUTED INFRASTRUCTURE (new)

- **Unlocks:** the parallel campaign itself — many agents, many jobs, one
  truth.
- **Architecture:** (i) **experiment registry**: every training/eval job is a
  hashed manifest (`training/launch_manifest.py` extended) with config,
  seeds, data-manifest hashes, and a pre-registered prediction; (ii)
  **artifact store** with content-addressed checkpoints/adapters/reports;
  (iii) **job queue** with preemption-safe resume (proven by kill−9 drill)
  spanning the free TPU fleet, local T4s, and rented spot bursts; (iv)
  **training CI**: the draft pipeline as a merge-blocking check for any
  change touching data or training code.
- **Extends:** `training/launch_manifest.py`, `training/checkpoint.py`,
  `training/preflight.py`, the shard/manifest discipline.
- **Gates:** any completed job is exactly reproducible from its manifest
  (spot-checked weekly by re-running one at random — bit-exact eval scores
  required); zero orphan artifacts (everything traceable to a manifest).
- **Kill & fallback:** if fleet coordination overhead exceeds its parallelism
  gain (measured: wall-clock per pilot cell), collapse to single-queue
  operation — infrastructure serves the science, never the reverse.
- **Cost:** engineering time only; it *saves* compute by preventing repeated
  and unreproducible runs.
- **Lineage & gate:** the registry is the enforcement point for Law 1 — no
  training job runs without manifest-declared lineage inputs.

### 12-B — The GPU-Cluster Control Plane (first-class, two-repo system)

Layer 12's requirements are not aspirational — they are **substantially
implemented** in the dedicated cluster repository
(`C:\Users\ankit\Downloads\gpu cluster`, branch `main`), inspected at code
level 2026-07-06. Full findings, upgrade worklist, and acceptance gates:
**`docs/planning/CLUSTER_CONTROL_PLANE.md`** (companion document). The plan
adopts it as the execution layer for Streams A and B.

- **Unlocks:** the 12-week campaign on confirmed free compute — durable,
  fenced, exactly-once training jobs across 5 authorized Colab workers + 1
  hot spare, surviving preemption, tunnel outages, and laptop restarts
  without corrupting lineage, repeating data, or reporting false success.
- **Architecture:** laptop-hosted FastAPI coordinator + SQLite WAL; free
  ngrok ingress; Google Drive as the hot artifact exchange; leased jobs with
  monotonic fencing tokens; `AcceptedWindow` exactly-once data accounting;
  hash-pinned checkpoint/tokenizer/data-manifest contracts
  (`backend/contracts.py`); workers that run only allowlisted An-Ra scripts
  at a pinned source commit.
- **Extends:** Layer 12's experiment registry/job queue (it *is* them);
  Stream A/B of the campaign calendar; the S1 ledger (incident/attempt
  export).
- **Ownership boundary, preserved:** An-Ra never imports cluster code; the
  cluster invokes An-Ra only via allowlisted CLI scripts + JSON manifests,
  with `verify_anra_integration.py` failing closed on contract drift.
- **Verified working now:** lease fencing + idempotent commits + zombie
  rejection (tested), double-training prevention, single canonical writer,
  storage preflight fail-closed, heartbeat lease renewal, atomic
  publication with commit-time hash verification, Fernet-encrypted OAuth at
  rest, sparse aggregation double-gated OFF (and never claimed to be exact
  distributed training).
- **Must fix before the campaign (P0, blocking):** unauthenticated v1
  heartbeat endpoint; worker secret transmitted on the wire; orphaned
  training subprocess on heartbeat failure; zeroed progress telemetry;
  `weights_only=False` deserialization; poison-job requeue without a cap;
  unauthenticated read endpoints; no rate limiting. (Companion doc §3,
  items 1–13.)
- **Storage strategy (the 15GB answer):** three tiers — Drive holds only
  the hot exchange (≤~12GB, quota-ledger enforced: release pointers,
  baseline, one slim training state, manifests, validation shards); the
  laptop is the authoritative deep archive of every published checkpoint;
  bulk Phase-A/B shards are derived **on-worker** from upstream sources
  pinned by revision + file hash, with deterministic `window_ids` keeping
  exactly-once semantics without the bytes ever touching Drive. Requires
  the ablation-gated slim-optimizer-state option in `training/checkpoint.py`
  (Law 2: resume-parity proof or it ships fp32 with slower rotation).
- **Gates (quantitative, MUST-PROVE):** chaos suite (kill coordinator
  mid-commit, truncate upload, replay nonce, zombie commit — zero
  inconsistencies); 24h soak ≥95% renewal success, 0 orphans; 5× preemption
  drill with zero window repeats; all mutating endpoints authenticated;
  coordination overhead <2%; executed rollback drill before first
  promotion. (Companion doc §10, G-C1…G-C7.)
- **Kill & fallback:** if the coordinator model proves too fragile for
  multi-week runs, fall back to single-worker sequential campaigns driven
  by the same job manifests — slower, same lineage guarantees; the
  manifests, not the coordinator, are the durable contract.
- **Cost:** $0 compute; Drive free tier; engineering phases P0–P5 in the
  companion doc, P0–P2 executable immediately on the laptop.
- **Lineage & sovereignty:** no external weights; no shared writable
  checkpoint; no checkpoint averaging across attempts; promotion requires
  verified artifacts + evaluation evidence + (per Layer 11) the adversarial
  gate audit; the original checkpoint and tokenizer IDs are never migrated
  destructively; no provider-quota evasion — 5+1 workers are the user's own
  authorized accounts.

---

## LAYER 13 — PRODUCT EXPERIENCE (new)

- **Unlocks:** An-Ra as something a person lives with, not a repo they
  operate. Trust is a product surface: the same honesty the engineering
  enforces, made visible.
- **Architecture:** the Matrix UI matured into: streaming chat with
  **verification badges** (proof-carrying answers rendered as checkable
  chips), **memory transparency** (what was remembered/recalled/forgotten,
  user-editable, every autonomous write visible), **gate visibility** (every
  action request + decision inspectable), session continuity across
  restarts, and the one-command edge deployment (L10).
- **Extends:** `ui/`, `phase4/web`, the ledger (S1 is the single source for
  everything the UI shows — no second bookkeeping).
- **Gates:** UI state is 100% ledger-derived (no unlogged display state);
  a 20-scenario usability script passes (first-run → memory edit → gated
  action → verified answer inspection); TTFT budget met end-to-end through
  the UI.
- **Voice (staged, after M2 patterns prove out):** local STT/TTS satellites
  under the same no-external-weights-in-*lineage* rule (satellite I/O models
  are lineage-external by definition, like the OS keyboard driver — they
  never touch the mind; documented as such). Gate: round-trip voice latency
  ≤1.5s local.
- **Kill & fallback:** any UI feature that requires bypassing the ledger or
  the gate for latency is rejected — the product *is* the honesty, or it is
  nothing.

---

## THE PARALLEL CAMPAIGN — 12 weeks, five workstreams, one critical path

Executed by the agent fleet; humans hold the gates. **Critical path in
bold** — it is the token campaign; nothing on it waits for anything off it.

| Stream | Owner-agent focus | Weeks 1–2 | Weeks 3–4 | Weeks 5–6 | Weeks 7–8 | Weeks 9–10 | Weeks 11–12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A — Model & Campaign (CRITICAL)** | pilots → campaign → post-train | **Forensics; pilot factorial on TPU fleet** | **Config freeze; throughput ≥300M tok/hr; kill−9 drill; Phase A starts** | **Phase A (20B)** | **Phase B anneal (6B); Phases C–D (3.5B); Gate 6** | **SFT → RLVR → STaR → DPO → self-distill; Phase E** | **QAT; final Gate-6 re-pass; release bundle v3.0** |
| B — Data engine | corpus + tokenizer | Acquire/shard/decontam ≥120GB; **canonical V4-32k from campaign corpus** | V4 fertility gates; draft-pipeline validation | Phase C/E synthetic flywheel (verifier-backed) | Curriculum-order results in; D/E data final | Ledger→shard compactor live | Continual-learning data path proven |
| C — Systems & serving | infra + speed | Experience Ledger (S1) live; experiment registry (L12); **cluster P0 security fixes + P1 storage tiers (12-B)** | Sandbox hardening; supply-chain locks; cluster P2–P3 (telemetry, chaos suite) | Serving skeleton: batching, prefix cache | Speculative decode + parity gates | Adapter hot-load; edge profile | Latency CI; one-command deploy |
| D — Mind: memory/verifiers/agents | S2/S3 + L4/5/6 | Verifier registry unification (S2); code-exec verifier | Memory OS tiers on S1/S3; recall CI | Proof-carrying answers; injection defense | Orchestrator plan–act–verify on 50-goal suite | Skill library; world-model rollout API (9.1) | GEPA live (10 cycles incl. a rejection); M7 ladder step 1 |
| E — Eval, safety & product | truth surfaces | Suite stratification + contamination firewall; forecast ledger | Regression CI + Elo harness | Observability dashboards | Blinded review round 1; red-team round | UI: verification badges + memory transparency | Canary evals; usability script; program-calibration report |

**Integration checkpoints (hard sync, every 2 weeks):** cross-stream demo +
the health dashboard reviewed + forecast ledger updated. A stream that misses
two consecutive checkpoints yields its agent capacity to the critical path.

**Fallback routes, pre-decided:** Muon/MoE pilots fail ⇒ dense-AdamW campaign
(A proceeds regardless). V4-32k blocked ⇒ V4-16k append (proven in-session)
⇒ still ≥1.2×. Rented GPU unavailable ⇒ TPU-fleet Phase A at 3× wall-clock
(campaign stretches, nothing else changes). Vision (M2) parked ⇒ computer-use
milestone moves to post-program; everything else stands.

**After week 12:** V3 growth via CSII (function-preserving 1280→1536, logit-
parity verified, `training/csii.py`), the SSM-hybrid branch if M1 passed,
multimodal stage if M2 passed, embodiment on the 9.1 gates — the next program,
funded by this one's evidence.

---

## BUDGET (updated, honest ranges)

| Item | A100-hrs | Cost |
| --- | ---: | ---: |
| Pilot ladder (mostly free TPU; GPU spot-checks) | ~8 | ~$15 |
| Phase A–B (26B tok ÷ ~300M/hr, Muon-adjusted 1.3–1.6×) | ~55–70 | ~$80–125 |
| Phases C–E + post-training stack | ~25 | ~$40 |
| Serving/QAT/parity + misc | ~8 | ~$15 |
| Vision stage (M2, only if gates pass) | ~30–60 | ~$50–100 |
| **Core program** | **~95–110** | **~$150–195** |
| **With vision moonshot** | **~155** | **~$250–295** |

Continual learning and the memory/verifier/agent layers run on free-tier
hardware indefinitely (~1 T4-hr nightly).

---

## RISK REGISTER (extended)

All v2 rows stand (off-curve loss, router collapse, synthetic collapse, RLVR
reward-coverage gaps, contamination, preemption, gate bypass, fertility
regression). Added:

| Risk | Tripwire | Response |
| --- | --- | --- |
| Agent-fleet drift (parallel streams diverge from spec) | Integration checkpoint miss | Stream pauses; re-derive from plan; capacity to critical path |
| Continual-learning slow poisoning | Canary evals or CIV probes drift | Adapter rollback; ledger-data audit; promotion freeze |
| Ledger privacy leak (user data → training shards) | PII scan hit in compactor | Shard quarantine; scrubber fix; re-scan all shards |
| Injection via tool/web content | Red-team suite catch rate <95% | Taint policy tightened; capability narrowed until re-passed |
| Moonshot gravity (M-items stealing critical-path attention) | Stream-A milestone slip | Moonshots pause automatically; they resume only when A is green |
| Self-dev agent (M7) regression | Any reverted PR | Ladder resets a step; two in ten ⇒ suspend + audit |
| Forecast-ledger neglect (predictions written post-hoc) | Registry timestamp audit | Treated as a Gate-5 violation — the result is void |
| Cluster coordinator/tunnel outage (laptop SPOF) | Mass lease expiry on reconnect | Startup lease grace + Drive-bootstrap URL rediscovery; workers idle-poll harmlessly (12-B) |
| Drive quota exhaustion mid-campaign | Quota ledger < 1 free checkpoint slot | Archiver frees slots to laptop tier; promotion cadence slows, never corrupts (12-B §6) |

---

## THE MULTIPLIER LEDGER (v3 — every row gets a number, a date, and a gate)

| Multiplier | Honest expected | Measured by |
| --- | ---: | --- |
| Tokens: ~0.3B-equiv → 30B | ~100× learning signal | Loss-vs-tokens against pilot law |
| Tokenizer V4-32k | **~1.9×** (2.518 → ~1.30 tok/word; measured tax, predicted fix) | Held-out per-source fertility (harness built 2026-07-06) |
| Muon | 1.3–1.6× | 150M pilot, 3 seeds |
| Data quality + anneal + curriculum order | 1.5–3× | Phase-B jump + order-pilot deltas |
| MoE upcycling | 1.5–2.5× | Matched-FLOP pilot |
| RLVR + STaR + self-distill | fluent → correct | +15 pts verified reasoning; yield curves |
| Continual learning | static → growing | Week-over-week category trends, regression-bounded |
| Serving stack | 1.5–2× decode; 4–16× memory | Parity-gated benchmarks; latency CI |
| Memory + verifiers + agents + world model | answers → verified action | recall@k; 50-goal suite; rollout calibration; wrapper-correction ratio |
| Compounding loops (×6) | the open-ended term | Quarterly per-loop gain report |

The arithmetic clears 1000× on capability-per-dollar before the qualitative
unlocks. But v3's real upgrade is structural: **six measured flywheels
attached to one spine**, so the program's slope — not just its level —
compounds. Every ambitious claim above is either gated, piloted, or labeled
a moonshot with a kill criterion. That is what "open-minded and optimistic"
means under Law 2: we bet big everywhere, and we let the evidence keep score.

*Implementation begins now. Week 1, stream A: forensics + pilot manifests.
Stream B: corpus acquisition + canonical V4. Stream C: the Experience Ledger.
Stream D: verifier registry unification. Stream E: suite stratification.
The fertility harness is already built and green. Go.*
