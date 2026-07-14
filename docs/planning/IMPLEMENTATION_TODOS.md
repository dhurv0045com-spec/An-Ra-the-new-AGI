# MASTER_UPGRADE v3 - Implementation TODOs

This is the live execution board for `MASTER_UPGRADE.md`. A checked box means
code plus focused tests exist; it does not waive a quantitative campaign gate.

## P0 - Week 1, shared foundations

### Stream A - Model and campaign

- [x] Obtain or locate the real 500M checkpoint without adding it to git.
      *(Owner artifact located 2026-07-10 at
      `C:\Users\ankit\Downloads\anra_frontier_500m.pt`; 2,000,680,247 bytes,
      SHA-256 `648354a42d68c22769450a3aaa249e93689b21fbe72e68b07dcc15c6f7f4d393`.
      Recorded source commit `e8d90d9...` is an ancestor of this branch.)*
- [x] Freeze checkpoint, tokenizer, config, and corpus-manifest hashes.
      *(`scripts/freeze_baseline_hashes.py` → `output/v2/baseline_freeze.json`,
      re-executed 2026-07-10 with checkpoint, tokenizer, config, and corpus
      identities all frozen.)*
- [x] Run exact tensor accounting and the 500 tokenizer probes.
      *(499,167,047 parameters; all 608 target tensors accounted for; zero
      missing, unexpected, or shape-mismatched tensors; exact core/native
      load. All 500 tokenizer probes match fingerprint `db1075ad...`.)*
- [x] Run deterministic generation and the 200-prompt recovery gate.
      *(Driver ready: `scripts/run_checkpoint_forensics.py --run-generation`
      — greedy, seed 0, cache off, coherence gate 80%, undertraining decision
      rule. Full CUDA gate launched 2026-07-10 on the RTX 4050; acceptance
      completed 2026-07-10: 200 diagnostic + 200 native + 200 replay prompts;
      exact load/replay passed, but coherence was 0.0% versus the 80% gate.
      Schema-7 CUDA replay completed 2026-07-11 with 0.0% acceptance and 100%
      EOS failure across diagnostic/native evidence.
      The final verdict is undertraining, not recovery; see
      `output/v2/stream_a_forensics.json`.)*
- [x] Repair the checkpoint-era temperature-control trainability contract.
      *(Schema 7 replaces the neutral `layer_temperature_bias` buffer with a
      bounded positive log-parameter, migrates legacy scales deterministically,
      rejects invalid scales, regularizes and telemeters the realized values,
      and routes the 28 parameters through the subsystem optimizer group.
      Current contract: 499,167,075; legacy artifact: 499,167,047. The code
      candidate is complete; three-seed removal-vs-trainable ablation remains a
      campaign selection gate.)*
- [x] Separate answer quality from prompt/scaffold training and validation loss.
      *(Conversational packing now emits explicit token-level answer masks to
      GPU and TPU paths. Training and immutable validation report total,
      weighted, answer-only, and scaffold-only CE with exact token counts;
      schema-7 checkpoints retain `best_answer_validation_loss`. Raw foundation
      shards correctly declare no answer boundary. Full suite: 593 passed,
      1 skipped.)*
- [x] Make conversational validation immutable and promotion evidence-derived.
      *(D/E examples are grouped by declared source/content identity or
      normalized record SHA-256 before tokenization; train/validation datasets
      are distinct and a hashed zero-overlap split manifest is emitted. Raw
      validation remains manifest-bound. Stage gates compute same-identity,
      newer, per-domain/answer regression evidence and reject trusted boolean
      claims. Full suite: 597 passed, 1 skipped.)*
- [x] Implement QK-Norm and the 3:1 sliding-window/full-attention backbone.
      *(Parameter-free per-head QK-Norm and explicit 1024-token sliding-window
      layers with every fourth layer full-attention now live in the canonical
      model contract. Old checkpoints restore their recorded legacy semantics;
      no silent attention migration occurs. `p150-qknorm-off` and
      `p150-swa-full` are now real trainer-mapped ablations. RTX 4050 smoke on
      the 57,374,343-parameter pilot produced finite logits/loss/gradients at
      498.7 MiB peak; three-seed acceptance evidence remains mandatory.)*
- [x] Implement executable MTP and sparse-upcycled MoE pilot axes.
      *(MTP adds two tied-vocabulary heads for +2/+3 future tokens and an
      explicit 0.2-weight trainer loss. MoE replaces each dense MLP with eight
      top-2 routed clones plus one shared expert; the step-zero function is
      exactly dense-equivalent and expert load uses optimizer-bound score-bias
      updates with no auxiliary loss. Signed axes reach the canonical trainer,
      parameter totals include all inactive experts, and checkpoint feature
      mismatches fail closed. RTX 4050 smokes produced finite losses/gradients.
      Acceptance still requires matched-token/active-FLOP three-seed pilots.)*
- [x] Implement the three curriculum-order pilot schedules.
      *(A compact-range deterministic sampler implements code-before-prose,
      math-density-ramp, and identity-mix-late over the signed expected-token
      budget. Relative multipliers preserve the immutable corpus distribution
      at 1.0, targeted source classes are mandatory, validation is untouched,
      and checkpoint training-recipe metadata rejects order changes on resume.
      All 17 non-V4/non-moonshot factorial cells are now trainer-mapped.)*
- [x] Convert the pilot factorial into pre-registered launch manifests, three seeds each.
      *(`training/pilot_factorial.py`: 23 cells, seeds 1301/2602/3903, signed
      schema-3 manifests with immutable data-manifest hashes and explicit
      train/validation roles, Law-1 scratch lineage; each seed is a separate
      signed run with a unique artifact path (69 manifests total), rather than
      three labels attached to one unseeded process;
      owner emits the set with
      `py -3.14 -m training.pilot_factorial --owner-authorized` once
      `ANRA_MANIFEST_SIGNING_KEY` is set. Exact `pilot-50m` and `pilot-150m`
      model profiles now build and run. Manifests bind distinct immutable
      train/validation shard inputs, an explicit V3-or-V4 tokenizer artifact,
      and exact token caps; optimizer fallbacks and unimplemented axes fail
      closed instead of silently running a baseline. V4 cells bind separately
      tokenized V4 shards and cannot inherit the process-global V3 hash.)*
      *(`scripts/run_pilot_queue.py` validates signatures, immutable inputs,
      forecast timestamps, blockers, seed/run identity, and completed artifacts;
      writes a resumable queue plan and per-run status/log evidence; excludes
      moonshots by default; and requires explicit `--execute` plus CUDA.)*
- [x] Record forecast-ledger predictions before every pilot launch.
      *(`training/forecast_ledger.py`: hash-chained append-only ledger;
      `build_pilot_launch_manifests` registers the forecast before the
      manifest exists and every manifest must pass the Gate-5
      `audit_pre_launch` timestamp audit.)*

### Stream B - Data engine and canonical V4

- [x] Define pinned, license-checked upstream corpus manifests.
      *(`training/corpus_manifest.py`: 7 pinned sources with license allowlist +
      immutable-revision checks + normalized weights; emits
      `output/v2/data_manifests/upstream_corpus_manifest.json`, content-hashed.)*
- [ ] Acquire, shard, deduplicate, and decontaminate at least 120 GB clean text.
      *(Execution update, 2026-07-10: one managed 30 GB pinned-source tranche
      is running after a 313 GB-free-disk preflight. The downloader now uses
      the standard streaming dataset contract without remote-code execution;
      completion, immutable manifests, source mix, and the full 120 GB target
      remain mandatory before this box can be checked.)*
      *(Blocked on network + storage + hours. Machinery exists:
      `scripts/download_training_data.py` — pinned revisions, MinHash dedup,
      PII/contamination cleaning, license gate, token-shard publisher.)*
      *(2026-07-11 audit: the interrupted 17.50GB artifact contains 3,347,036
      structurally valid, exactly deduplicated FineWeb-Edu records and no other
      source class. The WAL-safe resume index is finalized; safe acquisition of
      the remaining 30GB tranche sources is active.)*
      *(2026-07-13 repair: the first resume reached 21.58GB before exposing two
      upstream failures. Gated Stack v2 and script-only Dolma were replaced by
      immutable Common Pile Stack-v2-open-code and openly licensed ArXiv
      parquet revisions. Row licenses now fail closed individually. A real
      `120gb` native profile, live progress report, fsync-before-index commits,
      and chained incremental append audits are implemented. A fresh audit is
      complete: 21,582,998,123 bytes / 4,113,170 records / zero failures. The
      repaired 30GB resume passed that boundary and is actively appending from
      the resume-safe index toward the 27GiB native target. A fail-closed
      `scripts.execute_stream_b` continuation is queued to build the slice,
      canonical V4, and both tokenizer-specific shard families afterward. A
      second continuation starts the full 120GiB pinned acquisition only after
      that report reaches `status=complete`.)*
- [x] Produce the at-least-50 MB tokenizer campaign slice with held-out source splits.
      *(Builder shipped: `scripts/build_campaign_slice.py` — deterministic
      per-source held-out split, disjointness + >=50MB gate. Executed on the
      local corpus (3.4MB train, held-out disjoint); the 50MB itself is
      corpus-blocked.)*
      *(The large-corpus path is now bounded streaming rather than loading the
      17.5GB JSONL into RAM; readiness additionally requires all seven source
      classes and <=2-point mix deviation.)*
      *(2026-07-13: the canonical instruction source is acquired at its pinned
      revision: 120,000 SmolTalk rows / 230,753,834 bytes. A new verifier-bank
      builder emitted 2,249 unique, deterministically verified DFC records /
      4,195,921 bytes; unverified historical synthetic DFC is now rejected.
      Identity is explicitly replay-weighted and reported. The remaining slice
      blockers are the queued code and science downloads.)*
      *(Executed 2026-07-15: the fully audited 27.002 GiB native tranche produced
      a 64.024 MiB bounded slice across all seven required source classes. The
      deterministic held-out sets are disjoint, the 55/15/12/8/5/3/2 mix is
      within the two-point gate, and `ready_for_v4=true`. The bound canonical
      32,768-ID V4 was rebuilt with the V3 prefix unchanged, byte round-trip
      proof, 530,602,595-parameter contract, and 38.9932% projected token
      reduction.)*
- [x] Generalize the proven append migration to the canonical 32k V4 ceiling.
      *(`build_append_only_v4`/`audit_token_fertility` now take a `target_vocab_size`
      ceiling; canonical `CANONICAL_V4_VOCAB_SIZE = 32_768` with a pinned
      530,602,595-param contract; 16,384 retained as the proven fallback.
      Runtime/checkpoint/ssg gates accept both via `is_v4_vocab_size`. CLI:
      `scripts/build_v4_tokenizer.py`.)*
- [x] Prove IDs 0-8208 unchanged, round-trip, checkpoint migration, and fertility gates.
      *(`tests/test_canonical_v4_32k.py`, 10 tests: frozen-prefix identity,
      byte-safe round-trip, 32k param contract, checkpoint migration preserving
      legacy rows bit-for-bit with deterministic mean-init of appended rows,
      and the fertility-audit ceiling gate. Local-corpus audit confirms V4
      needs the >=50MB campaign corpus: 816k units / 3.7% reduction, ineligible.)*
- [ ] Run the 150M three-seed fertility-to-effective-compute pilot.
      *(Pre-registered in Stream A as `p150-v4tok`, three seeds, forecast in the
      ledger. The 32k ceiling is now built, so the blocker narrows to the
      canonical V4 candidate (>=50MB corpus) + GPU.)*

### Stream C - Systems and serving

- [x] Add schema-versioned append-only Experience Ledger events.
- [x] Add stable input hashes and tamper-evident event hashes.
- [x] Add validated trace replay.
- [x] Wire main generate/chat traces into the ledger.
- [x] Wire explicit tool/agent dispatch into the ledger.
- [x] Wire verifier outcomes into the ledger.
- [x] Wire owner-auth and sovereignty gate decisions into the ledger.
- [x] Make capture fail-open for serving and test injected write failure.
- [x] Add verifier/gate/PII-filtered training compaction with hashed manifests.
- [x] Add shard rotation, sealing, retention, and sealed-shard manifest verification.
- [x] Add crash/flush stress and sustained p50/p99 write-overhead CI benchmarks.
- [x] Implement Cluster P0 security fixes in the separate control-plane repository.
- [x] Implement Cluster P1 three-tier storage and quota ledger.

### Stream D - Mind: verifiers, retrieval, memory, agents

- [x] Add one auto-registering verifier registry and route the compatibility facade through it.
- [x] Define the single verifier request/result protocol.
- [x] Add registry conformance tests and duplicate-name rejection.
- [x] Move legacy handler branching into independently registered verifier modules.
- [x] Route training reward, inference DFC, synthetic validation, agents, and GEPA through it.
- [x] Harden code execution with the sandbox policy and resource limits.
- [x] Define the shared retrieval protocol for dense plus BM25 backends.
- [x] Route inference memory, agent skills, corpus dedup, and citation grounding through S3 adapters.
- [x] Bind memory writes, recalls, edits, and forgetting events to ledger trace IDs.

### Stream E - Evaluation, safety, and product

- [x] Stratify the private suite by capability and contamination source.
- [x] Add source-hash contamination firewall checks to evaluation CI.
- [x] Add the forecast-ledger schema and pre-launch timestamp audit.
      *(Delivered with Stream A: `training/forecast_ledger.py`, schema v1,
      hash-chained entries, `audit_pre_launch` voids post-hoc forecasts.)*
- [x] Establish regression CI and blinded Elo harness baselines.
- [x] Define UI ledger projections for verification, memory, and gate visibility.

## P1 - Weeks 3-6

- [ ] Freeze the winning model/curriculum configuration from three-seed pilots.
- [ ] Prove throughput and kill-minus-nine checkpoint recovery before Phase A.
      *(`training/recovery_drill.py` now performs a process-termination and
      exact checkpoint/optimizer restoration drill; the required measured
      frontier throughput and real-run recovery evidence remain blocked.)*
- [ ] Complete cluster telemetry, chaos suite, and 24-hour soak gates.
      *(`gpu cluster`: P2 now records append-only worker telemetry, derives
      throughput only from counter deltas, serves an operator-only campaign
      dashboard, and fences pause/drain/halt/resume plus worker replacement.
      P3 has fail-closed G-C1/G-C2/G-C3 evidence evaluators; simulated evidence
      cannot pass. P4 now requires An-Ra's signed, checkpoint-bound Gate-6,
      reproducibility, adversarial, and rollback evidence before promotion and
      can atomically restore a validated prior full checkpoint. The 27-test
      cluster suite passes. The still-open box requires actual
      Drive/worker chaos timing, a live 24-hour two-worker soak, and five live
      preemption drills.)*
- [x] Build hybrid retrieval and memory-tier recall CI.
      *(`evaluation/retrieval_recall.py`: deterministic recall@5/20/50 gate
      over the shared retriever contract, with tested pass/fail thresholds.)*
- [x] Ship proof-carrying answer contracts and injection-defense tests.
      *(`runtime/answer_contracts.py`: tamper-evident hash-only answer
      contracts; tainted retrieved memory is removed before chat context; the
      contract is ledgered and returned by `/generate` and `/chat`.)*
- [x] Build continuous batching, paged KV, and prefix-cache serving skeleton.
      *(`inference/serving_runtime.py` provides lineage-safe FIFO batching and
      bounded opaque KV pages; `inference/prefix_cache.py` remains the shared
      prefix-cache substrate.)*
- [ ] Run Phase A only after all blocking preflight gates pass.

## P2 - Weeks 7-12

- [ ] Complete Phases B-E and the full Gate 6 evaluation.
      *(The staged campaign machinery is present in `training/stages.py`; it
      cannot truthfully advance without the checkpoint, licensed campaign
      corpus, compute, and per-phase evaluation artifacts.)*
- [ ] Complete SFT, RLVR, STaR, DPO, and self-distillation with ablations.
      *(`training/dpo.py` and `training/post_training_ablations.py` complete
      the missing DPO objective and five-method evidence gate; actual training
      and ablation metrics remain compute-bound.)*
- [x] Add speculative decoding, adapter hot-load, QAT, and parity/latency CI.
      *(`inference/serving_gates.py`: promotion requires speculative benefit,
      token+distribution parity, <=1% QAT drift, and measured latency budget.)*
- [x] Prove plan-act-verify on the 50-goal suite.
      *(`agents/plan_act_verify.py`: ledgered fail-closed runner; 50
      deterministic harness cases pass. A real owner-held suite is still
      required before promotion.)*
- [x] Run GEPA for ten cycles including at least one correctly rejected proposal.
      *(`training/gepa_cycles.py`: proposal-only ten-cycle runner records
      reviewer evidence and requires a non-auto-applied justified rejection.)*
- [x] Ship ledger-derived UI trust surfaces and the 20-scenario usability script.
      *Developer UI renders `/traces/{trace_id}/trust`; `ui/usability.py`
      provides the versioned 20-scenario acceptance script.)*
- [x] Execute canary, adversarial gate, rollback, and release-bundle drills.
      *(`evaluation/release_drills.py` fail-closed canary/adversarial gates;
      rollback and signed-bundle drills are exercised in focused tests.)*

## Moonshots - pilot-gated, never on the critical path

`scripts/run_moonshot_pilots.py --execute-local` executes every local-safe
M1-M7 path, writes `output/v2/moonshot_local_execution.json`, and evaluates
acceptance evidence in `output/v2/moonshot_pilot_status.json`. Smoke evidence
cannot satisfy a campaign gate. Executed 2026-07-10: all seven local paths
ran and passed their smoke checks; M6 alone passed its actual deterministic
acceptance pilot. M1-M5 and M7 remain blocked on external evidence.

- [ ] M1 attention/SSM hybrid pilot.
      *(Local SSM branch executed and is finite/shape-correct. Acceptance
      remains blocked on an exact-150M, three-seed pilot proving >=0.98x
      short-context capability and >=1.5x long-context throughput.)*
- [ ] M2 in-house vision encoder and projector stages.
      *(The in-house encoder/projector path executed without external weights.
      Acceptance remains blocked on Gate 6, licensed image data, in-house
      training, >=30% reconstruction-MSE improvement, held-out 5k R@1 >=40%,
      and the 200-item vision-QA score >=60%.)*
- [ ] M3 latent-reasoning channel pilot.
      *(The recurrent latent path executed and is finite/shape-correct.
      Acceptance remains blocked on exact-150M, three-seed latent and
      token-thinking checkpoints proving >=1.15x reasoning score at matched
      inference FLOPs.)*
- [ ] M4 world-model rollout API and calibration gate.
      *(The offline-only rollout path executed. Acceptance remains blocked on
      a trained checkpoint plus held-out simulation and ledger tool-transition
      evidence: positive baseline gains, digital top-1 >=65%, calibration
      error <=10%, and action success >=70%.)*
- [ ] M5 trained retriever head behind the shared retrieval protocol.
      *(The two-tower S3 protocol path executed. Acceptance remains blocked on
      >=20,000 verified query-memory pairs, a trained checkpoint, and a
      held-out >=10% recall@5 gain over the hybrid baseline.)*
- [x] M6 formal-proof expansion where deterministic verifiers exist.
      *(`training/formal_proof_pilot.py` executed a fixed, hash-addressed
      100-case suite: 50 valid chains and 50 adversarial conclusion-injection
      certificates. The hardened explicit-rules verifier classified 100/100
      correctly (1.00 pass rate vs >=0.95 gate). Evidence is embedded in
      `output/v2/moonshot_local_execution.json`.)*
- [ ] M7 self-development ladder, starting at proposal-only authority.
      *(Proposal-only policy executed and proved no auto-apply or merge.
      Acceptance remains blocked on 10 real human-approved merged PRs, 10
      signed sovereignty records, zero reverts, and zero unauthorized applies.)*

## Review ledger (Fable 5 audit of Codex-checked work)

- **2026-07-13 - acquisition/provenance recovery.** Replaced two inaccessible
  upstream sources with public, exact-commit Common Pile parquet datasets;
  tightened per-row license validation so mixed permissive/disallowed rows
  cannot pass by substring; acquired the pinned 120k instruction tranche;
  generated a 4MB verifier-backed DFC corpus; prevented inferred DFC from
  entering the verified bucket; added explicit identity replay accounting,
  live source/rate telemetry, a 120GB acquisition profile, fsync ordering, and
  hash-chained incremental audit publication. Canonical V4 CLI execution now
  requires the ready seven-source slice and its exact content hash. Token
  publication now creates source-pure shards, materializes a minimum trainable
  identity replay shard, and the deterministic sampler enforces the declared
  campaign mix instead of inheriting raw corpus imbalance. Focused
  data/tokenizer verification: 33 passed. The audit/download and all later
  quantitative training gates remain open until their artifacts complete.

- **2026-07-11 - architecture/data/campaign hardening audit.** Added CUDA
  activation forensics, resumable 17.5GB corpus audit/indexing, exact pilot
  profiles, dense phase isolation, depth-scaled initialization, logit z-loss,
  QK-Norm/SWA, executable MTP/MoE/curriculum axes, per-cell tokenizer/shard
  binding, and fail-closed legacy trainer removal. Final consolidated suite:
  625 passed, 1 skipped; changed-file Ruff and diff checks pass. Data volume,
  V4, signed launches, three-seed outcomes, and soak/recovery remain open
  evidence gates.

- **2026-07-10 - full accumulated implementation audit.** The local mechanism
  layer was rerun end to end (575 non-GPU tests pass, 1 skipped; sibling
  cluster 24 tests pass). Corrected semantic answer-contract verification,
  complete verified-answer latency accounting, causal ablation evidence,
  pre-action authorization for irreversible plan steps, streamed adapter
  hashing, deterministic offline ghost-memory embeddings, bounded repository
  discovery, and local-workspace moonshot CLI resolution. No evidence-bound
  checkbox was changed by this audit.

- **2026-07-06 — Stream C (Experience Ledger, 9 boxes) + Stream D (verifier
  registry, 5 boxes) audited at code level. Verdict: sound.**
  - Verified working: schema-versioned append-only JSONL with tamper-evident
    per-event `event_hash`; validated replay; fail-open serving capture with
    injected-failure test; input stored as hash only (PII-safe); consistent
    `trace_id` threaded across chat/generate/gate; gate decisions classified
    allowed/denied; GEPA now gated on BOTH heuristic score and verifier
    verdict; auto-registering registry with duplicate rejection, alias
    normalization, result conformance ([0,1] score, positive tier, non-empty
    reason); legacy `VerifierHierarchy.score()` correctly routed through the
    registry; `_safe_exec` upgraded to the audit-hook `CodeSandbox` (fs/net/
    subprocess escapes confirmed blocked by live smoke test).
  - **Bug found and fixed (Fable 5):** `compact_for_training` split the
    train/validation firewall on `event_hash`, which embeds each event's
    timestamp+uuid — so the *same prompt* recorded twice could leak across
    both splits, violating the plan's source-hash-split contamination rule.
    Re-keyed the split on `inputs_hash` (input identity); added a regression
    test proving one input never lands on both sides. Also fixed one import-
    order lint in `execution/sandbox.py`.
  - Baseline after review: 477 passed, 1 skipped (was 458); ruff clean.

## Current next executable slice

Stream C is complete at code + focused-test level. Stream A's code-executable
half is done (2026-07-07): baseline freeze executed, forensics driver ready,
pilot factorial pre-registered behind the forecast ledger's Gate-5 audit.
Stream B's code-executable half is done (2026-07-07): the canonical 32k V4
append migration is generalized and proven (IDs 0-8208 frozen, byte-safe
round-trip, checkpoint migration, param contract), pinned license-checked
corpus manifests are emitted, and the campaign-slice + V4-build CLIs are
shipped. The >=50MB slice and bound canonical V4 gate completed on 2026-07-15.
Remaining Stream B work is execution-bound: recover and audit the owner-stopped
append (roughly 29GB), explicitly choose whether the campaign remains at that
volume or resumes toward >=120GB, publish tokenizer-bound shards, then run the
150M V4 pilot.

The legacy checkpoint forensics and CUDA activation profile are complete; the
artifact is a rejected baseline, not a continuation candidate. Remaining owner
action (A-manifests): set `ANRA_MANIFEST_SIGNING_KEY` then
`py -3.14 -m training.pilot_factorial --owner-authorized`; (B-corpus)
allow the managed audited `--profile 30gb --bucket base --resume` worker to
finish, then run `py -3.14 -m scripts.build_campaign_slice` and
`py -3.14 -m scripts.build_v4_tokenizer`. Publish the V3 and V4 shard families
before generating the signed factorial manifests.

P1 and P2's code-executable mechanisms are implemented and focused-test
verified. The cluster P2 telemetry/dashboard/control path and P3 fail-closed
chaos, soak, and preemption evidence validators are now implemented too. P4's
signed cross-repo promotion envelope and executable rollback are implemented;
their real G-C6/G-C7 evidence remains open. The cluster gate remains
intentionally unchecked until supplied with live evidence. The
remaining work is evidence-bound: P1's three-seed winner, measured throughput
and kill-9 recovery, live cluster chaos/soak; P2's actual Phases B-E,
post-training ablations, full Gate 6, and the external human/production
evidence behind the locally exercised gates.
