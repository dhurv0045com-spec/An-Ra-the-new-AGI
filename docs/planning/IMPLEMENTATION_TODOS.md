# MASTER_UPGRADE v3 - Implementation TODOs

This is the live execution board for `MASTER_UPGRADE.md`. A checked box means
code plus focused tests exist; it does not waive a quantitative campaign gate.

## P0 - Week 1, shared foundations

### Stream A - Model and campaign

- [ ] Obtain or locate the real 500M checkpoint without adding it to git.
      *(Blocked on the owner artifact. Locator ships in
      `scripts/run_checkpoint_forensics.py`: explicit arg →
      `ANRA_CHECKPOINT_PATH` → canonical path; reports blocked, exit 3.)*
- [x] Freeze checkpoint, tokenizer, config, and corpus-manifest hashes.
      *(`scripts/freeze_baseline_hashes.py` → `output/v2/baseline_freeze.json`,
      executed 2026-07-07: tokenizer/config/corpus frozen; checkpoint hash
      slot reports `blocked_on_artifact` until the file exists.)*
- [ ] Run exact tensor accounting and the 500 tokenizer probes.
      *(500 probes executed live 2026-07-07 and match the frozen fingerprint;
      tensor accounting is wired in the forensics driver but blocked on the
      checkpoint artifact.)*
- [ ] Run deterministic generation and the 200-prompt recovery gate.
      *(Driver ready: `scripts/run_checkpoint_forensics.py --run-generation`
      — greedy, seed 0, cache off, coherence gate 80%, undertraining decision
      rule. Blocked on the checkpoint artifact.)*
- [x] Convert the pilot factorial into pre-registered launch manifests, three seeds each.
      *(`training/pilot_factorial.py`: 23 cells, seeds 1301/2602/3903, signed
      schema-2 manifests, Law-1 scratch lineage; owner emits the set with
      `py -3.14 -m training.pilot_factorial --owner-authorized` once
      `ANRA_MANIFEST_SIGNING_KEY` is set.)*
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
      *(Blocked on network + storage + hours. Machinery exists:
      `scripts/download_training_data.py` — pinned revisions, MinHash dedup,
      PII/contamination cleaning, license gate, token-shard publisher.)*
- [ ] Produce the at-least-50 MB tokenizer campaign slice with held-out source splits.
      *(Builder shipped: `scripts/build_campaign_slice.py` — deterministic
      per-source held-out split, disjointness + >=50MB gate. Executed on the
      local corpus (3.4MB train, held-out disjoint); the 50MB itself is
      corpus-blocked.)*
- [x] Generalize the proven append migration to the canonical 32k V4 ceiling.
      *(`build_append_only_v4`/`audit_token_fertility` now take a `target_vocab_size`
      ceiling; canonical `CANONICAL_V4_VOCAB_SIZE = 32_768` with a pinned
      530,602,567-param contract; 16,384 retained as the proven fallback.
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
- [ ] Complete cluster telemetry, chaos suite, and 24-hour soak gates.
- [ ] Build hybrid retrieval and memory-tier recall CI.
- [ ] Ship proof-carrying answer contracts and injection-defense tests.
- [ ] Build continuous batching, paged KV, and prefix-cache serving skeleton.
- [ ] Run Phase A only after all blocking preflight gates pass.

## P2 - Weeks 7-12

- [ ] Complete Phases B-E and the full Gate 6 evaluation.
- [ ] Complete SFT, RLVR, STaR, DPO, and self-distillation with ablations.
- [ ] Add speculative decoding, adapter hot-load, QAT, and parity/latency CI.
- [ ] Prove plan-act-verify on the 50-goal suite.
- [ ] Run GEPA for ten cycles including at least one correctly rejected proposal.
- [ ] Ship ledger-derived UI trust surfaces and the 20-scenario usability script.
- [ ] Execute canary, adversarial gate, rollback, and release-bundle drills.

## Moonshots - pilot-gated, never on the critical path

- [ ] M1 attention/SSM hybrid pilot.
- [ ] M2 in-house vision encoder and projector stages.
- [ ] M3 latent-reasoning channel pilot.
- [ ] M4 world-model rollout API and calibration gate.
- [ ] M5 trained retriever head behind the shared retrieval protocol.
- [ ] M6 formal-proof expansion where deterministic verifiers exist.
- [ ] M7 self-development ladder, starting at proposal-only authority.

## Review ledger (Fable 5 audit of Codex-checked work)

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
shipped. Remaining Stream B work is data/compute-blocked: acquire >=120GB,
produce the >=50MB slice, then run the 150M V4 pilot.

Owner actions that unblock the rest: (A-checkpoint) restore the real
checkpoint then `py -3.14 scripts/run_checkpoint_forensics.py --run-generation`;
(A-manifests) set `ANRA_MANIFEST_SIGNING_KEY` then
`py -3.14 -m training.pilot_factorial --owner-authorized`; (B-corpus)
`py -3.14 scripts/download_training_data.py --profile 30gb` then
`scripts/build_campaign_slice.py` then `scripts/build_v4_tokenizer.py`.

Stream E continues: ledger-derived UI projections for verification, memory,
and gate visibility (the forecast-ledger schema + timestamp-audit box is
already delivered).
