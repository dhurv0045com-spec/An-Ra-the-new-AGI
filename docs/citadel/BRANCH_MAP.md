# BRANCH_MAP.md

Audit date: 2026-09-03. Auditor: Citadel bootstrap audit (direct git inspection + full-tree review).

Labels: **FACT** = verifiable from the repository (command or artifact cited). **INTERPRETATION** = Citadel's reading; may be wrong; must be re-derived when cited.

---

## Repository facts

**FACT.** Remote: `https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git`.
**FACT.** Branch tips at audit time (after `git fetch --all --prune`):

| Branch | Tip SHA | Commits | Merge base with esoes |
|---|---|---|---|
| `origin/esoes` | `85f44b7b449f2ee39a0e80203a2d7df04614983b` | 451 | — |
| `origin/triquetra` | `fa44ea3` | +47 ahead of esoes | `85f44b7` (the esoes tip itself) |
| `origin/cymek` | `92dcd56` | +1 ahead of esoes | `85f44b7` (the esoes tip itself) |
| `citadel` (new, local) | bootstrap at `85f44b7` | 0 | `85f44b7` |

**FACT.** Other remote branches exist (`main`, `core`, `core-exp`, `core-frozen-v4`, `core-vnext`, `senora`, `codxyz`, `experiment`, `iterate500`, `iterate900`). They are evidence sources only for this map; not audited here.

**FACT.** Both Triquetra and Cymek descend directly from the current esoes tip — they are siblings, not ancestors of each other.

**FACT.** The main local clone (`C:\Users\ankit\.zcode\workspace\default\An-Ra-the-new-AGI`) is checked out on `cymek` with uncommitted WIP (`v5_promotion/gates.py`, `v5_evaluation/checkpoint_adapter.py`, `v5_evaluation/firewall.py`, tests). That WIP is **not** part of any audited branch and was excluded from this audit.

---

## ESOES — research foundation

**Purpose (FACT, per `ESOES.md` / `README.md`):** clean-sheet cognition-first research branch for the next An-Ra Core generation (V5). Prior systems (V4, PGE, SFT, EXP) are evidence sources, not inherited implementations.

**Major added systems (FACT, all present at `85f44b7`):**

- `e0_cognition/` — E0 cognitive benchmark: causal contrast cases with mechanical assertions, model-view secrecy, 12 heuristic baselines with per-heuristic calibrated nulls, generator v0.4.0, sealed-fixture tooling that refuses in-repo sealed builds, statistical suite (exact tests, Wilson, bootstrap, Holm).
- `e1_tokenizer/` — static tokenizer tournament machinery; a real local 3-arm BPE tournament (16k/24k/32k) on 8.56M local bytes with determinism receipts.
- `e2_architecture/` — architecture canaries (signal propagation, QK norm, precision parity, RoPE conformance, real AdamW update invariants, cursor/resume) plus the candidate-scoring policy tournament machinery.
- `e3_data_objective/` — E3 data/objective experiment plan (blocked upstream).
- `v5_contracts/` — model spec (exact 250,216,960 params), run spec (5B tokens, 38,147 updates), training spec v1.0, launch readiness (fail-closed, `main_training_authorized` hardcoded `False`).
- `v5_training/` — training-state/checkpoint/transaction canaries, `4+4+2` budget, optimizer identity checks.
- `blueprint/` — V5 master blueprint, training spec, benchmark standard, decision log, launch gates (E1–E6 all PENDING).
- ~148 tests; CI byte-compares receipts against regenerated output.

**Major experimental conclusions (FACT for artifacts, verdicts in EVIDENCE_LEDGER.md):**

- E0 development certificate PASS (368 cases/112 pairs, all shortcut heuristics within calibrated null +10pp) — explicitly *infrastructure only, not a model result*.
- Two self-caught evaluator false greens (positional state shortcut; lexical/bag-of-words shortcut) repaired and documented (`artifacts/e0/shortcut_repair_receipt.json`).
- Scoring-policy tournament FAILED: both calibrated policies select the fewest-token role 1.000 in all 15 CUDA cells; `production_scoring_mode = null` (`artifacts/e2/scoring_policy_development.json`).
- Local mechanism canaries SUPPORTED (residual init, QK-norm scale control, BF16/FP32 parity, RoPE, real-update invariants); native-BF16 AdamW and native GQA (this stack) REJECTED.
- Inherited evidence base documents the founding negative: PGE continuation improved held-out loss 2.1884 → 1.9710 while copy/binding/composition stayed at zero/chance (`docs/esoes/EVIDENCE_AND_CONTEXT.md`, receipt in `core-vnext@054619f`).

**Unresolved work (FACT, per `blueprint/STATUS.md` + `LAUNCH_GATES.json`):** real E1 corpus/tokenizer artifact; learned E1–E5 runners and production trainer (unimplemented); P35 learned architecture/mixture evidence; sealed T2 custody; signed manifests; remote durability; all six launch gates PENDING; `main_training_authorized: false`.

**Relationship:** the shared research foundation. Citadel's sole ancestor.

---

## TRIQUETRA — experimental descendant (audit input)

**Purpose (INTERPRETATION, consistent with its own ledgers):** an external experimental notebook that took ESOES instruments onto real (weak) V4 checkpoints to find binding-related mechanisms; developed its own evidence discipline (firewalls, preregistrations, readiness gates) after catching two of its own false greens.

**Arc of the 47 commits (FACT, from `git log origin/esoes..origin/triquetra`):**

1. Executed ESOES's declared-but-unrun scoring tournament → `FAIL_DEVELOPMENT_POLICY` (also present on esoes tip artifacts).
2. Built `x_factor/` synthetic mechanism framework (X0–X7 ladder) — explicitly `SYNTHETIC_MECHANISM_DEMONSTRATION` only.
3. Real-model morning runs on a V4 SFT "accumulation child" checkpoint (since deleted from disk): X1-REAL, IBQ v1/v2, strength ladder, causal decomposition, binding factorial.
4. Afternoon preregister→receipt discipline on `anra-v4` step-30400: entity×value factorial (DEV + replication), competitive binding, query-value (QV) evidence matrix (DEV + frozen-seed replication + step-22517 comparison), structural-OOD E5.
5. Readiness-gate program: v1 pilot false green caught and downgraded; v2 calibration with primitive canaries; checkpoint registry; `WAITING_FOR_STRONGER_CHECKPOINT` declared; Cymek V5 arrival manifest prepared (identity fields unfilled).

**Key receipts (FACT for numbers; see EVIDENCE_LEDGER.md for verdicts):** `output/entity_value_factorial_dev.json` (+ `_rep.json`), `output/query_value_evidence_dev.json` (+ `_rep.json`, `_ckpt22517.json`), `output/structural_ood_e5.json`, `output/competitive_binding_dev.json`, `output/ibq_legacy_basis_verdict.json`, `output/readiness_v2_calibrate_30400.json`, `x_factor/registry/checkpoints.json`, master ledger `AN_RA_PROGRAM.md`.

**Major conclusions (FACT that these are recorded):** value-recency effect dominates repair (+43.0pp DEV, +33.7pp rep, single checkpoint); latent query-conditioned signal ≈ absent (raw rank 0.2500 = chance, QCS CI includes 0, both seeds); E5 duplication template-bound under structural shift (effect 0.0, p=1.0) — line closed; IBQ bases not qualified twice; X1-REAL invalidated as imbalanced false green; readiness v1 READY verdict downgraded to `CALIBRATION_ONLY`; no locally available checkpoint qualifies as a research subject (P1 single-fact canary 8.3%, B0 raw 0.0).

**Unresolved work (FACT per ledger):** everything is gated on a stronger checkpoint arriving through the frozen arrival contract (strict identity → canaries → calibrate → qualify); the E5 line is closed; the value-prior/position decomposition candidate mechanism is explicitly unfrozen/unpreregistered.

**Relationship (INTERPRETATION):** the project's only real-model cognition evidence. High audit value; all of it DEV-tier on floor-limited substrates. No training was run on this branch.

---

## CYMEK — production path (audit input)

**Purpose (FACT, per its blueprint/spec):** the executable V5 training system — how a validated idea would actually enter production training.

**The single commit (`92dcd56`, +~3,445 lines) adds (FACT):**

- `v5_tokenizer/` — frozen 24,576-entry byte-BPE interface, special IDs PAD 0/UNK 1/BOS 2/EOS 3, zero-unknown invariant, hash-bound identity receipts. Artifact itself still provisional (`artifact_sha256: None`).
- `v5_data/` — manifest-declared datasets (`DataManifest`/`SourceRecord`/`PackManifest`), exact-cluster split assignment (cluster = split unit), 8-gram contamination scan (fail-closed), mixture allocation (largest-remainder), bucketed packing (512/1024/2048/4096, frozen supercycle), block-diagonal packing semantics, coordinate cursor with exact real-token ledger.
- `v5_model/` — `V5A_250M` construction: 24,576 vocab, width 896, 26 layers, 14Q/7KV GQA, head_dim 64, SwiGLU 2,368, ctx 4,096, RoPE 10k, RMSNorm, tied embeddings, affine QK norm, exact receipt 250,216,960 params, deterministic per-seed init, `1/sqrt(2L)` residual output init.
- `v5_objectives/` — causal cross-entropy only (BOS/PAD excluded, segment-boundary masking, replica-global mean); query-swap contrastive present but disabled and gated to E3 Phase B.
- `v5_training/` — 131,072-token update contract, token-indexed WSD schedule (pure function of cumulative tokens; no rewarm on resume), single-writer checkpoint transactions, runner fencing, optimizer with tied-embedding ownership rules.
- `v5_evaluation/` — model adapter (never sees sealed fixtures, generation hard-capped at 64), Wilson LCB metrics, 100-case conditional-realization floor, tier system (tier0/tier1/sealed/fresh).
- `v5_promotion/` — ten conjunctive worst-family gates, signed decisions, chronology-proof checkpoint selection, durability receipts.
- `v5_remote/` — hash-bound remote job envelopes + six external identity slots; collection CLI; nothing ever submitted.
- 20+ test files covering the above.

**Frozen center (FACT):** 5B tokens; slices 0.65 natural / 0.20 code-math-formal / 0.15 verified-cognition; nine cognition family fractions inside the 750M slice (identity_copy 0.08 … faithful_realization 0.05); difficulty shares 0.34/0.355/0.305; tokens/update 131,072; WSD 50M warmup / 4.5B constant / 500M decay.

**Unresolved (FACT):** no corpus loaders (data must arrive in memory as token-id lists), no production trainer (canaries only), no tokenizer artifact, no real manifests (all identity hashes null), no remote submissions ever, `main_training_authorized: false`.

**Relationship (INTERPRETATION):** the migration target. Implementation stability of its contracts is separate from scientific optimality of the frozen mixture — a distinction ESOES itself draws and Citadel must preserve.

---

## Functional relationship (INTERPRETATION, not git ancestry)

```
ESOES (research foundation, 85f44b7)
 ├── TRIQUETRA (+47): what was learned on real (weak) checkpoints — audit input
 ├── CYMEK (+1): how validated ideas would enter production — contract reference
 └── CITADEL (this branch): independent controlled experiments from the clean foundation
        → validated claims become promotion candidates, ported deliberately with provenance records
```
