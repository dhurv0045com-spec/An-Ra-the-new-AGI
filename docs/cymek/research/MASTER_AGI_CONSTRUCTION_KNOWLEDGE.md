# AN-RA MASTER AGI CONSTRUCTION KNOWLEDGE

**Status:** phase 3 cross-branch research authority map; historical phase-2 material retained; not a claim of AGI
**Generated:** 2026-09-24
**Reviewed:** 2026-09-24
**Historical evidence cutoff:** 2026-09-13
**Current evidence basis:** [`../../research/EXPERIMENT_EVIDENCE_LEDGER.json`](../../research/EXPERIMENT_EVIDENCE_LEDGER.json) and [`../../research/EVIDENCE_SOURCE_MANIFEST_2026-09-24.json`](../../research/EVIDENCE_SOURCE_MANIFEST_2026-09-24.json)
**Phase-3 scope:** 90 experiments and 71 byte-identical imported evidence/provenance files; the source-manifest SHA-256 is `c164c735ec4e628a6311fe4c52117c8b58370321df61b00d4976c11769f0215f`.
**Pre-consolidation tip:** `research/evidence-consolidation-2026-09-25` at `90f77b7fa6ffd99f5a982263f03b2908298805ec` (`90f77b7f`), the parent of the consolidated head.
**Purpose:** give a human or autonomous research agent one file that explains what An-Ra currently knows about building a stronger general-learning Core, what construction is justified now, what is only a hypothesis, what failed, what remains unknown, and what evidence must exist before scaling.

> **Two-sentence definition of this file:** This is the shortest honest path from all current An-Ra evidence to a buildable research system: a conventional neural Core trained on auditable data, measured with causal/anti-shortcut evaluations, and surrounded by a fail-closed training, verification, retention, diagnosis, and promotion loop. It does **not** say we know how to build “perfect AGI”; it says exactly which pieces are demonstrated, which are implemented but unexecuted, which are speculative, and which experiment should change the design next.

---

## 0. Evidence snapshot and authority

This phase-3 synthesis re-freezes the historical 2026-09-13 phase-2 snapshot and adds the evidence reviewed through 2026-09-24. The historical branch table below is retained as provenance history; current authority is the phase-3 source manifest and the 90-entry machine ledger. Raw byte-identical source blobs, explicit receipts, and post-run audits outrank prose; implementation, preflight, custody, and external-only records retain their narrower classes.

The phase-3 ledger has **90 experiments** with status counts of 27 DEMONSTRATED, 16 SUPPORTED, 16 INCONCLUSIVE, 8 SUPERSEDED, 10 CONTRADICTED, 7 NOT_TESTED, 2 IMPLEMENTED_NOT_EXECUTED, 2 INVALIDATED, 1 SPECULATIVE, and 1 IN_PROGRESS. The 71 imported files are byte-identical evidence/provenance files; the manifest hash is `c164c735ec4e628a6311fe4c52117c8b58370321df61b00d4976c11769f0215f`. The consolidation is a standalone commit, not a merge or cherry-pick.

### 0.1 Phase-3 read order and import classification

Read the machine artifacts before this narrative:

1. [`../../research/EVIDENCE_SOURCE_MANIFEST_2026-09-24.md`](../../research/EVIDENCE_SOURCE_MANIFEST_2026-09-24.md) and its JSON counterpart for exact refs, byte identities, exclusions, and external-only artifacts.
2. [`../../research/EXPERIMENT_EVIDENCE_LEDGER.md`](../../research/EXPERIMENT_EVIDENCE_LEDGER.md) and [`../../research/EXPERIMENT_EVIDENCE_LEDGER.json`](../../research/EXPERIMENT_EVIDENCE_LEDGER.json) for all 90 rows, statuses, metrics, claims, and relations.
3. [`../../research/NEGATIVE_RESULTS_LEDGER.md`](../../research/NEGATIVE_RESULTS_LEDGER.md), [`../../research/CAUSAL_IDENTIFIABILITY_AUDIT.md`](../../research/CAUSAL_IDENTIFIABILITY_AUDIT.md), and [`../../research/CAUSAL_KNOWLEDGE_GRAPH.md`](../../research/CAUSAL_KNOWLEDGE_GRAPH.md) for claim ceilings and causal boundaries.
4. [`../../research/NEXT_PHASE_DECISION_MEMO.md`](../../research/NEXT_PHASE_DECISION_MEMO.md), [`../../research/NEXT_3_EXPERIMENTS.md`](../../research/NEXT_3_EXPERIMENTS.md), and [`../../research/EXPERIMENTS_TO_CANCEL_OR_DEFER.md`](../../research/EXPERIMENTS_TO_CANCEL_OR_DEFER.md) for the current decision order and stop rules.
5. This document and [`../../cymek/research/CROSS_BRANCH_EVIDENCE_AUDIT.md`](../../cymek/research/CROSS_BRANCH_EVIDENCE_AUDIT.md) for the human synthesis; branch-local ledgers are historical/append-only context.

The phase-3 import groups are:

| Imported source | Evidence brought into the current branch | Current boundary |
|---|---|---|
| `origin/cymek-500m-readiness` @ `dc915d44` | Complete R1C controlled development mechanism evidence | Narrow fixed-physical-V24576 mechanism result |
| archived `cymek-v51-canary-v2` @ `22d1bd4f` | Narrow canary mechanical pass and identity failure | No general V5.1 or production claim |
| `origin/cymek-next-core-architecture` @ `28e3cd02` | Formation-Mux S5 v8 completed/sealed result | Floor-limited NULLs, not mechanism exoneration |
| `origin/cymek-beta` @ `194b98c8` | Updated Formation-Mux recovery/CI records, blocked ROLE-TRANSFER-001 preregistration, and HORM-003/004 compact records | Recovery engineering is qualified but exact recovery is blocked; Role-Transfer is untested/unauthorized; HORM negatives are miniature |
| `origin/cymek-cs-transfer-001` @ `b52fe453` | CS protocol, amendments, engineering, and qualification records | Existing final CS interpretation remains controlled development-scale evidence |
| archived `eval-integrity-001` @ `915e23ff` | Citadel data/evaluation audits and T1D evidence | Corpus/evaluation not ready; old positive claims blocked |
| archived `codex__arkenstone-improvements` @ `516c2806` | Imported ARK-014 compact receipts | Narrow binding evidence; zero-failure-event retention screen |
| `origin/codex/cyhex-integrity-audit` @ `67220e2c` | Custody and provenance diagnostics | No recovered bytes or scientific revalidation |
| `origin/Gandiva` @ `49a3717a` | K8 engineering-partial and TPU-preflight handoff | Engineering-only; TPU not run |
| `origin/arkenstone-ark020-v4` @ `308a9865` | ARK-020 V4 readiness receipt | `DO_NOT_RUN/NOT_EXECUTED` |

### 0.2 Phase-3 branch coverage dispositions

A no-byte-import decision is an evidence-coverage classification, not a claim that a branch was ignored. The full ref matrix and exact reasons are in the source manifest; the principal dispositions are summarized here so the narrative does not accidentally promote an unimported or engineering-only branch.

| Ref | Phase-3 disposition | What remains authoritative here |
|---|---|---|
| `origin/Arkenstone` | excluded: historical evidence already represented | ARK-001…019 historical rows and negative ledgers; Guardian raw gap remains |
| `origin/BRAMASTRA` | excluded: superseded by descendant | BRAMASTRA phase-2 evidence; K8 is carried by the Gandiva import |
| `origin/arkenstone-astra` | excluded: packaging duplicate | no unique scientific bytes; implementation packaging only |
| `origin/citadel` | excluded: immutable archived descendant preferred | archived Citadel data/evaluation/T1D evidence is imported |
| `origin/cymek-v51-canary` | excluded: archived v2 descendant preferred | archived Canary-v2 result is the current narrow record |
| `origin/cyhex-hermes` | excluded: superseded by integrity-audit descendant | custody/provenance records are imported from `origin/codex/cyhex-integrity-audit` |
| `origin/triquetra` | excluded: no new post-cutoff unique evidence | historical weak-substrate diagnostics remain in the 90-row ledger |
| `origin/main` | base context, not an evidence source | merge-base and frozen repository context only |
| `origin/research/evidence-consolidation-2026-09-25` | target reference, not a source | pre-consolidation parent `90f77b7f`; standalone consolidation committed on the target branch |

Engineering-only, custody-only, and external-only records are not silently converted into scientific rows. Their exact artifact paths and hashes remain in the manifest and ledger.

### 0.3 Historical phase-2 branch snapshot (2026-09-13)

The following table is deliberately historical. It preserves the phase-2 branch evidence and includes `eval-integrity-001`, whose omission made the earlier “14 branches” statement incomplete. These heads describe the phase-2 audit snapshot, not the current phase-3 import refs and not a current branch topology claim.

| Branch | Historical head (2026-09-13) | Role | Highest-value historical evidence |
|---|---|---|---|
| `cymek-500m-readiness` | `f2c27a6` | Cymek V5 production core + controlled GPU campaigns | `docs/cymek/experiments/CYR-GPU-011..014-R1C/*`, `docs/cymek/research/*` |
| `Arkenstone` | `4ae9e3b` | Discovery program (ARK-001…022, Guardian/continual line) | `docs/arkenstone/CURRENT_STATE.md`, `experiments/ARK-017/RESULT_V2.md`, `experiments/ARK-018/FINAL_RESULT_AUDIT.md`, `experiments/ARK-019/FINAL_RESULT_AUDIT_V3.md` |
| `arkenstone-ark020-v4` | `65d1ef3` | ARK-020 V4 execution branch + durability amendments A1.1–A1.3 (no results) | `experiments/ARK-020-V4/ENGINEERING_AMENDMENT_A1.md` |
| `codex/arkenstone-improvements` | `1511321` | Runtime integrity + independent ARK-014 replication with raw receipts | `artifacts/arkenstone/ark014/ark014-cuda-2201-03/ARK-014_RESULT.json` |
| `arkenstone-astra` | `ccb84fb` | BRAMASTRA + research-environments code (no results) | — |
| `BRAMASTRA` | `02b94d3` | Discovery/binding lab + its own cross-branch audit | `docs/bramastra/RESULTS.md`, `docs/bramastra/EVIDENCE.md`, `artifacts/bramastra/*` |
| `triquetra` | `f23f0af` | Cognition laboratory (V4-substrate diagnostics); WAITING_FOR_STRONGER_CHECKPOINT | `AN_RA_PROGRAM.md`, `output/*` |
| `citadel` | `1d27f9b` | Independent auditor: T1-series, scorer tournament, 500M path audit, negatives registry | `docs/citadel/EVIDENCE_LEDGER.md`, `NEGATIVE_RESULTS.md`, `experiments/T1D/RESULTS.md`, `500M/PRODUCTION_PATH_AUDIT.md` |
| `esoes` | `85f44b7` | TPU-era V5 blueprint + founding negative + mechanism canaries (D-024…D-028) | `docs/esoes/EVIDENCE_AND_CONTEXT.md` |
| `eval-integrity-001` | `915e23f` | Independent evaluation-integrity audit and data/evaluation attack records | `docs/citadel/evaluation/*`, `docs/citadel/data/*` |
| `core-exp` | `51124de` | Historical V4-era self-model/policy line (unre-audited) + milestone 0001 | commit `20d8841` / tag `milestone/0001-honest-loop` |
| `core-frozen-v4` | `f72f193` | Frozen inference-only V4 core (32,768-token tokenizer) | README |
| `main` | `b620f1c` | Frozen V4 research system (2026-08-15) | — |
| `iterate500` / `iterate900` | `b438420` / `6fbd2c0` | Historical TPU/SFT engineering lineages | — |
| *(deleted)* `senora` | unreachable `30a8fa7` | Entire P35-CMS-1 + CAD program survives ONLY in unreachable commits | EVIDENCE_GAPS.md |

**Phase-3 topology correction:** the old warning that the current target is a parentless-squash/disconnected-shard topology is stale. The target/`origin/main` merge base is `010798094a43ea1ce2343abd79017212b873ec35`; `90f77b7f` has parent `0f293e8aea655e4c0809d3186e486627fbe98d63`; historical commit `28bf57a` has parent `1460741c94379bc9b51b2dc1640ef3a4ccbe31a3`. Unreachable historical objects and stash-only artifacts remain a separate preservation risk, not evidence that the current target is disconnected or parentless.

### 0.4 Changelog: historical phase-2 facts versus phase-3 corrections

The following phase-2 facts remain useful and are not silently deleted:

1. **ARK-017 V2 is executed:** `BOTH_LEVERS_SUFFICIENT`; HIGH failed 4/6, while LOW, CAP1X, exact-noncanonical 1/16 replay, joint treatment, and augmented-HIGH each had 0/6 failures. This supports a bounded retention interaction, not a universal replay or learning-rate law.
2. **ARK-018 V4 is executed and audited:** the Birth-content internalization threshold was not met (+0.000/+0.067 versus required +0.10), sealed science NLL cost was +3.68%/+3.91% versus matched science replay, and temporary-binding acquisition slowed. Assimilation is not identity, cognition, or AGI evidence.
3. **ARK-019 V3.1 is executed:** the official `CONTROLLER_NOT_SUPPORTED` verdict is formation-gated because the new capability never formed; old-skill recovery under replay/Guardian-like arms is not proof of prevention or internalization.
4. **CYR-GPU-013/R1B is executed:** the six-level class-space curve is `MIXED_OR_SEED_SENSITIVE_RESPONSE_CURVE`; the intermediate regime is reproducible in direction, not a universal optimum.
5. **CYR-GPU-012/R1 and 013/R1B change the representation question:** with active IDs fixed, declared tied class-space size can move held-out formation non-monotonically; monotonic “smaller is better” and “more classes is better” stories are false.
6. **BRAMASTRA’s phase-2 records remain historical evidence:** EOS supervision changes 0/32 to 32/32 across two seeds, while query-blind binding, discovery-controller, and depth-two results remain bounded negatives/inconclusive findings.

Phase-3 corrections supersede stale current-state claims:

- **R1C is complete 24/24:** `SOFTMAX_COMPETITION_NOT_SUFFICIENT`; K01 is **FIRED**; the mean `MASK_4096-FULL_24576` formation-AUC gap is `-0.11038062283737024`. This is controlled development mechanism evidence at fixed physical V24576, not a global softmax law or a V24576-optimality result.
- **CS-TRANSFER-001 is complete:** `PARTIAL_OR_INTERACTION`; physical V4096 is not a robust remedy, with one sealed reversal, three of four sealed pairs favoring V24576, and a raw Drive result whose recorded SHA-256 is `37e4bf741f9e55fc942fe4ade62d2d1c9619e4992e4a42ab507730809c12b150`.
- **Canary-v2 is complete:** 360 updates and 1,474,560 tokens passed mechanical gates but failed the frozen identity/copy formation threshold. Mechanical PASS is not cognition or production readiness.
- **Formation-Mux S5 v8 is complete 24/24 and sealed:** formal `CS-MECH-002` and `REP-FORM-003A` NULLs are retained, but the primary identity endpoint is near the zero floor; `M3-M2` is exploratory. The later v12/TIE-role frontier is only 2/24, with no sealed/final/checkpoint payload.
- **HORM-003 and HORM-004 are separate five-seed miniature `NOT_SUPPORTED` results:** HORM-004 success fraction is `0.0` on every seed, so it did not test rich dynamic appraisal. HORM-001/002 custody is blocked.
- **K8 is completed engineering-partial `INCONCLUSIVE`:** E0/E1 executed, E2 was not a positive cognition result, E3 was blocked by insufficient tool-training data, and E4/E5 were not run. The 100M TPU preflight was not run, made zero optimizer updates, and did not qualify TPU.
- **ARK-020-V4 is `DO_NOT_RUN/NOT_EXECUTED`:** resume and partial-identity defects are confirmed and all authorization flags are false. Guardian V4’s raw bundle remains missing; the transcription is not an executed authority.
- **Citadel corpus/evaluation are not ready:** latest-position shortcut, cross-split leakage, duplication, supply, tokenizer, entry-point, and fixture blockers prevent a production corpus or a new positive lift-off claim.
- **ARK-014 is narrow imported evidence:** order-robustness binding is supported across the local rerun, while the retention screen had zero failure events across three matched orders. It does not establish broad capability, transfer, or controller behavior.
- **Topology is corrected:** the target/`origin/main` merge base is `010798094a43ea1ce2343abd79017212b873ec35`; `90f77b7f` and `28bf57a` each have recorded parents. Unreachable-object preservation remains a separate risk.

**Authority rule:** raw byte-identical source blobs and explicit result receipts beat older prose; newer documentation without execution does not create a scientific result. A protocol, canary mechanical pass, custody audit, engineering build, preflight, partial snapshot, or external transcription cannot create a broader scientific claim. `cymek-500m-readiness` remains the production Core/operator context; Citadel, Triquetra, Arkenstone, BRAMASTRA, Gandiva, and the codex branches provide bounded evidence and challenger designs unless a result is explicitly promoted through the Cymek gate. The 90-row machine ledger, not a branch-local summary, is the current scientific index.

### Evidence labels used here

- **DEMONSTRATED** — executed evidence is strong enough for the exact scoped claim.
- **SUPPORTED** — useful executed evidence, but scope/replication/causal isolation is incomplete.
- **IMPLEMENTED** — code/plan exists and may be audited, but no scientific outcome exists yet.
- **SPECULATIVE** — plausible construction or mechanism requiring a decisive experiment.
- **REJECTED / CONTRADICTED** — the tested claim failed or a prior positive claim was invalidated.
- **NOT_DEMONSTRATED** — do not use the claim as fact.
- **INCONCLUSIVE** — evidence exists, but a control, floor, interaction, exposure, or causal-identification limit prevents the proposed conclusion.
- **IMPLEMENTED_NOT_EXECUTED** — code or readiness exists, but no scientific outcome was observed.
- **IN_PROGRESS** — a bounded custody or execution record is incomplete; it is not a verdict.
- **ENGINEERING_EVIDENCE** — receipt, build, resume, durability, canary, preflight, or custody evidence only.
- **SCIENTIFIC_EVIDENCE** — executed outcome evidence, always bounded by its claim ceiling.
- **BOTH** — a record may preserve engineering and scientific facts, but the classes must not be collapsed.
- **EXTERNAL_ONLY** — a raw bundle, checkpoint, row set, or transcription that has not been recovered and byte-verified in this branch.

A status is not a promotion decision. A `DEMONSTRATED` narrow result, a `SUPPORTED` result, a formal NULL, and a complete custody record all remain below the authorization boundary unless every conjunctive gate in the phase-3 decision model is passed.

---

# PART I — WHAT A SERIOUS AGI CONSTRUCTION ACTUALLY NEEDS

“Perfect AGI” is not a scientifically defined endpoint. For An-Ra, the useful target is a system that can **acquire reusable operations, generalize them under structural change, preserve them while learning more, diagnose failures, use tools/memory safely, and continue improving without evaluation leakage**.

A complete research stack therefore needs all of these layers:

```text
TRUSTWORTHY DATA
    ↓
REPRESENTATION / TOKENIZATION
    ↓
NEURAL CORE
    ↓
TRAINING OBJECTIVE
    ↓
OPTIMIZATION + CURRICULUM + REPLAY
    ↓
INTERNAL REPRESENTATIONS
    ↓
ADDRESSING / BINDING
    ↓
TRANSFORMATION / COMPOSITION
    ↓
GENERALIZATION + INVARIANCE
    ↓
RETENTION + RECOVERY + PLASTICITY
    ↓
MULTI-SKILL CONTINUAL LEARNING
    ↓
CAUSAL DIAGNOSIS / SELF-MODEL
    ↓
TOOLS + EXTERNAL MEMORY + VERIFICATION
    ↓
SEALED EVALUATION + FRESH REPLICATION
    ↓
PROMOTION / DURABILITY / SCALE
```

The Core and the complete AGI system are not the same thing. An-Ra’s current best architecture keeps the **Core conventional and causally inspectable**, while tools, permissions, durable memory, experiment routing, verification and promotion remain outside the Core until evidence shows a neural mechanism should be internalized.

---

# PART II — CURRENT BEST CONSTRUCTION

Sections 1–6 preserve the full phase-2 program synthesis and its detailed implementation constraints. Phase-3 status corrections are applied where they change current interpretation; historical constants and tables remain labeled as candidates or historical evidence rather than authorization.

## 1. System architecture: separate learning, execution, measurement, and authority

### 1.1 Neural Core

**Current best baseline: dense decoder-only Transformer.** Do not add MoE, recurrence, SSM blocks, latent-thought heads, learned routers, separate cognition heads, or neural long-term-memory modules merely because they sound “AGI-like.” The strongest reason is experimental: if the baseline fails, a simple system tells us which data/objective/representation pressure was inadequate; architecture soup destroys causal attribution.

The phase-2 V5-A production candidate in `cymek` is retained as an engineering baseline:

| Component | Current candidate | Status |
|---|---:|---|
| family | dense causal decoder Transformer | **SUPPORTED baseline** |
| parameters | 250,216,960 | **PROVISIONAL; scale not proven** |
| layers × width | 26 × 896 | **PROVISIONAL** |
| Q / KV heads | 14 / 7 | **STRONG INFERENCE** |
| head dim | 64 | **STRONG INFERENCE** |
| FFN | 2,368 SwiGLU | **STRONG INFERENCE** |
| attention | full causal every layer | **PROVISIONAL** |
| native context | 4,096 | **NOT YET LEARNING-VALIDATED** |
| norm | pre-RMSNorm + final RMSNorm | **SUPPORTED family** |
| QK norm | affine per-head RMS normalization | **DEMONSTRATED scale-control mechanism locally; cognition benefit unproven** |
| position | RoPE base 10,000, positions 0–4095 | **LOCAL CONFORMANCE** |
| embedding/output | tied | **PROVISIONAL** |
| bias/dropout | no linear bias, dropout 0 | **STRONG INFERENCE** |

Initialization: embedding, Q/K/V, gate and up use `Normal(0, 0.02)`; attention-output and FFN-down use `Normal(0, 0.02/sqrt(2L))`. Local mechanism tests support residual-output scaling as a way to control depth-dependent residual/gradient growth, but that is not yet evidence that it improves cognition.

Precision: one persistent FP32 parameter set, BF16 autocast compute, FP32 logits/loss/global-gradient-norm reductions, FP32 Adam moments. Native BF16 optimizer state was rejected after clip-norm overshoot; the mixed layout is the current safe local choice, not yet a universal TPU law.

### 1.2 External runtime / Connector

Keep the runtime controller outside the Core:

```text
TASK
 → Core attempt
 → independent verifier
 → observable failure record
 → smallest legal intervention
 → retry / repair
 → evidence receipt
 → optional training proposal
```

The verifier is the only success authority. Hidden answer labels may be used by an evaluator, but they must never leak into diagnosis features, curriculum generation, or runtime policy. External repair can prove that a failure is recoverable; it does **not** prove the neural Core has internalized the missing computation.

### 1.3 Independent evaluator and promoter

Training code must not own its own success metric. Evaluation consumes immutable checkpoints, sealed/fresh fixtures, tokenizer identity and protocol hashes; promotion consumes evaluation receipts and durability receipts, never mutable training state or “latest checkpoint.”

This separation is one of the strongest pieces of the project and should survive every future architecture change.

---

## 2. Representation and tokenizer: a first-order scientific bottleneck

### 2.1 Provisional tokenizer/representation candidate

Cymek’s provisional tokenizer contract is byte-level BPE with byte fallback, 24,576 entries, reserved `PAD=0, UNK=1, BOS=2, EOS=3`, zero expected UNKs, and no destructive normalization, case folding, whitespace rewrite, prefix-space insertion, or dropout. The original local tokenizer tournament put 24k between 16k and 32k in compression/parameter cost, but that result was only a planning prior because the local corpus was not representative; it does not authorize a production vocabulary or tokenizer change.

### 2.2 The representation evidence chain (CYR-GPU-011 → 012 → 013 → 014)

The latest completed Cymek-readiness experiments make representation impossible to treat as a minor implementation choice:

- **CYR-GPU-011 (DEMONSTRATED):** **COMPACT_BRIDGE**, real Cymek V5 4L/128w with the exact 19-symbol arithmetic representation, received only **44.89%** of the ARK-002B semantic exposure box yet reached M99 at update 1,400, sustained G50, and ended at **56.47%** held-out STANDARD exact-with-valid-EOS (maximum controller exact 59.38%). **PRODUCTION_BRIDGE**, same V5 geometry/task/objective with the frozen 24,576-token production representation, received **100%** of the exposure box (1,152,000 rows / 18,000 updates): train M99 but **0% held-out STANDARD**, **0/48 SEALED**, never G50 or G90.
- **CYR-GPU-012/R1 (DEMONSTRATED):** with the active arithmetic token IDs, segmentation, data, geometry, optimizer and seeds fixed, changing only the **declared tied embedding/output class-space size** produced V19 **12.94%** / **V4096 100%** / V24576 **0%** at the 512k-row endpoint. Non-monotonic; parameter displacement does not explain capability (V4096 moved farther than V19).
- **CYR-GPU-013/R1B (SUPPORTED, R1):** two fresh matched seeds × six levels at 128k rows: V19/V1024/V24576 ≈ 0; V4096 0.506/0.494 (most stable); V8192 0.718/0.0 and V16384 0.647/0.129 (seed-sensitive). Verdict `MIXED_OR_SEED_SENSITIVE_RESPONSE_CURVE`: the intermediate 4096–16384 region is the reproducible developmental regime; the exact optimum is not identified.
- **CYR-GPU-014/R1C (COMPLETE, CONTRADICTED, controlled development mechanism):** all 24/24 arms completed (4 matched seeds × 6 treatments × 3,000 updates; 72,000 optimizer updates total) with the physical 24,576-row matrix fixed in every arm. The verdict is `SOFTMAX_COMPETITION_NOT_SUFFICIENT`; K01 is **FIRED**; the preregistered `MASK_4096-FULL_24576` mean formation-AUC gap is `-0.11038062283737024` (paired gaps `[-0.1397923875432526, -0.24221453287197228, -0.019377162629757805, -0.04013840830449827]`). `MASK_4096` sustained above 50% on 1/4 seeds and `FULL_24576` on 0/4. This rejects inactive-softmax competition as the sufficient tested explanation; it does not disprove every softmax interaction, establish V24576 optimality, or authorize a production remedy.

Falsified simple stories: `smaller vocabulary → better capability`, `more classes/parameters → better capability`, and `mask inactive rows → robust production rescue`. A post-run audit also downgraded CYR-011's `COMMUTED=100%` flag: reversing operands changed the OOD tens-band role, so it is operand-role asymmetry evidence, not proven commutation invariance. The completed physical-class-space follow-up, `CS-TRANSFER-001`, is `PARTIAL_OR_INTERACTION`: physical V4096 is not a robust identity/copy remedy under the shared controlled development surface. Its mean development identity-AUC gap is `-0.034765625`, its sealed endpoint gap is `-0.1395833333333333`, one of four development pairs is positive, three of four sealed pairs favor V24576, and one sealed pair reverses. The raw Drive result remains external with recorded SHA-256 `37e4bf741f9e55fc942fe4ade62d2d1c9619e4992e4a42ab507730809c12b150`. Canary-v2 adds a separate narrow result: after 360 updates and 1,474,560 tokens, all mechanical gates passed but identity/copy formation failed. The remaining candidates are tied-row geometry, parameterization/initialization, optimizer/weight-decay/denominator interactions, representation, task effects, and their interactions.

### 2.3 Construction decision

Do **not** replace the general-language tokenizer with a 19-symbol arithmetic alphabet, and do **not** change the production tokenizer or vocabulary from these results. Keep the conservative general byte-fallback tokenizer for language. R1C and CS-TRANSFER are closed evidence, not pending mechanism campaigns: the tested inactive-softmax and physical-V4096 remedies are insufficient for their scoped questions. The next representation work is the cheap Formation-Mux control-capability/checkpoint-metric preflight, exact Output recovery, and—only if both gates pass—conditional completion of the frozen frontier. The prior `TIED-ROW-GEOMETRY-WD-001` placeholder is superseded by the more rigorous preregistered `ROLE-TRANSFER-001` design, which remains untested and execution-blocked on upstream completion, implementation, evaluator freeze, no-overlap proof, and independent qualification. Natural-language and larger-scale transfer wait for clean corpus/evaluation and validity gates. No production vocabulary, tokenizer, PRE500M, 250M, or 500M authorization follows.

---

## 3. Data: quality, causal coverage and replay matter more than raw bytes

### 3.1 Production mixture candidate

The phase-2 5B-token candidate was:

| Slice | Share | Tokens | Evidence status |
|---|---:|---:|---|
| high-quality natural text | 65% | 3.25B | **PROVISIONAL** |
| code/math/formal/structured | 20% | 1.00B | **PROVISIONAL** |
| mechanically verified cognition | 15% | 0.75B | **PROVISIONAL** |

This is a **historical phase-2 planning candidate**, not a current corpus recipe and not a production authorization. The current Citadel data gate is `FAIL`: no materialized production corpus, missing production entry point, unfrozen tokenizer artifact, unwired mixture enforcement and milestone evaluation, and real-corpus near-dedup/contamination screens still pending. The phase-2 5B candidate must not be treated as approved supply. The cognition fraction remains a hypothesis; any future comparison must be normalized for real tokenizer tokens, compute, provenance, and attack-screened evaluation.

### 3.2 Data contract

Every source should carry origin, acquisition, license/terms class, raw hash, filtering version, exact/near-dedup cluster, contamination result, split, tokenizer hash and post-filter token count. Splits must be source/dedup-cluster disjoint before outcome inspection; train/dev/sealed leakage invalidates a capability result even if training was expensive.

Synthetic cognition is useful only when truth is executable. The preferred source is programs, solvers, simulators, formal transforms or databases; LLM paraphrases remain provenance-tagged and bounded, never treated as ground truth merely because they sound natural.

The current Citadel evaluation audit makes the old tiered arithmetic surface non-qualifying for a new positive lift-off claim: latest-position accuracy is `1.000` on every tier; 173 dev and 357 test texts appear verbatim in train; train exact duplication is `13.5%`; and T1D had `15,000/15,000` generations terminate at `MAX_TOKENS` because EOS was not supervised. The historical T1-series nulls remain evidence within their confounds, but the surface cannot support a production cognition claim. `CORPUS-REGEN` must regenerate source/family-disjoint data, break shortcuts, supervise EOS, freeze fixtures, and pass contamination, leakage, supply, and sealed-firewall screens before natural-language or scale transfer.

### 3.3 Cognition families worth training

The phase-2 curriculum proposal below is retained as a **historical candidate family map**, not a demonstrated optimum or a current production mixture. The family definitions are more useful than the exact ratios.

| Family | Historical share inside cognition slice | Why it exists |
|---|---:|---|
| identity / exact copy | 8% | preserve arbitrary symbols and exact values |
| query-conditioned binding | 16% | let the current query select the right arbitrary binding |
| semantic-time state / precedence | 16% | use logical state rather than text position |
| interference-resistant retrieval | 10% | retrieve through distractors |
| relational composition | 20% | compose multi-step relations rather than retrieve endpoints |
| counterfactual sensitivity | 10% | change when relevant premises change |
| held-out-structure rule induction | 10% | infer/apply operations on novel structure |
| missing-information recognition | 5% | avoid unjustified guesses |
| faithful realization / format | 5% | emit the internally preferred answer correctly and terminate |

These weights are **not demonstrated optima**. The family definitions are better supported than the exact ratios.

### 3.4 Critical lesson: continued support can preserve a capability

ARK-015 is one of the most important data results in the entire program. Under canonical-only continuation, `NARROW_HIGH` lost robust order/query invariance in **8/8 matched pairs**, while `NARROW_LOW` lost it in **0/8** and `AUGMENTED_HIGH_REFERENCE` with continued presentation diversity lost it in **0/8**. Canonical accuracy stayed 1.0 even while broader invariance collapsed toward ~0.47 under narrow HIGH.

Therefore, forgetting/narrowing is not simply “the parameters moved too far.” High-plasticity training can preserve an invariant if the training stream continues to support it; current best hypothesis is an interaction between **plasticity/update scale × data support**.

### 3.5 Replay construction rule

When adding a new capability, preserve a small fixed replay/control stream from old capabilities and measure exact token fractions. ARK-013 showed that LOW LR alone did **not** solve prolonged cross-task interference without replay; every arm eventually lost T2 under pure new-skill training.

The exact replay fraction is **NOT_DEMONSTRATED**. Use controlled dose experiments, not folklore.

---

## 4. Objective: causal CE first, but train the behavior the evaluator requires

### 4.1 Current default

The clean baseline objective is causal cross-entropy over eligible target tokens, replica-global token mean, with label smoothing and z-loss zero. Query-swap auxiliary lambda and trace-loss lambda remain zero in the production candidate until a prospective experiment shows transfer beyond the task that trained it.

### 4.2 EOS/termination is a real training contract

Citadel T1D exposed a major measurement/training bug: **15,000/15,000 generations ended MAX_TOKENS** because EOS was used as a generation stop but was never a supervised target. That made content failure and termination failure inseparable; the corrected contract is `prompt + answer + EOS`, supervise answer content **and EOS**, and give a full-length answer one additional generation step to emit EOS.

This is not an AGI mechanism; it is a high-confidence mechanical rule. A system cannot be judged on a termination behavior it was never trained to produce.

### 4.3 Loss is not cognition

The founding negative result remains important: generic continuation improved held-out LM loss from **2.1884 → 1.9710** while preserved raw copy/context/binding/composition probes stayed at zero or chance. Citadel T1C/T1D repeated the broader lesson: large loss movement can occur while exact symbolic behavior remains almost unmoved.

Therefore use loss as a substrate metric, never as the success criterion for cognition.

### 4.4 Query-conditioned pressure remains a serious open objective question

Triquetra found raw query-conditioned candidate ranking near chance on the weak V4 substrate: raw rank-1 **25.0%** with chance 25%, QCS confidence interval including zero, and position effects around 19–35× larger than query-match effects. Correct-value recency could repair failures (~46–47%) while entity identity by itself repaired ~0%, which warns against casually calling a salience effect “addressing.”

A query-swap objective remains scientifically plausible because the current query often fails to control the answer representation, but it is **SPECULATIVE until a capable substrate, candidate-free transfer test and matched compute comparison show real gain**.

---

## 5. Optimization: acquire/recover and retain are different regimes

### 5.1 Safe baseline mechanics

Current V5 optimizer family:

```text
AdamW
β1=0.9, β2=0.95, eps=1e-8
weight_decay=0.1 on ndim>=2 tensors
no decay on ndim<2 norms/QK scales
global gradient clip = 1.0
```

The 5B V5 schedule was a phase-2 candidate, not a demonstrated current recipe. Token-indexed WSD, exact learning-rate constants, and large-run scale remain unvalidated and unauthorized. The current safe construction constraint is to preserve the verified optimizer ownership/precision contract while keeping any schedule decision behind corpus, formation, evaluation, and durability gates.

### 5.2 Demonstrated Micro retention law — scoped carefully

ARK-007R: after an already generalized T2 state, matched continuation at HIGH `1e-3` failed in **9/12 = 75%** forks while LOW `1e-5` failed in **0/12 = 0%**; risk difference LOW−HIGH = **−0.75**. This is strong Micro-T2 causal evidence that lower LR can protect an already-acquired capability under the tested stream.

ARK-010: after a prospectively defined instability event, continued HIGH recovered sustained G90 in **8/9 = 88.9%** states, while immediate LOW recovered in **2/9 = 22.2%**. Low LR therefore should not be used blindly when capability is absent.

ARK-011: after HIGH recovery, switching LOW reduced recurrent instability: HIGH recollapsed **3/6 = 50%**, SWITCH_LOW **0/6 = 0%**.

The best supported phase hypothesis is therefore:

```text
CAPABILITY ABSENT / NEED MOVEMENT
    → higher-plasticity acquisition or recovery
CAPABILITY CONFIRMED
    → lower-plasticity retention OR continued support/replay
CAPABILITY FALLS
    → recover, then protect again
```

But this is **not** a universal production scheduler. ARK-012 did not identify a unique numeric switch threshold; several behavioral thresholds aliased to the same switch time. ARK-016 also failed to isolate update-magnitude/trust-region mechanism because only 1/12 opportunities qualified.

### 5.3 What not to conclude

- “LOW LR is universally better” — false.
- “Large parameter movement causes forgetting” — unsupported; augmented HIGH moved farther and retained invariance in ARK-015.
- “Exact G90=.90 is a universal state threshold” — unsupported.
- “LR alone solves continual learning” — contradicted by ARK-013 no-replay interference.

---

## 6. Generalization and invariance: canonical accuracy is not enough

A model can score 100% on the narrow surface while losing the actual reusable operation. ARK-015 directly demonstrated this: canonical-only high-LR continuation retained canonical exact at 1.0 while order/query-order robustness collapsed, whereas LOW or continued augmentation preserved the broader capability.

Every important capability should therefore be tested on orthogonal axes:

```text
canonical
query-only change
order-only change
query + order change
relevant-fact intervention
irrelevant-fact intervention
structural OOD
natural/semi-natural analogue
```

Do not collapse these into one “reasoning score.” The latest CYR-GPU-011 post-run audit is another warning: a raw `COMMUTED=100%` flag was **not** accepted as commutation invariance because reversing operands also changed which operand occupied the held-out OOD role. A metric name is not evidence that the metric isolates what its name claims.

---

## 7. Capability formation is currently a more urgent Cymek problem than retention

### 7.1 Citadel low-budget evidence

T1D ran six arms across roughly 3.7–7.4M parameter models and 2–8M token budgets. Held-out exact remained **0–6.6%** across all arms, train exact **0–11%**, even though losses fell substantially. Curriculum, teacher, scale, masking and self-knowledge were therefore not sufficient for arithmetic lift-off at those tested budgets, though T1D had EOS and budget-confound limitations that prevent stronger universal conclusions.

Teacher primitives did move: held-out teacher microtask accuracy reached **51.5%** in the teacher arm while composed T2+ behavior stayed near zero. That is evidence for **primitive learning without compositional transfer**, not evidence that teacher data is useless.

### 7.2 CYR-GPU-009/011/012/013 refined the diagnosis

CYR-GPU-009's TINY model could saturate train probes but never reached candidate-free G90 under ~250k semantic row presentations; the later audit showed it received <22% of the ARK-002B positive-reference semantic exposure and was underdosed for a strong negative conclusion (CYR-GPU-010, which would have repeated that dose, was superseded pre-execution).

CYR-GPU-011 corrected the exposure confound and showed that **production representation can memorize but still produce 0% held-out exact even at the full 1,152,000-row reference box**, while compact representation entered a qualitatively different partially generalizing regime by 44.89% exposure. CYR-GPU-012/013 then isolated the surprising core: the effect tracks the **declared tied class-space size** (non-monotonically, seed-sensitively), not compact-vs-production tokenization per se.

Phase-3 closes the first mechanism question without closing formation. R1C completed 24/24 and found `SOFTMAX_COMPETITION_NOT_SUFFICIENT` at fixed physical V24576; CS-TRANSFER then found physical V4096 was not a robust remedy. Formation-Mux S5 v8 completed 24/24 and sealed, but the identity endpoint was `0.000–0.008` across all 24 arms while one-token termination reached `1.00`. The formal contrasts are therefore `INCONCLUSIVE_AT_ZERO_BASELINE`, not a mechanism exoneration. The later v12 frontier snapshot is only 2/24 and has no sealed result, final payload, or checkpoint tree. The current priority is a cheap control-capability and metric-resolution preflight, not a broad mechanism campaign from a floor.

---

## 7.3 Phase-3 post-cutoff evidence: completed, partial, blocked, and engineering-only

The following table is the human-readable phase-3 delta. The complete row set, metrics, relations, and artifact paths remain in [`../../research/EXPERIMENT_EVIDENCE_LEDGER.md`](../../research/EXPERIMENT_EVIDENCE_LEDGER.md) and its JSON counterpart; the source/ref decisions remain in the manifest.

| Area | Current record | Evidence class | Narrow claim ceiling |
|---|---|---|---|
| CYR-GPU-014-R1C | 24/24 complete; `SOFTMAX_COMPETITION_NOT_SUFFICIENT`; K01 fired; mean gap `-0.11038062283737024` | Scientific, controlled development mechanism | Fixed physical V24576 arithmetic development test; no global softmax law, optimality, vocabulary change, or scale |
| CS-TRANSFER-001 | Complete; `PARTIAL_OR_INTERACTION`; physical V4096 not robust; one sealed reversal; 3/4 sealed pairs favor V24576 | Scientific, controlled development causal evidence | Shared low-ID symbolic development surface only; raw Drive result external; no natural-language or scale transfer |
| CYMEK-V5.1-CANARY-V2 | 360 updates / 1,474,560 tokens; mechanical gates pass; `CANARY_V2_FAIL_FORMATION` | Scientific narrow canary plus engineering | Bounded canary family result; no general V5.1, cognition, or production claim |
| FORMATION-MUX-001-S5-V8 | 24/24 complete and sealed; formal NULLs at near-zero identity | Scientific but floor-limited/inconclusive | No tied-row, denominator, weight-decay, rendering, or architecture exoneration |
| FORMATION-MUX-001-V12-FRONTIER-PARTIAL | 2/24 TIE-role arms; no sealed/final/checkpoint payload; recovery engineering CI passed but original saved Output absent | Incomplete custody snapshot plus engineering qualification | No recovery execution, frontier verdict, or promotion |
| ROLE-TRANSFER-001 | Preregistered four-arm successor; no trainer, official arm, sealed evaluation, or result | Prospective protocol / engineering evidence only | No efficacy, mechanism, external benchmark, architecture, capability, or AGI claim |
| HORM-003 / HORM-004 | Separate five-seed miniature `NOT_SUPPORTED`; HORM-004 success fraction 0.0 on every seed | Scientific miniature negatives | HORM-004 did not test rich dynamic appraisal; HORM-001/002 custody remains blocked |
| BRAMASTRA-K8-20260922 | E0/E1 executed; E2 not positive cognition; E3 blocked; E4/E5 not run | Engineering partial | No tool-learning, cognition, AGI, or RSI inference |
| GANDIVA-TPU-100M-PREFLIGHT | Not run; zero optimizer updates and no qualification receipt | Engineering handoff | No TPU qualification or training result |
| ARK-020-V4 / Guardian | `DO_NOT_RUN/NOT_EXECUTED`; raw Guardian V4 bundle missing; resume and identity defects confirmed | Blocked engineering / external-only | No Guardian efficacy, continual-learning, or controller result |
| CITADEL-EVAL-001 / data gate | `NOT_READY` / `FAIL`; shortcut, leakage, duplication, supply, tokenizer, entry-point, and fixture blockers | Evaluation-integrity audit | Old T1 nulls remain historical; no production corpus or positive lift-off claim |
| ARK-014 and rerun | Narrow order-robustness binding evidence; retention screen had zero failure events across three matched orders | Scientific narrow / replication-bounded | No broad capability, transfer, or retention conclusion |

### 7.3.1 R1C and CS-TRANSFER: what changed in the causal picture

R1C held the physical 24,576-row tied matrix fixed and changed only training-logit participation. The primary `MASK_4096-FULL_24576` formation-AUC gaps were negative on all four matched seeds, with a mean of `-0.11038062283737024`; the preregistered positive mechanism threshold was not met. Individual `MASK_8192` successes were seed-dependent and do not rescue the mechanism claim. The returned result bundle is recorded as SHA-256 `2a9359e49f792f962774e42b4e5825b93fb7c91ddcad8f530c1c6f16c6f9bf9e`. The correct conclusion is that inactive-softmax competition alone is insufficient in this scope, not that every softmax interaction is absent or that V24576 is optimal.

CS-TRANSFER changed the actual physical tied matrix while matching the controlled development surface. The development mean identity-AUC gap was `-0.034765625`, the sealed endpoint gap was `-0.1395833333333333`, only 1/4 development pairs favored V4096, and 3/4 sealed pairs favored V24576 with one reversal. The sealed `4096−24576` gaps were `[-0.1833333333333333, +0.10833333333333331, -0.42499999999999993, -0.058333333333333334]`. The interaction/seed-sensitive verdict means physical reduction is not a robust remedy. It also means the next question is not a broad vocabulary sweep: it is whether tied-row geometry, initialization/parameterization, and optimizer/weight-decay/denominator dynamics interact away from a floor.

### 7.3.2 Canary and Formation-Mux: mechanical success is not formation

Canary-v2 passed its mechanical path after the longer exposure: development exact-with-valid-EOS was `0.519415`, sealed was `0.516641`, and the identity family was only `0.177005/0.189107`. The result says that a bounded execution stack can run and form several narrow families while failing a specific identity/copy endpoint. It does not turn the mechanical gates into cognition or production readiness.

Formation-Mux S5 v8 is the opposite kind of caution: the protocol completed 24/24 and sealed its formal contrasts, but identity formation was at `0.000–0.008` in every arm while termination reached `1.00`. The formal `CS-MECH-002` and `REP-FORM-003A` NULL labels are retained as the protocol’s recorded result, but the scientific interpretation is `INCONCLUSIVE_AT_ZERO_BASELINE`; the instrument lacks dynamic range for the intended multi-token identity contrast. The reported `M3-M2` difference of `+0.277083` is exploratory only. The later 2/24 TIE-role snapshot is a custody/data record, not a continuation or a verdict. Recovery-preflight engineering passed remotely after a preserved failed CI run, but the original checkpoint-bearing Output is absent, so no recovery execution or resumed arm exists.

### 7.3.3 ROLE-TRANSFER-001: stronger design, still no result

`ROLE-TRANSFER-001` supersedes the old tied-row placeholder as the design of record. It freezes four synchronized arms, pair-specific full-preclip norm matching, per-update hash-chained receipts, 12 fresh blocks across two replications, fixed two-million-token exposure per arm, clipping and gradient-manipulation gates, and a clean-room production-BPE endpoint within a new synthetic ontology. Its protocol hash is CI-qualified, but its trainer, evaluator freeze, no-overlap proof, exact-resume implementation, upstream frontier completion, and official execution authorization are absent. It is `NOT_TESTED`, not evidence for a mechanism.

### 7.3.3 HORM, K8, TPU, and custody boundaries

HORM-003 and HORM-004 are separate five-seed miniature prospective negatives, not one merged hormonal result. HORM-003 had only 3/5 sign-consistent seeds and a near-zero median loss difference. HORM-004 also had 3/5 sign consistency, but every treatment success fraction was `0.0`; the live appraisal signal was constant-failure and could not test rich dynamic conditioning. The historical HORM-001/002 result/checkpoint custody is blocked: observed JSONs match only an EOL reconstruction, checkpoints are missing at expected paths, and the audit explicitly does not authenticate, recover, or rerun the campaign.

K8 executed real CUDA engineering through E2 and stopped at E3 because the declared tool-training supply was impossible to satisfy; that is a data-contract failure, not a negative tool-learning result. The Gandiva 100M TPU preflight is an implemented zero-update backward handoff and was not run. Neither record creates a TPU qualification, cognition, tool-learning, AGI, or RSI result.

Formation-Mux checkpoint custody is also incomplete: the returned bundle contains 30 checkpoint receipts but zero checkpoint payloads, no source-commit/export-verification receipt, and the checkpoint tree remains in the original Kaggle Output. The partial archive contains the 2/24 frontier snapshot but no exact-resume state. A receipt, a ZIP name, a reconstructed hash, or a custody inventory is not raw-byte recovery. The phase-3 manifest lists the exact external paths and hashes; this document does not promote them.

### 7.3.4 Citadel and ARK-014: evaluation and narrow evidence

Citadel’s current `NOT_READY` audit is decisive for the old surface. The latest-position heuristic reaches `1.000` on each tier, 173 dev plus 357 test texts leak verbatim into train, train duplication is `13.5%`, and the prior T1D EOS defect made content and termination inseparable. The T1-series nulls remain useful historical negatives, but a new positive on that surface would be uninterpretable. `CORPUS-REGEN` is therefore a required parallel workstream, not a later cosmetic step.

The imported ARK-014 result is intentionally narrow. Its local rerun supports the order-robustness binding result under its frozen protocol, but the retention screen recorded zero failure events across three matched orders, so it cannot establish retention or a broad continual-learning mechanism. These distinctions are why the master synthesis points to the machine ledgers rather than treating every positive-looking number as a program-level capability.

---

## 7.4 Open unknowns and evidence prerequisites

The open-question register is deliberately kept separate from the maturity percentages. An unknown is not a negative result, and a blocked prerequisite is not permission to bypass the gate.

| Unknown | Current question | Evidence required before closure |
|---|---|---|
| U02 | Is the Guardian positive real, and does it beat static replay without task-ID leakage? | Recovered raw V4 bundle, byte audit, fresh matched replication, and a science-preserving parent |
| U03 | Which tied-row geometry, parameterization, initialization, and optimizer/WD/denominator interaction explains the remaining formation effect? | `FMUX-CONTROL-METRIC-PREFLIGHT`, exact recovery and conditional frozen-frontier completion, then the preregistered execution-blocked `ROLE-TRANSFER-001` validity sequence |
| U04 | Can a new skill form under a science-preserving parent at a viable dose? | Formation-positive parent and preregistered dose; not ARK-020 while `DO_NOT_RUN` |
| U05 | Which retention lever is necessary in which regime, and do doses transfer? | Matched shared-parent forks, multi-seed replication, and scale-transfer evidence |
| U06 | Can any assisted scoring policy pass nuisance/bias screens? | Certified scorer with length/tokenization attacks, sealed fixtures, and candidate-free comparison |
| U07 | Does the development representation effect transfer to natural language and larger models? | Clean corpus/evaluation, geometry gate, natural-transfer test, and fresh replication |
| U08 | What qualifies as a production corpus after the Citadel audit? | Materialized source manifests, supply, dedup, leakage, shortcut, tokenizer, and sealed-fixture receipts |
| U09 / U10 | Which subject/checkpoint qualifies for causal self-diagnosis, and how can unreachable historical programs be preserved? | Nondegenerate intervention basis, constant-baseline defeat, prospective replication, and raw-object preservation |
| U12 | Can a cheap Formation-Mux preflight move identity off the floor and resolve the metric? | Multi-seed control capability, checkpoint lineage, token/per-position/LCP diagnostics, and an away-from-floor endpoint |

The current next order is therefore a dependency chain, not a list of interchangeable experiments: qualify the control and metric, test a narrow mechanism only if identifiable, regenerate the corpus/evaluation in parallel, and defer transfer/scale until the preceding gates are clean.

---

## 8. Causal diagnosis and self-modeling: keep the machinery, reject the false greens

Triquetra’s strongest contribution is not a successful learned self-model; it is the discipline for deciding when self-model claims are identifiable.

### 8.1 What worked

- Controlled entity×value factorials showed correct-value recency is a strong elicitation axis: DEV value-only repair **46.7%** vs neutral **3.7%**, replicated at **46.2%** on a second seed.
- Separate selection from realization; rank-1 and free generation can fail differently.
- Use paired interventions and answer-blind/legal observations.
- Require a checkpoint readiness gate before mechanism studies.

### 8.2 What failed

The original X1-REAL self-model “PASS” was invalid. Prospective intervention prediction accuracy 0.9545 sounded excellent, but an always-negative predictor scored **0.9733** because positive-cell prevalence was only 0.0267. The basis was not qualified and the claim was correctly marked contradicted.

The first readiness gate also produced a false green on the weak V4 checkpoint; v2 correctly downgraded it to `NOT_READY / INSUFFICIENT / NOT_IDENTIFIABLE`. This is a critical research lesson: **a self-diagnosis model built on a floor-limited substrate can look accurate because the evaluation basis is degenerate**.

### 8.3 Construction decision

Keep a causal-diagnosis layer, but do not train a neural “self-model” until:

```text
base capability is in an identifiable performance regime
→ intervention basis has adequate positive/negative coverage
→ simple constant baselines are beaten
→ prediction is prospective
→ hidden outcome labels cannot leak into features
→ result replicates on fresh tasks/checkpoints
```

Until then, use the external verifier/experimenter as instrumentation rather than pretending metacognition has been learned.

---

## 9. Evaluation: the project’s strongest reusable asset

The canonical cognitive decomposition is:

```text
REPRESENT → ADDRESS → TRANSFORM → CHOOSE → REALIZE
```

Retention/recovery and transfer should be added as longitudinal axes rather than hidden inside a single final score.

Primary scientific evaluation should be candidate-free generation with explicit valid termination. Candidate ranking is diagnostic only until the scorer itself is nuisance-resistant; Citadel demonstrated that sum/token/byte and calibrated DC-PMI/contextual-calibration policies were badly length/tokenization biased, with the calibrated policies selecting the fewest-token role **100% across 15 CUDA cells**.

A promotion-quality result needs:

1. frozen protocol before outcomes;
2. source/fact/template/structure-disjoint CONTROL and SEALED sets;
3. paired causal interventions;
4. cheap heuristic attacks before model interpretation;
5. at least two independent subjects/seeds for a strong claim;
6. candidate-free behavior, not only assisted ranking;
7. retention/regression checks on old skills and natural substrate;
8. fresh replication after sealed success;
9. exact code/model/tokenizer/data/fixture hashes and receipts;
10. a written alternative explanation and evidence that would change the conclusion.

**Rule:** if loss improves and behavior does not, behavior wins. If a post-run audit shows a probe was confounded, the interpretation is downgraded even if the numeric score looks impressive.

---

## 10. Checkpointing, durability and reproducibility

Cymek’s production-engineering direction is sound and should be preserved:

```text
completed optimizer update
→ collective snapshot
→ immutable temporary generation
→ hash + structural validation
→ publish recovery/milestone pointer
→ durable upload
→ clean redownload
→ clean restore canary
→ only then mark durable
```

Full-resume state must include FP32 parameters, FP32 Adam moments/step, token-indexed schedule state, exact source/family token ledger, sampler/supercycle cursor, all RNG states, topology, model/tokenizer/data/pack/source/code identities. Resuming must not rewarm LR, double-advance token counters, or reconstruct sampler position from wall time.

Local checkpoint/transaction/corruption/cursor canaries are strong historical engineering evidence. The current Gandiva TPU preflight is not run: there is no TPU execution, optimizer update, checkpoint result, or qualification receipt. A future zero-update backward preflight, even if it passes, would prove only that a path ran; optimizer-state fit, one committed update, checkpoint/resume, sustained throughput, and scientific readiness would remain separate gates. There is no authorization for PRE500M, 250M, 500M, TPU qualification, or any production scale campaign.

---

## 11. Scale: engineering readiness is not scientific readiness

A historical Citadel production-path audit found packing, sampler/cursor, batch assembly, model path, causal loss, optimizer, TrainingState, checkpoint transaction and restore/continuation connected; historical blockers included production corpus materialization, top-level entry, frozen tokenizer identity, canonical schedule execution and wired milestone evaluation. The later phase-3 corpus/evaluation audit still reports `NOT_READY`.

The `cymek-500m-readiness` branch implemented major missing plumbing, but **data readiness and scientific authorization remain separate gates**. The phase-3 Citadel audit leaves the production corpus and evaluation `NOT_READY`; R1C and CS are narrow development-scale results; Formation-Mux is floor-limited; and no current record authorizes PRE500M, 250M, 500M, a production vocabulary/tokenizer change, cognition, AGI, TPU qualification, tool learning, or RSI.

Do not use “I can run the compute” as evidence that the run is scientifically justified. A large training run could be considered only after a future decision record passes clean corpus/evaluation, away-from-floor formation, geometry, transfer, retention, and durability gates. This document does not grant that authorization.

---

# PART III — THE INTEGRATED TRAINING ALGORITHM CANDIDATE

This phase-2 algorithm remains the best research construction candidate, not a proven AGI algorithm or an operator authorization. Phase-3 inserts a mandatory control-capability/metric-resolution preflight before any expensive mechanism contrast and keeps engineering, custody, and promotion evidence separate from scientific outcomes.

The following is the best current **research algorithm**, not a proven AGI algorithm.

```text
INPUTS
  immutable data/source manifests
  frozen tokenizer + model + objective + optimizer identities
  frozen development/control/sealed evaluation contracts

INITIALIZE
  conventional dense Core
  verified parameter count / tied weights / precision layout
  exact optimizer ownership and resume state

FOR each training phase:
  1. Train on deterministic, provenance-bound mixed data.
  2. Count actual non-padding target/input tokens exactly.
  3. At fixed token milestones, evaluate candidate-free:
       - substrate LM loss
       - representation
       - addressing/query sensitivity
       - transformation/composition
       - realization + EOS termination
       - invariance under irrelevant/order/surface changes
       - old-skill retention
       - new-skill acquisition
  4. Never select a mechanism from sealed outcomes.
  5. If capability is absent:
       keep acquisition/plasticity high enough to learn;
       do not freeze it merely to preserve a capability it does not possess.
  6. Once a capability is prospectively confirmed:
       compare retention controls under matched future data:
         HIGH / LOW / fixed-time / state-dependent policy / replay-support controls.
  7. When adding a new skill:
       keep an explicitly measured replay/support stream for old skills;
       report old and new capability separately.
  8. If capability falls:
       diagnose whether failure is representation, addressing,
       transformation, realization, data-support narrowing or optimization;
       apply the smallest intervention that distinguishes hypotheses.
  9. Checkpoint only after a completed update and publish only after clean restore.
 10. Choose a development checkpoint behaviorally, not because it is final.

PROMOTION
  one chosen immutable checkpoint
  → sealed evaluation once
  → source/family-disjoint fresh replication
  → natural-transfer check
  → regression/retention check
  → integrity/durability check
  → promote only if all conjunctive gates pass
```

The future **Capability Guardian** idea—reactively changing protection/replay based on CONTROL probes—remains a hypothesis. ARK-019 V3.1 is a formation-gated negative, ARK-019 V4 is only a transcribed external candidate whose raw bundle is missing, and ARK-020-V4 is `DO_NOT_RUN/NOT_EXECUTED` with confirmed resume and identity defects. Static sparse replay is an external reference protection; no Guardian/controller is installed or authorized.

---

# PART IV — EVIDENCE MATURITY PERCENTAGES

The original phase-2 maturity estimates are retained as a transparent heuristic and updated where phase-3 receipts change the boundary. They are not probabilities, authorization, or “percent to AGI.”

These percentages are **not “percent to AGI” and not probability of success**. They are a transparent engineering/research maturity estimate using this rubric: 0 = idea only, 25 = implemented/unexecuted, 50 = executed in one narrow regime, 75 = replicated and/or transferred across controlled regimes, 100 = replicated on real data and target-scale production conditions with no unresolved major confound.

| Subsystem | Evidence maturity | Why |
|---|---:|---|
| experiment integrity / receipts / fail-closed contracts | **90%** | 90-row ledger, byte-identical import manifest, explicit negative preservation, custody diagnostics, and strong fail-closed contracts; external bytes remain explicitly bounded |
| causal evaluation methodology | **85%** | rich decomposition, anti-shortcut, sealed/fresh design; scorer policy remains unresolved (`production_scoring_mode: null`) |
| checkpoint/resume/durability mechanics | **80%** | strong local and historical smoke evidence; current TPU preflight was not run; ARK-020-V4 confirms resume and identity defects remain |
| Core architecture mechanics | **65%** | conventional design + local QK/init/precision evidence; exact V5 learning benefit unproven |
| data governance / provenance design | **70%** | strong contracts and implementation; production-quality 5B corpus not yet qualified |
| actual production corpus readiness | **20%** | pipeline exists, but complete campaign supply/qualification is not demonstrated |
| representation/tokenizer scientific understanding | **55%** | R1/R1B map a non-monotonic seed-sensitive class-space effect; R1C rejects inactive-softmax sufficiency at fixed V24576; CS finds physical V4096 non-robust; geometry, optimizer/WD interaction, and transfer remain open |
| objective design | **45%** | CE mechanics + EOS contract solid cross-program; query-conditioned/compositional objective pressure unresolved |
| capability formation/generalization | **55%** | delayed transition replicated; class-space response curve mapped at development scale; production-representation lift-off still unsolved |
| retention/recovery under same-skill stress | **75% Micro / ~20% production-transfer** | replicated Micro T2 levers + ARK-017 both-levers result; mechanism and scale transfer unresolved |
| multi-skill continual learning | **15%** | ARK-019 V3.1 is formation-gated `CONTROLLER_NOT_SUPPORTED`; V4 is transcribed-only with raw bytes missing; ARK-020 is `DO_NOT_RUN/NOT_EXECUTED` |
| causal self-diagnosis / learned self-model | **15%** | instrumentation improved, but major positive self-model claim was invalidated and no qualified subject exists |
| real-text representation/retention transfer | **45%** | ARK-018 V4 remains a bounded executed negative/threshold miss; imported ARK-014 is narrow order-robustness evidence with zero retention failure events; natural-language transfer remains gated |
| target-scale 500M/5B scientific readiness | **20–25%** | engineering plumbing exists, but Citadel data/evaluation is not ready, formation is floor-limited, and no current authorization exists for any listed scale or production vocabulary |
| narrow formation/canary evidence | **45%** | Canary-v2 has a structured identity failure; S5 v8 is complete but floor-limited; v12 is 2/24 partial; neither supports promotion |
| HORM miniature program | **10%** | HORM-003/004 are bounded miniature negatives; HORM-004 has zero success variance and HORM-001/002 custody is blocked |
| K8/TPU engineering handoff | **35%** | K8 is completed engineering-partial and the 100M TPU preflight is not run; no tool-learning, qualification, or RSI result |
| demonstrated AGI | **0%** | no current result supports a general AGI claim |

A rough weighted program maturity for **doing credible AGI research** is much higher than the maturity of the AGI capability itself. The repository is becoming good at falsifying itself; the neural Core is still far from demonstrating broad continual general intelligence.

---

# PART V — THE MOST IMPORTANT NUMBERS WE ACTUALLY KNOW

The original phase-2 measurements are preserved below with their honest denominators and confounds. Phase-3 rows are appended or revised where a completed receipt supersedes an old pending or readiness claim.

| Finding | Number | Honest interpretation |
|---|---:|---|
| generic continuation LM loss | 2.1884 → 1.9710 | better loss did not produce tested cognition |
| Citadel T1D heldout arithmetic exact | 0–6.6% | no lift-off across tested low-budget arms; confounds recorded |
| T1D teacher primitive heldout | 51.5% | primitives learnable without composition transfer |
| Triquetra value-only repair | 46.7% DEV; 46.2% replication | value recency strongly elicits behavior on weak V4 |
| Triquetra raw query-conditioned rank | 25.0% at 25% chance | measurable query control approximately absent on that substrate |
| ARK-007R HIGH retention failure | 9/12 = 75% | high LR unstable after acquired Micro-T2 state |
| ARK-007R LOW retention failure | 0/12 = 0% | low LR protective in that matched Micro-T2 regime |
| ARK-010 HIGH post-collapse recovery | 8/9 = 88.9% | movement/plasticity helps reacquisition |
| ARK-010 immediate LOW recovery | 2/9 = 22.2% | freezing too early can prevent recovery |
| ARK-011 HIGH recurrent collapse | 3/6 = 50% | recovered state can destabilize again |
| ARK-011 HIGH→LOW recurrent collapse | 0/6 = 0% | post-recovery low switch protected in Micro T2 |
| ARK-015 narrow HIGH invariance failure | 8/8 = 100% | high plasticity + narrow support eroded broader capability |
| ARK-015 narrow LOW failure | 0/8 = 0% | low LR protected |
| ARK-015 augmented HIGH failure | 0/8 = 0% | continued diverse support also protected despite large movement |
| CYR-011 compact exposure | 44.89% | compact bridge timeboxed early |
| CYR-011 compact heldout STANDARD | 56.47% | partial generalization, not G90 |
| CYR-011 production exposure | 100% | full ARK reference semantic dose |
| CYR-011 production heldout STANDARD | 0% | memorization without structural heldout generalization |
| CYR-011 production SEALED | 0/48 = 0% | same negative on reserved measurement |
| CYR-012/R1 class-space endpoint (512k rows) | V19 12.94% / V4096 100% / V24576 0% | declared class-space size alone moves formation; non-monotonic |
| CYR-013/R1B response curve (128k rows, 2 seeds) | V4096 0.506/0.494; V8192 0.718/0.0; V16384 0.647/0.129; extremes ≈ 0 | intermediate regime reproducible, amplitude seed-sensitive |
| ARK-017 V2 primary screen | HIGH 4/6 fail; LOW/CAP1X/replay-1/16/joint/augmented 0/6 | both retention levers independently sufficient |
| ARK-018 V4 Birth internalization | +0.000 / +0.067 vs required +0.10 | preregistered threshold NOT MET (2 seeds) |
| ARK-018 V4 binding-acquisition slowdown | 1200 / >1500 vs 300 / 300 steps | replicated narrow plasticity cost of Birth-10% |
| ARK-019 V3.1 old-skill robust-min | PLASTIC_HIGH ≈0.005; STATIC 1/64 ≈0.882; Guardian ≈0.963 | plastic continuation destroys; replay protects; recovery is not prevention or internalization |
| R1C completed arms | 24/24; mean `MASK_4096-FULL_24576` gap `-0.11038062283737024`; K01 fired | inactive-softmax competition is not sufficient in the fixed-physical-V24576 development test |
| CS-TRANSFER development/sealed | mean development identity-AUC gap `-0.034765625`; sealed endpoint gap `-0.1395833333333333`; 1/4 development positive; 3/4 sealed favor V24576 | physical V4096 is not a robust remedy; interaction and one reversal remain |
| Canary-v2 | 360 updates; 1,474,560 tokens; identity `0.177005/0.189107` dev/sealed | mechanical path can pass while a narrow identity family fails |
| Formation-Mux S5 v8 | 24/24 sealed; identity `0.000–0.008` across arms; termination `1.00` | formal NULLs are floor-limited and cannot exonerate a mechanism |
| Formation-Mux v12 frontier | 2/24; no sealed/final/checkpoint payload | partial custody snapshot only |
| HORM-003 / HORM-004 | 3/5 sign-consistent seeds; HORM-004 success fraction `0.0/5` | miniature negatives; HORM-004 did not test rich dynamic appraisal |
| K8 | E0/E1 executed; E2 not positive; E3 blocked; E4/E5 not run | engineering-partial, not a tool-learning or cognition result |
| Citadel data/evaluation | latest-position `1.000` per tier; 530 cross-split leaks; 13.5% duplication | old surface cannot support a new positive lift-off claim |
| ARK-014 | order robustness supported in narrow rerun; retention failure events `0/3` matched orders | narrow binding evidence only |
| BRAMASTRA EOS experiment | 0/32 → 32/32 (both seeds) | EOS supervision mechanically required |

---

# PART VI — WHERE WE FAILED, IN SMALL FORM

A failure remains evidence only at the scope recorded by its receipt. Historical phase-2 failures are retained; phase-3 rows distinguish scientific negatives from engineering failures, blocked custody, and external-only material.

| Failure | What it taught us | Status now |
|---|---|---|
| lower LM loss without cognitive movement | loss cannot be the promotion metric | **DEMONSTRATED negative** |
| candidate scorers dominated by token/length bias | scoring instrument must be certified before model conclusions | **DEMONSTRATED failure** |
| T1D/T1C arithmetic did not lift off | low-budget CE can learn distribution/format without exact computation | **executed negative; T1D had known confounds** |
| EOS never trained in T1D | termination and content were conflated | **contract repaired conceptually/elsewhere** |
| curriculum did not accelerate Micro T2 | easy→hard staging is not automatically helpful | **REJECTED in tested Micro regime** |
| teacher rows did not create composition | primitives can improve while composition stays absent | **null for sufficient composition at tested dose** |
| selective-binding headline failed independent replication | format familiarity was mistaken for robust selection | **REJECTED** |
| entity-addressing interpretation | value recency, not entity identity, explained much of the repair | **reattributed** |
| X1 self-model “PASS” | class imbalance let a trivial predictor win | **CONTRADICTED** |
| readiness gate v1 | floor substrate produced a false green | **CONTRADICTED; v2 replaces it** |
| LOW LR as universal cure | failed under cross-task no-replay interference and post-collapse recovery | **REJECTED as universal** |
| exact LR switch threshold | time/state aliasing prevented identification | **NOT_IDENTIFIED** |
| update-cap mechanism | event rate too low to assign causal credit | **INCONCLUSIVE** |
| native BF16 optimizer state | clip-norm invariant violated | **REJECTED locally** |
| CYR-GPU-006 | hardware feasibility gate correctly stopped before science | **non-scientific preexecution failure** |
| CYR-GPU-009 | semantic exposure <22% of ARK reference | **valid narrow null, insufficient for strong acquisition conclusion** |
| CYR-GPU-010 | batch-16 design would still underdose semantic rows | **superseded before execution** |
| CYR-GPU-011 production representation | M99 but 0% heldout at full ARK exposure | **historical bottleneck; phase-3 representation questions remain open after R1C/CS** |
| CYR-GPU-012/013 | monotonic vocab stories falsified (V4096 100% > V19 12.94% > V24576 0%; intermediate regime seed-sensitive) | **DEMONSTRATED non-monotonic class-space effect** |
| CYR-GPU-014-R1C | 24/24 complete; `SOFTMAX_COMPETITION_NOT_SUFFICIENT`; K01 fired; mean gap `-0.11038062283737024` | **DEMONSTRATED controlled negative; no mask-only or production remedy follows** |
| ARK-017 V2 | both retention levers independently sufficient (HIGH 4/6 fail; LOW/CAP1X/replay/joint/augmented 0/6) | **DEMONSTRATED mechanism dissection** |
| ARK-018 V4 | Birth internalization threshold NOT MET (+0.000/+0.067 vs +0.10); science NLL +3.7–3.9%; binding-acquisition slowdown replicated | **DEMONSTRATED negative on primary; real-text transfer partially answered** |
| ARK-019 V3.1 | `CONTROLLER_NOT_SUPPORTED`; new skill never formed in any arm; PLASTIC_HIGH destroyed old skill 4/4 | **DEMONSTRATED: formation gates continual learning, Guardians recover but did not prevent** |
| ARK-019 V4 | transcribed external `GUARDIAN_CONTINUAL_PROXY_CANDIDATE`; raw bundle absent | **external-only/unverified; not an executed authority and not a controller result** |
| ARK-020 V4 | resume and partial-identity defects confirmed; A1 repair plan exists; all authorization flags false | **`DO_NOT_RUN/NOT_EXECUTED`; no scientific result** |
| Formation-Mux S5 v8 | identity endpoint at zero floor while formal contrasts print NULL | **floor-limited/inconclusive; no mechanism exoneration** |
| HORM-004 | five-seed miniature `NOT_SUPPORTED`; success fraction 0.0 on every seed | **bounded negative; rich dynamic appraisal unresolved** |
| K8 | E0/E1 engineering execution; E2 no positive cognition; E3 data blocker; E4/E5 absent | **engineering-partial `INCONCLUSIVE`** |
| Citadel | shortcut, leakage, duplication, supply, and fixture failures | **`NOT_READY`; old positive lift-off claims blocked** |

Failures are not waste; they remove bad explanations. But a failed experiment with a confound does not justify a universal negative claim.

---

# PART VII — PHASE-3 WHAT TO BUILD NEXT, IN ORDER

The phase-2 roadmap contained a representation-gate sequence, but its R1C launcher-repair and full-exposure compact-bridge steps are historical, not current actions. R1C and CS-TRANSFER are completed evidence. The current order begins with a cheap process correction: establish that the control can form the target capability and that the checkpoint metric can resolve it away from the floor. No expensive mechanism campaign may start from a floor-limited surface.

## Gate 1 — FMUX-CONTROL-METRIC-PREFLIGHT

**Purpose:** establish a multi-seed control capability and checkpoint metric resolution on the Formation-Mux identity surface. This is a cheap preflight, not a new broad formation campaign. It should inspect preserved checkpoints and custody where valid, and use a read-only custody/metric review when checkpoint continuity or metric resolution is uncertain.

**Pass conditions:** the control forms a measurable identity/copy signal above the floor; the checkpoint has an auditable lineage; exact, token-level, per-position, and longest-common-prefix diagnostics are calibrated against the same answer contract; and the result is not driven by a partial-generation or EOS artifact.

**Stop conditions:** if control capability or metric resolution cannot be demonstrated away from the floor, stop, preserve the evidence, and return to custody or measurement repair. A formal NULL, a mechanical pass, or a partial frontier snapshot is not a pass.

**Why first:** Formation-Mux S5 v8 completed 24/24 and sealed formal NULLs while identity remained at `0.000–0.008` in all arms; v12 is only 2/24. Without resolution, a tied-row or weight-decay contrast has no dynamic range.

## Gate 2 — exact Formation-Mux recovery and conditional frozen-frontier completion

Gate 1 does not recover missing state. Recovery-preflight engineering passed in GitHub Actions run `35925460488` after failed run `35923273288` exposed a preserved control-key defect, but the original checkpoint-bearing Kaggle saved Output is still absent. Obtain that exact Output and pass the pinned same-kernel recovery preflight before any continuation. If Gate 1 also proves that the control and metric resolve capability away from floor/ceiling, continue the frozen remaining 22 TIE-role development arms without changing protocol, seeds, exposure, endpoints, thresholds, or sealed policy. If either capability or custody fails, stop; CI qualification is not recovery and the partial archive is not resumable state.

## Gate 3 — ROLE-TRANSFER-001 implementation and readiness, preregistered but blocked

`ROLE-TRANSFER-001` supersedes the old `TIED-ROW-GEOMETRY-WD-001` placeholder as the design of record; the two must never run as duplicates. The frozen design uses `T0_CANONICAL`, `T3_RAW`, `T3_NORM`, and `P_NORM`; 12 fresh blocks in two replications; pair-specific full-preclip norm matching; per-update hash-chained receipts; at most 5% clipping; a 25% gradient-role manipulation gate; fixed two-million-token exposure per arm; and a clean-room within-new-ontology production-BPE endpoint.

No official execution is authorized. Before a separate authorization review, the upstream frontier must be complete, the new generator/trainer/evaluator and exact-resume state must be implemented, sealed commitments and no-overlap proofs must be frozen, and independent remote engineering qualification must pass. Protocol-hash CI is not a scientific result.

## Gate 4 — CORPUS-REGEN, in parallel and required

Regenerate and qualify the production corpus and evaluation surfaces while Gates 1–3 proceed. The package must include source/family-disjoint splits, exact and near-dedup evidence, contamination scans, shortcut baselines, EOS supervision, frozen fixtures, sufficient supply, a frozen tokenizer identity, and a wired milestone evaluator. Citadel’s current data and evaluation audits are `FAIL`/`NOT_READY`; this is a prerequisite for credible future natural-language or scale transfer, not a scale authorization.

## Gate 5 — later natural-language or scale transfer, only after all gates

A future natural-language or larger-scale transfer design is admissible only after:

```text
FMUX control capability and metric resolution pass
→ original Output recovery and conditional frozen-frontier completion pass
→ ROLE-TRANSFER-001 protocol, substrate, clipping, manipulation, and replication gates pass
→ regenerated corpus and evaluation pass leakage, shortcut, supply, and sealed-firewall screens
→ candidate-free transfer and retention/regression checks pass
→ fresh replication and durability receipts exist
→ a new explicit decision record authorizes the next scale step
```

This is a gate sequence, not automatic permission. No current record authorizes PRE500M, 250M, 500M, a production vocabulary or tokenizer change, cognition, AGI, TPU qualification, tool learning, or RSI.

## Closed and blocked branches

- **R1C:** closed 24/24 controlled mechanism result; do not repeat it as a mask-only campaign.
- **CS-TRANSFER-001:** closed controlled development result; do not treat V4096 as a robust remedy or natural-language transfer proof.
- **Historical full-exposure compact-bridge proposal:** superseded by the class-space response curve, R1C, CS, and the formation floor; it is not a successor action.
- **ARK-019 V4 Guardian transcription:** raw bundle missing; not an executed authority.
- **ARK-020-V4:** `DO_NOT_RUN/NOT_EXECUTED`; resume and identity defects remain, and all authorization flags are false.
- **Formation-Mux v12:** 2/24 partial custody snapshot; recovery-preflight engineering is qualified, but the original checkpoint-bearing Output is absent and no recovery execution occurred. No architecture promotion or mechanism conclusion.
- **ROLE-TRANSFER-001:** preregistered execution-blocked design of record; no trainer, official arm, sealed evaluation, or result. It supersedes the old tied-row placeholder and must not be duplicated.
- **K8/TPU:** K8 is engineering-partial; the TPU preflight is not run. Neither unlocks tool learning, qualification, cognition, or RSI.
- **Citadel old surface:** shortcut/leakage/supply failures block a new positive lift-off interpretation.

## Current decision boundary

The phase-3 machine ledgers and decision model are authoritative. The immediate scientific order is `FMUX-CONTROL-METRIC-PREFLIGHT`, then exact Output recovery and conditional frozen-frontier completion only if capability and custody pass. Preregistered `ROLE-TRANSFER-001` is the blocked successor of record; `CORPUS-REGEN` proceeds in parallel; later natural-language or scale transfer is downstream of all validity and clean corpus/evaluation gates. Until then, the correct action is evidence repair and qualification, not promotion.

---

# PART VIII — CURRENT BEST AGI RESEARCH HYPOTHESES

## H1 — representation/class-space can change whether structural capability emerges

**Status: SUPPORTED at development scale; mechanism narrowed but not identified.** CYR-011 established a condition-level representation divergence, CYR-012 isolated declared tied class-space as a non-monotonic factor, and CYR-013 replicated the intermediate direction across two fresh seeds. R1C completed 24/24 and rejected inactive-softmax competition as the sufficient tested explanation at fixed physical V24576. CS-TRANSFER then found physical V4096 non-robust and interaction/seed-sensitive. Tied-row geometry, parameterization/initialization, optimizer/weight-decay/denominator participation, task effects, and natural-language transfer remain open. No vocabulary or production-representation change follows.

## H2 — capability emergence can be delayed far beyond memorization

**Status: DEMONSTRATED at Micro symbolic scale.** ARK-002B replicated the qualitative memorize-first → delayed-generalize transition with large seed variance (rho 0.00 between M99 and G90 timing). This means stopping immediately after train saturation can miss a later structural transition.

## H3 — acquired capability can narrow without canonical accuracy falling

**Status: DEMONSTRATED at controlled Micro non-arithmetic scale.** ARK-015 retained canonical exact at 1.0 while order/query-order invariance eroded under narrow high-plasticity continuation.

## H4 — retention is an interaction between plasticity and ongoing support

**Status: STRONGEST CURRENT MECHANISM PICTURE; levers shown independently sufficient (ARK-017 V2: `BOTH_LEVERS_SUFFICIENT`); single necessary-mechanism attribution still open.** LOW protected; CAP1X protected; sparse treatment-exact replay protected; total parameter movement does not explain outcomes. Dose universality and scale transfer remain unresolved.

## H5 — acquisition/recovery and retention need different control regimes

**Status: SUPPORTED at Micro T2.** HIGH helps recover absent capability (8/9), LOW protects present capability (0/12), and HIGH→LOW after recovery reduces recurrence (0/6). Exact state thresholds were not identifiable (ARK-012) and scale transfer is unresolved.

## H6 — query-conditioned addressing is a real missing operation, but current weak-substrate evidence cannot justify a learned self-model

**Status: SUPPORTED as a bottleneck hypothesis / self-model NOT_DEMONSTRATED.** Triquetra measured chance-level query ranking and strong recency effects (replicated), BRAMASTRA independently showed aggregate binding accuracy is fully explained by query-blind copying, and the X1 self-model PASS was invalidated by a trivial-baseline audit.

## H7 — primitives do not automatically compose

**Status: SUPPORTED.** T1D teacher primitives learned to ~51.5% held-out while full arithmetic composition stayed near floor.

## H8 — better data/measurement should be tested before exotic architecture

**Status: STRONG INFERENCE.** No current evidence shows MoE, recurrence, SSM, latent thought or neural long-term memory is the binding bottleneck. Adding them before resolving representation mechanism, objective, data support and capability formation would reduce interpretability.

## H9 — new-skill formation gates every continual-learning verdict

**Status: DEMONSTRATED (formation-first lesson).** ARK-013 (T3 never acquired) and ARK-019 V3.1 (the new skill never formed, so the official controller verdict was `CONTROLLER_NOT_SUPPORTED` interpreted as a formation bottleneck) show that controller comparisons are not interpretable before the unprotected reference can acquire. The phase-3 V4/ARK-020 records are historical readiness artifacts only: V4 is transcribed without raw bytes, and ARK-020 is `DO_NOT_RUN/NOT_EXECUTED`. The law remains formation-first; it does not authorize a controller.

## H10 — a floor-limited formal NULL is a measurement failure until the instrument is qualified

**Status: SUPPORTED as a process correction.** Formation-Mux S5 v8 completed and sealed its protocol, but identity was near zero in every arm while one-token termination resolved. The next question is whether token-level, per-position, or checkpoint diagnostics can show capability before exact-match flips. A floor-limited NULL cannot select a mechanism, clear a floor, or authorize promotion.

## H11 — physical class-space effects may interact with optimizer and row geometry

**Status: OPEN, preregistered but execution-blocked.** R1C removes one softmax explanation and CS removes a simple physical-V4096 remedy, but neither identifies the remaining tied-row/parameterization/optimizer/WD/denominator interaction. `ROLE-TRANSFER-001` is now the design of record, with synchronized arms, full-preclip norm matching, 12 fresh blocks, clipping/manipulation gates, a clean-room endpoint, and fixed exposure. It has no trainer, official arm, sealed evaluation, or result and must not run beside the superseded placeholder or before recovery/frontier and readiness gates pass.

## H12 — real-text and scale transfer remain separate from development mechanism evidence

**Status: OPEN and blocked.** The historical ARK-018 result, narrow ARK-014 rerun, R1C, CS, Canary-v2, and Formation-Mux do not jointly establish natural-language transfer, production corpus readiness, or scale readiness. Regenerated corpus/evaluation, sealed fixtures, away-from-floor formation, geometry, retention, and fresh replication are all required.

## H13 — Guardian/controller claims require raw custody and a formation-capable science parent

**Status: OPEN and blocked.** ARK-019 V3.1 is a formation-gated negative, V4 is an unverified external transcription, and ARK-020 is not executed. A controller cannot be credited with preserving or acquiring a capability that the unprotected reference cannot form. Static replay remains an external comparison, not an internalized controller.

---

# PART IX — NON-NEGOTIABLE RESEARCH LAWS

1. **Behavior beats loss.** Loss is substrate evidence, not a cognition verdict.
2. **Execution artifacts beat prose.** A plan, README or newer commit is not a result.
3. **Preregister before outcome-sensitive execution.** Changing a metric after seeing the result creates a new exploratory claim.
4. **Candidate-free behavior is primary.** Assisted scoring is diagnostic unless its own bias screen is qualified.
5. **Train the behavior you evaluate.** If EOS terminates generation, EOS must be part of the training contract.
6. **Keep causal variables separate.** Do not call a combined query+order change “query sensitivity.”
7. **Shared-parent matched forks are the default for retention experiments.** Same bytes, same future stream, one treatment variable.
8. **A strong claim needs replication.** One seed can generate a hypothesis, not a universal law.
9. **Keep CONTROL and SEALED firewalled.** Sealed results cannot become training data or tuning feedback.
10. **Preserve negative results and false greens.** They are part of the system’s anti-self-deception memory.
11. **Do not promote from aggregate accuracy.** Inspect worst family, invariance, sensitivity, transfer, realization and retention.
12. **Scale only after the smaller experiment identifies what should scale.** Compute is not a substitute for causal clarity.
13. **No architecture soup.** Add one mechanism when a measured bottleneck demands it.
14. **No AGI/consciousness/identity claims from narrow task competence or Birth Book internalization.** Those conclusions require entirely different evidence.
15. **Preflight control capability and metric resolution cheaply before expensive mechanism campaigns.** A mechanism contrast is uninterpretable when the control cannot form the target behavior or the metric cannot resolve it. Do this before broad sweeps, architecture changes, or scale.
16. **Formation must be demonstrated away from both floor and ceiling.** A near-zero exact-match baseline, constant endpoint, saturated metric, or all-failure control cannot identify a mechanism; a ceiling can hide the same distinction. Report resolution, dynamic range, and alternative metrics.
17. **Engineering, partial, preflight, custody, and external-only evidence cannot create scientific outcomes.** These records can establish executability, custody status, or a blocker; they cannot establish cognition, mechanism credit, generalization, tool learning, TPU qualification, AGI, or RSI.
18. **A completed negative is not an invitation to repeat the same hypothesis.** Rerun only with a materially different preregistered question, a repaired instrument, or genuinely new evidence that changes identifiability.
19. **No promotion from a floor, a formal NULL, a mechanical PASS, aggregate accuracy, or a transcription.** Promotion requires a qualified control, orthogonal causal endpoints, clean transfer, retention/regression, fresh replication, and explicit authorization.
20. **Preserve custody boundaries.** Reconstructed hashes, manifests, checkpoint receipts, and partial archives are not raw-byte recovery; do not overwrite or silently normalize historical evidence.

---

# PART X — PHASE-3 ONE-PAGE AGENT HANDOFF

If an agent reads only this section, it should use the following read order and decision boundary.

## Read order

1. [`../../research/EVIDENCE_SOURCE_MANIFEST_2026-09-24.md`](../../research/EVIDENCE_SOURCE_MANIFEST_2026-09-24.md) and [`../../research/EVIDENCE_SOURCE_MANIFEST_2026-09-24.json`](../../research/EVIDENCE_SOURCE_MANIFEST_2026-09-24.json): exact source refs, 71 byte-identical files, SHA-256 `c164c735ec4e628a6311fe4c52117c8b58370321df61b00d4976c11769f0215f`, exclusions, and external-only artifacts.
2. [`../../research/EXPERIMENT_EVIDENCE_LEDGER.md`](../../research/EXPERIMENT_EVIDENCE_LEDGER.md) and [`../../research/EXPERIMENT_EVIDENCE_LEDGER.json`](../../research/EXPERIMENT_EVIDENCE_LEDGER.json): all 90 experiment rows, statuses, metrics, relations, and claim ceilings.
3. [`../../research/NEGATIVE_RESULTS_LEDGER.md`](../../research/NEGATIVE_RESULTS_LEDGER.md), [`../../research/CAUSAL_IDENTIFIABILITY_AUDIT.md`](../../research/CAUSAL_IDENTIFIABILITY_AUDIT.md), and [`../../research/CAUSAL_KNOWLEDGE_GRAPH.md`](../../research/CAUSAL_KNOWLEDGE_GRAPH.md): what is negative, blocked, floor-limited, or identifiable.
4. [`../../research/NEXT_PHASE_DECISION_MEMO.md`](../../research/NEXT_PHASE_DECISION_MEMO.md), [`../../research/NEXT_3_EXPERIMENTS.md`](../../research/NEXT_3_EXPERIMENTS.md), and [`../../research/EXPERIMENTS_TO_CANCEL_OR_DEFER.md`](../../research/EXPERIMENTS_TO_CANCEL_OR_DEFER.md): current order, conditional gates, and stop rules.
5. This document and [`../../cymek/research/CROSS_BRANCH_EVIDENCE_AUDIT.md`](../../cymek/research/CROSS_BRANCH_EVIDENCE_AUDIT.md): human synthesis and adversarial audit.

The historical phase-2 branch table and old ledgers are context, not current authority. The historical evidence cutoff is 2026-09-13; the phase-3 review date is 2026-09-24. Commit `90f77b7f` is the pre-consolidation parent; the current branch contains the standalone consolidation.

## Current evidence boundary

| Question | Current answer | Boundary |
|---|---|---|
| Inactive-softmax competition | R1C complete 24/24; `SOFTMAX_COMPETITION_NOT_SUFFICIENT`; K01 **FIRED**; mean gap `-0.11038062283737024` | Fixed physical V24576 controlled development mechanism only; no global softmax or optimality claim |
| Physical V4096 remedy | CS-TRANSFER complete `PARTIAL_OR_INTERACTION`; one reversal; V4096 not robust | Controlled development-scale only; raw Drive result external; no natural-language or scale transfer |
| Canary-v2 | 360 updates / 1,474,560 tokens; mechanical PASS; identity formation FAIL | Narrow canary; mechanical execution is not cognition or production readiness |
| Formation-Mux | S5 v8 24/24 sealed but floor-limited; v12 frontier 2/24 partial; recovery engineering CI passed but original Output absent | No mechanism exoneration, recovery execution, frontier verdict, or architecture promotion |
| ROLE-TRANSFER-001 | Preregistered execution-blocked four-arm design; no trainer, official arm, sealed evaluation, or result | No efficacy, mechanism, external benchmark, capability, architecture, or AGI claim |
| HORM-003/004 | Separate five-seed miniature `NOT_SUPPORTED`; HORM-004 success fraction 0.0 | HORM-004 did not test rich dynamic appraisal; HORM custody blocked |
| K8 / TPU | K8 engineering-partial `INCONCLUSIVE`; TPU preflight not run, zero updates | No tool learning, TPU qualification, cognition, AGI, or RSI |
| Guardian / ARK-020 | ARK-020 `DO_NOT_RUN/NOT_EXECUTED`; Guardian V4 raw bundle missing | No Guardian or controller result |
| Citadel | Data/evaluation `NOT_READY`; leakage, shortcut, duplication, supply, and fixture failures | No production corpus or positive lift-off claim |
| ARK-014 | Narrow order-robustness evidence; retention screen zero-failure-event | No broad capability or retention claim |

## Immediate order

1. `FMUX-CONTROL-METRIC-PREFLIGHT`: cheap multi-seed control-capability and checkpoint-metric-resolution audit. Use read-only custody/metric review if lineage or resolution is uncertain. Stop if the control remains at floor/ceiling or the instrument cannot resolve the endpoint.
2. `ROLE-TRANSFER-001`: design of record but preregistered and execution-blocked. Run only after FMUX capability/metric and exact-custody gates, conditional frozen-frontier completion, implementation, clean-room evaluator/no-overlap proof, and independent remote qualification. Never run the superseded tied-row placeholder beside it.
3. `CORPUS-REGEN`: in parallel; regenerate and qualify source/family-disjoint data, EOS supervision, shortcut/leakage/contamination screens, supply, tokenizer identity, and sealed fixtures.
4. Later natural-language or scale transfer: only after clean corpus/evaluation, away-from-floor/ceiling formation, geometry, retention/regression, fresh replication, durability, and explicit authorization gates.

R1C and CS-TRANSFER are closed evidence. The historical full-exposure compact-bridge proposal is superseded. ARK-019 V4 is not an executed authority because its raw bundle is missing. ARK-020-V4 is not an authorization. No result or ledger row authorizes a production vocabulary or tokenizer change, PRE500M, 250M, 500M, cognition, AGI, TPU qualification, tool learning, or RSI.

## Operating rules

Preserve the conventional dense decoder Core, external verifier, fail-closed custody, candidate-free primary evaluation, and orthogonal causal endpoints. Use a shared-parent matched fork and a preregistered claim ceiling for any retention test. Keep CONTROL and SEALED firewalled. Preserve failures and external references. Never promote from a floor, formal NULL, mechanical PASS, partial custody snapshot, aggregate accuracy, or transcription.

The program’s strongest result is disciplined falsification and several bounded discoveries: delayed generalization after memorization, state-dependent acquisition/recovery/retention behavior, non-arithmetic invariance narrowing, independently sufficient retention levers, a non-monotonic class-space response, and a completed narrow R1C mechanism negative. None establishes AGI or production readiness. The next valuable step is cheap capability-and-metric qualification, not a broad mechanism or scale campaign.
