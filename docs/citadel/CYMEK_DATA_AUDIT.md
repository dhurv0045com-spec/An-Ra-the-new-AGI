# CYMEK_DATA_AUDIT.md

The data system of `origin/cymek` @ `26a61f6`, audited 2026-09-03. Companion to
`CYMEK_EXECUTION_GRAPH.md` (pipeline connectivity) and `CYMEK_DATA_SHORTCUTS.md` (leakage risks).
Per-stage fields: input contract / output contract / determinism / provenance / resume /
failure behavior / tests / receipts / limitations / scientific assumptions.

---

## 0. Headline finding

**The intended training data does not exist and no code produces it.** Every committed execution
receipt consumed the same 48-file walk of Cymek's own repository (`v5_training/miniature.py:_load_corpus`,
~74K tokens, families `{natural, code_math_formal}`). `verified_cognition` tokens consumed by the
production path to date: **0**. The e0 training generator (`e0-train/0.2.0`) emits 3 ad-hoc
templates and is called by nothing outside the dev-certificate smoke test; the evaluation
generators (12 families) are certification infrastructure. Two of the nine planned cognition
families (`interference_retrieval` 75M, `faithful_realization` 37.5M tokens) have **no generator
at all**, and no natural/code corpus loader exists. The E3 plan's generator hash is `None`
(`artifacts/e3/static_plan.json`, `BLOCKED_UPSTREAM_INPUTS`).

## 1. Stage table

| Stage | Input contract | Output contract | Determinism | Provenance | Resume | Failure behavior | Tests | Receipts | Key limitations / assumptions |
|---|---|---|---|---|---|---|---|---|---|
| source → Document | none enforced (documents arrive in memory) | `Document(doc_id, text, source_id, domain, family, authorization_category, acquired_date)` | n/a (caller) | source_id + category restricted to 4 values | none (stateless) | `ValueError` on bad category | indirect | miniature receipt (48 repo files) | **No corpus loader of any format.** Byte-hash provenance computed then discarded (`miniature.py:125,146`); manifest hashes text, not bytes. ASSUMPTION: first-party synthetic/repo data is contamination-free. |
| manifest (dedup/split/scan) | `list[Document]` + salt + boundaries + count_tokens | `DataManifest` (sources, tokens_by_family, total) + audit | deterministic (`exact_clusters` by content SHA-256; `assign_split` = SHA-256(salt\0key) bucketed) | `manifest_sha256`; per-source `raw_sha256` | split is a pure function — re-runnable | duplicate source reuse rejected (`assert_source_disjoint`); boundaries must cover 4 splits | `test_v5_data_pipeline.py` (contamination fail-closed, cluster split) | miniature receipt | Splits assigned by **content hash, not by generator intent** — a generated sealed-eval case piped through this path would be hash-assigned, discarding its split label (latent contract gap; no adapter exists today). Exact-dup clustering only; the spec's promised *near*-duplicate gate (`training_spec.py:118`) is **not implemented**. |
| mixture | fractions (frozen center) | integer token allocation, exact by largest-remainder | deterministic | fractions hash-bound into run spec | n/a (pure) | `ValueError` off-sum/off-list | `test_v5_data.py` | none executed | Allocation math only — **nothing enforces mixture at consumption time**. Executed runs consumed a single unintended "mixture". Scientific status of 0.65/0.20/0.15 + 9 family fractions: see §2 table. |
| sampler | shard hashes, run_seed, epoch | permutation of shards | deterministic (sha256 of `run_seed/epoch/shard_hash`) | order hashable | re-derivable from cursor | n/a | `test_v5_data_pipeline.py` | **never executed** | Tested-only module; every executed run hand-sliced windows instead. |
| packer | `(doc_id, token_ids, source)` triples | `MultiPackedShard` sequences (BOS/EOS chunks, buckets 512/1024/2048/4096, per-segment IDs) | deterministic (doc_id sort; stream-fill) | `shard.sha256()` content-addressed; `pack_ledger` exact | final partial sequence = resume boundary | duplicate doc_id, bad markers raise | 9 tests (multi-doc, ledger, determinism, chunking) | miniature: efficiency 0.96673 | Packed segments are attention-isolated (block-diagonal mask) — but **causal twins (`…-base` / `…-changed`) pack adjacently** (doc_id sort); inert only while the mask is correct (see SHORTCUTS §pairing). Cognition examples (30–250 tok) all land in the 512 bucket — the planned bucket mix (0.25/0.25/0.30/0.20) is unreachable with cognition-only data. |
| cursor → batch | shards + sampler order + coordinates | `MicroBatch` (tokens, segment_ids, tokens_by_source, consumed) | deterministic | consumption cross-checked against `cursor.advance` (two accounting paths must agree exactly) | coordinates are the resume unit | stale coordinates / overrun / ledger disagreement raise | 9 tests | **never executed** | The exact edge Citadel's one-update cert exercises. |
| trainer/state | `TrainingState` + 4096-real-token updates | advanced state, receipts, checkpoints | deterministic per seed | every update carries `IdentityBindings` (8 hashes) | single-update transactions; `completed_update` rolls back to `committed_update`; LR never rewarms | abort on nonfinite loss/grad, token mismatch, identity mismatch, clip breach | 12+ tests (`test_v5_step_schedule_trainer.py`, `test_v5_training.py`, `test_v5_production_backend.py`) | miniature + 2 canaries | `tokens_per_update` fixed 4096 in all executed runs (spec allows 131,072 for production). |
| objective | packed batch + segment IDs | CE loss + supervised-token count | deterministic | loss bound to segment/mask semantics | n/a | zero eligible targets raises | `test_v5_objectives.py` | receipts record per-update loss | CE-only by frozen decision (D-041). Query-swap implemented, weight exactly 0, gated to E3 Phase B. |
| schedule | cumulative tokens | LR | pure function | self-hashing receipt | never rewarms | rejects float/step indexing | `test_v5_step_schedule_trainer.py` | **canonical WSD never executed** — runs bind `bounded_warmup_schedule` (constant 3e-4) | |
| evaluation | checkpoint payload + tokenizer + tasks | candidate scores, constrained choice, free generation | deterministic (greedy) | adapter identity hash-bound (checkpoint+payload+spec+tokenizer) | n/a | prefix-property violations fail closed; `VisibleTask` cannot carry gold (TypeError) | `test_v5_checkpoint_adapter.py`, `test_evidence_firewall.py` | miniature: 2 hardcoded tasks | Task supply is hardcoded (`MINIATURE_EVAL_TASKS`); no harness generates eval tasks from the e0 generators in the production path. Firewall is an in-process type discipline, not a process boundary. |
| promotion | evaluation dossier + signature | PROMOTE/REJECT/INCONCLUSIVE | deterministic | signature + detached hash; chronology-proof selection | milestones immutable | missing dossier entry ⇒ every gate FAILs; unknown gate inventory raises | 9 tests (synthetic dossiers) | **never fed real evidence** | |

## 2. Exact current mixture — and the epistemic class of every number

Frozen center (`v5_contracts/run_spec.py`, `v5_data/mixture.py`, mirrored in `training_spec.py`):

| Value | Class | Evidence |
|---|---|---|
| 5,000,000,000 total tokens | engineering constant | round budget; no learning curve justifies it |
| slices 0.65 natural / 0.20 code-math-formal / 0.15 verified-cognition | **preregistered hypothesis** (never measured) | E3 plan hypotheses 5/15/30% cognition; 15% is the "hypothesis" arm, frozen as launch default before any measurement |
| 9 cognition family fractions (0.08…0.20) | **preregistered hypotheses / design guesses** | relative weights track benchmark family importance, no measurement |
| difficulty shares 0.34/0.355/0.305 | **placeholder** | no code produces or enforces difficulty labels; only structural knobs exist in generators; e0's own histograms don't match these shares |
| bucket mix 0.25/0.25/0.30/0.20 + 20-microstep supercycle | engineering default | no cognition-only workload can realize it (all cognition examples fit bucket 512) |
| tokens/update 131,072 (executed: 4,096) | engineering constant (executed: cert scale) | |
| WSD 50M warmup / 4.5B constant / 500M decay, peak 3e-4 | engineering default (~standard practice) | executed runs use constant 3e-4 instead |
| `REQUIRED` 15 tokens/param minimum budget | engineering guard | prevents absurd under-training in the run spec |
| contamination scan n=8, fail-closed | engineering control (aligned with ESOES doctrine) | |

**Summary:** nothing in the data mixture has been *scientifically measured*. The fractions are
preregistered hypotheses frozen as defaults — legitimate as engineering baselines, invalid as
science until E3 (or Citadel's C1-class experiments) measures them.

## 3. Capability map (family → intended pressure → generator reality)

| v5 family (share of cognition slice) | Intended pressure | Generator reality |
|---|---|---|
| identity_copy 8% | verbatim span reproduction / realization | eval template exists; regex-solvable by design |
| query_binding 16% | select 1-of-N bindings under query swaps | eval generator hardened (causal pairs, swaps); **no train-side analogue** |
| semantic_state 16% | state tracking under semantic time, rollback, priority | eval generator hardened (shuffled serialization, cutoff≠timestamps); **no train-side analogue** |
| interference_retrieval 10% | retrieval under interference | **no generator at all** |
| relational_composition 20% | multi-hop composition | eval generator + direct-retrieval control; **no train-side analogue** |
| counterfactual_sensitivity 10% | answer flips with premise flip | eval generator (1-bit task); **no train-side analogue** |
| heldout_rule_induction 10% | infer latent rule from demos | eval generator (operation-name leak — see SHORTCUTS); **no train-side analogue** |
| missing_information 5% | abstention | eval generator is **constant-label** (always `<MISSING>`) |
| faithful_realization 5% | reliable output realization | measurement only (`measure_realization`); **no generator** |

Burden-of-proof note: where an eval generator exists, its *pressure* is plausible from the causal
pair contracts, but **transfer from training exposure has never been tested** — no training data
for these families has ever existed. The only training generator (registry/revision/transfer)
covers ~3 simplified surfaces and is serialization-leaky (see SHORTCUTS).

## 4. Resume and determinism guarantees (tested claims)

- Split assignment, packing, sampler order, mixture allocation: pure functions of
  (content, salt, seed, shard hashes) — re-runnable bit-exactly.
- Training resume: single-update transaction; interrupted volatile updates roll back;
  cursor/ledger byte-match enforced at restore; LR is a pure function of cumulative tokens
  (no rewarm possible by construction). Mechanically certified by the backend and by
  Citadel's independent certification runs.
- Generator determinism: `build_training_examples(seed, count)` is seeded and re-runnable.
