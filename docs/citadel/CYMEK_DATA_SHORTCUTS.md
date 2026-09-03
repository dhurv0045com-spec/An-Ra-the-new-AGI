# CYMEK_DATA_SHORTCUTS.md

Shortcut and leakage risks in the data that would feed Cymek's cognition slice, audited
2026-09-03 against `origin/cymek@26a61f6`. Scope: the generators that would produce training
data (`e0_cognition/training_generators.py`, `e0-cognition evaluation generators`) and the
pipeline's exposure to each risk. The evaluation generators were already red-teamed once
(`artifacts/e0/shortcut_repair_receipt.json` — the v0.3.0 false green); the *training* stub has
never been audited and is materially weaker. Each risk names its cheapest detection experiment.

## A. The training generator (e0-train/0.2.0) — the only wired cognition producer

| # | Risk | Evidence (file:line) | Cheapest detection |
|---|---|---|---|
| A1 | **Serialization/recency cue — no shuffling.** Relevant fact is always `facts[-1]` in revision mode; transfer mode lists the chain in path order with the distractor last. A model can answer from "last line" without any state tracking. | `training_generators.py:61-64,73-81` | Generate 10k examples; run E0's `latest_fact`/`nearest_position` baselines over them — predicted ≈1.0 on mode 1. (Fix pattern exists: eval side shuffles after graph build, `evaluation_generators.py:434-436`.) |
| A2 | **Lexical-overwrite cue.** Answer appears verbatim in the fact line containing the queried key; solvable by lexical lookup. | `training_generators.py:56-58`; cf. committed cert `lexical_overlap` = 1.0 on analogous eval families | Run `lexical_overlap`/`bag_of_words` baselines over emitted training examples (needs a `TrainingExample`→`CausalCase` adapter that does not exist). |
| A3 | **Template monoculture.** 3 templates cover the whole would-be curriculum; 9/12 eval families have no train-side analogue at all. | `training_generators.py` (123 lines total); call-path trace: only `certify.py:98` calls it | n-gram/template cross-match train↔eval; count distinct skeletons per family before any large generation. |
| A4 | **No baseline audit anywhere.** E0's 12-baseline heuristic suite runs only over eval `CausalCase`s; the training stub is never screened. | `baselines.py` input type; `certify.py:98-99` checks namespace disjointness only | Extend the E0 baseline gate to rendered training documents (same pooled-seed protocol). |

## B. Evaluation generators (would-be transfer targets / dev-tier metrics)

| # | Risk | Evidence | Cheapest detection |
|---|---|---|---|
| B1 | **Candidate-attestation cue.** Distractor candidates never appear in the context (binding: replacements, `evaluation_generators.py:190`; state: `:442-445`; rule: 8/16 candidates built from tokens absent from the demo context, `:660-666`). "Pick the candidate that appears in context" is a runnable heuristic. | committed cert: `lexical_overlap` 1.0 on 1-hop families | One-pass statistic: fraction of candidates absent from context per family; gate on it. |
| B2 | **Constant-label family.** `missing_information` always queries an absent entity — always-`<MISSING>` scores 1.0. | `evaluation_generators.py:562-582`; solver `reference_solvers.py:95` never exercised | Add present-entity control variants; gate on *balanced* accuracy (promotion gate already expects `missing_information_balanced_accuracy`, `training_spec.py:257` — nothing generates the negative case). |
| B3 | **Rule-operation name leak.** `operation = f"KEL-{prefix}-{index % 8}"` deterministically encodes the latent structure; `template_id` leaks it too. At training scale the name→rule map is memorizable. | `evaluation_generators.py:647-648,680` | Re-key operation names many-to-many across seeds; detect by training a lookup from operation string → answer vs demo-derived answer. |
| B4 | **1-bit counterfactual task.** Answer fully determined by `every` vs `no` in fact 0. | `evaluation_generators.py:591,613` | Quantifier-surface paraphrase variants; measure a quantifier-token-ablation solver's drop. |
| B5 | **Recency residue on 1-hop relations.** `latest_fact` = 0.625 on `relation_1_hop` (committed cert) — above its null, inside tolerance but notable. | `artifacts/e0/development_certificate.json` | Extend the analytic permutation-null pool to relation families. |
| B6 | **Held-out is lexical, not structural.** Split profiles share skeletons; "fresh" differs in lexicons/prefixes. Generator docstring admits it (`:639-645`). | `PROFILES`, `:31-56` | Structural-holdout variant (new skeletons) before any "generalization" claim. |

## C. Pipeline-level risks (independent of generator content)

| # | Risk | Evidence | Cheapest detection |
|---|---|---|---|
| C1 | **Causal twins co-packed.** `pack_documents` sorts by `doc_id`; sensitivity twins (`…-base`/`…-changed`) are adjacent and land in the same sequences. Block-diagonal masking makes this inert *only while the mask is correct*; any mask regression becomes systematic per-pair answer leakage. | `pack.py:201`; twin id scheme in generators | Packing fixture asserting no base/changed twins share a sequence (or interleave doc ids); add a mask-regression unit test that scores twin-in-sequence vs twin-split. |
| C2 | **Split labels discarded.** Manifest assigns splits by content hash; a generated sealed case routed through `build_data_manifest` would be hash-assigned, potentially into `training`. No adapter exists today (latent, not active). | `manifest.py:76-78` vs `CausalCase.split` / `TrainingExample.split_identity` | Adapter contract test: generator split label must survive to `SourceRecord.split`; refuse documents whose declared split ≠ hash-assigned split for eval-origin content. |
| C3 | **Near-duplicate gate missing.** Spec promises "exact plus near-duplicate cluster assignment" (`training_spec.py:118`); only exact hashing exists. Template-level repetition at 750M-token scale ≈ 10⁵–10⁶ repeats per skeleton. | `v5_data/split.py:44-54`; token math in CYMEK_DATA_AUDIT.md §3 | Sample 1M examples/family; gate on distinct-skeleton count and self-8-gram collision rate before scaling. |
| C4 | **Contamination scan never fed real benchmarks.** Executed runs pass `contamination_benchmarks={}` or 2 toy prompts. | miniature/canary drivers | Wire the e0 eval-suite prompts into every manifest build. |

## Priority

Blockers for *any* cognition-training claim, in order: **A1/A2** (the training data that exists is
shortcut-saturated — training on it would measure shortcut acquisition, not cognition),
**C1/C2** (pipeline can silently invalidate pair-based evaluation), **B1/B2** (eval families have
runnable heuristics that must be gated before any "above null" claim), then A3/B3.
C3 gates scale-up, not the first micro-experiment.
