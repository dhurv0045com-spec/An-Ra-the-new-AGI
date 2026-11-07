# ELENDRIL V6 — Grand Diagnostic Experiment
**Research protocol / prospective design**  
**Reference commit:** `0cd200730d43547294cdc9bf8540a478130cce9c` · **Prepared:** 7 November 2026  
**Registry SHA256:** `dcbc1be6acb28a42a89fe63078d066339c56682fa66e67a629e2e4c9c3562abe`  
**Not a trained result; no cloud CUDA executed.**

## 1. The scientific question
Can the existing Elendril neural learner acquire cost-aware **ANSWER / QUERY / ABSTAIN** behavior across new evidence worlds; maintain correct **post-query revision**; distinguish missing, contradictory, null, and unavailable evidence; and show any credible benefit from better learning objectives, representation, memory, reasoning depth, or scale? More importantly: **which failure class blocks the next capability and what experiment discriminates its cause?**

The V4 neural policy previously scored 0/4 on changing-state revision and issued no queries. V5 has correct infrastructure and an 8-update four-prompt QUERY/EOS overfit test, but no general learned-action campaign. This is the best existing bottleneck to begin with. Preserve both original results and the V5 implementation. Never treat engineered fixed baselines as neural success.

## 2. Data and scale are diagnostic variables, not prestige targets
- Target approximately **2–3 MB of unique, generated training records** before compression, with explicit SHA256 and counts. Use independently generated latent worlds, not duplicated strings to reach a byte goal. Add new disjoint validation and sealed confirmation sets; audit world IDs, queries, prompt collisions, templates, and near duplicates. If a wider generator is needed, implement it and audit it separately from the learner. Do not treat 2–3 MB as a large language corpus.
- Start with a **~3M-parameter** V5-compatible core. Condition subsequent ~8M and ~20M probes on measured learning, memory, timing and per-GPU VRAM. Verify the **exact** parameter inventory through `ModelSpec` and an instantiated model; table figures are approximate.
- “20 MB model” ≠ “20 million parameters”: 20M FP32 weights alone take roughly 80 MB (80,000,000 bytes), excluding gradients/optimizer/activation memory. No assertion of T4 fit before device canaries.
- Legacy 46-token symbolic path is the fairness control. The V5 260-token UTF-8 path is a separate representation experiment; it must not silently reinterpret or warm-start incompatible associative checkpoints. Hold semantic examples and compute as comparable as practicable.

### Candidate scales
| Profile | Width × layers | FFN width | Context | Nominal parameters |
|---|---:|---:|---:|---:|
| S3 | 256 × 4 | 768 | 128 | ~3.1M |
| S8 | 384 × 5 | 1,024 | 128 | ~8.0M |
| S20 | 512 × 7 | 1,408 | 192 | ~19.9M |

## 3. One notebook, staged and resource-bounded
Kaggle T4×2: treat each 16 GB GPU as an independent lane. Prefer parallel independent arms/seeds over unqualified distributed training; do not pool VRAM. Total target ~5–6 hours of wall time, including a **35-minute export reserve**; actual limits and quotas vary. The campaign must observe wall time and measured step times and prune or skip arms whose complete matched exposure cannot fit; do not record partial arms as valid comparisons. Avoid local CPU/GPU research training entirely (only short syntax/mechanics tests).

| Stage | Max elapsed allocation | Decision | Inputs / outputs |
|---|---:|---|---|
| S0 — Qualification | 30 min | Verify SHA, real two T4 UUIDs, data integrity, gradients, exact EOS, checkpoint roundtrip and fresh-session custody | Device/canary/manifest receipts |
| S1 — Screen | 105 min | Compare S3 CE, weighted-CE, curriculum, augmentation and an explicitly qualified learned control arm, with shared data/exposure | Paired, costed development results |
| S2 — Replicate | 105 min | Replicate the strongest candidates with new independent seeds | Confidence/cluster summaries |
| S3 — Conditional scale | 75 min | Only if S3 has a positive signal and forecasts fit; probe S8 then S20 | Quality/cost/VRAM frontier |
| S4 — Diagnose/export | 45 min | Run all qualified diagnostic panels, preserve NOT_QUALIFIED for unsupported capabilities, produce ZIP | Verdicts, graphs, raw attempts |

Each stage's max is a budget slice, not a promise to fill the time. Halt or shorten when results are decisive. Sum is 360 minutes including export. A single long **training** run may require a separate short **qualification / fresh-VM recovery** session first, as V5 admission contracts require. Never silently weaken that gate to fit “one run.”

## 4. The 15-panel diagnostic battery
The following panels report exact sample counts, uncertainty and raw predictions. An untrained or unsupported task is **NOT_QUALIFIED**, not a pass; a qualified task with poor score is a **FAIL**.

| # | Diagnostic | Primary observable |
|---|---|---|
| P01 | Positive-control learnability | train loss; exact content+EOS; invalid tokens |
| P02 | Known-evidence answering | correct; redundant query count; calibration |
| P03 | Missing-evidence query | query precision/recall; terminal accuracy; cost |
| P04 | Conflicting-evidence query | query rate; wrong-answer rate; later resolution |
| P05 | Null and unavailable abstention | ABSTAIN accuracy; invalid actions; utility |
| P06 | Post-query revision | end-to-end episode accuracy; intervention sensitivity |
| P07 | Cost-aware thinking and queries | utility; requests; cost/quality Pareto |
| P08 | Uncertainty and self-error detection | Brier; ECE; AUROC; accuracy/coverage; extra cost |
| P09 | Legal action and EOS integrity | exact+EOS; malformed outputs; loops; abstentions |
| P10 | Query swap and query blindness | paired both-correct; input intervention effect |
| P11 | Temporal/recency shortcuts | counterfactual pair accuracy; recency control |
| P12 | Representation and distractor OOD | separate/joint OOD accuracy; loss; failure clusters |
| P13 | Compositional reasoning 2→3 hop | exact graph pair; 3-hop; shortcut controls |
| P14 | Frozen-weight memory benefit | memory on/off; retrieval choice; cost; causal swaps |
| P15 | Sequential retention and transfer | A-before/after; B; C curves; total GPU work |

Each panel must have a frozen evaluation generator and deterministic oracle where applicable. V5's existing engineered controls include fixed, always-query, no-query, random, query-blind, recency and privileged oracle. Preserve origin attribution. New panels should not leak information through channel payloads, split labels, generation seeds, hidden truth, formatting markers or test-generated template classes.

## 5. Training arms and causal attribution
| Arm | Method | Stage | Rule |
|---|---|---|---|
| A0 | fixed policy; always-query; never-query; random; oracle separate | all panels where appropriate | no-train |
| A1 | CE AdamW, uniform exposure, established V5 core | screen | S3 |
| A2 | action-weighted CE, identical examples and exposure | screen | S3 |
| A3 | uncertainty-first curriculum; equal full-epoch row multiset | screen | S3 |
| A4 | independent alias/template augmentation; same budget and audit | screen | S3 |
| A5 | learned decision or evidence-seeking regularizer; explicit objective and baseline | screen-if-qualified | S3 |
| A6 | winning S3 mechanism; scale probe | only-after-screen | S8 |
| A7 | winning S3 mechanism; scale probe | only-after-efficiency-gate | S20 |
| A8 | best two preregistered S3 competitors, new seeds and sealed generator | only-after-screen | S3 |

At minimum, compare CE, action-weighted CE and uncertainty-first at the **same model size, init seeds, examples, supervised-token exposure, optimizer steps and train-time budget**, with declared exceptions where objectives cause different weighted denominators. New representation/augmentation comparisons must account for token count differences and confounds. Do not make a claim that an arm improved because it had more data, additional compute, or an easier evaluator.

## 6. Preregistered decision gates
Primary endpoint: full-episode utility `correct - 0.1 * verified_queries`, with exact content+EOS/invalid-action handling. Also report inference model calls, positions and GPU resources **separately** so utility does not hide extra thinking cost.

- Existing V5 prospective screen gates are the default: **>=90%** complete-episode accuracy, **>=80%** per public-state kind, **0** invalid actions and at most **0.05** excess requests/episode above fixed, across both preregistered screen seeds.
- Alternative versus CE: at least **+0.05 utility per episode in each seed**, and no more than five percentage points subgroup accuracy regression. These are screening gates, **not evidence of independent confirmation**.
- Only independently authored, sealed confirmation data can justify scientific promotion. Use world-level or cluster-aware confidence intervals; report every tested alternative and account for adaptive selection. Never select “best of 15” on the sealed test repeatedly.

### Mandatory result → next-decision logic
| Finding | Next scientific action |
|---|---|
| Tiny overfit/canary fails | Halt GPU search; debug objective, tokenizer, gradients or EOS; do not scale |
| Learns train but not new worlds | Check split leakage, template memorization, inductive bias, data coverage |
| Knows answers but never queries | Test action objective and learning signal; no memory/AGI claims |
| Queries always and passes accuracy | Add cost-aware training, matched query-cost comparisons |
| Good familiar worlds, poor aliases/templates | Target representation invariance; new independent generator |
| 2-hop good, 3-hop bad | Diagnose fixed-depth architecture and information flow |
| Retrieved memory helps but choice is scripted | Train decision-to-retrieve policy before claiming learned metacognition |
| A formation strong, B causes forgetting | Compare replay/protection under equal work; no transfer claim |
| S20 slower/worse than S3 | Stop scaling; identify data/model/compute bottleneck |
| Severe OOM/runtime/recovery failure | Stop and preserve partial receipts; fix infrastructure first |

## 7. Notebook + ZIP delivery contract
One top-level **`ELENDRIL_V6_KAGGLE_T4X2.ipynb`** must run all qualified experimental arms and panels. Before launch: owner-enabled pinned-source checkout, data SHA, safe native GPU/PyTorch inventory, real T4 UUIDs and VRAM, cross-device numerical policy, accurate measured forecasts, immutable checkpoint generations, same-device roundtrip and **fresh-session** custody reattachment. No auto-launch and no installation that destroys Kaggle's CUDA stack. Capture all results, including aborted arms, and create one downloadable **`ELENDRIL_V6_RESULTS.zip`**, plus a SHA256SUMS manifest. Save plots and a plain-English decision tree inside the ZIP. Ensure resumability when time/quotas end.

The draft notebook delivered with this protocol is intentionally **NOT YET A LAUNCHABLE TRAINING NOTEBOOK**: V6's dataset generator, 15-panel evaluator and campaign operator are new code Codex must implement. It has safe default cells, exact current Git source pin as a historical base, structured experiment plan and explicit fail-closed checks. After Codex implements and tests the V6 source, it must repin the notebook to the new immutable commit and qualify real T4 hardware before owner-run training.

## 8. Scope of claims
A successful action policy is evidence of learned decision selection within a narrow synthetic environment. It is not evidence of human-level cognition, subjective consciousness, general language ability, AGI, or autonomous recursive self-improvement. A clever learned improvement policy requires later prospective intervention-selection tests. The campaign's purpose is to reveal the next limiting factor, not manufacture a “singularity” headline.

### Official runtime references
- Kaggle notebook instructions: https://www.kaggle.com/docs/notebooks
- Kaggle T4×2 update (Sept 2026): https://www.kaggle.com/product-announcements/735239
- Kaggle runtime image: https://github.com/Kaggle/docker-python
- PyTorch CUDA memory: https://docs.pytorch.org/docs/stable/torch_cuda_memory
