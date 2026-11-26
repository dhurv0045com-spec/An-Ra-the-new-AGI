# AN-RA — Fundamental Reconstruction and Cognitive Mechanism Discovery
## Research Investigation, Defect Audit, Architectural/Algorithmic Design, and Gemini Implementation Mandate (2026-11-26)

> **Scope:** Branch `Elendril` (`77eb95d677e3ff5665d6c46472fefed71de9c7fe` baseline), strictly at **~20M (`S20`, `19.87M` core backbone)** plus `<100k` CPU canary. **No 50M (`S50`) scaling jobs are authorized or included in this phase.** All historical evidence and implementations (`Experiment Result/V8-X1/*`, `Experiment Result/V8-X2/*`, `Experiment Result/V8-X3/OWNER_RUN_01*`, `anra_v5/elendril/v8x/*`, `experiments/x2/*`, `experiments/x3/*`) remain preserved untouched.

---

## 1. Verified Experimental History — What Prior Experiments Actually Established

### 1.1 V8-X1 and V8-X2 Baseline Evidence
- **`anra_v5/elendril/v8x/`** established the clean historical baselines:
  - `X1TypedModel` (`models.py:124-390`) implements `B0`, `J`, `F`, `F-cap`, `P`, `P-capR`, `R`, `R-noedge`, `U-1`, `U-2`, `U-4`, `U2-untied` on top of a ~19.87M S20 core with strict **causal prefix masking** (`valid_prefix = (pos_grid <= pos_idx.unsqueeze(1)) & same_seg` at `models.py:230,321`) and **normalized probability-space copy–generate mixing** (`p_arg = (1.0 - g) * p_val + g * p_copy` at `models.py:341-342`).
  - **Architecture signal:** Explicit pointer attention (`P`) dramatically outperformed bare action heads (`J`/`F`) on direct symbol binding.
  - **Algorithmic caveat:** D2 (`PAIR-EQ`, `PAIR-EQ-CF`) bundles 4 linked views per base world inside every 16-state minibatch (4 unique base worlds/step vs up to 16 in D1-H interleaved) and alters preceding observation-stage exposure (`experiments/x3/d3_objectives.py:57-223`).
  - **R0–R3 pending ledger:** As reconciled in `OWNER_RUN_01.md:10`, 49/192 historical X2 obligations were credited with 125 mandatory and 18 conditional pending (`U-2`/`U-4` iterative depth degradation remained a genuine architectural puzzle).

### 1.2 V8-X3 Owner Run 01 (`7654543b`, Verified in `OWNER_RUN_01.md` & `OWNER_RUN_01_METRICS.json`)
- **Scale & Execution:** All **45/45 S20 fits** (15 configurations × seeds `7, 19, 41`, 1,024 updates/job, 3.54 worker-hours GPU) completed; all 12 conditional S50 jobs stayed closed (`0.0` GPU seconds). All 515 manifest entries in the 3.35 GB FULL archive verified cleanly.
- **Aggregate Held-Out Exactness (10 C-families, 20 cases/family):**
  - Highest confirmation macro-mean: **`S3 P-R-Iter / D3-Hybrid` = `16.67%`** (vs `S3 P-R-Iter / D2` = `8.17%`), OOD = **`10.83%`** (vs **`4.67%`**).
  - **In Stage S2 (`ARCH`):** All four matched controls (`AddMem` `13.83%`, `Fixed` `13.67%`, `NoCopy` `13.00%`, `NoRel` `14.00%`) beat integrated `P-R-Iter` (`11.33%`) on confirmation, while `P-R-Iter` only edged them out on OOD (`10.50%` vs `8.67–9.67%`).
  - **Complete Zeroes on 3 Key Cognitive Families:** **0/20** across all 45 fits on BOTH confirmation and OOD (and **0/12** on dev) for:
    - `C03_GRID_TRANSFORM_ARC` (`0/900` across all 45 models on confirm & OOD),
    - `C04_COMPOSITIONAL_SCAN` (`0/900` on confirm & OOD),
    - `C10_LANGUAGE_MATH_TRANSFER` (`0/900` on confirm & OOD).
  - **5 Learning Modes (4 seed-19 checkpoints):** Zero persistent-state accuracy (`0.0`), zero reset effect (`0.0`), `0.0` tool-free transfer vs `1.0` supplied oracle, `0.0` cross-domain probe, and `25–50%` on 8 C01 post-adaptation queries without pre-update baseline isolation.
- **Post-Hoc Checkpoint Forensic Probes (`OWNER_RUN_01.md:39-57`):**
  1. **Familiar vs Fresh Gap:** S3 `P-R-Iter / D3-Hybrid` seed-7 scored **28/30 (93.3%)** on exposed training prompts vs **4/30 (13.3%)** on fresh dev prompts — and specifically **9/9 on exposed train vs 0/9 on fresh dev** inside `C03/C04/C10`! Thus the model learned to memorize exact training strings rather than generalizable rules or transferable syntax.
  2. **Exposed-Data Confound in D2 vs D3:** D2 received 200 presentations per 25-batch cycle covering only **100 distinct cases** (duplicating `pair_cases` and dropping `independent_cases`), whereas D3-Hybrid received 400 presentations covering **319–329 distinct cases**. The S3 D3 > D2 gain (`+4.5` to `+12.5` pp confirm) is therefore confounded with doubling batch size and tripling unique case exposure.
  3. **Verified Future-Target Information Leakage:** Mutating answer payload bytes *after* fixed `ANSWER:` prefix changed causal backbone activations by `0.0`, yet changed `action_logits` beyond `1e-5` on **17/30** dev prompts (max diff **`11.726`**) and byte `logits` on **5/30** dev prompts (max diff **`0.05364`**) at S3 Hybrid checkpoints.

---

## 2. Four Independent Research Reviews

### Pass A — Architectural Investigation (`anra_v5/elendril/v8x/models.py` vs `experiments/x3/models.py`)
1. Historical `X1TypedModel` maintained strict **epistemic causality**: every reader and pointer attended exclusively to prefix positions `pos_grid <= pos_idx` (`anra_v5/elendril/v8x/models.py:230, 321`). `X3CognitiveModel` replaced this with `valid_tok = token_ids != UTF8.pad_id` (`experiments/x3/models.py:427`), exposing future gold answer tokens to slot readers (`line 467`), pointer keys (`line 668`), `P1` `comp_k/v` (`line 596-601`), `CounterexampleReader` (`line 619-623`), `OperatorTransportBank` (`line 639`), and `EpisodicMLP` (`line 649`).
2. `X3CognitiveModel` decoupled the **19-class action head** (`action_logits` receiving pointer probs, relational outputs, `P1` bilinear composition, `CE` signed gating) from the **actual generated bytes** (`gen_logits = F.linear(h + 0.25 * ctx_feat.unsqueeze(1), ...)` at `lines 687-688`), creating an **architectural split-brain**: all benchmark exactness is scored on auto-regressively generated bytes (`evaluate_x3_batched:658`), yet pointer attention (`ptr_sym_probs`) and normalized copy-generation fusion NEVER reach `gen_logits`!
3. Historical `R` (`v8x/models.py:109, 150-299`) had pointer access (`uses_pointer=True`) + Slot-Attention + directed `edge_mlp(3s -> eh -> 1)` + `rel_gru` + `r_to_ptr`. X3 `R` and `R-noedge` dropped the pointer completely (`models.py:250`) and replaced directed pairwise edges with `peer_mean = (slot_sum - w) / (K - 1)` (`lines 489-490`).
4. `U2`/`U4` failed in X2 and mixed in X3 because unanchored iterative recurrence lacks a clean causal contract with generation positions (`h_act` shifts every decoding token in `generate_bytes` at `line 735`).

### Pass B — Learning-Algorithm Investigation (`anra_v5/elendril/v8x/objectives.py` vs `experiments/x3/d3_objectives.py`)
1. `X3` claims `CE` and `D3-NoAux` strip auxiliary losses, yet `d3_objectives.py:583, 622-623` STILL add `0.25 * act_ce + 0.1 * meta_loss + 0.02 * ponder_loss` in every objective.
2. `D1` and `D2` in `d3_objectives.py:464-466` drop `independent_cases` and duplicate `pair_cases`, cutting unique world exposure by ~3x relative to `D3-Hybrid`.
3. Counterfactual loss (`d3_objectives.py:608-620`) only looks at indices `offset` and `offset + 1`, checks inequality via colliding `gold_class % 16`, and optimizes untargeted L1 divergence `relu(margin - 0.5 * |p_a - p_b|_1)` on `action_logits` — which can be minimized even when BOTH counterfactual states predict wrong answers (`y_hat != y*`) and ignores byte-level sequence correctness (`gen_logits`).
4. Cross-stage deduplication (`module.py:1794-1805`) and `X3_PAIRED_CONTRASTS` (`module.py:1382-1383`) broke R's hybrid-vs-ablation contrasts into `0.0` self-comparisons.

### Pass C — Data & Benchmark Investigation (`experiments/x3/cognitive_battery.py`)
1. **`C03_GRID_TRANSFORM_ARC`:** Train uses ONLY `flip_h`/`flip_v` and colors `0..3` (`lines 873, 921`); dev uses `rot90`/`transpose`; confirm uses `gravity_down`/`border_one` & colors `4..6`; OOD uses `tile_xor_rot180` & colors `7..9`. A model trained on X3 train has **never seen** digits `'4'..'9'` in grid transforms or any spatial rotation/gravity/border primitive, making `0/20` on confirm/OOD structurally inevitable under finite training!
2. **`C04_COMPOSITIONAL_SCAN`:** Train generates ONLY single-clause `{walk,run,look} {left,right} twice` (`lines 1025-1029`) — **literally 6 distinct prompts recycled forever**, with ZERO `jump`, ZERO `opposite`, ZERO `around`, ZERO `thrice`, ZERO `and`, ZERO `after`! So on dev/confirm/OOD (which combine `jump`, `opposite`, `around`, `thrice`, `and`, `after`), 0% held-out accuracy is mathematically guaranteed.
3. **`C10_LANGUAGE_MATH_TRANSFER`:** Train generates ONLY `denom=1` integer answers with a single phrasing template (`idx % 2` in `_generate_c10_case:1510-1519` while dev/confirm/OOD introduce `/2` rationals via `denom=2` on odd `idx` plus unseen phrasing `"splits (...) across 2 shares"`). The `'/'` rational format is never seen cleanly during C10 training alongside varied rates/units.
4. **`C09` split-tag shortcut & `canonical_structural_hash` split leakage:** `C09` embeds `split={ns}` verbatim (`lines 1453, 1462, 1467`) and `canonical_structural_hash` embeds `split_tag`, `split_motif`, `split_class`, `split`, `split_len` (`lines 1577-1645`), masking true cross-split structural isomorphisms.

### Pass D — Adversarial Scientific Review
Can prior X3 evidence establish that gradient surgery, pointer-attention, relational slots, or adaptive halting failed in principle? **No.** Every comparison in X3 was jointly confounded by:
- (i) train-vs-eval future-target leakage (`valid_tok`),
- (ii) train-vs-eval moving `apos` shift,
- (iii) pointer bypass around autoregressive byte logits,
- (iv) D2 100-case duplication vs D3 320-case exposure,
- (v) 6-command / 2-op / integer-only C04/C03/C10 training starvation paired with epoch recycling (1,024 steps over small pools -> 9/9 familiar, 0/9 fresh).

Therefore, **V8-X4 must cleanly decouple and test each hypothesis on unconfounded, leak-free, exposure-matched foundations.**

---

## 3. Complete Defect Registry (12 Confirmed A–L + 4 Newly Discovered M–P) & Mathematical Repairs

| ID | Defect | Exact Historical / X3 Location | Repaired X4 Invariant & Location |
|---|---|---|---|
| **A** | X3 `R`/`R-noedge` lost pointer & pairwise `edge_mlp` | `v8x/models.py:109,251` vs `x3/models.py:250,489` | `R-Ref` restores `uses_pointer=True` + Slot-Attention + pairwise `edge_mlp(3s->eh->1)` + `r_to_ptr`; `R-NoEdge` keeps pointer, ablates `edge_mlp`. |
| **B** | Pointer bypass during byte generation | `x3/models.py:666-689` | **Unified Causal Pointer-Assisted Byte Generation** routes `p_copy` directly into byte probability vector `p_unified` at every target step. |
| **C** | Future-target leakage in forward pass | `x3/models.py:427,467,594-668` | All readers, slots, pointers, `OpBank`, `CE-Reader` mask strictly `(token_ids != pad) & (pos_grid <= prompt_end.unsqueeze(1)) & same_seg`. Zero-diff (`<1e-6`) suffix mutation gate enforced. |
| **D** | Unnormalized weighted log-space mixture into `nll_loss` | `x3/models.py:676-679`, `d3_objectives.py:581` | **FP32 convex probability-space mixture**: `p = (1 - g) * p_gen + g * p_copy`, renormalized `p / p.sum(-1, keepdim=True)`, `log_p = log(p.clamp_min(1e-12))`. |
| **E** | Arbitrary `token_ids % 16` pointer fallback | `x3/models.py:390-394` | Eliminated. Copy gate `g_t` zeros out when copy evidence absent/ablated, cleanly falling back to `p_gen`. |
| **F** | Colliding `mod 16` auxiliary `gold_class` targets | `cognitive_battery.py:952,1059,1470,1534`, `d3_objectives.py:581` | Pure exact byte sequence log-likelihood `D0-ByteCE` + explicit semantic pair metadata (`pair_group_id`, `pair_relation`, exact partner target tokens). |
| **G** | D2 exposure confound + 1-pair untargeted L1 CF + hidden aux CE under `CE`/`NoAux` | `d3_objectives.py:464-466,581-625` | Matched-exposure batcher (`build_x4_matched_batches`); pure `D0-ByteCE` (`aux=0`); all-pairs targeted byte-sequence counterfactual margin + irrelevant-edit invariance. |
| **H** | Moving `apos` in `generate_bytes` vs fixed `apos` in training | `x3/models.py:735` vs `d3_objectives.py:328,568` | Fixed `prompt_end_positions` anchored at end of prompt during BOTH training and autoregressive decoding (+ reusable prefix cache). |
| **I** | Split tags inside `canonical_structural_hash` | `cognitive_battery.py:1577-1645` | Zero `split`/`split_tag`/`split_motif`/`split_class` in `canonical_structural_hash`. Verified split-label invariance. |
| **J** | Learning-mode Mode 2/3/4 eval flaws | `x3/models.py:850,916`, `x3/module.py:1510-1518` | Mode 3 uses `format_x4_typed_target(d)`; Mode 2 adds feedback vs no-feedback vs reset vs unrelated-history vs frozen controls; Mode 4 separates tool-free vs cross-domain sets. |
| **K** | Unconditional ACT loop & unanchored U state | `x3/models.py:478-516,547-570` | `U2-Fix`/`U4-Fix`/`U-P` implement anchored residual refinement `h(t+1) = h(t) + g(t) * LN(delta(h(t), h(0)))` + adaptive halting / step-trajectory `0/1/2/4/8` telemetry. |
| **L** | Self-contrasting R tuples & cross-stage silent dedup | `x3/module.py:1382-1383,1699,1794-1805` | Stage/cycle-qualified arm keys `(cycle, arch, obj, data, scale)` + explicit non-self-contrasting tuple assertion. |
| **M** *(New)* | Mismatched calibration confidence (`action_logits` vs `generate_bytes`) | `x3/module.py:656-678` | Calibration uses unified emitted sequence geometric mean token prob `exp(seq_log_prob / len)` from actual generated bytes. |
| **N** *(New)* | Explicit `split={ns}` prompt leak in C09 | `cognitive_battery.py:1453,1462,1467` | Removed all split strings from C09 prompts. |
| **O** *(New)* | C03/C04/C10 train primitive starvation | `cognitive_battery.py:873,921,1025,1510` | Repaired primitive-to-composition coverage (all 10 grid colors + core grid ops, all SCAN verbs/dirs/mults/connectives, integer + rational units) while holding out novel compositions. |
| **P** *(New)* | Cross-objective hidden `meta`/`ponder` terms | `x3/d3_objectives.py:622-625` | `D0-ByteCE` enforces `eq=0, cf=0, meta=0, ponder=0, act=0` identically. |

---

## 4. Unified Output Mechanism & Architectural Designs (§5, §9–§12)

### 4.1 Unified Causal Pointer-Assisted Byte Generation (§5)
For any token generation step `t` at or after `prompt_end`:
$$p_{\text{gen}}(v \mid x_{\le t}) = \text{softmax}(W_{\text{emb}} h_{\text{gen}, t} + b_{\text{lm}})_v \quad \text{(in FP32)}$$
$$a_{t, j} = \text{softmax}_{j \le \text{prompt\_end}}\!\left(\frac{q_t^\top k_j}{\sqrt{d_m}}\right), \qquad p_{\text{copy}}(v \mid x_{\le t}) = \sum_{j \le \text{prompt\_end}} a_{t, j} \mathbb{I}[x_j = v]$$
$$g_t = \sigma(w_g^\top [h_{\text{dec}, t}; c_{\text{prompt}}] + b_g) \cdot \mathbb{I}[\text{copy\_enabled}]$$
$$p_{\text{unified}}(v \mid x_{\le t}) = \frac{(1 - g_t) p_{\text{gen}}(v) + g_t p_{\text{copy}}(v)}{\sum_{u} [(1 - g_t) p_{\text{gen}}(u) + g_t p_{\text{copy}}(u)]}$$
Training minimizes negative log-likelihood directly on $\log p_{\text{unified}}$, unifying copy evidence selection, literal byte emission (`C03` grids, `C04` actions, `C10` rationals), and typed prefixes (`ANSWER:`, `QUERY`, `RETRIEVE`, `ABSTAIN`).

### 4.2 Principled Core & Novel Architectures (~20M `S20`, Strict Prefix-Causal)
- **`Base`**: Pure causal Transformer backbone.
- **`P-Ref`**: Historical pointer mechanism integrated directly into `p_unified` via causal prefix mask.
- **`R-Ref` & `R-NoEdge`**: Restored pointer (`R-Ref` WITH `uses_pointer=True` + Slot-Attention + directed pairwise `edge_mlp(3s->eh->1)` + `r_to_ptr`), vs matched `R-NoEdge` ablating only pairwise `edge_mlp`.
- **`P1-Alpha` vs `P1-ParallelRead`**: 3-hop sequential query-conditioned evidence reader ($w_{k+1} = \text{GRU}(\text{Attn}(w_k, K_{\le \text{end}}, V_{\le \text{end}}), w_k)$) vs matched parallel multi-head read.
- **`R1-Alpha` vs `R1-DenseControl`**: Query-conditioned slot attention + 4-type edge-conditioned message passing routed to both pointer/value and operation logits vs parameter-matched dense MLP.
- **`U2-Fix`, `U4-Fix`, `U2-Unanchored`, `U-P`**: Anchored residual refinement $h_{t+1} = h_t + \sigma(g_t)\,\text{LN}(\Delta(h_t, h_0))$ retaining direct access to $h_0$ at every step, evaluated across `0/1/2/4/8` steps via `eval_steps_override`, plus `U-P` combining anchored refinement with causal pointer.
- **`P1-OpBank` vs `P1-OpBank-DenseCtrl` (H1)**: Sparse-routed bank of $M=8$ low-rank ($r=16$) reusable residual operators $O_m = U_m V_m^\top$ applied after causal alignment reads, grafted with bounded capability-preserving live gates.
- **`P1-CE-Reader`**: Dual Support vs Counterevidence reader with signed conflict modulation.
- **`ANRA-X4-Int` (Cycle 4)**: Capability-preserving bounded residual composition of `P1-Alpha + R1-Alpha + U2-Fix + OpBank`.

---

## 5. Reconstructed Learning Algorithms (§6) & Data Curriculum (§8)

### 5.1 Matched-Exposure Batching & Distinct Objectives (`experiments/x4/objectives.py`)
Every objective receives identical minibatch tensors, case IDs, and eligible target bytes:
1. **`D0-ByteCE`**: Pure $\mathcal{L}_{\text{CE}}$ on $\log p_{\text{unified}}$ (strictly zero aux terms).
2. **`D0-Typed`**: $\mathcal{L}_{\text{CE}}$ + upweighted loss on the typed operation prefix token span `<OP>:`.
3. **`D1-CorrInvar`**: $\mathcal{L}_{\text{CE}}$ + correctness-gated invariance over explicitly linked `equiv_surface`/`equiv_perm` twins, weighted by detached anchor probability $p_{\text{anchor}}(y^*|x)$ so wrong predictions are never synchronized.
4. **`D2-ExactCF`**: $\mathcal{L}_{\text{CE}}$ over all paired views + **all-pairs** targeted counterfactual log-likelihood margin $\text{relu}(m - [\ell(x_a, y_a) - \ell(x_a, y_b)]) + \text{relu}(m - [\ell(x_b, y_b) - \ell(x_b, y_a)])$ + irrelevant-change invariance $|\ell(x_a, y_a) - \ell(x_{\text{irr}}, y_a)|$.
5. **`D3-Adaptive` vs `D3-SimpleWeight`**: `D0 + D1 + D2` with delayed warmup + PCGrad gradient conflict projection vs matched fixed scalar weights.
6. **`D4-ECCO` (Original Hypothesis H2)**: Error-Conditioned Counterfactual & Equivariant Credit Assignment — scales each counterfactual pair's structural margin & invariance loss by $w_{ab} = 1 + \beta |a_{\text{soft}}(a) - a_{\text{soft}}(b)|$ (detached), concentrating optimization onto pairs where one minimal variant succeeds and the twin fails.

### 5.2 Data & Curriculum Strategy (`experiments/x4/battery.py`)
- **22 Families in 6 Objective Groups (A–F)** covering all `X01–X12` and repaired `C01–C10`.
- **Fresh Procedural Stream (`fresh_stream`) vs Fixed Pools (`fixed_400`, `fixed_4000`, `fixed_40000`)** matching exact update/token budgets.
- **Primitive-to-Composition Curriculum (`curriculum_p2c`) vs Random Interleaved (`random_interleaved`)**: Phases 1–3 progress from atomic primitives (`k`-shot single ops, 1-clause SCAN, simple grid ops, base rational arithmetic) to full multi-step composition & counterfactual twins without changing total token budget.

---

## 6. Five Original Substantive Hypotheses (§22)

### H1 — Architecture: Compositional Sparse Low-Rank Neural Operator Bank (`P1-OpBank`)
- **Observed Problem:** Small (~20M) transformers either memorize problem instances in FFN weights (28/30 exposed vs 4/30 fresh in X3) or fail to transfer learned transformations (`C03`, `C04`, `C10` = 0%) because operations are entangled with surface tokens.
- **Proposed Mechanism:** Factor intermediate computation into $M=8$ low-rank ($r=16$) residual operators $O_m(e) = U_m(V_m^\top e)$ weighted by sparse query-conditioned routing $w(h_{\text{prompt}})$ strictly over causal prompt representations, enabling compositional reuse across tasks.
- **Why It Helps Intelligence:** Decouples *what operator to apply* (routing) from *how the operator transforms aligned state* ($U_m V_m^\top$) and *what surface tokens to emit* (causal pointer + LM head).
- **Competing Baseline:** Parameter-matched single dense residual MLP (`P1-OpBank-DenseCtrl`).
- **Discriminating Experiment:** Cycle 3 `P1-OpBank` vs `P1-OpBank-DenseCtrl` at S20 under identical `D4-ECCO`/`fresh_stream`, measuring held-out L2/L3 composition (`C03/C04/C10`) and operator-routing mutual information across task families.
- **Expected Outcome:** `P1-OpBank` beats `P1-OpBank-DenseCtrl` by $\ge +5$ pp on L3 unseen compositions with distinct family-aligned routing entropy.
- **Falsifier:** Zero improvement over dense control or degenerate uniform collapse ($H(w) \approx \log M$) across families.
- **Cost & Complexity:** $< 0.5\%$ S20 parameter overhead (`~85k` params at `d=512`), negligible wall-clock impact (`< 3%`).

### H2 — Algorithm: Error-Conditioned Counterfactual Credit Assignment (`D4-ECCO`)
- **Observed Problem:** Naive counterfactual objectives (`D2` untargeted L1) waste gradient signal on already-solved pairs or drive wrong-wrong divergence, while fixed scalar sums suffer gradient conflict on hard asymmetric edits.
- **Proposed Mechanism:** Weight all-pairs exact targeted sequence log-likelihood margins by detached error asymmetry $w_{ab} = 1 + \beta |p(y_a|x_a) - p(y_b|x_b)|$, paired with correctness-gated equivalence regularization.
- **Why It Helps Intelligence:** Focuses scarce update budget directly on minimal structural boundaries where the model currently uses brittle shortcuts rather than true causal predictors.
- **Competing Baseline:** `D2-ExactCF` (uniform pair weighting) and `D3-SimpleWeight` (fixed scalar weights) on identical batches.
- **Discriminating Experiment:** Cycle 2 matched-exposure comparison (`D4-ECCO` vs `D2-ExactCF` vs `D3-SimpleWeight` vs `D3-Adaptive`) on `P-Ref` & `R-Ref`, tracking counterfactual both-correct rate and irrelevant-edit sensitivity.
- **Expected Outcome:** Higher counterfactual both-correct rate and L2 robustness without sacrificing D0 clean accuracy.
- **Falsifier:** Equal or lower counterfactual both-correct rate vs `D2-ExactCF` or instability across seeds.
- **Cost & Complexity:** Zero extra forward passes over `D2-ExactCF` (piggybacks on existing pair logits, $< 1\%$ runtime overhead).

### H3 — Data/Curriculum: Anti-Memorization Fresh Procedural Stream + Primitive-to-Composition Scheduling
- **Observed Problem:** Fixed small corpora recycled over 1,024 updates produced extreme memorization (9/9 familiar vs 0/9 fresh on C03/C04/C10) alongside missing atomic primitives in train splits.
- **Proposed Mechanism:** Deterministic fresh procedural generation per `(seed, update_idx)` (strictly split-hash disjoint from dev/confirm/OOD) combined with 3-phase Primitive-to-Composition (`P2C`) curriculum.
- **Why It Helps Intelligence:** Eliminates memorization of fixed training prompt hashes and ensures every atomic verb/grid-op/rational concept is grounded before compositions appear.
- **Competing Baseline:** Fixed `400`, `4,000`, `40,000` pools and `random_interleaved` over the exact same total updates.
- **Discriminating Experiment:** Cycle 2 Data Axis (`fixed_400`, `fixed_4000`, `fixed_40000`, `fresh_stream` × `curriculum_p2c` vs `random_interleaved`) on `R-Ref` at matched token/update budget.
- **Expected Outcome:** `L0 - L1` memorization gap collapses dramatically under `fresh_stream`, and `P2C` unlocks non-zero C03/C04/C10 compositional acquisition.
- **Falsifier:** `fresh_stream` shows no `L1` gain over `fixed_40000` or `P2C` trails `random_interleaved` across all groups.
- **Cost & Complexity:** Deterministic CPU generation/prepack, zero extra GPU FLOPs per update.

### H4 — Capability Preservation: Live-Gradient Bounded Residual Arm Grafting (`ANRA-X4-Int`)
- **Observed Problem:** Combining mechanisms in `X3 S2` (`P-R-Iter`) degraded confirmation performance below all four single-mechanism controls because newly added arms distorted pretrained pointer/slot representations.
- **Proposed Mechanism:** Graft additional cognitive arms onto a shared pointer-relational substrate using small non-zero residual projection initialization (`gain = 0.05` on bounded `tanh`-clamped deltas with LayerNorm stability) rather than dead zero-gates or unscaled additive jumps, enforcing live gradients at step 0 and auditing against a max-regression guardrail (`<= 2` pp per group A–F).
- **Why It Helps Intelligence:** Satisfies the core AN-RA mandate: *adding an arm expands competence without breaking existing working arms*.
- **Competing Baseline:** Best single-arm Cycle 3 architecture vs unguarded direct additive integration.
- **Discriminating Experiment:** Cycle 4 (`ANRA-X4-Int` vs best single Cycle-3 model on fresh seeds `101, 202, 303` across all 22 families + `X4_CAPABILITY_REGISTRY`).
- **Expected Outcome:** `ANRA-X4-Int` achieves higher multi-group coverage than any individual arm while staying within the regression threshold across Groups A–F.
- **Falsifier:** `ANRA-X4-Int` trails single-arm winners by `> 2` pp on any established capability group.
- **Cost & Complexity:** Same S20 budget; additive per-arm overhead `< 3%` params.

### H5 — Challenging Project Assumption: Eliminating the Dual-Head Split-Brain (Unified Pointer-Byte LM vs Separate 19-Class Auxiliary Head)
- **Observed Problem:** X1/X3 split intelligence between a discrete 19-class action/mod-16 classifier (`action_logits`, carrying pointer/relational signals) and a separate byte head (`gen_logits`, starved of pointer attention) while pairing confidence from head 1 with correctness from head 2.
- **Proposed Mechanism:** Make **Unified Causal Pointer-Assisted Byte Generation** (`p_unified`) the sole primary decision, generation, counterfactual, and calibration interface (`seq_log_prob`), eliminating colliding `mod 16` class supervision and separate unlinked classification shortcuts.
- **Why It Helps Intelligence:** Every cognitive mechanism (pointers, relational slots, iterative refinement, counterexamples) must directly earn its keep by improving coherent semantic byte predictions across arbitrary open-ended answers (grids, action strings, rationals), aligning training objective with exact evaluation and calibration.
- **Competing Baseline:** `D0-Typed` / legacy dual-head supervision with separate `action_logits` auxiliary loss.
- **Discriminating Experiment:** Integrity Gate + Cycle 1/2 comparison (`D0-ByteCE` unified vs `D0-Typed`) on exact byte exactness, calibration ECE/Brier, and open-ended C03/C04/C10 vs A–P X01–X12.
- **Expected Outcome:** Unified interface improves calibration stability, removes mod-16 collapse, and allows pointer+relational gains to transfer smoothly to multi-byte answers (`C03/C04/C10`) as well as single-symbol `X01–X12`.
- **Falsifier:** Removing auxiliary action-class supervision harms X01–X12 exactness relative to `D0-Typed` even when pointer copy feeds bytes directly.
- **Cost & Complexity:** Reduces head complexity and removes `cls_head` reliance during training.

---

## 7. Four Resumable Cycles, Ambitious Honest Criteria (§19–§20), & Mandatory-Coverage Matrix (§24)

### 7.1 Four Resumable Cycles (Strictly S20, Zero S50 Jobs)
- **Cycle 1 (`C1`, 12 S20 jobs + CPU Integrity Gate):** Qualify mathematical/causal/eval integrity locally first (`integrity.py`), then run faithful foundations (`Base`, `P-Ref`, `R-Ref`, `R-NoEdge` × `D0-ByteCE` × `fresh_stream`, seeds `7, 19, 41`).
- **Cycle 2 (`C2`, 24 S20 jobs):** Controlled Algorithm × Data factorial (`P-Ref` & `R-Ref` across `D0-ByteCE`, `D0-Typed`, `D1-CorrInvar`, `D2-ExactCF`, `D3-Adaptive`, `D3-SimpleWeight`, `D4-ECCO` at matched batch exposure + `fixed_400`/`fixed_4000`/`fresh_stream` + `curriculum_p2c` vs `random_interleaved`, seeds `7, 19, 41`).
- **Cycle 3 (`C3`, 33 S20 jobs):** Architecture + Matched Controls (`P1-Alpha` vs `P1-ParallelRead`, `R1-Alpha` vs `R1-DenseControl`, `U2-Fix` & `U4-Fix` & `U-P` vs `U2-Unanchored`, `P1-OpBank` vs `P1-OpBank-DenseCtrl`, `P1-CE-Reader`, seeds `7, 19, 41`).
- **Cycle 4 (`C4`, 6 S20 jobs):** Independent confirmation on fresh seeds (`101, 202, 303`) comparing best single-arm winner vs `ANRA-X4-Int` on locked `confirm` & `ood_transfer` + full 22-family L0–L5 profile, X01–X12 interventions, C03/C04/C10 diagnostics, 5-mode learning evaluation, and capability-preservation audit.

### 7.2 Honest Two-Tier Qualification Criteria (§20)
- **`PRELIMINARY_IMPROVEMENT`:** Statistically consistent positive delta (`+3` pp macro or `+5` pp counterfactual/targeted family) over matched baseline with seed stability (`>= 2/3` seeds) and zero regression `> 3` pp.
- **`QUALIFIED` (Ambitious Competence):** Fresh-instance (`L1`) macro `>= 0.70`, Group A/C/D/E procedural core `>= 0.80`, meaning-preserving robustness (`L2`) retention `>= 0.85` of L1, counterfactual both-correct `>= 0.65`, unsupported guess rate `<= 0.05`, false-abstention rate `<= 0.05`, and capability-preservation max regression `<= 0.02` across every group A–F.

### 7.3 Mandatory-Coverage Matrix (§§1–24)

| Mandate Section | Implementing Component (`experiments/x4/*` / `tools/*` / `tests/*`) | Experiment / Check ID | Required Evidence / Contract |
|---|---|---|---|
| **§1–§2** History & 4 Passes | `research/v8-x4/AN_RA_FUNDAMENTAL_RECONSTRUCTION_MANDATE_2026-11-26.md`, `Experiment Result/V8-X3/OWNER_RUN_01*` | Historical Forensic Audit | Exact metrics + reproduced leak/exposure receipts |
| **§3** Integrity Gate (3.1–3.5) | `experiments/x4/integrity.py` (`run_foundational_integrity_gate`) | `X4-GATE-3.1..3.5` | `max_diff == 0.0` / `<1e-6` suffix non-leakage, math fixtures, FP16/0xB5 shield |
| **§4** Defects A–P Repairs | `models.py`, `objectives.py`, `battery.py`, `module.py` | `tests/test_elendril_x4_reconstruction.py` | Explicit unit regression passing per defect A–P |
| **§5** Unified Output Mechanism | `models.py` (`_compute_unified_copy_gen_log_probs`) | `X4-C1`, `X4-C2` (`D0-ByteCE` vs `D0-Typed`) | Normalized `sum(p)==1.0`, pointer directly informs bytes |
| **§6** D0–D4 Algorithms | `objectives.py` (`X4_OBJECTIVES`, `D4-ECCO`) | `X4-C2-*` | Exposure-matched manifest + pair-margin/equiv loss receipts |
| **§7–§8** Controlled Matrix & Data | `objectives.py` (`build_x4_matched_batches`), `battery.py` (`curriculum_p2c`, `fresh_stream`) | `X4-C2-*` | `unique_case_ids` equal across D0–D4; P2C vs interleaved |
| **§9–§12** P1/R1/U2/U4/OpBank/CE | `models.py` (`P1-Alpha`, `R1-Alpha`, `U2-Fix`, `U4-Fix`, `U-P`, `P1-OpBank`, `P1-CE-Reader`, `ANRA-X4-Int`) | `X4-C3-*` | Matched control contrasts + `0/1/2/4/8` U trajectory + routing MI |
| **§13** X01–X12 & Interventions | `battery.py` (`_generate_x1_family_case`, `build_x01_x12_interventions`) | `X4-C4-X-INTERV` | Privileged diagnostic intervention delta breakdown per X01–X12 |
| **§14–§15** 22 Families & L0–L5 | `battery.py` (`X4_22_FAMILIES`, `X4_FAMILY_GROUPS`, `score_x4_multidimensional_profile`) | `X4-PROFILE-L0L5` | Per-group A–F, L0–L5 table (with inapplicable reasons), unified calibration |
| **§16** C03/C04/C10 Recovery | `battery.py` (`diagnose_c03_c04_c10_failure_modes`) | `X4-DIAG-C03C04C10` | First wrong byte, parse validity, TF NLL vs free-gen exactness |
| **§17** 5 Learning Modes | `module.py` (`evaluate_x4_learning_modes`) | `X4-5MODES` | Mode 1 ICL, Mode 2 feedback/no-fb/reset/unrelated/frozen, Mode 3 typed + retention, Mode 4 distinct transfer |
| **§18** Capability Preservation | `module.py` (`evaluate_capability_preservation`) | `X4_CAPABILITY_REGISTRY.json` | Max group regression `<= 0.02` & per-arm registry |
| **§19–§20** 4 Cycles & Criteria | `module.py` (`build_x4_cycle_ledger`, `evaluate_x4_qualification_thresholds`) | `X4-C1..C4` | Resumable Palantiri ledger, 0 S50 jobs, honest QUALIFIED/PRELIM/NOT_QUALIFIED |
| **§21** Kaggle 2×T4 Engineering | `tools/build_elendril_x4_notebook.py`, `notebooks/ELENDRIL_V8_X4_KAGGLE_T4X2.ipynb` | `ELENDRIL_V8_X4_KAGGLE_T4X2.ipynb` | S20-only AMP/batched timing profile, `RESERVE_MINUTES=45`, `try/finally` salvage |
| **§22–§23** Hypotheses & Registry | `module.py` (`build_x4_failure_evidence_registry`) | `X4_FAILURE_EVIDENCE_REGISTRY.json` | H1–H5 falsification status & complete prior+new failure registry |
