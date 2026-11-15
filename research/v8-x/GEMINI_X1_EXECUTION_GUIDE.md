# ELENDRIL V8-X1 — From choosing evidence to computing with evidence

**Executable engineering and research handoff for Gemini**

Prepared by Codex, 2026-11-16, with a bounded Terra design review.

Repository: `dhurv0045com-spec/An-Ra-the-new-AGI`; destination branch: `Elendril`.

**Status: DESIGN COMPLETE; X1 IMPLEMENTATION AND GPU QUALIFICATION PENDING.** This document specifies work to implement. It does not certify a finished notebook, repaired runtime, trained relational reader, or scientific success. The owner runs substantial cloud experiments. Do not conduct research training on the local computer.

Companion file: `X1_DESIGN_REGISTRY.json` enumerates all42 proposed core jobs and their fixed common settings. It is explicitly a **design registry**, with no runtime source or launch authorization. Give Gemini both files; this guide supplies the contracts and scientific interpretation that the JSON intentionally does not duplicate.

## 1. Conclusion before implementation

The next useful leap is a learner that can **choose an information source, incorporate its response, bind the right entities and relations, and compute the correct answer under changes in naming, ordering and composition**. Selecting QUERY and then answering incorrectly is an unfinished computation, not a completed cognitive capability.

D1 contains a real positive observation worth preserving: initial agreement with the training teacher rose from 20.26% to 87.71% on missing-evidence cases and from 9.22% to 83.49% on conflicting-evidence cases. However, the six-run V8 owner experiment used B0/S20 throughout. **It changed the training recipe, not the neural architecture.** We should now perform the architectural comparisons that have not run, while retaining D1's information-seeking behavior as an explicit regression endpoint.

There is also a confirmed environment defect: the evaluator continues advertising a tool after it has been consumed. This is particularly damaging to interpretation of a policy that uses tools more often. It is an established implementation error; how much correcting it changes neural success remains unmeasured. It cannot explain incorrect first-step answers in already-answerable worlds. **The overall scientific cause remains unresolved.**

My improvement to the earlier direction is to separate instrument repair from architectural exploration. Correct transitions and qualify the measurement contract first, but **do not require B0 to succeed before training architectures intended to overcome B0's limitations**. Historical-checkpoint diagnosis and new architecture development proceed independently after shared mechanics pass. Missing historical weights do not stop X1.

X1 has three principal interventions:

1. **A relational reader R:** learn soft reusable state representations and exchange messages between them before selecting answer evidence. Compare with a conventional learned pointer, active capacity controls and a reader without cross-slot messages.
2. **D2 paired-world learning:** teach which public transformations should preserve a decision, which require an equivariant answer, and which require a changed conclusion. Compare against precisely the same paired examples trained with ordinary supervised loss.
3. **Bounded recurrent refinement U:** test whether repeated neural computation improves deeper composition, with weight-sharing and work controls. More iterations do not count as learned metacognition.

The main model floor is **S20, 19,871,616 backbone parameters**. Its size is an experimental choice, not a limit on the project's ambition. The campaign teaches up to three-hop problems and measures four- and five-hop extrapolation. It includes source revision, stale memory and two-tool compositions. Tiny models are allowed only as inexpensive mechanics fixtures. Do not replace S20 research jobs with S3 to make a launch fit.

The architecture and D2 core comprises **42 fresh S20 jobs, organized into seven resumable session groups of six jobs**. This is a substantial research program, not a claim that all work fits one Kaggle session. One notebook selects a session group, automatically runs its registered jobs after owner authorization and admission, saves after each job, and exports an independently verified bundle. Optional recurrence and transfer modules have separate budgets and activation records.

## 2. Evidence, source identity and checkpoints

### 2.1 Start here; preserve what already exists

The inspected code and audit baseline is:

```text
b727feeaa84f048b66f596a883448e581249b187
```

It was both local HEAD and the locally recorded `origin/Elendril` on inspection. The inspected worktree is `C:/Users/ankit/Downloads/An-Ra-Elendril-V8`, branch `codex/v8-mechanism-lab`. Do not assume another directory, a notebook filename, or the latest moving branch is the same source. Fetch, inspect status and worktrees, and preserve concurrent owner/agent edits. If Elendril has advanced, inspect the intervening changes and integrate this plan; do not reset to this hash or overwrite newer qualified work.

Read these sources before editing:

- `AGENTS.md` and root `progress.md`.
- `Experiment Result/V8/data-control-f5d28b8759d0/RESULTS.md` and `METRICS.json`.
- `tools/analyze_elendril_v8_results.py` and its test file.
- `research/v8/` and `anra_v5/elendril/v8/`.
- Curated reports in `Experiment Result/V6/`, `V7/`, `V7.1/`, `V7.15/`, `V7.16/`, `V7.17/`.
- Existing V5 contracts, tokenizer, optimizer, checkpoint and recovery machinery used by V8.

Preserve the original V8 runtime source:

```text
cf043942511f0b14c343445e0c3fe7570bd7c63b
```

The audit source and the owner-run runtime source are different identities. Historical scores must remain bound to the latter. New evaluator results receive a new identity, never overwrite old scores, and state whether neural inference was rerun.

### 2.2 Verified V8 observations that constrain this design

| Observation | Evidence | Correct interpretation |
|---|---|---|
| Six B0/S20 jobs completed | D0/D1, seeds 7, 19, 41 | Execution completion, not capability qualification |
| Raw final mean D0 15.92%, D1 8.59% | Original evaluator; seed-level values in the report | Both fail broad completion under that instrument; not an isolated data or architecture effect |
| D1 initial E14/E15 teacher agreement 87.71%/83.49% | Versus D0 20.26%/9.22% | Learned initial policy agreement, including the teacher's heuristic; not proof of optimal information seeking |
| D1 stale-availability records 6,754/11,520 | D0 2,051/11,520, final raw records | Differential exposure to a confirmed environment bug |
| D1 R09 initially answers 553/1,920 times, 431 wrong values | 122 exact initial answers; 909 unnecessary tool proposals | Important failure exists before any consumed-tool transition |
| Relevant-counterfactual both-correct D0 22/2,304; D1 5/2,304 | Linked original/changed worlds | No robust binding claim |
| D0 2,368 updates; D1 1,472 | Four epochs each, different state counts | D1 received 37.84% fewer updates; equal epochs are not matched exposure |
| Architecture arms J/F/P/U did not run | Owner archive inventory | Architecture ideas remain untested in this campaign |

The V8 audit replayed 126,720 recorded episodes and checked 24 milestones. Replay verifies recorded actions and environmental outcomes; it does not regenerate neural probabilities. The original analysis archive SHA256 is:

```text
f5d28b8759d0252e9c5e487e97cf1a817765def5a0c1523f7e2caeec097e7ef5
```

Historical V6 showed useful first queries without reliable final answers and weak changed-payload behavior. V7 showed some substantially better acquisition and payload use, but composition and seed stability remained weak. V7.17's controlled LR change from 3e-4 to 1e-4 improved mean confirmation from 62.80% to 84.52% on its own suite; its three-hop cases remained unsuccessful. These percentages are from different suites and cannot be plotted as a common accuracy trend. Preserve the LR evidence as the reason to start X1 at 1e-4, not as proof that optimization is solved.

### 2.3 What was actually found in Downloads

Found `C:/Users/ankit/Downloads/ELENDRIL_HITCHHIKER_FULL.zip`:

- 1,167,950,007 compressed bytes; 403 ZIP members; 1,997,853,511 uncompressed bytes from the central directory.
- Internal `analysis/identity.json`: schema `elendril-hitchhiker-archive-identity-v1`, phase `stage1`.
- Source `6859904a02c8083c4a0ab320c9c4a511899f73e4`.
- Protocol `11be27cf15a268501bb0c552438b2e12c0d02a3756e771160017e9d37043687f`.
- Fifteen `LATEST` checkpoint lineages: **S3**, arms B0/A/B/AB/C, seeds 7/19/41.

`C:/Users/ankit/Downloads/CHECKPOINT_INTAKE.json` also exists. It records an earlier archive hash and verification, but this design session inspected ZIP metadata/identity and did **not** repeat all 1.17 GB of content hashing or restore neural tensors. Reverify before using those weights. The earlier recorded archive SHA is `43a0c074cdcc3e12d8d00e0509663a2d47702586414317818ca0876579e487e2`.

No `ELENDRIL_V8_FULL.zip` was located at the top level or in the shallow Downloads search performed here. This is not a claim that it does not exist elsewhere. The Hitchhiker checkpoints are valuable separate evidence; **they are not the six V8 D0/D1 S20 checkpoints**. Never relabel them or use their outcomes as a frozen-weight correction of V8. If the user later supplies V8 FULL, validate source, protocol, per-job model spec, final update, tokenizer and checkpoint components before the V8 repair intervention. Meanwhile, all fresh X1 implementation can continue.

Use the Hitchhiker archive only in a separately named optional donor-analysis track. Do not import its code wholesale, conflate its Y1 plan with this X1 plan, or spend X1's S20 budget repeating unrelated S3 research.

## 3. Scientific contract and run vocabulary

Every report row must distinguish:

- **Observed symptom:** directly measured behavior, e.g. wrong first answer after an entity rename.
- **Established engineering defect:** violated source/runtime contract with a reproducer.
- **Supported mechanism:** a preregistered intervention changes measured behavior relative to appropriate controls.
- **Unresolved explanation:** plausible cause without discriminating evidence.

`COMPLETE` means required execution and delivery obligations are satisfied. It never means a scientific hypothesis passed. A valid campaign can finish with every scientific hypothesis unsupported. An interrupted campaign exports useful partial evidence but must not report full completion. Use separate fields `execution_status`, `delivery_status`, `scientific_status`, and `cause_status`.

Register each hypothesis with intervention, comparator, controlled variables, primary endpoint, falsifier, resource ceiling, required artifacts and classification. Register before viewing its scored development outcomes. Later amendments receive new IDs and rationale; preserve the original result.

### Hypothesis register

| ID | Intervention / comparator | Primary discriminating result | What weakens the hypothesis |
|---|---|---|---|
| H01 instrument | Frozen V8 weights, corrected vs legacy transitions | Paired change in post-tool completion and repeated-call behavior | No completion recovery despite corrected state; first-step failures persist |
| H02 surface representation | Same frozen weights/world, familiar vs unfamiliar renderer | Paired accuracy and answer-margin change | Similar failure in both renderings |
| H03 exposure | Historical D0/D1 at 1,472 and 2,368 updates, same registered horizon | Recipe-by-exposure response, with token counts | More exposure does not improve acquisition/binding |
| H04 output interface | J/F vs B0; algebraic null control | Better exact action/value learning and held-out completion | Improvement limited to legal syntax or teacher-forced labels |
| H05 evidence reading | P vs F and F-cap | Better counterfactual both-correct and raw answer completion | Capacity control matches gains, or pointer intervention has no effect |
| H06 relational computation | R vs P, R-noedge and P-capR | Composition/changed-edge benefit beyond pointer and active controls | No advantage, gains explained by added active capacity, or collapse to last-token copying |
| H07 functional equivariance | Paired CE+EQ vs identical paired CE, within P and R | Renaming/irrelevance robustness without losing correct binding | Consistent but wrong predictions, or augmentation alone explains gain |
| H08 sensitivity | CE+EQ+CF vs CE+EQ | Correct response to meaningful changes | Mere prediction change without correct new answers |
| H09 additional computation | U-1 vs U-2/U-4, shared-weight and independent-depth controls | Depth-dependent gain at measured inference cost | Extra compute helps only easy cases or equals an active feed-forward control |
| H10 preservation | Qualified skill A, same B training, no replay vs fixed replay | Less A loss while preserving B acquisition | A never acquired, or B tradeoff explains apparent retention |
| H11 forward transfer | Useful vs unrelated history vs fresh start, same C data | Faster C acquisition at a fixed criterion and work accounting | Initial familiarity/extra pretraining accounts for the difference |

H01/H02 require appropriate old checkpoints; H03 is a diagnostic module. H04–H08 form the new core. H09 is an ambitious scheduled extension. H10/H11 require actual acquisition for their interpretation, but failure of those prerequisites does not suppress the earlier architecture experiments.

## 4. Repair the instrument without erasing history

### 4.1 Established defects to fix

Inspect these current locations (line numbers can move):

- `anra_v5/elendril/v8/evaluation.py`: `_EpisodeRolloutState.apply_step`, `observe`, batched observation cache.
- `tools/elendril_v8_worker.py`: `_reconstruct_episode_steps`, profile, frozen evaluation and export integration.
- `anra_v5/elendril/v8/teacher.py`: teacher-state transitions.
- `anra_v5/elendril/v8/verifier.py`: independent episode replay.
- `anra_v5/elendril/v8/training.py`: checkpoint rotation currently prunes intermediate tensors.

For each allowed QUERY or RETRIEVE, consume that tool's availability exactly once **even when its response is empty, redundant or irrelevant**. Append only its public payload. Decrement the action budget exactly once. Repeated/unavailable proposals remain visible in the raw trace and follow the registered invalid-action outcome; never silently redirect them to ANSWER. ANSWER and ABSTAIN terminate. Tool calls consume a decision; budget four includes the terminal decision. Two tool calls plus an answer fit that budget.

Construct all next public observations from the corrected state. Version observation and verifier contracts. Preserve an explicitly named `legacy_v8` evaluation adapter for H01, loaded from or validated against the old immutable source. The legacy bug must not be the default new evaluator.

Replace the old cache key `(world_index, ordered_facts)` with the complete scoring-input identity: canonical public-input bytes/hash, tokenizer identity, observation version, ordered facts or their canonical serialization, both tool availability flags and remaining action budget. Include model/mode identity wherever cached values contain model-dependent probabilities. Construct the public input before lookup. No cross-checkpoint cache reuse.

Use the same public state contract for runtime and trace serialization; do not reconstruct availability from incomplete facts after the run. Independently replay transitions with a separate verifier implementation. A verifier must reject a stored next prompt that advertises a consumed tool even if the recorded final answer is correct. Avoid having actor and verifier call the same defective transition routine and thereby agree incorrectly.

### 4.2 Focused regression cases

Required model-free fixtures: query then answer; retrieve then answer; retrieve then query then answer; query then retrieve then answer; empty payload; identical payload already in facts; irrelevant response; unavailable first call; repeated consumed call; action budget exhaustion; terminal actions; reordered facts; cache collisions between equal facts/different flags; equal facts/different budget; sequential/batched equality. Assert exact next prompt bytes, flags, budget, charged calls and terminal state.

Add adversarial archive fixtures with a stale flag, mismatched prompt hash, wrong declared action count, duplicate step, missing step, incorrect terminal outcome and incomplete job inventory. They must fail verification without modifying original data.

### 4.3 Historical frozen intervention, if V8 weights are available

For all six original final S20 checkpoints, run legacy and corrected environments on the **same original 3,840 linked final cases per job**, exact weights and original renderer. Compare raw mode primarily; retain grammar/mask diagnostics under their own identities. No optimizer steps. Record initial-action equality: before any state diverges, identical deterministic inputs must give identical proposals. A difference there is an implementation/mapping problem, not an effect of consuming a tool later.

Measure per-world newly correct/newly incorrect, repeated calls, valid completion, post-tool exact answer and probability assigned to the recomputed correct action. Corrected trajectories can diverge after the first changed observation; do not try to splice their later actions into the old trajectory and call it neural reevaluation.

Additional fixed-weight renderer/payload interventions use separate records. They cannot be applied by merely editing the existing JSON scores. If old tensors are unavailable, emit `HISTORICAL_WEIGHTS_NOT_AVAILABLE`; preserve the original audit and continue the fresh core.

## 5. D1-H: harder tasks with interpretable difficulty

Keep historical D1 unchanged. New dataset identity: **`x1-d1h-v1`**. D1-H means a new harder curriculum, not a silent repair of old data. All architecture arms receive the same D1-H teacher-state schedule.

### 5.1 World semantics

Continue using public facts, a goal, optional query/memory responses and actions ANSWER(A–P), QUERY, RETRIEVE, ABSTAIN. The environment may compute semantic truth to create labels and score actions; the neural actor must receive only the public observation. No truth, answer paths, case IDs, scenario names, difficulty labels, gold spans, hidden memory or oracle-computed graphs in the actor input.

New world schema must distinguish:

```text
public_initial_facts; public_goal; public_tool_availability; action_budget
query_response; memory_response                # hidden until corresponding call
semantic_truth; provenance_rules; expected_valid_outcomes # evaluator/teacher only
base_world_id; split; family; transform; label_map         # bookkeeping only
```

Use directed relations with unique selected successors for answerable cases. Goals may require following a length-1, -2 or -3 relation sequence, then reading a terminal attribute. Define the depth convention explicitly: count relation transitions; attribute reading is separately recorded. Source/time precedence is declared in public task rules. An unresolved equal-authority conflict or missing required edge has no unique answer; label by the registered acquisition/abstention policy, not whichever fact happens to be last in a Python list.

Make stale-memory tasks observable: the public goal can require a specified evidence epoch or minimum source authority, and a retrieved fact exposes its epoch/source. For example, evidence at epoch1 is insufficient for a goal explicitly requiring epoch3; querying returns epoch3 evidence. Do not label a fact "stale" solely because the hidden world contains a newer value that the learner has no reason to seek. Conversely, if sufficient public evidence already meets the goal's declared validity rules, the teacher should answer even when more tools happen to be offered. Test this distinction with matched availability patterns.

Use A–P as the output code space and a public mapping when input aliases differ. Entity renaming and answer-code permutation are distinct transformations. The causal decoder's hidden states may contain contextual information; a slot is not assumed to equal one entity. Do not market slots as discovered entities without evidence.

### 5.2 Twelve balanced training families

Generate **512 unique base worlds per family = 6,144 training worlds**. Teacher-state count is derived, audited and frozen, not assumed to equal world count. Balance answer codes within answer-bearing strata; record any unavoidable remainder. Split before creating linked transformations.

| Family | Task | Main change relative to easy D1 |
|---|---|---|
| X01 | Direct evidence, one relation transition | Similar distractors, arbitrary row order and entity renaming |
| X02 | Two-hop relation composition | Competing branches; same entity appears in different roles |
| X03 | Three-hop composition | Reuse the learned relation operation three times |
| X04 | Source/time revision | Latest applicable authoritative fact may occur anywhere |
| X05 | Missing edge obtained by QUERY | Continue the computation using newly obtained evidence |
| X06 | Conflicting edge resolved by QUERY | Revise the answer implied by a plausible earlier fact |
| X07 | Missing edge obtained by RETRIEVE | Use relevant memory rather than blindly copy its last value |
| X08 | Stale memory followed by query correction | Retain useful context while replacing a contradicted relation |
| X09 | Two complementary missing facts | Both tools are necessary; vary which legal order is sufficient |
| X10 | Unanswerable, no useful tools | ABSTAIN without guessing |
| X11 | Already sufficient evidence, tools also offered | Avoid unnecessary tool use; availability must not determine action |
| X12 | Ambiguity or exhausted uninformative tools | Stop honestly after learning that no unique conclusion follows |

For X04–X09/X11, balance underlying path depths 1/2/3 as closely as integer quotas permit. X10/X12 contain matched answerable-looking controls; record semantic ambiguity separately from simple absence. X09 does not always imply a fixed tool order: when both orders work, score both as valid and report heuristic-teacher agreement separately. A deterministic teacher is a training convention, not the only correct policy.

**Non-anticipating teacher requirement — an additional design safeguard.** A teacher must not decide whether to query by peeking at the unrevealed payload. Before a tool call, two worlds with byte-identical public observations and availability must have the same teacher target/distribution, even if their hidden responses differ. Use this explicit default convention: answer if the currently public facts determine a unique conclusion; otherwise retrieve if memory is available; otherwise query if query is available; otherwise abstain. Recompute after each consumed tool. Source precedence belongs in the public rules. If a different convention is investigated, version it as a separate training intervention. An omniscient minimum-cost sequence is a privileged diagnostic upper bound, not a learnable target or the measure of initial teacher agreement.

Audit public-prefix collisions across the generated training states: inconsistent operation/value targets for identical visible inputs are a label-contract failure unless a probabilistic target distribution was explicitly designed and recorded. In particular, an empty/irrelevant future response cannot justify an early ABSTAIN label when an otherwise identical case receives QUERY. This protects the experiment from asking the model to predict information it cannot have. Grounding success is scored against what can be established through the allowed observations; a hidden answer alone does not make a correct-looking guess evidence-based reasoning.

Difficulty is a vector: hop count, public fact count, entity count, distractor count, branching, contradictory evidence, source priority, alias length and rendering. Use 5–14 entities and 0–6 irrelevant facts where framing fits. Do not force every cross-product cell to exist. Publish actual coverage and rejection counts by stratum. A comparison between deep/long/distracting examples and shallow/short/clean ones does not isolate depth.

Train on three registered renderer families: compact tuples, named relation statements and keyed records. Hold out a fourth independently implemented renderer. All renderers must express the same semantics, tools, budget and answer legend. No template contains a family label or target-correlated phrase outside the legitimate evidence. Keep a fixed original V8 renderer suite as a separate bridge; do not compare its scores directly with D1-H's aggregate.

### 5.3 Length, split and leakage rules

Use the existing UTF8 tokenizer with 260 symbols and the S20 context capacity of 640. **Maximum complete framed sequence = 640 tokens**, including public prefix, required control tokens and supervised action/EOS. The old V8 screen used 512; record the new limit in the X1 spec and apply it equally to every arm. Do not change tokenizer IDs or silently truncate evidence or labels. Reject and regenerate overlength worlds before split freeze; record the rejected difficulty/answer distribution. Evaluate every possible registered tool-observation state for length, not only the initial prompt.

Create structural identifiers invariant to renaming and row order. Keep linked views, counterfactual pairs, graph-isomorphic forms and their teacher states in the same split. Reserve relation-sequence combinations as well as world instances for compositional testing. Distinguish novel instances from held-out combinations, renderers and depths. Do not claim a held-out graph pattern if only its surface labels differ.

Training generation seed 8,100; development 8,200; confirmation 8,300; depth-stress 8,400. Derive substreams from `(dataset_schema, split, family, base_index, purpose)` using a versioned cryptographic digest, never Python's process-randomized `hash()`. These dataset seeds are independent of model initialization and order seeds. The owner/independent evaluator controls confirmation material until selection is frozen. If Gemini has inspected all confirmation outcomes, label the result development evidence and require fresh confirmation for promotion.

Use separate generation and semantic verification implementations. For small finite worlds enumerate all reachable valid tool/action paths and compare independently computed outcomes to teacher labels. Validate renderer round-trip semantics with parser/test code used only by the data audit; that parser must not feed graph structure to the learned actor.

Do not confuse agreement on a guessed final symbol with a justified terminal outcome. The independent verifier must check whether the available observations support that answer under the declared rules. Record lucky-but-unsupported guesses separately. Keep the historical score definition for historical reevaluation; this stronger X1 epistemic outcome is a new versioned endpoint.

### 5.4 Exact evaluation allocation

- Development: 32 base worlds ×12 families =384; six linked views each =2,304 episodes per final model.
- Milestone subset: a fixed stratified 8 of those bases per family =96; six views =576 episodes at updates 0/592/1,184/1,472. Final update 2,368 uses all development bases.
- Confirmation: 128 new bases ×12 families =1,536; six views =9,216 episodes per selected final model. Run only for a frozen candidate and its controls, in a separately admitted confirmation session.
- Depth stress: 64 bases for each depth 1/2/3/4/5 =320; canonical and relevant-counterfactual views =640 episodes per final model. Match total facts and length buckets across depths wherever feasible. Four/five-hop scores are extrapolation; failure does not prove a capacity ceiling.

The six views are canonical, entity/code permutation, irrelevant-fact intervention, row reordering, alternate rendering and a meaningful counterfactual. For an intrinsically unanswerable base, the last view may change answerability; record the applicable transition class. Counterfactual **both-correct answer** metrics include only declared pairs with two distinct determinate answers, with numerator and eligible denominator. Do not silently drop ineligible pairs from the all-family completion endpoint.

Primary mode is raw learned actions. Run the four historical diagnostic modes only on a fixed 96-base final subset. Grammar constraints and unavailable-action masks are diagnostics, not the primary capability claim. U's internal read iterations are neural computation, not extra external action opportunities.

## 6. Architectures: exact inheritance and comparators

Reuse `v5_model/core.py` and the qualified V5 foundation. Do not rebuild the transformer or introduce a new tokenizer during this screen. S20 is width512, 7 layers, 8 query heads, 2 KV heads, head dimension64, FFN1408, vocabulary260. Verify the actual spec and parameter inventory programmatically; reject mismatch. The existing backbone has 19,871,616 parameters.

| Arm | Neural mechanism | Comparison meaning |
|---|---|---|
| B0 | Original byte-autoregressive action generation | Historical interface reference; permits invalid strings |
| J | Dedicated 19-way action head | Joint typed interface vs B0; decoder/loss package also changes |
| F | Four-operation head and conditional 16-value head | Action/value factorization vs J |
| P | F plus learned copy distribution over visible A–P token positions | Conventional pointer evidence reader |
| F-cap | F plus active value residual MLP | Added-capacity control for P |
| R | P plus learned soft slots and query-conditioned cross-slot message passing | Relational-reader proposal |
| R-noedge | P plus the same slots, independent per-slot refinement | Cross-slot interaction control |
| P-capR | P plus an active dense query residual | Added-capacity control for R |

Existing J adds9,747 parameters; F adds10,260; P adds76,309; F-cap at width64 adds76,372, relative to the backbone. These are source-derived counts to recheck after implementation. R and its controls require exact new inventories, not guessed labels such as "20M equal size."

Typed joint probabilities must satisfy:

```text
p(ANSWER(a)|x) = p_op(ANSWER|x) * p_val(a|x)
p(QUERY|x), p(RETRIEVE|x), p(ABSTAIN|x) = corresponding operation probabilities
loss = -log p_op(gold_op) - 1[gold_op=ANSWER] log p_val(gold_value)
```

Average per supervised state, including non-answer states. Non-answer arguments contribute zero loss and no fake target. Inference chooses the maximum over the complete 19-way joint distribution; do not independently argmax operation and value, which implements a different policy. Keep J's exact algebraic factorization as a no-change control with probability/loss/gradient parity. B0 uses its existing byte-action likelihood and raw decoding; label B0→J as an interface package intervention rather than pure extra parameters.

P attends to **all eligible public-prefix A–P token positions in the same packed segment**, never a gold-derived span. Duplicate occurrences scatter-add into the same answer bin. Its learned copy/generative gate starts with bias−2; no candidates falls back to the generative value head without NaN. Never copy a hidden tool payload before that tool is called. Keep P's non-span path: answers absent from visible candidate positions remain representable through F, while tool/abstain probabilities remain in the operation head.

### 6.1 R: proposed relational reader, not an oracle graph

R is a concrete new model to implement, not a currently qualified component. It is inspired by learned slot aggregation and relational message passing; its utility in this byte-token setting is an open hypothesis.

For each action prefix let `H` be its causal token states `[L,512]`, `h` the final public-prefix state, and `M` a mask for public prefix tokens in the same packed example. Padding, another example and future answer tokens are excluded. Use float32 for attention softmax, normalization and losses.

1. **Token projection:** `x_i = W_x LN(H_i)`, dimension128. No gold token/edge/entity features.
2. **Eight learned initial slots:** a trainable `[8,128]` table, expanded across examples. Do not assign slot identities to gold entities. Two aggregation iterations, sharing parameters:
   - `a_ki = softmax_over_slots(k)(q(s_k)^T k(x_i)/sqrt(128))`, masked over valid tokens.
   - Renormalize each slot over its valid tokens: `w_ki = (a_ki + eps) / sum_valid_i(a_ki + eps)`, eps1e-8 on valid positions only.
   - `u_k = sum_i w_ki v(x_i)`.
   - `s_k = GRU_agg(u_k, s_k)` followed by `s_k += MLP_agg(LN(s_k))`, width128→256→128 with SiLU. Separate normalization parameters; share the block across the two iterations.
3. **Query-conditioned messages:** `q = W_goal h`, dimension128. For two shared message iterations, compute directed edge score `e_kl = MLP_edge([LN(s_k), LN(s_l), q])`, dimensions384→64→1 with SiLU. Mask `k=l`, normalize across sender `l`, compute `m_k = sum_l softmax(e_kl) W_msg s_l`, and update `s_k = GRU_rel(m_k,s_k)`. The two GRUs are distinct; each is shared only over its own iterations.
4. **Goal read:** `alpha_k = softmax_k(q^T LN(s_k)/sqrt(128))`; `r = sum_k alpha_k s_k`.
5. **Answer evidence:** modify only P's pointer query: `q_pointer = W_q h + W_r r`, dimension64. Keep P's candidate mask, keys, duplicate scatter, copy gate and F value/operation heads. Initialize `W_r` with the same small nonzero linear initialization convention as other heads. No initial zero gate that accidentally prevents all relational-path gradients.

The operation head does not receive a new explicit R input in this first experiment. Shared-backbone gradients can still change operation behavior, so do not claim its learned policy is mathematically unchanged. This separation isolates a useful first question: can a richer answer reader preserve tool selection while improving what happens after evidence arrives?

R has two message iterations, not a proven four/five-hop solver. The transformer and aggregation layers also contribute computation. Keep depth and iteration counts explicit; do not equate a slot pass with one semantic relation hop.

Required controls:

- **R-noedge:** identical aggregation and pointer path; replace cross-slot messages by an independently applied active per-slot MLP, retaining two shared refinement passes and a GRU update. Choose its MLP width to match R's added trainable parameters within1%; report measured work difference. No multiplication by zero, unused edge parameters or identity-only "control." A structural no-message intervention at fixed R weights is a separate lesion, not a trained capacity control.
- **P-capR:** add a dense residual to P's query through `h→hidden→64`, SiLU, with integer hidden width chosen to match the total new R parameters above P within1%. It must be on the actual forward/gradient path. Match parameters before looking at outcomes. If a single hidden width cannot meet tolerance, use a small additional active projection and publish the exact count; never add unused tensors.
- **Fixed-weight lesions:** set relational contribution to zero, permute edge destinations, and permute pointer value bins while preserving attention. Apply a sham identity operation with equal code path. Use them only as mechanistic sensitivity probes; off-manifold damage alone does not prove semantic relational reasoning.

Measure added parameters, active gradients, estimated operations and real median/p95 step time. Parameter matching is not compute matching. If a control is substantially faster, provide the learning-versus-work curves and an explicitly registered equal-work extension; do not mislabel the main equal-exposure comparison.

### 6.2 Initialization and contracts

Create a versioned X1 spec that includes aggregation dimension/count/iterations, message dimension/count, pointer dimension, normalization, precision, sequence limit, output mapping and schema. Existing V8 checkpoints retain their original specs. Reject loading a checkpoint into a structurally incompatible wrapper even if many tensor names match.

Seed the common backbone identically across every arm within seed. Initialize shared F heads and shared P projections from component-level seed streams that are independent of the complete arm name. In particular, P and R must start with byte-identical common parameters. New modules use separate deterministic component streams. Preserve the source convention of building the full model before optimizer construction. Save named parameter shapes, initial hashes, optimizer-group membership and first nonzero update checks. A fresh architecture run must not accidentally inherit a trained B0 backbone.

No predicted gold relations may be substituted by a deterministic parser in R. An oracle graph reader may be evaluated as an explicitly privileged upper bound, kept outside learned-model totals and disabled in production policies.

## 7. Training schedule and D2 objectives

### 7.1 Common architecture-screen training

Main seeds `(initialization, order)`: `(7,7), (19,19), (41,41)`. They are RNG keys, not inherently easy/hard tasks. Data splits are identical across these seeds. Save initialization, sampling and augmentation streams separately so later crossed-seed diagnostics can isolate them.

Use AdamW, peak LR1e-4, betas(.9,.95), epsilon1e-8, weight decay.1 with existing semantic no-decay grouping, gradient clip1.0, FP32 and the existing cosine schedule to .1 of peak. Preserve the exact source learning-rate function and report its formula/first/last values. Horizon **2,368 optimizer updates**, effective batch16 supervised states, microbatch2. No undocumented warmup, AMP switch or optimizer change. Accumulation is state-weighted; uneven batches must not receive equal microbatch weight by accident.

Use a deterministic, balanced/interleaved teacher-state schedule shared across arms. This is a new registered D1-H schedule, not a claim that interleaving won historically. Every effective batch draws from a frozen world/state ledger; balance family and trajectory position over a fixed cycle. Include terminal/post-tool states so long trajectories do not dominate by merely having more states. Save the selected state IDs and their order hash. Report world count, state count, actual presentations, unpadded/padded positions, answer-supervised positions, operations and wall time. Do not call this four epochs: the new corpus has a different number of states.

Save diagnostic observations at updates0/592/1,184/1,472/2,368. The1,472 point connects to historical D1 exposure under a declared different schedule; it does not reproduce old D1. Restore milestones from actual retained tensors when required, not their hashes alone.

Fresh frozen validation does not tune hyperparameters in place. If every arm fails to acquire even representative training controls, report the result and invoke a separately registered optimization diagnosis. Do not quietly raise LR, train longer only for preferred arms, or weaken outcomes to finish green.

### 7.2 D2 is a controlled objective experiment

D2 uses the same D1-H world semantics plus linked transformations. **Do not make D2 simultaneously harder than its CE comparator.** Both see identical paired data, minibatch order, labels, tokens, optimizer horizon and transformations. The six architecture/objective cells are P/R × {PAIR-CE, PAIR-EQ, PAIR-EQ-CF}; three seeds each =18 fresh runs.

Each effective batch contains four base-state groups with four views each =16 supervised states. A view group is original, a known entity/code permutation, an irrelevant/reordering transformation, and a relevant counterfactual. Use a stable schedule alternating irrelevant and row-order variants. Every view is supervised by its independently regenerated valid target. Within-group state pairing requires matching public information stage and tool availability; do not compare a pre-query state with a post-query state as if they should be invariant. Store a pair map and eligibility reason for each auxiliary term.

PAIR-CE trains on **all four views** using ordinary per-state CE. It is the augmentation control. PAIR-EQ adds a consistency loss. PAIR-EQ-CF adds a counterfactual answer-margin loss. Compare within this new dataset/schedule; main-screen P/R are context, not substitutes for PAIR-CE.

Let `p` be the typed joint19 distribution. For each meaning-preserving view pair `(x,T(x))`, with known answer-code permutation `Pi_T`, use:

```text
L_eq = mean_eligible_pairs JS(Pi_T p(.|x), p(.|T(x)))
JS(p,q) = .5 KL(p || (p+q)/2) + .5 KL(q || (p+q)/2)
L_PAIR_EQ = L_CE + 0.1 * L_eq
```

`Pi_T` permutes the16 ANSWER bins and fixes QUERY/RETRIEVE/ABSTAIN. Input-only alias changes with a fixed code legend use the identity output permutation. Use float32 log-space stable calculations and an epsilon only where mathematically needed; test exact identity/permutation cases. Gradients flow through both views. Do not enforce equality of individual latent slot indices: slots have no canonical entity ordering.

For a meaningful pair with two distinct determinate answer labels `y != y'` at matched ANSWER states, define conditional value margins:

```text
m(x)  = log p_val(y|x)  - log p_val(y'|x)
m(x') = log p_val(y'|x') - log p_val(y|x')
L_cf = mean_eligible_pairs .5 * (relu(0.5 - m(x)) + relu(0.5 - m(x')))
L_PAIR_EQ_CF = L_CE + 0.1 * L_eq + 0.1 * L_cf
```

CE already teaches all operation changes and new answers. The extra CF term is restricted to its declared domain; a relevant change can leave the correct operation QUERY or ABSTAIN unchanged. Do not force unrelated distributions apart merely because a hidden fact changed. Both-correct scoring is mandatory: always changing to a different wrong answer is not sensitivity competence.

Normalize auxiliary sums over eligible pairs, return a differentiable zero when none are eligible, and aggregate numerator/denominator across the entire effective batch. Since microbatch2 would split a four-view group, compute groups with a dedicated microbatch-of-one-group path or sequential view forwards whose outputs retain the necessary graph until group loss backward. Keep16 states/update and profile its actual memory. Do not detach logits to make it fit. If unsupported, reduce only groups-per-microbatch while retaining accumulation; never silently remove D2 terms.

These coefficient values are preregistered starting hypotheses, not optimized constants. Any coefficient sweep is a new module with its own budget. Slot/entity alignment probes are optional diagnostics trained on training data only; success of a probe is not proof that the policy uses its decoded information.

## 8. Coordinated campaign and launch order

### 8.1 Core session groups

One canonical X1 notebook and one registry enumerate these groups. Six jobs/group, three per arm, one independent process per GPU. Assign the same seed's paired arms to opposite GPUs when practical and alternate physical GPU assignment across seeds to reduce device confounding. Device0/1 identity is logged outside workers; a worker restricted to one GPU correctly sees logical `cuda:0`.

| Session ID | Fresh S20 jobs | Required data/objective |
|---|---|---|
| A0 | P ×3; R ×3 | D1-H, standard objective; first owner session |
| A1 | F ×3; F-cap ×3 | D1-H, standard objective |
| A2 | R-noedge ×3; P-capR ×3 | D1-H, standard objective |
| A3 | B0 ×3; J ×3 | D1-H, standard objective |
| D0 | P/R ×3 | Identical D2 pairs, PAIR-CE |
| D1 | P/R ×3 | Identical D2 pairs, PAIR-EQ |
| D2 | P/R ×3 | Identical D2 pairs, PAIR-EQ-CF |

Session IDs D0/D1/D2 here are **X1 session identifiers**, not historical D0/D1 dataset names; all job IDs must include `X1-` and explicit dataset/objective fields to prevent confusion. Prefer displaying `X1-PAIR-CE`, `X1-PAIR-EQ`, `X1-PAIR-EQ-CF` in the notebook UI.

**Default first launch is A0: P versus R.** This is a deliberate change from a baseline-first campaign that risks spending the owner's first session without testing the new mechanism. Repair the shared instrument, then put the conventional pointer and relational proposal into the first six-job run. Follow with the active controls; do not claim a relational mechanism from A0 alone. A0 can finish with informative evidence even if neither architecture reaches a capability threshold. B0/J remain required references, but their scientific success is not an admission requirement for A0.

All42 are scientifically useful even if B0 is weak. Safety/admission gates concern integrity, finite updates, device resources and timing. A failure to achieve80% on B0 does not cancel R or D2. A corrupted batch or incorrect label does block affected jobs until repaired. If a fix changes the shared experiment after earlier sessions run, version the protocol and repeat the necessary comparators; do not pool incompatible sessions as one clean comparison.

### 8.2 Optional diagnostic module: D1 exposure, not a prerequisite

If H01/H02 leave ambiguity worth resolving, register B0 on original D0/D1 at the same new2,368-update horizon, three seeds each, saving1,472 and2,368. Same initialization and order-stream policy across recipes; report token exposure separately. This yields a **recipe × exposure** comparison, not a clean single-data-property intervention. Original1,472-step D1 had a different scheduler horizon and must remain a separate historical result. If schedule horizon is the suspected mechanism, run the explicit2×2 horizon comparison with new job IDs rather than interpret intermediate checkpoints as identical to shorter completed training.

### 8.3 U: recurrence without conflating it with memory

Implement the registered extension even if B0 fails; run when its integrity canary and independent budget pass. Use the existing U design as a starting point, with the same public-prefix mask and typed F output. Compare U-1/U-2/U-4 with the same GRU/read parameters shared over iterations, using three seeds and the same D1-H schedule. Also train an untied two-step control, `U2-untied`, across the three seeds, for12 jobs. It has more parameters; state that fact. Pair work and capacity with measured curves instead of pretending these controls match everything simultaneously.

At fixed U-4 weights evaluate1/2/4/8 iterations on the depth-matched stress set, with latency and intermediate distributions. Eight iterations is extrapolation beyond training. A gain in accuracy per decision can cost more compute; report both. More recurrent steps are fixed engineered allocation, not learned adaptive thinking. A learned halt controller is a later intervention requiring matched fixed-budget controls and evidence that its decisions improve the accuracy/cost frontier.

### 8.4 Preservation and transfer: precise conditional extension

Only interpret retention after skill A reaches at least80% raw accuracy and70% counterfactual both-correct on a frozen development qualification set in all three seeds. These are proposed qualification thresholds, not current achievements. Choose one architecture using development data, freeze it and its training recipe before evaluating new confirmation worlds.

Define A as source-aware evidence binding, B as three-hop relation composition, and C as a previously untrained **composition of source revision with three-hop routing**, using disjoint relation combinations, worlds and templates. Do not train C combinations in A/B. For a source-revision operation genuinely learned in A, C tests reuse within this procedural domain; it does not demonstrate general real-world transfer.

Fork each qualified A checkpoint into two B continuations:1,184 updates with16 B states/update, no replay; and the same B examples/order plus4 stored A states/update, accumulated with a per-state mean over20 states. Extra replay work is explicitly charged. Add an equal-work B-only control with20 B states/update for the same1,184 updates. Do not claim equal-token or equal-compute retention from the16-versus20-state comparison. Evaluate A/B at0/296/592/1,184; save all four inference milestones.

For transfer to C compare useful A→B history, unrelated-history control of matched architecture/updates/tokens, and fresh initialization. Reset optimizer moments and scheduler for C in **all** conditions; preserve and report parameter initialization/history. Train the same C stream for at most1,184 updates, evaluate0/32/64/128/256/512/1,184, and measure work to80% accuracy plus70% CF-both-correct on two successive observations. Right-censor runs that never qualify; do not assign an invented threshold time. Report both incremental C learning cost and full lifetime learning cost. No transfer claim from a head start alone or from extra historical training.

This extension has a separate registry/budget; do not activate it automatically because a mean score looks encouraging. If A fails, retain the extension implementation and report `ACQUISITION_NOT_QUALIFIED`, not "retention failed."

## 9. Diagnosis that explains behavior

### 9.1 Required episode and training records

Each step records job/spec/source/checkpoint identity, base-world and view IDs, public observation hash, public tool flags, remaining budget, raw proposed action, executed action,19-way distribution when defined, correct-action set, teacher convention, observation appended, charged cost and terminal result. Keep public text for a fixed stratified trace set and enough deterministic data to reconstruct every scored observation. Hidden gold stays in the evaluator artifact, never the actor API.

Each update records supervised/auxiliary losses separately, eligible pair counts, LR, preclip gradient norm, clipping incidence, module gradient/update norms, actual state/token counts, sampler cursor, finite checks, step time and checkpoint events. Log at each update for scalar essentials; expensive per-module measurements at0/1/8/64 and registered milestones. Include representative train-state accuracy/value accuracy at every milestone, not only a tiny memorization canary.

Do not label a hidden mechanism from an outcome alone. Classify the first observable divergence as:

1. Correct initial action not selected despite sufficient/missing public information.
2. Relevant evidence not acquired because the tool choice is inappropriate or repeated.
3. Correct payload acquired but subsequent correct-answer probability/action remains low.
4. Correct operation but wrong value or malformed B0 output.
5. Correct new evidence contradicted by stale/irrelevant facts.
6. Previously correct skill deteriorates after later training (only with matched prior measurement).

Use language such as "post-observation answer failure" unless an intervention isolates evidence extraction, representation or decoding as the causal mechanism.

### 9.2 Intervention battery

Every final model receives the same selected intervention cases and budgets:

- Relevant returned-payload swap versus length-matched irrelevant swap.
- Same world with row order shuffled, alias permutation and familiar/unfamiliar renderer.
- Minimal sufficient evidence versus full distractor context; do not claim a pure distraction effect if length is uncontrolled.
- Evidence delivered via actual tool transition versus the byte-identical public prompt supplied directly. Equality is a useful statelessness check, not two independent causal treatments.
- Old fact plus correction in both orders, with explicit source/time precedence.
- Answer deliberately absent from the last fact; last-fact and majority-code deterministic baselines.
- Pointer disabled/value-bin permutation/sham for P and R; R message lesion/noedge/sham.
- Invalid/unavailable-action frequency, raw versus masked diagnosis, and grammar-only upper-bound analysis.

Only expose oracle graph or gold evidence spans in separately labeled diagnostic upper bounds. Never replace the model's actual query choice with oracle retrieval in the primary score. Preserve failed and ambiguous interventions. For every explanation state what was held fixed and what alternative explanation survives.

### 9.3 Primary endpoints and inference

Primary architecture endpoint: raw exact terminal success macro-averaged over the12 families, with equal family weights. Primary R mechanistic endpoint: canonical/relevant-CF both-correct on X02/X03/X05–X09 answer-determinate cases. Primary D2 endpoint: base-world all-correct across canonical, permutation and irrelevant views, plus CF-both-correct as a mandatory guardrail. Report simple per-view accuracies as well; conjunction scores alone obscure which operation fails.

For selecting a development candidate require mean gain at least5 percentage points on its registered primary endpoint versus its declared comparator, positive differences in at least2/3 seeds, no seed loss exceeding5pp, and no regression exceeding5pp on initial valid information-seeking decisions or post-tool completion. Preserve all numerical outcomes when a candidate misses a gate. The threshold is a research decision rule, not a mathematical proof.

For a provisional confirmed capability require new seeds101/211/307 on frozen candidate/control recipes, positive paired gains in all three, mean gain at least5pp, raw one-hop accuracy at least80%, relevant-CF both-correct at least70%, and correct-response preservation on irrelevant/renamed pairs at least90% **among initially correct cases**, with unconditional both-correct also reported. Require tool behavior guardrails above. A mechanism can show a promising gain below these absolute thresholds without earning a dependable-capability claim.

Use paired case comparisons and stratified bootstrap over independent base worlds, keeping all linked variants and all model-seed evaluations of a base together. Provide seed-specific estimates and a descriptive aggregate interval. Three seeds are too few for broad population claims; do not treat thousands of variants as independent model replications. Development selection is exploratory. Confirmation is a separate, frozen comparison, not repeated peeking until a small p-value appears. List every tested arm and contrast; any multiplicity-adjusted formal test must be preregistered separately.

Primary uncertainty intervals use10,000 fixed-seed bootstrap replicates on stored outcome arrays, not new inference. For small denominators show exact counts prominently. No mechanistic attribution solely from attention pictures, a linear probe, or an aggregate score.

### 9.4 Reports and charts

Produce one report that answers: what was acquired; what did not generalize; the first observable divergence; which controlled interventions changed it; remaining explanations; the smallest next discriminating experiment. Required charts: learning versus updates and tokens; learning versus measured GPU time; per-seed matched arm differences; accuracy by hop/length; initial-tool-choice→post-tool-answer funnel; paired correctness/change matrix; training-state versus held-out gap; component losses/update norms; optional retention/transfer curves when eligible.

Each chart embeds source/protocol/dataset IDs and counts in accompanying data. Never connect V6/V7/V8 accuracy points as one curve across different benchmarks. Publish `CAUSE UNRESOLVED` wherever evidence does not establish more. It is a valid and informative result if R fails, D2 only helps augmentation, or recurrence costs more without helping.

## 10. Kaggle execution, persistence and cost

### 10.1 Realistic budget

Historical V8 profiles reported about0.39 seconds/update for S20 on T4. At that rate2,368 updates cost15.4 worker-minutes;42 jobs cost about10.8 GPU worker-hours or5.4 hours at ideal two-GPU utilization **for training steps alone**. This is a lower-context historical extrapolation, not an X1 forecast.640-token inputs, R/D2, evaluation, checkpointing, source preparation and ZIP export can materially increase it. Do not promise the whole program in5–6 hours.

Each six-job session has an owner ceiling of330 minutes, with at least45 minutes reserved for safe stopping, verification/export and slow-host variability. Use measured component estimates, not a fixed training-only multiplier:

```text
predicted_session = setup + max(lane0_total, lane1_total)
                 + final_evaluation_not_already_charged
                 + export_and_verification + safety_reserve
```

Profile both physical GPUs with each architecture/loss path needed for that session: warmup, at least20 measured updates, representative long/context-heavy batches, batched rollout timing by family and checkpoint/export sizes. D2 must exercise eligible pair losses. Record median/p95 and use a conservative estimate. Canary/profile runs use separate scratch lineages and do not become research initializations.

If admission exceeds330 minutes or resource reserve, do not reduce registered seeds, updates, examples or difficulty. Partition the session into smaller job groups with unchanged job identities and exact resume. Save a revised scheduling receipt, not a revised scientific protocol. Both GPUs should do independent jobs; DDP is unnecessary for these models and would complicate recovery without a demonstrated gain.

### 10.2 Notebook release contract

Proposed notebook: `notebooks/ELENDRIL_V8_X1_KAGGLE_T4X2.ipynb`, generated by a deterministic builder. These are deliverables to create, not existing files.

Cells, in order:

1. Owner settings and visible summary: immutable40-hex source SHA, explicit session ID, optional checkpoint input path, output directory,330-minute cap, authorization flag. Default expensive execution disabled during development. Disabled state must display **NOT LAUNCHED** and the exact single setting to change; never silently claim success. An owner-armed release sets the flag true only following explicit owner authorization and includes the exact registered job list.
2. Source bootstrap: HTTPS fetch/archive of the exact commit, assert checked-out identity, import-path precedence, clean deterministic environment inventory. No moving `main`/`Elendril` installs or `pip install --upgrade torch` by default. Preserve Kaggle's CUDA-compatible torch unless verified requirements demand a change. Print concise dependency failures before campaign setup.
3. Data preparation: generate registered synthetic data from the pinned source automatically into writable work storage; verify manifests/structural leakage/length/semantics. **`/kaggle/input` is read-only mounted owner input.** Do not promise to download arbitrary files into it. Fresh core sessions require no manual dataset upload; historical/resume checkpoints require an actual mounted dataset or owner-provided accessible URL with expected SHA. Never invent a download URL for private local weights.
4. Two-T4 admission, CUDA allocation, finite forward/backward, exact-resume probe, parameter update checks, loss-path canaries, timing/disk forecast. Source/dependency failures stop with a precise receipt. Missing second GPU does not silently run a one-GPU protocol.
5. Execute the selected session's complete job registry with isolated workers, streamed concise progress, fail-safe checkpointing and periodic partial export.
6. Always finalize available evidence; verify archive inventory/hashes and required job/milestone/analysis coverage. Display downloadable ANALYSIS ZIP, FULL ZIP and FINAL_STATUS even after partial failure. Raise a clear error for unverified/missing required artifacts or failed jobs, retaining the original exception detail.

Check the rendered notebook text, parse/compile every code cell, execute its orchestration against CPU fake-worker fixtures, and rebuild byte-identically from the immutable source archive. Publish runtime implementation commit A first, then generate/publish notebook commit B that pins A. Do not attempt a self-referential notebook hash/commit. A must contain all runtime modules and the builder. Record A, B, notebook SHA and deterministic-build command in the handoff. A fresh archive checkout must pass the import and fixture tests; local editable imports are not evidence of reproducibility.

Kaggle packaging/session limits and installed Python/torch change. Before release, consult official Kaggle docs/runtime repository and PyTorch docs; record tested versions. CPU fixture success is **CPU-qualified, GPU admission pending**. Real owner canaries are the GPU evidence. Do not promise zero possible runtime failures; implement early detection, clear receipts and recovery.

### 10.3 Checkpoint retention and disk admission

The previous source solved disk exhaustion by deleting intermediate tensors. X1 must retain the milestones needed for its promised analysis without exceeding workspace limits.

- Keep an atomic full-resume latest and previous generation per active job: model, optimizer, scheduler, all RNG streams, sampler/view cursor, update number, loss/precision settings, source/spec/data hashes and parent lineage.
- Keep inference-only weights at updates0/592/1,184/1,472/2,368 for each job. Store common initial backbone once per seed with explicit reconstruction references and per-arm initial head weights, or store complete initial models if disk allows. Never deduplicate by filename alone.
- Keep full final-resume states for all completed jobs until exported and verified. Intermediate inference snapshots cannot resume optimization; label this limit.
- At every64 updates, save atomically and verify before rotating. Do not prune registered inference milestones or the last independently verified recovery generation. A failed save must leave the previous generation valid.
- Estimate raw files + staged ZIP + previous ZIP + manifests/temp files + at least2 GiB reserve. `df` free space alone may not reflect a platform output quota. Inspect mounts and applicable quota; `/tmp` is not automatically a separate or durable disk.
- Use at most six jobs per default session and split further if measured storage requires it. Do not delete completed evidence to create a nominally successful larger session.

An S20 FP32 tensor set is roughly80 MB; Adam moments and other resume components make a full state several times larger. Calculate actual serialized bytes before scheduling. Streaming ZIP avoids building an extra uncompressed tree but still needs space for the ZIP alongside source files. Export is synchronous with a stable snapshot; no two writers can advance the same lineage or archive.

Save durable outputs under `/kaggle/working` before the kernel ends. `/tmp` may be scratch, never the only copy of required results. Session-local existence is not external custody. After Save Version/Run All, the owner retrieves outputs; verify downloaded SHA independently. Cloud uploads require configured credentials/authorization, not assumptions.

### 10.4 Interruption and resume

Handle graceful deadline/termination by stopping at a safe update boundary, writing full state, closing logs and exporting a `PARTIAL` receipt with next jobs/cursors. Abrupt process death can occur without a final handler; recovery must find the last complete manifest without trusting staging directories. Never mark a partially written checkpoint as latest.

Resume verifies source/spec/data/loss/optimizer/schedule identities before load. Retain the recorded global schedule horizon and augmentation cursor. An interrupted data pair cannot be resumed as two unrelated examples. Restore both GPU/CPU RNG and isolated generators. On the same supported environment, compare uninterrupted versus save/restart trajectories for weights, optimizer, scheduler, cursor and next batch. Across hardware/library changes, do not promise bitwise equivalence; record a migration or reject exact-resume claims.

Full archive extraction must reject path traversal, absolute paths, symlinks, duplicate names, decompression-limit violations and inventory/hash mismatches before model loading. Do not unpickle arbitrary owner artifacts merely to inspect them. Use existing safe formats and bounded load paths; verify component versions and tensor shapes.

## 11. Concrete implementation work for Gemini

The filenames below are **proposed new deliverables**; existing code is listed separately. You may use a different internal arrangement only if the public contracts, registry and tests remain equivalent and the handoff explains it. Avoid a sprawling new framework.

### Existing components to reuse selectively

```text
v5_model/core.py                         backbone and packing contracts
v5_training/                            optimizer, schedule, checkpoint primitives
anra_v5/elendril/representation.py       UTF8 identity
anra_v5/elendril/v8/contracts.py          public/action/model conventions
anra_v5/elendril/v8/models.py             J/F/P/F-cap/U and algebraic null
anra_v5/elendril/v8/training.py           typed training integration
anra_v5/elendril/v8/evaluation.py         rollout/batched scoring, with repairs
anra_v5/elendril/v8/teacher.py            historical teacher, preserved
anra_v5/elendril/v8/verifier.py           independent outcome foundation
tools/elendril_v8_worker.py              isolated worker and diagnostics
tools/elendril_v8_campaign.py            scheduling pattern
tools/elendril_v8_export.py              verified export foundation
tools/build_elendril_v8_notebook.py      deterministic notebook pattern
```

### New bounded implementation surface

```text
anra_v5/elendril/v8x/contracts.py         X1 specs, world/view/loss/run identities
anra_v5/elendril/v8x/data.py              D1-H/D2 generation, pairing, split audit
anra_v5/elendril/v8x/models.py            R, R-noedge, P-capR; reused typed heads
anra_v5/elendril/v8x/objectives.py        CE/EQ/CF with eligible denominators
anra_v5/elendril/v8x/protocol.py          immutable jobs, milestones, budgets/gates
anra_v5/elendril/v8x/training.py          grouped views, resume cursor, model factory
anra_v5/elendril/v8x/evaluation.py        corrected state adapter and diagnostics
anra_v5/elendril/v8x/analysis.py          paired endpoints, uncertainty, decisions
tools/elendril_v8x_worker.py             isolated canary/profile/train/frozen worker
tools/elendril_v8x_campaign.py           session scheduling, deadline/resume
tools/elendril_v8x_export.py             analysis/full verification and final status
tools/build_elendril_v8x_notebook.py     reproducible notebook generation/check
tools/analyze_elendril_v8x_results.py    offline intake and curated reports
tests/test_elendril_v8x_*.py             bounded tests listed below
notebooks/ELENDRIL_V8_X1_KAGGLE_T4X2.ipynb
research/v8-x/IMPLEMENTATION_HANDOFF.md
```

Do not copy every V8 file into `v8x`. Reuse stable functions, isolate versioned behavior, and keep old source reproducible. If repair belongs in a common primitive, regression-test legacy behavior through the explicit historical adapter.

### CLI contracts to implement

These commands are acceptance targets, **not commands claimed to exist today**. Implement argument parsing/`--help` and fixture tests before documenting them as runnable:

```bash
python -m tools.elendril_v8x_campaign prepare --source <40hex> --session A0 --work-dir <dir>
python -m tools.elendril_v8x_campaign validate --source <40hex> --session A0 --work-dir <dir> --cpu-fixtures
python -m tools.elendril_v8x_campaign admit --source <40hex> --session A0 --work-dir <dir> --devices 0,1 --max-minutes 330
python -m tools.elendril_v8x_campaign run --source <40hex> --session A0 --work-dir <dir> --devices 0,1 --max-minutes 330 --owner-authorized
python -m tools.elendril_v8x_campaign resume --source <40hex> --session A0 --work-dir <dir> --devices 0,1 --max-minutes 330 --owner-authorized
python -m tools.elendril_v8x_export --work-dir <dir> --session A0 --output-dir <exports>
python -m tools.analyze_elendril_v8x_results --archive <analysis.zip> --output <curated-dir>
python -m tools.build_elendril_v8x_notebook --source <40hex> --output <notebook.ipynb>
python -m tools.build_elendril_v8x_notebook --source <40hex> --output <notebook.ipynb> --check
```

`prepare` performs no research training. `validate --cpu-fixtures` must not instantiate/train full S20 jobs. `admit` runs only bounded owner-cloud canaries/profiles. `run/resume` requires explicit authorization and a matching admission receipt. Authorization alone cannot bypass failed integrity checks. If disabled/unadmitted, return a clear non-success status, not an empty completed campaign.

### Commit-sized execution sequence

1. **Instrument repair:** reproduce stale transition/cache defects, fix defaults, preserve legacy adapter, independent replay tests. Commit with failing-before/passing-after fixture evidence.
2. **Data and registry:** implement D1-H semantics, transforms, hard-depth matching, split audits and the42 core job records. Freeze hashes and proposed endpoints. No cloud training yet.
3. **Mechanisms and loss:** integrate R/controls and grouped D2 loss; parameter/gradient/mask/algebraic tests. Preserve B0/J/F/P behavior at existing identities.
4. **Training/recovery:** exact grouped sampler state, checkpoints/milestones, update accounting, representative CPU microfixtures and interrupted worker tests.
5. **Evaluation/analysis:** full paired outcome ledger, first-divergence taxonomy, source-bound offline replay, bootstrap/charts, strict no-promotion defaults.
6. **Cloud orchestration:** two independent GPU workers, timing/disk admission, partial outputs, verified final exports, source archive/notebook checks.
7. **Publish runtime A then notebook B:** meaningful reversible commits to Elendril; no force push; append dated ≤5-line Codex/Gemini progress entries. Provide exact source, notebook and dependency receipts.
8. **Owner qualification and research:** run selected notebook sessions; retrieve verified outputs; archive only focused MD/essential JSON in Git; analyze before selecting extensions/confirmation.

Do not stop after writing an implementation roadmap. Gemini's assignment is to implement these contracts and deliver an operational notebook, with honest pending-GPU status. Codex's current assignment is this handoff, not its execution.

## 12. Verification requirements without wasteful local campaigns

Use short deterministic CPU fixtures and existing relevant tests. Do not repeatedly run the whole repository after documentation edits or run S20 research locally. Tiny models verify mechanics only and cannot qualify the scientific hypothesis.

| Area | Necessary acceptance evidence |
|---|---|
| Transition | All cases in section4; cache/trace/independent replay parity; original stale fixture rejected |
| Data | Semantic audit, deterministic regeneration, no split leakage, all tool states fit640, equal paired labels/maps, public-prefix target consistency, non-anticipating teacher, answer/length/difficulty distributions |
| Head contracts | Joint probabilities sum1; nonanswer loss/value-gradient semantics; J algebraic null; duplicate pointer bins; no candidates; packed-prefix exclusion |
| R/controls | Exact parameter inventory; all intended paths trainable; slot/message shapes; mask permutation behavior; sham identity; no gold inputs |
| D2 | Hand-computable equivariance/permutation losses; no eligible pairs; CF label changes; full-batch vs grouped accumulation gradient parity; linked views never split incorrectly |
| Optimization | Shared initial tensors, complete optimizer membership, finite parameter updates, clipping/schedule receipts, state-weighted accumulation |
| Recovery | Crash before/after manifest advancement; corrupt latest falls back to verified previous; complete restoration of model/optimizer/scheduler/RNG/pair cursor; incompatible specs rejected |
| Orchestration | Missing/one GPU, worker failure, time deadline, signal, insufficient disk, partial result, completed jobs with missing export, duplicate writer, source mismatch |
| Archives | ZIP inventory/hash verification, traversal/decompression bounds, missing mandatory milestone, tampered report, independently retrieved-file verification |
| Notebook | All code cells compile; fake-worker Run All path; disabled path visibly NOT LAUNCHED; exact source archive imports; builder byte parity; no placeholders in release |

Do not weaken strict checks to make a fixture pass. If nondeterministic GPU operations are encountered, resolve or explicitly version that numerical mode; do not quietly disable deterministic requirements. PyTorch does not promise cross-release/device reproducibility, so exact-recovery evidence is environment-specific. Use the official reproducibility guidance and store its relevant configuration.

Cloud canaries separately demonstrate actual T4 allocation, all active module gradients, bounded tiny-cohort acquisition where appropriate, exact save/restart continuation and realistic evaluation/export timing. A poor research baseline does not block a novel arm. A broken numerical path does. Do not throw away a valid failed scientific run because a target accuracy was not reached.

## 13. Evidence artifacts and final acceptance

Per-session output structure:

```text
identity.json                         source/protocol/data/session/run IDs
environment.json                      versions, GPU UUIDs, device mapping
admission.json                        time/memory/disk and bounded canaries
registry.json                         expected jobs/milestones/endpoints
data/manifest.json                    generator/render/split/transform identities
jobs/<job-id>/updates.jsonl
jobs/<job-id>/milestones/<update>/     scores, traces, diagnostics, weights refs
checkpoints/<job-id>/                 full-resume latest/previous/final
analysis/comparisons.json             paired differences and denominators
analysis/diagnosis.json               observed/supported/unresolved distinction
analysis/RESULTS.md
charts/                              figures plus numerical source tables
exports/ELENDRIL_X1_<session>_ANALYSIS.zip
exports/ELENDRIL_X1_<session>_FULL.zip
exports/FINAL_STATUS.json
```

ANALYSIS contains enough registered world/view data, observed actions, scalar training trajectories, identities and receipts for independent recorded-action replay. FULL additionally carries the checkpoint components and inference milestones promised above. Derived archives do not recursively contain themselves. Verify every declared member and required semantic coverage before final success.

Keep root final receipts outside a self-referential checksum cycle: the bundle manifest hashes payload members; a separate delivery receipt hashes the completed ZIPs and manifest. If an archive contains an earlier execution receipt, label it as such rather than rewriting it after hashing. Final delivery requires both expected ZIPs and matching hashes, not merely six completed jobs. Session delivery is not completion of the42-job program; the program aggregator checks every required session/job identity.

The curated Git destination is `Experiment Result/V8-X1/<campaign-identity>/RESULTS.md` plus essential `METRICS.json` and `PROVENANCE.json`. Keep owner ZIPs, giant traces and checkpoint tensors out of Git. Do not modify V6/V7/V8 historical results. The concise report must include exact commits, completed/failed/missing jobs, measured costs, observed outcomes, controlled interpretation, unresolved causes and next execution/scientific decision.

Implementation handoff acceptance:

- Corrected transitions and independent audit are tested, with legacy evidence preserved.
- All42 core jobs are real registry entries using S20 and the fixed protocol, not prose-only experiments.
- R, active controls and D2 objectives are integrated into the actual train/evaluate/recover path.
- Harder data and withheld combinations are semantically audited, and output/support limits are explicit.
- Notebook reproduces from an immutable source; local CPU/CI results and pending/observed GPU evidence are distinguished.
- Budget admission can partition execution without changing experimental work or discarding negative findings.
- Results export and restoration have a tested path after normal completion and interruption.
- No claims of AGI, solved cognition or recursive self-improvement appear without evidence.

## 14. How to decide what comes after X1

| Measured outcome | Interpretation to test | Next useful intervention |
|---|---|---|
| Fixed environment recovers post-tool success | State-contract error materially contributed | Preserve corrected evaluator; pursue remaining first-step/transfer failures |
| J/F improve syntax and training fit, little held-out binding | Interface helps acquisition but not robust evidence use | P/R and paired-world controls remain central |
| P beats F and F-cap, but R adds nothing | Conventional evidence selection sufficient at this difficulty | Test longer compositions/distractors before increasing architecture complexity |
| R beats P, R-noedge and P-capR on changed-edge/depth cases | Evidence supports benefit from trained cross-slot interaction in this domain | Independent replication; then iteration/slot-count and transfer tests |
| R and P-capR improve equally | Added active capacity/optimization remains a competing cause | Do not promote relational mechanism; use equal-work follow-up |
| R-noedge matches R | Slot aggregation may matter more than relational messages | Test slot capacity and aggregation depth with matched controls |
| D2 paired CE explains all gains | Augmented supervision helped; extra losses not needed | Keep simpler training; investigate a truly new shift |
| EQ improves stability but predictions stay wrong | Consistency is not grounded competence | Correct-label/CF learning, input sufficiency and optimization diagnosis |
| CF loss improves both-correct and transfers | Sensitivity objective supports useful functional structure | Replicate on independently generated transformations and new relation combinations |
| U improves deep tasks at acceptable work | Recurrent computation useful in this tested regime | Compare adaptive allocation to strongest fixed budget, not just more iterations |
| Every model fits training but fails new names/renderers | Generalization remains bottleneck | Controlled representation/tokenization study with migration-safe identities |
| Every model fails representative training acquisition | Measurement/data/optimization not yet sufficient | Exact gradients/labels, short acquisition trajectories, preregistered LR/horizon controls |
| Qualified skills survive B and accelerate C | Domain-specific cumulative learning evidence | Broaden task semantics, evaluate lifetime cost, then scale the validated mechanism |

Scaling toward larger models remains a valid future direction. X1 does not require pretending20M is enough for broad language competence. It asks whether the proposed mechanism yields reliable, reusable operations under controlled conditions and whether the learning process actually uses it. A useful positive result is a capability that survives unfamiliar names, changed evidence and composition. A useful negative result identifies which intervention did not help and eliminates an attractive but unsupported explanation.

## 15. References and design provenance

The source-specific findings above come from the inspected repository and owner archives, not external literature. Terra's review sharpened the P-versus-R D2 comparator, active controls, slot-permutation caution and compute accounting. The following primary sources motivate parts of the design without establishing that they work in ELENDRIL:

- [Object-Centric Learning with Slot Attention](https://arxiv.org/abs/2006.15055): learned iterative slot aggregation. R is a proposed adaptation to causal token states, not a reproduction or a proven object-discovery system.
- [Universal Transformers](https://arxiv.org/abs/1807.03819): shared recurrent computation as an architectural hypothesis. X1 U is a bounded reader refinement, not a full implementation of that architecture.
- [PyTorch reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html): environment-specific reproducibility, RNG and deterministic-operation requirements.
- [Official Kaggle runtime repository](https://github.com/Kaggle/docker-python) and [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks): release engineer must verify the current owner runtime and platform requirements before notebook qualification.

**Instruction to Gemini:** implement this as one coherent, testable extension of the existing learner. Preserve provenance, repair the real defect, and make the architectural experiment happen. When a specified mechanism cannot be implemented or a comparison cannot be matched, report the precise limitation and version the design; do not silently replace it with an easier experiment. Deliver working code and the owner-run notebook, while leaving every scientific conclusion contingent on the resulting evidence.
