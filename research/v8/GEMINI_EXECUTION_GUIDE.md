# ELENDRIL V8 — mechanism discovery and cumulative learning
## Independent conclusions and executable handoff for a new engineering agent

**Author:** Codex. **Date:** 2026-11-14, Asia/Calcutta.
**Deliverable:** research design and implementation instructions only. No V8 model, experiment, notebook or scientific result has been implemented or executed by writing this document.
**Audience:** Gemini or another agent starting without the previous conversation.
**Repository:** https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI
**Verified Elendril HEAD:** `0c9cb284d2c8c949e17ac342959c6f2bf351bd30` (remote checked while preparing this handoff).
**Scientific parent source:** `724af4cb508173e0eb3f4397d5dcb4a42a66712f`, the source actually used for V7.17 training.

Read this entire document before changing code. Sections marked **BUILD FIRST** are a concrete implementation contract. Conditional stages are specified follow-ups, not permission to launch an unlimited campaign. A future implementing agent should implement and validate the first stage, deliver an owner-operated notebook, and continue independent engineering while GPU results are pending. Only the owner launches research training. The present request authorizes writing this guide, not executing its instructions now.

## 1. Independent conclusion: what should change, and why

**Build an experiment that separates learning the task, selecting an action, and computing the answer. Keep the S20 neural backbone, introduce small independently switchable changes, and compare them against equally well-trained controls.** Do not assemble all eight proposed mechanisms into one architecture before any component has earned its cost.

V7.17 supplies an unusually useful clue: an unchanged S20 network improved substantially when peak LR fell from 3e-4 to 1e-4. Yet it failed all tested novel one-hop chains. The next question is not simply whether seven transformer layers can perform three sequential lookups. It is whether the learner recognizes an unfamiliar request, grounds the available operations, and selects a value from public evidence at all.

My three main disagreements with the supplied suggestions are:

1. **The proposed S3-first tournament conflicts with the owner's S20 minimum and would create another scale interpretation problem.** Use S20 for research. Tiny CPU models are allowed only as mechanical fixtures. S8 is the historical 7.97M profile; no S9 exists in the inspected registry. Do not invent S9 or substitute it silently. S3/S8 results remain historical comparators, not new default research runs.
2. **“Architecture first” is insufficiently controlled if architecture, output interface and training distribution all change together.** A pointer head can appear successful because answers were converted into convenient slots; a policy head can appear successful because invalid outputs were removed. Introduce joint-head, grammar, data-only, and compute controls explicitly. A better data/objective baseline winning is a useful result, not a failure to innovate.
3. **Formation, retention and transfer cannot be interpreted simultaneously before their prerequisites exist.** A Guardian protecting an unacquired task is meaningless. Recurrent reasoning is not justified by a three-hop failure when a novel one-hop task also fails. Specify the entire cumulative program now, but admit its stages separately.

My additional candidate is **action/answer factorization**, testing competition in the current output distribution. It has no pointer, recurrent workspace, privileged task labels, availability features or auxiliary oracle loss. It separates the probability of ANSWER from the conditional distribution over answer values, and compares that architecture with a matched joint categorical head. This is a new ELENDRIL hypothesis, not a claim of a new invention in machine learning. It is distinct from simply supplying availability bits: it changes how the learner allocates probability and gradients even when the public input is identical.

I rank the initial rivals as follows:

| Rival | Hypothesis | What would count against it |
|---|---|---|
| Strong plain learner | Coverage/optimization explains much of the failure | Properly balanced data still fails while a matched structural intervention improves |
| Factorized action/answer head | Control actions and values interfere in the shared readout | Only output restriction helps; factorized head does not beat the matched joint head |
| Conventional pointer reader | Reading a relevant public value is the bottleneck | Gains vanish under row permutations, unavailable answers, or matched extra-parameter controls |
| Shared recurrent refinement, conditional | Additional learned computation helps composition | Gains are explained by extra inference work or do not exceed a nonrecurrent equal-cost competitor |

Availability masks are engineered controls, not a fifth learned architecture. Fast memory, invariance losses, adapters and Guardian are later branches of the same learner, not mandatory first-release modules.

The first desired achievement is a reproducible **raw learned-policy improvement on independently generated one-hop and counterfactual tasks**, with an attributable intervention and controlled cost. No plan can guarantee that outcome or eliminate every implementation error. This guide instead makes failures observable and limits the consequences of mistakes.

## 2. Scientific starting point the implementing agent must preserve

V7.17 optimization ran eight S20 jobs: baseline seeds 7/19/41; lower-LR counterparts; and two crossed initialization/order jobs. Each used four epochs, 2368 updates, batch16/micro2 and fresh weights.

| Initialization/order | Peak LR | Base /112 | Confirmation /112 |
|---|---:|---:|---:|
| 7/7 | 3e-4 | 46 | 54 |
| 19/19 | 3e-4 | 80 | 79 |
| 41/41 | 3e-4 | 75 | 78 |
| 7/7 | 1e-4 | 91 | 88 |
| 19/19 | 1e-4 | 92 | 99 |
| 41/41 | 1e-4 | 90 | 97 |
| 7/19 | 3e-4 | 57 | 56 |
| 19/7 | 3e-4 | 45 | 47 |

Lower LR increased mean confirmation 62.80%→84.52% and full-training exact-action accuracy 63.43%→83.94%. Lower-LR novel one-hop, two-hop and three-hop raw results were each 0/48 across three seeds. In 46/48 three-hop cases the first action was RETRIEVE despite Memory=0; two were ABSTAIN. Alias accuracy regressed from 55/336 to 40/336. This is demonstrated recipe sensitivity with unresolved remaining mechanisms, not a demonstrated capacity ceiling.

The three baseline runs reproduce V7.16 compact raw metrics across all 16 cohorts. The audited analysis archive contained 44 independently replayed evaluation reports and 32 audited diagnostic milestones. Recorded-action replay does not regenerate neural probabilities or restore model tensors. Separate final delivery and FULL archive were not supplied in that audit. Do not claim to have old weights from an analysis ZIP.

Exact historical files to read at the verified parent commit:

* `Experiment Result/V7.17/f443d3f4141b4163bbc130d46ebd7395/RESULTS.md` and `METRICS.json`.
* `research/v7.17/README.md`, `ENGINEERING_AUDIT.json`.
* `anra_v5/elendril/v717_protocol.py`, `v717_training.py`, `v717_measurement.py`.
* `tools/elendril_v717_campaign.py`, `elendril_v717_worker.py`, `elendril_v717_report.py`.
* `tools/analyze_elendril_v717_results.py`: intentionally bound to one exact owner archive; do not generalize it by removing identity checks.

Owner archive SHA256: `10d024a5daa64b24a63e549b6c8a78c31c8eff141772ec84efee4b5a9f02e1fc`.
Owner run: `f443d3f4141b4163bbc130d46ebd7395`.
Training-data identity: `7786244187963f75e6cda5b10e4d4e0d81927b7000154126fff538dea15539e5`.
The familiar corpus has 7335 worlds, 9472 supervised states and about 2.2 MB of prompt/target text. It is not natural-language pretraining.

The previously planned representation-training phase is **NOT_RUN**. Do not relabel it as completed or replace its registered 3e-4 recipe with 1e-4 under the same identity. V8 gets new identities.

## 3. Repository reconnaissance and boundaries — BUILD FIRST

Local inspected worktree: `C:/Users/ankit/Downloads/An-Ra-Elendril-V7`, local branch `codex/elendril-v7`, upstream `origin/Elendril`. Another agent must discover its own path; do not assume these Windows paths exist elsewhere.

Run these read-only commands first:

```text
git status --short --branch
git rev-parse HEAD
git fetch origin Elendril
git log -5 --oneline origin/Elendril
git diff --stat
```

Read AGENTS.md before editing. Preserve dirty work. If remote HEAD differs, inspect the intervening changes; record the actual base and revise path references. Do not reset to this document's commit. For future implementation, create a reversible branch `codex/v8-mechanism-lab` from the reviewed Elendril tip, or a distinct worktree if dirty work exists. A colliding branch must be inspected, not overwritten. Do not merge donor branches wholesale. A completed implementation can be proposed for Elendril after review; no force push.

Existing code responsibilities:

| Existing file | Responsibility / integration constraint |
|---|---|
| `v5_model/core.py` | `initialize(spec, seed)`, causal `forward_hidden`, tied embedding output; currently asserts the exact base parameter receipt |
| `v5_contracts/model_spec.py` | ModelSpec and parameter identities; architecture changes require a new versioned specification |
| `anra_v5/elendril/backend.py` | Validates packed masks and generation identity; no task truth or answer repair |
| `foundation_training.py` | Instantiates model before optimizer, validates parameter ownership and gradients, performs update transactions and checkpoint binding |
| `v7_training.py`, `v717_training.py` | Deterministic epoch permutations and separate initialization/order seeds |
| `v6_data.py` | Existing public cases, renderer, supervised trajectories and UTF8 framing; preserve historical generator |
| `v6_evaluation.py`, `v71_evaluation.py` | Public actor loop and outcome replay; do not supply their truth objects to the actor |
| `v5_training/checkpoint.py` | Content-addressed model/optimizer/RNG/cursor state, committed generations, milestones |
| `tools/elendril_v71_export.py` | Dual ZIP integrity and restoration; old phase/protocol allowlists are not generic V8 support |

**Important integration trap:** the trainer constructs its model, optimizer, tensor inventory and checkpoint identities in that order. Attaching new parameters afterward silently excludes them from optimization/identity. Introduce an explicit validated model factory before optimizer construction, with the unchanged historical factory as default. Likewise, typed policies cannot be squeezed into byte logits and called equivalent. Define an explicit action-policy adapter with its own identity and loss contract.

No local research training, including S20 canaries. CPU tests must use bounded miniature fixtures and a few updates. Owner-run Kaggle 2×T4 is the target. A local directory search during planning did not establish custody of a V7.17 FULL ZIP; unrelated files with “full” in their names are not substitutes.

## 4. Common actor, observation and truth contract — BUILD FIRST

Retain one cognitive loop:

```text
public observation + public tool availability + bounded recalled memory
    → neural policy → proposed action → environment validity check
    → real public observation/outcome → next policy decision
```

Create immutable types under a new `anra_v5/elendril/v8/` package:

```python
PublicInput(text_utf8, query_available: bool, memory_available: bool,
            action_budget: int, observation_version: str)
PolicyProposal(operation, argument, joint_log_probability,
               raw_output, model_calls, evaluated_positions, mode, model_identity)
EvaluationOutcome(correct, terminal_status, charged_cost, verifier_identity)
```

Operation vocabulary for the first architecture screen is ANSWER, QUERY, RETRIEVE, ABSTAIN. ANSWER has one argument in A–P. Other arguments are fixed by the current single-channel environment. The byte baseline still exposes all native outputs, including malformed output and unsupported THINK; those count as invalid. Do not silently add THINK training. A later recurrent reader performs internal computation, not an externally scripted deliberation action.

No actor input may contain case ID, world ID, task family, split, expected answer, required operation, number of necessary hops supplied by an evaluator, or hidden channel payload. A hop count stated in the user's public question is allowed. Explicit public query/memory availability bits are allowed only in arms that declare them; the raw-text control receives their textual rendering.

Teachers and verifiers are separate modules. Teacher state labels may train a model; they may not become runtime features. At evaluation the actor receives only PublicInput, never the case object or a callback that can inspect hidden truth. Returned tool observations come from the environment, not a label lookup inside the model.

For availability-mask evaluation, mask QUERY when query_available=False and RETRIEVE when memory_available=False. Never mask answer values according to truth, filter to values present only in a gold route, or force ANSWER merely because the evaluator says information is sufficient. ABSTAIN remains possible. Use the same public availability source for all arms.

Always log the unconstrained proposal and the actually executed proposal separately. A raw invalid action remains a raw failure even if a masked condition later succeeds. Direct-answer-only evaluation changes the task and is a diagnostic upper bound, not an autonomous policy score.

## 5. Architectural contracts — BUILD FIRST for B0, J, F and P

### 5.1 Shared backbone and profiles

Start with the existing S20: UTF8 vocabulary260, width512, seven layers, eight query heads, two KV heads, head dimension64, FFN1408, context640. The base inventory is 19,871,616 parameters. Preserve its tokenizer identity and tied embedding. Use raw UTF8 inputs with maximum framed sequence length512 for the first screen. Longer contexts and tokenizers are separate future interventions.

All main arms start from identical backbone tensor values for a given initialization seed. Initialize each head in a separate RNG context so constructing it cannot alter backbone weights. Save hashes before training. Use `init_seed` and `order_seed` as distinct fields; main paired runs use (7,7), (19,19), (41,41), without claiming those settings are universally favorable. Extra head RNG seed is a deterministic hash of architecture ID and init_seed, recorded in the manifest.

### 5.2 B0: unchanged byte policy, strongest current recipe

Model output `[B,L,260]`; next-token CE on target action byte plus EOS, with prompt/marker/padding excluded exactly as in V7.17. Peak LR1e-4, cosine, clipping1, FP32, batch16/micro2. Four full epochs. Preserve the original raw policy and separately evaluate grammar/availability conditions. This is the default fallback if candidates fail.

Do not compare a new architecture only with the old 3e-4 model. Do not claim cross-interface CE values are numerically comparable: B0 averages supervised byte/EOS tokens, whereas typed heads use one action loss per state. Report both objective loss and common exact-action/episode metrics.

### 5.3 J: joint categorical head — required output-interface control

Take `h` at the final public ANSWER-marker position, shape `[B,512]`, before any target token. All prompt information is causally accessible. Compute `z = h @ W_J + b_J`, shape `[B,19]` for A–P, QUERY, RETRIEVE, ABSTAIN. Optimize mean categorical CE per supervised state.

J removes byte/EOS generation as an uncontrolled difference between F/P and B0. Serialize the selected action using an engineered deterministic adapter; generated EOS is not a learned achievement for J/F/P. Keep the backbone's tied embedding table; do not allocate another embedding. Added head count is 512×19+19=9747.

### 5.4 F: factorized action and argument — the independent candidate

From the same h, predict operation logits `[B,4]` with a linear head and answer logits `[B,16]` with a separate linear head. No extra availability features or gold task indicator. Parameters added: (512×4+4)+(512×16+16)=10260, only 513 more than J. Report the difference; do not add dummy parameters and call the models exactly matched.

Let p_op=softmax(z_op), p_val=softmax(z_val). Define the joint action distribution:

```text
p(ANSWER(a)) = p_op[ANSWER] * p_val[a]
p(QUERY)    = p_op[QUERY]
p(RETRIEVE) = p_op[RETRIEVE]
p(ABSTAIN)  = p_op[ABSTAIN]
```

Training loss per state: `-log p_op[gold_op] - 1[gold_op=ANSWER] * log p_val[gold_arg]`. Average over states, not independently over the answer subset; otherwise class weighting changes. Use stable log_softmax. Decode by argmax over the **19 joint probabilities**, not argmax operation followed by argmax argument. The latter can choose a low-probability answer merely because total ANSWER mass is large. For masked inference, form the joint distribution first, remove only unavailable operations, then renormalize.

J and F share PublicInput, datasets, optimization schedule, output action set, serialization and losses measured per state. J→F is the primary factorization contrast. B0→F alone does not isolate factorization.

**Algebraic null:** group J's 16 answer probabilities into ANSWER mass and divide them by that mass to produce conditional answer probabilities. Recombining must exactly recover J probabilities, joint argmax and gradients within numerical tolerance. This proves that mere mathematical re-expression is not a learned architectural effect; F earns its difference through separately parameterized functions.

### 5.5 P: conventional pointer-assisted F — binding competitor

Use F's operation head. From backbone hidden states `H:[B,L,512]`, compute query `q=hWq` and keys `K=HWk`, both dimension64; scores `q·K/sqrt(64)`. Candidate positions are **every public-prefix byte token whose decoded byte is A–P**. Include occurrences in questions, instructions and distractors; do not parse “value columns,” follow edges, or mark relevant facts. Exclude target suffix, padding, and other packed examples. Repeated occurrences remain distinct positions.

Softmax scores over candidate positions; scatter-add their mass into 16 answer-symbol bins to obtain p_copy. Mix with F's answer distribution:

```text
g = sigmoid(w_g · h + b_g)
p_arg = (1-g)*p_val + g*p_copy
```

Use the same joint operation/argument distribution and per-state likelihood as F. No gold pointer alignment loss in the first screen. Gradients flow through mixture, attention and backbone; integer token-to-bin mapping is not differentiable and does not need to be. If there are no candidates, set p_arg=p_val with a documented empty-source branch; never softmax an all-negative-infinity row. No detach of attention or hidden states.

Initialize gate bias -2, all other head weights by the registered initializer. This is approximate baseline behavior, not exact neutral parity. An explicit `copy_enabled=False` must bypass pointer computation and reproduce F exactly with shared F weights. Pointer parameters must be excluded from the disabled model inventory, not left as trainable unused tensors to evade all-gradient checks.

This is a conventional mechanism adapted to this testbed. [Pointer-generator networks](https://nlp.stanford.edu/pubs/see2017get.pdf) establish the prior concept of mixing generation and copying; they do not establish that it will learn ELENDRIL relations. Single-byte A–P answers do not establish arbitrary string copying. Alias tests rename input entities while preserving a public A–P output convention; arbitrary-name output requires a separately versioned span/sequence task.

**Required active-capacity comparator F-cap:** replace the pointer branch with a learned residual bottleneck on h, `h' = h + W2 SiLU(W1 h + b1)+b2`, used by the value head. Enumerate positive integer bottleneck widths and choose the closest added-parameter count to P; require ≤1% difference in added parameters or report no match. All comparator parameters must affect outputs and receive gradients. It has no source-position addressing. This controls whether P merely benefits from added trainable computation. Profile cost separately.

### 5.6 Conditional recurrence U — specification, not a first-phase dependency

Only after one-hop qualifies, test a shared refinement state with T∈{1,2,3,4}. Initialize z0 from h. At each step attend from z_t to public H using shared q/k/v projections of width64, project the read vector to width512, and apply a shared GRUCell `[512→512]` to obtain z_(t+1). Read final F heads from z_T. No gold intermediate entity or required step count enters the forward pass. Start with fixed T=3, no adaptive halting. Use the same T on train and primary evaluation; T-sweeps are separately labelled interventions.

Compare U with T=1 and with a nonshared three-block refinement model at matched measured inference budgets. Parameter matching and compute matching are different comparisons. Shared recurrence has established precedent in [Universal Transformers](https://arxiv.org/abs/1807.03819); this restricted adaptation is a proposal, not a claim that recurrence will extrapolate.

## 6. Dataset and independent evaluation specification — BUILD FIRST

### 6.1 Two datasets, no hidden recipe drift

**D0:** exact inherited V7.17 training worlds and supervision. Preserve its seed606, UTF8 identity and 9472 samples. Never regenerate it with altered functions and retain the old hash.

**D1:** a new coverage-controlled corpus with 3584 training worlds, 512 in each of seven families: direct one-hop, two-hop, known temporal/source selection, missing/query, conflicting/query, memory retrieval, and a combined null/unavailable family. Only that last family's 512 worlds are divided into 256 null and 256 unavailable cases; the other six families each contain512. Every world yields one episode; teacher-generated additional states remain explicit and counted. Thus 3584 worlds does not mean 3584 supervised examples. Four epochs means four passes over the actual sample list. All D1 arms use precisely the same examples and order plan.

This is a starting bounded corpus, not an assertion that text size implies adequate learning. Log generated UTF8 bytes and supervised state counts. Do not duplicate examples merely to reach 2–3 MB. If runtime/context constraints require different quotas, amend before research, regenerate every arm, and publish the changed recipe. D0↔D1 is a data intervention; it is not exposure-matched and cannot establish architecture causality.

### 6.2 World construction rules

For direct and route worlds, sample 4–12 distinct entities from A–P, balance the terminal answer approximately uniformly with per-split max/min target-count ratio ≤1.1. Create acyclic unambiguous routes. For ambiguous/missing cases, specify unanswerability and tool availability independently. Construct timestamps and row order independently of route order: position, highest timestamp, first/last row, lexicographic symbol and number of rows must not identify the answer better than the declared heuristics after balancing.

Generate 0, 2 or 4 distractor edges uniformly, avoiding unintended alternate valid routes. Temporal selection has explicit same-source update semantics; cross-source unresolved conflict needs a query or abstention if unavailable. The teacher must not silently reuse the evaluation solver. Include null cases as true unknown outcomes, distinct from an obtainable unknown.

Availability must not uniquely identify task family. Include answerable worlds both with and without available tools; unavailable worlds with irrelevant tools; cases where retrieval is available but unnecessary. Match visible fact counts/length distributions across answerable and unanswerable groups where feasible. Publish cross-tabs of family × target operation × availability × length bin; do not claim perfect balance unless measured.

Partition by a canonical semantic graph identity **before rendering**. A world, its rename, row permutation, temporal-equivalent rewrite, counterfactual partner and all teacher observation states belong to one partition. A new timestamp offset or label permutation must not make a training world into an “independent” test world. Use graph canonicalization appropriate to at most12 entities; if canonicalization is approximate, label its coverage and test explicit isomorphism families. Never use task success to decide partition assignment.

Distinguish **new instances of a shared task structure** from **new graph structures**. All isolated one-edge graphs are isomorphic: demanding disjoint isomorphism classes while putting that same motif in every split is impossible. Construct the primary structural split from full multi-entity worlds, including distractor edges and semantic source/time roles, with sufficient distinct graph classes. Keep standalone one-edge examples as explicitly shared-structure acquisition probes, not structural-generalization evidence. Audit the number of available canonical classes before assigning quotas; if insufficient, fail dataset admission and revise the generator prospectively. Counterfactual pairing uses the connected component of the pairing relation as the split unit, so a counterfactual that matches another base world cannot leak across partitions.

Train with two surface templates, row permutations, and half canonical/half training-only aliases (e.g. a reserved prefix set). Development holds out a third template and different alias prefix set. Confirmation uses a separately implemented renderer and world constructor with no reuse of train world-building or target-selection functions. Sharing a common task specification is allowed. If authored by the same model/team, call this **separate implementation with shared authorship**, not independent external authorship.

### 6.3 Evaluation cohorts and custody

Before scaling generation, implement hand-written semantic fixtures. For example, public edges `A→C; C→F; B→D`, question “from A after two edges?”, and an explicit fixed output-code legend imply ANSWER(F). Replacing only `C→F` with `C→H` implies ANSWER(H); replacing only `B→D` with `B→E` must leave ANSWER(F) unchanged. These are relevant and irrelevant counterfactuals, respectively. Hide `C→F` and make the query channel legitimately reveal it: the expected episode becomes QUERY, observation, ANSWER(F). With neither evidence nor a channel able to resolve it, ABSTAIN is appropriate. The teacher may know hidden truth to construct supervision, but the actor cannot read it before the observation. Replacing the query response by `C→H` must change the answer; score that paired intervention separately from ordinary query selection.

Define timestamp/source semantics in the public task specification and fixtures: later observations supersede earlier observations from the same source; equal-time contradictory observations remain unresolved; disagreement between sources remains unresolved unless a public authority rule or subsequent authoritative query resolves it. Never infer source authority from row position. Budget exhaustion, repeated unproductive calls, malformed output and an unavailable action must each have explicit terminal/status rules and denominators shared by all arms. Exact compatible historical behavior stays in D0; changed environment semantics receive a D1 environment version. Any unresolved semantic choice must be locked with these fixtures before dataset generation, not improvised during evaluation.

Historical suite remains a fixed development regression suite, never a new sealed confirmation set. Primary new development: 128 base worlds per relevant scenario, with four linked variants per world: canonical, row shuffle, fresh alias, and one relevant-fact counterfactual. Provide an equal-length irrelevant-fact counterfactual as a separate negative control. Primary confirmation: 256 fresh semantic worlds per relevant scenario, same paired transformations, generated only after architecture/recipe selection is locked. Precommit generator code and a seed commitment; owner/evaluator supplies the confirmation seed after model selection. If unavailable, the stage remains development-only.

A multi-hop confirmation suite excludes all its graph-isomorphism families from training and development. Hold out three- and four-hop lengths; do not claim composition extrapolation if training contains their target trajectories. Do not concatenate evaluator-only route explanations into actor prompts.

Each record needs: `world_id`, `parent_world_id`, `pair_id`, `semantic_hash`, `split`, `family`, public observation, separately stored hidden channels/truth, target provenance, renderer ID, token length, allowed public tools and content hashes. Actor dataloaders receive a redacted view by construction. Files containing evaluator truth must never be a model's context or retrieval corpus.

### 6.4 First frozen or fresh diagnostic

If genuine V7.17 FULL weights become available, verify archive/source/job identities and restore the three low-LR checkpoints. Evaluate raw, grammar, availability-mask and direct-answer-only policies on identical new public worlds, without optimizer updates. Log changed action rankings and terminal outcomes, not only reduced invalid counts.

If weights are absent, record `PARENT_WEIGHTS_UNAVAILABLE`. Do not stall independent architecture implementation. A newly trained D0/B0 baseline provides a separate fresh lineage; never label it the restored owner model. The old analysis ZIP cannot support new inference.

## 7. Exact initial run registry and resource comparisons

These are **proposed registrations**, to be materialized in `v8/protocol.py` before owner execution. No jobs below have run.

| Phase | Arms | Init/order seeds | Training jobs | Purpose |
|---|---|---|---:|---|
| `archive-diagnostic` | supplied low-LR parents, four policy modes | existing 7/19/41 | 0 | New inference only; conditional on FULL custody |
| `data-control` | B0/D0 vs B0/D1 | 7/7,19/19,41/41 | 6 | Test whether the revised corpus/sampler recipe repairs the plain learner |
| `head-screen` | J/D1, F/D1 | same three | 6 | Compare typed readout/objective, then isolate factorization within that interface |
| `reader-screen` | P/D1, F-cap/D1 | same three | 6 | Copy/read mechanism versus active extra capacity |
| `replication` | best eligible candidate and its exact relevant comparator | fresh 101/101,211/211,307/307 | 6 | New paired model replication; confirmation opened only here |

B0/D1 from data-control and F/D1 from head-screen may be reused in later comparisons only if source, data, initialization, full order plan, schedule and evaluation conditions match. If implementation evolves between phases, regenerate matched controls under the new source and charge them; do not compare across unrecorded code drift. Source changes confined to documentation must still be explicitly handled by identity policy, not bypassed.

Initial total is 18 development trainings in three sessions, plus six conditional confirmation trainings. Do not describe 24 scenarios as 24 training jobs, or all scenario/seed combinations as independent experiments. Do not launch all phases automatically from one Run All. One notebook can expose a validated PHASE field and run one declared phase per session.

Fixed first recipe: AdamW using inherited optimizer grouping and defaults recorded verbatim, peak LR1e-4, cosine to the inherited terminal ratio, clipping1, FP32, batch16/micro2, four epochs. Log the actual default betas, epsilon and decay from source before locking the registry; do not guess them. No scheduler retuning based on confirmation. Uniform full-epoch order for D1; preserve original blocked semantics for D0, and label this data-control arm as a combined corpus/sampler recipe comparison. Architecture contrasts all share D1 uniform order.

If a candidate fails positive acquisition, investigate implementation/optimization before calling the architecture impossible. Any LR tuning is a predeclared development-only second study: test {3e-5,1e-4,3e-4} for both candidate and its comparator, with equal tuning budget and unchanged confirmation custody. This option does not permit unbounded retries or selecting one lucky seed.

**Two scientific cost views are mandatory:**

1. Equal training exposure: same D1 samples, four epochs and order; report all extra architecture work, auxiliary tokens and measured inference latency.
2. Equal measured training work: save scheduled checkpoints and evaluate at a common compute budget established from production profiles, using processed tokens and GPU reservation as distinct measures. If a candidate's final advantage disappears at matched work, call it an accuracy–cost tradeoff, not a free architectural improvement. Parameter-count parity alone is not compute parity.

For P versus F-cap, require actual active added-parameter counts and operation estimates before launch. For J versus F, acknowledge the 513-parameter difference and report actual runtime; its size is tiny but not literally zero. No unused padding parameters or zero-gradient stubs.

## 8. Checkpoints, diagnostics and attribution

Evaluate untrained, epoch1, epoch2 and epoch4 checkpoints. Keep a checkpoint at each milestone and every64 completed updates for crash recovery. Do not reduce epochs or omit late arms when the clock runs out. An interrupted phase is PARTIAL, with job-level completion states.

**First-release evaluation allocation:** the primary final-checkpoint cohorts are R09, R10, E13, E14, E15 and A19, each with128 base worlds and the five linked forms in §6 (640 episodes per cohort). G05/G06/F01/F02/F04/G07 are annotations/contrasts on those same worlds, not duplicate cohorts. A17 repeats each cohort under the declared policy modes. Initial/epoch1/epoch2 evaluations use a fixed first32-world subset per cohort, with all five forms; epoch4 uses all128. Full training-state diagnostics remain required at every milestone. Historical regression evaluation stays separate. Conditional scenarios are NOT_RUN until admitted, and confirmation uses256 base worlds per applicable cohort only after selection. Record exact episode counts, maximum actions and mode counts in the registry; profile that complete workload, including decoding and logging, before admission. If it cannot fit, publish a prospective phase split or larger owner-run budget; do not quietly shrink evaluation after results are visible.

At each milestone retain:

* Full training action correctness and value accuracy conditional on a required answer; byte EOS metrics only where EOS is learned.
* Development interactive exact episode correctness, useful-query rate, unnecessary-query rate, invalid actions and abstention appropriateness.
* Separate teacher-state and autonomous rollout measurements, with exact denominators and the same cohort when reporting their gap.
* First observable divergence: public prompt, model proposal, legal availability, environment response, expected outcome stored only in evaluator record, and subsequent actions.
* Pair-level invariance and both-correct rate, semantic-counterfactual response changes, irrelevant-counterfactual stability.
* Per-family failures and raw/grammar/availability-only outcomes. A mask is never folded into the raw score.
* Full loss/LR/gradient-clipping trajectories, update counters, head/backbone gradient norms and optimizer ownership receipts. Gradient presence is not proof of successful learning.

For P log copy-gate values and attention entropy as descriptive diagnostics. Do not equate attention with explanations. Intervene at fixed weights: disable copy, permute candidate values while holding attention positions fixed, and use balanced sham permutations. These can localize dependence on the pointer path; they do not prove the latent reader implements an abstract relation algorithm.

For F log operation mass and conditional argument accuracy. A lower invalid-action rate without more correct final answers is a policy-validity improvement only. If the joint head J matches F, do not claim factorization helped.

Before every comparison, assert same case identities and checkpoint bindings. Report seed-level effects, per-world paired effects, wins and regressions. Do not turn shared worlds across seeds into hundreds of independent model replications.

## 9. Predeclared advancement rules

These thresholds are **proposed operational targets**, not empirical facts or guarantees. Freeze them before their corresponding run. They may be changed prospectively with a new protocol ID, never retroactively.

**Engineering admission:** valid dataset/oracle checks, finite updates, complete parameter ownership, bounded resources, exact same-runtime recovery, and positive-control acquisition. Failure stops the affected phase as an engineering/learnability result; preserve all diagnostics.

**Development candidate:** mean raw one-hop exact episode accuracy ≥80%; mean paired improvement ≥10 percentage points over its relevant strong comparator; at least2/3 seed differences positive and none below -5 points; canonical raw accuracy regression no worse than5 points. No success claim from masking alone. Require relevant-fact counterfactual both-correct rate ≥70% and irrelevant-fact stability ≥90%, together with accuracy, because always emitting one symbol can be perfectly stable.

**Confirmation candidate:** preregister one winner and one comparator, then use three fresh paired model seeds and owner-held confirmation worlds. Require all three seed differences positive, mean gain ≥10 points, same capability/robustness thresholds and no disqualifying regression. Report individual seed/world uncertainty. A paired world bootstrap may quantify variation across worlds **conditional on those three trained seeds**; it cannot establish population-level reliability across initializations. If model-seed effects are unstable, request a new registered five-seed study rather than overinterpreting a narrow confidence interval.

**Composition admission:** direct one-hop and familiar two-hop qualify independently first. Then assess held-out lengths and new graph structures. Training on three-hop labels creates a three-hop acquisition study, not two-to-three extrapolation.

**First achievement language:** “Intervention X improved raw one-hop/counterfactual performance under these matched conditions, with these costs and limitations.” Do not say AGI, general reasoning, a solved architecture, or automatic self-improvement.

## 10. Retention, fast memory and transfer — fully specified conditional program

### 10.1 Retention comes after acquisition

Define A as public entity/value selection with temporal corrections; B as a separately generated two-hop mapping task with disjoint semantic graph families. Qualify each independently on 256 new development worlds, raw accuracy≥80% and relevant counterfactual both-correct≥70%. Use equal B training exposure across continuations. A task that never qualified cannot subsequently “forget.”

From each qualified A parent fork four continuation arms with three paired seeds:

* N: B-only, no replay.
* R: B plus uniform random replay from A training support.
* S: B plus fixed capability-stratified replay.
* G: B plus adaptive allocation within the same replay quota.

Store 128 A training worlds, balanced across eight preregistered skill/condition strata with16 each. These strata may select replay examples but must not be supplied to the actor. Replay four old supervised states per16 new B states, with the same total replay quota in R/S/G and the same B permutation. Charge all replay, storage and monitor work. N has lower cost; add N-extra, when affordable, spending the same extra update budget on B examples to distinguish replay content from extra optimization.

G is deliberately simple: every64 B updates, evaluate a fixed 64-world development monitor separate from replay, confirmation and final evaluation. Compute each stratum's error rate, use EMA with coefficient0.5, and allocate the next block's fixed replay quota proportionally to error+0.05 with a minimum one sample per stratum when the block quota allows. Tie-break by stable stratum ID. No changes to total replay count, LR or update caps in this first Guardian test. R and S receive the same monitoring calls for cost-matched comparison but do not adapt. Persist EMA, allocation counts, buffer IDs and monitor cursor in checkpoints.

Success target: old-A accuracy improves≥10 points versus N while new-B accuracy is within5 points; G must additionally beat S at matched replay+monitor cost or reach the same qualification using a separately preregistered smaller replay budget. If S matches G, prefer S. This tests a transparent replay allocator, not a learned metacognitive controller.

Do not transplant ARK-017 movement caps as universal bounds into S20. Their norms depend on parameterization, optimizer and task. Reimplement a cap only in a separate, normalized, matched intervention with measured unclipped/clipped update paths.

### 10.2 Fast memory and correction

Keep neural weights frozen for the designated memory-only comparison. Use the existing bounded public experience interface, capacity32 records initially. Records contain public observations, provenance and time/version, never evaluator answers. Compare no retrieval, learned RETRIEVE, and engineered retrieval under the same record budget; add shuffled and stale-but-length-matched memory controls. Charge retrieval and context tokens.

Inserting an observed fact into an engineered store is an engineered state update. Attribute learning without weight updates only to measured improvement from accumulated legitimate experience, and disclose which write/read decisions are learned. Independent future episodes must require reuse, not identical world replay. No secret prefetch of the target fact.

### 10.3 Adapters and recurrence

Adapters are optional if shared weights show interference despite sparse replay. Their router sees public context only, never task IDs. Count total stored parameters across all adapters; compare learned routing with an explicitly labelled oracle-routing upper bound, not a hidden oracle. Test whether isolation also prevents positive transfer. Pair each integration with its component-alone results and a removal ablation.

Recurrence U is eligible only when single-step grounding qualifies. Use the exact shared-state contract in §5.6 and matched-compute comparisons. Do not combine it with replay, a new tokenizer and a new loss in its first causal test.

### 10.4 Forward transfer and cumulative acquisition

C combines source revision with relational traversal using previously unseen semantic structures and a separately authored rendering family. Before C training, evaluate C: if already above the qualification threshold, call it zero-shot transfer, not faster acquisition.

Compare four histories: fresh model; A→B; unrelated-history U1→U2 matched in training tokens/updates and comparable public-format exposure; A-only with budget-matched unrelated continuation. Use the same model architecture, C sequence, optimizer-reset policy, initial C LR, maximum C exposure and evaluation schedule. Report C learning curves every64 updates, plus retained A/B. Reset optimizer state for every history at the C boundary in the primary comparison; optimizer-state continuation is a separate ablation.

Time-to-acquisition is the first scheduled C checkpoint reaching≥80% on the fixed acquisition-monitor set and remaining above it at the next checkpoint. Evaluate separate final held-out C worlds afterward. Runs that never reach it are right-censored at the maximum budget; do not drop them or substitute that maximum as an observed success time. Success target is≥10% fewer C examples or measured training work versus the unrelated-history control, with A/B still qualified. Also report total lifetime A+B+C work; cheaper marginal C acquisition need not mean cheaper lifetime training.

No adaptive architecture rewriting is admitted by a successful A→B→C proxy. A future self-improvement study needs a frozen proposer, allowed change space, independent verifier and prospective unseen task sequences.

## 11. Scenario ledger: all 24 suggestions, with explicit scope

Common defaults for active first-release scenarios: S20; seeds7/19/41; untrained/epoch1/epoch2/epoch4; raw primary, availability-only secondary; separate-development/confirmation worlds as §6. Rows below are evaluator conditions, not standalone training jobs.

| ID | Implementation and contrast | Stage / status at handoff |
|---|---|---|
| F01 | Public entity aliases with fixed answer-code legend; separate arbitrary-string copying later | Formation; planned. Do not overclaim new output vocabulary |
| F02 | Multiple entities, changed relevant attribute, row/distractor controls | Formation; planned |
| F03 | Reverse relation question with unique inverse versus ambiguous inverse | Diagnostic extension; planned, no training on held-out inverse labels |
| F04 | Longer aliases and different notation within512-token bound | Formation; planned; >512 separately deferred |
| G05 | Novel input aliases and consistent symbol permutations | Formation primary robustness; planned |
| G06 | Row shuffle and independent timestamps | Formation primary robustness; planned |
| G07 | Matched-length distractors, same-source revision, cross-source conflict | Formation; planned |
| G08 | Separate generator/renderer, explicit author provenance | Confirmation; conditional |
| R09 | Direct one-hop, raw/J/F/P comparison | First primary capability endpoint; planned |
| R10 | Two-hop routes, counterfactual edge replacement | Composition prerequisite; planned |
| R11 | Held-out three-hop, unrelated/sham clamp controls | After R09/R10; conditional |
| R12 | Source revision plus route task C | Transfer; conditional |
| E13 | Fully sufficient facts with tools available/unavailable independently | Formation; planned |
| E14 | Missing fact, legitimate query, changed returned payload | Formation; planned |
| E15 | Conflicting sources and subsequent authoritative observation | Formation; planned |
| E16 | Stale memory followed by public correction | Memory stage; conditional |
| A17 | Raw versus public-availability-only intervention | First diagnostic; planned |
| A18 | Available queries at costs0/0.1/0.5/1 | Fixed-policy repricing first; learned cost conditioning NOT_RUN |
| A19 | Null/unavailable versus obtainable unknown | Formation; planned |
| A20 | Wrong proposed answer followed by public corrective evidence | Diagnostic extension; learned self-error detector NOT_RUN |
| C21 | Qualified A→B, no replay | Retention baseline; conditional |
| C22 | Same B with fixed random/stratified sparse replay | Retention; conditional |
| C23 | Same replay quota, adaptive allocation and charged monitoring | Retention; conditional |
| C24 | A→B→C versus unrelated-history/fresh controls | Transfer; conditional |

All rows have actual status NOT_RUN at this planning handoff. The runner must distinguish NOT_IMPLEMENTED, IMPLEMENTED_NOT_RUN, PREREQUISITE_NOT_QUALIFIED and EXECUTED instead of populating missing rows with zeros.

## 12. Mapping every proposed research question to this design

| Letter question | Decision here |
|---|---|
| 01 baseline reconstitution | Restore available genuine parents; fresh lineages if absent; no invented weights |
| 02 policy split | Raw/grammar/availability/direct-answer diagnostic, with changed-task labels |
| 03 token/value/EOS split | Preserve for B0; typed arms explicitly lack learned EOS |
| 04 smaller-scale LR pilots | Deferred by owner's S20 minimum, not silently run |
| 05 generators/canaries | Mandatory before owner research |
| 06 binding/pointer | P versus F and F-cap |
| 07 iterative workspace | U after grounding, with matched inference work |
| 08 affordances | Engineered mask first; explicit-feature training is a separate prospective arm |
| 09 fast state | Bounded public-memory stage, separate retrieval controls |
| 10 invariant loss | Conditional: paired augmentation-only baseline first; add loss only with equal paired-data exposure |
| 11 qualify A/B | Required before retention; no failed acquisition called forgetting |
| 12 plasticity/replay | N/R/S first; LR/caps separately registered |
| 13 support replay | Random versus stratified supports with fixed memory/replay cost |
| 14 modularity | Conditional on observed interference; learned versus oracle routing separate |
| 15 Guardian | G must outperform fixed S or earn lower-cost qualification |
| 16 composition | Independent R10/R11/R12, no hidden intermediate labels |
| 17 transfer | Matched histories, pre-C evaluation, censored acquisition curves |
| 18 interactions | Best two components only after individual effects; test removal and joint regression |
| 19 scale/domain replication | Fresh model seeds and separate implementation first; larger scale later, no automatic S3/S8/S9 campaign |
| 20 challenger tournament | Prospective phase registry; no moving target inside a running campaign |

## 13. Proposed file-level work packages and interfaces

**These files and commands are to be implemented; they do not currently exist.** Do not tell the owner to run them until implementation and qualification succeed.

| Package | New paths | Acceptance artifact |
|---|---|---|
| W0 source/provenance | `research/v8/SOURCE_AUDIT.json`, `DESIGN_DECISIONS.md` | Actual refs, dirty-tree handling, donor evidence tiers and availability |
| W1 contracts/registry | `anra_v5/elendril/v8/contracts.py`, `protocol.py` | Serializable exact model/data/job identities, immutable ordered job matrix |
| W2 data/verifier | `v8/data.py`, `v8/teacher.py`, `v8/verifier.py`, `v8/holdout.py` | Split-isomorphism audit, oracle agreement, contradictory-label checks, shortcut baselines |
| W3 models/policies | `v8/models.py`, `v8/policies.py` | B0/J/F/P/F-cap receipts, no-op/algebraic parity, truth firewall |
| W4 training | `v8/training.py` plus a minimal factory hook in the existing trainer | Model built before optimizer; all parameters owned; old paths unchanged |
| W5 measurement | `v8/evaluation.py`, `v8/analysis.py` | Raw/masked separation, paired denominator validation, first divergence, cost/uncertainty |
| W6 workers/export | `tools/elendril_v8_worker.py`, `elendril_v8_campaign.py`, `elendril_v8_export.py` | Exact continuation, fail-closed completion, partial exports and restoration |
| W7 notebook | `tools/build_elendril_v8_notebook.py`, `notebooks/ELENDRIL_V8_KAGGLE_T4X2.ipynb` | Immutable-source rebuild and real owner-run GPU receipts |
| W8 later learning | `v8/replay.py`, `v8/memory.py`, `v8/transfer.py` | Only admitted after their prerequisite reports; no disconnected default modules |

Proposed CLI, once implemented:

```text
python -m tools.elendril_v8_campaign --phase data-control --source <40hex> --output <outside-repo> --minutes 330 --owner-enabled
python -m tools.elendril_v8_campaign --phase head-screen --source <40hex> --output <outside-repo> --minutes 330 --owner-enabled
python -m tools.elendril_v8_campaign --phase reader-screen --source <40hex> --output <outside-repo> --minutes 330 --owner-enabled
```

No command may silently default to CUDA research when imported. Registry generation must be pure. Unknown arm IDs, phase/job mismatch, source mismatch, transformed datasets without new identity, or unsupported checkpoint versions must fail before training.

Implement one work package at a time, inspect the diff, run the relevant checks and commit meaningful units. Append dated Codex/Gemini entries to `progress.md`, maximum five nonempty lines each. Do not invent prior executions in the register.

## 14. Numerical, training and recovery invariants

* Causal masks remain block-diagonal across packed examples. Pointer masks additionally exclude future target tokens, padding and other segments. Test adversarially with identical labels in another packed segment.
* Use only the hidden state at the last public prefix token for action heads. A forward pass containing teacher targets must not let the head read their hidden states.
* Microbatch objective denominators refer to the whole logical batch. F's answer loss is zero on non-ANSWER states; empty ANSWER subsets must not divide by zero. Compare accumulated gradients with an independent reference.
* New model factory returns the complete module/spec/tokenizer identity before optimizer creation. Count trainable unique storages once, including tied embeddings. Every expected parameter must update in a suitable positive example; inactive conditional paths need a declared coverage test, not a blanket weakening of gradient assertions.
* Factorized value-head gradients can legitimately be absent on a batch containing no ANSWER labels. Handle this through an explicit per-objective expected-active-parameter contract, or guarantee ANSWER-containing logical batches in a separately declared common sampler. Do not silently replace missing gradients with zeros to pass ownership checks. AdamW decay/step behavior for inactive heads must be specified and tested consistently with resume.
* Prefer the existing FP32 path initially. Mixed precision is a separate engineering qualification with overflow accounting, not an invisible speed fix.
* Version `V8ModelSpec` to include base spec, head type, pointer/recurrent dimensions, policy interface, vocabulary and output mapping. Hash the complete active tensor inventory, optimizer groups and objective configuration. The old core's base-only receipt must remain valid for B0; a wrapper asserts base plus explicit head receipt.
* Old checkpoints may be loaded for explicit weights-only migration into a declared new lineage; initialize new heads deterministically and reset optimizer unless an exact compatible migration is defined. Never claim exact resume across changed architectures/objectives.
* Save model, optimizer, scheduler horizon, gradient-scaler state if applicable, sampler plan/cursor, all RNG states, accumulation boundary, data/source identities, memory/replay/controller state and current learning phase. Publish only completed update transactions.
* Kill/restart tests must cover interruption before update, mid-update, between payload and manifest publication, and during ZIP replacement. Resume the last committed generation; no partially mutated state may be published as a completed step.
* Checkpoint schema mismatch is a rejected migration, not an excuse to use `strict=False`. Corruption must be reported and rollback must identify the last independently verified generation.

## 15. Validation sequence and stop rules

**CPU mechanics:** tiny models ≤100k parameters, ≤32 examples, ≤8 updates per case. Test actual tensor code and actual serialization. These fixtures establish mechanics, never learned S20 capability. Cover dimensions, target exclusion, empty pointer source, repeated source symbols, packed-segment leakage, F joint argmax, J algebraic null, active parameter inventory, exact no-copy F parity, deterministic order, frozen evaluation, interrupted restore and mutated archive rejection.

**Owner GPU positive acquisition:** for each active architecture, use a fixed 32-world training-only cohort covering all seven families and valid/invalid availability contexts, preserving every teacher state. Allow at most256 updates, peak LR1e-4, batch16/micro2, same registered schedule for all arms. Require ≥95% training-state exact actions, ≥90% interactive training episodes and final mean loss≤20% of initial measured loss. Save the whole curve and failures. This is a memorization/learnability canary, not generalization. If a model cannot pass, record POSITIVE_CANARY_FAIL; inspect whether the failure is code, optimization or the task. Any modified canary recipe needs a recorded pre-research amendment applied fairly to controls.

**Production profile:** both physical T4s, actual S20 module, longest admitted batch, real optimizer, at least16 measured updates after warmup, all evaluation modes, checkpoint save and fresh-process resume. Test save at update2, compare uninterrupted to restored update4 with the same full scheduler horizon and exact payload expectations. Do not shorten the horizon to four updates for the recovery branch.

**Integration test:** orchestrate a tiny complete campaign, forced worker failure, missing mandatory report, stale ZIP, wrong source pin and simulated deadline. Test the exact published notebook cells using safe subprocess stubs for launch wiring; this is additional to real tensor tests, not a replacement. Rebuild the notebook from a Git archive of its pinned source and compare bytes/normalized notebook structure.

Once focused checks pass, run the existing CPU suite and evidence audit once for the trainer/checkpoint changes. Repeat only affected checks after a fix unless new concerns justify a broader run. No test may be weakened because a candidate fails. Preserve failure traces and repair or reject the candidate.

## 16. Kaggle admission, custody and release contract

One notebook, one selected phase per session, two isolated physical T4 workers. Do not use DataParallel/DDP merely to occupy both GPUs; independent jobs use the devices efficiently and avoid treating two16GB cards as one32GB card. Capture UUIDs, actual free VRAM, Python/PyTorch/CUDA versions, GPU capability, source and protocol hashes. Use native installed PyTorch; dependency replacement requires separate compatibility validation.

Initial metadata defaults: `RUN_STUDY=False`, `PHASE='data-control'`, `MINUTES=330`. Display an unmistakable “NOT LAUNCHED: owner switch is false” status when disabled, with exact enable instructions. On explicit owner authorization for that release, the publication may default True and Run All proceeds after canaries. Do not silently skip training and print success. Do not override an owner's later authorized preference.

Generate public procedural input automatically in `/kaggle/working`; `/kaggle/input` is for attached immutable archives. No manual input attachment is needed for fresh D0/D1 training. A FULL parent archive is needed only for the corresponding restore/frozen diagnostic phase. Never download a dataset name that has not been published or fabricate a Kaggle API credential.

**Forecast:** use measured per-arm optimizer-step time, complete diagnostic timing, serialization time and actual per-job update counts. Sum the work assigned to each lane, multiply training/evaluation estimates by2, add measured canary/startup/export work and at least600 seconds final reserve. Admit only if the whole selected phase fits remaining time. Production observation V7.17 used111.48 controller minutes for eight S20 jobs; a six-job phase would be about83.6 minutes only under identical per-job work, which V8 does not guarantee. Treat this as context, not an admission result.

**Storage:** FP32 weights alone are about79.5MB for the base model; weights plus two Adam moments are roughly238.5MB before metadata, extra heads and transient copies. Compute storage from actual tensor inventories. Include every retained milestone, recovery copy, existing raw outputs, previous ZIP, staging ZIP and reserve. Measure peak VRAM with optimizer state and actual longest tensors; attention/activations can dominate. If VRAM fails, reducing microbatch while preserving logical batch requires fresh parity/timing qualification; never silently truncate context or discard examples.

At a graceful deadline, finish or reject the current optimizer transaction, checkpoint the last valid boundary and export PARTIAL. Do not start a job that cannot safely reach its recovery/export reserve. SIGTERM handling and outer timeout behavior must agree; abrupt VM loss can still destroy session-local outputs. State this accurately.

Exports: `ELENDRIL_V8_ANALYSIS.zip`, `ELENDRIL_V8_FULL.zip`, `FINAL_STATUS.json`. Do not recycle misleading V71 labels in V8. Analysis includes protocol/source/data identities, raw episodes, per-step histories, diagnostic records, matched comparisons, controls, costs and errors. FULL additionally contains model/optimizer/RNG/replay/memory state. Save incremental analysis after each job and durable raw checkpoints throughout; periodically verify a recovery snapshot according to measured disk/time budget. Never promise final ZIP survival after a VM is destroyed before output is saved.

Completion is a state machine: `PREPARED → ADMITTED → RUNNING → EXECUTED → EXPORT_VERIFIED → COMPLETE`; errors may produce `NO_GO`, `PARTIAL` or `INVALID_RESULTS`. COMPLETE requires all registered jobs, all required milestones/modes, independent recorded-action replay, valid checkpoint components, both current-run verified ZIPs and a final delivery receipt. A campaign marker created before export is not the final receipt. Fresh-VM restore is a separate qualification. Archive-only scoring replay is not neural restoration.

Pin runtime source commit first; generate notebook referencing that exact40hex commit; test reconstruction from its Git tree; publish notebook in a subsequent commit. Owner downloads analysis, FULL and final status and saves Kaggle output. GitHub result folders receive only focused reports and essential curated JSON, never raw ZIPs or checkpoints.

## 17. Donor and integration ledger

These are provenance limits, not permission to claim historical successes anew.

| Donor | Inspected identity/status | Reuse and required qualification |
|---|---|---|
| Elendril neural/training/checkpoint core | Verified repository at 0c9cb284; V7.17 runtime724af4cb | Reuse infrastructure, preserve baseline parity, add explicit factory/spec extension |
| V7.17 owner results | Archive/actions/measurements audited; FULL absent in that audit | Motivation and controls; no invented restored model |
| Cymek branch | Local remote ref `feac2948466832516b7f048435483fd95e53b078`; formation records exist | Audit specific donor file/commit before reuse; branch name is not a qualified mechanism |
| ARK-017 V2 | `origin/Arkenstone` inspected at `4ae9e3b6dc806d6ada7b7b2f05d32271d072d543`; report names runner `377c4743f8017e3455f576eafb75bb8ab9c50284` | Reported audited micro replay/movement result; raw bundle not reaudited for this guide; implement sparse replay as a competing hypothesis |
| ARK-019 V4 | Report names runner `8e858c614d764335100dfd41dda6f8e0d0c877a7`; explicitly transcribed external audit | Motivation for retention; static replay remains serious control; no automatic Guardian promotion |
| Pointer reader P | New V8 adaptation of established pointer-generator idea | Independent enabled/disabled and active-capacity comparisons |
| Factorized policy F | New ELENDRIL proposal, unexecuted | J control and algebraic null before scientific attribution |
| Recurrent U | Conditional proposal with established recurrence precedent | One-hop prerequisites and compute comparison |
| Memory/Guardian/adapters | Conditional integrations, not built by this guide | Standalone qualification, joint factorial/removal tests, costs and rollback |

Every donor record must include immutable source, license, raw-evidence tier (`AUDITED_HERE`, `REPORTED_EXTERNAL_AUDIT`, `UNVERIFIED`, `PROPOSED`), interface assumptions, expected parameter changes, standalone effect, interaction result and rollback path. Do not infer Triquetra/Brahmastra capabilities from their names; record them as unreviewed unless the implementing agent actually inspects relevant evidence.

## 18. Decisions when the evidence disagrees with the proposal

| Observation | Required decision |
|---|---|
| B0/D1 repairs one-hop as well as all added modules | Prefer B0 for this claim; attribute to changed training recipe/coverage and investigate which data factor mattered |
| J improves over B0, F does not improve over J | The combined typed-readout/objective intervention is sufficient here; factorization has no demonstrated benefit. This does not isolate output restriction from objective/EOS changes |
| F beats J across paired seeds and independent worlds | Retain factorization provisionally; inspect operation versus argument effects |
| P beats F but not F-cap | Added capacity/work remains an explanation; no pointer-specific promotion |
| P wins canonical worlds but fails permutations/counterfactuals | Record shortcut/generalization failure; redesign source reading or data before scaling |
| Availability mask fixes legality but not correctness | Preserve safety/control gain; task grounding remains unresolved |
| Exact tiny cohort cannot be acquired | Investigate instrument/optimization; no capacity ceiling claim |
| One-hop qualifies, three-hop fails | Admit controlled recurrence or intermediate supervision study with fair extra-supervision controls |
| Static replay matches Guardian | Prefer static replay; do not promote adaptive control |
| Retention protects A but prevents B/C learning | Record stability–plasticity tradeoff; modular isolation is not cumulative intelligence |
| A+B gives zero-shot C | Report zero-shot transfer; redesign C if measuring acquisition speed |
| All mechanisms fail under sound controls | Keep the negative result; examine task semantics, supervision and architecture assumptions; do not relax confirmation thresholds |

An amendment must state the observation, rival explanations, proposed changed variable, frozen prior results, new protocol/source identities, costs and disconfirming outcome. New hypotheses are welcome between registered runs. Never amend a running study to remove a difficult seed or redefine success.

## 19. Handoff completion checklist for Gemini

Before reporting “ready for owner execution,” provide:

1. Exact commits and an implementation ledger showing which work packages are complete, conditional or deferred.
2. Machine-readable registry with the exact six jobs in the selected phase, seed roles, data/sample hashes, architecture receipts and total update counts.
3. Data/teacher/verifier audits and explicit actor-truth firewall proof; no synthetic “independent” authorship claim.
4. Actual CPU tensor/recovery tests and a passing CI link. Mock notebook wiring alone is insufficient.
5. A notebook rebuilt from its immutable runtime source, with correct launch status, fresh-data handling, output download links and recovery instructions.
6. Resource admission code that measures both T4s and budgets the entire phase, including diagnostics, replay and ZIPs.
7. A result template that separates engineered controls, learned raw policy, historical evidence, observed failures and unresolved mechanisms.
8. Clear owner action: what phase to run, required inputs if any, how long the cap is, what files to return, and whether GPU qualification has actually happened.

After owner execution, independently verify artifacts, all seeds and paired cohorts before writing scientific conclusions. Preserve regressions and failed hypotheses. Curate reports under `Experiment Result/V8/<run_id>/`; leave large owner files outside Git. Do not implement a new campaign simply because an aggregate score improved; use the decision table to select the next discriminating comparison.

## 20. Sources, evidence boundaries and CEO decision

Primary technical references checked during planning:

* [See, Liu and Manning, pointer-generator networks](https://nlp.stanford.edu/pubs/see2017get.pdf): established copy/generate mixture, not proof of relational reasoning.
* [Dehghani et al., Universal Transformers](https://arxiv.org/abs/1807.03819): recurrence with shared transformations as a comparator, not a promised solution here.
* [Chaudhry et al., On Tiny Episodic Memories](https://arxiv.org/abs/1902.10486): supports taking simple replay seriously as a competing baseline; results are not automatically transferable to this architecture.
* [Rolnick et al., Experience Replay for Continual Learning](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html): precedent for replay-based continual learning, not an ELENDRIL retention result.

Local sources actually inspected include the supplied architecture letter, current V7.17 protocol/results, the S20 registry, core forward/hidden-state interface, trainer model/optimizer construction, checkpoint-bound metadata, and the cited ARK-017/ARK-019 report excerpts. The historical companion Downloads documents referenced by the letter were not independently reaudited for this guide. Their claims remain reported context unless the implementing agent checks their evidence.

The CEO has three defensible options: a lower-risk coverage/control study, the recommended tightly controlled head-and-reader comparison, or a broader recurrence/memory program with greater cost and attribution risk. I recommend the middle option with the first study as its baseline foundation. It preserves architectural ambition, gives a plain learner a fair opportunity to win, and produces evidence that can determine which larger investment is warranted. The long-term objective remains cumulative, transferable learning; this first V8 release earns a small, precise part of that objective rather than declaring the entire architecture solved.
