# Evidence that would justify building further

AGI is the owner's objective. The immediate scientific question is whether the connected learner acquires useful transferable behavior more efficiently and retains it better than strong simpler controls. A finite suite cannot certify unrestricted general intelligence. It can expose missing abilities, reject ineffective mechanisms and support progressively broader claims.

## 1. Capability and implementation are different receipts

| Level | Evidence required | Permitted wording |
|---|---|---|
| Contract implemented | Public interfaces, types and focused negative tests | The interface exists and rejects these invalid cases |
| Integrated locally | Actual command/orchestration path with traceable fixture inputs | The local build connects these components |
| Gradient connected | Bounded backward checks and objective masks/counts | The specified supervision reaches these parameters |
| Accelerator checked | Intended actual-backend path, measured resources, fresh-process restore | This configuration ran and resumed on this device |
| Capability supported | Independent task instances, strong controls, uncertainty and full accounting | This treatment improved the named capability under this protocol |
| Continual improvement supported | Multiple acquisition/retention cycles with independent examination | These changes produced measured gains while meeting these retention bounds |
| AGI | Not established by the above alone | Research objective; scope of evidence must remain explicit |

Every handoff uses these distinctions. The agent must never turn a unit test of a fake planner into learned planning evidence or a random-head inference trace into inquiry learning.

## 2. Required measurements

For every episode record exact-answer success, valid EOS/stopping, action legality, fallback count, inquiry/submission costs, elapsed time, model calls/tokens, retrieval cost, termination/truncation and protocol budget. Aggregate success per world and per family. Keep token prediction loss, world content-field loss, Brier/log score where defined, action teacher agreement and value error as diagnostics, not substitutes for task success.

For goal pairs, report both-correct, both-wrong, same-answer rate, legitimate identical-answer exclusions and goal-swap score difference. For planning, report policy-only versus search decisions, realized benefit, simulated nodes, parse/unknown mass and goal-relevant versus irrelevant inquiry. For memory, report eligible sources retrieved, copied-answer cases, memory-off delta and same-mechanism exclusion status.

For learning, count unique examples/mechanisms, presentations, supervised targets, bytes/tokens, real interactions, optimizer updates, teacher computation and actual device time. Report capability versus each meaningful budget axis; one scalar cost conversion may be included only with its declared weights. A model using twice the search compute may be useful but does not establish equal-compute superiority.

Implement one versioned `compare_resources(parent, child, policy)` function over ResourceVector, used by the examiner and candidate publication gate. Each comparison protocol declares its claim class: **equal allowance** (identical specified caps, both treatments within every cap), **equal realized budget** (specified actual dimensions agree within preregistered tolerances), or **cost-quality frontier** (no equality claim; report the vector and tradeoff). Exact integers use exact equality where selected; device-time tolerance must be explicit and hardware/runtime compatibility verified. The first primary task comparison uses equal real-interaction and inference-token allowances plus a declared training allocation; teacher and retrieval allowances are included when enabled. Unmeasured required quantities block that comparison, and ignored dimensions remain visible as limitations. Never label all forms simply cost-matched.

## 3. Comparisons that can reject the design

| Hypothesis | Treatment and mandatory control | Result that rejects or restricts the claim |
|---|---|---|
| Shared trajectory learning helps decisions | Token-only versus token+world+action+value; same core and data exposure accounting | No task gain or loss of language/retention that outweighs the gain |
| Counterfactual grounding improves goal use | Pair term on/off; matched groups and answer exposure | Lower training pair loss without held-out both-correct or goal-dependent improvement |
| Predicted outcomes support planning | Policy-only, current root scorer and recursive planner | Search decisions do not depend on outcomes, or do not beat equal-budget controls |
| Inquiry selects useful evidence | Fixed, random, greedy, deeper and oracle diagnostic | High surprise with no answer benefit; no success/cost advantage on complementary queries |
| Memory helps transfer | Memory-off, lexical retrieval, optional learned selection | Gains explained by forbidden same-mechanism copies or excessive additional context |
| Retention mechanism preserves learning | Fixed replay versus controller/distillation candidates | Preserved old accuracy achieved only by failing to acquire new skills |
| Curriculum improves experience efficiency | Uniform/stratified schedule versus frontier selector | Selected-family gains with worse full-suite coverage or no equal-cost advantage |
| Repeated improvement is productive | Frozen parent, fixed-schedule child and proposed child | Apparent progress comes from more training, reused exam answers or selected successful cycles |

Avoid an enormous all-switch factorial. Build the answer-only control, qualify one new connection, freeze its design, then assess the next connection and a small number of motivated interactions. Keep rejected mechanisms disabled rather than accumulating every idea in the default model. A simpler configuration winning is a valid architecture decision.

## 4. Pairing, uncertainty and initial promotion policy

The independent sampling unit for task generalization is the mechanism/world cluster, not each query from that world. The independent training unit is a separately trained acquisition seed, not many forks of one acquired model. Pair parent and child by exact task/case IDs, labels/equivalence rules, budgets, memory regime and evaluation settings. Validate pairing before score subtraction.

Use a versioned hierarchical resampling procedure when both training seeds and world clusters are available: resample independent paired training seeds, then paired world clusters within the declared task stratum, retaining all case pairs in each selected cluster. Record RNG seed, replicate count, aggregation order and interval method. A single trained seed provides conditional task uncertainty only; it cannot establish robustness to training randomness. With few seeds, label interval stability limitations and prefer additional independent confirmation over many cosmetic bootstrap replicates.

Proposed initial confirmation policy for bounded success-rate tasks: five independent training seeds when an allocation permits, at least 100 distinct held-out mechanism clusters per primary family, point acquisition gain at least 0.03, positive adjusted lower confidence bound for acquisition, and every protected-family lower bound above -0.02. Use family-wise alpha 0.05 divided across the preregistered acquisition and protected-family claims, or another explicitly reviewed simultaneous procedure. These are engineering decision thresholds, not a universal definition of useful learning or guaranteed statistical power. Different outcome scales need an explicit corresponding policy. Do not execute this template if compute or task diversity cannot support it.

Screening can use fewer seeds and worlds but cannot be relabeled confirmation. Report inconclusive when intervals do not support the policy. Commit protocol and split identities before launch; deviations produce a new exploratory label. Do not select the best seed, choose the favorable metric after observing results, or omit rejected candidate cycles.

## 5. What historical branches teach and do not establish

The [B2 evidence snapshot](../build_20260912/evidence/SOURCES.json) records source commits, paths and hashes. The [evidence decisions](../build_20260912/EVIDENCE_DECISIONS.md) are the detailed interpretation. These are committed reports; not every raw accelerator artifact was available for independent reanalysis.

- Arkenstone acquisition forks show preservation/acquisition tradeoffs and a recovery result in which increased plasticity often helped reacquisition. Near-freezing is a confound for easy retention. Therefore measure new learning and displacement, and keep reacquisition distinct from stabilization.
- Cymek's arithmetic readiness results show severe representation/training sensitivity and seed dependence. Full-vocabulary generation and output-space treatments must be separated. An intermediate vocabulary result does not imply a universal optimum, and an unexecuted treatment plan supplies no efficacy evidence.
- BRAMASTRA D02's small matched inquiry comparison did not establish a reliable general advantage. It motivates complementary-query diagnostics and stronger controls, not a claim that a particular inquiry teacher has solved reasoning.

To compare BRAMASTRA with a historical branch, first freeze its exact source and verified runnable configuration. Adapt it only at the public task interface; document any missing ability, teacher, retrieval or evaluator difference. Use the same task instances, split definitions, answer scoring and real resource allowance. Compare both an equal-core/equal-data mechanism control where feasible and an end-system cost/quality comparison. Report parameter count, initialization priors and different inference computation.

Do not copy old branch headline percentages into a leaderboard with new tasks. If a branch cannot run the new protocol, mark it not evaluated on that protocol. “Better than all branches” requires comparable evidence for every named comparison; the present program has not produced it. The design target is a stronger integrated system, and the comparison harness is how that target becomes accountable.

## 6. Broadening the objective

Track a capability map rather than one AGI number: unfamiliar-mechanism acquisition, goal grounding, uncertainty calibration, useful inquiry, compositional transfer, long-horizon planning, memory use, language/code understanding, tool execution, retention and improvement across cycles. Store each cell as unsupported, failed, exploratory or confirmed under a linked protocol. No amount of progress in a single easy family fills other cells automatically.

Introduce new domains when the current system's failure analysis identifies missing representations, supervision or experience. Measure transfer against training a matched learner without the earlier domain experience. Distinguish additional compute from reusable knowledge. Require sustained improvements across independent cycles and held-out mechanisms before calling the system a useful self-improving learner. Keep open the possibility that the architecture needs substantial revision; the acceptance process exists to discover that efficiently.
