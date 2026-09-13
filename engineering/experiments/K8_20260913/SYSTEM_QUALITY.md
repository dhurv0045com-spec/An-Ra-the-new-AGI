# Multiple themes, one accountable learner

This document explains the engineering intent behind K8 and strengthens acceptance of I01–I06. [experiment.md](../../../experiment.md) remains authoritative for treatments, thresholds, data sizes and time. These themes introduce no extra training arms. Proposed follow-up questions at the end require a later allocation.

## 1. Representation: preserve what the learner actually knows

Use one canonical public serialization for the goal, observations, previous actions, tool feedback and any workspace. Private generator state can establish ground truth in a verifier, but cannot enter model tokens or candidate construction through an indirect helper. Keep public rendering separate from verifier access in the API, rather than trusting callers to remove hidden keys afterward.

A prepared example needs a stable episode identity, mechanism/split identity, prefix tokens, supervised spans, candidate identities, eligibility masks and provenance. Variable-length padding is not evidence. Mask it explicitly. Train-time and inference-time serialization must agree, including EOS and the location of candidate actions relative to history. If context overflows, record and reject or apply the prespecified policy; silently dropping the contradictory observation changes the task.

Acceptance evidence: render the same public state through preparation and inference and compare bytes; perturb private state while keeping public observations fixed and verify unchanged model inputs; vary padding and verify unchanged eligible reductions. Publicly identical states with different hidden answers are legitimate uncertainty, not necessarily broken data. Their grouping must prevent cross-split leakage while preserving the intended task distribution.

## 2. Learning: optimize the quantity the configuration claims

For objective j over an accumulation window, define a differentiable numerator S_j as the sum of eligible losses and N_j as its eligible count. The intended loss is the sum of lambda_j * S_j / N_j for active terms. A mean of microbatch means generally gives a different result when eligible counts vary. Count each objective separately; do not divide a sparse action objective by all padded tokens.

Compute window denominators before backward, or use an equivalent method that demonstrably preserves this formula without retaining an unbounded graph. If N_j is zero, the term contributes no loss and is recorded as ineligible. If N_j is positive but the term is missing, nonfinite or detached, fail the update. If every term is ineligible, record the exposure without inventing a successful optimizer step.

Answer accuracy, action quality, world prediction and value estimates have distinct targets. A scalar value label must derive from the declared return convention, not a hidden solution copied into a float. World loss supervises the declared outcome span. Pair loss uses the prespecified margin and score normalization. Do not let convenient tensor shapes redefine the mathematics.

Acceptance evidence: compare a small window processed together with the same examples split into unequal microbatches, with stochastic layers disabled for that equivalence check. Compare gradients on named parameters, not only the final scalar. Local checks may use backward but no optimizer step. E0 verifies actual intended parameter changes and finite AMP behavior within its allocation.

## 3. Cognition: information must change decisions

The cognitive hypothesis is that retaining and interpreting observations helps choose useful actions under uncertainty. Additional text, a belief dictionary or a planning function is not itself evidence. Require a trace from current public history to checkpoint scores to selected action to resulting observation. Distinguish an action proposed in imagination from an action actually executed.

Use the existing E2 contradictions, complementary queries and goal swaps to test information use. The same observation can warrant different actions under different goals; the same goal can require revision after contradictory evidence. Aggregate success alone can conceal a policy that always asks the most common query. Report paired outcomes and action changes for these existing cases, including cases where a change was unnecessary or harmful.

A reference Bayesian updater provides a control and a diagnostic ceiling in its declared finite setting. It must not lend private beliefs to the learned arm. Likewise a workspace must contain allowed evidence or learned state, not a teacher's solved answer. Report serialization and context cost so a workspace gain is not silently attributed to reasoning when it merely receives additional information.

## 4. World models and planning: predictions are uncertain branches

Represent a planning node by public state identity, goal identity, remaining real and imagined budgets, depth and branch provenance. A predicted outcome is not an observation. A planner may reason over it, but may not write it into the real episode history as if a tool returned it.

Finite-support prediction needs deduplicated public outcomes and normalized scores under the declared support. Its probabilities are conditional on that support. If support omits possible outcomes, calibration within it does not establish an accurate open-world model. Record how support was constructed and ensure hidden truth did not choose it.

Keep the existing E2 depth/node/call limits. Count batching honestly: a single model invocation evaluating many candidates can still consume substantial tokens. Report real actions, imagined nodes, model calls and processed tokens separately. The planner must not quietly obtain a larger action budget than the policy comparator. Invalid or exhausted branches terminate predictably and remain visible in the trace.

## 5. Tool competence: reward verified state changes

Use typed actions and bounded arguments for the declared finite tools. Give each invocation a request ID, input artifact identity, result identity, status and measured cost. A repeated request after a timeout must not accidentally duplicate a write. Resolve outputs inside the disposable episode directory and validate resulting content through the independent task verifier.

The model's text saying it wrote a correct result is not task completion. Completion follows the tool receipt and verifier. Preserve errors and recovery attempts as observations. Ensure a failed action cannot receive a success label merely because the tool returned parseable JSON. Tool names alone also do not demonstrate transfer: held-out compositions must require combining operations in configurations excluded from training by mechanism grouping.

No new repair-learning arm is added here. Existing tool-phase traces should identify invalid arguments, execution failures, incorrect compositions and successful recovery separately. This tells the next engineer whether to improve representation, action supervision or environment feedback.

## 6. Continual learning: new competence must survive comparison with old competence

E3 asks about a specified replay-and-objective package, not every possible continual-learning method. Preserve its T0/T1 distinction. Record actual new-task and replay exposure, sampling identities and protected-family outcomes. Equal optimizer steps do not imply equal examples of each type.

The protected evaluator must not write examples, gradients or outcome-driven weights into the learner. Select no checkpoint using protected results unless that selection was explicitly part of the frozen protocol. Report the declared endpoint and all required checkpoints; retain regressions even when new-tool performance improves.

A useful interpretation separates acquisition, retention and cost. If acquisition improves while retention deteriorates, do not average them into an unexplained success score. If replay preserves old tasks but prevents new learning, the package has a different limitation. The next decision follows these separate quantities.

## 7. Architecture plasticity: distinguish changing execution from inventing architecture

E4 reuses two existing blocks through learned gates. It tests whether this predefined change in execution helps under the declared comparison. It does not test unconstrained architecture discovery. The proposed zero-gate migration must preserve parent behavior in a controlled deterministic forward check, and the parameter inventory must show that blocks share storage rather than being cloned accidentally.

At zero gates, inspect the gate gradient; do not demand a nonzero gradient through an inactive residual contribution to reused weights. At nonzero gates, verify that the additional computation can reach the shared blocks. Use separate named checks to avoid declaring the entire construction broken or correct from one indiscriminate gradient assertion. S0 must carry the declared disabled gate slots without executing the treatment inadvertently.

Optimizer parameter registration must include each shared parameter once. Restore must reconstruct sharing, gate values and architecture identity before optimizer state is applied. Time and token counters include additional block execution even when parameter count barely changes. A result can improve accuracy while losing efficiency; report both.

## 8. RSI: let selection affect a successor, then test selection itself

The essential chain is measured training archive -> P0 context -> generated method -> validated dispatch -> P1 update -> fresh task decisions. Every arrow requires an identity and a timestamp/event ordering constraint. A method-origin string assigned by the host does not establish a learned proposal.

Keep three roles distinct: the fixed adaptation anchor used for method trials, the proposer being trained to select methods, and the successor proposer produced by the selected update. Accidentally updating the anchor makes later methods benefit from earlier training and destroys the comparison. Accidentally scoring P1 on its training archive measures memorization of method outcomes.

P_fixed controls the extra archive and training time; P0 controls the absence of the successor update. Capture all fresh-task method choices before confirmation outcomes become available. Preserve invalid proposals and failed trials with their charged costs. Apply an explicit predeclared invalid-proposal policy; do not secretly regenerate until a favorable method appears. If the method grammar forces validity, record that prior and still test origin and dispatch.

The improvement hypothesis is falsifiable: P1 may fail to outperform P_fixed, may select expensive unhelpful methods, or may regress on protected tasks. Those are real outcomes. Success would support a bounded learned method-selection loop in these task families; it would not establish sustained recursive gains, unrestricted source rewriting or general intelligence.

## 9. Runtime: recover the experiment, not just the weights

A checkpoint includes the model, optimizer, scaler, sampler position, random states and the relevant controller/proposer state. A campaign also needs the persistent allocation and job lineage. Restoring one without the other can repeat examples, duplicate trials or erase consumed time.

Publish checkpoint completeness only after the payload is durably written and its manifest/hash is committed. An incomplete latest checkpoint must not replace the last complete restore point. Recovery must validate source, data, configuration and architecture compatibility before resuming. Test interruption at reservation, checkpoint publication and result-commit boundaries using simulated clocks and small fixtures locally.

The supervisor owns accounting; workers own their bounded jobs. A worker crash remains charged and is not a new free run. A live second supervisor must be refused. Export is a required phase within the same allowance. Missing payloads are reported as missing rather than replaced with manifest-only claims of recoverability.

## 10. What follows the first campaign

Choose the next milestone from the failure pattern, not the most impressive example. If E1 fails to learn, inspect target correctness, actual exposure and optimization before expanding architecture. If E1 learns but E2 fails, examine observation use and planner model error. If E3 forgets, distinguish replay availability from replay effectiveness. If E4 helps only with substantially more computation, design a later cost-matched comparison. If P1 only beats P0, investigate whether ordinary meta-training explains the gain before claiming RSI.

Broader language understanding, long-horizon memory, open-ended tool composition, learned experiment design and multiple generations of retained improvement remain important research themes. They require new datasets, controls and allocations after this campaign establishes trustworthy foundations. Record a ranked follow-up list with the observation that motivates each experiment, its falsifying outcome and its cheapest adequate resource estimate. Do not run these extensions inside K8 by stealing time from controls or export.
