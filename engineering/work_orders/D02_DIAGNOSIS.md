# D02 diagnosis before additional training

Chief assignment, 2026-09-09. Read the D02 execution packet and chief result. The observed learned accuracy contrasts were 0/92 and +2/92. Determine whether the weak difference begins in the teacher intervention or in the trained learner. Do not tune against these development outcomes.

## Ownership and allowance

One Luna executor owns inquiry diagnostic source/tests, `engineering/reports/W04/`, and a new immutable `artifacts/bramastra/d02_diagnosis_*` directory. Existing experiment artifacts and discovery source are read-only. CPU only, two threads, at most two minutes of diagnostic computation. No training, accelerator, network, Git or additional agents under this packet. Implementation effort is driven by the acceptance criteria, not an elapsed-time target.

## Part A: verify intervention strength without a learner

Reconstruct the frozen 256-episode teaching datasets for seeds 801/802 from the same 73 training worlds. Verify both tensor identities against their recorded training manifests before deriving conclusions. If source or data identity differs, stop and report it; do not call a newly generated dataset the historical dataset.

Group rows by remaining horizon, keeping terminal states out of policy-supervision denominators. For each group report row count, positive-gain support for each teacher, exact best-action-set agreement, partial overlap, disjoint best sets, and the fraction where only depth-two provides supervision. Apply the trainer's actual tie tolerance and legal masks. A numerical gain difference without a changed target distribution is not a changed imitation target.

For initial states with changed supervision, independently enumerate the two-query observation branches and report expected terminal target entropy under: uniform legal inquiry; fixed coverage; one-step teacher with uniform ties and uniform legal fallback at zero gain; depth-two teacher with the same tie/fallback rules. Teacher continuation must use the observed posterior. This is a privileged training-population diagnostic, never learned evaluation evidence.

Do not enumerate paths through hidden answers and call them available observations. Weight branches by their actual training-prior mass; confirm branch probabilities sum to one and posterior subsets partition the prior. Include the four-hypothesis parity construction as an independent exact check, plus an irrelevant-query example. Report query costs separately; this frozen experiment has zero cost.

## Part B: inspect historical learned behavior

From saved raw evaluation rows, verify legal, nonrepeated, target-excluding action sequences. Compare the two learned arms' complete query sequences on each paired world/target. Report unchanged sequences, changed sequences, and correctness/Brier differences within each group, separately by seed and family. Inspect probabilities even where binary correctness ties. These are descriptive associations, not causal decompositions: shared predictor weights also changed.

The original campaign did not save model checkpoints. Therefore policy imitation accuracy and counterfactual predictor re-evaluation cannot be recovered from its raw prediction rows. Explicitly mark these unavailable. Do not silently retrain to fill them in. If Part A shows a meaningful intervention but Part B remains unresolved, propose a separately frozen follow-up that saves checkpoints and training-state imitation diagnostics for chief approval.

## Acceptance and chief decision

Provide exact commands, source/data identities, raw compact counts, independently checked branch expectations and a handoff. Negative findings are valid. If optimal teacher targets rarely differ, redesign the task distribution before scaling. If targets differ materially but learned action sequences do not, the next test concerns imitation. If sequences differ without useful predictive gains, investigate information use and representation. None of these cases alone establishes a route to AGI.
