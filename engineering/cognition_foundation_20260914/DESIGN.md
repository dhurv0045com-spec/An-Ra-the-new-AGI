# Evidence-conditioned cognition and a bounded creative extension

## 1. Public state is a data contract

Define a versioned `PublicState` with goal, ordered received events, admitted evidence, legal action descriptors and remaining budgets. Preserve original immutable events separately from derived evidence. Keep private mechanism parameters, oracle answers and unreceived observations outside this object. Hash the entire canonical public state, not the goal alone. Identical public observations under different private worlds must yield identical prompts; changing a relevant public observation must change the prompt.

Replace overlapping abbreviations with a bijective versioned encoding. A variable identifier and its value must occupy distinct fields. Preserve types: integer 7, string "7", false, null and missing are distinct. Support every field emitted by the live environments and tools. Unknown fields must either survive in an extension map or trigger explicit schema refusal; silently dropping them is unacceptable. Round-trip nested results, tool arguments, errors, identifiers and contradictory evidence. Length-based string slicing is not semantic compression.

Train and inference must call the same renderer and action codec. Record their schema identities in prepared data, checkpoints and evaluation receipts. Existing data produced by a lossy renderer cannot silently remain valid under a repaired schema. Invalidate or regenerate affected artifacts in new directories. Do not mix incompatible training targets and inference prompts.

Context admission must reserve space for goal, budgets, response boundary and required decision evidence before accepting optional material. If the necessary information cannot fit, return an explicit context-exhausted outcome with omitted IDs. A deterministic summary is a declared prior and must retain source IDs and typed facts. It must not invent a resolved answer. Do not solve overflow by cutting JSON bytes. Measure the largest legal action and prediction encodings against the existing 24-token response envelope. If impossible, design compact typed output codes with a declared vocabulary and matching training targets; any envelope change requires an explicit K8 protocol amendment and recalibration before training.

## 2. Evidence with subjects, time and support

An evidence record needs `record_id`, `subject`, `predicate`, typed `value`, `valid_time`, `source_event_id`, `status`, and optional `supersedes`. Derive subject and predicate from the received observation schema, not from event kind alone. For example, reading x=7 and y=8 yields different subjects. A door changing from closed at t0 to open at t1 is a state transition. Two incompatible assertions about the same subject/predicate at the same relevant time constitute a conflict. An unknown timestamp does not license an invented simultaneity judgment.

Retain immutable history. Derive active, superseded, conflicting or unknown status without deleting inconvenient evidence. Distinguish exclusive single-valued predicates from multi-valued sets. Receiving another item in a container must not necessarily contradict the first item. Duplicate observations must not count as independent support. Use actual environment time semantics and disclose them as engineered structure.

The workspace is a representation aid, not an oracle. The model remains responsible for action selection. Any deterministic selection or conflict rule must be recorded as a prior so a positive result can be attributed correctly.

## 3. Conditional world prediction

The predictor consumes the rendered public state, a typed candidate action and an explicitly marked imagined prefix. Its output is a typed predicted observation or state delta, uncertainty, terminal prediction and optional value. Validate numerical finiteness, probabilities in [0,1], allowed keys, and semantic applicability. A parse failure remains a failed prediction; a neutral fallback cannot become a valid confident prediction by default.

The production learner needs targets compatible with this interface. Trace the current world-loss targets through the codec, training window and decoder/head to the E2 consumer. If the current objective trains a different representation, build an explicit compatible adapter or repair the target construction while preserving the registered objective weights. A JSON instruction alone does not train an initially random decoder to predict transitions. Supply a backward-only proof that valid transition targets reach the intended parameters and masked/private fields do not enter model inputs. This proves connectivity, not learned quality.

## 4. Genuine bounded search

Represent an imagined node as parent ID, root action ID, full public-state hash, imagined-state hash, action, predicted delta, uncertainty, accumulated declared cost, remaining budgets and origin. Apply the first prediction to a separate imagined state before requesting the second. Recompute subsequent legal actions only from that imagined state and public rules. Never mutate real history with predictions. If valid imagined transition semantics are unavailable, report a one-step planner; do not advertise two-step search.

Avoid root starvation. With eight nodes and up to eight roots, evaluate all roots once before considering children. If more than eight actions exist, use a registered public-only candidate rule and record excluded action IDs. For fewer roots, distribute remaining expansions using deterministic ties and uncertainty-aware priorities. Every predictor call consumes the real call budget, including failed calls. Every node consumes the node budget. An imagined action reduces the imagined path budget, not the real episode budget until execution. Keep the existing four real inquiries/tools plus submission and sixteen-call limits.

Score paths using explicit terminal success predictions and measured/declared action costs. Avoid adding unrelated probabilities across time. Bind the executed action to its selected root, selected path and prediction IDs. Reordering equivalent legal action lists must not change the winner except through documented ties or candidate selection. Test a delayed-reward example where the best first action depends on its predicted second state; a scripted predictor is appropriate for verifying search mechanics, but its success is fixture evidence.

## 5. Honest calibration and attribution

Join predictions to realized outcomes by episode, decision, root/action and horizon. A prediction of immediate feedback must be scored against immediate feedback. An episode-success prediction is scored against terminal success only with its forecast horizon documented. Do not select the first node opportunistically. Store raw prediction/outcome pairs, eligible counts and exclusions. Report Brier score for success probabilities and a clearly named error measure for structured feedback. Report fallback and invalid-prediction rates separately, without silently dropping their failed episodes from task success.

Keep model-origin, trained-parent lineage, fixture origin and demonstrated capability as separate fields. One model-origin event cannot certify every event in an aggregate. Retain matched mechanism IDs, seed, checkpoint and treatment identities in each row. Compare paired modes over the same admitted episodes; context or parser failures count as outcomes, not reasons to replace difficult tasks.

## 6. Creative hypothesis: evidence repair before commitment

After the foundation passes, investigate whether the learner can recognize a specific missing or conflicting fact and choose a cheap action that repairs its decision state. The proposed mechanism is a small evidence-repair controller sharing the randomly initialized core. It predicts a decision dependency: which received fact supports the planned answer, what alternative remains plausible, and which legal observation could distinguish the alternatives. This is a bounded hypothesis about cognition, not a claim that a new module yields AGI.

Use a typed `DecisionCertificate` containing candidate action ID, supporting evidence IDs, unresolved proposition IDs, proposed discriminating query ID and predicted repair benefit. A structural checker validates references and legality only; it does not tell the model the correct answer. A prediction unsupported by received evidence stays hypothetical. Do not require verbose natural-language rationales or treat plausible explanations as correctness evidence.

Construct matched training examples from public trajectories: sufficient evidence; one critical observation withheld; irrelevant observation withheld; contradictory same-time observation; normal state transition; goal changed while evidence stays fixed. Teachers can label whether a query distinguishes finite hidden hypotheses, but those labels are training supervision and never an inference oracle. Group splits by mechanism equivalence to prevent rephrased copies crossing into confirmation data.

The proposed auxiliary target is the next useful observation or stop decision, plus support-ID selection. Start with supervised targets in a separate future protocol, not an unregistered addition to K8's frozen B objective. Learn to stop when evidence is sufficient; do not reward query count. Proposed utility is verified task improvement minus observation cost and old-skill degradation, evaluated against equal-budget baselines.

For eventual evaluation compare the repaired workspace policy with the controller enabled and disabled, plus a random-query control. Include source-ID permutation with remapping, irrelevant-evidence injection, missing-evidence pairs, and goal swaps. Success requires selective useful inquiry and maintained answer accuracy, not longer traces. A controller that always requests another observation fails the sufficiency test. A controller that reads private labels fails the experiment.

Connection to RSI: only after measured cognition exists, allow the proposer to choose among already registered repair curricula or representation variants. Every candidate starts from the same permitted parent, has measured trial costs, and is compared on fresh tasks plus protected retention. Keep the evaluator and budgets outside the candidate's editable scope. Improved self-report, extra training, or hand-picked demonstrations are not recursive improvement. Defer this extension until baseline evidence justifies using a future allocation; do not crowd the present K8 run with another arm.
