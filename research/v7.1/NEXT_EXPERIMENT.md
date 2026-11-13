# After V7.1 core: isolate the surface failure before retraining

Evidence basis: run `20c80f66e76e4b589aed2053afbddffc`, analyzed 2026-11-09. This is a development diagnostic plan informed by observed failures; the original protocol, model outputs and negative results remain unchanged. It is not a new sealed confirmation test.

The next instrument is implemented in `anra_v5/elendril/v71_followup.py`: `build_suite()`, `planned_contrasts()` and `audit(suite, train=actual_training_dataset)`. It reuses the sixteen original V7.1 base route worlds, where familiar-shape performance can be measured directly. The exact original baseline prompts are retained. No gold value is inserted into evidence or selected by the model actor.

Cross three input factors: original timestamps versus a +9000 shift preserving ties/order; all original rows versus only the relevant path edges; original versus reversed row serialization. For each of those eight combinations, test four explicitly different questions: the original two-hop question; its first-hop subproblem; a one-hop question starting at the public first intermediate; and a plain current-value lookup at that intermediate. The last three change the question and must not be called identical-prompt internal-state interventions. Each question has eight matched factor conditions: **32 cohorts, 512 records and 48 one-factor contrasts of 16 world pairs each**. Interpret the registered same-question factor contrasts; do not pool every cross-question/world comparison into causal evidence.

Competing explanations and falsifiers:

| Explanation | Discriminating observation at fixed weights | Limit |
|---|---|---|
| Timestamp/token-distribution sensitivity | Moving only timestamps causes reproducible paired changes while graph, answer and other surface factors stay fixed. No change weakens a simple timestamp account. | Identifies an input sensitivity, not its neural representation. |
| Dependence on distractor structure | Removing only irrelevant rows changes success on the same route and timestamp conditions. | Changed length/positions remain part of that intervention. A later length-matched distractor control would separate them. |
| Serialization dependence | Reversing only the public rows changes answers for the same state; check all three seed blocks. | Already demonstrated broadly in V7.1; this tests whether it explains the route-domain contrast. |
| Instruction/operation bottleneck | Plain lookup succeeds while the one-hop question over the same relevant edge fails. | Questions differ; this is a behavioral boundary, not proof of an internal module. |
| Residual composition difficulty | One-hop/direct lookup are reliable under a shared surface but two-hop remains weak. | Only then is a composition-specific intervention well motivated; still not a demonstrated internal cause. |

No optimizer updates are needed for this first phase. Evaluate the nine fixed core checkpoints with raw and grammar decoding, retain every action and prompt, and measure work. Recheck model/spec/tokenizer/source identities and finite tensors; verify the FULL ZIP and explicit checkpoint generations before inference. The 1.96 GB FULL archive has not been supplied locally. **No frozen results or cloud-execution readiness are claimed yet.** The original core notebook is preserved; its rerun would repeat training rather than execute this follow-up.

CPU qualification: five dataset tests plus four offline-audit tests pass. `FOLLOWUP_QUALIFICATION.json` records complete public-teacher correctness and exact disjointness against original training identity `7786244187963f75e6cda5b10e4d4e0d81927b7000154126fff538dea15539e5`. Some transformed variants map to the procedural train partition; this is disclosed rather than hidden, and exact comparison to actual training cases/states passes. Sixteen shared worlds and three initializations sharply limit generality.

A second, separately registered memory diagnostic should contrast equal-length neutral versus stale same-key memory histories and their presentation order, preserving the same current answer. It must ensure neutral rows actually survive the engineered goal-filtered retrieval step. This separates history-length/position effects from contradictory-binding effects before changing training objectives. That diagnostic is proposed, not implemented or run here. Larger models and a new supervision mechanism should compete only after these smaller tests identify a reproducible bottleneck and include retention/robustness checks that the present intermediate regimen fails.
