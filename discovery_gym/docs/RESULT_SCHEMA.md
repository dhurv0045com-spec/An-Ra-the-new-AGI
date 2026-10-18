# Result and Checkpoint Identity Schema

## Episode result

`anra-discovery-gym/episode-result/v1` is evaluator-owned. It binds one independent latent world to one policy run and includes public transcript events separately from evaluator recomputation. Required fields include policy identity, hidden-truth flag, action and experiment costs, invalid actions, budget overrun, final answer, success, predictions, commitments, attribution, direct outcome recomputation, and an independent legality replay. The receipt's protocol hash must equal the declared protocol hash, not a second hash over the already-hashed protocol object.

The public transcript is a projection of the result. It excludes structure ID, split, rule, target assignment, generation seed, expected evaluator values, and oracle status. A public result must never be used as proof of a hidden label.

## Experiment identity

Every receipt binds:

- source commit and source-closure hash;
- generator and evaluator source hashes;
- protocol hash;
- suite-manifest hash;
- policy ID and model ID;
- seed;
- checkpoint and resume-state hashes when a model exists.

## Checkpoint identity

DG0 emits an explicit identity with `model_id=none`, null checkpoint/resume/optimizer hashes, and `status=NOT_CREATED_DG0`. A future learner must replace every null with a content hash or a typed `NOT_APPLICABLE` reason. A filename, parameter count, or prose claim is not a checkpoint identity.

## Independence

The statistical unit is the latent world. Multiple targets, queries, renderings, or trajectories from one world do not increase the independent sample count for a causal or transfer claim.
