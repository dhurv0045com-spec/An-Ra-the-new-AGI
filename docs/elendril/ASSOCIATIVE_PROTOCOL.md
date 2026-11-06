# ELD-DEV-002 frozen protocol

ELD-DEV-001 found zero query-swap success even on training examples. This screen asks whether same-update counterfactual exposure, explicit answer ranking, and a learned causal memory read head can recover the signal. The read head has no fixture parser, entity table, gold candidates or externally written facts. It can copy from prior tokens using contextual addresses and a learned 3 × 3 pattern of token equalities. A learned gate mixes that distribution with the dense next-token distribution. The token equality operation is an architectural inductive bias.

## Matched contrasts

| Arm | Training method | Interpretation against previous arm |
| --- | --- | --- |
| Shuffled CE | Dense, shuffled within each exposure block, answer + EOS CE | Exposure-matched control |
| Paired CE | Dense, both views in the same update, answer + EOS CE | Batch co-occurrence |
| Paired contrast | Same paired stream, CE + 0.5 times softplus(1 + distractor logit − gold logit) | Extra query-conditioned margin |
| Associative | Same paired contrast with learned read head and identical initial dense-core weights | Architecture addition at matched examples and updates |
| Associative permuted | Same architecture, recipe and row order; within-world answer assignments flipped by fixed seeded coins | Dependence on coherent supervision |

The head adds parameters and work. Record actual cost and treat the architecture contrast as data/update-matched. Equal training time, FLOPs or parameter count would require a separate comparison.

Run seeds 7, 19 and 41 on one CPU thread, one process at a time. Each 21-update exposure block visits every row exactly twice, and the experiment runs 16 blocks for 336 updates per arm. Paired variants use identical index streams for a seed. Hold LR, batch size, context, masks, EOS supervision, generation cap, and data splits fixed. Use 3 new tokens per example. Run base development at 0, 84, 168, 252 and 336. Evaluate all four stress panels once at 336. A run stops on nonfinite arithmetic, contract failure, system pressure or a 180-second process limit. Keep prior checkpoints.

## Decision rules

The primary endpoint is both queries correct with valid EOS on the original development worlds. Check acquisition on the clean training labels and report training and development exact EOS. For any stage to clear formation, require at least 60% both-correct in training, 60% both-correct in development, and 75% exact EOS on development for every seed. For a gain attributable to a stage in this screen, require at least 20 percentage points of development both-correct improvement over the previous arm in every seed. For the associative head require at least 20 points over the permuted-label arm in each seed. A failed manipulation check invalidates its contrast. Zero both-correct on neutralized queries is required.

The stress panels preserve the seven development worlds. Alias replaces both queried entity symbols and the fact separator. Distractor inserts an unqueried third fact at a deterministic varied position. Query-neutral replaces both query symbols in a pair with the same symbol while preserving scoring truth. The first two panels test specific distribution shifts and share latent worlds with the primary panel. Report per-seed counts for each. Seven worlds and fixed render transformations support a development screen. Future broad claims require a larger independent instrument, a matched-capacity architecture comparator, and controlled retention and forward-transfer experiments.
