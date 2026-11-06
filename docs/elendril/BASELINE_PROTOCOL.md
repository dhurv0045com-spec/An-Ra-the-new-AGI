# ELD-DEV-001 frozen development protocol

The experiment asks whether a fresh Cymek-derived dense decoder can learn the serializer's copy control and answer counterfactual binding queries on disjoint two-value worlds. A fresh-initialization snapshot establishes the starting floor. A world-wise label-permutation arm tests whether the coherent query rule matters. Fixed first/last and random choices screen positional success. These are development observations for one finite symbolic surface.

## Arms and budget

Run copy, clean binding, and permuted binding at seeds 7, 19 and 41. The copy arm is a separate mechanical positive control. Compare clean and permuted binding within seed. The binding pair uses identical initialization, row indices and prompts, vocabulary, optimizer, gradient bound, generation limit, schedule and number of updates. Independently flip or preserve the two training labels with one seeded fair coin per latent world using seed + 100000. This preserves the frequency of each label within each world and exposes the model to inconsistent world-to-world query rules. Measure every arm on clean held-out worlds. Accidental finite-sample alignment of the corrupt rule is possible.

Use CPU FP32 with one Torch intra-op thread, 350 updates per arm, batch size 16, constant AdamW LR 0.003 and pre-clip limit 1.0. Evaluate at updates 0, 50, 150 and 350 using greedy free generation with an EOS stop and a maximum of five new tokens for copy and three for binding. Execute one process at a time. Stop on a nonfinite update, contract violation, checkpoint error, system pressure, or three minutes in any one process. A stopped arm remains incomplete. Fixed endpoints govern interpretation.

## Endpoints and falsifiers

- Copy floor gate. Every seed must achieve at least 80% development exact plus valid EOS and improve by at least 70 percentage points from initialization. The format-copy control is expected to solve this family.
- Binding gate. Every clean seed must reach at least 75% exact plus valid EOS and at least 60% both-correct query swaps. Every matched clean arm must exceed its permuted control by at least 20 percentage points on both-correct query swaps. First-fact and last-fact controls should score exactly 50% exact and 0% both-correct.
- Failure to meet a gate means this configuration failed qualification on the specified surface and budget. Interpret no retention or forward transfer contrast from a failed learnability gate.
- Report per-seed exact and both-correct counts, all-view cluster counts, token accuracy and prefix resolution. Views share worlds, and all seeds share one development split. Report seed ranges and paired deltas descriptively. The seven binding development worlds and one fixed rendering support no population or confirmatory claim.

Record real processed non-BOS, nonpadding tokens separately from answer+EOS targets. Preserve the individual greedy token streams, per-update loss and clipping values, sample indices, checkpoint identities, elapsed CPU/wall times, and evaluated forward positions. Checkpoint manifests are hashes of local transaction inventories. Score from receipt predictions in a separate process.
