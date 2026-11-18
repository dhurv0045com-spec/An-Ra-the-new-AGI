# Avendi V8-Y2 frozen results

**The six Y1 checkpoints failed two-key lookup.** Each checkpoint solved **0/256 complete one-hop families and 0/256 complete three-hop families**. Each family asks about both keys before and after the destinations are exchanged. The archived first-token scores independently settle the primary result. In every family, at least one cell ranked the wrong endpoint above the correct one. No tie affected this conclusion.

**The clearest example is F00 seed 7.** It scored 510/1,024 exact one-hop answers, yet its first byte matched the last rendered row’s value on 1,020/1,024 cases. Changing the requested key left the first byte unchanged on 510/512 pairs. Exchanging the two destinations changed it on 508/512 pairs. This model responds to changed content while almost always overlooking which key the question names. This is a behavioral finding on development data. The internal circuit and the training history that produced it remain open questions.

## What ran

Owner evidence from run `avendi-frozen-63c10bbd2f`, audited 2026-11-19. The earlier CPU intake `avendi-frozen-0acbe87814` lacked weights. This later run restored both Y1 S3 arms at update 2,368 for seeds 7, 19 and 41. F00 is the historical baseline. F11 is the combined Y1 treatment. The owner run recorded 478.7 seconds. All six models saw the same 2,304 dev cases, giving 13,824 model presentations. They comprise 256 literal cases plus 256 four-cell families at each of two depths. Repeated model presentations share the underlying fixtures.

| Stage | Result | Evidence |
| --- | --- | --- |
| P0 fixture | Qualified | Local full-size CPU regeneration matched the archived P0 receipt. Public traversal solved 256/256 families at each depth. All seven other reference policies solved 0/256, although a query-blind policy gets 512/1,024 individual cases correct. Removing either required input created 512 contradictory identical-input pairs per intervention at each depth. |
| P1 restoration | Qualified in the owner run | Six matching parent checkpoint manifests. The run reports 1,024 original Y1 base cases per checkpoint with exact token agreement and maximum absolute first-logit difference 0. |
| P2 frozen diagnosis | Failed | Both arms missed all three gates. Each seed required at least 95% exact literal answers, 80% complete one-hop families and 70% complete three-hop families. F00 is the registered diagnostic anchor on a tie. |
| P3–P6 | Unexecuted | No acquisition comparison, transition intervention, conditional-action study, or correction result. There are no model confirmation or transfer scores in this run. |

## Scores by checkpoint

“Exact” is the run’s recorded equality to the correct byte followed by EOS. “A–P” is a local recomputation of whether the gold byte ranked first among the 16 possible answer letters. All literal cells are answerable. Each hop panel contains 1,024 cases per checkpoint.

| Arm | Seed | Literal exact /256 | Literal A–P /256 | One-hop exact /1,024 | One-hop A–P /1,024 | Three-hop exact /1,024 | Three-hop A–P /1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F00 | 7 | 61 | 106 | 510 | 512 | 160 | 160 |
| F00 | 19 | 28 | 83 | 2 | 283 | 105 | 107 |
| F00 | 41 | 13 | 89 | 0 | 208 | 145 | 145 |
| F11 | 7 | 8 | 16 | 409 | 419 | 52 | 102 |
| F11 | 19 | 1 | 59 | 313 | 360 | 103 | 103 |
| F11 | 41 | 41 | 63 | 412 | 458 | 155 | 155 |

F11 gets 1,134/3,072 individual one-hop cases correct against F00’s 512/3,072. The paired family advantage is zero at every seed and both depths. All 3,072 model-by-family evaluations miss even the weaker criterion of preferring the gold endpoint to the single rival in all four cells. Restricting the first byte to A–P also produces zero complete families. These results rule out suppressing abstention alone as a fix for this frozen greedy decoder. Zero observed family differences give a degenerate paired family bootstrap interval of [0, 0] on these data. That interval cannot establish equivalence on new seeds or tasks.

## What changes the models’ answers

These post hoc counts come from archived first logits and the reconstructed public rows. For each one-hop family, two pairs change the query alone and two change the destinations alone.

| Arm and seed | Same byte after query change /512 | Same byte after destination swap /512 | Byte equals last row value /1,024 | First byte is abstain /1,024 |
| --- | ---: | ---: | ---: | ---: |
| F00 7 | 510 | 4 | 1,020 | 0 |
| F00 19 | 510 | 503 | 0 | 1,015 |
| F00 41 | 508 | 484 | 0 | 992 |
| F11 7 | 501 | 44 | 832 | 22 |
| F11 19 | 492 | 51 | 616 | 211 |
| F11 41 | 506 | 55 | 823 | 137 |

The six models reach the same family score through different behaviors. Two F00 checkpoints mostly abstain on one-hop cases. The third repeatedly emits the last value. F11 usually changes its answer when values move, while a change to the requested key rarely changes its choice. At three hops even F11 seed 41 emits the last row’s value on 1,018/1,024 cases and keeps the same first byte on 512/512 query changes. Agreement with a position-based rule identifies a candidate shortcut. Swapping row order while keeping the graph and question fixed is the direct test of that rule.

**Literal readout depends strongly on position.** Each literal prompt has one non-null value in four possible positions, balanced at 64 cases per position per checkpoint. The question adds a marked-register instruction and uses three null distractors. Across all six checkpoints, correct first bytes by position were **0, 2, 77 and 73**, each out of 384 presentations. Within A–P the corresponding counts were **25, 89, 122 and 180**. F00 seed 7 got 61/64 correct with the value in row three and 0/192 elsewhere. F11 seed 41 got 41/64 with the value last and 0/192 elsewhere. These are different examples across positions, so a matched permutation is needed for a causal position claim.

**The largest observable literal error occurs before EOS.** The first action was abstention in 1,256/1,536 literal presentations. All 152 correct literal first bytes also counted as exact in the run’s summaries. Across both hop panels, the summaries attribute only five additional misses to EOS after a correct first byte. EOS following a *forced* answer remains unmeasured. A global inability to output a letter is too broad an explanation for the observed position- and seed-specific successes.

## Verification and evidence limits

| Claim | Audit status | How it was checked |
| --- | --- | --- |
| Archive bytes | Verified locally | Read all 20 ZIP members, checked inventory, lengths and member hashes against the manifest and SHA256SUMS. The package contains 19 manifest-listed files plus the manifest. |
| Oracle and fixture | Reproduced locally | Regenerated P0 at scale factor 1. The receipt, split custody checks and planted-collision detection matched. The six logit files matched the rebuilt dev case order exactly. |
| First actions and endpoint preferences | Recomputed locally | All six [2,304 × 260] float32 arrays were finite and matched their sidecar digests. Used case IDs, target bytes and public row order to check token winners, A–P ranks, signed margins and family results. No mismatches on first-logit-identifiable comparisons. |
| Recorded action replay | Unavailable for Y2 full sequences | The runner sets this status from P0. The exporter drops per-case raw tokens and retains first logits, family summaries and counts. These establish the zero-family upper bound for greedy generation, but cannot independently reconstruct exact case-level EOS sequences. |
| Checkpoint identity and restoration | Parent identity cross-checked locally. Restore reported by the owner run | All six hashes and update numbers match the original Y1 evidence. This ZIP contains no weights. The campaign records its source pin but suppresses source-check exceptions, so the recorded commit is not independent runtime attestation. |
| Regenerated Y1 per-case agreement | 6,144/6,144 reported. Reference side cross-checked locally | Rejoined the 1,024 parent base IDs per checkpoint and verified each archived Y1 first action against its original logits. The Avendi receipts report zero sequence mismatches and zero maximum logit difference at tolerance 1e-5. New Y1 parity vectors and weights are absent from this ZIP. |
| Training reproduction | Unavailable | This was a frozen evaluation. No local neural inference or training was performed during this audit. |

The observed artifact SHA-256 is `d30607012647c51f6e71c87ea2f21132dccb7e7dcfd738cb99d462aed9033775` (12,942,632 bytes). The recorded source is `428664245c11400464a050fce4422ade4b3a1b76`. The registry design SHA-256 is `13f8f46c9065115b9bbdecb2884edba2ff88a3bf9e01cba2afd613739c344fbd`. Full per-seed counts, checkpoint hashes, parent archive hash, parity receipts and audit scope are in [METRICS.json](METRICS.json).

## Next experiment for Gemini

1. **Run one frozen, matched row-permutation diagnostic first.** Keep each one-hop family’s facts, key names, values, timestamps, source labels and targets fixed. Render both row orders and both query keys before and after the destination swap. Score both row orders on the same checkpoint. A correct lookup changes when the query changes and remains correct when rows move. A last-row policy follows the moved value. Report exact family success across all eight cells, paired query flips, row-order flips and changes in signed gold-minus-rival logit. Begin with F00 seed 7 as the positive example of last-row agreement and F00 seed 19 as an abstention comparison. Preserve the original dev scores as the baseline.
2. **Separate literal format, position and action choice.** Rotate identical labeled rows through all four positions. On the same facts cross the native question with the extra marked-register wording. Run raw decoding and an A–P-only first step followed by a real model EOS step. Log target rank, chosen byte, next token, raw correctness and abstention per cell. A wording change that restores lookup implicates the renderer. Improvement only under forced answers identifies an action-choice contribution. Persistent row-order dependence isolates a behavioral target for training.
3. **Then run the registered P3 comparison.** Use three paired fresh S3 initializations (7, 19, 41), cross-entropy versus the same cross-entropy plus the registered endpoint-margin loss. Hold initial parameters, 1,024 training families, four-cell batches, order, updates and compute constant. Monitor train and held-out families at 0, 64, 256, 512 and 1,024 updates. Require all three complete seed pairs, lock on dev, then open confirmation once. The current job evaluates confirmation inside each training job and its gate accepts the intersection of seed sets. Separate selection from confirmation and enforce the full seed set before making a qualification claim. Margin support requires at least +10 percentage points mean dev family gain, positive at every seed, at least 80% complete confirm families per treatment seed and at most a 5-point paired literal drop on dev. If both qualify without specific margin support, use CE. If train improves and dev stays at floor, inspect held-out role and row-order dependence before moving to depth.
4. **Keep the next export small and independently replayable.** For each scored case retain its ID, prompt hash, checkpoint/decoder identity, up to three raw token IDs and their actual length. Keep the existing compressed first logits in EVIDENCE, the short report and metrics in SUMMARY, and weights in FULL. Recompute replay from the saved tokens. A corrupted EOS and an exchanged case must each fail the verifier. The present run provides no neural evidence for P4–P6 or for a performance benefit attributable to Planthir.

The registered router names P3 after the joint literal and lookup failure. The written design prioritizes checking the literal renderer in this branch. The two frozen diagnostics above address that dependency and leave the registered P3 comparison intact. For actors continuing this branch, Y1 discovery items 05–08 and 10 now have new frozen development evidence. Item 21 still needs full Y2 action replay. Training, transfer, retention and correction hypotheses remain open. [Registered campaign and discovery ledger](../../../research/avendi/GEMINI_AVENDI_V8_Y2.md) · [Previous Y1 results](../../V8-Y1/y1-all-9c80885013/RESULTS.md)
