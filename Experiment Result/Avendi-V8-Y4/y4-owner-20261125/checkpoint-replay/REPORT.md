# Avendi Y4 · Frozen checkpoint replay

**26 November 2026.** The Y4 span parser and categorical graph executor work together on familiar byte prompts. Across five seeds they solved **632/640 complete counterfactual families (98.75%)**. The same byte model's direct answer path solved **0/640**. On the same graphs rendered with brackets, a different arrow and a different row separator, the connected system solved **20/640 (3.13%)**. Replacing just the learned parser with the public reference parser restored **640/640**. This localizes two distinct problems: selecting and emitting the answer on familiar prompts, and recovering rows from unfamiliar syntax. All measurements use existing weights.

## Four ways of solving exactly the same prompts

Each model–case evaluation starts with a query, a graph, and a public step count. An eight-cell family only passes if every query, endpoint assignment and presentation order is correct. Each panel has 1,024 distinct cases evaluated by five matched seed pairs.

| Parser → executor | Familiar correct cases / 5,120 | Familiar complete families / 640 | Bracketed correct cases / 5,120 | Bracketed complete families / 640 |
|---|---:|---:|---:|---:|
| Oracle → oracle | 5,120 | 640 | 5,120 | 640 |
| Oracle → frozen true-step categorical | 5,120 | 640 | 5,120 | 640 |
| Frozen byte spans → oracle | 5,102 | 632 | 429 | 20 |
| **Frozen byte spans → frozen true-step categorical** | **5,102** | **632** | **440** | **20** |
| Frozen byte spans → frozen IIT categorical | 5,102 | 632 | 441 | 20 |
| Direct greedy byte answer through EOS | 1,439 | 0 | 1,139 | 0 |

**Where the first errors occur.** On familiar prompts, 5,102 parses were exact and every one produced the correct executor answer. All remaining 18 parses were invalid at the typed interface. The bridge answered correctly on 3,667 cases where the same byte model's direct output failed. For example, seed 7 read `E3->G6 ; G4->B6` and the query `G4` correctly. Its direct decoder generated `G6` followed by EOS. Passing its extracted rows and query to the executor returned `B6`. This places the familiar-format gap after extracting the spans in this assembled system; it does not distinguish the native model's attention, hidden binding and output head as causes.

On bracketed prompts the parser recovered all rows and the queried entity exactly on **421/5,120** evaluations. All 421 then gave the correct endpoint. Some imperfect parses still produced the correct answer (19 extra with the true-step executor). The row list was exact on 452 evaluations, while the query alone was exact on 3,506. The parser produced a usable typed graph on only 473. At depths 2–4 the bracketed panel had **zero exact parses** and **zero complete families**. Seed 7, for example, read `[E3=>G6] [G4=>B6]` as two rows ending in `6` and extracted the query as `3` instead of `E3`. Returning a correct endpoint occasionally on a malformed parse earns no parsing credit.

| Depth | Familiar complete families with learned bridge / 160 | Bracketed complete families with learned bridge / 160 | Direct byte complete families, either format |
|---|---:|---:|---:|
| 1 | 158 | 20 | 0 |
| 2 | 160 | 0 | 0 |
| 3 | 160 | 0 | 0 |
| 4 | 154 | 0 | 0 |

Seed-level familiar family counts were **127, 125, 128, 127, 125** out of 128. Bracketed counts were **3, 0, 11, 6, 0**, in seed order 7, 19, 41, 73, 101. The parser and its historical sham have **identical weights in all 56 state-dictionary tensors at every seed**. The earlier training run therefore gives no separate estimate for the span-supervision treatment against that sham.

Before the new tests, I replayed 32 archived dev cases per byte checkpoint. All **160/160** first-byte argmax decisions and **160/160** stored exact-response flags agreed. The largest absolute difference across their 260-way first logits was below **3.82 × 10⁻⁶**. The original evidence retained the first logits and exact-response flags rather than the full generated token sequence, so this comparison cannot certify exact historical sequence identity. [Five-seed measurements and cases](BRIDGE.json) · [Totals and paired comparisons](SUMMARY.json)

## Which punctuation broke the parser?

A follow-up crossed three independent edits on the same 16 graphs (four per depth, 128 cases per seed). The public reference parser accepts every form. “Space only” replaces the ` ; ` separator with a single space. Each number counts exact recovery of **all** rows and the query; chance endpoint hits cannot increase it.

| Edge arrow | Brackets around each row | Space only | Exact parses / 640 | Complete families / 80 |
|---|---|---|---:|---:|
| `->` | No | No | 640 | 80 |
| `=>` | No | No | 77 | 4 |
| `->` | Yes | No | 172 | 10 |
| `->` | No | Yes | 105 | 8 |
| `->` | Yes | Yes | 102 | 7 |
| `=>` | No | Yes | 17 | 0 |
| `=>` | Yes | No | 28 | 0 |
| `=>` | Yes | Yes | 45 | 2 |

Even the single-character arrow edit dropped exact parsing from 640/640 to 77/640. Each single edit independently reduced parsing accuracy on the same graphs. The differences across combinations are descriptive on this small nested sample. [Renderer measurements](RENDERER.json) · [Factorial replay code](../../../../tools/avendi_y4_renderer_ablation.py)

## What to run next

Keep the frozen true-step executor. Use a short parser-only experiment with identical graph families and two training arms. The first repeats the familiar rendering; the second gets equal exposure to varied arrows, brackets and row delimiters. Balance query, endpoint assignment and presentation order. Lock out whole renderer combinations until evaluation. Record exact row recovery, exact query recovery and eight-cell bridge success on both the original and held-out forms. This tests whether varying the training syntax can repair the observed bottleneck without sacrificing the working format. The direct byte decoder remains a separate benchmark. A later properly labeled donor-training comparison is warranted only if the question is the *incremental effect of IIT*, which these frozen runs do not estimate.

## What the checkpoints establish

The FULL archive contains 120 trained lineages. Every LATEST pointer matched the checkpoint digest in its job receipt. The 511 payloads shared with the previously audited EVIDENCE archive matched byte-for-byte by digest. I verified 1,718 listed FULL manifest entries, 1,717 checksum-list entries and the per-object component digests. The 4,503,835,147-byte FULL has SHA-256 `7d006b2e72c18ed5a763d873cb23c5bba053e6ce8426c79ad13cb1857c2bb4a1`. Weight loading used strict state-dictionary matching and PyTorch's weights-only loader. The six Y1 historical lineages remain a separate missing input. [Custody record](CUSTODY.json)

I used CPU inference with frozen weights and five matched seeds (7, 19, 41, 73, 101). The new diagnostic set contains 32 families per depth at one through four hops, with eight matched cases per family. The same 128 graphs were rendered twice. One rendering uses familiar `A0->B2 ; …` syntax; the other uses `[A0=>B2] […]`. The key, query, target, depth, row order and counterfactual assignments remain paired across renderings. The sampler excludes unordered role quadruples from the known byte-model split pools and typed train/confirmation pools. Shared graph motifs, alphabet and training corpus limit the generalization claim. The five seed evaluations reuse the same cases; the denominators below count model–case or model–family evaluations. These are fresh post-run diagnostics.

The four crossings use the same public bytes. The oracle parser decodes the graph; the learned parser extracts row key, value and query spans with the saved A-pointer checkpoint. Both routes use the **publicly supplied hop count**. The oracle executor follows exact edges. The learned executor is the saved C categorical model paired by seed. Invalid IDs, empty rows and duplicate keys fail the learned interface. Endpoint correctness counts all cases including invalid parses. The direct byte score separately requires the correct two-byte entity followed by EOS. The bridge returns a typed entity ID; serializing that ID is an engineered output step. During evaluation the parser receives an empty answer placeholder and the typed encoder receives a dummy scoring label. Neither reaches inference as a gold answer.

## Causal state replay

I reconstructed the training donor bank and checked it with the graph walker. **Zero of the 256 original donor graphs reached the labeled donor state at step one**, and in all 256 cases the actual state had no valid continuation in the intended recipient graph. This changes the interpretation of the earlier zero-gain result. It describes the implemented training contrast. Y4 did not provide a valid test of correctly targeted interchange intervention training.

For this follow-up only, the intervened donor node was renamed consistently in both ends of its incident edges. Assertions verify that the donor reaches the intended intermediate state and retains its own distinct endpoint. The learned intervention takes its **actual soft state after one forward step** and inserts it into the recipient. A separate run inserts the oracle one-hot. A third supplies a wrong donor on a graph with a known distinct suffix. I scored the original recipient suffix and a last-edge rewire as a pair.

| Five-seed arm | Actual donor, both suffixes correct | Oracle donor, both correct | Wrong donor followed its actual suffix | Identity patch preserved prediction |
|---|---:|---:|---:|---:|
| True-step, ordinary supervision | 3,840/3,840 | 3,840/3,840 | 3,840/3,840 | 3,840/3,840 |
| True-step + historical IIT objective | 3,838/3,840 | 3,838/3,840 | 3,839/3,840 | 3,840/3,840 |

Each arm saw 256 pairs at each of depths 2, 3 and 4, with a step-one intervention. The two paired-suffix failures were at depth four in IIT seed 73. This supports causal reuse of the current entity *at this named state in this typed architecture*. The state space and value-copy operation are built into the model. The test does not identify an equivalent internal variable in the free-running byte model. [Donor measurements](DONOR.json)

## Reproduction

From the repository root, use the two commands below with the owner's archive. The renderer ablation adds an optional third command. All run without training. The bridge output records per-seed case masks and failure examples, so query or assignment slices can be rescored without loading a checkpoint again. [Bridge code](../../../../tools/avendi_y4_frozen_bridge.py) · [Donor replay code](../../../../tools/avendi_y4_frozen_donor.py)

```sh
python -m tools.avendi_y4_frozen_bridge AVENDI_Y4_FULL.zip BRIDGE.json
python -m tools.avendi_y4_frozen_donor AVENDI_Y4_FULL.zip DONOR.json
python -m tools.avendi_y4_renderer_ablation AVENDI_Y4_FULL.zip RENDERER.json
```
