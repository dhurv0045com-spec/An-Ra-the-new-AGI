# ARK-004A — DEVELOPMENTAL TRANSITION MAPPING (preregistered; PLAN committed+pushed before training)

## Scientific question
What measurable behavioral or internal-state changes occur BEFORE structural
OOD generalization emerges on T2, and does any such factor predict the
transition timing beyond strong trivial baselines (step, loss, train exact,
logit margin)?

## Fixed configuration (identical to ARK-002B; no architecture drift)
- Model: Micro 4L/128w/4H, compact 19-token vocab, ~0.8M params.
- Data: frozen ARK-002B manifest split_sha256 0dd930569704...
  (TASK_MANIFEST.json copied verbatim into this directory and hash-bound).
- Optimizer AdamW lr 1e-3 betas (0.9,0.95) eps 1e-8 wd 0.1; batch 64.
- Budget: <= 24000 steps or 2400s wall box per seed; eval every 200 steps.
- Device: CUDA (RTX 4050); sequential seeds.

## Seed list (frozen before execution; genuinely fresh; none used before)
101, 202, 303, 404.
Historical seeds 13/29/47 are context only.

## Behavioral metrics (every eval)
train exact (100-row sample), S0 structural-OOD exact (197 test rows),
per-output-position accuracy, correct-token logit margin at tens and ones
answer positions, errors-by-position, counterfactual locality (100 fixed
cases), sustained M99/G50/G90/G95, post_mem_delay_90, OOD-AUC after M99,
supervised token count.

## Internal-state probes (minimal set; frozen diagnostic counterfactuals)
Frozen probe sets built ONCE from the test band:
- P-ONES: pairs differing only in ub (ones of b; no-carry preserved).
- P-TENS: pairs differing only in ta within 6..7 (tens of a; no-carry preserved).
At every eval, for each layer L (block outputs 0..3 + final norm):
1. REPRESENTATION DELTA: mean L2 distance between hidden states of the pair
   at the two answer-prediction positions (tens-position = last prompt token;
   ones-position = next token).
2. COLUMN SELECTIVITY: for P-ONES pairs, the fraction of total answer-position
   logit change that lands on the ones output position; symmetric for P-TENS.
   (Selectivity -> 1 means the representation/logits factorize by column.)
3. CROSS-COLUMN INTERFERENCE: 1 - selectivity (reported, same measurement).
Explicitly excluded: linear decoders, rank/norm/cosine dashboards (no
falsifiable role in this question yet).

## Prediction test (Triquetra-style, preregistered)
Does any factor measured in the EARLY post-memorization window (first 2000
steps after M99) predict G90 step across seeds?
- Candidate factors: mean column selectivity (tens, ones), mean margin,
  OOD-AUC-so-far.
- Trivial baselines: M99 step, training loss at M99, train exact at M99,
  mean margin at M99.
- Validation: leave-one-seed-out Spearman rank correlation, n=4 (weak power —
  honestly labeled). A precursor QUALIFIES only if its LOO advantage over the
  best trivial baseline is consistent in direction on >= 3 of 4 folds and the
  pooled LOO correlation exceeds every baseline's.

## Primary endpoint
LOO predictive advantage of tens-column selectivity (or the best internal
probe) over the best trivial baseline for G90 timing.

## Secondary endpoints
- Trajectory shape of selectivity vs OOD (does factorization precede, co-occur
  with, or follow OOD emergence?).
- Seed spread of all sustained thresholds on fresh seeds.

## Success criteria
- Precursor qualifies (per the prediction test) -> SUPPORTED COGNITION-PRECURSOR
  CANDIDATE (TENTATIVE if LOO power is marginal), and ARK-004B intervention is
  designed from it.
- Failure criteria: no factor beats baselines -> Case B negative result; write
  it, update the bottleneck graph, and move to the next hypothesis family.

## What these results do NOT justify
No AGI claim. No core promotion. No architecture change. No mechanism name
beyond neutral descriptions. n=4 cannot establish universal laws.

## Stopping rules
- Seed aborts at its wall box (recorded; sustained rules handle truncation).
- If two seeds show NO OOD emergence at all by box end, continue remaining
  seeds (the 002B record predicts box-edge variance) and analyze honestly.

## Compute budget
<= 4 x 2400s GPU wall = ~2.7h worst case.
