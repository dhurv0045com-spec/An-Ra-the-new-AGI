# C0 — Preregistration: Can an answer-blind selection policy pass the calibrated bias screen?

Status: **PREREGISTERED — NO RESULTS** (written 2026-09-03, before any C0 execution).
Branch: `citadel` @ base `85f44b7b449f2ee39a0e80203a2d7df04614983b`.
Ledger context: EVIDENCE_LEDGER E3/T7, NEGATIVE_RESULTS N3/N4/N18, BOTTLENECK_RANKING B1.

## Question

Does a candidate-selection policy whose scoring computation never receives the candidate set
(answer-blind free generation + normalized matching) pass the same calibrated bias screen that
rejected every likelihood-based scoring policy, while the screen retains the power to detect a
planted query-conditioned signal?

## Existing evidence

Repository-supported facts only (full citations in EVIDENCE_LEDGER.md):

- E3/N3: the preregistered scoring tournament (256 triplets × 3 rotations × 5 dev seeds × 3
  tokenizers) rejected domain-PMI and contextual-calibration — both select the fewest-token role
  1.000 in all 15 CUDA cells (`artifacts/e2/scoring_policy_development.json`,
  `production_scoring_mode: null`, 0.0757 GPU-hours). Rotation geometry itself validated
  (first_position_equivalence 0.3333 in 15/15 cells).
- Null receipts: on untrained exact middle-P35 weights, sum/byte/token aggregation selects the
  fewest-token candidate 100% / 83.33% / 50–66.67% (`artifacts/e2/local_{cpu,cuda}_scoring_null.json`) —
  i.e., the bias screen is meaningful with untrained weights, so C0 needs no training.
- N18: small-N scorer calibration is underpowered and rejected; the fixture is the sanctioned vehicle.
- T5/N10 (triquetra): a behavioral exact-match generation scorer passed triquetra's own
  answer-blind firewall and produced replicated factorial effects on a weak checkpoint — but was
  never tested against ESOES's calibrated screen. No generation-based policy has ever been screened.
- E4: E0's heuristic gate convention is `accuracy − calibrated null ≤ 0.10` on point estimates,
  pooled over independent seeds.

## Primary hypothesis

**H1:** An answer-blind generation-based selection policy exhibits worst-axis bias
B = max_a |Acc_a − p0_a| ≤ 0.10 across all preregistered decoy axes and all rotation positions
on the frozen 256-group development fixture — i.e., it passes the screen that all five
likelihood-based policies failed.

## Competing hypothesis

**H2:** Generation-based matching carries its own systematic surface bias — degenerate or short
generations disproportionately match short/fewest-token candidates, format artifacts match
marked-prefix/first-token patterns, or tie-breaking induces position bias — so it also fails the
screen. Under H2 the defect is plausibly in the task/candidate construction itself (axis
correlations are properties of the fixture), and benchmark revision, not policy search, is the fix.

## Why this experiment

- `production_scoring_mode = null` is the recorded blocker for *every* learned cognition
  comparison in the program (training-program Stage 0; E1–E3 comparisons; any Citadel training
  experiment's primary metric). No measurement can be trusted until a policy passes or the
  instrument is redefined.
- Highest information gain per cost in the bottleneck ranking (B1): no training, no GPU-heavy
  compute, machinery and fixture already exist and are tested.
- It cleanly discriminates two explanations with different fixes: "policy family was too narrow"
  (fix: adopt generation-based scoring) vs "candidate construction is intrinsically confounded"
  (fix: revise candidate construction before any training claims).
- Obvious alternatives are worse: training experiments before a valid scorer waste compute and
  produce unmeasurable or biased outcomes; waiting for a stronger checkpoint leaves the
  instrument blocker untouched.

## Independent variable

Exactly one: **the candidate-selection policy**. Levels:

1. `sum` (likelihood sum) — negative control
2. `token_normalized` — negative control
3. `byte_normalized` — negative control
4. `domain_pmi` — negative control
5. `contextual_calibration` — negative control
6. `generation_match` — **the candidate under test**: greedy free generation (≤ 32 new tokens,
   EOS-terminated, temperature 0) from the query+context prompt only; the generated text is
   normalized (casefold, whitespace collapse) and matched against candidates by exact normalized
   string equality; no match ⇒ no selection ⇒ scored incorrect; multiple matches ⇒ tie-break by
   fixture candidate order (fixed, declared here).
7. `constructed_oracle` — positive control: a policy with genuine query-conditioned preference
   built from evaluator-side gold through the sanctioned oracle path (demonstrates the screen
   detects real signal when it exists).

## Fixed variables

- Fixture: the development fixture compilation from `e2_architecture/scoring_policy_fixture.py`
  (256 development groups + 256 identity-distinct fresh groups; unique shortest-byte/fewest-token
  roles; 3 first-token roles; balanced hidden labels; 6 surface families). Fixture SHA-256
  recorded at compile; expected to match `artifacts/e2/scoring_policy_fixture.json` (`3adb9bed…`).
- Fixture seeds: 95101–95105 (development compile seeds, existing convention).
- Tokenizer set: the three local-tournament byte-BPE artifacts (16,384 / 24,576 / 32,768) with
  SHA-256s recorded from `artifacts/e1/local_tournament/` receipts.
- Model: untrained exact middle-P35 (16×384, 35,414,400 params), init seed 47031, weights
  hash-recorded before use. No training, no fine-tuning, no checkpoint.
- Scoring adapter: suffix-only teacher forcing for likelihood policies (unchanged from the
  parity receipts); generation path for level 6 as specified above.
- Rotation contract: all 3 positions per triplet, unchanged.
- Null definitions: casewise uniform-candidate chance for candidate-set axes; permutation null
  for serialization-coupled axes; first-position equivalence 1/3 — all as implemented in the
  existing tournament machinery (unchanged code path).
- Hardware plan: primary run CUDA; CPU/CUDA parity pair on a fixed 10% subset (existing parity
  harness, tolerance as recorded: max abs err ≤ 0.01 selection-affecting).

## Models/checkpoints

- No trained checkpoint is used or produced.
- Untrained P35 init: seed 47031; parameter SHA-256 recorded in the run receipt.
- Branch SHA at run time recorded; fixture SHA-256 recorded; tokenizer artifact SHA-256s recorded.

## Data

- Fixture compiled by the existing generator (hash-pinned above); no external data.
- No sealed fixture is touched; the sealed tier does not exist yet (recorded state).
- No train/eval leakage question arises (no training), and the fixture's identity-distinct
  fresh groups are used only if a replication pass is triggered (below).

## Controls

- Negative controls (levels 1–5): must reproduce their recorded failures — sum/byte selecting
  the fewest-token role at ≥ 0.65 and both calibrated policies at ≥ 0.95, matching the recorded
  receipts. If they do not reproduce, the screen is broken → IMPLEMENTATION_FAILURE (this is the
  strongest single confound check: the instrument must reproduce known results before any new
  verdict is interpreted).
- Positive control (level 7): must pass the bias screen trivially (it is constructed unbiased)
  AND achieve gold-selection Wilson 95% LCB ≥ null + 0.10 — proving the fixture retains power to
  detect a real query-conditioned signal (a screen that cannot detect signal is worthless even
  when a policy "passes").
- Random baseline: uniform candidate selection reported alongside for reference (null ≈ 0.25/0.333).

## Metrics

Primary (declared first): **worst-axis bias**
`B(policy) = max over axes a ∈ {fewest_token, shortest_utf8_byte, marked_prefix, first_token(×3 roles), surface_family, hidden_label, position(×3 rotations)} of |Acc_a − p0_a|`,
pooled over the 5 development seeds and 256 groups. Gate value 0.10 (ESOES heuristic-gate convention).

Secondary: per-axis Wilson 95% intervals; gold-selection accuracy vs null (power view);
generation parse/match-rate distribution (reported, with no-match-scored-incorrect semantics);
CPU/CUDA parity of selections (exact match of argmax selections required).

## Statistical treatment

- Wilson 95% intervals on every reported rate (two-sided, z = 1.96), existing metrics module.
- Paired exact sign tests (policy vs null) where the comparison is paired within group.
- Holm–Bonferroni across the policy×axis family for the secondary interval criterion.
- Pooling across the 5 seeds before gate evaluation (single-seed pass insufficient, per E0 policy).

## Success threshold

Declared before any run: `generation_match` achieves **B ≤ 0.10 on every axis and rotation**
(point estimates, pooled over seeds) **and** the positive control achieves its power criterion
**and** all five negative controls reproduce their recorded failures.

## Failure threshold

Declared before any run: `generation_match` shows **B > 0.10 on any axis with the Wilson 95% LCB
of B_a exceeding 0.10** — bias beyond the gate, beyond noise → policy FAILS the screen;
H2 is supported; `production_scoring_mode` stays null and benchmark/candidate-construction
revision becomes the next preregistered target.

## Confound checks

1. Screen-validity: negative controls reproduce recorded failures (see Controls).
2. Screen-sensitivity: positive control detects the planted signal (see Controls).
3. Candidate-blindness of level 6: static source assertion that the generation prompt never
   includes candidate strings (audit + test; borrowed concept from triquetra's answer-blind
   guard, reimplemented locally, provenance recorded).
4. Determinism: identical rerun reproduces identical selections bit-for-bit (same device).
5. Parity: CPU vs CUDA selections identical on the 10% parity subset.
6. Generation plumbing: EOS behavior, max-token cap, normalization function unit-tested;
   tie-break rule fixed and declared above.
7. No threshold, axis, or policy may be added/changed after results are seen; amendments require
   a new preregistration file that supersedes this one, never silent edits.

## Compute budget

- Ceiling: 2.0 GPU-hours total (RTX 4050-class) or CPU equivalent; expected ≤ 0.5 GPU-hours
  (few thousand short greedy generations on a 35.4M model + cheap suffix scoring).
- Storage: receipts under `docs/citadel/experiments/C0/receipts/`, ≤ 100 MB.
- No training, no remote jobs, no sealed consumption.

## Stop condition

One complete pass of the 7-level grid on the development fixture + the parity pair, then stop:
write receipts, record the verdict in the ledger, and update `BOTTLENECK_RANKING.md`/`OPEN_QUESTIONS.md`.
No reruns, no threshold edits. If IMPLEMENTATION_FAILURE occurs, fix implementation and re-run
only with an appended amendment note; thresholds remain unchanged.

## Possible outcomes

1. **`generation_match` passes (H1).** Belief change: a valid answer-blind selection instrument
   exists; `production_scoring_mode` has a sanctioned development candidate; E1–E3 comparisons
   and Citadel training experiments (Q-C, Q-D) become measurable. Next: C1 micro-scale training
   baseline (Q-C) using this policy. Caveat recorded: untrained-weight bias screen does not
   certify behavior on a trained model — re-screen with the first trained checkpoint.
2. **`generation_match` fails (H2).** Belief change: the bias is plausibly structural to
   candidate construction; benchmark revision (balanced-length candidates, construction-level
   debiasing) is preregistered as C0.1 before any further policy search; scorer blocker remains.
3. **Mixed (passes some tokenizers/axes, fails others).** Belief change: policy is usable but
   fragile; scope of validity recorded per tokenizer/axis; treat as TENTATIVE instrument, do not
   promote; investigate the discriminating axis.
4. **Negative controls fail to reproduce (any level).** IMPLEMENTATION_FAILURE — the screen or
   fixture drifted; repair and re-preregister before interpreting anything. No scientific verdict.
5. **Positive control fails its power criterion.** The fixture cannot detect real signal at this
   sample size → the screen is not a valid gate for any policy; fixture power must be fixed first.
   No verdict on `generation_match` is drawn.
