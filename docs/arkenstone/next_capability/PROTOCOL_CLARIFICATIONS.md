# ARK-014 prospective protocol clarifications

Date: 2026-09-12. Written **before** the learned ARK-014 comparison was
executed. Packet: ARK-NEXT-001. These clarifications freeze interpretations
that experiments/ARK-014/PLAN.md left open; they do not alter any frozen
threshold, seed, budget or verdict name. They are applied identically to
baseline and candidate.

## 1. Audit of the inherited objective (result: reused unchanged)

The inherited supervised objective (`loss_and_positions` in
`experiments/ARK-001/run_ark001.py`, reused by ARK-009 through ARK-013)
supervises, for each row, the tokens after the prompt: the answer's BOS token,
the answer characters, EOS, and — in tasks with mixed answer lengths — the
padding positions after EOS inside a batch (they share the mask
`positions >= prompt_len - 1`).

For ARK-014 every prompt is exactly 14 characters (`k=v+k=v+k=v/q=`) and every
answer is one digit, so every encoded row is 18 tokens and the mask covers
exactly 3 positions per row (answer-BOS, answer digit, EOS) with zero padding
positions. The mixed-length padding concern **cannot arise in this task**.

Decision: the historical objective is reused **unchanged** (same function, same
source). Its identity is bound in every ARK-014 receipt via the
`run_ark001.py` source hash recorded by the shared `ReceiptWriter`. Test
`tests/test_ark014.py::ObjectiveAuditTests` asserts the 3-supervised-position
invariant and the uniform row length. Had a correction been required, it would
have been defined here first and versioned separately; it is not.

## 2. BIND_CONTROL / BIND_SEALED hash split (exact rule)

PLAN.md says the 100 held-out fact-sets are split "deterministically by
fact-set signature hash". The exact frozen rule, implemented in
`ark014_binding.split_control_sealed` and applied before query or order
expansion:

- rank held-out fact-sets by `sha256(json(canonical_signature))`, where the
  canonical signature is the sorted list of `[key, value]` pairs, JSON-serialized
  with sorted keys and no whitespace;
- ties (none exist for distinct signatures) break by the signature itself;
- even ranks (0th, 2nd, ...) are BIND_CONTROL; odd ranks are BIND_SEALED;
  50 fact-sets each.

Membership and assignment hashes are recorded in `ARK-014_TASK_MANIFEST.json`.
The train/held-out fact-set universe reproduces the historical ARK-009 anchor
hashes exactly (asserted at build time).

## 3. Diagnostics construction (exact rules)

Per held-out fact-set `f = ((k1,v1),(k2,v2),(k3,v3))` in stored order, with
mapping `m` and key order `K = (k1,k2,k3)`, each diagnostic enumerates exactly
three query rows (150 rows per diagnostic per split):

- CANONICAL: present `f`, query each `k` in `K`; answer `m[k]`.
- ORDER_ONLY: present `reversed(f)`, query the same keys in the same canonical
  order; answer unchanged per key.
- QUERY_ONLY: present `f`, for the canonical row at position `i` query
  `K[(i+1) mod 3]`; answer `m[K[(i+1) mod 3]]`.
- QUERY_ORDER: present `reversed(f)`, and — reproducing the ARK-009 swap
  construction verbatim — for the canonical row querying `k`, find `k`'s
  position in the reversed key order and query the next key cyclically; answer
  is the queried key's value under `m`.

Every row's answer is asserted equal to the symbolic lookup of its query in the
presented fact-set's mapping (fixed prior), for all four diagnostics.

## 4. Order augmentation (exact function)

`ORDER_AUGMENTED` permutes the presentation order of the three facts as a pure
function of `(acquisition_seed=2201, optimizer_step, batch_position,
semantic_example_id)`: fold the four integers through splitmix64 and reduce
modulo 6; index into the fixed permutation table
`itertools.permutations(range(3))`. The function consumes no RNG state, so the
matched semantic minibatch stream (a `torch.Generator` seeded 2201) is
byte-identical across arms. Query and answer are never changed. For retention
continuation, `optimizer_step` is the fork-relative step (1..6000), so HIGH and
LOW arms see identical semantic examples and identical order permutations.

## 5. Frozen interpretations of ambiguous verdict language

These were frozen before execution and are encoded in
`run_ark014.summarize`/`_verdict`; they are implementation interpretations,
not new scientific laws.

1. **"Materially improves"** is only consulted when BOTH regimes qualify (if
   the canonical arm fails, the qualification contrast itself decides
   ORDER_ROBUSTNESS_REPAIRED). Definition: ORDER_AUGMENTED must exceed
   CANONICAL_TRAIN by >= 0.05 on BOTH BIND_CONTROL ORDER_ONLY and BIND_CONTROL
   QUERY_ORDER, each measured as the mean over the last three evals of the
   respective arm's trajectory. Raw deltas are reported alongside.
2. **Only CANONICAL_TRAIN qualifies** is not covered by a plan verdict; the
   runner reports `ORDER_AUGMENTATION_NOT_SUPPORTED_CANONICAL_QUALIFIED`.
3. **Retention execution statuses**: `NOT_EXECUTED` (blocked before any
   continuation arm), `INCOMPLETE_ARMS` (at least one arm lacks the full 6,000
   steps), `EXECUTED` (all frozen orders completed both LRs). A matched
   retention comparison requires both LRs of one order to complete 6,000
   steps; partial arms are never compared.
4. **TRANSFER_NOT_SUPPORTED_SCREEN vs inconclusive**: the protection screen
   requires no reverse discordance. Frozen here: TRANSFER_NOT_SUPPORTED_SCREEN
   additionally requires zero reverse discordance (HIGH failed where LOW did
   not, in at least one order, and LOW never failed where HIGH did not). A
   mixed outcome (LOW failed in an order where HIGH did not) is reported as
   `ROBUST_BINDING_ACQUIRED_BUT_TRANSFER_INCONCLUSIVE`, not as a negative
   transfer result.
5. **Sealed-qualified at fork**: an order counts toward the screen only if the
   FIRST recorded sealed eval of that order's fork meets the sealed
   qualification thresholds. Sealed trajectories are otherwise used only to
   detect the first 3 consecutive sealed-qualification failures.
6. **Sealed usage**: BIND_SEALED is measured (a) once per acquisition arm,
   strictly after the qualification decision and checkpoint snapshot, and
   (b) every 200 retention steps for the frozen endpoint analysis. No code
   path reads sealed values to select checkpoints, change LR, stop training or
   choose which regime advances. `tests/test_ark014.py::ControllerDisciplineTests`
   demonstrates decisions are invariant to arbitrary sealed mutations.

## 6. Compute box for the recorded runs

Runs launched under this clarification record the device, torch/python
versions, wall budget and an authorization note in the receipt
(`compute_box`). The historical ~100 TPU-hours/week estimate is not treated as
an entitlement; only owner-authorized local hardware is used.
