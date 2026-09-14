# ARK-014 — ORDER-ROBUST NON-ARITHMETIC BINDING AND LR-RETENTION TRANSFER

## Status

**PREREGISTERED BEFORE EXECUTION.**

## Why this exists

ARK-009 reached ordinary fact-set-disjoint held-out exact = 1.0 on both seeds, but failed the composite query+order diagnostic. Audit showed that the diagnostic changed query and fact order simultaneously. Because every fact-set already appears with all three possible queries, same-order query changes are already covered by the ordinary test. The unresolved weakness is primarily robustness to fact ordering and its interaction with query identity.

ARK-014 repairs that test and asks whether the LR-retention effect transfers once the non-arithmetic binding subject is genuinely order-robust.

## Task

Reuse the ARK-009 symbolic key-value retrieval semantics:
- six keys 0..5 and six values 0..5;
- each example contains three key=value facts and asks for one value;
- complete fact-set split before query expansion;
- task seed 4242;
- 400 train fact-sets, 100 test fact-sets;
- zero fact-set overlap.

Split the 100 held-out fact-sets deterministically by fact-set signature hash:
- 50 `BIND_CONTROL` fact-sets;
- 50 `BIND_SEALED` fact-sets.

The split occurs before query or order expansion. BIND_SEALED is measurement-only.

## Orthogonal diagnostics

For each held-out fact-set, evaluate:
1. `CANONICAL`: original stored fact order, all three queries.
2. `ORDER_ONLY`: same query and answer, reversed fact order.
3. `QUERY_ONLY`: same fact order, alternate query. This is expected to be redundant with CANONICAL because all three queries are already enumerated; it is still reported explicitly for audit clarity.
4. `QUERY_ORDER`: reversed order plus alternate query, reproducing the difficult composite style from ARK-009.

The runner must report each metric separately. No composite metric may be described as specifically measuring query-conditioning unless its variables are isolated.

## Acquisition arms

Matched acquisition seed: 2201.

Two training regimes, same architecture/optimizer and same number of optimizer steps:

- `CANONICAL_TRAIN`: original ARK-009 training examples with fixed fact order.
- `ORDER_AUGMENTED`: same semantic fact-set/query examples, but the three fact pairs are deterministically permuted as a pure function of `(acquisition_seed, optimizer_step, batch_position, semantic_example_id)` so training receives order diversity without changing answer semantics.

Both use:
- Micro width128 / 4 layers / 4 heads;
- CompactVocab;
- AdamW lr1e-3, betas=(0.9,0.95), eps=1e-8, wd=0.1;
- batch64, clip1.0;
- eval every200;
- max 24,000 steps.

## Qualification

Controller qualification uses **BIND_CONTROL only** and requires 3 consecutive evals with:
- CANONICAL >=0.90;
- ORDER_ONLY >=0.85;
- QUERY_ORDER >=0.85.

QUERY_ONLY is reported but is not an additional gate because it is semantically redundant with the all-query canonical set.

At qualification, snapshot exact model/optimizer/RNG state and measure BIND_SEALED once for analysis only.

If neither acquisition regime qualifies, retention transfer is blocked and the experiment still answers whether order augmentation repaired the ARK-009 failure.

## Retention transfer screen

Only an acquisition regime that qualifies on BIND_CONTROL is forked. Prefer ORDER_AUGMENTED if both qualify; if both qualify, both may be retained if budget permits, but the runner must state the policy before inspecting sealed outcomes.

Frozen continuation seeds:
- 7701
- 7702
- 7703

For each order, fork from identical qualified state:
- HIGH lr=1e-3
- LOW lr=1e-5

Run 6,000 matched continuation steps on the same semantic training regime used for acquisition. Arms consume identical semantic examples and identical order permutations. Evaluate all four CONTROL and SEALED diagnostics every 200 steps.

Primary transfer retention endpoint on BIND_SEALED:
- qualification retention, where sealed qualification requires CANONICAL>=0.90, ORDER_ONLY>=0.85, QUERY_ORDER>=0.85;
- first 3 consecutive failures of that sealed qualification;
- paired HIGH vs LOW failure counts and risk difference.

## Verdicts

- `ORDER_ROBUSTNESS_REPAIRED`: ORDER_AUGMENTED qualifies while CANONICAL_TRAIN does not, or materially improves ORDER_ONLY and QUERY_ORDER robustness with the preregistered thresholds reached.
- `ROBUST_BINDING_ACQUIRED_BUT_TRANSFER_INCONCLUSIVE`: robust acquisition qualifies but too few retention events occur for a directional LR claim.
- `NONARITHMETIC_LR_PROTECTION_SCREEN`: at least 2 matched continuation orders qualify on sealed at fork, HIGH has >=1 sealed retention failure, LOW has none, and no reverse discordance. This is a screen, not full replication because acquisition seed n=1.
- `TRANSFER_NOT_SUPPORTED_SCREEN`: robust binding qualifies and enough failures occur, but LOW does not protect.
- `ROBUST_BINDING_NOT_ACQUIRED`: neither arm reaches qualification.

## Claim limits

One acquisition seed cannot establish non-arithmetic replication. A positive retention screen is only authorization for a later multi-seed transfer replication; it is not a universal cognition result.