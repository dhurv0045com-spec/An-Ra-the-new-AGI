# TASK VALIDITY ANALYSIS — V3 skills C and D

## C: two-hop composition with INDEPENDENTLY ORDERED blocks

Prompt: `Links: [x gives m;]* [m makes y;]* Trace: x to` -> y.

- **Information available:** both mappings in context; the queried x; 3 candidate y.
- **Rule inferred:** x -> m via the gives-block, m -> y via the makes-block.
- **Sealed derivability:** every prompt carries both maps; answer logically determined.
- **Independent orders (V3 repair):** XM and MY block orders come from separate hash
  namespaces (`c_orders`); no mode structurally aligns a chain's two positions.
- **Positional-shortcut statistic (measured, n=3000, canonical):** same-ordinal Y equals
  the true answer **0.3377** of the time (~chance 1/3). Under the V2 tied renderer the
  same statistic is **1.0000** — the mutation test proves the detector catches the old
  build if it is ever reintroduced.
- **Composition-oracle statistic:** recovering x -> matching M -> matching Y from the
  RENDERED prompt yields the true answer **1.0000** (n=1000) — the intended operation is
  sufficient and necessary (same-ordinal picking is not).
- **Other shortcuts:** direct (x,y) memorization impossible across disjoint factsets;
  y-frequency balanced by bijective assignment (test); 1-hop bypass impossible (Trace
  answers are never intermediates; test).
- **Chance:** 1/3. **Generalization:** thresholds on held-out factsets with permuted,
  independently ordered blocks.

## D: genuine inverse retrieval

Prompt: `Owners: o1 holds b1; o2 holds b2; o3 holds b3; Who holds b2 ?` -> o2.

- Facts are presented FORWARD (owner -> object); the query is an OBJECT; the answer is
  its OWNER — the model must invert the presented relation's direction.
- The ordered (object, owner) query-answer pair is NEVER presented as a fact (test);
  under the V2 pre-reversed construction it always was (mutation test).
- Chance: 1/3 within the factset's three owners (1/6 across the owner group); test
  asserts owner-group membership of answers and object-group membership of queries.
- Splits disjoint, deterministic, frequency-balanced; order modes actually reorder.

## A/B

Unchanged from V2 (V4-identical constructions, splits 524218/524219, three-mode
order-robustness, chance 1/3 within factset candidates).
