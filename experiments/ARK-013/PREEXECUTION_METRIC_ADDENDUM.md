# ARK-013 PRE-EXECUTION METRIC ADDENDUM

**Committed before execution.** The scientific PLAN used the phrase "materially lower" for the fixed-LOW plasticity-cost comparison but did not assign a numeric margin. The implemented runner must not be allowed to choose that margin after seeing results.

Operationalization frozen here:

`PLASTICITY_COST_DEMONSTRATED` vote on a matched triplet requires BOTH:
- `FIXED_LOW T3_SEALED final <= FIXED_HIGH T3_SEALED final - 0.10`, and
- `FIXED_LOW T2_SEALED final >= FIXED_HIGH T2_SEALED final + 0.10`.

A majority of completed matched triplets is required for the program-level boolean `plasticity_cost_demonstrated`.

The already-preregistered ADAPTIVE Pareto criterion remains unchanged: within 0.05 of FIXED_HIGH on new-skill final (or area) while improving old-skill final (or area) by at least 0.10, on at least 2 triplets spanning both fresh acquisition seeds.

This addendum resolves an analysis ambiguity before any ARK-013 result exists; it does not alter arms, seeds, data, or training budget.