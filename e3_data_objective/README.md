# E3 data and objective plan

E3 is staged. Phase A compares 5%, 15%, and 30% mechanically verified
cognition data under CE while preserving the 65:20 natural/code ratio in the
remaining tokens. Phase B can be instantiated only after Phase A selects one
mixture and one adjacent mixture; it compares CE against query-swap λ 0.05 and
0.15. The trace arm is disabled unless a hashed composition-transfer failure
explicitly triggers it.

```text
python -m e3_data_objective.plan --output artifacts/e3/static_plan.json
```

The default status is `BLOCKED_UPSTREAM_INPUTS`. This package contains no model
or trainer.
