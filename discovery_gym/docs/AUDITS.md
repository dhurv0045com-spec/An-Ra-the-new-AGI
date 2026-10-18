# Shortcut and Leakage Audits

## Executable audit

Run:

```text
python -m discovery_gym dg0 --output <new-output-directory>
```

The command refuses to overwrite an existing output directory. It writes `audit.json` and a result receipt. The audit exits nonzero if a critical check fails.

## Critical checks

1. **Generator and split integrity:** unique world IDs, complete provenance hashes, no semantic or structural overlap across splits, a target sampled from the complement of an anchor neighborhood, and no intervention path into any target-candidate assignment.
2. **Public evidence firewall:** every initial public view has only the declared public fields and no hidden values.
3. **Legality and cost accounting:** legal descriptors carry exact costs; repeated and target-context actions are rejected, charged, and do not create a second accepted result.
4. **Independent evaluator recomputation:** every accepted transition and final verdict agrees with direct rule evaluation; the oracle ceiling is available only as a diagnostic.
5. **Shortcut baseline audit:** no-experiment, surface/position, and task-identity controls remain below the evaluator ceiling. Their rates are reported, not hidden.
6. **Confirmation firewall:** DG0 records `model_status=NOT_RUN` and forbids policy, threshold, or generator tuning on confirmation outcomes.

## Leakage cases covered

- hidden rule, variant, structure ID, target assignment, generation seed, evaluator label, and future outcome keys;
- structure or split labels encoded in an opaque world ID;
- renamed or isomorphic rules crossing a split;
- a direct target-context probe or intervention;
- an intervention whose resulting assignment is in the target-candidate set;
- a result label or evaluator verdict entering the policy view;
- a policy that submits its own correctness label;
- a repeated or invalid experiment that appears free;
- a changed target answer with unchanged legal evidence;
- use of confirmation outcomes to choose a threshold or policy.

## What the audit does not prove

The audit does not prove that a neural model can learn, that the finite rule catalog resembles natural causal discovery, that transparent information gain is optimal, or that the generator is externally independent. A shortcut audit is evidence about benchmark validity, not a capability result.
