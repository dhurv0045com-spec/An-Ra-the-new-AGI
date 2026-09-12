# ARK-013 PRE-EXECUTION ADDENDUM — CARRY TASK FEASIBILITY

**Committed before runner implementation or execution.**

Static combinatorial audit found that the PLAN.md clause requiring both (a) operand-A OOD tens band 6..7 and (b) a guaranteed two-digit final sum leaves too few unique forced-carry OOD operand pairs after the required commutation-overlap filter to supply the planned 200-row OOD set.

To preserve the more important preregistered properties — forced ones-column carry, structural operand-A band split, 500 train / 200 OOD examples, deterministic membership, and zero commutation overlap — the runner SHALL allow the final sum to be either two or three digits. Operand B tens may therefore range 1..9.

Everything else in PLAN.md is unchanged:
- train operand-A tens 1..5;
- OOD operand-A tens 6..7;
- ones digits must satisfy ua+ub>=10;
- deterministic dataset seed 131313;
- 500 train, 200 OOD after filtering;
- CONTROL/SEALED firewall;
- same three training arms and 12k matched horizon.

This is a feasibility correction discovered before code/execution, not an outcome-driven modification.