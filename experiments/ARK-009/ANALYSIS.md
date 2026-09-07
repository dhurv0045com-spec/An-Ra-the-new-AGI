# ARK-009 ANALYSIS — non-arithmetic transfer gate

## Status

**EXECUTED, STRICT QUALIFICATION NOT MET; RETENTION TRANSFER NOT EXECUTED.**

The machine receipt says `TRANSFER_BLOCKED_BY_ACQUISITION`, but the scientific wording needs nuance: both seeds reached **1.00 ordinary fact-set-disjoint held-out exact**, while neither passed the combined ordinary-test + query-swap qualification.

| seed | ordinary peak | query-swap peak | final ordinary | final swap | qualified |
|---:|---:|---:|---:|---:|---|
| 1201 | 1.000 | 0.383 | 1.000 | 0.310 | NO |
| 1202 | 1.000 | 0.360 | 1.000 | 0.337 | NO |

Both ran to 24,000 steps. Fact-set overlap is 0.

## Critical diagnostic confound

The implemented `query-swap` changes **two variables simultaneously**: it reverses fact order and asks for a different key. Therefore its low score does not isolate query conditioning; it may reflect order sensitivity, query sensitivity, or their interaction.

Demonstrated: perfect ordinary exact on held-out complete fact-sets under canonical presentation; poor robustness under the composite query+order intervention; the preregistered gate correctly blocked retention transfer.

Not demonstrated: robust variable binding under isolated interventions, query-blindness specifically, or transfer of ARK-007R LR protection.

Next repair: orthogonalize `QUERY_ONLY` (same facts/order, different query), `ORDER_ONLY` (same query, permuted order), and `QUERY+ORDER`.
