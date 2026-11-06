# Benchmark validity audit

The independent [fixture audit](../../artifacts/elendril/eld-dev-001/fixture-audit.json) verifies disjoint latent-cluster IDs and rendered prompts within each family. Oracle exact/EOS equals the fixture denominator and the malformed deliberate-failure control scores zero in all six families.

| Family | Train rows | Development rows | Development clusters | Validity boundary |
| --- | ---: | ---: | ---: | --- |
| Copy | 384 | 128 | 128 | The prompt-parser ceiling is 128/128. This family calibrates learnability, alignment and EOS. |
| Binding | 168 | 56 | 7 | Both first- and last-fact controls reach 28/56 exact and 0/28 both-correct. Assignment and order are balanced within each value-pair world. |
| Abstention | 252 | 84 | 7 | First/last controls reach 28/84 exact. Missing key is fixed to Z and requires randomization for a semantic abstention claim. |
| State/order | 40 | 16 | 8 | Both orders are present for noncommuting cases. Development uses two held-out starting values. |
| Composition | 42 | 14 | 1 | The held-out pair exercises both orders. One latent pair supports a scaffold check, with insufficient independent clusters for an effect estimate. |
| Primitive | 20 | 8 | 8 | The primitive-operation fixture checks mapping on held-out inputs before staged composition. |

Binding's raw 50% position baseline makes the both-correct endpoint indispensable. The paired loss/sampling successor should keep the answer vocabulary and fixture unchanged for its first development comparison. An independent follow-up must diversify query aliases, answer labels and rendering, freeze a global split before rendering, and increase the number of genuinely independent worlds and operation pairs. The present public source allows access to all development gold. The model API isolates prompt strings and the process-independent scorer reproduces the result from committed outputs. That boundary is a correctness contract for this code path. It is insufficient for untrusted candidate code or prospective sealed evaluation.
