# Development capability benchmark

The v1 generator admits only train and development partitions. All prompt strings are reversible under the symbolic vocabulary. Generated examples carry an immutable task ID and latent-cluster ID. The model adapter receives only the prompt and a correlation ID. Development truth is joined after the greedy prediction stream is frozen. The generator and truth are available in source for reproducibility and this development partition is public.

| Family | Construction | Unit and expected failure control |
| --- | --- | --- |
| Copy | Return three symbols after the C prefix. Hold out complete triples. | Three-token resolution and EOS. A prompt parser is the exact positive ceiling. |
| Binding | Two distinct values assigned to X/Y. Render both assignments, both presentation orders and both queries. | The unordered value pair is a split cluster. Both-correct query swaps use identical facts and differing queries. A positional or query-blind policy scores one correct in each pair. |
| Abstention | The two-fact world additionally supports a Z query whose value is absent. | Report both-correct on present X/Y and exact on the full fixture. For a confirmatory variant rotate the missing key and answer representation to rule out a Z-only rule. |
| State/order | Apply two noncommuting primitive operations modulo seven. Both orders are in the same cluster, with disjoint starting-state values for development. | Require exact result and both orders in a cluster. Condition on primitive skill and include order-blind control in a prospective study. |
| Composition | Hold out the entire J/L pair in both orders. J adds one, K doubles, L negates and M adds three modulo seven. The single-operation family provides primitive checks. | The current held-out pair is a single cluster. A subsequent design must independently hold out multiple operation sets and isolate all cross-family aliases. |
| Retention | Compare the same A probe before and after acquisition of B, and after C where available. A-only controls drift. | Join the identical frozen A fixture, report paired cluster changes, and require independently measured B acquisition. |
| Forward transfer | Compare C after A/B with fresh C, holding total cost fixed. Also show C-exposure-matched learning curves as a diagnostic. | Match candidate seed and C world block. A benefit must exceed fixed replay or strategy baselines where applicable. |

Train/development partitions are disjoint within each family. Binding and abstention share underlying worlds by design. State/order and composition overlap semantically. A future multi-family study must split globally by shared latent program before rendering. Treat each existing family as a separate instrument until that split is implemented.

The longitudinal module joins scored cluster counts at identical fixture, model and seed identities. Retention requires positive intervening training exposure. Forward transfer requires equality across the full cumulative work vector. These checks support later comparison without authorizing a staged run. Causal interpretation still requires the preregistered acquisition and intervention controls.

## Metrics and validation

Report exact complete answer plus valid EOS, valid EOS alone, content token accuracy by position, longest common prefix, both-correct swaps, and all views correct per cluster. Invalid special tokens, missing termination and extra output cannot earn exact credit. Keep raw emitted token IDs, including malformed or incomplete streams, with exact denominators. Query swaps are paired diagnostics. Latent clusters are the uncertainty unit, and paired seeds are the replication unit. Per-view rows are dependent. Tiny development fixture counts support descriptive ranges. A prospective study should preregister a paired cluster/bootstrap or randomization procedure with interval width calibrated using a separate development panel.

The audit executes oracle, deliberate failure, fixed output, length, format-copy, first-fact query-blind, latest-value, and seeded random-choice controls. Random choice is a declared candidate policy with a finite observed score. Free-generation chance has a different interpretation. Confirmatory fixture design should add prompt/template deduplication, rendered collision checks, order/length attacks, held-out entity aliases, and independently implemented truth generation.

Reserve confirmation and sealed splits with separate seed and template custody. The custodian creates them and gives the verifier a committed input/truth manifest and the candidate prompt-only access. The chooser receives development receipts. All confirmatory access and unlock decisions require owner review.
