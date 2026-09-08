# ARK-014 ANALYSIS — order-robust non-arithmetic binding and retention transfer

## Status

**EXECUTED. Acquisition verdict: `ORDER_ROBUSTNESS_REPAIRED`. Transfer verdict: `ROBUST_BINDING_ACQUIRED_BUT_TRANSFER_INCONCLUSIVE`.**

The repaired diagnostics cleanly separated the ARK-009 failure. With canonical fixed-order training, the model again reached perfect canonical/query-only exact but remained near chance-like order robustness: final CONTROL ORDER_ONLY and QUERY_ORDER were 0.333, and SEALED was 0.327. It never qualified in 24k steps.

With deterministic order-augmented training on the same semantic task, the single acquisition seed qualified quickly: qualification onset 1,400, confirmation 1,800. At the fork, BIND_SEALED was 1.000 canonical/query-only and 0.987 ORDER_ONLY/QUERY_ORDER.

This is a strong screen showing that the previous ARK-009 weakness was primarily presentation-order brittleness rather than failure to retrieve queried values in canonical order.

## Retention-transfer screen

Three matched continuation orders were run from the qualified ORDER_AUGMENTED checkpoint, HIGH `1e-3` vs LOW `1e-5`.

- sealed-qualified pairs: 3/3;
- HIGH retention failures: 0/3;
- LOW retention failures: 0/3;
- mean sealed qualification retention: 1.0 for both arms;
- mean ORDER_ONLY/QUERY_ORDER area was ~0.9995 HIGH and ~0.9999 LOW.

Because HIGH produced no failures, there is no event contrast from which to infer LR protection. This is a genuine low-event-rate transfer screen, not evidence that the effect transfers and not evidence that it fails to transfer.

## Scientific interpretation

**DEMONSTRATED SCREEN (one acquisition seed):** deterministic fact-order augmentation repairs the specific order-robustness failure that blocked ARK-009 qualification.

**NOT DEMONSTRATED:** non-arithmetic LR-retention protection. Both HIGH and LOW stayed essentially perfect in the 6k continuation horizon, so the retention mechanism was not stress-tested.

The acquisition result is useful, but order augmentation is a known invariance/data-augmentation idea. The important program advance is that Arkenstone now has a non-arithmetic subject that can pass an orthogonalized robustness gate, enabling a future fresh multi-seed transfer experiment with a regime that actually generates retention events.
