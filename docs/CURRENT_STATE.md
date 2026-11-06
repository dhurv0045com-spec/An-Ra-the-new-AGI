# Current state

**V4 milestone: MECHANICS_QUALIFIED.** The lean research runtime now runs a complete public-evidence episode through the frozen Elendril decoder, budgeted actions, independent outcome verification and versioned episodic admission. Route and changing-state revision use separate task generators. No optimizer updates were made.

| Four episodes per arm | Fixed public-evidence control | Frozen neural checkpoint |
| --- | ---: | ---: |
| Route with admitted memory | 4/4 | **4/4** |
| Route with no memory | 0/4 | 0/4 |
| Route with same-token irrelevant context | 0/4 | 1/4 |
| Changing-state revision | **4/4** | 0/4 |

Retrieval in the memory arm was a fixed operation. The decoder generated every neural answer with its original weights. The status-aware revision renderer distinguished missing, explicit null and conflicting source values. The frozen policy answered H incorrectly three times, emitted one invalid action and never queried the available channel. Its earlier value-only rendering also failed all four revision episodes. The combined two-family model floor failed. Forty neural model calls consumed 912 input positions across all 20 neural episodes under the status-aware rendering. The memory and irrelevant-context arms each consumed 204 positions. This is evidence for using a retrieved public fact on four related routes from the historical development fixture. The fixed retrieval decision and reused generator bound that result. [Frozen scorecard](COGNITIVE_SCORECARD.md)

The status-aware demo used source `15d423e16ea41a96e7942370b13304625f4f1c88`, original checkpoint `96a15fd5e7a3de495d93c049fe6dde49f4874a2c148ac64c762dd20c49c9bc76` from ELD-DEV-004 and weight fingerprint `8734cfb9957f510a703c0d04e5f68a305e0af9552238c28f178e848dc9c2e863` before and after. A representative trace is a charged fixed RETRIEVE followed by a model-origin ANSWER F. The raw receipts retain every decision, rejected outcome, request and evidence identity. [V4 receipt manifest](../artifacts/elendril/cognitive-v4/manifest.json)

V3 topology-split fixtures, factorized rendering screens and shared-read alternatives are preserved. The sparse intermediate path has forward/gradient parity tests and no measured CUDA efficiency result. The two graph notebooks are default-disabled and do not exercise the V4 cognitive loop. [Experiment protocols](EXPERIMENT_PROTOCOLS.md) · [Cloud operations](OPERATIONS.md)

The next gate is a supervised positive control for **when to request evidence**. Generate disjoint revision worlds requiring a query in some cases and a direct answer or abstention in others. Compare learned action and answer choice against always-query and no-query baselines at the same inference and request budgets. Begin cloud action training only after the objective, oracle and split are frozen and the integrated verifier can score both cost and correctness.
