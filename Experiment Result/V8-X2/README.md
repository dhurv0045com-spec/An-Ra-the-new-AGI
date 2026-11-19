# ELENDRIL V8-X2 Experiment Results

This directory stores curated owner-retrieved reports, metrics, provenance receipts, and test-first cards for **Elendril V8-X2** ("Can a learner revise a binding and use it again?"). Raw ZIPs, checkpoint tensors, and full episode dumps remain outside Git.

- **Current Status:** `CPU_QUALIFIED_READY_FOR_OWNER_CANARY` (no S20 19.87M GPU training or confirmation executed locally).
- **Pre-run Scaffold:** `TEMPLATE_METRICS.json` records the immutable 126-base-equivalent design registry structure, primary endpoints, and separate 4-axis qualification status fields (`execution_status`, `delivery_status`, `scientific_status`, `causal_status`).
- **A0 Historical Cross-Check:** Original X1 A0 results remain preserved under `Experiment Result/V8-X1/A0/RESULTS.md` and can be cross-checked against the compact A0 ZIP via `python -m tools.rescore_x1_a0 <A0_ANALYSIS_ZIP> <OUT_JSON>` (expected 31,488 episodes).
