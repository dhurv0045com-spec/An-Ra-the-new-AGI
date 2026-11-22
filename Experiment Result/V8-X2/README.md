# ELENDRIL V8-X2 Experiment Results

This directory stores curated owner-retrieved reports, metrics, provenance receipts, and test-first cards for **Elendril V8-X2** ("Can a learner revise a binding and use it again?"). Raw ZIPs, checkpoint tensors, and full episode dumps remain outside Git.

- **Current status:** The original 42-job X1 core (`I_X1_CORE`, reproduced `36/36` in Run 3) and `7/12` `IIIa_ORIGINAL_U` jobs (`U-1`, `U-2`, `U-4 s7`) have audited outcomes (`49/192` obligation rows complete). The broader X2 campaign remains partial. See the [first owner run](OWNER_RUN_01.md), [second owner run](OWNER_RUN_02.md), [third owner run](OWNER_RUN_03.md), and [Run 3 per-seed metrics](OWNER_RUN_03_METRICS.json) ([Run 2 metrics](OWNER_RUN_02_METRICS.json)).
- **Pre-run Scaffold:** `TEMPLATE_METRICS.json` records the immutable 126-base-equivalent design registry structure, primary endpoints, and separate 4-axis qualification status fields (`execution_status`, `delivery_status`, `scientific_status`, `causal_status`).
- **A0 Historical Cross-Check:** Original X1 A0 results remain preserved under `Experiment Result/V8-X1/A0/RESULTS.md` and can be cross-checked against the compact A0 ZIP via `python -m tools.rescore_x1_a0 <A0_ANALYSIS_ZIP> <OUT_JSON>` (expected 31,488 episodes).
