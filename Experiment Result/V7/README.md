# V7 owner-run evidence

**Owner results archived — CAUSE UNRESOLVED.** [Focused results, correctness and failures](c9981309a94847a4a97d8115c56d75c8/RESULTS.md) cover the completed nine-run study at source `413a3ecc`. The final status, compact measurements, trajectories and reproduction script are preserved beside the report. Historical V6 evidence remains unchanged in its own folder.

After the registered study, retain `ELENDRIL_V7_FULL.zip`, `ELENDRIL_V7_ANALYSIS.zip` and `FINAL_STATUS.json`. The final status file certifies delivery separately from the pre-export receipts inside the ZIPs; a ZIP cannot contain a certificate of its own final hash.

From the published source checkout, archive a returned full ZIP or saved-output directory:

```text
python -m tools.analyze_elendril_v7 --input <full-ZIP-or-output-directory> --run-id <unique-run-name>
```

The default destination is `Experiment Result/V7/<unique-run-name>/`. Intake verifies ZIP contents, independently replays recorded actions when datasets are present, preserves original text receipts under `raw/analysis`, and writes derived summaries, scientific reports and a hash inventory without changing the input. An existing run name is rejected. Analysis-only ZIPs can be archived but lack datasets for independent action replay; this limitation is explicit. Keep full checkpoint ZIPs outside Git and preserve their recorded hashes and custody.

Reports separate symptoms, engineering-reference disagreements, competing hypotheses, intervention outcomes and demonstrated causes. They report matched case trajectories within seed, cohort, dataset and checkpoint. A change from correct to incorrect is an observed loss, not proof of catastrophic forgetting; engineered-reference disagreement is not proof of incorrect cognition. Available value/EOS probes support investigation but do not establish a cause. Root cause stays unresolved until a controlled mechanism correction survives held-out, independently replicated comparisons.
