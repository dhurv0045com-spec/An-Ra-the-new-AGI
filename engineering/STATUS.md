# Current engineering state

## Active: FINAL-K8 — build verified, owner launch pending

Latest implementation: 41a48089+ (BRAMASTRA). The build verifier
(`python -m bramastra_lab.research.campaigns.k8 verify-build`) reports
**VERIFIED: 24/24 requirements PASS** with zero local optimizer commits,
and the evidence-bound startup gate admits. The full registered data
bundle (4096/256/256/128 mechanisms per family, 4096+256 tool tasks,
meta 24/6/6) is generated, validated, and carries the information-sufficiency
witness per mechanism. The historical 15 September review findings
(compiler slicing, copied P0/P1 choices, always-false gate, absent
verify-build) are all closed in later commits; see
[reports/FINAL_K8/BUILD_READINESS.json](reports/FINAL_K8/BUILD_READINESS.json)
for the derived verdicts.

## 22 September partial campaign (owner-run, superseded by repairs)

The owner's first Kaggle campaign passed E0 and E1 (16,000 committed
updates) and correctly stopped at E3: the bundle then carried only 256
tool-training rows against a calibrated 4,000-row demand
([K8_20260922_RESULT.md](../docs/bramastra/K8_20260922_RESULT.md)).
Repairs since: tool inventory raised to 4,096 training + 256 held-out
tasks; `discover_bundle` now refuses any attached bundle below the E3
minimum before GPU work (diagnosed, never silently used); verify-build's
bundle receipt records tool cardinality; E3/E4 save checkpoints every 200
steps/updates like E1; xprobe contract failures repaired (tests in
tests/test_research_xprobe.py).

## Completion and authority

Ready for owner experiment: **true, with runtime gates pending** — E0
hardware qualification (G01–G04) remains the mandatory first phase of the
owner's single 600-minute two-T4 allocation (training stop 570, export
reserve 30; Kaggle GPU sessions allow 12 h). No local optimizer updates
were performed at any point. The owner launches via
[notebooks/bramastra_k8.ipynb](../notebooks/bramastra_k8.ipynb) using
[final_delivery/RUN_EXPERIMENT.md](final_delivery/RUN_EXPERIMENT.md).
