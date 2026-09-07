# IMPROVEMENTS — every adopted improvement, dated, attributed

Format: `date | agent | improvement | source | evidence`

---

2026-09-06 | arkenstone-agent | **Lift-off dose mapping instrument** | self-discovered | ARK-001: 200-400 steps for T1; replicated seeds 13/29

2026-09-06 | arkenstone-agent | **Commutation-free dataset manifest** | self-found during ARK-002B prep | split_sha 0dd930569704; zero overlap asserted

2026-09-06 | arkenstone-agent | **Sustained-threshold metrics (M99/G50/G90/G95)** | self-designed | ark_metrics.py + tests

2026-09-06 | arkenstone-agent | **Plan-commit-before-training discipline** | adopted from program research rules | ARK-002B onward

2026-09-06 | arkenstone-agent | **Fork-at-trigger experimental design** | self-designed after ARK-003 wall-time confound | run_005.py onward

2026-09-06 | arkenstone-agent | **Terminal/EOS supervision as a training contract** | BRAMASTRA evidence | Arkenstone harness verified EOS supervision

2026-09-06 | arkenstone-agent | **Capacity-accounting correction** | BRAMASTRA audit | interpretation rule recorded in feature ledger

2026-09-06 | arkenstone-agent | **Continuation probe / source snapshot / content-hash ledger verification** | program integrity work | provenance utilities + verifier

2026-09-08 | ChatGPT + user Colab | **Pinned syntax-safe Colab runner + GPU smoke test before long training** | repeated notebook indentation/XLA failures | V5 smoke receipt PASS; snapshot reload exact; continuation hash reproducible

2026-09-08 | ChatGPT audit | **External result-bundle validation before GitHub import** — re-hash every JSON receipt against the runner canonicalization, preserve source bundle SHA, and never rewrite preregistration history | user-uploaded `ARKENSTONE_V5_RESULTS.zip` | 9/9 receipt hashes matched; source ZIP SHA256 `e40355e0e0212406985c2445d2a8388b5d704ad722fd4c4b4c3455a5e64f0c7f`

2026-09-08 | user Colab + ChatGPT audit | **Fresh-checkpoint paired retention replication** — identical continuation data for HIGH/LOW across 3 independent acquisitions | ARK-007R | HIGH collapse90 9/12 vs LOW 0/12; direction on all acquisitions

2026-09-08 | ChatGPT audit | **State must be explicit in optimizer interventions** — preservation and reacquisition are distinct objectives | ARK-007R + ARK-010 | LOW protects when capability is present; HIGH recovers more often after instability

2026-09-08 | ChatGPT audit | **Orthogonal intervention diagnostics** — never call a perturbation query-specific if it simultaneously changes presentation order | ARK-009 red-team | future binding diagnostic split into QUERY_ONLY / ORDER_ONLY / QUERY+ORDER
