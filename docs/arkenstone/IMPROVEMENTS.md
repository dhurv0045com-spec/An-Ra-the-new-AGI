# IMPROVEMENTS — every adopted improvement, dated, attributed

Format: `date | agent | improvement | source | evidence`

---

2026-09-06 | arkenstone-agent | **Lift-off dose mapping instrument** — first measurement of steps-to-exact-match in program history | self-discovered (no branch had it) | ARK-001: 200-400 steps for T1; replicated seeds 13/29

2026-09-06 | arkenstone-agent | **Commutation-free dataset manifest** — removed 48 sorted-pair train/test overlaps from the T2 holdout | self-found during ARK-002B prep | split_sha 0dd930569704; zero overlap asserted in tests

2026-09-06 | arkenstone-agent | **Sustained-threshold metrics (M99/G50/G90/G95)** — replaced max-snapshot claims with ≥3-consecutive-eval rule | self-designed | ark_metrics.py + 6 unit tests

2026-09-06 | arkenstone-agent | **Plan-commit-before-training discipline** — every PLAN is committed and pushed before execution; every RESULT binds plan_commit_sha | adopted from citadel's agent.md rule + BRAMASTRA's manifest-before-training pattern | ARK-002B PLAN 90818c1, ARK-003 PLAN 9d06a49, ARK-004A PLAN 9e2fe85, ARK-004B PLAN aa84179, ARK-005 PLAN 3d5e97a

2026-09-06 | arkenstone-agent | **Fork-at-trigger experimental design** — one shared pre-trigger training cloned into N arms at the detected transition, removing step-count confounds by construction | self-designed (needed after ARK-003 wall-time confound) | committed addendum c7dd792; implemented in run_005.py

2026-09-06 | arkenstone-agent | **Terminal/EOS supervision as a training contract** — the model must be taught when the answer ends | adopted from BRAMASTRA (0/32→32/32, 2 seeds); Citadel T1C never supervised a terminator | AGI_FEATURE_LEDGER; Arkenstone harness already supervised EOS (verified)

2026-09-06 | arkenstone-agent | **Capacity-accounting correction** — MINI/MID scale arms are ~95% embedding; "2.3x scale" differs by <0.1M non-embedding params | adopted from BRAMASTRA EVIDENCE.md audit | interpretation rule recorded in AGI_FEATURE_LEDGER

2026-09-06 | arkenstone-agent | **Continuation probe (save→update→reload→same update→verify)** — every experiment receipt can include a determinism proof | adopted from BRAMASTRA experiment.py L180-227 | ark_provenance.py; 4 unit tests

2026-09-06 | arkenstone-agent | **Source-snapshot in receipts** — exact runner code captured at execution time, not just hash | adopted from BRAMASTRA source_snapshot/ pattern | ark_provenance.py; 2 unit tests

2026-09-06 | arkenstone-agent | **Fixed transparent research-nomination rule** — code that names the next experiment from measured evidence | adopted from BRAMASTRA decide() pattern (GAP 1 partial fix) | ark_provenance.py nominate_next(); 4 unit tests

2026-09-06 | arkenstone-agent | **Content-hash ledger verification** — governed files hash-stamped; drift detected mechanically; no commit self-reference | self-designed (GAP 2/4 fix) | verify_ledgers.py v2; caught real ARK-001 receipt canonicalization drift on first run

2026-09-06 | arkenstone-agent | **Dual-scheme legacy receipt verification** — accommodates both compact and default-separator JSON canonicalization without rewriting history | self-designed after historical-receipt mutation was caught | verify_ledgers.py v2; restores immutability

2026-09-06 | arkenstone-agent | **Single-session Colab notebook** — full program device-adaptive TPU/CUDA/CPU, resumable, budgeted, receipts+download | Citadel's notebook pattern, Arkenstone content | experiments/COLAB/arkenstone_all.ipynb; harness validated on GPU
