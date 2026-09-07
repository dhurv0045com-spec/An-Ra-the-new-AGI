# FAILURES — every experiment failure and falsified claim, dated, attributed

Format: `date | agent | failure | evidence | lesson`

---

2026-09-06 | arkenstone-agent | **ARK-004A precursor direction INVERTED** — claimed "higher tens-selectivity -> EARLIER generalization" but rho(+0.60) vs G90_step means LATER | REANALYSIS.py from raw receipts; original LOO not reproducibly specified | recompute correlation signs explicitly; never narrate direction without stating the target variable

2026-09-06 | arkenstone-agent | **ARK-004A selectivity is a MARKER not a PRECURSOR** | REANALYSIS.json temporal-ordering table | temporal ordering must be checked before claiming precursor

2026-09-06 | arkenstone-agent | **ARK-003 curriculum arm DELAYED memorization** — M99 5400 vs 1400 for flat | RESULT_B.json | easy-first staging is not free

2026-09-06 | arkenstone-agent | **ARK-003 aligned teacher showed NO acceleration under equal wall budget**; step confound remains | RESULT_C/D.json | step/token/compute matching must be explicit

2026-09-06 | arkenstone-agent | **ARK-005 weight-decay removal does NOT prevent post-G90 decay** | RESULT_C_seed606.json | H-WD not supported at micro scale

2026-09-06 | arkenstone-agent | **ARK-005 EMA consolidation does NOT prevent post-G90 decay** | RESULT_D_seed606.json | simple EMA consolidation insufficient

2026-09-06 | arkenstone-agent | **ARK-005 LR x0.1 only DELAYS collapse** | RESULT_B_seed606.json | stronger LR intervention needed

2026-09-06 | arkenstone-agent | **ARK-001 harness ByteVocab PAD/BOS asymmetry** | superseded artifact | encode/decode must share answer contract

2026-09-06 | arkenstone-agent | **ARK-001/002a commutation leakage** — 48 sorted-pair overlaps | ERRATUM_002a.json | assert commutation-free split

2026-09-06 | arkenstone-agent | **ARK-003 PLAN referenced an uncommitted manifest** | ERRATUM_003.json | bound artifacts must exist before execution

2026-09-06 | arkenstone-agent | **ARK-003 wall-time confound** | ERRATUM_003.json | step-matched accounting required

2026-09-06 | arkenstone-agent | **ARK-001 historical receipt was mutated to satisfy verifier** | restored historical file; verifier redesigned | verifier must accommodate immutable history

2026-09-06 | arkenstone-agent | **ARK-005 plan_commit_sha stored short** | RESULT receipts | receipts must store full 40-char SHA

2026-09-06 | arkenstone-agent | **ARK-005 S1 composition eval vacuous** | pre-execution addendum | audit combinatorial coverage before defining diagnostic

2026-09-08 | ChatGPT audit | **ARK-009 query-swap is not a one-variable intervention** — V5 code reverses the fact order and changes the queried key simultaneously | experiments/ARK-009/ANALYSIS.md + V5 runner | separate QUERY_ONLY, ORDER_ONLY and QUERY+ORDER; do not diagnose query-conditioning from a composite perturbation

2026-09-08 | user Colab T4 | **ARK-009 strict transfer qualification not met** — ordinary held-out exact reached 1.0, but composite diagnostic peaked only 0.383/0.360 | experiments/ARK-009/RESULT.json | retention-transfer question was never reached; do not call it a transfer failure

2026-09-08 | user Colab T4 | **ARK-010 immediate-low-LR recovery hypothesis falsified** — HIGH recovered sustained G90 8/9, LOW only 2/9 | experiments/ARK-010/RESULT.json | intervention utility depends on capability state; preservation and reacquisition are different objectives

2026-09-08 | ChatGPT audit | **`collapse90` was too easy to narrate as irreversible forgetting** — most HIGH recovery continuations regain sustained G90 | experiments/ARK-010/ANALYSIS.md | call it a post-G90 instability episode unless irreversibility is separately demonstrated
