# FAILURES — every experiment failure and falsified claim, dated, attributed

Format: `date | agent | failure | evidence | lesson`

---

2026-09-06 | arkenstone-agent | **ARK-004A precursor direction INVERTED** — claimed "higher tens-selectivity → EARLIER generalization" but rho(+0.60) vs G90_step means LATER | REANALYSIS.py from raw receipts; original LOO not reproducibly specified | recompute correlation signs explicitly; never narrate direction without stating the target variable

2026-09-06 | arkenstone-agent | **ARK-004A selectivity is a MARKER not a PRECURSOR** — selectivity move does not precede OOD emergence (after P10 in 2/4 seeds) | REANALYSIS.json temporal_ordering table | temporal ordering must be checked before claiming "precursor"

2026-09-06 | arkenstone-agent | **ARK-003 curriculum arm DELAYED memorization** — M99 at 5400 vs 1400 for flat; zero OOD at box end | RESULT_B.json trajectory | easy-first staging is not free; it consumes budget that flat training uses better

2026-09-06 | arkenstone-agent | **ARK-003 aligned teacher showed NO acceleration** — C/D ran ~5300 steps vs A's 9600 (suffix token cost) | RESULT_C/D.json; wall-time confound documented | step-matched accounting is required before any teacher-fails claim

2026-09-06 | arkenstone-agent | **ARK-005 weight-decay removal does NOT prevent post-G90 decay** — identical trajectory to control | RESULT_C_seed606.json | H-WD not supported at micro scale

2026-09-06 | arkenstone-agent | **ARK-005 EMA consolidation does NOT prevent post-G90 decay** — identical trajectory to control | RESULT_D_seed606.json | simple consolidation machinery is insufficient

2026-09-06 | arkenstone-agent | **ARK-005 LR-decay only DELAYS collapse, does not prevent it** — RET90 0.464 vs 0.143, but still collapsed | RESULT_B_seed606.json | H-LR weak-positive; stronger intervention needed

2026-09-06 | arkenstone-agent | **ARK-001 harness bug: ByteVocab PAD/BOS asymmetry** — loss 0.0 / exact 0.0 (impossible signature caught) | superseded artifact in git history | encode/decode must share the answer-encoding contract

2026-09-06 | arkenstone-agent | **ARK-001/002a commutation leakage** — 48 sorted-pair overlaps in the T2 holdout | ERRATUM_002a.json | sorted-pair filter now asserted in manifest builder

2026-09-06 | arkenstone-agent | **ARK-003 PLAN referenced an uncommitted manifest** — TASK_MANIFEST.json first entered git in a later commit | ERRATUM_003.json; git cat-file verified | bound artifacts must be committed before or in the PLAN commit

2026-09-06 | arkenstone-agent | **ARK-003 wall-time confound** — PLAN said equal steps; C/D got ~45% fewer steps from suffix token cost | ERRATUM_003.json | step-matched accounting required for causal claims

2026-09-06 | arkenstone-agent | **ARK-001 RESULT receipt_sha256 mutated** — gap-review commit changed the historical hash to satisfy a new verifier | restored from aa84179; verifier redesigned to dual-scheme | history is immutable; the verifier accommodates history

2026-09-06 | arkenstone-agent | **ARK-005 plan_commit_sha is 7 chars not 40** — runner stored short SHA | RESULT_* receipts | always store full 40-char SHA in receipts

2026-09-06 | arkenstone-agent | **ARK-005 S1 composition eval vacuous** — train pool covers all 55 ones-pairs; empty set caused crash | pre-execution addendum in PLAN | audit pool coverage before defining composition metrics
