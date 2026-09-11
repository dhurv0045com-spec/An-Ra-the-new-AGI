# ARK-020 V3 — RED TEAM (pre-freeze, mission #34)

Every attack below was mounted against live V3 code. Dispositions are backed by tests.

| attack | disposition |
|---|---|
| C ordinal shortcut | BROKEN by independent XM/MY namespaces; measured same-ordinal rate 0.3377 (~1/3) over 3000 canonical prompts; per-mode test caps alignment < 0.60; tied-renderer mutation scores 1.0000 proving the detector works |
| C direct X→Y shortcut | impossible: (x,y) never co-queried in training; disjoint sealed factsets (test) |
| C token-frequency shortcut | blocked: bijective Y assignment per factset (test asserts all 3 Y per factset) |
| C template/position shortcut | distinct Trace template; answer never an intermediate (test) |
| C split leakage | disjoint 5-way chain-set splits (test) |
| D accidentally forward | facts presented (owner, object); query object; answer owner; ordered (object→owner) pair never presented (test); V2-style pre-reversal caught by mutation test |
| D query-answer exposure | (q, a) adjacency never presented (test); mutation shows V2 build exposes it |
| D token-role leakage | owner/object group membership asserted (test) |
| Controller SEALED contamination | structural: controller entry points take CONTROL state only (test pins) |
| Validation overuse | validation only for prospective confirmation gates |
| Protection triggered by future info | controller observes only current CONTROL observations |
| Stale V4 reuse | parent/dose reuse identity-checked; contract tests pin V4 signatures and exercise real call shapes |
| Checkpoint mismatch | identity = schema + seeds(B/C/D) + arm + dose + parent_sha + task_hash + cap16x; 5 corruption mutations fail closed (test) |
| Resume duplication | dedupe_trajectory; per-session PARTIAL receipts |
| RNG mismatch | cpu_rng + cuda_rng hashed in smoke; semantic (not byte) hashing per mission #4 |
| Stream-seed mismatch | per-phase seeds bound into identity; stream-independence test (task samples AND real-text starts) |
| Notebook Drive state mismatch | cell 0 mounts Drive BEFORE clone/scan; drive_ok=False → STOP — DRIVE UNAVAILABLE |
| Concurrent sessions | structured lock; ACTIVE → WAIT; MALFORMED → STOP; test covers all three |
| Global/phase step confusion | phase-relative primary; golden tests distinguish median-vs-median from per-run |
| Median/per-run confusion | preregistered median-vs-median; golden test with one slow set passes, all-slow fails |
| Incomparable static baseline | static arms share task exposure; duty/replay accounted |
| Cherry-picked sets | all seeds frozen in preregistration |
| Incomplete result accepted | decide() returns INCONCLUSIVE_INCOMPLETE_MATCHED_SETS (golden test) |
| Calling composition "reasoning" | forbidden by claim ceiling (PLAN, PREREGISTRATION) |
| Calling recovery "prevention" | PREVENTION/RECOVERY/RETENTION reported separately; never collapsed |
| Calling proxy multi-skill learning "AGI" | authorization flags all false; claim ceiling explicit |

VERDICT: no unresolved fatal defect. Remaining accepted risks: C formation speed at 12
slots (formation gate converts failure to INCONCLUSIVE), 3–5 session operator load,
V4 evidence still provenance-marked until byte-level bundle audit.
