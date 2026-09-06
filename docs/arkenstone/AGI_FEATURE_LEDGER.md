# AGI FEATURE LEDGER (Arkenstone)

One entry per proposed feature/mechanism with honest novelty, evidence, cost,
and status. "AGI feature" is never claimed casually.

| Feature | Prior status | Novelty class | Capability target | Mechanism | Evidence | Cost | Confidence | Core status |
|---------|--------------|---------------|-------------------|-----------|----------|------|------------|-------------|
| Lift-off dose measurement instrument | No branch measured lift-off | NEW_MEASUREMENT (for program) | locate whether/where symbolic capability can move at all | dose-response curves at micro scale | ARK-001+002a: lift-off 200-400 steps; OOD saturation at ~9k steps; replicated 2 seeds | CPU-minutes | HIGH | PROMOTED (standing instrument) |
| Structural-band holdout + per-position decomposition eval | absent in branch receipts | NEW_MEASUREMENT | distinguish memorization from algorithm extraction | band-split + per-digit accuracy | ARK-001/002a: ones-vs-tens asymmetry (0.91/0.37 mid-transition; 1.0/1.0 saturated) | none (metrics) | HIGH | PROMOTED (standing instrument) |
| Vocabulary reduction for symbolic training | T1D arm E designed, never run | EXTENSION | faster/better symbolic learning | smaller embedding, denser signal | ARK-001: byte-level 24,576 vocab lifts off at SAME step (200) as 19-token vocab | none | HIGH (at micro) | REJECTED at micro scale; revisit only at P35+ scale |
| Micro-teacher rows (digit/subproblem supervision) | citadel T1D arm C unexecuted; ARK-003 arm C EXECUTED | ALREADY_KNOWN (designed) | tens-column binding bottleneck | teacher decomposes the binding-heavy position | ARK-003: no acceleration at equal wall budget; confound: C/D ran ~45% fewer steps (suffix token cost) — step-matched rerun open | recorded | MEDIUM | CANDIDATE (null-in-budget; confounded) |
| Curriculum (easy->hard tiers) | citadel T1D arm B unexecuted; ARK-003 arm B EXECUTED | ALREADY_KNOWN (designed) | staged exposure | easy-first staging | ARK-003: delays T2 memorization (M99 5400 vs 1400), zero OOD at box end | recorded | HIGH (negative at micro) | REJECTED at micro scale |
| Interference-pair binding generator v2 (data) | cymek, test-asserted | REPRODUCTION_ONLY (independent verification) | trustworthy cognition training data | pair-preserving interference | BINDING-V2-REDTEAM receipt: CONSISTENT, 3 seeds + trained escalation baseline | CPU-minutes | MEDIUM-HIGH | VERIFIED (cymek's artifact) |
| Column-selectivity precursor + dense transition instrumentation | absent | NEW_PROGRAM_MEASUREMENT | predict/shorten the post-memorization delay | counterfactual factorization probes | ARK-004A: LOO 4/4, beats time/loss; post-G90 instability documented | probe cost ~2s/eval | MEDIUM | CANDIDATE (ARK-004B gate OPEN) |
