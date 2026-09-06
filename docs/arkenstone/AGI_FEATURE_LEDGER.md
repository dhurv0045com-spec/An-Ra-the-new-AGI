# AGI FEATURE LEDGER (Arkenstone)

One entry per proposed feature/mechanism with honest novelty, evidence, cost,
and status. "AGI feature" is never claimed casually.

| Feature | Prior status | Novelty class | Capability target | Mechanism | Evidence | Cost | Confidence | Core status |
|---------|--------------|---------------|-------------------|-----------|----------|------|------------|-------------|
| Lift-off dose measurement instrument | No branch measured lift-off | NEW_MEASUREMENT (for program) | locate whether/where symbolic capability can move at all | dose-response curves at micro scale | ARK-001+002a: lift-off 200-400 steps; OOD saturation at ~9k steps; replicated 2 seeds | CPU-minutes | HIGH | PROMOTED (standing instrument) |
| Structural-band holdout + per-position decomposition eval | absent in branch receipts | NEW_MEASUREMENT | distinguish memorization from algorithm extraction | band-split + per-digit accuracy | ARK-001/002a: ones-vs-tens asymmetry (0.91/0.37 mid-transition; 1.0/1.0 saturated) | none (metrics) | HIGH | PROMOTED (standing instrument) |
| Vocabulary reduction for symbolic training | T1D arm E designed, never run | EXTENSION | faster/better symbolic learning | smaller embedding, denser signal | ARK-001: byte-level 24,576 vocab lifts off at SAME step (200) as 19-token vocab | none | HIGH (at micro) | REJECTED at micro scale; revisit only at P35+ scale |
| Micro-teacher rows (digit/subproblem supervision) | citadel T1D arm C, unexecuted | ALREADY_KNOWN (designed) | tens-column binding is the grokking bottleneck (ARK-001/002a asymmetry) | teacher decomposes the binding-heavy position | pending ARK-003 | TBD | — | CANDIDATE (design now evidence-backed) |
| Curriculum (easy->hard tiers) | citadel T1D arm B, unexecuted | ALREADY_KNOWN (designed) | reach OOD lift-off within budget at higher tiers | staged exposure | pending ARK-002b/003 dose-ratio map | TBD | — | CANDIDATE |
| Interference-pair binding generator v2 (data) | cymek, test-asserted | REPRODUCTION_ONLY (independent verification) | trustworthy cognition training data | pair-preserving interference | BINDING-V2-REDTEAM receipt: CONSISTENT, 3 seeds + trained escalation baseline | CPU-minutes | MEDIUM-HIGH | VERIFIED (cymek's artifact) |
