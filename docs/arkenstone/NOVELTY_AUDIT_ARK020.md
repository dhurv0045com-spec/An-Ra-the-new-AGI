# ARK-020 NOVELTY AUDIT (mission §14) — written before execution

Honest classification against the continual-learning literature. "Novel because our code
is new" is explicitly rejected as a standard.

## Mechanism-by-mechanism

| mechanism | closest known work | classification |
|---|---|---|
| Experience replay against catastrophic forgetting | rehearsal buffers (standard CL) | **ALREADY_KNOWN** |
| Sparse replay (1/16–1/64 of batch) protecting an acquired capability | experience replay; buffer-size studies | **EXTENSION / NEW_EMPIRICAL_DISCOVERY at micro scale** — the extreme sparsity finding (1 slot/batch sufficient, R2 secondary 0/3) is far below typical rehearsal budgets and is receipt-backed on a real-text substrate |
| Replay permitting *larger* parameter movement than failing narrow training | none directly; related to "replay ≠ regularization" discussions | **NEW_EMPIRICAL_DISCOVERY** (R2 primary; falsifies simple movement-magnitude accounts) |
| Update-magnitude cap as protection (trust-region on applied delta) | trust-region methods, GEM-style gradient constraints | **COMBINATION** (known family; the applied-delta projection with LOW-LR shadow calibration is a concrete new instantiation) |
| Two-lever sufficiency (cap OR replay) | EWC (penalty), GEM (projection), replay (data) | **NEW_EMPIRICAL_DISCOVERY** at this scale: each lever independently sufficient in matched forks |
| State-conditional replay allocation by capability health | adaptive/conditional replay; MIR selects maximally-interfered examples; loss-driven scheduling | **COMBINATION / EXTENSION** — risk-based per-capability allocation with margin+degradation triggers is a known idea class; the preregistered margin/degradation trigger set and de-escalation ladder are new instantiations, not a new mechanism |
| De-escalation after sustained recovery (protection duty reduction) | less explored; some adaptive CL schedules | **EXTENSION** (explicit prospective de-escalation + duty accounting) |
| Capability registry (multi-capability health tracking) | CL task-metric monitoring | **COMBINATION** (engineering synthesis, not new science) |
| Reactive vs predictive controller comparison | predictive scheduling appears in online CL heuristics | **EXTENSION** — the preregistered head-to-head with prevention/recovery distinction is the contribution |
| PREVENTION vs RECOVERY vs RETENTION as separate reported quantities | occasionally discussed, rarely separated | **EXTENSION** (measurement discipline) |
| Closed-loop "Guardian" framing | control-theoretic CL; homeostatic plasticity analogies | **COMBINATION** |
| Multi-skill sequential battery with structurally distinct skills | CL benchmarks (e.g., split tasks) | **ALREADY_KNOWN** as a design pattern; the specific C (rule induction with never-trained sealed keys) subject is a new construction |

## What would count as genuinely new after ARK-020 executes

1. The **protection-cost slope** (duty/slots as retained capabilities grow 1→2→3) compared
   Guardian vs static — no standard benchmark reports this; if the Guardian's slope is
   flatter at matched protection, that is a **NEW_EMPIRICAL_DISCOVERY** candidate.
2. **PREDICTIVE_ADDS_VALUE**: prevention at equal cost beating reactive recovery — if the
   flag fires, it upgrades state-conditional replay from heuristic to validated controller
   property (**NEW_EMPIRICAL_DISCOVERY** at micro scale).
3. Cross-family protection (does replay trained on binding protect a rule-induction
   capability?) — unaddressed in the sparse-replay literature we know of; any clean
   positive or negative is reportable evidence.

## Explicit non-claims

No claim of a universal replay law, no production scheduler authorization, no novelty for
replay itself, no AGI-adjacent reading of "capability registry". Sparse replay is a known
tool; what this program adds is measured dose boundaries, mechanism separation, and
controller-level evidence on a real-text proxy.
