# ARK-004A RED TEAM

Attacks and answers:
- n=4 LOO is weak power -> acknowledged in PLAN and ANALYSIS; claim level is
  TENTATIVE SUPPORTED, not REPLICATED. ARK-004B replication across arms adds
  seeds.
- Selectivity could encode training time -> the baseline M99 step (rho 0.00,
  2/4) directly tests "just time"; selectivity beats it 4/4 LOO. Additionally
  selectivity is measured over a FIXED window relative to each seed's OWN M99,
  normalizing time.
- Margin rho=-1.00 with n=4 -> p(two-sided, n=4, no ties) ~ 1/12 per
  permutation; not claimed as a discovery; recorded as anti-directed.
- Probes could perturb training -> probes are no_grad, eval-only, fixed sets;
  training loop is byte-identical to ARK-002B's.
- Collapsed seed (101) could indicate eval bugs -> greedy evaluator is
  unit-tested; the collapse is itself replicated behavior across steps (OOD
  fell from >=0.9 sustained to 0.188 over 8000 steps — gradual, not a
  one-eval glitch; see trajectory).
- Commutation leakage -> frozen manifest is the ARK-002B commutation-free
  split (zero sorted-pair overlap, asserted in code and tests).
- Post-hoc window choice -> the M99+2000 window was preregistered in PLAN.md
  (committed 9e2fe85) before execution.
