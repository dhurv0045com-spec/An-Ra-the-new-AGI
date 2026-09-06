# ARK-005 ANALYSIS (executed: 8 runs, 2 seeds x 4 arms, 24k steps each)

## Results (RET90 = fraction of post-G90 evals >= 0.90; gap = peak - final)

| Arm | seed 505 (trigger 16200) | seed 606 (trigger 18600) |
|-----|---------------------------|---------------------------|
| A control | RET90 0.975, gap 0.002, final 1.0 — STABLE | RET90 0.143, collapse@19200, final 0.822 — DECAYING |
| B lr-decay x0.1 | RET90 1.0, gap 0.0, final 1.0 — STABLE | RET90 0.464, collapse@21200, final 0.822 — DECAYING (delayed ~2000 steps) |
| C wd-removal | RET90 1.0, gap 0.0, final 1.0 — STABLE | RET90 0.107, collapse@19200, final 0.812 — identical to control |
| D EMA-0.999 | RET90 1.0, gap 0.0, final 1.0 — STABLE | RET90 0.107, collapse@19200, final 0.822 — identical to control |

## Findings
1. **Ceiling effect on seed 505**: the control consolidated on its own (RET90
   0.975) — no arm could demonstrate an improvement. The experiment only had
   discriminating power on seed 606, where the control decayed.
2. **On the decaying seed, LR-decay is the only arm that moved**: collapse
   delayed ~2000 steps (21200 vs 19200), RET90 3.2x control (0.464 vs 0.143).
   Direction consistent with H-LR (constant 1e-3 too high post-transition).
   But it DELAYED the collapse; it did not prevent it — B also collapsed
   eventually. Weak, single-seed evidence.
3. **Weight-decay removal (C) and EMA consolidation (D) had NO effect** — both
   track the control's trajectory exactly on seed 606. H-WD and the simple
   consolidation machinery are not supported at this scale.
4. **No arm meets the preregistered consolidation-candidate bar** (RET90 >=
   0.95 on both seeds). Per the futility rule, the retention program at micro
   scale records a WEAK-POSITIVE hint for LR decay only.

## Provenance notes (honest flaws)
- The receipts were produced by the original independent-run implementation
  (8 parallel GPU processes), not the fork-at-trigger addendum; the addendum
  was committed while these runs were already in flight and does not describe
  them.
- plan_commit_sha in these receipts is a 7-char short SHA ("3d5e97a"), not the
  full 40-char hash (a runner bug); the plan commit is unambiguously
  resolvable, but future receipts must store the full hash.
- 8-way GPU contention slowed all runs equally; training math was unaffected
  (identical seeds/config produce identical trajectories regardless of wall
  speed), so the retention measurements are valid.

## Verdict
H-LR: WEAK-POSITIVE (delay, single decaying seed). H-WD: NOT_SUPPORTED.
H-EMA: NOT_SUPPORTED. H-STATE/H-DATA: untested. Retention remains an open
problem; the next discriminating experiment should test LR schedules more
aggressively (e.g. 10x decay, or drop to 0 at trigger) on decaying seeds only.
