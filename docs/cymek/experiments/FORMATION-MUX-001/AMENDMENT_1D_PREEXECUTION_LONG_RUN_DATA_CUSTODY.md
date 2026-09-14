# FORMATION-MUX-001 — Amendment 1D (pre-execution)

Status: **PROSPECTIVE / NO OFFICIAL FORMATION-MUX OUTCOMES OBSERVED**

This amendment is made before the first official Kaggle T4 x2 campaign run. It does not alter the causal questions, arms, matched seeds, primary metrics, sealed endpoint, verdict thresholds, physical vocabulary, optimizer treatments, or representation contrast frozen by S4.

## Why this amendment exists

The S4 surface contains 600 training examples per family (3,600 total). CS-MECH-002 presents 2,000 updates × 16 rows = 32,000 training examples per arm, so the old surface would be reused about 8.9 times. REP-FORM-003A similarly consumes 500,000 processed tokens and can cycle through the small training surface. That is unnecessary for a deterministic synthetic mechanism experiment and can turn formation into memorization of a small surface.

The operator was also intentionally conservative about a single Kaggle session wall. The user has sufficient compute time and prefers the campaign to continue for as many sessions as necessary rather than shorten the science. Therefore session duration becomes purely an engineering/resume concern, not a scientific stop rule.

## Prospective changes

1. **Training surface expansion only.** Generate 10,000 unique examples per family = 60,000 training rows. Development remains 80/family (480 total) and sealed remains 120/family (720 total). The grammar, tokenizer, family definitions, contamination rules, shortcut screens, split firewall, and deterministic seed remain unchanged.
2. **Reason for 60,000 rows.** CS-MECH consumes 32,000 row presentations per arm, so it cannot wrap the 60,000-row permutation. REP-FORM consumes 500,000 processed tokens; 60,000 rows also gives enough headroom to avoid the extreme repeated-surface regime while keeping the manifest small enough for Kaggle RAM and fast startup.
3. **Checkpoint durability.** CS-MECH writes an exact-resume checkpoint every 200 optimizer updates. REP-FORM remains token-matched scientifically and writes exact-resume checkpoints every 10,000 processed non-padding tokens, which is intentionally more frequent than its scientific evaluation cadence. Every checkpoint also writes a human-readable progress snapshot containing update count, processed/supervised token counts, latest development measurement already available, current formation summary, timing, and checkpoint hash.
4. **No adaptive science.** Progress snapshots are diagnostic only. They cannot change arms, thresholds, seeds, endpoints, token/update budgets, data, or sealed access. Official verdicts still use the frozen development formation metric plus one-time sealed endpoint.
5. **Long-run execution.** The campaign may span multiple Kaggle sessions. A session should run until near the platform wall, then stop launching new work, atomically package exact-resume state, and continue in a later session from the same frozen science identity. The campaign itself has no scientific wall-time truncation.
6. **No raw sealed persistence.** S4 sealed custody remains unchanged: workers only receive training + development; sealed rows are regenerated and commitment-verified only after development is frozen.

## Data budget

Frozen unique surface:

- training: 60,000 rows (10,000 × 6 families)
- development: 480 rows (80 × 6)
- sealed: 720 rows (120 × 6)
- total latent examples: 61,200

This is a development-scale causal dataset, not a pretraining corpus. The scientific exposure remains 2,000 updates for each CS-MECH arm and 500,000 processed tokens for each REP-FORM arm.

## Claim ceiling

Unchanged: development-scale causal mechanism/representation evidence only. This amendment does **not** authorize a production vocabulary change, PRE500M, 250M, 500M, or any capability/superiority claim.