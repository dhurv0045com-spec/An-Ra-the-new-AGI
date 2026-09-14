# Packet A runner handoff

Source baseline: Arkenstone `933d4f3` (working tree included the shared runtime changes from the preceding executor). Owned files changed: `experiments/ARK-012/run_ark012.py`, `experiments/ARK-013/run_ark013.py`, `tests/test_discovery_v6_campaigns.py`, and this report.

## Implemented

- ARK-012 counts only `EXECUTED` sources with the exact frozen schedule key set, all required sealed metrics, and `completed_steps == 8000`. A late switch earns protection only when that same arm has G90 confirmation, gains over `LOW_IMMEDIATE`, and has no recurrent sealed drop. The HIGH collapse signal alone cannot satisfy this condition.
- ARK-013 counts only triplets with the exact three arm keys, all four metric groups, and `completed_steps == 12000`. Adaptive Pareto qualification requires final and area metrics to agree in the same direction, using the frozen 0.05 noninferiority and 0.10 gain boundaries. The code labels this as a conservative implementation interpretation, not a new preregistered law.
- Both campaigns emit `BUDGET_BLOCKED` results when campaign entry reserve is unavailable (35 minutes for ARK-012, 50 minutes for ARK-013), preserving skipped source/triplet rows. Existing within-campaign stop gates remain 15 and 18 minutes respectively.
- Completed schedule/arm evidence is written before a later arm failure. The inherited answer-loss objective remains unchanged; the shared runtime review flags its BOS/padding supervision limitation.

## Verification

Command:

```text
python -m unittest discover -s tests -p test_discovery_v6_campaigns.py -v
```

Result: 6 tests passed, CPU-only, approximately 0.13 seconds. Coverage includes deterministic T3CARRY construction and firewall invariants, ARK-012 same-arm recurrence protection and short-horizon rejection, ARK-013 area-only false-positive rejection and forward-direction gain acceptance, and completed evidence preservation with fake runtimes.

No training, GPU, network, dependency installation, or Git publishing was performed. Scientific campaign outcomes remain unrun and unsupported by this packet. The root agent should run the integrated runtime suite and final preflight after reviewing these files.

Primary integration note: after this six-test packet, the chief added explicit reverse-direction Pareto, budget, duplicate-identity and premature-negative checks. The final suite and final-source acceptance are recorded in HANDOFF.md.
