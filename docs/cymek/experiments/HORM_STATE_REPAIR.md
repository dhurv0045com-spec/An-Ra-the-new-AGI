# HORM state-construction repair (engineering only)

Baseline: `1cb7a58`. Historical experimental plans, results and custody
inventories are unchanged. This repair is not a rerun or capability claim.

## Changes

- HALState accepts list/tuple history inputs, owns a tuple container, rejects
  more than 64 entries, and copies each entry into a read-only mapping.
- Entry keys must be hormone names or `step`; values are converted to finite
  floats. Partial entries remain accepted. This is structural validation,
  not authentication of historical measurements or chronology.
- The direct HAL wrapper rejects temperature bounds outside finite `(0, 1.5]`,
  matching the existing configuration rule. Default and valid-bound numerical
  behavior is unchanged by the guard.

## Execution evidence

- Original history regressions: 3 failed / 15 passed after fixing test import
  mistakes. Failures were missing rejection and caller mutation leaking.
- Empty-list ownership regression: 1 passed after tuple normalization.
- Container validation: 4 failed / 35 passed, then 39 passed after the guard.
- Final affected local CPU suite: **53 passed in 6.70s**, exit 0:
  `test_hormonal_contracts.py`, `test_hormonal_state.py`,
  `test_hormonal_integration.py`, `test_hormonal_integration_canary.py`.
- Includes actual tiny-model state-dict save/load, exact recompute gradients,
  forward-state mutation before backward, and no-op/active projection checks.
- Wrapper tests were added after the guard; this part was not test-first.
  No claim of a pre-repair failing wrapper pytest run is made.

Resource check before bounded tests: 16 logical CPUs, 4.03 GiB available RAM,
GPU idle (0 MiB allocated). Two CPU threads; no campaign or GPU training.

## Compatibility and provenance limits

MappingProxyType entries are read-only but are not directly JSON/pickle or
`dataclasses.asdict` serialization targets. Use `dict(entry)` for explicit
log export; existing `as_receipt()` excludes the log. Tiny-model state-dict
roundtrip is tested, not arbitrary HALState pickle/deepcopy compatibility.

The two repaired source files no longer match hashes recorded by historical
HORM-001/002 preregistrations. Original-source replay requires original sources;
these changes must not be presented as the frozen experimental implementation.
The earlier custody reports remain snapshots at their recorded identities,
not fresh attestations of the repaired working tree. Checkpoints remain
unrecovered and no original scientific verifier was made to pass.
