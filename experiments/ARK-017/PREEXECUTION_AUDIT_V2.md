# ARK-017 V2 PRE-EXECUTION AUDIT

Date: 2026-09-09

## Scope

Static review of:
- `experiments/ARK-017/PLAN.md`
- `experiments/ARK-017/PLAN_V2_ADDENDUM.md`
- `experiments/ARK-017/run_ark017.py`
- `experiments/ARK-017/run_ark017_v2.py`
- reused ARK-014 binding implementation
- reused Discovery V7 delta-cap primitive

No ARK-017 GPU result existed at audit time.

## V1 defect found before execution

**Sparse replay treatment-fidelity defect.** The V1 runner selected exactly four replay positions but rendered them with ARK-014's six-way augmentation, which includes identity order. Therefore “4/64 alternative-order examples” was not guaranteed.

Status: **FIXED BEFORE EXECUTION** in V2.

V2 selects only among the five non-identity permutations for sparse replay, asserts selected replay rows differ from canonical, and keeps all non-replay rows byte-identical canonical.

## V2 execution checks

### Scientific identity
- frozen ARK-014 binding manifest required;
- fresh acquisition seeds unchanged: 2601/2702/2803;
- continuation semantic seeds unchanged: 10801/10802;
- primary 8k horizon unchanged;
- CONTROL decides qualification; SEALED remains measurement-only;
- primary verdict still comes only from the original six-arm core experiment;
- secondary dose screens are conditional, preregistered and cannot rewrite the primary verdict.

### Fork identity
V2 uses the exact acquisition snapshot for every matched arm. LOW runs first only to materialize a movement reference trace; every later arm restarts from the same acquisition snapshot and consumes the same semantic IDs.

### Cap semantics
The cap treatment intentionally keeps HIGH AdamW hyperparameters/moment evolution while limiting the **applied parameter delta** after each step. It therefore tests whether applied movement is sufficient to explain protection; it is not claimed to reproduce LOW optimizer state.

### Replay semantics
Sparse replay changes presentation order only. Key/value semantics, answer, semantic IDs, batch size and update count are unchanged. Full augmented HIGH remains a specificity reference and may include identity permutations exactly as the acquisition distribution did.

### Runtime integrity
V2 smoke is required to test:
- manifest identity;
- exact non-canonical sparse replay;
- CUDA forward/backward;
- model + optimizer snapshot restore;
- identical next update from identical forks;
- multi-step LOW-reference trace consumed by capped HIGH;
- finite optimizer state.

The Colab launcher must fail closed unless smoke passes.

## Remaining limitations

1. **Micro scale.** Even a strong result is not a production optimizer law.
2. **Cap intervention is nonstandard.** AdamW moment state is not rescaled when the parameter delta is capped; that is intentional for causal separation but requires later optimizer-native implementation if promoted.
3. **One sparse replay family.** 1/16 is primary; 1/32 and 1/64 are only conditional secondary efficiency screens.
4. **Budget gates.** A 240-minute safety budget can yield a partial secondary screen. Primary matched sets take precedence.
5. **No architecture claim.** The experiment studies training dynamics on the existing Micro subject.

## Audit verdict

**READY_FOR_PINNED_GPU_SMOKE AFTER COLAB LAUNCHER IS BOUND TO THE V2 RUNNER COMMIT.**

No scientific outcome is claimed by this audit.