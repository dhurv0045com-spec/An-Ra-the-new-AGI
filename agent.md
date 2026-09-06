# AGENT.md — Citadel handover (machine-readable status for the operator's agent)

> Convention (binding): rewritten at the END of every Citadel work cycle,
> committed to the `citadel` branch ONLY, then `git push origin citadel`.
> Other branches are read-only audit inputs — never modified, never pushed.
> A CPU/CUDA run is NEVER a TPU result. No fabricated device results.
> Preregistration and results never share a commit. Download ceiling <10 GB;
> this cycle: 0 bytes.
>
> STANDING OPERATOR POLICY: prefer batched preregistered experiment suites
> over repeated round-trips. Tiny experiments are automated internal
> gates/preflight only, never repeated operator-facing work.
> Citadel validates Cymek; it does not replace Cymek.

## STATUS

CURRENT_CITADEL_SHA: see `git log origin/citadel -1` (certificate-regen
handover commit; executable code_sha unchanged by this docs-only commit)
CERTIFIED_CODE_SHA: `a7230712afc9b6fca9b25ad20912524ffc28c729`
AUDITED_CYMEK_SHA: `28bf57a0d299a2c13a99fe0046616c00a1b8530c`
RUNTIME_PIN_SHA: `28bf57a0d299a2c13a99fe0046616c00a1b8530c` (== live HEAD)

DEVELOPMENT_CERTIFICATE = PASS (7/7 files, regenerated at the exact
  executable identity; torch environment used so the four torch-optional
  tests ran rather than skipped)
LOCAL_TESTS = 91 PASS / 5 torch-optional skips across 7 files at the system
  interpreter; 91/91 under the torch environment (all skips verified PASS)
  (t1d 38; t1c 10; t1_canary 6; notebooks 2; bootstrap 6; cymek_checkpoint 7;
   one_shot 20 — includes the reserved PRE50M final-update regression, the
   completed-state negative control, PRE50M phase-status propagation,
   PRE50M-only bundle verification, the notebook torture set, and the T1E
   EOS-helper contracts)
NOTEBOOK_TORTURE = PASS
CYMEK_ALIGNMENT = PASS (pin == live origin/cymek HEAD)

T1D = EXECUTED / ARCHIVED — DO NOT RERUN
  (all arms SCIENTIFIC_FAIL, cross-arm INCONCLUSIVE; RESULTS.md/RESULTS.json)
PRE50M = GREEN (ready_for_50m_training: true, zero blockers — certified in the operator's 2026-09-06 T1D rerun bundle, sha256 c3f6643bf8aa88ff)
  (smoke budget fixed: reserved final update funds the resume proof;
  regression against the real Cymek TrainingState contract incl. the
  'a completed run cannot advance' negative control)
T1E = PLAN ONLY / NOT EXECUTED (docs/citadel/experiments/T1E/PLAN.md,
  PENDING OPERATOR; EOS-supervised helpers unit-tested)
50M CYMEK TRAINING = GATE OPEN (PRE50M green received)
  (NEXT_50M_DECISION.ready_for_50m_training == true required first)

## DOWNLOADS

```text
ITEM | SOURCE/PURPOSE | BYTES | CUMULATIVE BYTES
(none this cycle)
TOTAL_DOWNLOADED_GB = 0.0
```

## QUESTIONS_FOR_OPERATOR

```text
NONE
```

## BIGGEST BLOCKER

NONE LOCAL. The only remaining step before the 50M milestone is the
PRE50M-only TPU run.

## NEXT ACTION

The PRE50M gate is GREEN (the operator's T1D rerun bundle carried
NEXT_50M_DECISION.ready_for_50m_training = true with zero blockers, plus a
full six-arm replication of the null). Control hands to Cymek for the real
50,000,000-token milestone under its own production contracts. Citadel's
standing next work: T1E awaits operator approval; cymek's 3 unpushed local
commits get audited when pushed.

## CYMEK_REQUIRED_CHANGE

```text
NONE FOUND this cycle
```

## T0 / T1 / T1B / T1C (history)## T0 / T1 / T1B / T1C (history)

```text
T0: PASS (unchanged, still applicable)
T1: FAIL (loss-learned, exact-flat; historical result unchanged)
T1B: SUPERSEDED_BY_T1C (preserved, unexecuted)
T1C: EXECUTED — 4 arms FAIL, cross-arm INCONCLUSIVE (mode collapse, no
memorization; objective/data/2.3x scale moved nothing at 4M)
```

## T1D + PRE50M (preregistered, pending execution)

Arms A/B/C/D/E on the tiered ladder (~130 MB); PRE50M smoke on SCALE2 7.4M
through production transactions; NEXT_50M_DECISION machine gate.
"50M checkpoint" = 50M training TOKENS (Cymek training_spec, re-verified);
no 50M-param spec exists. Notebook: notebooks/citadel_colab_t1d.ipynb.

## DOWNLOADS

```text
ITEM | SOURCE/PURPOSE | BYTES | CUMULATIVE BYTES
(none — no pip installs, no datasets, no checkpoints, no artifacts)
TOTAL_DOWNLOADED_GB = 0.0
```

## QUESTIONS_FOR_OPERATOR

```text
NONE
```

## BIGGEST BLOCKER

NONE REMAINING THAT LOCAL VALIDATION CAN ADDRESS — next is the operator run.

## NEXT ACTION

Run `notebooks/citadel_colab_t1d.ipynb` once from Cell 0 through F in a
fresh/restarted Colab TPU session and return `CITADEL_T1D_RESULTS.zip`.

## CROSS-BRANCH STATE (2026-09-06)

BRAMASTRA (0235001, from triquetra): independently confirmed the
termination-supervision flaw (third discovery after T1C-confound and T1D's
15,000/15,000 MAX_TOKENS run); audited Cymek's data readiness and found
four gaps (P35-A prereg DRAFT with PENDING_MATERIALIZATION hashes;
inventory overcounts; qualification omission possible; declared != runnable
data) - recorded as post-PRE50M CYMEK_OBSERVATIONS, PRE50M semantics frozen.
Arkenstone (2a5ab55, from cymek pin; 95 files): executed real TPU training -
ARK-004A cognition transition universal on 4 fresh seeds (precursor claim
later INVERTED by its own reanalysis - precursor humility lesson); ARK-005
retention NULL (no arm meets consolidation bar; LR-decay weakly delays
collapse) - consolidation must be designed, not assumed. Provenance
mechanics (continuation probes, source snapshots) noted for adoption.
Cymek LOCAL movement: local cymek branch is 3 commits ahead of
origin/cymek (28bf57a), UNPUSHED (causal-eval/RNG/sampler/data-identity
repairs + CoreSubjectManifest/registry/protocol engine). Citadel pin stays
at 28bf57a until pushed; on push, audit the delta against T1D/PRE50M-
critical surfaces and re-run checkpoint/resume contract tests before
repinning. Full record: CROSS_BRANCH_INGESTION.md.

## CYMEK_REQUIRED_CHANGE

```text
NONE FOUND this cycle (checkpoint, data-interface, and contract surfaces
checked against current Cymek; Citadel adapts at its own layer only)
```

