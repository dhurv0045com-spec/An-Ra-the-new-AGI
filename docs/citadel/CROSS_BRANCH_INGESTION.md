# CROSS-BRANCH INGESTION — BRAMASTRA + Arkenstone (2026-09-06)

Citadel's standing rule: other branches are audit inputs, never merged. This
records what the sibling agent branches demonstrated, what Citadel ingests,
and what stays branch-local. Read-only inspection of `origin/BRAMASTRA`
(0235001) and `origin/Arkenstone` (2a5ab55); the local `cymek` branch has
also moved (28bf57a → 4abeaeb, 3 commits, **unpushed** — origin/cymek still
shows the Citadel pin 28bf57a).

## BRAMASTRA (descends from triquetra)

**Independent convergence on the termination contract.** BRAMASTRA's
experiments and their Citadel roadmap (CITADEL_100X_ROADMAP.md, auditing
Citadel at 28ff690) found the same flaw Citadel's T1D run then proved
empirically: the encoder never appends a termination token, the eligible
mask covers answer characters only, and generation therefore ends MAX_TOKENS
unless a stop character happens to appear. Their prescriptions match T1E's
implemented fix (EOS appended before packing, same segment, EOS target
eligible under the shifted loss) and add two requirements T1E now adopts:

1. **Loss-averaging declaration**: answer/EOS loss must declare
   target-averaged vs separately weighted, identical across arms.
2. **Edge verification set**: single-char and multi-char answers, adjacent
   packed records, exactly-full sequences, padding tails — verified before
   any T1E run.

**Cymek data-readiness audit** (CYMEK_100X_ROADMAP.md, auditing the same pin
Citadel uses) found four issues Citadel's PRE50M data-interface cert does
not currently catch:

- the P35-A preregistration is `DRAFT` with `PENDING_MATERIALIZATION`
  cognition-generator hashes — declared data is not yet materialized data;
- `build_data_status` sums receipt-local unique counts without
  cross-receipt union/deduplication (inventory overcount);
- an empty qualification map produces no unqualified-family blocker
  (missing evidence can be omitted);
- the older gap-audit ("disconnected components") is historical — Citadel's
  runtime/checkpoint integration should be assessed for reuse, not assumed
  absent (BRAMASTRA correctly credits this).

**Citadel response (this cycle):** PRE50M semantics are frozen, so the
four issues are recorded as **known gaps for the post-PRE50M milestone**
(see agent.md CYMEK_OBSERVATIONS) — they gate the real 50M run's data
preparedness, not the smoke. T1E PLAN adopts the loss-averaging declaration
and the edge-verification set.

## Arkenstone (descends from the Cymek pin 28bf57a; 95 files, ~49k lines)

Real training executed on TPU with production-adjacent machinery. Findings
Citadel ingests:

- **ARK-004A**: a cognition transition mapped on 4 fresh seeds — universal,
  with a tens-selectivity precursor (LOO 4/4) and post-G90 instability; a
  follow-up reanalysis INVERTED an earlier precursor claim (VERDICT B+C) and
  cancelled ARK-004B. Lesson for Citadel: precursor claims need inversion
  analysis before they become gates.
- **ARK-005 retention**: 4 arms × 2 seeds × 24k steps; **no arm meets the
  consolidation bar**; LR-decay weakly delays capability collapse; wd/EMA
  null. Directly relevant to the operator's "child that keeps learning"
  goal: consolidation must be designed and measured, not assumed. T1E and
  the future 50M run should retain post-training retention probes.
- **Provenance mechanics** worth adopting: continuation probes, source
  snapshots, nomination rules; ledger verifier keyed on content-hash drift
  (Citadel already hashes receipts; the drift-detection idea is noted).
- **The same termination-supervision flaw** was found via BRAMASTRA
  ingestion ("Citadel T1C stopping confound") — third independent
  confirmation.

## Cymek local movement (UNPUSHED)

The local `cymek` branch is at 4abeaeb — 3 commits ahead of origin/cymek
(28bf57a): causal-evaluation repair, exact-resume RNG, canonical sampler
path, data identity, optimizer contract, P35 identity repair;
CoreSubjectManifest + checkpoint registry + evaluation protocol engine +
preregistered experiment contracts; Triquetra handshake artifact binding.

**Citadel pin action:** none yet — origin/cymek (the pushed authority) is
unchanged at the pin. When those commits are pushed, Citadel must audit the
delta against the T1D/PRE50M-critical surfaces (model/objective/optimizer/
checkpoint/packing/cursor/tokenizer-identity) before repinning; the
"exact resume RNG" and "optimizer contract" repairs touch surfaces our
resume-identity tests cover, so repinning requires re-running the
checkpoint-contract and resume tests against the new pin.

## Adopted now (this cycle)

1. T1E PLAN: loss-averaging declaration + edge-verification set added
   (BRAMASTRA prescription).
2. agent.md: BRAMASTRA's four Cymek data-readiness gaps recorded as
   CYMEK_OBSERVATIONS for the post-PRE50M milestone.
3. Cross-branch state recorded here; cymek movement flagged for audit on
   push.


## ADDENDUM — Cymek topology: two DISCONNECTED histories (verified)

`git merge-base 28bf57a 4abeaeb` exits 1: **no common ancestor**. The pin
(28bf57a) is a re-rooted lineage (root: "835c868 first upload") whose tip
commit's exclusive delta is one receipt refresh; the local `cymek` branch
(4abeaeb) descends from 26a61f6 — a lineage Citadel knows — plus three
commits (95dbc97, 674c1ca, 4abeaeb).

File-level divergence at the tips:
- production modules ONLY at the pin: `v5_training/mutation.py` (real-
  mutation certification), `v5_training/provenance.py`, 
  `v5_tokenizer/artifact.py`, plus pack/cursor/mixture/checkpoint_adapter
  present on both lines but only the pin's are production-audited;
- the local lineage LACKS mutation/provenance/artifact entirely.

Consequence: this is NOT a fast-forward repin — it is a choice between two
disconnected implementations. Citadel therefore added **lineage guards** to
`runtime_bootstrap.REQUIRED_RELATIVE_PATHS` (mutation.py, provenance.py,
artifact.py, pack.py, cursor.py, mixture.py, checkpoint_adapter.py): a
runtime resolved from the wrong lineage fails `verify_files` loudly at
BOOTSTRAP instead of silently training against an unaudited variant
(regression: bootstrap case G).

Reconciliation belongs to the Cymek side: the two histories must be
explicitly merged or one must be declared canonical before Citadel
considers repinning.
