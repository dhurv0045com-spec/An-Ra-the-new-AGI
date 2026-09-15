# FORMATION-MUX-001 — returned-bundle audit and corrected interpretation

Date: 2026-09-15. Bundle: `FORMATION_MUX_001_RESULTS.zip`
SHA-256: `sha256:859489d9babc532a4ed1e785af4e4993541b93730f7166dff2946a26cc996bc5` (operator-returned, Kaggle T4 x2).

## Operational audit — PASS

- 24/24 official arms COMPLETE on two Tesla T4s; no engineering failures.
- Both sealed sets consumed legally (markers COMPLETE, once, after dev freeze).
- Environment receipt: dual Tesla T4; engineering_only flag absent (official run).
- Receipts internally consistent (per-arm manifests, progress cadence, checkpoint
  receipts, sealed commitment).

## Corrected verdict interpretation — the preregistered NULL is NOT licensed

The preregistered verdicts print NULL for both experiments because every
preregistered paired delta is below threshold. But the deltas sit on a ZERO
BASELINE: identity formation AUC is 0.000-0.008 in ALL 24 arms of BOTH
experiments. The verdict mapping presupposed nonzero baseline formation in at
least the reference arms; with the capability absent everywhere, the contrasts
had no dynamic range. The mechanically correct label for the contrasts is
**INCONCLUSIVE_AT_ZERO_BASELINE** — no conclusion about extra-row weight
decay, trainability, or denominator participation (in either direction) is
licensed by this campaign.

## What the campaign DID measure — a strong structured finding

The instrument resolves learning cleanly on the same grammar, same training,
same arms:

- **termination (1-token counting answer): 1.00 exact** by update ~300,
  sustained to 2000 — fully formed.
- **composition (two-hop, 1-token answer): ~0.29** at endpoint, still climbing.
- **missing_info (abstention): ~0.14 flat** — the abstain path is learned
  (eos_rate is 1.0 everywhere: valid stopping is universal).
- **identity/copy (multi-token exact-sequence answer): 0.0000 flat for all
  2000 updates, every arm, both renderings.**
- binding and state_order (multi-token answers): ~0.0-0.05.

Two exonerations fall out at this baseline: the tied-row treatments are
indistinguishable (M0-M3 identical zeros), and rendering is exonerated —
R1 (symbol-isomorphic, NO composite segmentation) also sits at exactly 0.0,
so segmentation is not the identity blocker.

Prime suspect: **exact-match metric resolution for multi-token answers.** The
identity family's all-or-nothing sequence-exact score gives near-zero
evaluated signal until the policy is nearly perfect, while 1-token families
(termination) resolve immediately. The model may hold substantial partial
capability invisible to the primary metric. This is an instrument-resolution
question, and it is testable: token-level accuracy / longest-common-prefix /
per-position teacher-forced accuracy on the same checkpoints would reveal
formation long before exact-match flips.

## Disposition (supersedes the recorded next_action)

1. Do NOT conclude the tied-row mechanism line is null — it is untested at
   this baseline. Do NOT start another mechanism sweep either.
2. Next single action: a metric-resolution audit on the PRESERVED final
   checkpoints (token-level diagnostics; no new training required) to decide
   whether identity/copy is (a) forming but unmeasured, or (b) genuinely
   absent — this changes which roadmap item follows.
3. The mechanism contrasts (CS-MECH-002) may then be re-adjudicated on a
   resolved instrument; the frozen science commit remains the protocol
   authority for that decision.
