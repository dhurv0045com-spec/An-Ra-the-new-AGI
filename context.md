# Hitchhiker context

Updated: 2026-11-14. Branch: `hitchhiker`.

## Purpose and requested workflow

The user proposed mechanisms A–H for compact cognitive generalization and asked for a technical, evidence-producing experiment design before implementation. The first deliverable is complete: [conclusion and experiment design](research/hitchhiker/CONCLUSION_AND_EXPERIMENT_DESIGN.md). The [original supplied proposal](research/hitchhiker/USER_ARCHITECTURAL_PROPOSAL.md) is preserved byte-for-byte, with its hypotheses and reported numbers unchanged.

The documentation deliverable established the reviewed design ([conclusion and experiment design](research/hitchhiker/CONCLUSION_AND_EXPERIMENT_DESIGN.md)) alongside the [original supplied proposal](research/hitchhiker/USER_ARCHITECTURAL_PROPOSAL.md). The Hitchhiker evidence-producing experiment harness, 7-factor counterfactual suite, cognitive health probes, Stage 0 diagnostic decoders, deterministic Stage 1–2 training arms, and bounded CPU mechanical verification are now implemented locally; frozen S20-low-lr checkpoint restoration and GPU Stage 1–3 training runs remain pending.

## Source identities

| Item | Identity |
|---|---|
| Branch base, from `Elendril` | `2affffb806b7f73b00bc21aa90538019f37ce904` |
| Audited V7.17 checkout | `0c9cb284d2c8c949e17ac342959c6f2bf351bd30` |
| Recorded optimization run source | `724af4cb508173e0eb3f4397d5dcb4a42a66712f` |
| Recorded optimization run | `f443d3f4141b4163bbc130d46ebd7395` |
| Local Hitchhiker worktree | `C:/Users/ankit/Downloads/An-Ra-hitchhiker` |
| Local audited V7 checkout | `C:/Users/ankit/Downloads/An-Ra-Elendril-V7` |

The branch base is an ancestor of the audited V7 source (`0c9cb284d2c8c949e17ac342959c6f2bf351bd30`), which is integrated in the working tree (`MERGE_HEAD`) and bound in `anra_v5/elendril/hitchhiker_protocol.py`. Existing V6 documents retain their historical context.

The audited [V7.17 result report](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/0c9cb284d2c8c949e17ac342959c6f2bf351bd30/Experiment%20Result/V7.17/f443d3f4141b4163bbc130d46ebd7395/RESULTS.md) records earlier execution and replay. This document does not claim those experiments were run again for Hitchhiker.

## Findings that govern implementation

1. Lower LR improved mean confirmation from 62.80% to 84.52%, but novel one-hop and three-hop remained 0/48 pooled seed/case observations. These are three seeds on shared worlds, not 48 independent worlds.
2. UTF8 is a byte tokenizer. Novel aliases do not allocate new entity embedding rows. Formatting, sequence length, positions, and task wording are competing explanations.
3. Training randomizes entity associations across worlds, then repeats a fixed corpus. Freshness is testable, but static global entity-to-answer lookup is not an established generator defect.
4. Chain diagnostics jointly change wording, timestamps, distractors, and row order. Matched counterfactuals are needed to isolate causes.
5. Full grammar decoding already failed to rescue lower-LR three-hop results. An availability-only mask is a separately attributed diagnostic, not a demonstrated solution.
6. The six unrun representation jobs are baseline versus row reversal at 3e-4, not slot or binding training. Amendments need new protocol identities.
7. Registered model scales are S3, S8, S20. A new S9 requires an explicit architecture and parameter receipt.
8. External canonicalization, action masks, and algorithmic verification must remain distinguishable from learned representation formation.

## Next execution sequence

1. Integrate the reviewed implementation and establish source/protocol provenance.
2. Locate and verify original frozen checkpoints. The committed report says weights were absent from the supplied analysis ZIP; availability of the external FULL archive remains unestablished here.
3. Run matched frozen-checkpoint diagnostics with raw, availability-only, and existing grammar decoders. Log complete first-byte distributions and complete outcomes.
4. Register a five-arm S3 screen across seeds 7, 19, 41: static baseline, fresh worlds, entity relabeling, fresh-plus-relabeling, and targeted availability/task-coverage supervision. Total: 15 runs.
5. Add justified interaction and slot-control experiments; promote the selected recipe to matched S8/S20 comparisons. Keep recurrent reasoning and associative-memory changes as subsequent attributable experiments.
6. Evaluate fresh confirmation worlds, causal answer changes, renaming consistency, distractors, revisions, missing bridges, and independently acquired retention/transfer. Publish failures and inconclusive outcomes as well as successes.

The full design specifies proposed thresholds, compute matching, uncertainty, falsifiers, and an implementation prompt. Freeze final counts and criteria before new runs. Passing engineering tests does not establish cognition.

## Current handoff state

- Complete: repository audit, critique of A–H, proposed experimental protocol, original proposal preservation, integration of `0c9cb284d2c8c949e17ac342959c6f2bf351bd30` (`MERGE_HEAD`), and Hitchhiker implementation across `anra_v5/elendril/hitchhiker_{protocol,data,diagnostics,training,evaluation}.py`, `tools/elendril_hitchhiker_runner.py`, and `tests/test_elendril_hitchhiker.py`.
- Verified locally: `tests/test_elendril_hitchhiker.py` (`9 passed`), `python -m tools.elendril_hitchhiker_runner --mode cpu-mechanical-check` (receipt at `artifacts/elendril/hitchhiker/mechanical_verification.json`, `claim_status="IMPLEMENTED"`, `formation_gate="MECHANICAL_ONLY_NOT_FORMATION_QUALIFIED"`), and `python -m tools.verify_elendril_evidence` (`410 passed, 1 skipped` in `pytest -q`, with 3 historical notebook `HEAD`-ancestry checks awaiting commit of `MERGE_HEAD`).
- Pending: mounting external S20-low-lr checkpoints for live Stage 0 audit (`CHECKPOINT_AUDIT_PENDING`), GPU execution of Stage 1–3 training/confirmation runs, and all capability claims (second-wave mechanisms E–H remain deferred).
- No V8/V8.1 specification was located in the searched research/docs trees. V8 compatibility in the design is a proposed interface contract, not a verified match to an unseen specification.
