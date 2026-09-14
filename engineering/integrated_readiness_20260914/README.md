# Consolidated integration milestone: a learnable, connected cognition system

This is the current chief response. Complete the existing cognition repairs as part of this single connected milestone; do not start another disconnected feature branch. Read [REVIEW.md](REVIEW.md), [CONTRACTS.md](CONTRACTS.md), [EXECUTION.md](EXECUTION.md), and [ACCEPTANCE.md](ACCEPTANCE.md). The earlier [F1–F6 design](../cognition_foundation_20260914/README.md) and [O01–O10 operational obligations](../next_phase_20260914/README.md) remain required where this packet does not explicitly refine them.

## Current decision

**Not ready for the owner experiment.** At the start of this review both HEAD and origin/BRAMASTRA were `5020533`. The external agent had uncommitted changes in `cognition/episode.py`, `campaigns/phases/e2.py`, and a new `tests/test_research_k8_foundation.py`. This review includes that work in progress, identified by exact file hashes in the diagnostic receipts. It is not a claim that those changes were pushed, finished, or accepted.

Twenty-six focused foundation tests passed. The full new suite stalled in the planner episode test and was interrupted; the two episode/integration cases were excluded from the subsequent bounded test run. Seven cross-component diagnostic criteria failed. The second diagnostic receipt establishes that all inspected source hashes stayed stable during the run. No learned model or optimizer was used in those diagnostics; an oracle is used only to construct review counterexamples.

The main failure is no longer merely missing classes. The data, objectives and decision consumers disagree about what the model is learning. Some tasks withhold information necessary to determine the answer. The proposed runtime can also fail to terminate. Fixing only the isolated renderer tests will not close this milestone.

## What the chief has delivered

- A reproducible, bounded [audit program](audit.py), [first receipt](BASELINE_01.json), [expanded receipt](BASELINE_02.json), and [focused test receipt](TESTS_01.json).
- A consolidated source review covering cognition, supervision, action/prediction interfaces, controls, accelerator preparation and RSI.
- Concrete algorithm and data decisions, ten work packages, ownership boundaries, acceptance evidence and a copyable implementation prompt.

The chief did not edit or commit the agent's three implementation files. This packet owns only its new directory and the dispatch notices in AGENTS.md, engineering/README.md, engineering/STATUS.md, experiment.md, and the K8 agent prompt. Two bounded Luna source audits contributed findings; the chief checked the cited critical paths before incorporating them.

## What to build next

Build one learnable vertical slice first: a real public task state, available information, an observed transition, correctly conditioned targets, a trained scoring/decoding interface, a legal action, an independent verdict, and a correctly charged receipt. Extend that same slice to all registered families and treatments. Then make measured proposer training and independent successor evaluation use those exact contracts.

The evidence-repair controller remains a creative future hypothesis. The useful new idea for this milestone is an **information sufficiency witness**: before admitting a task as solvable, demonstrate which allowed observations can distinguish the answers. The witness is a dataset/validation artifact, never an oracle handed to the learner. This prevents an expensive run from testing an impossible information problem and then misdiagnosing it as model failure.

Keep the owner's one future two-T4 campaign at 480 elapsed minutes, stop training at minute 450, preserve export time and all retry charges. Local optimizer updates remain unallocated. No new accelerator run is launched by this dispatch. Code-ready, GPU-qualified and experimentally supported are separate decisions; this packet alone establishes none of the last two. AGI remains the research objective, not the name of a passed test.
