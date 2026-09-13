# Engineering decisions

## E009 — Complete the learning connections before claiming an integrated learner

Adopted for implementation design, 2026-09-13; efficacy unqualified. Source audit at `baa655d` found that action/value heads lack training objectives, the planner does not recursively value typed public outcomes, memory is absent from canonical input, and collection stops at the ledger. The [master program](master_program_20260913/README.md) assigns M00–M18 to connect those behaviors through the existing shared decoder and trainer, preserving an answer-only control.

Candidate isolation, separate objective denominators, a finite-support initial world planner, explicit ResourceVector comparisons and fixture-versus-learned publication guards are required design decisions. The central research hypothesis is counterfactual inquiry followed by consolidation; no improvement magnitude is asserted. The owner requested the whole build, so after M00 dependency checks/report the executor may continue provisional implementation instead of stopping at the former B2.2 return boundary. Chief acceptance, from-scratch weights and cumulative resource limits remain in force.

## E001 — Chief designs; agents implement

Date: 2026-09-07. Status: adopted from owner direction. The chief owns architecture, algorithms, data structures, experiments and acceptance. Coding and experimental execution are delegated. Existing in-flight prototype work is completed as a bounded delegated task.

## E002 — BRAMASTRA owns its architecture

Status: adopted. Other branches are historical evidence, not runtime dependencies or default design authorities. New integrated modules use `bramastra_lab/research/`; the existing `discovery` code remains a named prototype until reviewed components are adopted through explicit adapters.

## E003 — Test learned investigation before adding scale by default

Status: adopted as research priority. Core uncertainty is whether the learner can select useful experience and retain its gains. Larger models remain candidates when learning curves indicate capacity limitations. Code size and parameter count are not progress measures by themselves.

## E004 — One-step information gain is a bootstrap control

Status: adopted. An exact parity construction shows individually uninformative observations can jointly determine a target. The current teacher is therefore insufficient as the sole investigation objective. W04 compares depth-two teaching and outcome-trained policies; no neural superiority is assumed.

## E005 — Independent evidence before promotion

Status: adopted. A candidate cannot certify itself through training loss or its own report. Fresh paired transfer and retention results, cumulative forgetting and full cost determine a bounded promotion decision. Test pool feedback that guides development is retired from future independent claims.

## E006 — Work packets sized by outcomes, not padded time

Status: adopted. Packets target 3–5 hours of substantial engineering with clear ownership and deliverables. Actual duration varies. Agents should finish criteria or report blockers, not generate unnecessary code or consume compute to satisfy a duration target.

## E007 — No premature AGI claim

Status: adopted. A completed engineering program may yield evidence about learning mechanisms. No current result establishes AGI, unlimited self-improvement or a guaranteed compute budget sufficient for either.

## E008 — Contract acceptance includes adversarial cases

Status: adopted after W01's first review. Dataclass names and frozen annotations do not enforce immutable, well-typed or nonleaking data. Acceptance includes nested record consistency, immutable payloads, collision-resistant canonicalization, normalized split identities and exact prototype field mapping. See [the W01 review log](reports/W01_REVIEW.md).
