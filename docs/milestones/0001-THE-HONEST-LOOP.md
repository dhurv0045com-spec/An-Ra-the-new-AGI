# Milestone 0001 — THE HONEST LOOP

**Date:** 2026-08-22 · **Branch:** `core-exp` · **Tag:** `milestone/0001-honest-loop`

Something big happened here.

An-Ra stopped being a repository of impressive module names and became one
small machine that measures itself and tells the truth about what it finds.
No capability was invented in this milestone. Something rarer was built:
an executable intelligence loop whose every success claim is verified, whose
every failure is diagnosed from measured evidence, and whose substrate
limitation is stated in numbers rather than adjectives.

## The errors that were reverse-engineered out of existence

1. A prototype whose intervention generator read the planted failure label —
   deleted before this branch; its lesson (structural `ObservedCase` / `HiddenGroundTruth`
   separation) is now enforced by tests.
2. Completers that could manufacture success labels — abolished; the
   verifier is the only authority on success.
3. Tokenizer identity (500 probe encodes + a SHA-256 over the full vocab)
   recomputed on *every* prefill/step — generation was validation-bound.
4. Errored intervention arms counted as measured non-flips — a Core fault
   could masquerade as "no intervention helped."
5. Outcome labels outside the diagnosis vocabulary; abstention silently
   undercounted.
6. Case IDs that encoded the hidden family — a greppable leakage bypass.
7. A suite docstring claiming measurements that never ran.
8. `generate()` accepting foreign tokenizers and burning a discarded final
   forward pass.
9. The step-20k artifact — whose tokenizer is byte-identical to canonical —
   rejected on contract *format*; legacy binding now checks substance.
10. A CUDA device-index trap (`torch.device("cuda") != cuda:0`) that broke
    every GPU `forward_step` after prefill; CPU tests could never see it.
11. An entire GPU evaluation built on stateless 1-token decoding — evidence
    that measured its own artifact, not the model. Replaced.

## The truths that were measured

- **In-context information use: 0/5 everywhere** (steps 5k / 20k / 30.4k,
  both protocols, nonce facts). The substrate cannot lift a fact from
  context into an answer.
- **Verbatim copying in natural language: 3/5 → 4/5 → 0/5.** Capability
  peaked at step 20,000; the continuation to 30,400 *destroyed* it
  (echolalia, rare-token salad, sampling-diversity collapse 1.00 → 0.57).
  Zero improvements were found past 20k.
- **Natural language strictly dominates the structured tag protocol.**
  Next-token entropy after a tag prompt: 4.9–7.0 (lost). After natural
  prompts: 0.01–0.47 (near-certain). The tag grammar is out-of-distribution.
- **Chat form was learned** (H:/ANRA: turn-taking, grammar, register);
  content was not (no persona, no knowledge, instruction echo instead of
  extraction). No tool calling in any convention at any step.
- **The full cognitive-credit battery on real V4: 20/20 honest abstentions.**
  The system reports what happened, never what would look good.

## What now exists

One reference runtime — `anra.run(task)` — binding task → attempt → Core →
verification → single-variable interventions → measured diagnosis → repair
→ verified learning candidates. A causal probe battery (P1–P6) with matched
controls. A strict loader, a validated executor (CPU and CUDA), 79 passing
tests, and evidence directories instead of promises.

## The verdict this milestone carries forward

The bottleneck is not the Connector, not the protocol, not the code.
It is the substrate: context-to-answer binding, best addressed by training
restarted from the step-20k weights under a decayed schedule — promoted
only through this battery, never around it.

> The target is not a repository containing AGI ideas.
> It is a machine whose behavior we can understand, intervene on, measure,
> and improve. As of this milestone, the measuring is real.

— recorded at the close of the coherence pass, core-exp.
