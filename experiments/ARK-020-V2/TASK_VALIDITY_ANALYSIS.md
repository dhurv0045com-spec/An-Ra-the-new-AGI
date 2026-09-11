# TASK VALIDITY ANALYSIS — Skill C (two-hop relational composition)

*Mission §5 requirement: written before V2 implementation freezes; answers what
information is available, what rule must be inferred, why sealed answers are derivable,
what shortcuts exist, how they are blocked, what chance is, and what counts as
generalization.*

## The V1 failure (root cause, not patched)

V1's skill C asked for the successor of a token under an arbitrary fixed permutation.
The prompt (`Chain: k next`) contained **no mapping information** — for sealed keys that
were never trained, the (key, successor) pair existed nowhere in any training example or
in the prompt itself. The task was information-theoretically impossible: no amount of
learning could produce above-chance sealed accuracy. Root cause: the task conflated
"rule induction" with "arbitrary unseen lookup". V1 is preserved as evidence; C is
replaced, not patched.

## V2 skill C: two-hop relational composition

**Construction (per factset):** 6 X tokens, 3 M tokens, 3 Y tokens (disjoint, in C's
token group). A factset is 3 chains x→m→y: choose 3 of 6 X tokens, assign the 3 M tokens
bijectively, assign the 3 Y tokens bijectively. C(6,3) × 3! × 3! = 720 possible chain
sets; 600 are drawn deterministically and split 400 train / 50 parent-control /
50 main-control / 50 validation / 50 sealed (seed 524220).

**Prompt (both maps present in context; answer requires chaining):**

```
Links: x1 gives m1; x2 gives m2; x3 gives m3; m1 makes y1; m2 makes y2; m3 makes y3; Trace: x2 to
```
answer = y2 — reachable only as x2 → m2 → y2. The intermediate m2 is never the answer of
a Trace query, and no (x, y) pair is ever directly queried without the m-hop being
available in context.

## The seven required answers

1. **What information is available?** The full chain structure in context: 3 x→m pairs
   and 3 m→y pairs, in some order. During training, Trace queries are asked on TRAIN
   factsets (so the model also sees correct Trace behavior on 400 factsets).
2. **What rule must be inferred?** Trace = resolve the queried x to its m via the
   "gives" segment, then resolve that m to its y via the "makes" segment — a two-hop
   lookup through an intermediate, robust to the order in which the six facts appear.
3. **Why can the SEALED answer be derived?** Sealed factsets are disjoint from
   train/control/validation factsets, but every sealed prompt contains both required
   mappings in its own context. The answer is logically determined by the prompt + the
   chaining rule; no external memory of the specific factset is needed.
4. **Shortcuts possible?**
   - *Direct (x,y) memorization*: impossible for sealed factsets (disjoint); a model
     that memorized train (x,y) compositions cannot transfer except via the rule.
   - *Token-frequency*: each y token appears as the composition answer equally often
     across the drawn split (balanced by bijective assignment; asserted by test).
   - *Template shortcut (answer = nearest semantic type)*: candidate answers are the 3
     y tokens; m and x tokens are answer-eligible distractors, so the model must
     distinguish token roles — that is part of the capability, not a leak.
   - *1-hop bypass*: a model that answers with the queried x's m fails (m is never a y);
     the m→y hop is mandatory. A model that ignores x and outputs the most recent y
     fails on 2 of 3 orderings (robustness modes catch order-sensitive degeneracy).
5. **How are shortcuts blocked?** Disjoint splits (memorization), bijective balance
   (frequency), three-mode evaluation (order degeneracy), distinct template (no
   confusion with A/B machinery), and the never-an-intermediate property of Trace
   answers (1-hop bypass). All asserted in `test_skill_c_validity`.
6. **Chance performance?** 1/3 (three possible y tokens; uniformly balanced). The 0.85
   third-mode and 0.90 canonical thresholds are far above chance.
7. **What counts as real generalization?** ≥ thresholds on held-out factsets with
   permuted orderings, given that the same model was trained on disjoint factsets —
   i.e., the chaining *operation*, not the chains.

## Learnability rationale

Two-hop in-context lookup is the classic multi-hop binding extension of the A/B family;
the ARK-014/015/017/V4 lineage shows single-hop lists are learned within ~1–2k updates
at this scale, and the chained variant adds one indirection with both maps visible.
Residual risk (formation too slow at 12 slots/1500 updates) is handled the honest way:
the plastic-reference formation gate returns `INCONCLUSIVE_FORMATION_INSTABILITY_PHASE_C`
instead of misattributing to protection — and V4's evidence shows dose selection matters,
so 12 slots is chosen mid-range of V4's working 8–16 window.

## Role of token disjointness (mission §9)

Skill vocabularies stay disjoint so that interference measured at phase k+1 is
attributable to *learning a new capability*, not to *reusing token ids* (which would
confound retention failures with output-space collisions — exactly the CYR-GPU-011/012
class of representation effects). Disjointness is causal hygiene, not difficulty
theater; it is cheap here because 48 eligible single-token words exist.

## D (inverse retrieval) validity, briefly

Same list machinery as A/B with (value, key) pairs and a "Who holds v ?" query: the
answer is derivable from context by reversing the lookup direction; chance = 1/6
(6 candidate key tokens); shortcuts blocked by the same disjointness/balance/three-mode
mechanisms. It is retained because reversed-direction lookup is a genuinely different
computation from forward binding while reusing proven machinery.
