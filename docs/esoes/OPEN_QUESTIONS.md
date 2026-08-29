# ESOES Open Questions

These are questions the project should answer experimentally before freezing the next major Core training path. They are intentionally open-ended. A good result may reject the current intuition.

## STEP 2 disposition

The original list below remains as the audit trail, but it must not trigger 104 independent experiments. `V5_MASTER_BLUEPRINT.md` collapses it into six decisive experimental questions:

| Priority | Decisive question | Current answer | Resolver |
|---:|---|---|---|
| 1 | Is the cognition benchmark causally valid and shortcut-free? | OPEN; no model result is meaningful before certification | E0 |
| 2 | Which tokenizer maximizes cognition per raw byte/FLOP? | 24,576 byte-fallback is provisional | E1: 16k/24k/32k |
| 3 | Does more depth/full attention/less KV compression improve OOD cognition at fixed budget? | 28×768, full attention, 4 KV, QK norm is provisional | E2 |
| 4 | Are structured contrasts sufficient under LM loss, or does query-swap contrast add transfer? | LM remains base; query-swap is the sole auxiliary candidate | E3 |
| 5 | What cognition share/order and optimizer regime preserves the substrate? | 15%, uniform mixing, WSD, 262k tokens/update are provisional | E3–E4 |
| 6 | Does the winning effect survive scale and fresh natural domains? | OPEN / UNKNOWN | E5 at ~102M |

### Questions answered enough to stop debating

- **Direct 300M–3B launch:** rejected. Evidence is insufficient; run the 195M path only after proxy and mid-scale gates.
- **Final checkpoint equals best:** rejected by V4 evidence.
- **Many novel modules in the first V5 baseline:** rejected because they destroy causal attribution.
- **Same-query margin objective:** rejected by SFT7 evidence.
- **Runtime intervention equals native cognition:** rejected; assisted and raw capability remain separate.
- **Connector replacement:** rejected. The Connector keeps tools, durable memory, long-horizon control, verification, risk, and credit assignment.

### Questions deliberately left open

- Whether query-swap contrast transfers from verified synthetic examples to natural reasoning.
- Whether 195M has enough capacity for robust three-hop composition.
- Whether byte-level architectures eventually beat subword V5-A at equal compute.
- Whether 300M provides more value than additional clean tokens after V5-A.

Do not reopen a resolved direction without new evidence. Do not close an experiment-gated value by implementing it first.

---

## A. What exactly should the Core know how to do?

1. Which cognitive primitives are truly foundational versus skills that can safely remain in the Connector?
2. Is robust query-conditioned binding the first bottleneck, or is the deeper bottleneck contextual representation itself?
3. How much composition should a ~100M, ~300M, and ~1B model realistically be expected to learn from foundation training?
4. Should “missing information” detection be treated as a foundation capability or post-training behavior?
5. Which capabilities need to be native because external repair is too expensive, brittle, or distribution-sensitive?
6. What evidence would convince us that a Core has internalized an intervention rather than merely memorized the training format?

---

## B. Architecture

7. At fixed parameters/FLOPs, does deeper/narrower actually improve multi-step composition over wider/shallower?
8. Does V4's 18-layer/896-width shape leave too little sequential computational depth?
9. Is 4k context enough for the first V5, or does longer context change the nature of the cognition experiments?
10. Does full attention materially outperform hybrid/sliding attention on binding and state tracking at equal compute?
11. How many KV heads are needed before GQA compression starts harming fine-grained contextual binding?
12. Does QK norm help or hurt sharp query-dependent candidate discrimination?
13. Is tied embedding/output still the right choice for a cognition-focused small model?
14. Would a modest recurrent/state mechanism improve state tracking enough to justify extra complexity?
15. Are explicit memory/state modules useful before the base decoder can solve controlled context tasks, or would they mask a weak Core?
16. At what scale, if any, does MoE become scientifically useful rather than just operationally complex?

---

## C. Tokenizer / representation

17. Does 16k/24k vocab improve compositional handling of novel identifiers compared with 32k?
18. At what point does smaller vocab create too much sequence-length inflation and reduce effective context?
19. Does tokenizer choice materially affect exact copying of nonce strings?
20. Should code/math symbols receive special treatment, or should a simpler general tokenizer be preferred?
21. Could byte-level or byte-fallback behavior improve robustness enough to justify cost?
22. Does the tokenizer accidentally create lexical shortcuts in synthetic cognitive tasks?

---

## D. Data mixture

23. What fraction of foundation tokens should be cognition-targeted before ordinary language/world modeling degrades?
24. Is 20–35% cognitive curriculum too high, too low, or only appropriate at small scale?
25. Should cognitive examples be uniformly mixed or scheduled in phases?
26. Does mixing synthetic cognition throughout training work better than concentrated curriculum bursts?
27. How much high-quality code/math/science data is needed for structural reasoning benefits?
28. Which natural-data domains contribute most to transfer onto binding/composition?
29. Can we measure the marginal cognitive value of each data family per token/FLOP?
30. How aggressively should repeated structures be deduplicated when repetition itself may help teach an algorithm?
31. How do we prevent synthetic tasks from dominating style/distribution while still shaping internal computation?

---

## E. Curriculum

32. Should curriculum difficulty advance based on token count or competence thresholds?
33. Does learning one-binding → four-binding → eight-binding produce genuine cardinality scaling?
34. Is composition learned better after strong binding/state tracking, or should composition appear from the beginning?
35. Does explicit gradual difficulty help or merely overfit a progression pattern?
36. How should curriculum mix hard failures versus examples near the model's current competence frontier?
37. Can the Connector automatically identify the next useful curriculum difficulty without leaking evaluation answers?

---

## F. Objectives

38. How much can standard autoregressive LM training learn if the data contains strong controlled contrasts?
39. Is the SFT6 query-conditioned improvement primarily due to the objective or the data structure?
40. Does an explicit contrastive/margin loss create better OOD binding than LM-only training?
41. Can an auxiliary objective improve internal selection while damaging free realization?
42. Should selection and realization receive separate objectives?
43. Can counterfactual query normalization be converted into a training objective that causes RAW scores to become intrinsically normalized?
44. What would count as evidence that the Core has internalized the normalization computation?

---

## G. Optimization

45. What learning-rate schedule best preserves cognitive capability during long continuation training?
46. Did the final V4 continuation's behavioral regression reflect LR schedule, data phase, random variance, or measurement noise?
47. Is WSD preferable to cosine for cognition-rich mixtures?
48. How sensitive are cognitive metrics to batch size / gradient noise scale?
49. Does weight decay influence exact memory/binding differently from broad LM loss?
50. Should some curriculum phases use lower LR to avoid erasing earlier cognitive circuits?
51. Would replay/interleaving prevent the retention-versus-specialization tradeoffs seen in earlier SFT?
52. What optimizer-state/provenance invariants should be checked every N updates rather than only at startup/checkpoint time?

---

## H. Scale

53. What is the minimum model size at which 2-hop composition becomes learnable on genuinely OOD forms?
54. Do cognition-rich data gains observed at 50M survive at 150M and 300M?
55. Is there a qualitative capability threshold with depth/parameter scale, or mostly smooth improvement?
56. Should V5 target ~300M, or would a better-trained 180M Core be more informative first?
57. At what point does scaling parameters beat spending the same compute on better curriculum/data?
58. When would a 1B or 3B run become justified by evidence rather than ambition?

---

## I. Evaluation

59. What is the smallest benchmark battery that reliably predicts later cognitive quality?
60. How many cases are needed before selecting an intermediate checkpoint over a final checkpoint?
61. How do we test copy/context representation without conflating it with instruction following?
62. How should candidate-scoring tasks be designed so that chance, candidate priors, and lexical artifacts are controlled?
63. How can we make composition tests impossible to solve by direct lexical retrieval?
64. How do we build OOD splits across entities, templates, topology, and difficulty simultaneously?
65. What metrics should define a cognition aggregate without hiding catastrophic weakness in one family?
66. Should promotion use a minimum/worst-family threshold rather than only an average?
67. How often should fresh fixtures be generated versus preserving sealed fixed benchmarks for longitudinal curves?
68. What behavioral tests must run automatically on every immutable checkpoint?

---

## J. Internal state / mechanistic diagnostics

69. Can query-conditioned candidate preference be localized to particular layers or attention pathways?
70. Does SFT6 create a new signal or amplify a weak signal already present in the parent?
71. At what depth does the correct entity/value binding become linearly or probabilistically separable?
72. Can layerwise logit-lens/probe measurements predict which training examples will improve binding?
73. Is the realization failure caused by later-layer drift, token prior competition, or decoding dynamics?
74. Does counterfactual normalization approximate a computation that could be learned inside attention/MLP circuits?
75. Can the model's own observable score geometry predict which cognitive mechanism is failing without structural task labels?

---

## K. Self-model / cognitive credit assignment

76. Can a policy predict repair choice when external task structure is held identical and only internal score state differs?
77. How much of the current routing success is a structural router versus genuine model-state diagnosis?
78. Can a structure-only baseline match the learned policy?
79. Which internal-state features add predictive value beyond candidate count, output arity, and format?
80. Can intervention outcome data be collected without evaluator leakage or outcome-conditioned sampling?
81. Can a clean one-variable VIE bank be built mechanically?
82. How should diagnostic confidence be calibrated rather than assigned after one successful flip?
83. Can a cost-sensitive policy choose the cheapest effective intervention without sacrificing too much success?
84. When should the system choose NO_CHANGE?
85. When is training actually justified instead of a runtime repair?

---

## L. Internalization from Connector to Core

86. How many verified intervention examples are required before converting a repair into training curriculum?
87. What is the correct way to turn RAW→NORMALIZED repairs into training examples without teaching fixture-specific shortcuts?
88. After internalization training, should success be measured by higher raw accuracy, lower intervention frequency, stronger margins, or all three?
89. How do we detect when training has merely learned to imitate the external repair's output rather than its computation?
90. Can several discovered repairs be internalized without catastrophic interference?
91. Should the Connector maintain interventions indefinitely as diagnostic probes even after the Core improves?

---

## M. Checkpoints and training infrastructure

92. What immutable checkpoint cadence provides enough temporal resolution to catch behavioral peaks without excessive storage?
93. Can milestone checkpoints be safely persisted from XLA with full optimizer state?
94. Should behavioral canaries run during TPU training or asynchronously after checkpoint upload?
95. How should cumulative token accounting work across packs so lifetime totals are impossible to misinterpret?
96. What evidence must a checkpoint contain before it can be called a valid parent?
97. How should “best checkpoint” be selected when LM loss and cognition disagree?
98. Should the final model be an ensemble of evidence (best substrate vs best cognition) only for research, or must one checkpoint always win?

---

## N. What could falsify the whole V5 direction?

99. What if cognition-targeted pretraining improves synthetic benchmarks but not natural reasoning?
100. What if standard generic pretraining catches up once model scale increases?
101. What if the apparent SFT advantage is mostly instruction-format specialization?
102. What if binding/composition are better implemented in the Connector than internalized in a small Core?
103. What if architecture changes matter far less than data quality?
104. What if the best foundation is simply a very strong conventional LM plus minimal external cognition?

These possibilities must remain admissible.

---

# Highest-priority questions to answer first

Before freezing V5 v1.0, answer these first:

1. **What exact cognition benchmark defines success?**
2. **Can cognition-rich LM data produce OOD binding/state/composition without special objectives?**
3. **Does depth improve cognition at fixed model/compute budget?**
4. **Which vocabulary size gives the best cognition/compute tradeoff?**
5. **What cognition-data fraction improves primitives without damaging general substrate?**
6. **Does SFT6's selection signal come primarily from data contrasts or its objective?**
7. **Can the same external task structure have different optimal interventions based only on model-state evidence?**
8. **What checkpoint promotion rule prevents another “lower loss, worse behavior” mistake?**
9. **What is the smallest scale at which the winning recipe survives?**
10. **What evidence is strong enough to justify the first expensive ~300M V5 run?**

---

# Question to return to

> **If the Connector were removed tomorrow, what computations would we wish the Core had already learned — and what exact training experiences would cause those computations to emerge?**
