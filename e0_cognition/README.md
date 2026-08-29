# E0 cognition research package

This package is the first executable ESOES component. It is deliberately limited to benchmark and generator research; it is not a model, trainer, tokenizer, corpus pipeline, or production V5 dependency.

## What is certified now

- deterministic development-suite regeneration from explicit version and seed;
- typed hidden causal truth and model-facing views that exclude answers/candidates;
- mechanical query, relevant-fact, irrelevant-fact, order, and state counterfactual contracts;
- separate development/sealed/fresh namespaces for symbols, relations, domains, and templates;
- distinct training-generator templates and internally recorded graph, counterfactual, difficulty, seed, provenance, and split fields;
- separate selection, query-lift, and realization measurements;
- random, candidate-order, lexical, positional, bag-of-words, broken-state, direct-retrieval, and full-oracle controls;
- independent surface-text reference solvers across every generated family;
- exact uniform-candidate chance, Wilson intervals, candidate/context-position and answer-format axes, approximate power planning, exact paired sign tests, and 10,000-resample paired bootstrap protocol;
- deterministic development receipt and E0 regression/property tests, including a 20-seed generator sweep and pooled eight-seed shortcut red-team.

Run:

```text
python -m unittest discover -s tests -v
python -m e0_cognition.certify --output artifacts/e0/development_certificate.json
python -m e0_cognition.device_benchmark --output artifacts/e0/local_device_probe.json
```

`device_benchmark` measures the E0 harness on CPU and, when PyTorch with CUDA
is installed, runs a small matched CPU/CUDA matrix smoke test. It is bounded
for laptop hardware and is not a model-quality or training-throughput claim.

## What is not certified

The current `PASS` is an infrastructure-development certificate, not proof that a model reasons and not the full E0 promotion exit. State cases are semantic-time queries with shuffled serialization, and rule cases hold out latent operand structures across splits. Easy retrieval families are intentionally solvable by lexical retrieval; causal query pairs, matched multi-hop controls, and natural transfer are what must distinguish manipulation. Raw-Core, constrained, assisted, and intervention-dependence outcomes are separate contracts. Before E1 model training, add source-disjoint natural fixtures and create an externally held T2 seed/fixture whose content never enters Git.

The hard-coded SEALED seeds used by tests are namespace sentinels only. They are not promotion fixtures and carry no scientific result.
