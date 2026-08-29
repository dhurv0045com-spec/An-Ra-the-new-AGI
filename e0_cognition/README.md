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
- exact uniform-candidate chance, Wilson intervals, answer-position balance, and approximate power planning;
- deterministic development receipt and 14 regression/property tests, including a 20-seed generator sweep.

Run:

```text
python -m unittest discover -s tests -v
python -m e0_cognition.certify --output artifacts/e0/development_certificate.json
```

## What is not certified

The current `PASS` is an infrastructure-development certificate, not proof that a model reasons and not the full E0 promotion exit. Easy retrieval families are intentionally solvable by lexical retrieval; causal query pairs, matched multi-hop controls, and natural transfer are what must distinguish manipulation. Before E1 model comparison, finish context-position and answer-format balancing, preregister paired/exact confidence procedures per metric, add source-disjoint natural fixtures, and create an externally held T2 seed/fixture whose content never enters Git.

The hard-coded SEALED seeds used by tests are namespace sentinels only. They are not promotion fixtures and carry no scientific result.
