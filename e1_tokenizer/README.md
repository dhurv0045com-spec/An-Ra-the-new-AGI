# E1 tokenizer tournament harness

E1 does not select a tokenizer from compression alone. This package defines the artifact-bound static audit that every 16k/24k/32k candidate must pass before matched P35 training.

An external tokenizer adapter produces `esoes-e1-candidate-encoding/v1` JSON for every committed probe and binds it to the tokenizer artifact SHA-256. `esoes-e1-audit` verifies byte-exact round trip, zero unknowns, token ranges, sequence inflation, and domain metrics. Serious corpus probes remain external and must be referenced by hash. Passing reports enter a Pareto comparison over sequence cost and vocabulary cost; cognition, code/math, and byte-normalized model loss later decide among non-dominated candidates.

No tokenizer winner exists yet. The harness makes the experiment executable without pretending that static canaries answer the learning question.

`e1_tokenizer.tournament` adds the preregistered three-arm plan (16,384 / 24,576 /
32,768 entries). It requires one external corpus manifest and enforces identical
raw-byte and measured-FLOP budgets; until those artifacts exist its status is
`BLOCKED_EXTERNAL_CORPUS`, not a tokenizer recommendation.

For bounded local development, `e1_tokenizer.local_tournament` independently
trains all three byte-level BPE arms from the same hash-bound text records. It
uses a content-hash holdout so duplicate lines cannot cross train/evaluation,
audits every candidate, repeats the 24k build for byte-level determinism, and
reports held-out tokens/byte by source domain. Example:

```text
python -m e1_tokenizer.local_tournament \
  --source LABEL::DOMAIN::PATH \
  --output-directory artifacts/e1/local_tournament
```

This command needs the optional `tokenizers` package. Its result is explicitly
`DEVELOPMENT_STATIC_PASS`, never an E1 promotion: local sources do not replace
representative external custody or matched P35 byte-normalized loss/cognition.
