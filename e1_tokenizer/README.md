# E1 tokenizer tournament harness

E1 does not select a tokenizer from compression alone. This package defines the artifact-bound static audit that every 16k/24k/32k candidate must pass before matched P35 training.

An external tokenizer adapter produces `esoes-e1-candidate-encoding/v1` JSON for every committed probe and binds it to the tokenizer artifact SHA-256. `esoes-e1-audit` verifies byte-exact round trip, zero unknowns, token ranges, sequence inflation, and domain metrics. Serious corpus probes remain external and must be referenced by hash. Passing reports enter a Pareto comparison over sequence cost and vocabulary cost; cognition, code/math, and byte-normalized model loss later decide among non-dominated candidates.

No tokenizer winner exists yet. The harness makes the experiment executable without pretending that static canaries answer the learning question.
