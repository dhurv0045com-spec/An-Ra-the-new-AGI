# ESOES Decision Log

This file prevents silent redesign.

## 2026-08-29 — Branch created

Base: `core-vnext` at `054619f20851317e9b1c49b6f31599f6444a8280`.

Decision:

- create a separate research/design branch rather than modifying the active V4 training branch;
- treat V4, EXP, and VNEXT as evidence, not immutable architecture;
- do not launch a major V5 training run until the cognition-first blueprint is stress-tested and frozen;
- preserve open questions explicitly rather than allowing execution agents to answer them implicitly in code.

Current status: **V5 design open; no final architecture/tokenizer/data/optimizer/scale decision frozen.**