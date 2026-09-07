# W09 — Language/code data and natural interfaces

**Status:** inventory/design ready; data execution depends on accessible sources and W01. **Effort:** 3–5 hours. **Role:** data and interface engineer. **Compute:** CPU inventory, tokenizer fixtures and bounded preprocessing; no large download/pretraining by default.

Read DATA_CONTRACTS.md and SYSTEM_ARCHITECTURE.md capacity/breadth sections. Own `research/data/`, language adapters, data tests and `engineering/reports/W09/`.

## Deliverable

Create a verified small corpus pipeline and natural-language/code interfaces for the existing qualified tasks. The objective is to test whether learned mechanisms survive natural descriptions and program interfaces, not to replace environment learning with a large unexamined text scrape.

Inventory accessible local material with source/license/provenance, exact hashes, semantic/document deduplication and split assignment. Produce a training-only tokenizer policy: either a declared inherited tokenizer prior with known limitations or a tokenizer trained only on the permitted training corpus. No pretrained neural weights.

## Required comparison

Structured task inputs versus independently authored natural instructions, renamed variables and distractors. Keep the underlying semantic tasks paired. Hold out renderings and task structures separately so syntax shift and mechanism shift are not conflated.

For code tasks, use a small bounded interpreter or controlled test runner. A hidden oracle solving the program outside the learner is a diagnostic, not model competence.

## Acceptance evidence

- Global unique counts come from content union, not sums of overlapping receipt totals.
- Tokenizer training data excludes declared evaluation material; unknown provenance is reported.
- Packed document boundaries, loss masks and EOS targets are correct.
- Natural-language variants are semantically checked independently and do not reveal answers through naming/templates.
- Corpus/sample/pack/tokenizer manifests reproduce the same batches.
- A compact paired dataset and evaluation adapter are usable by W03/W08 without network access.
- The handoff proposes a bounded B1 language-support experiment only if data quality and measured runtime justify it.

Do not claim arbitrary raw token volume produces AGI. Do not introduce pretrained-model-generated answers without explicit provenance and a separate from-scratch interpretation. If source access is missing, deliver the validator and qualified local fixture, naming exactly what remains unavailable.
