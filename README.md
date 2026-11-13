# Hitchhiker — Elendril mechanism investigation

Hitchhiker is an experimental branch for determining which mechanisms improve grounded lookup, variable binding, action selection, and multi-hop generalization. The current deliverable is a repository-grounded research design; implementation and new training are pending.

- [User's original architectural proposal](research/hitchhiker/USER_ARCHITECTURAL_PROPOSAL.md), preserved verbatim from the supplied attachment.
- [Conclusion and experiment design](research/hitchhiker/CONCLUSION_AND_EXPERIMENT_DESIGN.md): code audit, refined mechanisms A–H, matched diagnostics, a 15-run initial screen, falsifiers, and implementation handoff.
- [Branch context and next steps](context.md): source identities, evidence limits, and the next execution sequence.

**Conclusion:** isolate task interpretation, action legality, evidence binding, and composition before combining architectural changes. V7.17 demonstrates optimization sensitivity and transfer failures; it does not establish memorization as the sole cause or a capacity ceiling. Proposed improvements must survive changed facts, renamed entities, reordered evidence, and unsolvable cases.

**Source boundary:** this branch was created from `Elendril` at `2affffb806b7f73b00bc21aa90538019f37ce904`. The audit reviewed later V7.17 code at `0c9cb284d2c8c949e17ac342959c6f2bf351bd30`; that implementation has not been integrated into this branch. Resolve this source gap before implementing the proposed experiments. The documentation below describes the inherited V6 foundation, not newly executed Hitchhiker work.

## Inherited ELENDRIL foundation

ELENDRIL studies cumulative, resource-efficient learning through one neural cognitive loop. V6 adds a coordinated, gated raw-evidence campaign with 15 core diagnostic panels and four additional frozen-checkpoint experiments on the V5 representation, training and recovery foundation. The frozen V4 learner and ELD-DEV-001–004 evidence remain preserved.

- [Current state](docs/CURRENT_STATE.md), [handoff](docs/HANDOFF.md) and [cognitive kernel](docs/COGNITIVE_KERNEL.md)
- [Foundation decisions](docs/FOUNDATION_AUDIT.md), [qualification](docs/FOUNDATION_QUALIFICATION.md) and [algorithm comparisons](docs/ALGORITHM_COMPARISONS.md)
- [Hypotheses](docs/HYPOTHESIS_REGISTER.md), [next cognitive experiment](docs/NEXT_COGNITIVE_EXPERIMENT.md) and [owner operations](docs/OPERATIONS.md)
- [Evidence custody](docs/EVIDENCE_AND_PROVENANCE.md) and [archived programs](docs/ARCHIVED_PROGRAMS.json)
- Current owner notebook: [V6 Kaggle two T4s](notebooks/ELENDRIL_V6_KAGGLE_T4X2.ipynb), [launch instructions](docs/V6_CAMPAIGN.md), [protocol package](research/v6/README_FIRST.md) and [prospective implementation amendment](research/v6/IMPLEMENTATION_AMENDMENT_2026-11-07.md).
- Preserved V5 reproductions: [Colab one T4](notebooks/ELENDRIL_ACTION_COLAB_T4.ipynb), [Kaggle two T4s](notebooks/ELENDRIL_ACTION_KAGGLE_T4X2.ipynb).

Use Python 3.11+, the appropriate native PyTorch, and `python -m pip install -e ".[test]"`. Local hardware is for lightweight correctness tests and inference only.

~~~text
python -m pytest -q
python -m tools.verify_elendril_evidence
python -m elendril --help
python -m elendril inspect-checkpoint --root <checkpoint-root> --lineage <lineage>
python -m elendril verify-checkpoint --root <checkpoint-root> --lineage <lineage>
python -m elendril resume-dry-run --root <checkpoint-root> --lineage <action-lineage>
~~~

Owner-run notebooks keep all launch switches OFF. CUDA and fresh cloud-session recovery remain unverified until owner receipts return. No capability promotion follows from a passing test suite. [License](LICENSE)
