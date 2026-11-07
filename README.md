# ELENDRIL

ELENDRIL studies cumulative, resource-efficient learning through one neural cognitive loop. V5 strengthens its representation, training and recovery foundation while preserving the frozen V4 learner and ELD-DEV-001–004 evidence.

- [Current state](docs/CURRENT_STATE.md), [handoff](docs/HANDOFF.md) and [cognitive kernel](docs/COGNITIVE_KERNEL.md)
- [Foundation decisions](docs/FOUNDATION_AUDIT.md), [qualification](docs/FOUNDATION_QUALIFICATION.md) and [algorithm comparisons](docs/ALGORITHM_COMPARISONS.md)
- [Hypotheses](docs/HYPOTHESIS_REGISTER.md), [next cognitive experiment](docs/NEXT_COGNITIVE_EXPERIMENT.md) and [owner operations](docs/OPERATIONS.md)
- [Evidence custody](docs/EVIDENCE_AND_PROVENANCE.md) and [archived programs](docs/ARCHIVED_PROGRAMS.json)

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
