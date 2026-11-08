# ELENDRIL

ELENDRIL studies cumulative, resource-efficient learning through one neural cognitive loop. V7 investigates V6's query-to-answer failure and unstable training-order findings through a preregistered, matched-data study on the existing V5/V6 core. Historical implementations and ELD-DEV-001–004 evidence remain preserved.

- [Current state](docs/CURRENT_STATE.md), [handoff](docs/HANDOFF.md) and [cognitive kernel](docs/COGNITIVE_KERNEL.md)
- [Foundation decisions](docs/FOUNDATION_AUDIT.md), [qualification](docs/FOUNDATION_QUALIFICATION.md) and [algorithm comparisons](docs/ALGORITHM_COMPARISONS.md)
- [Hypotheses](docs/HYPOTHESIS_REGISTER.md), [next cognitive experiment](docs/NEXT_COGNITIVE_EXPERIMENT.md) and [owner operations](docs/OPERATIONS.md)
- [Evidence custody](docs/EVIDENCE_AND_PROVENANCE.md) and [archived programs](docs/ARCHIVED_PROGRAMS.json)
- Current diagnostic study: [V7 Kaggle two T4s](notebooks/ELENDRIL_V7_KAGGLE_T4X2.ipynb), [owner instructions and handoff](research/v7/HANDOFF.md), [protocol](research/v7/PROTOCOL.md) and [historical evidence audit](research/v7/EVIDENCE.md). Owner preflight remains pending; V7 research has not run.
- Preserved V6 campaign: [notebook](notebooks/ELENDRIL_V6_KAGGLE_T4X2.ipynb), [launch instructions](docs/V6_CAMPAIGN.md), [protocol package](research/v6/README_FIRST.md) and [implementation amendment](research/v6/IMPLEMENTATION_AMENDMENT_2026-11-07.md).
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

The V7 notebook requires the owner to set `RUN_STUDY = True`; it automatically obtains source and generates the data. Both physical GPUs must pass learning and same-session fresh-process recovery controls before training. Fresh-VM recovery remains unqualified; historical notebook launch behavior is preserved. No capability promotion follows from a passing test suite. [License](LICENSE)
