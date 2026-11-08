# ELENDRIL

ELENDRIL studies cumulative, resource-efficient learning through one neural cognitive loop. V7.1 core completed nine owner-run models; its verified results show narrow evidence dependence alongside brittle serialization, action-availability and memory-correction behavior. Historical implementations and ELD-DEV-001–004 evidence remain preserved. Underlying neural causes remain unresolved; paired fixed-checkpoint diagnostics precede further training.

- [Current state](docs/CURRENT_STATE.md), [handoff](docs/HANDOFF.md) and [cognitive kernel](docs/COGNITIVE_KERNEL.md)
- [Foundation decisions](docs/FOUNDATION_AUDIT.md), [qualification](docs/FOUNDATION_QUALIFICATION.md) and [algorithm comparisons](docs/ALGORITHM_COMPARISONS.md)
- [Hypotheses](docs/HYPOTHESIS_REGISTER.md), [next cognitive experiment](docs/NEXT_COGNITIVE_EXPERIMENT.md) and [owner operations](docs/OPERATIONS.md)
- [Evidence custody](docs/EVIDENCE_AND_PROVENANCE.md) and [archived programs](docs/ARCHIVED_PROGRAMS.json)
- Preserved V7.1 core reproduction: [Kaggle two T4s](notebooks/ELENDRIL_V71_KAGGLE_T4X2.ipynb), [launch and recovery instructions](research/v7.1/README.md), [protocol](research/v7.1/PROTOCOL.md), [hypotheses](research/v7.1/HYPOTHESES.md) and [CPU data qualification](research/v7.1/DATA_QUALIFICATION.json).
- [Completed V7.1 core results](Experiment%20Result/V7.1/20c80f66e76e4b589aed2053afbddffc/RESULTS.md) preserve raw evidence, negative findings, measured costs and the [next paired diagnostic iteration](research/v7.1/NEXT_EXPERIMENT.md). The original core notebook remains a reproduction, not the new follow-up launcher.
- Completed V7: [preserved notebook](notebooks/ELENDRIL_V7_KAGGLE_T4X2.ipynb), [handoff](research/v7/HANDOFF.md), [protocol](research/v7/PROTOCOL.md), [historical evidence audit](research/v7/EVIDENCE.md) and [owner-run results](Experiment%20Result/V7/c9981309a94847a4a97d8115c56d75c8/RESULTS.md), documenting all nine runs and measured successes and failures.
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

The owner-authorized V7 notebook launches with Run All; `RUN_STUDY = True` is already set. It automatically obtains source and generates data. Both physical GPUs must pass learning and same-session fresh-process recovery controls before training. Completion additionally requires all registered results and both verified current-run ZIPs. Fresh-VM recovery remains unqualified. No capability promotion follows from passing tests or completing a campaign. [License](LICENSE)
