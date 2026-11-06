# ELENDRIL

ELENDRIL studies whether neural learners can acquire reusable skills, retain them, transfer prior learning to new tasks, and improve through independently verified interventions.

The active tree contains the research model and operator, the small V5 dependency closure required for exact replay, regression tests, and ELD-DEV-001 through ELD-DEV-004 evidence. Older AN-RA programs live at their archived Git identities. [Evidence and provenance](docs/EVIDENCE_AND_PROVENANCE.md)

## Research entry points

- [Current state](docs/CURRENT_STATE.md) and [roadmap](docs/RESEARCH_ROADMAP.md)
- [Ranked hypotheses](docs/HYPOTHESIS_REGISTER.md) and [experiment protocols](docs/EXPERIMENT_PROTOCOLS.md)
- [Cloud operations and launch gates](docs/OPERATIONS.md)
- [Colab single-T4 screening](notebooks/ELENDRIL_SCREEN_COLAB_T4.ipynb)
- [Kaggle dual-T4 confirmation](notebooks/ELENDRIL_CONFIRM_KAGGLE_T4X2.ipynb)

## Local qualification

Use Python 3.11 or later and install PyTorch for the actual platform. Local work is limited to source checks, receipt replay, unit tests and tiny correctness canaries. Train research campaigns only in owner-run cloud notebooks.

~~~text
python -m pip install -e ".[test]"
python -m elendril --help
python -m pytest -q
~~~

The `elendril` command and the historical `python -m anra_v5.elendril` command run the same entry point. Historical receipt replay uses each receipt's original source where needed. [Official PyTorch installation](https://pytorch.org/get-started/locally/)

[License](LICENSE)
