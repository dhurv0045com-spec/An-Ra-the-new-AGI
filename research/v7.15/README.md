# V7.15 — Elendril research release

**Exact version: V7.15.** V7.1 core results remain historically separate. This release adds source-pinned Kaggle execution for scale, frozen diagnostics, continual learning and historical V7 comparison. CPU and publication qualification passed with 372 tests and 14 subtests. The owner S8 GPU batch has since executed and its analysis passed independent replay.

- [Archived S8 results and next steps](../../Experiment%20Result/V7.15/3981e34a2ffe4b4b9528b4b4d588883b/RESULTS.md)
- [Kaggle notebook and detailed launch instructions](RELEASE.md)
- [Completion plan, scientific gates and additional future studies](COMPLETION_PLAN.md)
- [Current experiment delivery status](../../Experiment%20Result/V7.15/STATUS.json)
- [Previous V7.1 experiment report](../../Experiment%20Result/V7.1/20c80f66e76e4b589aed2053afbddffc/RESULTS.md)

The runnable notebook is in [notebooks/ELENDRIL_V715_KAGGLE_T4X2.ipynb](../../notebooks/ELENDRIL_V715_KAGGLE_T4X2.ipynb). Select T4×2 and enable Internet. Run All defaults to a six-job S8 batch without input files; the three S20 seed batches complete the 12-job matrix across sessions; frozen diagnostics and continual learning require the original V7.1 core FULL archive.

Scientific questions include route composition without obvious shortcuts, sensitivity to row order and timestamps, memory correction against equal-length controls, query availability, balanced evidence changes, model-size scaling and retained acquisition. Do not conflate engineered controls, CPU tests, archival delivery and neural cognition.

- [Source-pinned release qualification and limitations](RELEASE_QUALIFICATION.json)
