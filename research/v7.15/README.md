# V7.15 — Elendril research release

**Exact version: V7.15.** V7.1 core results remain historically separate. This release adds source-pinned Kaggle execution for scale, frozen diagnostics, continual learning and historical V7 comparison. **CPU/publication qualification passed (370 tests, 14 subtests); actual V7.15 GPU phases still require owner Kaggle execution.**

- [Kaggle notebook and detailed launch instructions](RELEASE.md)
- [Completion plan, scientific gates and additional future studies](COMPLETION_PLAN.md)
- [Current experiment delivery status](../../Experiment%20Result/V7.15/STATUS.json)
- [Previous V7.1 experiment report](../../Experiment%20Result/V7.1/20c80f66e76e4b589aed2053afbddffc/RESULTS.md)

The runnable notebook is in [notebooks/ELENDRIL_V715_KAGGLE_T4X2.ipynb](../../notebooks/ELENDRIL_V715_KAGGLE_T4X2.ipynb). Select T4×2 and enable Internet. Run All defaults to the 12-job larger-model scale phase without input files; frozen diagnostics and continual learning require the original V7.1 core FULL archive.

Scientific questions include route composition without obvious shortcuts, sensitivity to row order and timestamps, memory correction against equal-length controls, query availability, balanced evidence changes, model-size scaling and retained acquisition. Do not conflate engineered controls, CPU tests, archival delivery and neural cognition.

- [Source-pinned release qualification and limitations](RELEASE_QUALIFICATION.json)
