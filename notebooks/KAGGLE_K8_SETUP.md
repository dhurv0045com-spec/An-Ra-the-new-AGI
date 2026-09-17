# K8 Kaggle launch setup

The K8 notebook is designed to run from immutable Kaggle input artifacts. It
does not clone a branch, install the repository, download packages or generate
the full data bundle inside the GPU session. That keeps the run reproducible
and makes a missing input fail before any campaign allocation.

## Attach these inputs

Attach two Kaggle Datasets to the notebook.

1. **BRAMASTRA source.** Its root, or a directory one level below its root,
   must contain both `pyproject.toml` and `bramastra_lab/`. Export the exact
   source revision to a Dataset; do not attach a ZIP file that remains
   compressed.
2. **K8 bundle.** Its root, or a directory one level below its root, must
   contain `manifest.json` whose `schema` is `bramastra-k8-data/v1`. Prepare
   and validate this full bundle before launching the GPU notebook:

   ```text
   python -m bramastra_lab.research.campaigns.k8 prepare --out <bundle> --training-mechanisms 4096 --controller-mechanisms 256 --development-mechanisms 256 --confirmation-mechanisms 128 --tool-mechanisms 256 --tool-heldout 64 --meta-train 24 --meta-validate 6 --meta-confirm 6
   python -m bramastra_lab.research.campaigns.k8 validate --bundle <bundle>
   ```

Select **GPU T4 x2**. Internet is not required for the notebook path. Kaggle
mounts inputs below `/kaggle/input` and gives the notebook a writable
`/kaggle/working`; saving a notebook version retains working outputs.
The setup cell also checks that the selected image supplies `torch>=2.6`,
`numpy`, and `pytest`, which are required by the source and build verifier.
If one is absent, select a compatible image or add the pinned dependency
before starting the campaign; do not repair a dependency after E0 begins.

## Optional overrides

If the source or bundle is mounted in an unusual location, set
`BRAMASTRA_REPO` or `BRAMASTRA_BUNDLE_DIR` to its directory before running the
first code cell. For an independent campaign, set `BRAMASTRA_RUN_ID` to a new
safe filename token. Leave it unchanged for a restart of the same session so
E0 and the full run use the same campaign ledger and allocation deadline.

The setup cell prints the selected source root, bundle path, run ID, source
identity, configuration identity and both visible CUDA devices. It fails early
when it cannot find a valid input, when the package cannot be imported, or when
Kaggle exposes any GPU count other than two. The build-verification cell uses a
fresh report directory and the export cell uses a fresh output directory; no
prior evidence directory is deleted.
