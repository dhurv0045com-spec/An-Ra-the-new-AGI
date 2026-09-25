# K8 Kaggle launch setup

The Gandiva K8 notebook can acquire its own source and generated input bundle.
With Internet enabled, it clones the `Gandiva` branch into `/kaggle/working` and,
when no valid K8 bundle is attached, generates the full deterministic bundle
before build verification or GPU training begins. The cloned revision and the
generated bundle manifest become part of the run evidence.

## Attach these inputs

For the automatic path, select **GPU T4 x2** and enable **Internet** in Kaggle
Session options, then run the normal self-sustaining notebook from its first
cell. It uses the public An-Ra Git remote and `Gandiva` branch by default.

Attaching two Kaggle Datasets remains the offline/reproducible alternative.

1. **Gandiva source.** Its root, or a directory one level below its root,
   must contain both `pyproject.toml` and `bramastra_lab/`. Export the exact
   source revision to a Dataset; do not attach a ZIP file that remains
   compressed.
2. **K8 bundle.** Its root, or a directory one level below its root, must
   contain `manifest.json` whose `schema` is `bramastra-k8-data/v1`. Prepare
   and validate this full bundle before launching the GPU notebook:

   ```text
   python -m bramastra_lab.research.campaigns.k8 prepare --out <bundle> --training-mechanisms 4096 --controller-mechanisms 256 --development-mechanisms 256 --confirmation-mechanisms 128 --tool-mechanisms 4096 --tool-heldout 256 --meta-train 24 --meta-validate 6 --meta-confirm 6
   python -m bramastra_lab.research.campaigns.k8 validate --bundle <bundle>
   ```

Kaggle mounts inputs below `/kaggle/input` and gives the notebook a writable
`/kaggle/working`; saving a notebook version retains working outputs.
The setup cell also checks that the selected image supplies `torch>=2.6`,
`numpy`, and `pytest`, which are required by the source and build verifier.
If one is absent, select a compatible image or add the pinned dependency
before starting the campaign; do not repair a dependency after E0 begins.

K8 uses **FP32** by default. This is intentional: its first live two-T4 E0
run exposed an FP16 loss-scale overflow before the first optimizer update.
Do not switch the notebook to `fp16_autocast` without a separately successful
AMP calibration and retry policy.

## Downloading the completed run

The final notebook cell packages the entire generated run root into a verified
ZIP in `/kaggle/working`, writes a SHA-256 receipt beside it, and displays a
download link when Kaggle exposes one. It also runs automatically after a
registered command fails, so failure logs are not lost. Save a Kaggle notebook
version and download the ZIP from the **Output** panel if the in-notebook link
is not available. Kaggle cannot force a browser download without a user click.

## Optional overrides

Set `BRAMASTRA_GIT_URL` or `BRAMASTRA_GIT_REF` only when intentionally using a
different public Git remote or branch. If a source or bundle is mounted in an
unusual location, set `BRAMASTRA_REPO` or `BRAMASTRA_BUNDLE_DIR` to its
directory before running the first code cell. For an independent campaign, set
`BRAMASTRA_RUN_ID` to a new safe filename token. Leave it unchanged for a
restart of the same session so E0 and the full run use the same campaign ledger
and allocation deadline.

The setup cell prints the selected source root, bundle path, run ID, source
identity, configuration identity and both visible CUDA devices. It fails early
when it cannot find a valid input, when the package cannot be imported, or when
Kaggle exposes any GPU count other than two. The build-verification cell uses a
fresh report directory and the export cell uses a fresh output directory; no
prior evidence directory is deleted.

The environment-variable names retain the `BRAMASTRA_` prefix for notebook
compatibility; the default branch is Gandiva. For branch scope and current
owner-run state, see the [Gandiva guide](../GANDIVA.md).
