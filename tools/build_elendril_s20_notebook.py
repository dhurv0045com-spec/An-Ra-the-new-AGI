"""Publish the registered six-job S20 study with a measured S8-based allowance."""
import argparse
import json
from pathlib import Path

from tools.build_elendril_v715_notebook import build as build_release

DEST = Path(__file__).resolve().parents[1] / 'notebooks/ELENDRIL_V715_S20_KAGGLE_T4X2.ipynb'


def build(source):
    notebook = build_release(source)
    notebook['cells'][0]['source'] = (f'''# ELENDRIL V7.15 S20 · Kaggle T4×2

Run All starts all six registered S20 jobs: uniform and blocked CE at seeds
7, 19 and 41, four epochs each, with fresh 19,871,616-parameter models.
Training exposure matches the completed S8 comparison. No input attachment is required.

The S8 controller took 45.68 minutes before finalization; twice that is 91.36
minutes, an approximate planning target. Its conservative admission forecast
was 98.86 minutes. This notebook permits 200 minutes (approximately twice
that forecast), including measured admission and export reserve. Actual T4
canaries determine whether the entire six-job study fits. Runtime is unmeasured
for S20. An unaffordable batch produces NO_GO with saved diagnostic receipts.

Select GPU T4×2 and enable Internet. Native Kaggle PyTorch is used.
Pinned runtime source: {source}.

Checkpoints, raw episode traces and partial results are preserved in the
source-specific output directory. Download ELENDRIL_V71_ANALYSIS.zip,
ELENDRIL_V71_FULL.zip and FINAL_STATUS.json from the final cell links.
Keep original ZIPs outside Git. COMPLETE requires registered reports and
both verified archives. Hardware qualification remains owner-run.

For separate predeclared sessions, PHASE may be scale-s20-seed7,
scale-s20-seed19 or scale-s20-seed41. Never count a seed pair as the full study.
''').splitlines(keepends=True)
    config = ''.join(notebook['cells'][1]['source'])
    config = config.replace('PHASE = "scale-s8"', 'PHASE = "scale-s20"')
    config = config.replace('MINUTES = 330', 'MINUTES = 200')
    notebook['cells'][1]['source'] = config.splitlines(keepends=True)
    notebook['metadata']['elendril_s20_status'] = 'OWNER_GPU_QUALIFICATION_PENDING'
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--output', type=Path, default=DEST)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(args.source_sha), indent=2, ensure_ascii=False) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
