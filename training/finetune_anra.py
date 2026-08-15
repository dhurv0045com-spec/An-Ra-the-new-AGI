"""Retired identity fine-tune implementation retained for forensic imports.

This module used to expose a runnable ``python -m training.finetune_anra``
path.  That path wrote an independent ``identity`` checkpoint through the
pre-SFT training route, so it could silently fork a V4 model away from its
signed base-checkpoint and SFT lineage.  The implementation remains importable
for historical report inspection, but the command-line entrypoint is now
fail-closed.  Canonical post-training is ``python -m training.sft_v4`` via the
reviewed V4 SFT notebook/contract.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from anra.anra_paths import ensure_dirs, get_dataset_file
from scripts.build_brain import train_anra_v2

from training.sparse_lora import SparseLoRAEstimateConfig, write_sparse_lora_report_from_dataset
from training.v2_runtime import canonical_v2_checkpoint, v2_report_path, write_json

ensure_dirs()


def finetune_identity(
    *,
    data_path: str | None = None,
    max_minutes: int = 12,
    max_examples: int = 8000,
    sparse_lora_mode: str = "logging",
    optimizer_name: str = "auto",
) -> dict[str, object]:
    started_at = time.time()
    resolved_data_path = Path(data_path) if data_path else get_dataset_file()
    sparse_lora_report = None
    if sparse_lora_mode != "off":
        sparse_lora_report = write_sparse_lora_report_from_dataset(
            resolved_data_path,
            config=SparseLoRAEstimateConfig(
                mode="logging_only",
                max_examples=min(512, max(1, max_examples)),
            ),
            output_path=v2_report_path("sparse_lora_report"),
        )
    result = train_anra_v2(
        data_path=str(resolved_data_path),
        checkpoint_path=str(canonical_v2_checkpoint("identity").name),
        resume_from=str(canonical_v2_checkpoint("brain").name),
        max_minutes=max_minutes,
        answer_loss_weight=2.0,
        max_examples=max_examples,
        own_ratio=0.45,
        identity_ratio=0.35,
        teacher_ratio=0.08,
        symbolic_ratio=0.04,
        replay_ratio=0.08,
        optimizer_name=optimizer_name,
        continuation_phase="D",
    )
    report = {
        "generated_at": time.time(),
        "started_at": started_at,
        "stage": "identity_finetune_v2",
        "checkpoint": str(canonical_v2_checkpoint("identity")),
        "sparse_lora": sparse_lora_report,
        "result": result,
    }
    write_json(v2_report_path("finetune_report"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-migration",
        action="store_true",
        help="Print the supported V4 SFT migration target without training.",
    )
    args, unknown = parser.parse_known_args()
    guidance = {
        "status": "retired_identity_finetune_entrypoint",
        "reason": (
            "This old path can create an unreviewed identity child checkpoint "
            "outside the signed V4 SFT lineage."
        ),
        "canonical_sft": "python -m training.sft_v4 --help",
        "canonical_foundation": "python -m training.train_unified --help",
        "received_legacy_arguments": unknown,
    }
    print(json.dumps(guidance, indent=2))
    if not args.show_migration:
        raise SystemExit(2)


finetune_identity_current = finetune_identity


if __name__ == "__main__":
    main()
