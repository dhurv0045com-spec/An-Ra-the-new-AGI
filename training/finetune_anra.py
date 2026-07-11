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
    parser = argparse.ArgumentParser(description="Canonical identity fine-tune entrypoint")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--max_minutes", type=int, default=12)
    parser.add_argument("--max_examples", type=int, default=8000)
    parser.add_argument("--sparse_lora_mode", choices=["off", "logging"], default="logging")
    parser.add_argument(
        "--optimizer", choices=["auto", "adamw", "muon", "scale", "galore"], default="auto"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            finetune_identity(
                data_path=args.data_path,
                max_minutes=args.max_minutes,
                max_examples=args.max_examples,
                sparse_lora_mode=args.sparse_lora_mode,
                optimizer_name=args.optimizer,
            ),
            indent=2,
        )
    )


finetune_identity_current = finetune_identity


if __name__ == "__main__":
    main()
