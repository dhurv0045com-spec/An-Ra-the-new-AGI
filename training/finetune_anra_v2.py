from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anra_paths import ensure_dirs, get_dataset_file
from scripts.build_brain_v2 import train_anra_v2
from training.v2_config import V2_REPORT_FILES
from training.v2_runtime import canonical_v2_checkpoint, v2_report_path, write_json

ensure_dirs()


def finetune_identity_v2(
    *,
    data_path: str | None = None,
    max_minutes: int = 12,
    max_examples: int = 8000,
) -> dict[str, object]:
    started_at = time.time()
    result = train_anra_v2(
        data_path=str(Path(data_path) if data_path else get_dataset_file()),
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
    )
    report = {
        "generated_at": time.time(),
        "started_at": started_at,
        "stage": "identity_finetune_v2",
        "checkpoint": str(canonical_v2_checkpoint("identity")),
        "result": result,
    }
    write_json(v2_report_path("finetune_report"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--max_minutes", type=int, default=12)
    parser.add_argument("--max_examples", type=int, default=8000)
    args = parser.parse_args()
    print(
        json.dumps(
            finetune_identity_v2(
                data_path=args.data_path,
                max_minutes=args.max_minutes,
                max_examples=args.max_examples,
            ),
            indent=2,
        )
    )
