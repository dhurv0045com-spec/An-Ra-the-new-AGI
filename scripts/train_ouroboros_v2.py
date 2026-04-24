from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anra_paths import ensure_dirs, get_dataset_file
from scripts.build_brain_v2 import train_anra_v2
from training.v2_runtime import canonical_v2_checkpoint, v2_output_file, write_json

ensure_dirs()


def train_ouroboros_v2(
    *,
    data_path: str | None = None,
    max_minutes: int = 10,
    max_examples: int = 9000,
) -> dict[str, object]:
    started_at = time.time()
    identity_ckpt = canonical_v2_checkpoint("identity")
    resume_name = identity_ckpt.name if identity_ckpt.exists() else canonical_v2_checkpoint("brain").name
    result = train_anra_v2(
        data_path=str(Path(data_path) if data_path else get_dataset_file()),
        checkpoint_path=str(canonical_v2_checkpoint("ouroboros").name),
        resume_from=resume_name,
        max_minutes=max_minutes,
        answer_loss_weight=1.9,
        max_examples=max_examples,
        own_ratio=0.40,
        identity_ratio=0.20,
        teacher_ratio=0.20,
        symbolic_ratio=0.10,
        replay_ratio=0.10,
    )
    report = {
        "generated_at": time.time(),
        "started_at": started_at,
        "stage": "ouroboros_v2",
        "checkpoint": str(canonical_v2_checkpoint("ouroboros")),
        "result": result,
    }
    write_json(v2_output_file("v2_ouroboros_report.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--max_minutes", type=int, default=10)
    parser.add_argument("--max_examples", type=int, default=9000)
    args = parser.parse_args()
    print(
        json.dumps(
            train_ouroboros_v2(
                data_path=args.data_path,
                max_minutes=args.max_minutes,
                max_examples=args.max_examples,
            ),
            indent=2,
        )
    )
