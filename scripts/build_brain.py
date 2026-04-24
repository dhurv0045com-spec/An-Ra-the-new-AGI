from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anra_paths import ensure_dirs, inject_all_paths
from scripts.build_brain_v2 import train_anra_v2

inject_all_paths()
ensure_dirs()


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical An-Ra base trainer (V2 mainline)")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint_path", default="anra_v2_brain.pt")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--max_minutes", type=int, default=30)
    parser.add_argument("--answer_loss_weight", type=float, default=1.75)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--own_ratio", type=float, default=None)
    parser.add_argument("--identity_ratio", type=float, default=None)
    parser.add_argument("--teacher_ratio", type=float, default=None)
    parser.add_argument("--symbolic_ratio", type=float, default=None)
    parser.add_argument("--replay_ratio", type=float, default=None)
    args = parser.parse_args()

    result = train_anra_v2(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        resume_from=args.resume_from,
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_minutes=args.max_minutes,
        answer_loss_weight=args.answer_loss_weight,
        max_examples=args.max_examples,
        own_ratio=args.own_ratio,
        identity_ratio=args.identity_ratio,
        teacher_ratio=args.teacher_ratio,
        symbolic_ratio=args.symbolic_ratio,
        replay_ratio=args.replay_ratio,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
