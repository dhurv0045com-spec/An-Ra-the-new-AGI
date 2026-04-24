from __future__ import annotations

import argparse
import json

from training.finetune_anra_v2 import finetune_identity_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical identity fine-tune entrypoint (V2 mainline)")
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


if __name__ == "__main__":
    main()
