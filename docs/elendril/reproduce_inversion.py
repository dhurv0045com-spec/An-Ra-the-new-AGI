"""Reproduce ELD-DEV-002I from the registered model source checkout."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
import time

from anra_v5.elendril.__main__ import committed_source, write_receipt
from anra_v5.elendril.benchmark import fixture
from anra_v5.elendril.codec import digest
from anra_v5.elendril.counterfactual import CounterfactualBatches
from anra_v5.elendril.scoring import score
from anra_v5.elendril.training import BATCH_SIZE, Trainer, configure_cpu, data_sha, generate


PILOT_SOURCE = "77edcbd8dce7b73d7562e619351a9cc496c6ef84"


def invert(rows):
    pairs = CounterfactualBatches(rows, BATCH_SIZE)
    return tuple(replace(row, truth=replace(row.truth, gold=rows[pairs.partners[index]].truth.gold))
                 for index, row in enumerate(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(7, 19, 41), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("output must be a new file inside an existing directory")
    if committed_source() != PILOT_SOURCE:
        raise ValueError("inversion diagnostic requires the registered source checkout")
    configure_cpu()
    started = time.perf_counter()
    train = invert(fixture("binding", "train"))
    dev = fixture("binding", "development")
    inverted_dev = invert(dev)
    trainer = Trainer(train, seed=args.seed, steps=336, root=args.root, lineage=f"inverted-{args.seed}",
                      source_commit=PILOT_SOURCE, method="associative")
    telemetry = []
    for update in range(1, 337):
        telemetry.append(trainer.step())
        if update % 84 == 0:
            trainer.save()

    def evaluate(rows, alternative=None):
        prompts = [(row.visible.task_id, row.visible.prompt) for row in rows]
        predictions, work = generate(trainer.model, prompts, max_new_tokens=3)
        raw = [asdict(prediction) for prediction in predictions]
        result = {"metrics": score(rows, predictions), "predictions": raw,
                  "work": work, "predictions_sha256": digest(raw)}
        if alternative is not None:
            result["alternative"] = score(alternative, predictions)
        return result

    write_receipt(args.output, {"source_commit": PILOT_SOURCE,
                  "operator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                  "seed": args.seed, "steps": 336, "training_tokens": trainer.state.cumulative_tokens,
                  "identity": trainer.state.identities.canonical(), "updates": telemetry,
                  "wall_seconds": time.perf_counter() - started,
                  "train_sha256": data_sha(train), "development_sha256": data_sha(inverted_dev),
                  "final_checkpoint_sha256": trainer.parent,
                  "train_inverted": evaluate(train), "development_inverted": evaluate(inverted_dev, dev)})


if __name__ == "__main__":
    main()
