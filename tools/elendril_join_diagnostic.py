"""Read-only first-read and alias-factor diagnostics for ELD-DEV-004."""

import argparse
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
from unittest.mock import patch

from anra_v5.elendril.__main__ import committed_source, write_receipt
from anra_v5.elendril.codec import decode, digest, encode
from anra_v5.elendril.join import join_fixture, parse, solve
from anra_v5.elendril.scoring import score
from anra_v5.elendril.training import Trainer, configure_cpu, generate


SOURCE = "6615886b1a37fd535682bdee38157e4c44cd0783"


def assess(model, rows):
    predictions, work = generate(model, [(r.visible.task_id, r.visible.prompt) for r in rows],
                                 max_new_tokens=3)
    raw = [asdict(prediction) for prediction in predictions]
    return {"score": score(rows, predictions), "work": work, "predictions": raw,
            "prediction_sha256": digest(raw)}


def diagnostic(trainer):
    model = trainer.model
    rows = join_fixture("development")
    alias = join_fixture("development", "alias")
    aliases_only, separator_only = [], []
    for base, renamed in zip(rows, alias):
        sep = "#" if "#" in base.visible.prompt else ";"
        aliases_only.append(replace(renamed, visible=replace(
            renamed.visible, prompt=renamed.visible.prompt.replace("-", sep))))
        separator_only.append(replace(base, visible=replace(
            base.visible, prompt=base.visible.prompt.replace(sep, "-"))))
    evidence = {"aliases_only": assess(model, tuple(aliases_only)),
                "separator_only": assess(model, tuple(separator_only))}
    observations, interventions, correct_first = [], [], 0
    original_join = model._join_scores
    for index, row in enumerate(rows):
        edges, root = parse(row.visible.prompt)
        _, other_root = parse(rows[index ^ 1].visible.prompt)
        middle = solve(edges, root, 1)
        other_middle = solve(edges, other_root, 1)
        wrong_terminal = solve(edges, other_root, 2)
        prompt_length = len(row.visible.prompt) + 1
        first_tokens = []

        def inspect(first, tokens, positions):
            if tokens.size(1) == prompt_length:
                first_tokens.append(int(first[0,-1].argmax()))
            return original_join(first, tokens, positions)

        with patch.object(model, "_join_scores", side_effect=inspect):
            baseline = assess_one(model, row)
        assert len(first_tokens) == 1
        first_token = first_tokens[0]
        correct_first += first_token == encode(middle)[0]

        def intervene(first, tokens, positions):
            if tokens.size(1) == prompt_length:
                first = first.clone()
                first[0, -1].zero_()
                first[0, -1, encode(other_middle)[0]] = 1.0
            return original_join(first, tokens, positions)

        with patch.object(model, "_join_scores", side_effect=intervene):
            altered = assess_one(model, row)
        observations.append({"task_id": row.visible.task_id, "middle": middle,
                             "first_token": first_token, "baseline": asdict(baseline),
                             "intervened": asdict(altered)})
        interventions.append(replace(row, truth=replace(row.truth, gold=wrong_terminal)))

    from anra_v5.elendril.codec import Prediction

    evidence.update(first_hop_exact=[correct_first, len(rows)], observations=observations,
                    forced_wrong_middle=score(tuple(interventions), (
                        Prediction(r["task_id"], r["intervened"]["tokens"]) for r in observations)))
    return evidence


def assess_one(model, row):
    return generate(model, [(row.visible.task_id, row.visible.prompt)], max_new_tokens=3)[0][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if committed_source() != SOURCE:
        raise ValueError("use a clean checkout of the original ELD-DEV-004 source")
    configure_cpu()
    results = []
    for seed in (7, 19, 41):
        trainer = Trainer(join_fixture("train"), seed=seed, steps=384, root=args.root,
                          lineage=f"two-read-join-v1-binding-{seed}", source_commit=SOURCE,
                          method="two-read", resume=True)
        assert trainer.state.complete
        results.append({"seed": seed, "checkpoint_sha256": trainer.parent, **diagnostic(trainer)})
    write_receipt(args.output, {"source_commit":SOURCE, "operator_sha256":hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest(), "optimizer_updates":0, "results":results})


if __name__ == "__main__":
    main()
