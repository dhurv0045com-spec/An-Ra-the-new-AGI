"""V3 scientific contracts exercised without a local training campaign."""

from dataclasses import asdict
from pathlib import Path

import pytest

from anra_v5.elendril import graph_v3
from anra_v5.elendril.__main__ import main
from anra_v5.elendril.codec import BOS, Prediction, answer_tokens, canonical, digest
from anra_v5.elendril.counterfactual import CounterfactualBatches


@pytest.mark.parametrize("surface, development_rows", (
    ("graph-v3", 96), ("graph-v3-replication", 192),
))
def test_topology_split_oracles_and_trainer_identities(surface, development_rows):
    audit = graph_v3.graph_audit(surface)
    assert (audit["training_rows"], audit["development_rows"]) == (256, development_rows)
    assert (audit["training_clusters"], audit["development_clusters"]) == (16, 6)
    base = audit["panels"]["base"]["controls"]
    assert base["oracle"]["exact_eos"] == [development_rows, development_rows]
    assert base["one-edge"]["exact_eos"] == [0, development_rows]
    for name in ("first", "last", "frequency"):
        assert base[name]["query_swap_both_correct"] == [0, development_rows // 2]
    for transform in graph_v3.TRANSFORMS:
        rows = graph_v3.graph_fixture(surface, "train", transform=transform)
        paired = CounterfactualBatches(rows, 16)
        assert paired.block_steps == 32
        assert audit["training_fixture_sha256"][transform] == digest([
            (asdict(row.visible), asdict(row.truth)) for row in rows
        ])


def test_independent_classes_and_semantic_rerender():
    train = graph_v3.graph_fixture("graph-v3", "train")
    augmented = graph_v3.graph_fixture("graph-v3", "train", transform="augment")
    assert len(train) == len(augmented)
    assert [row.visible.task_id for row in train] == [row.visible.task_id for row in augmented]
    assert [row.visible.cluster_id for row in train] == [row.visible.cluster_id for row in augmented]
    assert any(old.truth.gold != new.truth.gold for old, new in zip(train, augmented))
    for row in augmented:
        depth, edges, root = graph_v3.parse(row.visible.prompt)
        assert all("A" <= token <= "P" for edge in edges for token in edge)
        assert graph_v3.solve(edges, root, depth) == row.truth.gold
    screen, replica = (graph_v3.graph_fixture(surface, "development") for surface in graph_v3.SURFACES)
    assert not ({row.visible.cluster_id for row in screen} &
                {row.visible.cluster_id for row in replica})


def test_factorial_shift_and_rewire_control():
    raw = graph_v3.graph_fixture("graph-v3", "development")
    for panel in ("alias-symbol", "alias-separator", "alias-combined"):
        shifted = graph_v3.graph_fixture("graph-v3", "development", panel)
        for before, after in zip(raw, shifted):
            assert graph_v3.normalize(before.visible.prompt)[0] == graph_v3.normalize(after.visible.prompt)[0]
    rewired = graph_v3.graph_fixture("graph-v3", "development", "rewire")
    for before, after in zip(raw, rewired):
        assert sorted(before.visible.prompt) == sorted(after.visible.prompt)
        assert before.truth.gold != after.truth.gold
    for panel in ("absent", "ambiguous"):
        rows = graph_v3.graph_fixture("graph-v3", "development", panel)
        assert sum(row.truth.gold == "!" for row in rows) == len(rows) // 2


def test_canonical_receipt_independently_decodes_and_rejects_alteration(tmp_path: Path):
    from anra_v5.elendril.scoring import score

    rows = graph_v3.graph_fixture("graph-v3", "development")
    prompts, inverse = graph_v3.inference_inputs(rows, "base", "recurrent-canon")
    original = tuple(Prediction(row.visible.task_id, answer_tokens(row.truth.gold)) for row in rows)
    native = []
    for prediction in original:
        mapping = {raw: canonical_id for canonical_id, raw in inverse[prediction.task_id].items()}
        native.append(Prediction(prediction.task_id, tuple(mapping.get(i, i) for i in prediction.tokens)))
    assert graph_v3.restore_predictions(tuple(native), inverse) == original
    assert graph_v3.restore_predictions((Prediction(original[0].task_id, answer_tokens("Z")),),
                                        inverse)[0].tokens[0] == BOS
    raw = [asdict(prediction) for prediction in original]
    projected = [asdict(prediction) for prediction in native]
    snapshot = {
        "predictions": raw, "predictions_sha256": digest(raw),
        "model_predictions": projected, "model_predictions_sha256": digest(projected),
        "model_prompts_sha256": digest(prompts), "metrics": score(rows, original),
    }
    receipt = {"arm": "binding", "surface": "graph-v3", "method": "recurrent-canon", "status": "partial",
               "development_sha256": digest([asdict(row) for row in rows]), "evaluations": [snapshot]}
    target = tmp_path / "receipt.json"
    target.write_bytes(canonical(receipt))
    main(["rescore", "--receipt", str(target)])
    snapshot["model_predictions"][0]["tokens"] = answer_tokens("Z")
    snapshot["model_predictions_sha256"] = digest(snapshot["model_predictions"])
    target.write_bytes(canonical(receipt))
    with pytest.raises(ValueError, match="canonical decoding mismatch"):
        main(["rescore", "--receipt", str(target)])
