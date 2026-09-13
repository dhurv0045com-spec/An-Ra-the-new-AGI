"""Reproduce chief data/evaluation findings without a model or optimizer step.

Run from the repository: python engineering/reports/B2_CHIEF_20260913/probe.py
Prints observations, not a PASS certification. Temporary fixtures are deleted
by TemporaryDirectory; historical evidence and source files are not mutated.
"""
from pathlib import Path
import contextlib
import io
import json
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bramastra_lab.research.commands import prepare_data, _read_prepared, _load_rows
from bramastra_lab.research.config import BuildConfig
from bramastra_lab.research.data.manifest import load_dataset
from bramastra_lab.research.evaluation.scoring import (
    RawOutcome, clustered_bootstrap_delta, decide_promotion, PromotionConfig,
)


def main():
    results = {"optimizer_updates": 0, "model_instantiations": 0}
    with tempfile.TemporaryDirectory(prefix="bramastra-chief-") as directory:
        root = Path(directory)
        examples = [
            {"example_id": "left", "prompt_events": [["goal", "A?"]],
             "answer": "1", "group": "pair-1", "family": "binding"},
            {"example_id": "right", "prompt_events": [["goal", "B?"]],
             "answer": "2", "group": "pair-1", "family": "binding"},
        ]
        (root / "rows.jsonl").write_text("".join(json.dumps(x)+"\n" for x in examples))
        manifest = {"schema": "bramastra-dataset-manifest/v1", "name": "chief-fixture",
                    "license": "test-only", "provenance": "chief-generated",
                    "entries": [{"path": "rows.jsonl", "split": "training",
                                 "kind": "trajectory", "trainable": True}]}
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        original_identity = load_dataset(str(path)).identity
        manifest["entries"][0]["trainable"] = False
        path.write_text(json.dumps(manifest))
        results["trainability_change_preserves_dataset_identity"] = (
            original_identity == load_dataset(str(path)).identity)
        config_raw = {"model": {"profile": "tiny"}}
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config_raw))
        config = BuildConfig.from_dict(config_raw)
        prepared = root / "prepared"
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_data(str(path), str(prepared), str(config_path))
        rows = _load_rows(str(prepared), "training")
        results["nontrainable_rows_emitted_for_training"] = len(rows)
        results["prepared_pair_group_ids"] = [r["pair_group_id"] for r in rows]
        results["prepared_semantic_ids_are_example_ids"] = [
            r["provenance"]["task_semantic_id"] for r in rows]
        before = _read_prepared(str(prepared), config)["identity"]
        rows[0]["tokens"][1] = 65
        (prepared / "rows-training.jsonl").write_text(
            "".join(json.dumps(r)+"\n" for r in rows))
        results["changed_prepared_tokens_accepted_with_old_identity"] = (
            _read_prepared(str(prepared), config)["identity"] == before
            and _load_rows(str(prepared), "training")[0]["tokens"][1] == 65)

    def outcome(identifier, label, prediction):
        return RawOutcome(identifier, "measurement", "development", "f", "world",
                          prediction, True, label, 0.0)
    report = clustered_bootstrap_delta(
        [outcome("different-task-a", "a", "wrong")],
        [outcome("different-task-b", "b", "b")], iterations=10)
    results["unmatched_cases_same_world_accepted_delta"] = report["delta"]
    report = decide_promotion(reference_metrics={"complete_answer_rate": .1},
        candidate_metrics={"complete_answer_rate": .9},
        reference_family={"f": {"complete_answer_rate": .5}},
        candidate_family={"f": {"complete_answer_rate": .5}},
        paired=None, config=PromotionConfig(), evidence_complete=True)
    results["promotion_without_uncertainty_or_pair_receipt"] = report["decision"]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
