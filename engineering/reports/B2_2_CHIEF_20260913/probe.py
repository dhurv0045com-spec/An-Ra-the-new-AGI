"""Non-learning reproductions against acfa249; no model or optimizer creation."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bramastra_lab.research.commands import _load_rows, _read_prepared, prepare_data
from bramastra_lab.research.config import BuildConfig
from bramastra_lab.research.data.manifest import load_dataset
from bramastra_lab.research.evaluation.scoring import (
    EvidenceBundle, EvaluationProtocol, PromotionConfig, decide_promotion,
)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


result = {"source_commit": "acfa249", "model_instantiations": 0, "optimizer_updates": 0}
with tempfile.TemporaryDirectory(prefix="bramastra-chief-b22-") as temporary:
    base = Path(temporary)
    rows_path = base / "rows.jsonl"
    rows = [{"example_id": "excluded", "text": "excluded source", "trainable": False},
            {"example_id": "eligible", "text": "eligible source", "trainable": True}]
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest_path = base / "manifest.json"
    write_json(manifest_path, {"schema": "bramastra-dataset-manifest/v1",
               "name": "chief-row-scope-fixture", "license": "CC0",
               "provenance": "chief authored non-learning fixture",
               "entries": [{"path": "rows.jsonl", "split": "training",
                            "kind": "language", "trainable": True}]})
    handle = load_dataset(str(manifest_path))
    result["per_row_trainability_false_then_true"] = {
        example.example_id: example.trainable for example in handle.examples}
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)), encoding="utf-8")
    handle = load_dataset(str(manifest_path))
    result["per_row_trainability_reversed_order"] = {
        example.example_id: example.trainable for example in handle.examples}

    config_path = base / "config.json"
    write_json(config_path, {"model": {"profile": "tiny"}})
    prepared_dir = base / "prepared"
    with contextlib.redirect_stdout(io.StringIO()):
        prepare_data(str(ROOT / "engineering/reports/B2/fixtures/manifest.json"),
                     str(prepared_dir), str(config_path))
    config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
    old_identity = _read_prepared(str(prepared_dir), config)["identity"]
    data_path = prepared_dir / "rows-training.jsonl"
    prepared_rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    prepared_rows[0]["tokens"][1] = (prepared_rows[0]["tokens"][1] + 1) % 260
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in prepared_rows).encode("utf-8")
    data_path.write_bytes(payload)
    prepared_manifest_path = prepared_dir / "prepared.json"
    prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    prepared_manifest["split_integrity"]["training"]["rows_sha256"] = hashlib.sha256(payload).hexdigest()
    write_json(prepared_manifest_path, prepared_manifest)
    accepted = _read_prepared(str(prepared_dir), config)
    loaded_rows = _load_rows(str(prepared_dir), "training")
    result["changed_rows_and_digest_accepted_under_old_prepared_identity"] = (
        accepted["identity"] == old_identity and loaded_rows == prepared_rows)

protocol = EvaluationProtocol("chief-proof", requires_uncertainty=True)
base_bundle = dict(parent_identity="parent", child_identity="child", protocol=protocol,
                   pool="confirmation", data_identity="dataset",
                   reference_metrics={"complete_answer_rate": 0.5},
                   candidate_metrics={"complete_answer_rate": 0.8},
                   reference_family={"f": {"complete_answer_rate": 0.5}},
                   candidate_family={"f": {"complete_answer_rate": 0.8}},
                   sealed_fresh=True)
negative_ci = {"clusters": 10, "ci_low": -0.3, "ci_high": -0.1,
               "ci_includes_zero": False, "delta": -0.2}
result["promotion_with_entirely_negative_inconsistent_interval"] = decide_promotion(
    EvidenceBundle(**base_bundle, clustered_uncertainty=negative_ci), PromotionConfig())["decision"]
base_bundle["candidate_family"] = {"f": {"complete_answer_rate": float("nan")}}
positive_ci = {"clusters": 10, "ci_low": 0.1, "ci_high": 0.5,
               "ci_includes_zero": False, "delta": 0.3}
result["promotion_with_nan_protected_family"] = decide_promotion(
    EvidenceBundle(**base_bundle, clustered_uncertainty=positive_ci), PromotionConfig())["decision"]
print(json.dumps(result, indent=2, allow_nan=False))
