"""B03 focused tests: manifest loading, split integrity and group sampling."""
import json
import os
import tempfile
import unittest

from bramastra_lab.research.data.manifest import (
    DATA_AVAILABLE,
    DATA_NOT_READY,
    DatasetError,
    load_dataset,
)
from bramastra_lab.research.data.sampler import GroupSampler, SamplerState

MANIFEST = {
    "schema": "bramastra-dataset-manifest/v1",
    "name": "tiny-fixture",
    "license": "CC0-operator-declared",
    "provenance": "operator-local-fixture",
    "entries": [],
}


def write_fixture(directory: str, files: dict[str, str], manifest: dict | None = None):
    for name, text in files.items():
        if not text.endswith("\n"):
            text += "\n"
        with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
            handle.write(text)
    manifest = manifest or MANIFEST
    path = os.path.join(directory, "manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return path


TRAIN_LINES = [
    {"example_id": "t1", "prompt_events": [["goal", {"question": "2+2?"}]], "answer": "4"},
    {"example_id": "t2", "prompt_events": [["goal", {"question": "3+3?"}]], "answer": "6"},
    {"example_id": "p1", "prompt_events": [["goal", {"question": "5+1?"}]], "answer": "6",
     "group": "g1"},
    {"example_id": "p2", "prompt_events": [["goal", {"question": "1+5?"}]], "answer": "6",
     "group": "g1"},
]

DEV_LINES = [
    {"example_id": "d1", "prompt_events": [["goal", {"question": "4+4?"}]], "answer": "8"},
]


class ManifestTests(unittest.TestCase):
    def make_manifest(self, directory: str, *, extra_entries: list | None = None,
                      overrides: dict | None = None) -> str:
        manifest = json.loads(json.dumps(MANIFEST))
        manifest["entries"] = [
            {"path": "train.jsonl", "split": "training", "kind": "trajectory"},
            {"path": "dev.jsonl", "split": "development", "kind": "trajectory"},
        ] + (extra_entries or [])
        if overrides:
            manifest.update(overrides)
        return write_fixture(directory, {
            "train.jsonl": "\n".join(json.dumps(line) for line in TRAIN_LINES),
            "dev.jsonl": "\n".join(json.dumps(line) for line in DEV_LINES),
        }, manifest)

    def test_load_reports_availability_identity_and_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_manifest(tmp)
            handle = load_dataset(path)
            self.assertEqual(handle.availability, DATA_AVAILABLE)
            self.assertEqual(handle.split_inventory["training"]["examples"], 4)
            self.assertEqual(handle.split_inventory["development"]["examples"], 1)
            self.assertEqual(handle.group_count, 1)
            self.assertTrue(handle.identity)
            # Changing input bytes invalidates identity.
            with open(os.path.join(tmp, "dev.jsonl"), "a", encoding="utf-8") as handle_file:
                handle_file.write(json.dumps(
                    {"example_id": "d2", "prompt_events": [["goal", {"q": 1}]],
                     "answer": "z"}) + "\n")
            reloaded = load_dataset(path)
            self.assertNotEqual(handle.identity, reloaded.identity)

    def test_missing_file_is_data_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_manifest(tmp)
            os.remove(os.path.join(tmp, "train.jsonl"))
            with self.assertRaises(DatasetError) as caught:
                load_dataset(path)
            self.assertEqual(caught.exception.status, DATA_NOT_READY)

    def test_empty_manifest_is_data_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.loads(json.dumps(MANIFEST))
            manifest["entries"] = []
            path = write_fixture(tmp, {"train.jsonl": ""}, manifest)
            with self.assertRaises(DatasetError) as caught:
                load_dataset(path)
            self.assertEqual(caught.exception.status, DATA_NOT_READY)

    def test_missing_manifest_is_data_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DatasetError) as caught:
                load_dataset(os.path.join(tmp, "absent.json"))
            self.assertEqual(caught.exception.status, DATA_NOT_READY)

    def test_renamed_duplicate_cannot_cross_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.loads(json.dumps(MANIFEST))
            manifest["entries"] = [
                {"path": "train.jsonl", "split": "training", "kind": "trajectory"},
                {"path": "dev.jsonl", "split": "development", "kind": "trajectory"},
            ]
            path = write_fixture(tmp, {
                "train.jsonl": json.dumps(
                    {"example_id": "a", "prompt_events": [["goal", {"q": 1}]], "answer": "x"}),
                # Different example_id and rendering keys; same semantics.
                "dev.jsonl": json.dumps(
                    {"example_id": "b", "prompt_events": [["goal", {"q": 1}]], "answer": "x"}),
            }, manifest)
            with self.assertRaises(DatasetError) as caught:
                load_dataset(path)
            self.assertIn("duplicate semantics across splits", str(caught.exception))

    def test_pair_group_cannot_span_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.loads(json.dumps(MANIFEST))
            manifest["entries"] = [
                {"path": "train.jsonl", "split": "training", "kind": "trajectory"},
                {"path": "dev.jsonl", "split": "development", "kind": "trajectory"},
            ]
            path = write_fixture(tmp, {
                "train.jsonl": json.dumps(
                    {"example_id": "a", "prompt_events": [["goal", {"q": 1}]], "answer": "x",
                     "group": "g"}),
                "dev.jsonl": json.dumps(
                    {"example_id": "b", "prompt_events": [["goal", {"q": 2}]], "answer": "y",
                     "group": "g"}),
            }, manifest)
            with self.assertRaises(DatasetError) as caught:
                load_dataset(path)
            self.assertIn("spans multiple splits", str(caught.exception))

    def test_url_paths_and_unknown_fields_reject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.loads(json.dumps(MANIFEST))
            manifest["entries"] = [{"path": "https://example.com/data.jsonl",
                                    "split": "training", "kind": "trajectory"}]
            path = write_fixture(tmp, {}, manifest)
            with self.assertRaises(DatasetError) as caught:
                load_dataset(path)
            self.assertIn("downloads are not authorized", str(caught.exception))

        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.loads(json.dumps(MANIFEST))
            manifest["entries"] = [{"path": "train.jsonl", "split": "training",
                                    "kind": "trajectory", "surprise": True}]
            path = write_fixture(tmp, {"train.jsonl": ""}, manifest)
            with self.assertRaises(DatasetError):
                load_dataset(path)

    def test_inference_metadata_hides_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            handle = load_dataset(self.make_manifest(tmp))
            allowed_keys = {"example_id", "kind", "split", "semantic_identity",
                            "group_id", "source"}
            for example in handle.examples:
                metadata = example.inference_metadata()
                self.assertEqual(set(metadata), allowed_keys)
                # No value outside content-hash fields may equal the answer.
                for key in ("example_id", "kind", "split", "source"):
                    self.assertNotEqual(metadata[key], example.answer())


class GroupSamplerTests(unittest.TestCase):
    @staticmethod
    def build_examples() -> list:
        from dataclasses import dataclass, field

        @dataclass
        class FakeExample:
            example_id: str
            group_id: str | None
            kind: str = "trajectory"
            split: str = "training"

        return [FakeExample(example_id=f"e{index}", group_id=group)
                for index, (group, _question, _answer) in enumerate([
                    ("g1", "5+1?", "6"), ("g1", "1+5?", "6"),
                    (None, "2+2?", "4"), (None, "3+3?", "6"),
                    ("g2", "7+1?", "8"), ("g2", "1+7?", "8")])]

    def test_batches_keep_pairs_whole(self) -> None:
        examples = self.build_examples()
        groups = [(examples[0], examples[1]), (examples[4], examples[5]),
                  (examples[2],), (examples[3],)]
        sampler = GroupSampler(groups, batch_size=2, seed=3)
        batches = [[example.example_id for example in sampler.take_batch()]
                   for _ in range(4)]
        for batch in batches:
            for group_id in ("g1", "g2"):
                members_in_batch = sum(1 for example_id in batch if self._group_of(
                    group_id, examples) and example_id in self._ids_of_group(group_id, examples))
                if members_in_batch:
                    self.assertEqual(members_in_batch,
                                     len(self._ids_of_group(group_id, examples)))
        self.assertEqual(sampler.examples_consumed, sum(len(batch) for batch in batches))

    @staticmethod
    def _ids_of_group(group_id: str, examples: list) -> set:
        return {example.example_id for example in examples if example.group_id == group_id}

    def _group_of(self, group_id: str, examples: list) -> bool:
        return any(example.group_id == group_id for example in examples)

    def test_resume_reproduces_the_next_group(self) -> None:
        examples = self.build_examples()
        groups = [(examples[0], examples[1]), (examples[2],), (examples[3],),
                  (examples[4], examples[5])]
        reference = GroupSampler(groups, batch_size=2, seed=9)
        reference.take_batch()
        reference.take_batch()
        expected_rest = [reference.take_batch() for _ in range(4)]

        resumed = GroupSampler(groups, batch_size=2, seed=9)
        resumed.take_batch()
        resumed.take_batch()
        resumed.restore(resumed.state())
        actual_rest = [resumed.take_batch() for _ in range(4)]
        self.assertEqual(
            [[example.example_id for example in batch] for batch in expected_rest],
            [[example.example_id for example in batch] for batch in actual_rest])

    def test_resume_from_saved_state_reproduces_stream(self) -> None:
        examples = self.build_examples()
        groups = [(examples[0], examples[1]), (examples[2],), (examples[3],),
                  (examples[4], examples[5])]
        sampler = GroupSampler(groups, batch_size=2, seed=11)
        sampler.take_batch()
        saved = sampler.state()
        saved_dict = saved.to_dict()
        restored = GroupSampler(groups, batch_size=2, seed=11)
        restored.restore(SamplerState.from_dict(saved_dict))
        self.assertEqual(
            [example.example_id for example in sampler.take_batch()],
            [example.example_id for example in restored.take_batch()])

    def test_unpaired_control_uses_same_examples_without_groups(self) -> None:
        examples = self.build_examples()
        groups = [(examples[0], examples[1]), (examples[2],), (examples[3],),
                  (examples[4], examples[5])]
        paired = GroupSampler(groups, batch_size=6, seed=5)
        control = GroupSampler(groups, batch_size=6, seed=5, mode="unpaired_control")
        paired_batch = paired.take_batch()
        control_batch = control.take_batch()
        self.assertEqual(
            sorted(example.example_id for example in paired_batch),
            sorted(example.example_id for example in control_batch))
        # Control order is not required to differ for every seed, but the
        # grouping guarantee is gone: members of g1 need not be adjacent.
        control_ids = [example.example_id for example in control_batch]
        self.assertEqual(len(control_ids), 6)

    def test_epoch_rollover_is_deterministic_and_counted(self) -> None:
        examples = self.build_examples()
        groups = [(examples[2],), (examples[3],)]
        sampler = GroupSampler(groups, batch_size=2, seed=1)
        first_epoch = sampler.epoch
        sampler.take_batch()
        sampler.take_batch()  # exactly two units -> rolls into the next epoch
        self.assertGreater(sampler.epoch, first_epoch)
        self.assertEqual(sampler.groups_consumed, 4)
        self.assertEqual(sampler.examples_consumed, 4)


if __name__ == "__main__":
    unittest.main()
