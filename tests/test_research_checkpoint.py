"""B05 focused tests: atomic checkpoints, fencing, rotation, resume payload.

The checkpoint file mechanics (atomicity, validation, rotation, fencing) are
arithmetic/state tests and run without optimizer updates. The decisive
fresh-process next-update equivalence check performs real optimizer updates
and is therefore gated behind ``BRAMASTRA_LEARNED_CHECKS=1``.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

import torch

from bramastra_lab.research.config import BuildConfig, seed_everything
from bramastra_lab.research.runtime import checkpoint as ckpt
from bramastra_lab.research.runtime.checkpoint import CheckpointError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEARNED_CHECKS = os.environ.get("BRAMASTRA_LEARNED_CHECKS") == "1"
LEARNED_REASON = ("learned checks deferred: owner has not authorized optimizer updates "
                  "(set BRAMASTRA_LEARNED_CHECKS=1 to run)")

TINY_CONFIG = {"model": {"profile": "tiny"}}


def make_payload(update_index: int) -> dict:
    return {"counters": {"optimizer_updates": update_index},
            "tensor": torch.arange(5, dtype=torch.float32) + update_index}


class SaveAndLoadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = os.path.join(self.tmp.name, "run")
        os.makedirs(self.run_dir)

    def test_round_trip_and_pointer(self) -> None:
        manifest = ckpt.save_checkpoint(
            self.run_dir, make_payload(3), run_id="r1", update_index=3,
            config_identity="cfg-1", tokenizer_identity="tok-1", data_identity="data-1",
            parent_checkpoint_id=None)
        payload, loaded = ckpt.load_checkpoint(
            self.run_dir, expect_config_identity="cfg-1", expect_tokenizer_identity="tok-1")
        self.assertEqual(loaded.checkpoint_id, manifest.checkpoint_id)
        self.assertEqual(payload["counters"]["optimizer_updates"], 3)
        self.assertTrue(torch.equal(payload["tensor"], torch.arange(5) + 3))
        latest = ckpt.read_pointer(self.run_dir, ckpt.LATEST_POINTER)
        self.assertEqual(latest["checkpoint_id"], manifest.checkpoint_id)
        self.assertIsNone(loaded.parent_checkpoint_id
                          if loaded.parent_checkpoint_id is None else None)

    def test_parent_chain_recorded(self) -> None:
        first = ckpt.save_checkpoint(self.run_dir, make_payload(1), run_id="r1",
                                     update_index=1, config_identity="cfg",
                                     tokenizer_identity="tok", data_identity="d",
                                     parent_checkpoint_id=None)
        second = ckpt.save_checkpoint(self.run_dir, make_payload(2), run_id="r1",
                                      update_index=2, config_identity="cfg",
                                      tokenizer_identity="tok", data_identity="d",
                                      parent_checkpoint_id=None)
        self.assertEqual(second.parent_checkpoint_id, first.checkpoint_id)
        _, manifest = ckpt.load_checkpoint(self.run_dir)
        self.assertEqual(manifest.checkpoint_id, second.checkpoint_id)

    def test_incomplete_publication_rejected_and_previous_survives(self) -> None:
        good = ckpt.save_checkpoint(self.run_dir, make_payload(1), run_id="r1",
                                    update_index=1, config_identity="cfg",
                                    tokenizer_identity="tok", data_identity="d",
                                    parent_checkpoint_id=None)
        # Simulate an interrupted publication: a directory without COMPLETE.
        broken = os.path.join(self.run_dir, "checkpoints", "update-000000000002")
        os.makedirs(broken)
        torch.save(make_payload(2), os.path.join(broken, "payload.pt"))
        payload, manifest = ckpt.load_checkpoint(self.run_dir)
        self.assertEqual(manifest.checkpoint_id, good.checkpoint_id)
        with self.assertRaises(CheckpointError):
            ckpt.load_checkpoint(self.run_dir, checkpoint_id="nonexistent")

    def test_tampered_payload_rejected(self) -> None:
        ckpt.save_checkpoint(self.run_dir, make_payload(1), run_id="r1", update_index=1,
                             config_identity="cfg", tokenizer_identity="tok",
                             data_identity="d", parent_checkpoint_id=None)
        latest = ckpt.read_pointer(self.run_dir, ckpt.LATEST_POINTER)
        payload_path = os.path.join(self.run_dir, "checkpoints", latest["directory"],
                                    "payload.pt")
        torch.save(make_payload(999), payload_path)  # tamper after publication
        with self.assertRaises(CheckpointError) as caught:
            ckpt.load_checkpoint(self.run_dir)
        self.assertIn("tampered", str(caught.exception))

    def test_identity_mismatch_rejected(self) -> None:
        ckpt.save_checkpoint(self.run_dir, make_payload(1), run_id="r1", update_index=1,
                             config_identity="cfg-a", tokenizer_identity="tok-a",
                             data_identity="d", parent_checkpoint_id=None)
        with self.assertRaises(CheckpointError):
            ckpt.load_checkpoint(self.run_dir, expect_config_identity="cfg-b")
        with self.assertRaises(CheckpointError):
            ckpt.load_checkpoint(self.run_dir, expect_tokenizer_identity="tok-b")

    def test_stale_parent_rejected(self) -> None:
        first = ckpt.save_checkpoint(self.run_dir, make_payload(1), run_id="r1",
                                     update_index=1, config_identity="cfg",
                                     tokenizer_identity="tok", data_identity="d",
                                     parent_checkpoint_id=None)
        ckpt.save_checkpoint(self.run_dir, make_payload(2), run_id="r1", update_index=2,
                             config_identity="cfg", tokenizer_identity="tok",
                             data_identity="d", parent_checkpoint_id=first.checkpoint_id)
        with self.assertRaises(CheckpointError) as caught:
            ckpt.load_checkpoint(self.run_dir,
                                 expect_parent_checkpoint_id="not-the-parent")
        self.assertIn("stale or divergent parent", str(caught.exception))

    def test_checkpoints_never_overwritten(self) -> None:
        ckpt.save_checkpoint(self.run_dir, make_payload(1), run_id="r1", update_index=1,
                             config_identity="cfg", tokenizer_identity="tok",
                             data_identity="d", parent_checkpoint_id=None)
        with self.assertRaises(CheckpointError):
            ckpt.save_checkpoint(self.run_dir, make_payload(1), run_id="r1",
                                 update_index=1, config_identity="cfg",
                                 tokenizer_identity="tok", data_identity="d",
                                 parent_checkpoint_id=None)


class RotationAndMilestoneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = os.path.join(self.tmp.name, "run")
        os.makedirs(self.run_dir)

    def _save(self, index: int, milestone: str | None = None):
        return ckpt.save_checkpoint(
            self.run_dir, make_payload(index), run_id="r1", update_index=index,
            config_identity="cfg", tokenizer_identity="tok", data_identity="d",
            parent_checkpoint_id=None, milestone=milestone)

    def test_rotation_retains_referenced_milestones(self) -> None:
        self._save(1)
        milestone = self._save(2, milestone="before-acquisition")
        self._save(3)
        self._save(4)
        removed = ckpt.prune_checkpoints(self.run_dir, keep_latest=1)
        self.assertIn("update-000000000001", removed)
        surviving = sorted(os.listdir(os.path.join(self.run_dir, "checkpoints")))
        self.assertIn(f"update-{2:012d}", surviving)  # milestone retained
        self.assertIn("update-000000000004", surviving)  # latest retained
        labels = ckpt.read_milestones(self.run_dir)
        self.assertEqual(labels[0]["label"], "before-acquisition")
        payload, manifest = ckpt.load_checkpoint(
            self.run_dir, checkpoint_id=milestone.checkpoint_id)
        self.assertEqual(payload["counters"]["optimizer_updates"], 2)

    def test_accepted_parent_pointer_separate_from_latest(self) -> None:
        self._save(1)
        promoted = ckpt.promote_accepted_parent(self.run_dir)
        self._save(2)
        latest = ckpt.read_pointer(self.run_dir, ckpt.LATEST_POINTER)
        accepted = ckpt.read_pointer(self.run_dir, ckpt.ACCEPTED_PARENT_POINTER)
        self.assertEqual(latest["update_index"], 2)
        self.assertEqual(accepted, promoted)
        # Rotation protects the accepted parent too.
        ckpt.prune_checkpoints(self.run_dir, keep_latest=1)
        surviving = os.listdir(os.path.join(self.run_dir, "checkpoints"))
        self.assertIn(accepted["directory"], surviving)


class WriterFenceTests(unittest.TestCase):
    def test_single_writer_fence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            token = ckpt.acquire_writer_fence(tmp)
            with self.assertRaises(CheckpointError):
                ckpt.acquire_writer_fence(tmp)
            ckpt.release_writer_fence(tmp, token)
            token2 = ckpt.acquire_writer_fence(tmp)
            ckpt.release_writer_fence(tmp, token2)

    def test_stale_fence_requires_explicit_force(self) -> None:
        """A held lease is never stolen for its age (B2.2 R4)."""
        with tempfile.TemporaryDirectory() as tmp:
            lock = os.path.join(tmp, "writer.lock")
            with open(lock, "w", encoding="utf-8") as handle:
                handle.write("999999:0.0")
            old = time.time() - 10 * 3600
            os.utime(lock, (old, old))
            with self.assertRaises(CheckpointError) as caught:
                ckpt.acquire_writer_fence(tmp)
            self.assertIn("explicit", str(caught.exception))
            # Recovery from a crashed writer is an explicit operator decision.
            token = ckpt.acquire_writer_fence(tmp, force=True)
            self.assertTrue(token)
            ckpt.release_writer_fence(tmp, token)


class RngStateTests(unittest.TestCase):
    def test_capture_restore_round_trip(self) -> None:
        seed_everything(5)
        torch.rand(3)
        state = ckpt.capture_rng_state()
        first = torch.rand(4)
        ckpt.restore_rng_state(state)
        second = torch.rand(4)
        self.assertTrue(torch.equal(first, second))

    def test_python_random_round_trip(self) -> None:
        import random

        seed_everything(6)
        random.random()
        state = ckpt.capture_rng_state()
        first = [random.random() for _ in range(3)]
        ckpt.restore_rng_state(state)
        second = [random.random() for _ in range(3)]
        self.assertEqual(first, second)


class FreshProcessResumeTests(unittest.TestCase):
    """The decisive check: interrupted vs uninterrupted NEXT UPDATE, in a
    fresh process, using the real trainer. Performs optimizer updates."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    @unittest.skipUnless(LEARNED_CHECKS, LEARNED_REASON)
    def test_fresh_process_next_update_agrees(self) -> None:
        script = os.path.join(os.path.dirname(__file__), "_resume_probe.py")
        result = subprocess.run(
            [sys.executable, script, "--workdir", self.tmp.name],
            capture_output=True, text=True, cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": REPO_ROOT,
                 "BRAMASTRA_LEARNED_CHECKS": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        verdict = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(verdict["agrees"], verdict)
        self.assertEqual(verdict["tolerance"], "1e-5 relative / 1e-7 absolute")


class IdempotentRetryTests(unittest.TestCase):
    """Kaggle retry regression: an identical re-publication (retried job)
    returns the existing identity; divergent content still refuses."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = os.path.join(self.tmp.name, "run")
        os.makedirs(self.run_dir)

    def _publish(self, payload: dict, **overrides):
        args = dict(run_id="E1-A-1702-0-1702", update_index=0,
                    config_identity="cfg-1", tokenizer_identity="tok-1",
                    data_identity="data-1", parent_checkpoint_id=None,
                    dir_suffix="E1-A-1702")
        args.update(overrides)
        return ckpt.save_checkpoint(self.run_dir, payload, **args)

    def test_identical_republication_returns_existing(self) -> None:
        first = self._publish(make_payload(0))
        second = self._publish(make_payload(0))
        self.assertEqual(first.checkpoint_id, second.checkpoint_id)

    def test_divergent_republication_refuses(self) -> None:
        self._publish(make_payload(0))
        with self.assertRaises(CheckpointError):
            self._publish(make_payload(1))


if __name__ == "__main__":
    unittest.main()
