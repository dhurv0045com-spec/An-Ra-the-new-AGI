"""B2.2 C2/C3 synthetic tests: loop traces with fake callbacks and checkpoint
publication safety — no model instantiation and no optimizer updates."""
import json
import os
import tempfile
import unittest

import torch

from bramastra_lab.research.runtime import checkpoint as ckpt
from bramastra_lab.research.runtime.checkpoint import CheckpointError


class FakeTrainer:
    """Records the exact event sequence; no tensors, no optimizer."""

    def __init__(self, grad_accum_steps=1):
        self.grad_accum_steps = grad_accum_steps
        self.counters = type("C", (), {"optimizer_updates": 0, "attempted_batches": 0,
                                       "microbatches": 0, "replay_entries_consumed": 0})()
        self.events = []

    def accumulate(self, tag, pair_rows=None):
        self.events.append(("accumulate", tag))
        self.counters.microbatches += 1

    def finalize_update(self):
        self.events.append(("update",))
        self.counters.optimizer_updates += 1


class FakeSampler:
    def __init__(self, tags):
        self.tags = list(tags)
        self.index = 0

    def take_batch(self):
        tag = self.tags[self.index % len(self.tags)]
        self.index += 1
        return tag


def run_loop(*, grad_accum_steps, target_updates, replay_schedule, replay_rows,
             sampler_tags, on_empty="refuse", start_update=0):
    """A faithful trace harness for the shared loop's scheduling decisions.

    Mirrors commands._training_loop's decision structure (replay slot per
    update boundary from the persisted scheduler, empty-slot policy, update
    accounting) without any model or data.
    """
    trainer = FakeTrainer(grad_accum_steps)
    trainer.counters.optimizer_updates = start_update
    events = []
    empty_skips = 0
    updates_done = 0
    while trainer.counters.optimizer_updates < start_update + target_updates:
        use_replay = replay_schedule is not None and replay_schedule.slot_due()
        for _micro in range(trainer.grad_accum_steps):
            if use_replay:
                rows = replay_rows
                if not rows:
                    if on_empty == "refuse":
                        raise RuntimeError("empty replay attempt refused")
                    events.append(("replay_empty_skip",))
                    rows = None  # slot consumed; fall back to data
                if rows:
                    trainer.accumulate(("replay", rows))
                    continue
            trainer.accumulate(("data", next(iter(sampler_tags))))
        trainer.finalize_update()
        updates_done += 1
    return {"trainer": trainer, "events": trainer.events, "updates_done": updates_done,
            "empty_skips": empty_skips}


class LoopTraceTests(unittest.TestCase):
    def test_p03_split_run_matches_uninterrupted(self) -> None:
        """p=.3 is non-divisible for small counts: the persisted credit must
        make a split invocation choose the same replay/data events."""
        from bramastra_lab.research.experience.replay import ReplaySchedule

        uninterrupted = ReplaySchedule(0.3)
        pattern = [uninterrupted.slot_due() for _ in range(7)]

        first = ReplaySchedule(0.3)
        head = [first.slot_due() for _ in range(3)]
        resumed = ReplaySchedule.from_state(first.state())
        tail = [resumed.slot_due() for _ in range(4)]
        self.assertEqual(pattern, head + tail)

    def test_split_events_identical_at_non_divisible_boundary(self) -> None:
        from bramastra_lab.research.experience.replay import ReplaySchedule

        def trace(schedule, target, start=0):
            harness = run_loop(grad_accum_steps=1, target_updates=target,
                               replay_schedule=schedule,
                               replay_rows=["ep1"], sampler_tags=["d"])
            return tuple(tuple(event) for event in harness["events"])

        schedule_a = ReplaySchedule(0.3)
        full = trace(schedule_a, 5)
        schedule_b = ReplaySchedule(0.3)
        head = trace(schedule_b, 2)
        resumed = ReplaySchedule.from_state(schedule_b.state())
        tail = tuple(tuple(event) for event in
                     run_loop(grad_accum_steps=1, target_updates=3,
                              replay_schedule=resumed, replay_rows=["ep1"],
                              sampler_tags=["d"], start_update=2)["events"])
        self.assertEqual(full, head + tail)

    def test_empty_replay_produces_no_fictional_step(self) -> None:
        from bramastra_lab.research.experience.replay import ReplaySchedule

        schedule = ReplaySchedule(1.0)  # every update is a replay slot
        with self.assertRaises(RuntimeError):
            run_loop(grad_accum_steps=1, target_updates=1, replay_schedule=schedule,
                     replay_rows=[], sampler_tags=["d"], on_empty="refuse")
        # Configured fallback: the slot is consumed, a data batch is drawn,
        # and the update still happens through data (never a fake replay).
        schedule = ReplaySchedule(1.0)
        harness = run_loop(grad_accum_steps=1, target_updates=1,
                           replay_schedule=schedule, replay_rows=[],
                           sampler_tags=["d"], on_empty="skip")
        self.assertEqual(harness["trainer"].counters.optimizer_updates, 1)
        self.assertIn(("accumulate", ("data", "d")), harness["events"])

    def test_configured_accumulation_groups_microbatches(self) -> None:
        from bramastra_lab.research.experience.replay import ReplaySchedule

        schedule = ReplaySchedule(1.0)
        harness = run_loop(grad_accum_steps=3, target_updates=2,
                           replay_schedule=schedule, replay_rows=["r"],
                           sampler_tags=["d1", "d2", "d3"])
        trainer = harness["trainer"]
        self.assertEqual(trainer.counters.optimizer_updates, 2)
        self.assertEqual(trainer.counters.microbatches, 6)
        # Every update consumed exactly grad_accum_steps micro batches.
        updates = sum(1 for event in trainer.events if event[0] == "update")
        accumulates = sum(1 for event in trainer.events if event[0] == "accumulate")
        self.assertEqual(accumulates, updates * 3)


class PublicationSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = os.path.join(self.tmp.name, "run")
        os.makedirs(self.run_dir)
        self.token = ckpt.acquire_writer_fence(self.run_dir)
        self.addCleanup(ckpt.release_writer_fence, self.run_dir, self.token)

    def _payload(self, update):
        return {"counters": {"optimizer_updates": update},
                "tensor": torch.arange(3, dtype=torch.float32) + update}

    def _save(self, update, *, token=None, expected_parent=None):
        return ckpt.save_checkpoint(
            self.run_dir, self._payload(update), run_id="r1", update_index=update,
            config_identity="cfg", tokenizer_identity="tok", data_identity="data-1",
            parent_checkpoint_id=None, writer_token=token or self.token,
            expected_parent=expected_parent)

    def test_stale_writer_cannot_publish(self) -> None:
        self._save(1)
        with self.assertRaises(CheckpointError) as caught:
            self._save(2, token="not-the-lease")
        self.assertIn("writer lease", str(caught.exception))

    def test_stale_parent_cannot_publish(self) -> None:
        first = self._save(1)
        self._save(2, expected_parent=first.checkpoint_id)
        # A stale writer that saw an older parent is refused even with the
        # right lease: it cannot infer a new parent from whatever LATEST says.
        with self.assertRaises(CheckpointError) as caught:
            self._save(3, expected_parent=first.checkpoint_id)
        self.assertIn("expected parent", str(caught.exception))
        # The correct current parent publishes cleanly.
        latest = ckpt.read_pointer(self.run_dir, ckpt.LATEST_POINTER)
        self._save(3, expected_parent=latest["checkpoint_id"])

    def test_redirected_latest_pointer_rejected_on_load(self) -> None:
        self._save(1)
        pointer_path = os.path.join(self.run_dir, "checkpoints", ckpt.LATEST_POINTER)
        with open(pointer_path, "r", encoding="utf-8") as handle:
            pointer = json.load(handle)
        pointer["directory"] = ".." + os.path.sep + "evil"
        with open(pointer_path, "w", encoding="utf-8") as handle:
            json.dump(pointer, handle)
        with self.assertRaises(CheckpointError):
            ckpt.load_checkpoint(self.run_dir)

    def test_mismatched_latest_pointer_rejected_on_load(self) -> None:
        self._save(1)
        pointer_path = os.path.join(self.run_dir, "checkpoints", ckpt.LATEST_POINTER)
        with open(pointer_path, "r", encoding="utf-8") as handle:
            pointer = json.load(handle)
        pointer["checkpoint_id"] = "0" * 64  # redirected identity
        with open(pointer_path, "w", encoding="utf-8") as handle:
            json.dump(pointer, handle)
        with self.assertRaises(CheckpointError) as caught:
            ckpt.load_checkpoint(self.run_dir)
        self.assertIn("redirected", str(caught.exception))

    def test_data_identity_mismatch_rejected_on_load(self) -> None:
        self._save(1)
        with self.assertRaises(CheckpointError) as caught:
            ckpt.load_checkpoint(self.run_dir, expect_data_identity="other-data")
        self.assertIn("data identity", str(caught.exception))

    def test_no_unrestricted_pickle_fallback(self) -> None:
        """Incompatible payloads fail with a clear error, never a fallback."""
        self._save(1)
        latest = ckpt.read_pointer(self.run_dir, ckpt.LATEST_POINTER)
        payload_path = os.path.join(self.run_dir, "checkpoints",
                                    latest["directory"], "payload.pt")
        torch.save({"evil": object()}, payload_path)
        with self.assertRaises(Exception) as caught:
            ckpt.load_checkpoint(self.run_dir)
        self.assertNotIn("weights_only=False", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
