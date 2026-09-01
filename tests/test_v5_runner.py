from __future__ import annotations

import json
import unittest

from v5_training.runner import RunController, RunStatus, RunnerState


class V5RunnerTests(unittest.TestCase):
    def test_failure_preserves_last_durable_parent_and_recovery_reuses_it(self) -> None:
        controller = RunController(target_update=2)
        self.assertEqual(controller.start().status, RunStatus.RUNNING)
        controller.complete_update(update=1)
        controller.begin_checkpoint(update=1)
        failed = controller.fail(code="writer_lost")
        self.assertEqual(
            (failed.completed_update, failed.committed_update, failed.last_checkpoint_sha256),
            (1, 0, None),
        )
        recovered = controller.recover()
        self.assertEqual(recovered.status, RunStatus.RUNNING)
        self.assertEqual(recovered.completed_update, recovered.committed_update)
        controller.complete_update(update=1)
        controller.begin_checkpoint(update=1)
        first = controller.commit_checkpoint(checkpoint_sha256="a" * 64)
        self.assertEqual(
            (first.completed_update, first.committed_update, first.last_checkpoint_sha256),
            (1, 1, "a" * 64),
        )
        controller.complete_update(update=2)
        controller.begin_checkpoint(update=2)
        failed = controller.fail(code="upload_timeout")
        self.assertEqual(
            (failed.completed_update, failed.committed_update, failed.last_checkpoint_sha256),
            (2, 1, "a" * 64),
        )
        recovered = controller.recover()
        self.assertEqual((recovered.completed_update, recovered.committed_update), (1, 1))
        controller.complete_update(update=2)
        controller.begin_checkpoint(update=2)
        controller.commit_checkpoint(checkpoint_sha256="b" * 64)
        terminal_running_state = controller.state
        with self.assertRaises(ValueError):
            controller.begin_checkpoint(update=3)
        self.assertEqual(controller.state, terminal_running_state)
        self.assertEqual(controller.complete().status, RunStatus.COMPLETED)

    def test_pending_checkpoint_cannot_be_skipped_or_completed(self) -> None:
        controller = RunController(target_update=2)
        controller.start()
        with self.assertRaises(ValueError):
            controller.complete()
        with self.assertRaises(ValueError):
            controller.begin_checkpoint(update=2)
        controller.complete_update(update=1)
        controller.begin_checkpoint(update=1)
        with self.assertRaises(ValueError):
            controller.commit_checkpoint(checkpoint_sha256="bad")
        with self.assertRaises(ValueError):
            controller.begin_checkpoint(update=1)

    def test_checkpoint_can_follow_a_seventy_six_update_gap(self) -> None:
        controller = RunController(target_update=77)
        controller.start()
        for update in range(1, 77):
            controller.complete_update(update=update)
        self.assertEqual((controller.state.completed_update, controller.state.committed_update), (76, 0))
        with self.assertRaises(ValueError):
            controller.begin_checkpoint(update=75)
        pending = controller.begin_checkpoint(update=76)
        self.assertEqual(pending.pending_update, 76)
        committed = controller.commit_checkpoint(checkpoint_sha256="c" * 64)
        self.assertEqual((committed.completed_update, committed.committed_update), (76, 76))

    def test_recovery_rolls_completed_work_back_to_durable_commit(self) -> None:
        controller = RunController(target_update=77)
        controller.start()
        controller.complete_update(update=1)
        controller.begin_checkpoint(update=1)
        controller.commit_checkpoint(checkpoint_sha256="d" * 64)
        for update in range(2, 78):
            controller.complete_update(update=update)
        self.assertEqual((controller.state.completed_update, controller.state.committed_update), (77, 1))
        controller.fail(code="worker_lost")
        recovered = controller.recover()
        self.assertEqual((recovered.completed_update, recovered.committed_update), (1, 1))

    def test_runner_state_round_trips_canonically_and_rejects_bad_schema(self) -> None:
        state = RunnerState.initial(target_update=3)
        encoded = json.dumps(state.canonical(), sort_keys=True)
        self.assertEqual(RunnerState.from_dict(json.loads(encoded)), state)
        with self.assertRaises(ValueError):
            RunnerState.from_dict({**state.canonical(), "extra": True})


if __name__ == "__main__":
    unittest.main()
