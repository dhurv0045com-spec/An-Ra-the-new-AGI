from __future__ import annotations

from training.train_xla import _runtime_topology


class _PrelaunchEightCoreRuntime:
    def world_size(self) -> int:
        return 1

    def global_device_count(self) -> int:
        return 8


class _SingleCoreRuntime:
    def world_size(self) -> int:
        return 1

    def global_device_count(self) -> int:
        return 1


def test_prelaunch_process_is_not_mistaken_for_tpu_slice_size() -> None:
    assert _runtime_topology(_PrelaunchEightCoreRuntime()) == (1, 8)


def test_single_device_runtime_remains_single_worker() -> None:
    assert _runtime_topology(_SingleCoreRuntime()) == (1, 1)
