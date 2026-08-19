from __future__ import annotations

from pathlib import Path

from training.train_xla import _runtime_topology


ROOT = Path(__file__).resolve().parents[1]


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


def test_parent_launcher_does_not_initialize_worker_xla_stack() -> None:
    """PJRT must be initialized only after the launcher creates workers."""
    source = (ROOT / "training" / "train_xla.py").read_text(encoding="utf-8")
    run_body = source.split("def run(args: argparse.Namespace) -> None:", 1)[1]
    launch_start = run_body.index("if callable(launch):")
    parent_setup = run_body[:launch_start]

    assert "_require_xla()" not in parent_setup
    assert "import torch_xla.runtime" not in parent_setup
    assert "import torch_xla.core.xla_model" not in parent_setup


def test_tpu_stop_flag_uses_supported_sum_collective() -> None:
    source = (ROOT / "training" / "train_xla.py").read_text(encoding="utf-8")
    assert "xm.all_reduce(xm.REDUCE_SUM, stop)" in source
    assert "xm.all_reduce(xm.REDUCE_MAX, stop)" not in source


def test_tpu_loader_does_not_unroll_gradient_accumulation_into_one_hlo_graph() -> None:
    source = (ROOT / "training" / "train_xla.py").read_text(encoding="utf-8")
    assert "batches_per_execution=1" in source
    assert 'batches_per_execution=int(config["grad_accum_steps"])' not in source
