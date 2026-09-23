"""CPU-only contract tests for the Kaggle TPU adapter and replica math.

No torch_xla runtime or optimizer update is used here. A tiny replica double
checks the scalar normalization contract; actual TPU fit and execution still
require the Kaggle owner run.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from bramastra_lab.research.experience.supervision import SupervisionWindow
from bramastra_lab.research.learning.router import route_window
from bramastra_lab.research.runtime.tpu import (
    TPU_RUNTIME_NOT_SELECTED,
    XLAReplicaBackend,
    broadcast_replica_parameters,
    inspect_tpu_runtime,
    launch_tpu_workers,
    make_xla_data_loader,
    replica_training_sampler,
    state_dict_sha256,
    verify_replica_initialization,
    write_replica_initialization_receipt,
)


class _ReplicaDouble:
    world_size = 2

    def __init__(self, *, forced_counts: dict[str, int] | None = None) -> None:
        self.forced_counts = forced_counts or {}
        self.local_counts: dict[str, int] = {}

    def reduce_counts(self, counts: dict[str, int]) -> dict[str, int]:
        self.local_counts = dict(counts)
        return {term: self.forced_counts.get(term, count * self.world_size)
                for term, count in counts.items()}

    def materialize_counts(self, counts: dict[str, int]) -> dict[str, int]:
        return {term: int(value) for term, value in counts.items()}

    def reduce_metric_sum(self, local_sum: torch.Tensor) -> torch.Tensor:
        return local_sum * self.world_size

    def reduce_gradients(self, optimizer) -> None:
        raise AssertionError("CPU contract tests must not reduce or step gradients")

    def mark_step(self) -> None:
        raise AssertionError("CPU contract tests must not flush an XLA step")

    def optimizer_step(self, optimizer) -> None:
        raise AssertionError("CPU contract tests must never call optimizer.step")


class _NoUpdateReplicaDouble(_ReplicaDouble):
    """Exercise trainer finalization without reducing or updating weights."""

    def __init__(self, *, forced_counts: dict[str, int] | None = None) -> None:
        super().__init__(forced_counts=forced_counts)
        self.reduced_head_grads: dict[str, torch.Tensor | None] = {}
        self.step_requested = False
        self.mark_requested = False
        self.named_head_parameters: dict[str, torch.nn.Parameter] = {}

    def reduce_gradients(self, optimizer) -> None:
        # The test model's action/value head tensors are recorded by name via
        # the attributes attached by the test before finalize_update.
        for name, parameter in self.named_head_parameters.items():
            gradient = parameter.grad
            self.reduced_head_grads[name] = (
                None if gradient is None else gradient.detach().clone())

    def optimizer_step(self, optimizer) -> None:
        # Deliberately do not invoke optimizer.step(): local diagnostics may
        # exercise the boundary without committing parameter updates.
        self.step_requested = True

    def mark_step(self) -> None:
        self.mark_requested = True


class _StopBeforeCommit(RuntimeError):
    pass


class _NoCommitReplicaDouble(_ReplicaDouble):
    def __init__(self, *, forced_counts: dict[str, int] | None = None) -> None:
        super().__init__(forced_counts=forced_counts)
        self.reduce_called = False
        self.step_requested = False
        self.mark_requested = False

    def reduce_gradients(self, optimizer) -> None:
        self.reduce_called = True

    def mark_step(self) -> None:
        self.mark_requested = True

    def optimizer_step(self, optimizer) -> None:
        self.step_requested = True
        raise _StopBeforeCommit("test stopped immediately before optimizer commit")


class TPUContractTests(unittest.TestCase):
    def test_tpu_inspection_refuses_when_runtime_is_not_selected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            report = inspect_tpu_runtime()
        self.assertEqual(report["status"], TPU_RUNTIME_NOT_SELECTED)
        self.assertFalse(report["training_started"])

    def test_replica_backend_refuses_visible_but_unlaunched_tpu(self) -> None:
        with patch.dict(os.environ, {"PJRT_DEVICE": "TPU"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "only allowed inside"):
                XLAReplicaBackend.current("xla:0")

    def test_launcher_marks_workers_and_restores_parent_environment(self) -> None:
        xla = types.ModuleType("torch_xla")
        observed = []

        def fake_launch(worker, args=()):
            observed.append(os.environ.get("BRAMASTRA_XLA_REPLICA_LAUNCH"))
            worker(*args)

        xla.launch = fake_launch
        modules = {"torch_xla": xla}
        with patch.dict(sys.modules, modules), patch.dict(
                os.environ, {"PJRT_DEVICE": "TPU"}, clear=True):
            launch_tpu_workers(lambda: None)
            self.assertIsNone(os.environ.get("BRAMASTRA_XLA_REPLICA_LAUNCH"))
        self.assertEqual(observed, ["1"])

    def test_parameter_broadcast_uses_xla_master_and_flushes(self) -> None:
        xla = types.ModuleType("torch_xla")
        xla.__path__ = []
        core = types.ModuleType("torch_xla.core")
        core.__path__ = []
        xm = types.ModuleType("torch_xla.core.xla_model")
        calls = []
        xm.broadcast_master_param = lambda model: calls.append(("broadcast", model))
        xm.mark_step = lambda: calls.append(("mark_step", None))
        core.xla_model = xm
        modules = {"torch_xla": xla, "torch_xla.core": core,
                   "torch_xla.core.xla_model": xm}
        model = torch.nn.Linear(3, 2)
        with patch.dict(sys.modules, modules), patch.dict(
                os.environ,
                {"PJRT_DEVICE": "TPU",
                 "BRAMASTRA_XLA_REPLICA_LAUNCH": "1"}, clear=True):
            broadcast_replica_parameters(model)
        self.assertEqual(calls, [("broadcast", model), ("mark_step", None)])

    def test_parameter_broadcast_refuses_outside_replica_launch(self) -> None:
        with patch.dict(os.environ, {"PJRT_DEVICE": "TPU"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "only allowed inside"):
                broadcast_replica_parameters(torch.nn.Linear(2, 2))

    def test_state_fingerprint_covers_tensor_names_shapes_dtypes_and_values(self) -> None:
        state = {"weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                 "bias": torch.tensor([0.5], dtype=torch.float32)}
        same = {name: tensor.clone() for name, tensor in state.items()}
        changed = {name: tensor.clone() for name, tensor in state.items()}
        changed["weight"][0, 1] = 3.0
        changed_dtype = {"weight": state["weight"].to(torch.bfloat16),
                         "bias": state["bias"].to(torch.bfloat16)}
        self.assertEqual(state_dict_sha256(state), state_dict_sha256(same))
        self.assertNotEqual(state_dict_sha256(state), state_dict_sha256(changed))
        self.assertNotEqual(state_dict_sha256(state),
                            state_dict_sha256(changed_dtype))
        self.assertNotEqual(
            state_dict_sha256({"scalar": torch.tensor(1.0)}),
            state_dict_sha256({"scalar": torch.tensor(2.0)}))

    def test_replica_initialization_requires_exact_identical_receipts(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

        model = TinyModel()
        with tempfile.TemporaryDirectory() as directory:
            for rank in range(2):
                write_replica_initialization_receipt(
                    directory, rank=rank, world_size=2,
                    config_identity="cfg-sha256", model=model)
            report = verify_replica_initialization(
                directory, expected_replicas=2)
            self.assertEqual(report["status"], "REPLICAS_IDENTICAL")
            self.assertFalse(report["training_started"])
            self.assertEqual(report["parameter_count"], 6)
            with self.assertRaises(FileExistsError):
                write_replica_initialization_receipt(
                    directory, rank=0, world_size=2,
                    config_identity="cfg-sha256", model=model)

    def test_replica_verifier_rejects_boolean_rank_and_missing_directory(self) -> None:
        model = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            write_replica_initialization_receipt(
                directory, rank=0, world_size=1,
                config_identity="cfg-sha256", model=model)
            path = os.path.join(directory, "replica-000.json")
            with open(path, encoding="utf-8") as handle:
                receipt = __import__("json").load(handle)
            receipt["rank"] = True
            with open(path, "w", encoding="utf-8") as handle:
                __import__("json").dump(receipt, handle)
            with self.assertRaisesRegex(RuntimeError, "receipt mismatch"):
                verify_replica_initialization(directory, expected_replicas=1)
        with self.assertRaisesRegex(RuntimeError, "directory does not exist"):
            verify_replica_initialization(os.path.join(directory, "missing"))

    def test_replica_initialization_detects_divergent_model(self) -> None:
        model0 = torch.nn.Linear(2, 2)
        model1 = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            write_replica_initialization_receipt(
                directory, rank=0, world_size=2,
                config_identity="cfg-sha256", model=model0)
            write_replica_initialization_receipt(
                directory, rank=1, world_size=2,
                config_identity="cfg-sha256", model=model1)
            with self.assertRaisesRegex(RuntimeError, "do not share exact"):
                verify_replica_initialization(directory, expected_replicas=2)

    def test_training_sampler_splits_without_replica_overlap_or_padding(self) -> None:
        dataset = torch.utils.data.TensorDataset(torch.arange(19))
        shards = [replica_training_sampler(
            dataset, rank=rank, world_size=4, seed=93) for rank in range(4)]
        indices = [list(iter(sampler)) for sampler in shards]
        self.assertEqual([len(shard) for shard in indices], [4, 4, 4, 4])
        flattened = [index for shard in indices for index in shard]
        self.assertEqual(len(set(flattened)), len(flattened))
        self.assertEqual(len(flattened), 16)
        self.assertTrue(all(0 <= index < len(dataset) for index in flattened))
        for sampler in shards:
            sampler.set_epoch(1)
        next_indices = [list(iter(sampler)) for sampler in shards]
        self.assertEqual(len({index for shard in next_indices for index in shard}), 16)
        self.assertNotEqual(indices, next_indices)

    def test_training_sampler_rejects_invalid_rank_and_too_small_dataset(self) -> None:
        dataset = torch.utils.data.TensorDataset(torch.arange(2))
        with self.assertRaises(ValueError):
            replica_training_sampler(dataset, rank=2, world_size=2)
        with self.assertRaises(ValueError):
            replica_training_sampler(dataset, rank=0, world_size=3)

    def test_xla_loader_constructs_distinct_rank_shards_before_prefetch(self) -> None:
        dataset = torch.utils.data.TensorDataset(torch.arange(20))
        shards = []
        for rank in range(4):
            xla = types.ModuleType("torch_xla")
            xla.__path__ = []
            runtime = types.ModuleType("torch_xla.runtime")
            runtime.world_size = lambda: 4
            runtime.global_ordinal = lambda rank=rank: rank
            distributed = types.ModuleType("torch_xla.distributed")
            distributed.__path__ = []
            parallel_loader = types.ModuleType(
                "torch_xla.distributed.parallel_loader")

            def wrap(loader, device):
                return {"host_loader": loader, "device": device}

            parallel_loader.MpDeviceLoader = wrap
            distributed.parallel_loader = parallel_loader
            modules = {
                "torch_xla": xla,
                "torch_xla.runtime": runtime,
                "torch_xla.distributed": distributed,
                "torch_xla.distributed.parallel_loader": parallel_loader,
            }
            with patch.dict(sys.modules, modules), patch.dict(
                    os.environ, {"PJRT_DEVICE": "TPU"}):
                sampler, device_loader = make_xla_data_loader(
                    dataset, "xla:0", batch_size=2, seed=41)
                self.assertIs(device_loader["device"], "xla:0")
                sampler.set_epoch(0)
                shards.append([int(item) for (values,) in
                               device_loader["host_loader"]
                               for item in values.tolist()])
        flattened = [item for shard in shards for item in shard]
        self.assertEqual(len(flattened), 20)
        self.assertEqual(len(set(flattened)), 20)

    def test_replica_mean_produces_one_global_token_mean(self) -> None:
        window = SupervisionWindow(
            weights={"token": 1.0}, enabled_terms=frozenset({"token"}))
        window.add("token", 2)
        local_sum = torch.tensor(10.0, requires_grad=True)
        combined, report = route_window(
            window, {"token": local_sum},
            global_denominators={"token": torch.tensor(6.0)},
            replica_gradient_scale=3)
        self.assertAlmostEqual(float(combined.detach()), 5.0)
        self.assertEqual(
            report["applied"]["token"]["denominator"], "global_replica_sum")
        combined.backward()
        # The replica mean reduction will average R-scaled local gradients;
        # this local derivative is R / global_count = 3 / 6.
        self.assertAlmostEqual(float(local_sum.grad), 0.5)

    def test_replica_term_with_no_local_rows_keeps_zero_contribution(self) -> None:
        window = SupervisionWindow(
            weights={"token": 1.0, "action": 0.5},
            enabled_terms=frozenset({"token", "action"}))
        window.add("token", 4)
        total = torch.tensor(8.0, requires_grad=True)
        zero_action_sum = total * 0.0
        combined, _ = route_window(
            window,
            {"token": total, "action": zero_action_sum},
            global_denominators={"token": torch.tensor(12.0),
                                 "action": torch.tensor(2.0)},
            replica_gradient_scale=3)
        self.assertAlmostEqual(float(combined.detach()), 2.0)
        combined.backward()
        self.assertAlmostEqual(float(total.grad), 0.25)

    def test_pair_loss_shards_reduce_to_global_group_mean(self) -> None:
        weight = 0.1
        global_count = 3
        replica_count = 2
        parameter = torch.tensor(1.0, requires_grad=True)
        local_derivatives = []
        for local_count, local_sum_factor in ((1, 2.0), (2, 3.0)):
            window = SupervisionWindow(
                weights={"pair": weight}, enabled_terms=frozenset({"pair"}))
            window.add("pair", local_count)
            local_sum = parameter * local_sum_factor
            local_objective, _ = route_window(
                window, {"pair": local_sum},
                global_denominators={"pair": torch.tensor(float(global_count))},
                replica_gradient_scale=replica_count)
            gradient, = torch.autograd.grad(local_objective, parameter,
                                            retain_graph=True)
            local_derivatives.append(float(gradient))
        reduced_gradient = sum(local_derivatives) / replica_count
        expected_global_mean_gradient = weight * (2.0 + 3.0) / global_count
        self.assertAlmostEqual(reduced_gradient, expected_global_mean_gradient)

    def test_invalid_replica_scale_refuses(self) -> None:
        window = SupervisionWindow(
            weights={"token": 1.0}, enabled_terms=frozenset({"token"}))
        window.add("token", 1)
        with self.assertRaises(ValueError):
            route_window(window, {"token": torch.tensor(1.0)},
                         replica_gradient_scale=0)

    def test_k8_trainer_routes_token_counts_across_replicas_without_step(self) -> None:
        from bramastra_lab.research.config import BuildConfig
        from bramastra_lab.research.experience.sequences import build_answer_row, collocate
        from bramastra_lab.research.learning.k8_trainer import K8Trainer
        from bramastra_lab.research.models import IntegratedModel

        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        trainer = K8Trainer(config, IntegratedModel(config), device="cpu",
                            precision="fp32", replica_backend=_ReplicaDouble(),
                            require_allocation=False)
        row = build_answer_row(
            [("goal", {"q": "1+1?"})], "2",
            provenance={"kind": "trajectory", "episode_id": "replica-e1",
                        "task_semantic_id": "replica-task", "split": "training",
                        "source": "test", "collection_policy": "fixed",
                        "family": "f"}, max_tokens=64)
        batch = collocate([row], max_seq=64)
        trainer.accumulate(batch)
        self.assertEqual(trainer.counters.optimizer_updates, 0)
        self.assertEqual(trainer._active_global_counts["token"],
                         batch.target_count * 2)
        self.assertTrue(any(parameter.grad is not None
                            for parameter in trainer.model.parameters()))

    def test_trainer_keeps_globally_active_action_term_on_empty_local_shard(self) -> None:
        from bramastra_lab.research.config import BuildConfig
        from bramastra_lab.research.experience.sequences import build_answer_row, collocate
        from bramastra_lab.research.experience.supervision import SupervisionWindow
        from bramastra_lab.research.learning.k8_trainer import K8Trainer
        from bramastra_lab.research.models import IntegratedModel

        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        model = IntegratedModel(config)
        backend = _ReplicaDouble(forced_counts={"action": 1})
        trainer = K8Trainer(config, model, device="cpu", precision="fp32",
                            replica_backend=backend, require_allocation=False)
        row = build_answer_row(
            [("goal", {"q": "2+2?"})], "4",
            provenance={"kind": "trajectory", "episode_id": "replica-e2",
                        "task_semantic_id": "replica-task", "split": "training",
                        "source": "test", "collection_policy": "fixed",
                        "family": "f"}, max_tokens=64)
        batch = collocate([row], max_seq=64)

        def window_builder(target_count: int) -> SupervisionWindow:
            window = SupervisionWindow(
                weights={"token": 1.0, "world": 0.0, "action": 0.5,
                         "value": 0.0, "pair": 0.0, "pg": 0.0},
                enabled_terms=frozenset({"token", "action"}))
            window.add("token", target_count)
            return window

        trainer.accumulate_full_window(
            batch, window_builder=window_builder, extra_terms_fn=lambda: {})
        self.assertEqual(backend.local_counts["action"], 0)
        self.assertEqual(trainer._active_global_counts["action"], 1)
        self.assertIsNone(model.action_head.weight.grad)
        self.assertEqual(trainer.counters.optimizer_updates, 0)

    def test_pair_cognition_loss_is_global_and_reaches_precommit_boundary(self) -> None:
        from bramastra_lab.research.config import BuildConfig
        from bramastra_lab.research.experience.sequences import build_answer_row, collocate
        from bramastra_lab.research.experience.supervision import SupervisionWindow
        from bramastra_lab.research.learning.k8_trainer import K8Trainer
        from bramastra_lab.research.models import IntegratedModel

        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        model = IntegratedModel(config)
        backend = _NoCommitReplicaDouble()
        trainer = K8Trainer(config, model, device="cpu", precision="fp32",
                            replica_backend=backend, require_allocation=False)
        provenance = {
            "kind": "trajectory", "episode_id": "pair-a",
            "task_semantic_id": "pair-test", "split": "training",
            "source": "test", "collection_policy": "paired",
            "family": "reasoning",
        }

        def row(question: str, answer: str, episode_id: str):
            return build_answer_row(
                [("goal", {"question": question})], answer,
                provenance={**provenance, "episode_id": episode_id},
                max_tokens=64)

        own_rows = [row("2+2?", "4", "pair-a"),
                    row("3+4?", "7", "pair-b")]
        swapped_rows = [row("2+2?", "7", "pair-a-swap"),
                        row("3+4?", "4", "pair-b-swap")]

        def window_builder(target_count: int) -> SupervisionWindow:
            window = SupervisionWindow(
                weights={"token": 1.0, "world": 0.0, "action": 0.0,
                         "value": 0.0, "pair": 0.1, "pg": 0.0},
                enabled_terms=frozenset({"token", "pair"}))
            window.add("token", target_count)
            window.add("pair", len(own_rows))
            return window

        before = {name: parameter.detach().clone()
                  for name, parameter in model.named_parameters()}
        trainer.accumulate_full_window(
            collocate([own_rows[0]], max_seq=64), window_builder=window_builder,
            extra_terms_fn=lambda: {}, pair_rows=(own_rows, swapped_rows))

        self.assertEqual(trainer._active_global_counts["pair"], 4)
        self.assertIsNotNone(trainer._pending_pair_loss_value)
        self.assertGreater(
            sum(float(parameter.grad.abs().sum()) for parameter in model.parameters()
                if parameter.grad is not None), 0.0)
        with self.assertRaises(_StopBeforeCommit):
            trainer.finalize_update()
        self.assertTrue(backend.reduce_called)
        self.assertTrue(backend.step_requested)
        self.assertEqual(trainer.counters.optimizer_updates, 0)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_remote_only_pair_rows_contribute_zero_local_gradient_safely(self) -> None:
        from bramastra_lab.research.config import BuildConfig
        from bramastra_lab.research.experience.sequences import build_answer_row, collocate
        from bramastra_lab.research.experience.supervision import SupervisionWindow
        from bramastra_lab.research.learning.k8_trainer import K8Trainer
        from bramastra_lab.research.models import IntegratedModel

        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        model = IntegratedModel(config)
        backend = _NoCommitReplicaDouble(forced_counts={"pair": 1})
        trainer = K8Trainer(config, model, device="cpu", precision="fp32",
                            replica_backend=backend, require_allocation=False)
        row = build_answer_row(
            [("goal", {"question": "5+6?"})], "11",
            provenance={"kind": "trajectory", "episode_id": "pair-empty-rank",
                        "task_semantic_id": "pair-test", "split": "training",
                        "source": "test", "collection_policy": "paired",
                        "family": "reasoning"}, max_tokens=64)

        def window_builder(target_count: int) -> SupervisionWindow:
            window = SupervisionWindow(
                weights={"token": 1.0, "world": 0.0, "action": 0.0,
                         "value": 0.0, "pair": 0.1, "pg": 0.0},
                enabled_terms=frozenset({"token", "pair"}))
            window.add("token", target_count)
            return window

        trainer.accumulate_full_window(
            collocate([row], max_seq=64), window_builder=window_builder,
            extra_terms_fn=lambda: {})
        self.assertEqual(backend.local_counts["pair"], 0)
        self.assertEqual(trainer._active_global_counts["pair"], 1)
        self.assertEqual(trainer._pending_pair_loss_value, 0.0)
        self.assertTrue(backend.mark_requested)
        with self.assertRaises(_StopBeforeCommit):
            trainer.finalize_update()
        self.assertTrue(backend.reduce_called)
        self.assertEqual(trainer.counters.optimizer_updates, 0)

    def test_finalize_leaves_globally_inactive_cognition_heads_untrained(self) -> None:
        from bramastra_lab.research.config import BuildConfig
        from bramastra_lab.research.experience.sequences import build_answer_row, collocate
        from bramastra_lab.research.learning.k8_trainer import K8Trainer
        from bramastra_lab.research.models import IntegratedModel

        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        model = IntegratedModel(config)
        backend = _NoUpdateReplicaDouble()
        backend.named_head_parameters = {
            "action": model.action_head.weight,
            "value": model.value_head.weight,
        }
        trainer = K8Trainer(config, model, device="cpu", precision="fp32",
                            replica_backend=backend, require_allocation=False)
        before = {name: parameter.detach().clone()
                  for name, parameter in model.named_parameters()}
        row = build_answer_row(
            [("goal", {"q": "3+4?"})], "7",
            provenance={"kind": "trajectory", "episode_id": "replica-e3",
                        "task_semantic_id": "replica-task", "split": "training",
                        "source": "test", "collection_policy": "fixed",
                        "family": "f"}, max_tokens=64)
        trainer.accumulate(collocate([row], max_seq=64))

        trainer.finalize_update()

        self.assertTrue(backend.step_requested)
        self.assertEqual(backend.reduced_head_grads, {"action": None, "value": None})
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_finalize_zero_fills_only_globally_active_remote_cognition_head(self) -> None:
        from bramastra_lab.research.config import BuildConfig
        from bramastra_lab.research.experience.sequences import build_answer_row, collocate
        from bramastra_lab.research.experience.supervision import SupervisionWindow
        from bramastra_lab.research.learning.k8_trainer import K8Trainer
        from bramastra_lab.research.models import IntegratedModel

        config = BuildConfig.from_dict({"model": {"profile": "tiny"}})
        model = IntegratedModel(config)
        backend = _NoUpdateReplicaDouble(forced_counts={"action": 1})
        backend.named_head_parameters = {
            "action": model.action_head.weight,
            "value": model.value_head.weight,
        }
        trainer = K8Trainer(config, model, device="cpu", precision="fp32",
                            replica_backend=backend, require_allocation=False)
        before = {name: parameter.detach().clone()
                  for name, parameter in model.named_parameters()}
        row = build_answer_row(
            [("goal", {"q": "5+6?"})], "11",
            provenance={"kind": "trajectory", "episode_id": "replica-e4",
                        "task_semantic_id": "replica-task", "split": "training",
                        "source": "test", "collection_policy": "fixed",
                        "family": "f"}, max_tokens=64)

        def window_builder(target_count: int) -> SupervisionWindow:
            window = SupervisionWindow(
                weights={"token": 1.0, "world": 0.0, "action": 0.5,
                         "value": 0.0, "pair": 0.0, "pg": 0.0},
                enabled_terms=frozenset({"token", "action"}))
            window.add("token", target_count)
            return window

        trainer.accumulate_full_window(
            collocate([row], max_seq=64), window_builder=window_builder,
            extra_terms_fn=lambda: {})
        trainer.finalize_update()

        self.assertTrue(backend.step_requested)
        self.assertIsNotNone(backend.reduced_head_grads["action"])
        self.assertEqual(float(backend.reduced_head_grads["action"].norm()), 0.0)
        self.assertIsNone(backend.reduced_head_grads["value"])
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)


if __name__ == "__main__":
    unittest.main()
