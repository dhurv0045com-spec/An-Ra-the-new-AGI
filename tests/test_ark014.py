"""Focused ARK-014 checks: task firewall, augmentation, controller discipline,
matched-fork replay, evidence survival and frozen verdict logic."""
import copy
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "COLAB"))
sys.path.insert(0, str(REPO / "experiments" / "ARK-001"))


def load_runner(name):
    path = REPO / "experiments" / "ARK-014" / name
    spec = importlib.util.spec_from_file_location(f"test_ark014_{name.replace('.', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


binding = load_runner("ark014_binding.py")
runner = load_runner("run_ark014.py")

TASK = binding.build_binding_task()
META = runner._row_meta(TASK)

SEALED_PROMPTS = {p for d in binding.DIAGNOSTICS
                  for p, _ in binding._diagnostic_rows_grouped(TASK, "BIND_SEALED", d)}


class RecordingWriter:
    def __init__(self, *args, **kwargs):
        self.saves = []

    def save(self, filename, payload):
        self.saves.append((filename, copy.deepcopy(payload)))


def bind_ark11(device):
    from discovery_v6_common import bind_ark11_runtime, load_ark11
    ark11 = load_ark11()
    ark11.DEVICE = device
    ark11.RUNNER_HEAD = "test-head"
    ark11.RUNNER_SOURCE_SHA256 = "test"
    return ark11


def train_rows_and_step(ark11, model, optimizer, rows, device):
    loss, _ = ark11.loss_and_positions(model, vocab_of(ark11), rows, device)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss


def vocab_of(ark11):
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = ark11.CompactVocab()
    return _VOCAB


_VOCAB = None


def fixed_batch(example_ids, step, batch_size=8):
    idx = torch.tensor(example_ids[:batch_size])
    return runner._augmented_rows("ORDER_AUGMENTED", META, idx, step, 2201, None, {})


class TaskContractTests(unittest.TestCase):
    def test_manifest_deterministic(self):
        rebuilt = binding.build_binding_task()
        self.assertEqual(rebuilt["manifest"]["manifest_sha256"],
                         TASK["manifest"]["manifest_sha256"])

    def test_no_factset_crosses_a_split_boundary(self):
        manifest = TASK["manifest"]
        train = {tuple(map(tuple, s)) for s in
                 (binding.signature_json(f) for f in TASK["train_factsets"])}
        control = {tuple(map(tuple, s)) for s in manifest["control_factsets"]}
        sealed = {tuple(map(tuple, s)) for s in manifest["sealed_factsets"]}
        self.assertEqual(len(train), 400)
        self.assertEqual(len(control), 50)
        self.assertEqual(len(sealed), 50)
        self.assertEqual(train & control, set())
        self.assertEqual(train & sealed, set())
        self.assertEqual(control & sealed, set())
        self.assertEqual(manifest["factset_overlap"], 0)
        self.assertTrue(manifest["split_rule"]["applied_before_expansion"])

    def test_historical_ark009_anchor_reproduced(self):
        anchor = TASK["manifest"]["historical_ark009_anchor"]
        self.assertEqual(TASK["manifest"]["train_factset_sha256"], anchor["train_factset_sha256"])
        self.assertEqual(TASK["manifest"]["heldout_factset_sha256"], anchor["test_factset_sha256"])

    def test_all_order_variants_preserve_answer_semantics(self):
        for split in ("BIND_CONTROL", "BIND_SEALED"):
            for diagnostic in binding.DIAGNOSTICS:
                for row in TASK["diagnostics"][diagnostic][split]:
                    self.assertEqual(row["answer"],
                                     binding.symbolic_answer(row["facts"], row["query"]))

    def test_diagnostics_change_exactly_their_variable(self):
        for split in ("BIND_CONTROL", "BIND_SEALED"):
            canon = TASK["diagnostics"]["CANONICAL"][split]
            for name in ("ORDER_ONLY", "QUERY_ONLY", "QUERY_ORDER"):
                for a, b in zip(canon, TASK["diagnostics"][name][split]):
                    order_changed = a["facts"] != b["facts"]
                    query_changed = a["query"] != b["query"]
                    if name == "ORDER_ONLY":
                        self.assertTrue(order_changed)
                        self.assertFalse(query_changed)
                        self.assertEqual(a["answer"], b["answer"])
                    elif name == "QUERY_ONLY":
                        self.assertFalse(order_changed)
                        self.assertTrue(query_changed)
                    else:
                        self.assertTrue(order_changed)
                        self.assertTrue(query_changed)


class AugmentationTests(unittest.TestCase):
    def test_pure_function_deterministic(self):
        facts = TASK["train_factsets"][7]
        self.assertEqual(binding.augment_facts(facts, 2201, 12, 3, 19),
                         binding.augment_facts(facts, 2201, 12, 3, 19))

    def test_preserves_semantics_and_membership(self):
        for example_id in range(0, 1200, 97):
            for step in (1, 2, 4999):
                for pos in (0, 32, 63):
                    facts = binding.augment_facts(
                        META[example_id]["facts"], 2201, step, pos, example_id)
                    self.assertEqual(tuple(sorted(facts)),
                                     tuple(sorted(META[example_id]["facts"])))
                    self.assertEqual(binding.symbolic_answer(facts, META[example_id]["query"]),
                                     META[example_id]["answer"])

    def test_changed_inputs_change_order_across_a_sweep(self):
        base = [binding.augmentation_permutation_index(2201, s, p, e)
                for s in range(1, 60) for p in range(4) for e in range(0, 120, 7)]
        self.assertEqual(len(set(base)), 6)
        other = [binding.augmentation_permutation_index(2202, s, p, e)
                 for s in range(1, 60) for p in range(4) for e in range(0, 120, 7)]
        self.assertNotEqual(base[:40], other[:40])

    def test_augmentation_does_not_touch_torch_rng(self):
        facts = TASK["train_factsets"][3]
        before = torch.get_rng_state().clone()
        binding.augment_facts(facts, 2201, 1, 0, 0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


class ControllerDisciplineTests(unittest.TestCase):
    def _acquire_with_mutated_sealed(self, sealed_mutation):
        """Short fake acquisition: CONTROL evals forced to qualify; SEALED
        measurements mutated by the caller-supplied function."""
        import time as _time
        from discovery_v6_common import RunContext
        ark11 = bind_ark11(torch.device("cpu"))
        original = runner._evaluate_diagnostics

        def patched(_ark11, model, vocab, rows_by_diagnostic, device):
            out = {}
            for diagnostic, rows in rows_by_diagnostic.items():
                if rows[0][0] in SEALED_PROMPTS:
                    out[diagnostic] = sealed_mutation
                else:
                    out[diagnostic] = {"CANONICAL": 0.95, "ORDER_ONLY": 0.90,
                                       "QUERY_ONLY": 0.90, "QUERY_ORDER": 0.90}[diagnostic]
            return out

        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext(torch.device("cpu"), "test-head", _time.time(), 5.0,
                             Path(tmp) / "fake-run")
            with patch.object(runner, "_evaluate_diagnostics", patched):
                arm = runner.acquire_arm(
                    ark11, regime="ORDER_AUGMENTED", task=TASK, meta=META,
                    device=ctx.device, ctx=ctx, writer=RecordingWriter(),
                    checkpoints_dir=ctx.output_dir / "checkpoints",
                    max_steps=3, eval_every=1, log=lambda *a, **k: None)
        return arm

    def test_controller_decision_ignores_sealed_values(self):
        arms = [self._acquire_with_mutated_sealed(value) for value in (0.0, 0.44, 1.0)]
        fingerprints = set()
        for arm in arms:
            self.assertEqual(arm["status"], "QUALIFIED")
            self.assertEqual(arm["qualification_confirmation_step"], 3)
            self.assertEqual([t["step"] for t in arm["trajectory"]], [1, 2, 3])
            fingerprints.add(json.dumps(arm["trajectory"], sort_keys=True))
        # One identical decision stream despite three different sealed streams.
        self.assertEqual(len(fingerprints), 1)
        # And the sealed values really were different each time.
        sealed_sets = {json.dumps(a["sealed_at_qualification"], sort_keys=True) for a in arms}
        self.assertEqual(len(sealed_sets), 3)

    def test_qualification_thresholds_frozen(self):
        self.assertEqual(binding.QUALIFICATION_THRESHOLDS,
                         {"CANONICAL": 0.90, "ORDER_ONLY": 0.85, "QUERY_ORDER": 0.85})
        self.assertEqual(binding.QUALIFICATION_CONSECUTIVE, 3)


class MatchedUpdateTests(unittest.TestCase):
    def _setup(self):
        ark11 = bind_ark11(torch.device("cpu"))
        torch.manual_seed(2201)
        vocab = ark11.CompactVocab()
        model = ark11.Micro(vocab.size, 128)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                                      eps=1e-8, weight_decay=0.1)
        return ark11, vocab, model, optimizer

    def test_same_snapshot_order_lr_reproduces_next_update(self):
        ark11, vocab, model, optimizer = self._setup()
        rows = fixed_batch(list(range(8)), 1)
        train_rows_and_step(ark11, model, optimizer, rows, torch.device("cpu"))
        snapshot = ark11.snapshot_state(model, optimizer)
        hashes = []
        for _ in range(2):
            _, fork_model, fork_optimizer = ark11.load_fork(snapshot, 1e-3)
            train_rows_and_step(ark11, fork_model, fork_optimizer, rows, torch.device("cpu"))
            hashes.append(runner.parameter_sha(fork_model))
        self.assertEqual(hashes[0], hashes[1])

    def test_different_lr_changes_next_update(self):
        ark11, vocab, model, optimizer = self._setup()
        rows = fixed_batch(list(range(8)), 1)
        train_rows_and_step(ark11, model, optimizer, rows, torch.device("cpu"))
        snapshot = ark11.snapshot_state(model, optimizer)
        _, high_model, high_optimizer = ark11.load_fork(snapshot, 1e-3)
        _, low_model, low_optimizer = ark11.load_fork(snapshot, 1e-5)
        train_rows_and_step(ark11, high_model, high_optimizer, rows, torch.device("cpu"))
        train_rows_and_step(ark11, low_model, low_optimizer, rows, torch.device("cpu"))
        self.assertNotEqual(runner.parameter_sha(high_model), runner.parameter_sha(low_model))


class ObjectiveAuditTests(unittest.TestCase):
    def test_supervised_positions_are_answer_bos_digit_eos(self):
        ark11 = bind_ark11(torch.device("cpu"))
        torch.manual_seed(0)
        vocab = ark11.CompactVocab()
        model = ark11.Micro(vocab.size, 128)
        rows = [(binding.render_prompt(META[i]["facts"], META[i]["query"]), META[i]["answer"])
                for i in range(16)]
        _, count = ark11.loss_and_positions(model, vocab, rows, torch.device("cpu"))
        self.assertEqual(int(count.item()), 3 * len(rows))

    def test_no_padding_heterogeneity_in_ark014_rows(self):
        # Every ARK-014 prompt is 14 characters and every answer one digit, so
        # each encoded row has identical length: the inherited objective's
        # mixed-length padding concern cannot arise in this task (documented in
        # the dated protocol clarification).
        from run_ark001 import CompactVocab
        vocab = CompactVocab()
        lengths = set()
        for split in ("BIND_CONTROL", "BIND_SEALED"):
            for diagnostic in binding.DIAGNOSTICS:
                for prompt, answer in binding._diagnostic_rows_grouped(TASK, split, diagnostic):
                    lengths.add(len(vocab.encode(prompt)) + len(vocab.encode(answer)) + 1)
        self.assertEqual(len(lengths), 1)


class EvidenceSurvivalTests(unittest.TestCase):
    def _fake_campaign(self, tmp, explode_on_call):
        """Two short acquisition arms forced QUALIFIED, then retention forks
        where the Nth arm raises an injected failure."""
        from discovery_v6_common import RunContext
        ark11 = bind_ark11(torch.device("cpu"))
        ctx = RunContext(torch.device("cpu"), "test-head", time.time(), 30.0,
                         Path(tmp) / "survive")
        writer = RecordingWriter()
        checkpoints = ctx.output_dir / "checkpoints"
        acquisitions = []
        for regime in ("CANONICAL_TRAIN", "ORDER_AUGMENTED"):
            arm = runner.acquire_arm(ark11, regime=regime, task=TASK, meta=META,
                                     device=ctx.device, ctx=ctx, writer=writer,
                                     checkpoints_dir=checkpoints, max_steps=2,
                                     eval_every=1, log=lambda *a, **k: None)
            arm["status"] = "QUALIFIED"
            acquisitions.append(arm)
        writer.save("ARK-014_PARTIAL.json", {"acquisitions": runner._public(acquisitions),
                                             "retention": []})
        snapshot = runner._snapshot_from_checkpoint(ark11, ctx, acquisitions,
                                                    "ORDER_AUGMENTED", checkpoints)
        retention = []
        calls = {"n": 0}
        original = runner.retention_fork

        def exploding_fork(**kwargs):
            calls["n"] += 1
            if calls["n"] == explode_on_call:
                raise RuntimeError("injected later-arm failure")
            return original(ark11, **kwargs)

        try:
            with patch.object(runner, "retention_fork", exploding_fork):
                for order_seed in runner.CONTINUATION_ORDER_SEEDS:
                    for lr, label in ((runner.LR_HIGH, "HIGH"), (runner.LR_LOW, "LOW")):
                        retention.append(runner.retention_fork(
                            regime="ORDER_AUGMENTED", snapshot=snapshot,
                            order_seed=order_seed, lr=lr, task=TASK, meta=META,
                            device=ctx.device, ctx=ctx, checkpoints_dir=checkpoints,
                            steps=2, eval_every=1, log=lambda *a, **k: None))
                        writer.save("ARK-014_PARTIAL.json", {
                            "acquisitions": runner._public(acquisitions),
                            "retention": runner._public(retention)})
        except RuntimeError as exc:
            writer.save("ARK-014_FAILURE.json", {
                "status": "FAILED", "exception": str(exc),
                "acquisitions": runner._public(acquisitions),
                "retention": runner._public(retention),
                "summary": runner.summarize(acquisitions, retention,
                                            protocol_scale="DIAGNOSTIC_SCALE_NOT_PREREGISTERED"),
            })
        return writer

    def test_completed_arms_survive_injected_later_arm_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = self._fake_campaign(tmp, explode_on_call=5)
            failure = [p for name, p in writer.saves if name == "ARK-014_FAILURE.json"]
            self.assertEqual(len(failure), 1)
            # The 5th arm exploded; the four completed arms are in the final
            # partial receipt and the failure receipt.
            partials = [p for name, p in writer.saves if name == "ARK-014_PARTIAL.json"]
            self.assertEqual(len(partials[-1]["retention"]), 4)
            self.assertEqual(len(failure[0]["retention"]), 4)
            completed = [r for r in failure[0]["retention"] if r["status"] == "COMPLETED"]
            self.assertEqual(len(completed), 4)
            self.assertNotEqual(failure[0]["summary"]["verdict"], "NONARITHMETIC_LR_PROTECTION_SCREEN")

    def test_incomplete_arms_receive_explicit_status(self):
        from discovery_v6_common import BudgetExhausted, RunContext
        ark11 = bind_ark11(torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext(torch.device("cpu"), "test-head", time.time(), 1e-6,
                             Path(tmp) / "budget-starved")
            writer = RecordingWriter()
            with self.assertRaises(BudgetExhausted):
                runner.acquire_arm(ark11, regime="ORDER_AUGMENTED", task=TASK, meta=META,
                                   device=ctx.device, ctx=ctx, writer=writer,
                                   checkpoints_dir=ctx.output_dir / "checkpoints",
                                   max_steps=10, eval_every=1, log=lambda *a, **k: None)
            incomplete = [p for name, p in writer.saves
                          if name == "ARK-014_ARM_ORDER_AUGMENTED_INCOMPLETE.json"]
            self.assertEqual(len(incomplete), 1)
            self.assertEqual(incomplete[0]["status"], "INCOMPLETE_BUDGET_STOP")
            self.assertIn("checkpoint", incomplete[0])


class VerdictLogicTests(unittest.TestCase):
    @staticmethod
    def arm(regime, status):
        return {"regime": regime, "status": status, "trajectory": [],
                "checkpoint": {"filename": "x.pt", "sha256": "a", "parameter_sha256": "b",
                               "step": 1}}

    @staticmethod
    def retention(order_seed, high_failure, low_failure, at_fork=True, completed=True):
        def arm(label, failure):
            trajectory = [{"step": 200, "sealed_qualified": at_fork,
                           "control_qualified": True}]
            outcome = [not failure] * 3
            for i, qualifies in enumerate(outcome):
                trajectory.append({"step": 1000 + 200 * i,
                                   "sealed_qualified": qualifies,
                                   "control_qualified": True})
            return {"regime": "ORDER_AUGMENTED", "order_seed": order_seed, "lr_label": label,
                    "status": "COMPLETED" if completed else "INCOMPLETE_BUDGET_STOP",
                    "completed_steps": runner.RETENTION_STEPS if completed else 1,
                    "sealed_qualification_failures": runner._first_sustained_failures(trajectory),
                    "sealed_qualification_at_fork": at_fork,
                    "trajectory": trajectory,
                    "sealed_metrics": {}, "control_metrics": {}}
        # The runner records one flat arm row per (order, LR); tests mirror that.
        return [arm("HIGH", high_failure), arm("LOW", low_failure)]

    def qualified_pair(self):
        return [self.arm("CANONICAL_TRAIN", "NO_QUALIFICATION"),
                self.arm("ORDER_AUGMENTED", "QUALIFIED")]

    def test_not_acquired_when_neither_qualifies(self):
        summary = runner.summarize(
            [self.arm("CANONICAL_TRAIN", "NO_QUALIFICATION"),
             self.arm("ORDER_AUGMENTED", "NO_QUALIFICATION")],
            [], protocol_scale="PREREGISTERED_FROZEN")
        self.assertEqual(summary["verdict"], "ROBUST_BINDING_NOT_ACQUIRED")

    def test_qualified_but_no_failure_events_is_inconclusive(self):
        retention = [a for o in (7701, 7702, 7703) for a in self.retention(o, False, False)]
        summary = runner.summarize(self.qualified_pair(), retention,
                                   protocol_scale="PREREGISTERED_FROZEN")
        self.assertEqual(summary["verdict"], "ROBUST_BINDING_ACQUIRED_BUT_TRANSFER_INCONCLUSIVE")

    def test_protection_screen(self):
        retention = [a for order, h, l in ((7701, True, False), (7702, True, False), (7703, False, False)) for a in self.retention(order, h, l)]
        summary = runner.summarize(self.qualified_pair(), retention,
                                   protocol_scale="PREREGISTERED_FROZEN")
        self.assertEqual(summary["verdict"], "NONARITHMETIC_LR_PROTECTION_SCREEN")
        paired = summary["paired_retention"]
        self.assertEqual(paired["high_failure_orders"], 2)
        self.assertEqual(paired["low_failure_orders"], 0)
        self.assertAlmostEqual(paired["risk_difference_low_minus_high"], -2 / 3)

    def test_transfer_not_supported_screen(self):
        retention = [a for order, h, l in ((7701, True, True), (7702, True, False), (7703, True, False)) for a in self.retention(order, h, l)]
        summary = runner.summarize(self.qualified_pair(), retention,
                                   protocol_scale="PREREGISTERED_FROZEN")
        self.assertEqual(summary["verdict"], "TRANSFER_NOT_SUPPORTED_SCREEN")

    def test_incomplete_or_blocked_retention_is_never_a_comparison(self):
        partial = [a for order, h, l, c in ((7701, True, False, True), (7702, True, False, False)) for a in self.retention(order, h, l, completed=c)]
        summary = runner.summarize(self.qualified_pair(), partial,
                                   protocol_scale="PREREGISTERED_FROZEN")
        self.assertEqual(summary["verdict"], "ROBUST_BINDING_ACQUIRED_BUT_TRANSFER_INCONCLUSIVE")
        blocked = [{"regime": "ORDER_AUGMENTED", "status": "BUDGET_BLOCKED_BEFORE_RETENTION"}]
        summary = runner.summarize(self.qualified_pair(), blocked,
                                   protocol_scale="PREREGISTERED_FROZEN")
        self.assertEqual(summary["verdict"], "ROBUST_BINDING_ACQUIRED_BUT_RETENTION_NOT_EXECUTED")

    def test_reverse_discordance_blocks_protection_screen(self):
        retention = [a for order, h, l in ((7701, True, False), (7702, False, True), (7703, True, False)) for a in self.retention(order, h, l)]
        summary = runner.summarize(self.qualified_pair(), retention,
                                   protocol_scale="PREREGISTERED_FROZEN")
        self.assertEqual(summary["verdict"], "ROBUST_BINDING_ACQUIRED_BUT_TRANSFER_INCONCLUSIVE")

    def test_canonical_only_qualification_is_not_a_repair(self):
        summary = runner.summarize(
            [self.arm("CANONICAL_TRAIN", "QUALIFIED"),
             self.arm("ORDER_AUGMENTED", "NO_QUALIFICATION")],
            [], protocol_scale="PREREGISTERED_FROZEN")
        self.assertEqual(summary["verdict"], "ORDER_AUGMENTATION_NOT_SUPPORTED_CANONICAL_QUALIFIED")


class ReceiptIdentityTests(unittest.TestCase):
    def test_receipt_writer_binds_ark014_sources(self):
        from discovery_v6_common import ReceiptWriter, RunContext
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext(torch.device("cpu"), "test-head", time.time(), 5.0,
                             Path(tmp) / "receipt-run")
            writer = ReceiptWriter(
                ctx, experiment_id="ARK-014", plan_sha=runner.PLAN_COMMIT_SHA,
                runner_path=runner.RUNNER_PATH,
                extra_plan_shas={"ark014_plan_commit_sha": runner.PLAN_COMMIT_SHA},
                extra_source_paths=[Path(runner.__file__).parent / "ark014_binding.py"])
            self.assertIn("experiments/ARK-014/run_ark014.py", writer.sources)
            self.assertIn("experiments/ARK-014/ark014_binding.py", writer.sources)
            payload = {}
            path = writer.save("PROBE.json", payload)
            saved = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(saved["ark014_plan_commit_sha"], runner.PLAN_COMMIT_SHA)
            self.assertTrue(saved["receipt_sha256"])


if __name__ == "__main__":
    unittest.main()
