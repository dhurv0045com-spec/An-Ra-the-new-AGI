"""B2.2-chief F5/F6 focused tests: public accumulation semantics, hard
pre-execution accounting, failed-run accounting. No optimizer steps."""
import json
import os
import tempfile
import unittest

from bramastra_lab.research.config import BuildConfig
from bramastra_lab.research.learning.trainer import Trainer, TrainerStateError
from bramastra_lab.research.runtime.smoke import (
    LIMITS,
    SessionLedger,
    SmokeBudgetExhausted,
)


def tiny_trainer(grad_accum_steps=1):
    seed = 5
    import torch

    torch.manual_seed(seed)
    config = BuildConfig.from_dict({
        "model": {"profile": "tiny"},
        "training": {"grad_accum_steps": grad_accum_steps}})
    model = __import__("bramastra_lab.research.models", fromlist=["IntegratedModel"]) \
        .IntegratedModel(config)
    return Trainer(config, model)


class TrainingStepGuardTests(unittest.TestCase):
    def test_single_microbatch_window_allowed(self) -> None:
        trainer = tiny_trainer(grad_accum_steps=1)
        self.assertEqual(trainer.grad_accum_steps, 1)
        self.assertIsInstance(trainer, Trainer)

    def test_configured_multi_microbatch_rejects_step(self) -> None:
        trainer = tiny_trainer(grad_accum_steps=2)
        with self.assertRaises(TrainerStateError) as caught:
            trainer.training_step(_fake_batch())
        self.assertIn("grad_accum_steps=2", str(caught.exception))
        self.assertIn("accumulate/finalize_update", str(caught.exception))

    def test_docstring_contract_declares_restriction(self) -> None:
        self.assertIn("grad_accum_steps == 1", Trainer.training_step.__doc__)


def _fake_batch():
    """A minimal batch-shaped object; accumulate validation runs first."""
    import torch

    from bramastra_lab.research.experience.sequences import (
        build_answer_row,
        collocate,
    )

    row = build_answer_row(
        [("goal", {"q": 1})], "7",
        provenance={"kind": "trajectory", "episode_id": "e1",
                    "task_semantic_id": "t", "split": "training",
                    "source": "test", "collection_policy": "fixed",
                    "family": "f"},
        max_tokens=64)
    return collocate([row], max_seq=64)


class LedgerAccountingTests(unittest.TestCase):
    def temp_ledger(self, used_updates=0, used_seconds=0.0):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "SESSION_LEDGER.json")
        SessionLedger(path)
        if used_updates or used_seconds:
            state = json.load(open(path, encoding="utf-8"))
            state["cpu_optimizer_updates"] = used_updates
            state["cpu_learned_smoke_seconds"] = used_seconds
            json.dump(state, open(path, "w", encoding="utf-8"), indent=2, sort_keys=True)
        return path

    def test_pre_execution_accounting_refuses_over_budget(self) -> None:
        ledger = SessionLedger(self.temp_ledger(used_updates=LIMITS["cpu_optimizer_updates"]))
        with self.assertRaises(SmokeBudgetExhausted) as caught:
            ledger.require_learned_allowance(updates=1, seconds=1.0,
                                             what="unit probe")
        self.assertIn("new owner allocation", str(caught.exception))

    def test_pre_execution_accounting_admits_within_budget(self) -> None:
        ledger = SessionLedger(self.temp_ledger(used_updates=190))
        ledger.require_learned_allowance(updates=6, seconds=60.0, what="unit probe")
        self.assertTrue(True)  # no refusal

    def test_failed_run_accounting_records_consumption(self) -> None:
        """Injected failure after partial work still records what was spent."""
        path = self.temp_ledger(used_updates=100)
        ledger = SessionLedger(path)
        ledger.check_can_run(device="cpu", updates=5, seconds=90.0,
                             reserve_for_resume=False)
        try:
            ledger.record(device="cpu", updates=2, seconds=12.0,
                          what="partial work before injected failure",
                          evidence="injected")
            raise RuntimeError("injected failure")
        except RuntimeError:
            ledger.record(device="cpu", updates=0, seconds=1.0,
                          what="failure accounting entry", evidence="injected")
        state = ledger.snapshot()
        self.assertEqual(state["cpu_optimizer_updates"], 102)
        self.assertGreaterEqual(len(state["history"]), 2)
        whats = [entry["what"] for entry in state["history"]]
        self.assertIn("partial work before injected failure", whats)
        self.assertIn("failure accounting entry", whats)

    def test_subprocess_entry_cannot_bypass_shared_allowance(self) -> None:
        """A learned probe launched outside train --smoke still consults the
        shared ledger and refuses when it is exhausted."""
        import subprocess
        import sys

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ledger_path = self.temp_ledger(
            used_updates=LIMITS["cpu_optimizer_updates"],
            used_seconds=LIMITS["cpu_learned_smoke_seconds"])
        completed = subprocess.run(
            [sys.executable, os.path.join(repo_root, "tests", "_b22_resume_probe.py"),
             "--workdir", os.path.join(tempfile.gettempdir(), "b22-guard-probe"),
             "--ledger", ledger_path],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": repo_root})
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("new owner allocation", completed.stderr)

    def test_ledger_totals_are_additive_and_precise(self) -> None:
        path = self.temp_ledger()
        ledger = SessionLedger(path)
        ledger.record(device="cpu", updates=1, seconds=6.0,
                      what="failed attempt", evidence="recorded")
        ledger.record(device="cpu", updates=6, seconds=30.0, what="comparison one")
        ledger.record(device="cpu", updates=6, seconds=30.0, what="comparison two")
        state = ledger.snapshot()
        self.assertEqual(state["cpu_optimizer_updates"], 13)
        self.assertAlmostEqual(state["cpu_learned_smoke_seconds"], 66.0)


if __name__ == "__main__":
    unittest.main()
