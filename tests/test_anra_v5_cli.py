from __future__ import annotations

import unittest

from anra_v5.cli import SUBCOMMANDS, main


class AnraV5CliTests(unittest.TestCase):
    def test_every_registered_module_exposes_a_callable_main(self) -> None:
        import importlib

        self.assertGreaterEqual(len(SUBCOMMANDS), 30)
        for name, (module_name, _description) in SUBCOMMANDS.items():
            with self.subTest(command=name):
                module = importlib.import_module(module_name)
                self.assertTrue(callable(getattr(module, "main", None)))

    def test_unknown_command_fails_closed(self) -> None:
        self.assertEqual(main(["no-such-command"]), 2)

    def test_help_lists_commands(self) -> None:
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(main(["--help"]), 0)
        text = buffer.getvalue()
        self.assertIn("readiness", text)
        self.assertIn("collect", text)
        self.assertIn("transaction", text)

    def test_bare_invocation_reports_usage_without_action(self) -> None:
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(main([]), 2)
        self.assertIn("usage", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
