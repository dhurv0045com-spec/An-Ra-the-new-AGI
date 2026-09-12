"""Shared CLI/command error type (lives outside cli.py so that
``python -m bramastra_lab.research.cli`` and importing modules resolve one
class even though the CLI module executes as ``__main__``)."""


class CommandError(RuntimeError):
    """A CLI command failed for a reported, operator-actionable reason."""
