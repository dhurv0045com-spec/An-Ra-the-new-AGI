"""An-Ra: the reference runtime façade.

    import anra
    result = anra.run("...", checkpoint="anra-v4.pt", expected="...")
    print(result.status, result.answer, result.changed_variable)

See ``connector/runtime.py`` for the executable loop this exposes.
"""

from connector.runtime import RunResult, Step, run

__all__ = ["run", "RunResult", "Step"]
__version__ = "0.6.0"
