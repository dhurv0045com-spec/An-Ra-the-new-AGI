"""Remote-accelerator execution contracts for the V5 training path.

Local GPU/CPU execution is out of scope for this package: it only describes a
job to run on a remote accelerator host and binds a returned result back to
that exact request. Nothing here submits jobs, stores credentials, or executes
training compute. See ``v5_remote/README.md`` for the submission flow.
"""

from .collect import collect
from .job_spec import RemoteJob
from .result import RemoteResult, bind_result, submission_envelope

__all__ = [
    "RemoteJob",
    "RemoteResult",
    "bind_result",
    "collect",
    "submission_envelope",
]
