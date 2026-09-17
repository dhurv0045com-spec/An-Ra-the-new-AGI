"""Decision helper for the unexecuted DOSE-001 proposal; no launch authority."""


def should_escalate(trace, *, complete):
    """Require the exact completed five-point flat trajectory."""
    return (complete is True
            and [row.get('step') for row in trace] == [0, 50, 100, 150, 200]
            and all(row.get('correct') == 27 for row in trace))
