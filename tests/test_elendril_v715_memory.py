"""V7.15 memory interventions must preserve same-source content controls."""
from anra_v5.elendril.v715_memory import build_suite, audit


def test_memory_suite_is_deterministic_and_teacher_qualified():
    suite=build_suite()
    assert len(suite)==6 and sum(len(v) for v in suite.values())==96
    assert audit(suite)["status"]=="PASS"


def test_equal_length_and_corrected_content_with_order_intervention():
    suite=build_suite()
    for i in range(16):
        a=suite["consistent-two"][i].memory.events
        b=suite["stale-corrected"][i].memory.events
        c=suite["reversed-stale-corrected"][i].memory.events
        assert len(a)==len(b)==len(c)==2
        assert a[1].value==b[1].value
        assert a[0].value!=b[0].value
        assert tuple(reversed(b))==c
        assert suite["removed"][i].truth.expected is None
        assert suite["irrelevant"][i].truth.expected is None
