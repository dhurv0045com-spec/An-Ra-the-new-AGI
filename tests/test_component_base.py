from engine.component_base import BaseComponent, ComponentProtocol


class DemoComponent(BaseComponent):
    name = "demo"
    version = "1.2.3"

    def run(self, payload):
        self._record_call(success=True, latency_ms=12.5)
        return {"success": True, "payload": payload}


def test_base_component_tracks_metrics():
    component = DemoComponent()
    component.run({"x": 1})
    component._record_call(success=False, latency_ms=7.5, error="boom")

    metrics = component.metrics()
    assert metrics.calls_total == 2
    assert metrics.calls_success == 1
    assert metrics.calls_failed == 1
    assert metrics.total_latency_ms == 20.0
    assert metrics.last_error == "boom"


def test_base_component_success_rate():
    component = DemoComponent()
    component.run({})
    component.run({})

    assert component.metrics().success_rate == 1.0
    assert component.metrics().to_dict()["avg_latency_ms"] == 12.5


def test_protocol_is_checkable():
    component = DemoComponent()

    assert isinstance(component, ComponentProtocol)
    assert component.health()["status"] == "ok"
