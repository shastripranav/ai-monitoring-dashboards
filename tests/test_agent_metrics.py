from prometheus_client import CollectorRegistry

from ai_monitor import AgentMetrics


class TestAgentMetrics:
    def setup_method(self):
        self.registry = CollectorRegistry()
        self.metrics = AgentMetrics(prefix="test", registry=self.registry)

    def test_record_routing(self):
        self.metrics.record_routing("finance")
        self.metrics.record_routing("finance")
        self.metrics.record_routing("operations")

        finance = self.registry.get_sample_value(
            "test_agent_routing_total",
            {"agent_name": "finance", "from_supervisor": "default"},
        )
        ops = self.registry.get_sample_value(
            "test_agent_routing_total",
            {"agent_name": "operations", "from_supervisor": "default"},
        )
        assert finance == 2.0
        assert ops == 1.0

    def test_observe_response(self):
        self.metrics.observe_response("finance", 0.5)
        count = self.registry.get_sample_value(
            "test_agent_response_duration_count", {"agent_name": "finance"},
        )
        assert count == 1.0

    def test_observe_iterations(self):
        self.metrics.observe_iterations(3)
        count = self.registry.get_sample_value(
            "test_agent_iterations_total_count", {},
        )
        assert count == 1.0

    def test_record_error_and_fallback(self):
        self.metrics.record_error("planning", "timeout")
        self.metrics.record_fallback("planning")

        err = self.registry.get_sample_value(
            "test_agent_errors_total",
            {"agent_name": "planning", "error_type": "timeout"},
        )
        fb = self.registry.get_sample_value(
            "test_agent_fallback_total", {"agent_name": "planning"},
        )
        assert err == 1.0
        assert fb == 1.0
