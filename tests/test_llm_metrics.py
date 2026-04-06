import time

from prometheus_client import CollectorRegistry, generate_latest

from ai_monitor import LLMMetrics


class TestLLMMetrics:
    def setup_method(self):
        self.registry = CollectorRegistry()
        self.metrics = LLMMetrics(prefix="test", registry=self.registry)

    def test_record_call_increments_counter(self):
        self.metrics.record_call(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, latency_ms=500,
        )
        val = self.registry.get_sample_value(
            "test_llm_requests_total",
            {"provider": "openai", "model": "gpt-4o", "status": "success"},
        )
        assert val == 1.0

    def test_record_call_tracks_tokens_separately(self):
        self.metrics.record_call(
            provider="openai", model="gpt-4o",
            input_tokens=150, output_tokens=340, latency_ms=800,
        )
        inp = self.registry.get_sample_value(
            "test_llm_tokens_total",
            {"provider": "openai", "model": "gpt-4o", "direction": "input"},
        )
        out = self.registry.get_sample_value(
            "test_llm_tokens_total",
            {"provider": "openai", "model": "gpt-4o", "direction": "output"},
        )
        assert inp == 150.0
        assert out == 340.0

    def test_context_manager_tracks_latency(self):
        with self.metrics.track_call(provider="openai", model="gpt-4o-mini") as tracker:
            time.sleep(0.05)
            tracker.set_tokens(100, 200)

        count = self.registry.get_sample_value(
            "test_llm_request_duration_seconds_count",
            {"provider": "openai", "model": "gpt-4o-mini"},
        )
        assert count == 1.0

        # tokens should also be recorded
        inp = self.registry.get_sample_value(
            "test_llm_tokens_total",
            {"provider": "openai", "model": "gpt-4o-mini", "direction": "input"},
        )
        assert inp == 100.0

    def test_context_manager_records_errors(self):
        try:
            with self.metrics.track_call(provider="openai", model="gpt-4o"):
                raise ValueError("test error")
        except ValueError:
            pass

        err = self.registry.get_sample_value(
            "test_llm_errors_total",
            {"provider": "openai", "model": "gpt-4o", "error_type": "ValueError"},
        )
        assert err == 1.0

        status = self.registry.get_sample_value(
            "test_llm_requests_total",
            {"provider": "openai", "model": "gpt-4o", "status": "error"},
        )
        assert status == 1.0

    def test_cost_calculation(self):
        self.metrics.record_call(
            provider="openai", model="gpt-4o",
            input_tokens=1000, output_tokens=500, latency_ms=1000,
        )
        cost = self.registry.get_sample_value(
            "test_llm_cost_dollars_total",
            {"provider": "openai", "model": "gpt-4o"},
        )
        # gpt-4o: input=2.50/1M, output=10.00/1M
        expected = 1000 * 2.50 / 1_000_000 + 500 * 10.00 / 1_000_000
        assert abs(cost - expected) < 1e-9

    def test_error_recording_increments_error_counter(self):
        self.metrics.record_call(
            provider="anthropic", model="claude-3-5-sonnet",
            latency_ms=500, error=True, error_type="timeout",
        )
        val = self.registry.get_sample_value(
            "test_llm_errors_total",
            {"provider": "anthropic", "model": "claude-3-5-sonnet", "error_type": "timeout"},
        )
        assert val == 1.0

    def test_metrics_format_is_valid_prometheus(self):
        self.metrics.record_call(
            provider="openai", model="gpt-4o",
            input_tokens=100, output_tokens=50, latency_ms=500,
        )
        output = generate_latest(self.registry)
        assert b"test_llm_requests_total" in output
        assert b"provider=" in output
